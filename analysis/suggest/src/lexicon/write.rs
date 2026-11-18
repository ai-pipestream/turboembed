//! Compiling accumulated `(surface, stem, count)` observations into a flat
//! `TSUGLEX1` double-array image.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fmt;
use std::num::NonZeroU32;

use turbo_stem::Stemmer;
use turbo_tokenize::{WordType, word_tokens};

use super::{Counts, ENDIAN, EntryRow, FREE, HEADER, MAGIC, PAGE_INDEX, ROOT_CHECK, TrieCell, VERSION, checksum};
use crate::aligned::ImageBuf;
use crate::fold::{canonical_surface_lowercase, fold_surface_key};

const MAX_WORD_UTF16: NonZeroU32 = NonZeroU32::new(255).unwrap();

/// Default number of ranked completions stored per prefix node and per stem.
pub const DEFAULT_TOP_K: usize = 16;

/// Why [`SurfaceLexiconBuilder::build_bytes`] refused an image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LexiconBuildError {
    TooLarge(&'static str),
}

impl fmt::Display for LexiconBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LexiconBuildError::TooLarge(what) => write!(f, "lexicon {what} exceeds u32 capacity"),
        }
    }
}

impl std::error::Error for LexiconBuildError {}

/// Accumulates `(surface_token, stem, count)` observations during ingest or
/// shard compaction and compiles a zero-copy `TSUGLEX1` lexicon image.
#[derive(Clone, Debug)]
pub struct SurfaceLexiconBuilder {
    observations: BTreeMap<(String, String), u64>,
    top_k_prefix: usize,
    top_k_stem: usize,
}

impl Default for SurfaceLexiconBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SurfaceLexiconBuilder {
    /// Creates an empty builder retaining up to [`DEFAULT_TOP_K`] entries per
    /// prefix and per stem.
    pub fn new() -> Self {
        SurfaceLexiconBuilder { observations: BTreeMap::new(), top_k_prefix: DEFAULT_TOP_K, top_k_stem: DEFAULT_TOP_K }
    }

    /// Sets the maximum number of surface completions stored per prefix node.
    pub fn with_top_k_prefix(mut self, k: usize) -> Self {
        self.top_k_prefix = k.max(1);
        self
    }

    /// Sets the maximum number of surface realizations stored per stem.
    pub fn with_top_k_stem(mut self, k: usize) -> Self {
        self.top_k_stem = k.max(1);
        self
    }

    /// Number of unique `(surface, stem)` pairs accumulated so far.
    pub fn len(&self) -> usize {
        self.observations.len()
    }

    /// Returns `true` if no `(surface, stem)` observations have been recorded.
    pub fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }

    /// Accumulates `count` occurrences of `(surface, stem)`.
    ///
    /// Canonicalizes `surface` to NFC lowercase (preserving diacritics) so
    /// `"Courtesy"` and `"courtesy"` merge into `"courtesy"` while `"Café"`
    /// preserves `"café"`. Blank strings or zero counts are ignored.
    pub fn observe(&mut self, surface: &str, stem: &str, count: u64) {
        if count == 0 {
            return;
        }
        let canon = canonical_surface_lowercase(surface.trim());
        let stem_trimmed = stem.trim();
        if canon.is_empty() || stem_trimmed.is_empty() {
            return;
        }
        let entry = self.observations.entry((canon, stem_trimmed.to_owned())).or_default();
        *entry = entry.saturating_add(count);
    }

    /// Extracts UAX #29 alphabetic/alphanumeric word tokens from `text`,
    /// computes their stems with `stemmer`, and records document-frequency
    /// (`df = 1` per unique `(surface, stem)` in `text`) or term-frequency
    /// counts.
    pub fn observe_document(&mut self, text: &str, stemmer: &dyn Stemmer, count_doc_freq: bool) {
        let mut spans = Vec::new();
        let mut types = Vec::new();
        word_tokens(text, MAX_WORD_UTF16, &mut spans, &mut types);
        let mut seen: HashSet<(String, String)> = HashSet::new();
        for (sp, wt) in spans.into_iter().zip(types) {
            if wt == WordType::Numeric {
                continue;
            }
            let raw = &text[sp.start as usize..sp.end as usize];
            let canon = canonical_surface_lowercase(raw);
            if canon.is_empty() || !canon.chars().any(char::is_alphabetic) {
                continue;
            }
            let stem = stemmer.stem(&canon);
            if stem.is_empty() {
                continue;
            }
            if count_doc_freq {
                if seen.insert((canon.clone(), stem.clone())) {
                    self.observe(&canon, &stem, 1);
                }
            } else {
                self.observe(&canon, &stem, 1);
            }
        }
    }

    /// Compiles the accumulated observations into an aligned [`ImageBuf`].
    pub fn build_image(&self) -> Result<ImageBuf, LexiconBuildError> {
        Ok(ImageBuf::copy_from(&self.build_bytes()?))
    }

    /// Compiles the accumulated observations into raw `TSUGLEX1` image bytes.
    pub fn build_bytes(&self) -> Result<Vec<u8>, LexiconBuildError> {
        let mut unique_stems: Vec<String> = self.observations.keys().map(|(_, st)| st.clone()).collect();
        unique_stems.sort();
        unique_stems.dedup();
        let stem_map: HashMap<&str, u32> =
            unique_stems.iter().enumerate().map(|(i, s)| (s.as_str(), i as u32)).collect();

        let mut items: Vec<(String, String, u32, u64)> = self
            .observations
            .iter()
            .map(|((surf, st), &cnt)| (fold_surface_key(surf), surf.clone(), stem_map[st.as_str()], cnt))
            .collect();
        // Global rank order: highest count first, then canonical surface UTF-8 ascending, then stem_id.
        items.sort_by(|a, b| b.3.cmp(&a.3).then_with(|| a.1.cmp(&b.1)).then_with(|| a.2.cmp(&b.2)));

        let mut prefix_trie = MemTrie::new(self.top_k_prefix);
        let mut stem_lists: Vec<Vec<u32>> = vec![Vec::new(); unique_stems.len()];
        for (entry_id, (folded, _, stem_id, _)) in items.iter().enumerate() {
            let eid = entry_id as u32;
            prefix_trie.insert_ranked(folded, eid);
            let list = &mut stem_lists[*stem_id as usize];
            if list.len() < self.top_k_stem {
                list.push(eid);
            }
        }
        let mut stem_trie = MemTrie::new(self.top_k_stem);
        for (sid, stem) in unique_stems.iter().enumerate() {
            stem_trie.set_terminal(stem, &stem_lists[sid]);
        }

        let all_cps: Vec<u32> = prefix_trie.code_points().chain(stem_trie.code_points()).collect();
        let (codes, ascii, page_index, pages) = build_alphabet(&all_cps);
        let (prefix_cells, prefix_top_ids) = prefix_trie.compile(&codes)?;
        let (stem_cells, stem_top_ids) = stem_trie.compile(&codes)?;

        let mut payload = String::new();
        let mut stem_ends = Vec::with_capacity(unique_stems.len());
        for st in &unique_stems {
            payload.push_str(st);
            stem_ends.push(u32::try_from(payload.len()).map_err(|_| LexiconBuildError::TooLarge("payload"))?);
        }
        let mut entries = Vec::with_capacity(items.len());
        for (_, surf, stem_id, cnt) in &items {
            payload.push_str(surf);
            let surface_end = u32::try_from(payload.len()).map_err(|_| LexiconBuildError::TooLarge("payload"))?;
            entries.push(EntryRow {
                surface_end,
                stem_id: *stem_id,
                count_lo: *cnt as u32,
                count_hi: (*cnt >> 32) as u32,
            });
        }

        let counts = Counts {
            prefix_cells: u32::try_from(prefix_cells.len()).map_err(|_| LexiconBuildError::TooLarge("prefix_cells"))?,
            stem_cells: u32::try_from(stem_cells.len()).map_err(|_| LexiconBuildError::TooLarge("stem_cells"))?,
            entries: u32::try_from(entries.len()).map_err(|_| LexiconBuildError::TooLarge("entries"))?,
            stems: u32::try_from(stem_ends.len()).map_err(|_| LexiconBuildError::TooLarge("stems"))?,
            prefix_top_ids: u32::try_from(prefix_top_ids.len())
                .map_err(|_| LexiconBuildError::TooLarge("prefix_top_ids"))?,
            stem_top_ids: u32::try_from(stem_top_ids.len()).map_err(|_| LexiconBuildError::TooLarge("stem_top_ids"))?,
            pages: (pages.len() / 256) as u32,
            codes: codes.len() as u32,
            payload: u32::try_from(payload.len()).map_err(|_| LexiconBuildError::TooLarge("payload"))?,
        };
        let total_len = usize::try_from(counts.len()).map_err(|_| LexiconBuildError::TooLarge("image"))?;
        let mut out = Vec::with_capacity(total_len);
        out.extend_from_slice(&MAGIC);
        let head = [
            VERSION,
            ENDIAN,
            counts.prefix_cells,
            counts.stem_cells,
            counts.entries,
            counts.stems,
            counts.prefix_top_ids,
            counts.stem_top_ids,
            counts.pages,
            counts.codes,
            counts.payload,
            0,
        ];
        for w in head {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out.extend_from_slice(&[0u8; 8]); // checksum slot at bytes 56..64
        debug_assert_eq!(out.len(), HEADER);

        let put_words = |buf: &mut Vec<u8>, ws: &[u32]| {
            for w in ws {
                buf.extend_from_slice(&w.to_le_bytes());
            }
        };
        put_words(&mut out, &ascii);
        put_words(&mut out, &page_index);
        put_words(&mut out, &pages);
        for c in &prefix_cells {
            put_words(&mut out, &[c.base, c.check, c.top_offset, c.top_len]);
        }
        for c in &stem_cells {
            put_words(&mut out, &[c.base, c.check, c.top_offset, c.top_len]);
        }
        put_words(&mut out, &prefix_top_ids);
        put_words(&mut out, &stem_top_ids);
        for e in &entries {
            put_words(&mut out, &[e.surface_end, e.stem_id, e.count_lo, e.count_hi]);
        }
        put_words(&mut out, &stem_ends);
        out.extend_from_slice(payload.as_bytes());
        out.resize(total_len, 0);
        let sum = checksum(&out);
        out[56..64].copy_from_slice(&sum.to_le_bytes());
        Ok(out)
    }
}

#[derive(Default)]
struct MemNode {
    kids: BTreeMap<u32, usize>,
    top: Vec<u32>,
}

struct MemTrie {
    nodes: Vec<MemNode>,
    cap: usize,
}

impl MemTrie {
    fn new(cap: usize) -> Self {
        MemTrie { nodes: vec![MemNode::default()], cap }
    }

    fn insert_ranked(&mut self, key: &str, entry_id: u32) {
        let mut cur = 0usize;
        if self.nodes[cur].top.len() < self.cap {
            self.nodes[cur].top.push(entry_id);
        }
        for ch in key.chars() {
            let next = if let Some(&nxt) = self.nodes[cur].kids.get(&(ch as u32)) {
                nxt
            } else {
                let nxt = self.nodes.len();
                self.nodes.push(MemNode::default());
                self.nodes[cur].kids.insert(ch as u32, nxt);
                nxt
            };
            cur = next;
            if self.nodes[cur].top.len() < self.cap {
                self.nodes[cur].top.push(entry_id);
            }
        }
    }

    fn set_terminal(&mut self, key: &str, ids: &[u32]) {
        let mut cur = 0usize;
        for ch in key.chars() {
            let next = if let Some(&nxt) = self.nodes[cur].kids.get(&(ch as u32)) {
                nxt
            } else {
                let nxt = self.nodes.len();
                self.nodes.push(MemNode::default());
                self.nodes[cur].kids.insert(ch as u32, nxt);
                nxt
            };
            cur = next;
        }
        self.nodes[cur].top.extend(ids.iter().copied().take(self.cap));
    }

    fn code_points(&self) -> impl Iterator<Item = u32> + '_ {
        self.nodes.iter().flat_map(|n| n.kids.keys().copied())
    }

    fn compile(&self, codes: &HashMap<u32, u32>) -> Result<(Vec<TrieCell>, Vec<u32>), LexiconBuildError> {
        let states = self.nodes.len();
        let mut pos = vec![0u32; states];
        let mut base = vec![0u32; states];
        let mut used = vec![true];
        let mut free = 1usize;
        let mut queue = VecDeque::from([0usize]);

        while let Some(s) = queue.pop_front() {
            let mut kids: Vec<(u32, usize)> = self.nodes[s].kids.iter().map(|(&cp, &t)| (codes[&cp], t)).collect();
            if kids.is_empty() {
                continue;
            }
            kids.sort_unstable_by_key(|&(c, _)| c);
            let c0 = kids[0].0 as usize;
            let top = kids[kids.len() - 1].0 as usize;
            let mut p = free.max(c0);
            let b = loop {
                if used.len() <= p + top {
                    used.resize(p + top + 1, false);
                }
                let cand = p - c0;
                if !used[p] && kids.iter().all(|&(c, _)| !used[cand + c as usize]) {
                    break cand;
                }
                p += 1;
            };
            base[s] = u32::try_from(b).map_err(|_| LexiconBuildError::TooLarge("trie base"))?;
            for (c, t) in kids {
                let cell_idx = b + c as usize;
                used[cell_idx] = true;
                pos[t] = u32::try_from(cell_idx).map_err(|_| LexiconBuildError::TooLarge("trie cell"))?;
                queue.push_back(t);
            }
            while free < used.len() && used[free] {
                free += 1;
            }
        }
        let n_cells = used.iter().rposition(|&u| u).map_or(1, |i| i + 1);
        if n_cells as u64 >= u64::from(ROOT_CHECK) {
            return Err(LexiconBuildError::TooLarge("trie cells"));
        }
        let mut cells = vec![TrieCell { check: FREE, ..TrieCell::default() }; n_cells];
        cells[0].check = ROOT_CHECK;
        let mut top_ids = Vec::new();
        for (s, node) in self.nodes.iter().enumerate() {
            let c_idx = pos[s] as usize;
            cells[c_idx].base = base[s];
            cells[c_idx].top_offset =
                u32::try_from(top_ids.len()).map_err(|_| LexiconBuildError::TooLarge("top_ids"))?;
            cells[c_idx].top_len = u32::try_from(node.top.len()).map_err(|_| LexiconBuildError::TooLarge("top_len"))?;
            top_ids.extend_from_slice(&node.top);
            for &t in node.kids.values() {
                cells[pos[t] as usize].check = pos[s];
            }
        }
        Ok((cells, top_ids))
    }
}

#[allow(clippy::type_complexity)]
fn build_alphabet(cps: &[u32]) -> (HashMap<u32, u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut uses: HashMap<u32, u32> = HashMap::new();
    for &cp in cps {
        *uses.entry(cp).or_default() += 1;
    }
    let mut ordered: Vec<(u32, u32)> = uses.into_iter().collect();
    ordered.sort_unstable_by_key(|&(cp, n)| (std::cmp::Reverse(n), cp));
    let codes: HashMap<u32, u32> = ordered.iter().enumerate().map(|(i, &(cp, _))| (cp, i as u32 + 1)).collect();
    let mut ascii = vec![0u32; 128];
    let mut page_index = vec![0u32; PAGE_INDEX];
    let mut pages = Vec::new();
    let mut sorted_cps: Vec<u32> = codes.keys().copied().collect();
    sorted_cps.sort_unstable();
    for cp in sorted_cps {
        let code = codes[&cp];
        if cp < 128 {
            ascii[cp as usize] = code;
        } else {
            let page = &mut page_index[(cp >> 8) as usize];
            if *page == 0 {
                pages.resize(pages.len() + 256, 0);
                *page = (pages.len() / 256) as u32;
            }
            pages[(*page as usize - 1) * 256 + (cp & 255) as usize] = code;
        }
    }
    (codes, ascii, page_index, pages)
}
