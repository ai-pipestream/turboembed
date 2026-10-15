//! Character-level Aho-Corasick glossary matcher with UAX #29 word boundary
//! checks and optional offset-preserving pre-normalization.

use std::sync::Arc;

use turbo_analysis::{Span, ViewBuilder};
use turbo_normalize::DynNormalizer;
use turbo_tokenize::uax29::word_boundaries;
use turbo_unicode::java::code_point;

use crate::selection;
use crate::{GlossaryEntry, GlossaryError, GlossaryMatch, GlossaryMatcher, is_blank};

#[derive(Clone, Copy)]
struct CandidateHit {
    start: u32,
    end: u32,
    pattern_index: u32,
}

/// A [`GlossaryMatcher`] backed by a character-level Aho-Corasick automaton
/// with UAX #29 word boundary checks and longest-match-first overlap
/// resolution.
///
/// Matches are filtered so both the start and end offsets fall on UAX #29
/// word boundaries in the original input text. Overlapping matches are
/// resolved deterministically by preferring the earliest start offset, then
/// the longest span, and finally the earliest entry in the input list.
pub struct AhoCorasickGlossaryMatcher {
    entries: Vec<GlossaryEntry>,
    ignore_case: bool,
    normalizer: Option<Arc<dyn DynNormalizer>>,
    edge_code_points: Vec<Box<[u32]>>,
    edge_targets: Vec<Box<[u32]>>,
    fail: Box<[u32]>,
    outputs: Vec<Box<[u32]>>,
    term_byte_lengths: Box<[u32]>,
    term_cp_lengths: Box<[u32]>,
}

impl std::fmt::Debug for AhoCorasickGlossaryMatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AhoCorasickGlossaryMatcher")
            .field("entries_len", &self.entries.len())
            .field("ignore_case", &self.ignore_case)
            .field("has_normalizer", &self.normalizer.is_some())
            .finish()
    }
}

impl AhoCorasickGlossaryMatcher {
    /// Builds an Aho-Corasick glossary matcher for `entries`.
    ///
    /// # Errors
    ///
    /// Returns [`GlossaryError::EmptyEntries`] if `entries` is empty.
    pub fn new(entries: &[GlossaryEntry], ignore_case: bool) -> Result<Self, GlossaryError> {
        Self::build(entries, ignore_case, None)
    }

    /// Builds an Aho-Corasick glossary matcher for `entries` that normalizes
    /// both glossary terms and input text through `normalizer`, mapping matches
    /// in the normalized text back to original byte spans via the normalizer's
    /// [`turbo_analysis::Alignment`].
    ///
    /// # Errors
    ///
    /// Returns [`GlossaryError::EmptyEntries`] if `entries` is empty, or
    /// [`GlossaryError::BlankNormalizedTerm`] if any entry's term normalizes
    /// to an empty or whitespace-only string.
    pub fn with_normalizer(
        entries: &[GlossaryEntry],
        ignore_case: bool,
        normalizer: Arc<dyn DynNormalizer>,
    ) -> Result<Self, GlossaryError> {
        Self::build(entries, ignore_case, Some(normalizer))
    }

    fn build(
        entries: &[GlossaryEntry],
        ignore_case: bool,
        normalizer: Option<Arc<dyn DynNormalizer>>,
    ) -> Result<Self, GlossaryError> {
        if entries.is_empty() {
            return Err(GlossaryError::EmptyEntries);
        }
        let entries_vec = entries.to_vec();
        let mut term_byte_lengths = Vec::with_capacity(entries_vec.len());
        let mut term_cp_lengths = Vec::with_capacity(entries_vec.len());

        let mut builder = AutomatonBuilder::new();
        for (i, entry) in entries_vec.iter().enumerate() {
            let normalized_owned;
            let raw_term = if let Some(norm) = &normalizer {
                let mut vb = ViewBuilder::new(entry.term());
                norm.normalize_dyn(entry.term(), &mut vb);
                normalized_owned = vb.finish().0;
                if is_blank(&normalized_owned) {
                    return Err(GlossaryError::BlankNormalizedTerm {
                        entry_id: entry.id().to_owned(),
                        term: entry.term().to_owned(),
                    });
                }
                normalized_owned.as_str()
            } else {
                entry.term()
            };
            term_byte_lengths.push(raw_term.len() as u32);
            let cp_len = builder.insert(raw_term, i as u32, ignore_case);
            term_cp_lengths.push(cp_len);
        }

        let (edge_code_points, edge_targets, fail, outputs) = builder.build();
        Ok(Self {
            entries: entries_vec,
            ignore_case,
            normalizer,
            edge_code_points,
            edge_targets,
            fail,
            outputs,
            term_byte_lengths: term_byte_lengths.into_boxed_slice(),
            term_cp_lengths: term_cp_lengths.into_boxed_slice(),
        })
    }

    #[inline]
    fn fold_code_point(&self, cp: u32) -> u32 {
        if self.ignore_case { code_point::to_lower_case(cp) } else { cp }
    }

    #[inline]
    fn step(&self, mut state: u32, cp: u32) -> u32 {
        loop {
            let next =
                Self::find_target(&self.edge_code_points[state as usize], &self.edge_targets[state as usize], cp);
            if let Some(target) = next {
                return target;
            }
            if state == 0 {
                return 0;
            }
            state = self.fail[state as usize];
        }
    }

    #[inline]
    fn find_target(keys: &[u32], targets: &[u32], cp: u32) -> Option<u32> {
        let len = keys.len();
        if len <= 8 {
            for i in 0..len {
                if keys[i] == cp {
                    return Some(targets[i]);
                }
                if keys[i] > cp {
                    return None;
                }
            }
            return None;
        }
        keys.binary_search(&cp).ok().map(|idx| targets[idx])
    }
}

impl GlossaryMatcher for AhoCorasickGlossaryMatcher {
    fn match_text(&self, text: &str) -> Vec<GlossaryMatch> {
        let normalized_pair = self.normalizer.as_ref().map(|norm| {
            let mut vb = ViewBuilder::new(text);
            norm.normalize_dyn(text, &mut vb);
            vb.finish()
        });
        let scan_text = match &normalized_pair {
            Some((norm_str, _)) => norm_str.as_str(),
            None => text,
        };
        if scan_text.is_empty() {
            return Vec::new();
        }

        // When ignore_case is enabled on non-ASCII scan_text, a lowercase mapping
        // such as U+212A (3 bytes) -> 'k' (1 byte) or U+1E9E (3 bytes) -> U+00DF (2 bytes)
        // can change UTF-8 byte width while preserving code-point count. Track code-point
        // start offsets in that case so norm_start is always exact.
        let need_cp_offsets = self.ignore_case && !scan_text.is_ascii();
        let mut cp_offsets: Vec<u32> =
            if need_cp_offsets { Vec::with_capacity(scan_text.len() + 1) } else { Vec::new() };

        let mut lazy_boundaries: Option<Vec<u32>> = None;
        let mut hits: Vec<CandidateHit> = Vec::new();
        let mut state: u32 = 0;

        for (cp_idx, (byte_idx, ch)) in scan_text.char_indices().enumerate() {
            if need_cp_offsets {
                cp_offsets.push(byte_idx as u32);
            }
            let cp = self.fold_code_point(ch as u32);
            state = self.step(state, cp);
            let norm_end = (byte_idx + ch.len_utf8()) as u32;
            let out = &self.outputs[state as usize];
            if !out.is_empty() {
                if need_cp_offsets {
                    cp_offsets.push(norm_end);
                }
                for &pattern_index in out.iter() {
                    let norm_start = if need_cp_offsets {
                        let cp_len = self.term_cp_lengths[pattern_index as usize] as usize;
                        cp_offsets[cp_idx + 1 - cp_len]
                    } else {
                        norm_end - self.term_byte_lengths[pattern_index as usize]
                    };
                    let orig_span = match &normalized_pair {
                        Some((_, alignment)) => alignment.span_to_parent(Span::new(norm_start, norm_end)),
                        None => Span::new(norm_start, norm_end),
                    };
                    if is_word_boundary(text, orig_span.start as usize, &mut lazy_boundaries)
                        && is_word_boundary(text, orig_span.end as usize, &mut lazy_boundaries)
                    {
                        hits.push(CandidateHit { start: orig_span.start, end: orig_span.end, pattern_index });
                    }
                }
                if need_cp_offsets {
                    cp_offsets.pop();
                }
            }
        }

        if hits.is_empty() {
            return Vec::new();
        }

        hits.sort_by(|a, b| {
            a.start.cmp(&b.start).then_with(|| b.end.cmp(&a.end)).then_with(|| a.pattern_index.cmp(&b.pattern_index))
        });

        let selected = selection::select(hits, |h| Span::new(h.start, h.end));
        selected
            .into_iter()
            .map(|hit| {
                let entry = &self.entries[hit.pattern_index as usize];
                GlossaryMatch::from_entry(Span::new(hit.start, hit.end), entry)
            })
            .collect()
    }
}

/// Checks whether `index` (a UTF-8 byte offset in `text`) is a UAX #29 word
/// boundary, using an ASCII fast path before lazily computing the full
/// boundary table with [`word_boundaries`].
#[inline]
pub(crate) fn is_word_boundary(text: &str, index: usize, lazy_boundaries: &mut Option<Vec<u32>>) -> bool {
    let bytes = text.as_bytes();
    let length = bytes.len();
    if index == 0 || index == length {
        return true;
    }
    if index > length || !text.is_char_boundary(index) {
        return false;
    }
    let prev = bytes[index - 1];
    let next = bytes[index];
    if prev < 0x80 && next < 0x80 {
        let prev_word = is_ascii_simple_word_byte(prev);
        let next_word = is_ascii_simple_word_byte(next);
        // WB5, WB8, WB9, WB10, WB13a, WB13b: never split adjacent ASCII
        // ALetter / Numeric / ExtendNumLet characters.
        if prev_word && next_word {
            return false;
        }
        // A transition between an ASCII word character and an ASCII non-word
        // character is always a UAX #29 boundary unless the non-word side is
        // one of the context-sensitive mid characters (WB6/WB7/WB11/WB12).
        if prev_word != next_word {
            let non_word = if prev_word { next } else { prev };
            if !is_ascii_context_sensitive_mid_byte(non_word) {
                return true;
            }
        }
    }
    let boundaries = lazy_boundaries.get_or_insert_with(|| {
        let mut out = Vec::new();
        word_boundaries(text, &mut out);
        out
    });
    boundaries.binary_search(&(index as u32)).is_ok()
}

#[inline]
const fn is_ascii_simple_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

#[inline]
const fn is_ascii_context_sensitive_mid_byte(b: u8) -> bool {
    matches!(b, b'\'' | b'"' | b'.' | b':' | b',' | b';')
}

struct AutomatonBuilder {
    keys: Vec<Vec<u32>>,
    targets: Vec<Vec<u32>>,
    outputs: Vec<Vec<u32>>,
}

impl AutomatonBuilder {
    fn new() -> Self {
        Self { keys: vec![Vec::new()], targets: vec![Vec::new()], outputs: vec![Vec::new()] }
    }

    fn insert(&mut self, term: &str, pattern_index: u32, ignore_case: bool) -> u32 {
        let mut state: u32 = 0;
        let mut cp_len: u32 = 0;
        for ch in term.chars() {
            let cp = if ignore_case { code_point::to_lower_case(ch as u32) } else { ch as u32 };
            state = self.get_or_add_child(state, cp);
            cp_len += 1;
        }
        self.outputs[state as usize].push(pattern_index);
        cp_len
    }

    fn get_or_add_child(&mut self, state: u32, cp: u32) -> u32 {
        let k = &self.keys[state as usize];
        match k.binary_search(&cp) {
            Ok(idx) => self.targets[state as usize][idx],
            Err(ins) => {
                let next_state = self.keys.len() as u32;
                self.keys.push(Vec::new());
                self.targets.push(Vec::new());
                self.outputs.push(Vec::new());
                self.keys[state as usize].insert(ins, cp);
                self.targets[state as usize].insert(ins, next_state);
                next_state
            }
        }
    }

    #[allow(clippy::type_complexity)]
    fn build(mut self) -> (Vec<Box<[u32]>>, Vec<Box<[u32]>>, Box<[u32]>, Vec<Box<[u32]>>) {
        let num_states = self.keys.len();
        let mut edge_code_points = Vec::with_capacity(num_states);
        let mut edge_targets = Vec::with_capacity(num_states);
        for (k, t) in self.keys.into_iter().zip(self.targets) {
            edge_code_points.push(k.into_boxed_slice());
            edge_targets.push(t.into_boxed_slice());
        }

        let mut fail = vec![0u32; num_states];
        let mut queue = Vec::with_capacity(num_states);

        for &target in edge_targets[0].iter() {
            fail[target as usize] = 0;
            queue.push(target);
        }

        let mut head = 0;
        while head < queue.len() {
            let r = queue[head];
            head += 1;
            let r_keys = &edge_code_points[r as usize];
            let r_targets = &edge_targets[r as usize];
            for i in 0..r_keys.len() {
                let cp = r_keys[i];
                let u = r_targets[i];
                queue.push(u);

                let mut f = fail[r as usize];
                let fallback = loop {
                    if let Some(target) = AhoCorasickGlossaryMatcher::find_target(
                        &edge_code_points[f as usize],
                        &edge_targets[f as usize],
                        cp,
                    ) {
                        break target;
                    }
                    if f == 0 {
                        break 0;
                    }
                    f = fail[f as usize];
                };
                fail[u as usize] = fallback;
                if !self.outputs[fallback as usize].is_empty() {
                    let inherited = self.outputs[fallback as usize].clone();
                    self.outputs[u as usize].extend_from_slice(&inherited);
                }
            }
        }

        let compiled_outputs = self.outputs.into_iter().map(Vec::into_boxed_slice).collect();

        (edge_code_points, edge_targets, fail.into_boxed_slice(), compiled_outputs)
    }
}
