//! Compiling a glossary into an image: the in-heap automaton's states laid
//! out as a double array.

use std::borrow::Cow;
use std::collections::{HashMap, VecDeque};

use turbo_glossary_layers::Glossary;

use super::{Cell, Counts, ENDIAN, FREE, HEADER, MAGIC, NO, PAGE_INDEX, ROOT_CHECK, VERSION, checksum};
use crate::error::BuildError;
use crate::text::{Case, TextMatcher};

/// Writes the image a [`super::MappedGlossaryMatcher`] reads.
pub struct MappedGlossaryWriter;

impl MappedGlossaryWriter {
    /// The image of [`TextMatcher::new`]`(glossary, case)`.
    pub fn write(glossary: &Glossary, case: Case) -> Result<Vec<u8>, BuildError> {
        Self::write_matcher(&TextMatcher::new(glossary, case)?, glossary)
    }

    /// The image of [`TextMatcher::with_normalizer`]: the patterns as
    /// `normalize` writes them, the ids and terms as registered.
    pub fn write_normalized<'g>(
        glossary: &'g Glossary,
        case: Case,
        normalize: impl FnMut(&'g str) -> Cow<'g, str>,
    ) -> Result<Vec<u8>, BuildError> {
        Self::write_matcher(&TextMatcher::with_normalizer(glossary, case, normalize)?, glossary)
    }

    /// The image of `matcher`, built from `glossary`.
    pub fn write_matcher(matcher: &TextMatcher, glossary: &Glossary) -> Result<Vec<u8>, BuildError> {
        let (mut edges, [fail, depth, entry, dict], longest) = matcher.tables();
        if let Some(&e) = entry.iter().filter(|&&e| e != NO).max()
            && e as usize >= glossary.len()
        {
            return Err(BuildError::NoEntry { entry: turbo_glossary_layers::EntryIx(e) });
        }
        let (codes, ascii, page_index, pages) = alphabet(&edges);
        for e in &mut edges {
            e.1 = codes[&e.1];
        }
        edges.sort_unstable();
        let place = place(fail.len(), &edges);
        let pos = |s: u32| if s == NO { NO } else { place.pos[s as usize] };
        let mut cells = vec![Cell { check: FREE, ..Cell::default() }; place.cells];
        for s in 0..fail.len() {
            let c = &mut cells[place.pos[s] as usize];
            c.base = place.base[s];
            c.fail = pos(fail[s]);
            c.entry = entry[s];
            c.dict = pos(dict[s]);
            c.depth = depth[s];
        }
        cells[0].check = ROOT_CHECK;
        for &(s, _, t) in &edges {
            cells[place.pos[t as usize] as usize].check = place.pos[s as usize];
        }
        let mut payload = String::new();
        let mut id_ends = Vec::with_capacity(glossary.len());
        let mut term_ends = Vec::with_capacity(glossary.len());
        for (_, e) in glossary.iter() {
            payload.push_str(e.id);
            id_ends.push(payload.len());
        }
        for (_, e) in glossary.iter() {
            payload.push_str(e.term);
            term_ends.push(payload.len());
        }
        let too_large = |what: &'static str| BuildError::TooLarge { what };
        let counts = Counts {
            case: matcher.case(),
            cells: u32::try_from(cells.len()).map_err(|_| too_large("cells"))?,
            entries: u32::try_from(glossary.len()).map_err(|_| too_large("entries"))?,
            longest,
            pages: (pages.len() / 256) as u32,
            codes: codes.len() as u32,
            payload: u32::try_from(payload.len()).map_err(|_| too_large("payload"))?,
        };
        let len = usize::try_from(counts.len()).map_err(|_| too_large("image"))?;
        let mut out = Vec::with_capacity(len);
        out.extend_from_slice(&MAGIC);
        let case_word = match counts.case {
            Case::Sensitive => 0,
            Case::Insensitive => 1,
        };
        let head = [VERSION, ENDIAN, case_word, counts.cells, counts.entries, longest, counts.pages, counts.codes];
        head.iter().chain([&counts.payload, &0]).for_each(|w| out.extend_from_slice(&w.to_le_bytes()));
        out.extend_from_slice(&[0; 16]);
        debug_assert_eq!(out.len(), HEADER);
        let words = |out: &mut Vec<u8>, ws: &[u32]| ws.iter().for_each(|w| out.extend_from_slice(&w.to_le_bytes()));
        words(&mut out, &ascii);
        words(&mut out, &page_index);
        words(&mut out, &pages);
        for c in &cells {
            words(&mut out, &[c.base, c.check, c.fail, c.entry, c.dict, c.depth]);
        }
        let ends = |v: &[usize]| v.iter().map(|&e| e as u32).collect::<Vec<u32>>();
        words(&mut out, &ends(&id_ends));
        words(&mut out, &ends(&term_ends));
        out.extend_from_slice(payload.as_bytes());
        debug_assert_eq!(out.len() as u64, HEADER as u64 + 4 * counts.words() + u64::from(counts.payload));
        out.resize(len, 0);
        out[56..64].copy_from_slice(&matcher.fingerprint().to_le_bytes());
        let sum = checksum(&out);
        out[48..56].copy_from_slice(&sum.to_le_bytes());
        Ok(out)
    }
}

/// Codes for the code points the edges carry, the most used first, and the
/// ASCII table, page index and pages that look them up.
#[allow(clippy::type_complexity)]
fn alphabet(edges: &[(u32, u32, u32)]) -> (HashMap<u32, u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut uses: HashMap<u32, u32> = HashMap::new();
    for e in edges {
        *uses.entry(e.1).or_default() += 1;
    }
    let mut by_use: Vec<(u32, u32)> = uses.into_iter().collect();
    by_use.sort_unstable_by_key(|&(cp, n)| (std::cmp::Reverse(n), cp));
    let codes: HashMap<u32, u32> = by_use.iter().enumerate().map(|(i, &(cp, _))| (cp, i as u32 + 1)).collect();
    let mut ascii = vec![0u32; 128];
    let mut page_index = vec![0u32; PAGE_INDEX];
    let mut pages = Vec::new();
    let mut cps: Vec<u32> = codes.keys().copied().collect();
    cps.sort_unstable();
    for cp in cps {
        let code = codes[&cp];
        if cp < 128 {
            ascii[cp as usize] = code;
            continue;
        }
        let page = &mut page_index[(cp >> 8) as usize];
        if *page == 0 {
            pages.resize(pages.len() + 256, 0);
            *page = (pages.len() / 256) as u32;
        }
        pages[(*page as usize - 1) * 256 + (cp & 255) as usize] = code;
    }
    (codes, ascii, page_index, pages)
}

struct Placement {
    /// The cell of each state.
    pos: Vec<u32>,
    /// The base of each state: its children are at `base + code`.
    base: Vec<u32>,
    cells: usize,
}

/// Places the states, breadth first from the root at cell 0: each state's
/// children go at the first base whose cells are all free. `edges` are
/// `(state, code, target)` sorted by state then code.
fn place(states: usize, edges: &[(u32, u32, u32)]) -> Placement {
    let mut first = vec![0usize; states + 1];
    for e in edges {
        first[e.0 as usize + 1] += 1;
    }
    for i in 0..states {
        first[i + 1] += first[i];
    }
    let mut pos = vec![NO; states];
    let mut base = vec![0u32; states];
    let mut used = vec![true];
    let mut free = 1usize;
    let mut queue = VecDeque::from([0u32]);
    pos[0] = 0;
    while let Some(s) = queue.pop_front() {
        let kids = &edges[first[s as usize]..first[s as usize + 1]];
        let Some(&(_, c0, _)) = kids.first() else { continue };
        let top = kids[kids.len() - 1].1 as usize;
        let mut p = free.max(c0 as usize);
        let b = loop {
            if used.len() <= p + top {
                used.resize(p + top + 1, false);
            }
            let b = p - c0 as usize;
            if !used[p] && kids.iter().all(|k| !used[b + k.1 as usize]) {
                break b;
            }
            p += 1;
        };
        base[s as usize] = b as u32;
        for &(_, c, t) in kids {
            used[b + c as usize] = true;
            pos[t as usize] = (b + c as usize) as u32;
            queue.push_back(t);
        }
        while free < used.len() && used[free] {
            free += 1;
        }
    }
    let cells = used.iter().rposition(|&u| u).map_or(1, |i| i + 1);
    Placement { pos, base, cells }
}
