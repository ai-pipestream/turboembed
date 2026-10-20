//! Glossary terms found by their characters: an Aho-Corasick automaton
//! over code points, OpenNLP's `AhoCorasickGlossaryMatcher`.

use std::borrow::Cow;
use std::hash::{DefaultHasher, Hash, Hasher};

use turbo_analysis::Span;
use turbo_glossary_layers::{EntryIx, Glossary, is_blank};
use turbo_unicode::java::to_lower_char;

use crate::boundaries::Boundaries;
use crate::error::BuildError;
use crate::select::{Candidate, GlossaryMatch, select};

/// How a character matcher compares letters.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum Case {
    /// Character for character.
    #[default]
    Sensitive,
    /// Through the simple lowercase mapping of each code point
    /// (`Character.toLowerCase(int)`), which keeps every character one
    /// character. Expansions such as sharp s to "ss" belong to a
    /// normalized view.
    Insensitive,
}

impl Case {
    #[inline]
    pub(crate) fn fold(self, c: char) -> u32 {
        match self {
            Case::Sensitive => c as u32,
            Case::Insensitive => to_lower_char(c) as u32,
        }
    }
}

/// No state, no entry.
const NO: u32 = u32::MAX;

/// [`TextMatcher::tables`]: the edges, the per-state columns and the
/// longest pattern.
pub(crate) type Tables<'a> = (Vec<(u32, u32, u32)>, [&'a [u32]; 4], u32);
const ROOT: u32 = 0;

/// The goto edges below the root: `(state, code point)` packed into a
/// `u64`, in one open addressing table with linear probing, kept below
/// seven tenths full.
#[derive(Clone, Debug)]
struct EdgeTable {
    keys: Vec<u64>,
    targets: Vec<u32>,
    len: usize,
}

const EMPTY: u64 = u64::MAX;

impl EdgeTable {
    fn new() -> EdgeTable {
        EdgeTable { keys: vec![EMPTY; 16], targets: vec![0; 16], len: 0 }
    }

    #[inline]
    fn slot(key: u64, mask: usize) -> usize {
        // A multiplicative hash: the high bits of the product spread both
        // the state and the code point.
        (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) as usize & mask
    }

    #[inline]
    fn get(&self, state: u32, cp: u32) -> Option<u32> {
        let key = (u64::from(state) << 32) | u64::from(cp);
        let mask = self.keys.len() - 1;
        let mut i = Self::slot(key, mask);
        loop {
            let k = self.keys[i];
            if k == key {
                return Some(self.targets[i]);
            }
            if k == EMPTY {
                return None;
            }
            i = (i + 1) & mask;
        }
    }

    fn insert(&mut self, state: u32, cp: u32, target: u32) {
        if (self.len + 1) * 10 > self.keys.len() * 7 {
            let size = self.keys.len() * 2;
            let old_keys = std::mem::replace(&mut self.keys, vec![EMPTY; size]);
            let old_targets = std::mem::replace(&mut self.targets, vec![0; size]);
            self.len = 0;
            for (k, t) in old_keys.into_iter().zip(old_targets) {
                if k != EMPTY {
                    self.put(k, t);
                }
            }
        }
        self.put((u64::from(state) << 32) | u64::from(cp), target);
    }

    fn put(&mut self, key: u64, target: u32) {
        let mask = self.keys.len() - 1;
        let mut i = Self::slot(key, mask);
        while self.keys[i] != EMPTY {
            i = (i + 1) & mask;
        }
        self.keys[i] = key;
        self.targets[i] = target;
        self.len += 1;
    }

    fn heap_bytes(&self) -> usize {
        self.keys.capacity() * 8 + self.targets.capacity() * 4
    }
}

/// Finds glossary terms by their characters, then keeps the ones whose
/// ends are on boundaries the caller gives, and resolves overlaps
/// leftmost first, then longest, then by registration order.
///
/// Terms match literally (or through [`Case::Insensitive`]), so a term of
/// several words matches only with the separators it was registered
/// with. Text normalized beforehand (whitespace collapsed, dashes or case
/// folded) is a view of its own: match over the view with the terms
/// normalized the same way ([`TextMatcher::with_normalizer`]), and the
/// view's alignment takes the matches back to the source.
///
/// The automaton is built once; matching takes `&self`, so one matcher
/// serves any number of threads.
#[derive(Clone, Debug)]
pub struct TextMatcher {
    case: Case,
    /// The root's edges for ASCII code points; `ROOT` where there is none.
    root_ascii: Box<[u32; 128]>,
    edges: EdgeTable,
    /// Per state: the failure link, how many characters it is from the
    /// root, the entry whose term ends there (the first registered, or
    /// `NO`), the next state down the failure chain that has an entry,
    /// and whether it has any edge.
    fail: Vec<u32>,
    depth: Vec<u32>,
    entry: Vec<u32>,
    dict: Vec<u32>,
    leaf: Vec<bool>,
    longest: u32,
    /// A hash of the patterns as stored and the entry each one reports.
    fingerprint: u64,
}

impl TextMatcher {
    /// The automaton over the terms of `glossary` as they are.
    pub fn new(glossary: &Glossary, case: Case) -> Result<TextMatcher, BuildError> {
        Self::with_normalizer(glossary, case, Cow::Borrowed)
    }

    /// The automaton over the terms of `glossary` as `normalize` writes
    /// them: the normalizer the view to be matched was made with. A term
    /// that normalizes to blank is refused. Two terms that come out the
    /// same are one pattern, which reports the first one registered.
    pub fn with_normalizer<'g>(
        glossary: &'g Glossary,
        case: Case,
        mut normalize: impl FnMut(&'g str) -> Cow<'g, str>,
    ) -> Result<TextMatcher, BuildError> {
        if glossary.is_empty() {
            return Err(BuildError::Empty);
        }
        let mut m = TextMatcher {
            case,
            root_ascii: Box::new([ROOT; 128]),
            edges: EdgeTable::new(),
            fail: vec![ROOT],
            depth: vec![0],
            entry: vec![NO],
            dict: vec![NO],
            leaf: vec![true],
            longest: 0,
            fingerprint: 0,
        };
        let mut h = DefaultHasher::new();
        case.hash(&mut h);
        // The children of each state as a linked list, for the breadth
        // first pass below: the first child, and each child's next
        // sibling with the code point that leads to it.
        let mut first_child: Vec<u32> = vec![NO];
        let mut next_sibling: Vec<u32> = vec![NO];
        let mut label: Vec<u32> = vec![0];
        for (ix, e) in glossary.iter() {
            let pattern = normalize(e.term);
            if is_blank(&pattern) {
                return Err(BuildError::BlankTerm { entry: ix });
            }
            let mut state = ROOT;
            for c in pattern.chars() {
                let cp = case.fold(c);
                let next = m.edge(state, cp);
                state = match next {
                    Some(n) => n,
                    None => {
                        let made = m.fail.len() as u32;
                        m.fail.push(ROOT);
                        m.depth.push(m.depth[state as usize] + 1);
                        m.entry.push(NO);
                        m.dict.push(NO);
                        m.leaf.push(true);
                        first_child.push(NO);
                        next_sibling.push(first_child[state as usize]);
                        label.push(cp);
                        first_child[state as usize] = made;
                        m.leaf[state as usize] = false;
                        if state == ROOT && cp < 128 {
                            m.root_ascii[cp as usize] = made;
                        } else {
                            m.edges.insert(state, cp, made);
                        }
                        made
                    }
                };
            }
            if m.entry[state as usize] == NO {
                m.entry[state as usize] = ix.0;
                ix.hash(&mut h);
                state.hash(&mut h);
                pattern.chars().map(|c| case.fold(c)).for_each(|cp| cp.hash(&mut h));
            }
            m.longest = m.longest.max(m.depth[state as usize]);
        }
        m.fingerprint = h.finish();
        // Failure links breadth first, so a state's link is set before
        // its children's.
        let mut queue = std::collections::VecDeque::new();
        let mut c = first_child[ROOT as usize];
        while c != NO {
            queue.push_back(c);
            c = next_sibling[c as usize];
        }
        while let Some(s) = queue.pop_front() {
            let mut c = first_child[s as usize];
            while c != NO {
                let f = m.step(m.fail[s as usize], label[c as usize]);
                m.fail[c as usize] = f;
                m.dict[c as usize] = if m.entry[f as usize] != NO { f } else { m.dict[f as usize] };
                queue.push_back(c);
                c = next_sibling[c as usize];
            }
        }
        Ok(m)
    }

    #[inline]
    fn edge(&self, state: u32, cp: u32) -> Option<u32> {
        if state == ROOT && cp < 128 {
            let t = self.root_ascii[cp as usize];
            return (t != ROOT).then_some(t);
        }
        if self.leaf[state as usize] {
            return None;
        }
        self.edges.get(state, cp)
    }

    /// The state after `cp` from `state`, following failure links on a miss.
    #[inline]
    fn step(&self, mut state: u32, cp: u32) -> u32 {
        loop {
            if let Some(n) = self.edge(state, cp) {
                return n;
            }
            if state == ROOT {
                return ROOT;
            }
            state = self.fail[state as usize];
        }
    }

    pub fn case(&self) -> Case {
        self.case
    }

    /// A hash of the patterns the automaton holds and the entry each one
    /// reports, which differs between matchers built with different
    /// normalizers or case.
    pub(crate) fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    /// How many states, the root included.
    pub fn states(&self) -> usize {
        self.fail.len()
    }

    /// The bytes the matcher holds on the heap; the allocator's own
    /// overhead is not counted.
    pub fn heap_bytes(&self) -> usize {
        self.edges.heap_bytes()
            + 128 * 4
            + self.fail.capacity() * 4
            + self.depth.capacity() * 4
            + self.entry.capacity() * 4
            + self.dict.capacity() * 4
            + self.leaf.capacity()
    }

    /// The automaton as plain tables, for [`crate::MappedGlossaryWriter`]:
    /// every goto edge `(state, code point, target)`, in no order, and per
    /// state the failure link, depth, entry and next state with an entry.
    pub(crate) fn tables(&self) -> Tables<'_> {
        let mut edges: Vec<(u32, u32, u32)> = Vec::with_capacity(self.fail.len());
        for (cp, &t) in self.root_ascii.iter().enumerate() {
            if t != ROOT {
                edges.push((ROOT, cp as u32, t));
            }
        }
        for (&k, &t) in self.edges.keys.iter().zip(&self.edges.targets) {
            if k != EMPTY {
                edges.push(((k >> 32) as u32, k as u32, t));
            }
        }
        (edges, [&self.fail, &self.depth, &self.entry, &self.dict], self.longest)
    }

    /// Calls `hit` with every occurrence of every pattern in `text`, in
    /// the order their ends are reached, before any boundary check or
    /// overlap resolution. Spans are byte offsets into `text`.
    pub fn scan(&self, text: &str, mut hit: impl FnMut(Span, EntryIx)) {
        // The byte offset of each of the last `ring.len()` characters, so
        // a hit's start is found from its length in characters. No hit is
        // longer than the text.
        let size = ((self.longest as usize).min(text.len()) + 1).next_power_of_two();
        let mask = size - 1;
        let mut ring = vec![0u32; size];
        let mut seen = 0usize;
        let mut state = ROOT;
        for (i, c) in text.char_indices() {
            ring[seen & mask] = i as u32;
            seen += 1;
            state = self.step(state, self.case.fold(c));
            let end = (i + c.len_utf8()) as u32;
            let mut s = if self.entry[state as usize] != NO { state } else { self.dict[state as usize] };
            while s != NO {
                let start = ring[(seen - self.depth[s as usize] as usize) & mask];
                hit(Span::new(start, end), EntryIx(self.entry[s as usize]));
                s = self.dict[s as usize];
            }
        }
    }

    /// Appends to `out` the matches in `text` whose both ends are in
    /// `boundaries`, in text order and without overlaps.
    pub fn find(&self, text: &str, boundaries: &Boundaries, out: &mut Vec<GlossaryMatch>) {
        let mut candidates = Vec::new();
        self.scan(text, |span, entry| {
            if boundaries.around(span) {
                candidates.push(Candidate { span, tokens: 0, entry });
            }
        });
        select(&mut candidates, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn glossary(entries: &[(&str, &str)]) -> Glossary {
        let mut g = Glossary::new();
        for (id, term) in entries {
            g.push(id, term).unwrap();
        }
        g
    }

    fn every_offset(text: &str) -> Boundaries {
        Boundaries::from_offsets(text.len() as u32, 0..=text.len() as u32)
    }

    fn hits(m: &TextMatcher, text: &str) -> Vec<(u32, u32, u32)> {
        let mut out = Vec::new();
        m.scan(text, |s, e| out.push((s.start, s.end, e.0)));
        out.sort_unstable();
        out
    }

    #[test]
    fn every_occurrence_with_failure_links() {
        let g = glossary(&[("1", "he"), ("2", "she"), ("3", "his"), ("4", "hers")]);
        let m = TextMatcher::new(&g, Case::Sensitive).unwrap();
        assert_eq!(hits(&m, "ushers"), [(1, 4, 1), (2, 4, 0), (2, 6, 3)]);
    }

    #[test]
    fn byte_offsets_over_multibyte_text() {
        let g = glossary(&[("de", "Größe"), ("jp", "東京"), ("emoji", "🙂 ok")]);
        let m = TextMatcher::new(&g, Case::Insensitive).unwrap();
        let text = "Die GRÖßE in 東京 🙂 OK";
        let found = hits(&m, text);
        let spans: Vec<&str> = found.iter().map(|&(s, e, _)| &text[s as usize..e as usize]).collect();
        assert_eq!(spans, ["GRÖßE", "東京", "🙂 OK"]);
    }

    #[test]
    fn duplicates_report_the_first_entry_and_blank_patterns_are_refused() {
        let g = glossary(&[("a", "Cat"), ("b", "cat")]);
        let m = TextMatcher::new(&g, Case::Insensitive).unwrap();
        assert_eq!(hits(&m, "CAT"), [(0, 3, 0)]);
        let m = TextMatcher::new(&g, Case::Sensitive).unwrap();
        assert_eq!(hits(&m, "cat Cat"), [(0, 3, 1), (4, 7, 0)]);
        let g = glossary(&[("a", "x"), ("b", "-")]);
        let err = TextMatcher::with_normalizer(&g, Case::Sensitive, |t| Cow::Owned(t.replace('-', " "))).unwrap_err();
        assert_eq!(err, BuildError::BlankTerm { entry: EntryIx(1) });
        assert_eq!(TextMatcher::new(&Glossary::new(), Case::Sensitive).unwrap_err(), BuildError::Empty);
    }

    #[test]
    fn boundaries_and_overlaps() {
        let g = glossary(&[("ny", "New York"), ("nyc", "New York City"), ("york", "York City"), ("cat", "cat")]);
        let m = TextMatcher::new(&g, Case::Sensitive).unwrap();
        let text = "New York City concatenate cat";
        let mut out = Vec::new();
        m.find(text, &every_offset(text), &mut out);
        let got: Vec<(&str, u32)> = out.iter().map(|x| (&text[x.span.range()], x.entry.0)).collect();
        assert_eq!(got, [("New York City", 1), ("cat", 3), ("cat", 3)]);
        let mut words = Vec::new();
        turbo_tokenize::word_boundaries(text, &mut words);
        let mut out = Vec::new();
        m.find(text, &Boundaries::from_offsets(text.len() as u32, words), &mut out);
        let got: Vec<&str> = out.iter().map(|x| &text[x.span.range()]).collect();
        assert_eq!(got, ["New York City", "cat"]);
    }

    #[test]
    fn a_long_term_and_a_short_text() {
        let long = "ab".repeat(50_000);
        let g = glossary(&[("long", &long), ("ab", "ab"), ("é", "éé")]);
        let m = TextMatcher::new(&g, Case::Sensitive).unwrap();
        // The text is shorter than the longest term, and every hit fits it.
        assert_eq!(hits(&m, "abéé"), [(0, 2, 1), (2, 6, 2)]);
        assert_eq!(hits(&m, "ab"), [(0, 2, 1)]);
        assert_eq!(hits(&m, &long).len(), 50_001);
    }

    #[test]
    fn empty_text_has_no_matches() {
        let g = glossary(&[("a", "a")]);
        let m = TextMatcher::new(&g, Case::Sensitive).unwrap();
        let mut out = Vec::new();
        m.find("", &Boundaries::from_offsets(0, [0]), &mut out);
        assert!(out.is_empty());
    }
}
