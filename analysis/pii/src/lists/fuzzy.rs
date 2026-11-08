//! Bounded Levenshtein matching of deny list terms over word tokens, with
//! an index that only compares a text word with the terms whose first word
//! could lie within the edit budget (#1935).
//!
//! A first word of at most [`INDEXED_LEN`] characters is indexed by its
//! deletion neighbourhood: the hashes of every string made by deleting up to
//! its own word budget of characters. When two words are within `k` edits,
//! deleting at most `k` characters from each gives the same string, so a
//! text word only needs the terms that share one of its own deletion hashes.
//! A hash collision only adds a candidate, which the banded Levenshtein then
//! rejects, so the index never changes a result. Longer first words are kept
//! in buckets by character length.

use std::collections::{BTreeMap, HashMap};
use std::hash::{BuildHasherDefault, Hasher};

use turbo_analysis::Span;
use turbo_glossary::{Candidate, Case, EntryIx};
use turbo_unicode::category::{GeneralCategory, general_category};
use turbo_unicode::java::to_lower_char;

use crate::text::letter_or_digit;

/// First words up to this many characters go in the deletion index.
const INDEXED_LEN: usize = 24;

#[derive(Clone, Debug)]
pub(super) struct FuzzyTerm {
    pub(super) entry: EntryIx,
    pub(super) words: Vec<Vec<char>>,
    pub(super) separators: Vec<String>,
}

/// Passes a `u64` that is already a hash straight through.
#[derive(Default)]
struct PassThrough(u64);

impl Hasher for PassThrough {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
        }
    }
    fn write_u64(&mut self, n: u64) {
        self.0 = n;
    }
}

type VariantMap = HashMap<u64, Vec<u32>, BuildHasherDefault<PassThrough>>;

/// The fuzzy terms of a deny list and the index over their first words.
#[derive(Clone, Debug, Default)]
pub(super) struct FuzzyIndex {
    pub(super) terms: Vec<FuzzyTerm>,
    variants: VariantMap,
    /// Distinct `(first word length, its word budget)` of indexed terms.
    shapes: Vec<(usize, u8)>,
    long: BTreeMap<usize, Vec<u32>>,
}

/// Work counters of one scan: term and window pairs checked in full.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct FuzzyStats {
    pub(super) compared: usize,
}

impl FuzzyIndex {
    pub(super) fn new(terms: Vec<FuzzyTerm>) -> FuzzyIndex {
        let mut index = FuzzyIndex { terms, ..FuzzyIndex::default() };
        let mut hashes = Vec::new();
        for (t, term) in index.terms.iter().enumerate() {
            let first = &term.words[0];
            let budget = max_word_edits(first.len());
            if first.len() > INDEXED_LEN {
                index.long.entry(first.len()).or_default().push(t as u32);
                continue;
            }
            if !index.shapes.contains(&(first.len(), budget)) {
                index.shapes.push((first.len(), budget));
            }
            hashes.clear();
            deletion_hashes(first, budget, &mut hashes);
            hashes.sort_unstable();
            hashes.dedup();
            for &h in &hashes {
                index.variants.entry(h).or_default().push(t as u32);
            }
        }
        index
    }

    pub(super) fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// Appends the candidates for the windows of `text` within `max_edits` of
    /// a term: for each window start and word count, the lowest entry that
    /// matches, which is the one `turbo_glossary::select` would keep of all
    /// the terms matching there.
    pub(super) fn scan(
        &self,
        text: &str,
        case: Case,
        max_edits: u8,
        capitalised_only: bool,
        candidates: &mut Vec<Candidate>,
    ) -> FuzzyStats {
        let mut stats = FuzzyStats::default();
        let words = word_spans(text);
        if words.is_empty() || self.terms.is_empty() {
            return stats;
        }
        let folded: Vec<Vec<char>> = words.iter().map(|&(s, e)| fold_chars(&text[s..e], case)).collect();
        let g = usize::from(max_edits);
        let mut useful = vec![false; INDEXED_LEN + g + 1];
        for &(len, budget) in &self.shapes {
            let r = usize::from(budget.min(max_edits));
            useful[len.saturating_sub(r)..=len + r].fill(true);
        }
        let mut band = Band::default();
        let mut hashes = Vec::new();
        let mut seen = vec![0usize; self.terms.len()];
        let mut picked: Vec<u32> = Vec::new();
        let mut found: Vec<usize> = Vec::new();
        for i in 0..words.len() {
            let word = &folded[i];
            picked.clear();
            if useful.get(word.len()).copied().unwrap_or(false) {
                hashes.clear();
                deletion_hashes(word, max_edits, &mut hashes);
                for h in &hashes {
                    if let Some(ts) = self.variants.get(h) {
                        for &t in ts {
                            if seen[t as usize] != i + 1 {
                                seen[t as usize] = i + 1;
                                picked.push(t);
                            }
                        }
                    }
                }
            }
            for (_, ts) in self.long.range(word.len().saturating_sub(g)..=word.len() + g) {
                picked.extend_from_slice(ts);
            }
            // Every term of `k` words gives the window at `i` the same span,
            // and selection keeps the lowest entry among equal spans and
            // token counts, so the first matching term of each `k` in entry
            // order is the only candidate that can be kept.
            picked.sort_unstable();
            found.clear();
            for &t in &picked {
                let ft = &self.terms[t as usize];
                if found.contains(&ft.words.len()) {
                    continue;
                }
                stats.compared += 1;
                if let Some(c) = window_match(text, &words, &folded, i, ft, max_edits, capitalised_only, &mut band) {
                    candidates.push(c);
                    found.push(ft.words.len());
                }
            }
        }
        stats
    }
}

/// Scratch rows of the banded Levenshtein.
#[derive(Default)]
pub(super) struct Band {
    prev: Vec<usize>,
    curr: Vec<usize>,
}

/// The candidate for term `ft` on the window of text words starting at `i`,
/// if every word is within its remaining edit budget.
#[allow(clippy::too_many_arguments)]
pub(super) fn window_match(
    text: &str,
    words: &[(usize, usize)],
    folded: &[Vec<char>],
    i: usize,
    ft: &FuzzyTerm,
    max_edits: u8,
    capitalised_only: bool,
    band: &mut Band,
) -> Option<Candidate> {
    let k = ft.words.len();
    if i + k > words.len() {
        return None;
    }
    let (start, end) = (words[i].0, words[i + k - 1].1);
    if capitalised_only && !is_capitalised(&text[start..end]) {
        return None;
    }
    if !separators_match(text, &words[i..i + k], &ft.separators) {
        return None;
    }
    let mut total = 0u8;
    for m in 0..k {
        let rem = (max_edits - total).min(max_word_edits(ft.words[m].len()));
        total += levenshtein_within(&folded[i + m], &ft.words[m], rem, band)?;
    }
    Some(Candidate { span: Span::new(start as u32, end as u32), tokens: k as u32, entry: ft.entry })
}

/// Pushes the hash of every string made by deleting up to `budget`
/// characters (at most `2`) from `word`, the word itself included.
fn deletion_hashes(word: &[char], budget: u8, out: &mut Vec<u64>) {
    let n = word.len();
    out.push(hash_skipping(word, n, n));
    if budget >= 1 {
        for a in 0..n {
            out.push(hash_skipping(word, a, n));
        }
    }
    if budget >= 2 {
        for a in 0..n {
            for b in a + 1..n {
                out.push(hash_skipping(word, a, b));
            }
        }
    }
}

/// FNV-1a over the characters of `word` except positions `a` and `b`
/// (an index of `word.len()` skips nothing), then length mixed in.
fn hash_skipping(word: &[char], a: usize, b: usize) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut len = 0u64;
    for (j, &c) in word.iter().enumerate() {
        if j == a || j == b {
            continue;
        }
        h = (h ^ u64::from(c as u32)).wrapping_mul(0x100_0000_01b3);
        len += 1;
    }
    (h ^ len.wrapping_mul(0x9e37_79b9_7f4a_7c15)).wrapping_mul(0x100_0000_01b3)
}

pub(super) fn is_capitalised(text: &str) -> bool {
    text.chars()
        .next()
        .is_some_and(|c| c.is_uppercase() || matches!(general_category(c), GeneralCategory::Lu | GeneralCategory::Lt))
}

pub(super) fn fold_chars(text: &str, case: Case) -> Vec<char> {
    match case {
        Case::Sensitive => text.chars().collect(),
        Case::Insensitive => text.chars().map(to_lower_char).collect(),
    }
}

/// The per-word edit floor: `0` under 4 characters, `1` for 4 or 5, `2` from 6.
pub(super) fn max_word_edits(word_len: usize) -> u8 {
    ((word_len.saturating_sub(2)) / 2).min(2) as u8
}

pub(super) fn word_spans(text: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start: Option<usize> = None;
    for (i, c) in text.char_indices() {
        if letter_or_digit(c) {
            if start.is_none() {
                start = Some(i);
            }
        } else if let Some(s) = start.take() {
            out.push((s, i));
        }
    }
    if let Some(s) = start {
        out.push((s, text.len()));
    }
    out
}

pub(super) fn build_fuzzy_term(entry: EntryIx, term: &str, case: Case) -> Option<FuzzyTerm> {
    let spans = word_spans(term);
    if spans.is_empty() {
        return None;
    }
    let words: Vec<Vec<char>> = spans.iter().map(|&(s, e)| fold_chars(&term[s..e], case)).collect();
    if words.iter().all(|w| max_word_edits(w.len()) == 0) {
        return None;
    }
    let separators: Vec<String> = spans.windows(2).map(|w| term[w[0].1..w[1].0].to_owned()).collect();
    Some(FuzzyTerm { entry, words, separators })
}

fn separators_match(text: &str, window: &[(usize, usize)], term_seps: &[String]) -> bool {
    for (w, want) in window.windows(2).zip(term_seps) {
        let gap = &text[w[0].1..w[1].0];
        if want.chars().all(char::is_whitespace) {
            if gap.is_empty() || !gap.chars().all(char::is_whitespace) {
                return false;
            }
        } else if gap != want {
            return false;
        }
    }
    true
}

pub(super) fn levenshtein_within(a: &[char], b: &[char], max: u8, band: &mut Band) -> Option<u8> {
    let (prev, curr) = (&mut band.prev, &mut band.curr);
    let max_usize = usize::from(max);
    if a.len().abs_diff(b.len()) > max_usize {
        return None;
    }
    if max == 0 {
        return (a == b).then_some(0);
    }
    let blen = b.len();
    if blen == 0 {
        return Some(a.len() as u8);
    }
    let inf = max_usize + 1;
    if prev.len() <= blen {
        prev.resize(blen + 1, inf);
        curr.resize(blen + 1, inf);
    }
    for (j, slot) in prev.iter_mut().enumerate().take((max_usize + 1).min(blen) + 1) {
        *slot = j;
    }
    for (i, &ca) in a.iter().enumerate() {
        let r = i + 1;
        let lo = r.saturating_sub(max_usize).max(1);
        let hi = (r + max_usize).min(blen);
        curr[lo - 1] = if lo == 1 { r } else { inf };
        if hi < blen {
            curr[hi + 1] = inf;
        }
        let mut row_min = curr[lo - 1];
        for j in lo..=hi {
            let cost = usize::from(ca != b[j - 1]);
            let v = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
            curr[j] = v;
            row_min = row_min.min(v);
        }
        if row_min > max_usize {
            return None;
        }
        std::mem::swap(prev, curr);
    }
    (prev[blen] <= max_usize).then_some(prev[blen] as u8)
}

#[cfg(test)]
#[path = "fuzzy_tests.rs"]
mod tests;
