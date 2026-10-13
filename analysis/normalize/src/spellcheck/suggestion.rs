//! What a lookup returns and how much of it.

use std::cmp::Ordering;

/// How many suggestions a lookup returns, as in the SymSpell reference
/// implementation and OpenNLP's `Verbosity`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Verbosity {
    /// The one suggestion with the smallest edit distance, the most
    /// frequent among those. The cheapest.
    Top,
    /// Every suggestion at the smallest edit distance found.
    Closest,
    /// Every suggestion within the maximum edit distance.
    All,
}

/// One suggestion: a dictionary word, its edit distance from the term
/// looked up, and its count in the dictionary. The word is borrowed from
/// the engine.
///
/// The order is OpenNLP's `SuggestItem` order: smaller edit distance
/// first, then higher count, then the term in UTF-16 code unit order (the
/// order of Java's `String.compareTo`), so that distinct suggestions never
/// compare equal.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Suggestion<'a> {
    pub term: &'a str,
    pub edit_distance: usize,
    pub count: u64,
}

impl Ord for Suggestion<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.edit_distance
            .cmp(&other.edit_distance)
            .then(other.count.cmp(&self.count))
            .then_with(|| self.term.encode_utf16().cmp(other.term.encode_utf16()))
    }
}

impl PartialOrd for Suggestion<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
