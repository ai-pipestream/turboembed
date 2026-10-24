//! A first-word index over multi-word phrases, for matching
//! normalized token streams in one sweep instead of one window
//! scan per phrase.
//!
//! Phrases are grouped by first word (sorted, binary searched),
//! longest first within a group, so `matches_at` answers the
//! longest phrase starting at a token in time proportional to
//! that group's phrases, not to the whole table.

/// Phrases of normalized words, each carrying the index it was
/// built with. Build once, match many token streams.
#[derive(Clone, Debug, PartialEq)]
pub struct PhraseIndex {
    /// `(first word, start, end)` ranges into `phrases`, sorted
    /// by first word.
    groups: Vec<(String, u32, u32)>,
    /// `(rest of the phrase, payload)`, longest first in a group.
    phrases: Vec<(Vec<String>, u32)>,
}

impl PhraseIndex {
    /// Indexes `(words, payload)` phrases; empty phrases are
    /// skipped. Words are matched as given: normalize them (and
    /// the tokens) the same way on both sides.
    pub fn new(phrases: impl IntoIterator<Item = (Vec<String>, u32)>) -> PhraseIndex {
        let mut all: Vec<(Vec<String>, u32)> = phrases.into_iter().filter(|(w, _)| !w.is_empty()).collect();
        all.sort_by(|a, b| a.0[0].cmp(&b.0[0]).then(b.0.len().cmp(&a.0.len())).then_with(|| a.0.cmp(&b.0)));
        let mut groups: Vec<(String, u32, u32)> = Vec::new();
        for (i, (words, _)) in all.iter().enumerate() {
            match groups.last_mut() {
                Some(g) if g.0 == words[0] => g.2 = (i + 1) as u32,
                _ => groups.push((words[0].clone(), i as u32, (i + 1) as u32)),
            }
        }
        PhraseIndex { groups, phrases: all }
    }

    fn group_at<S: AsRef<str>>(&self, tokens: &[S], at: usize) -> Option<&[(Vec<String>, u32)]> {
        let first = tokens.get(at)?.as_ref();
        let g = self.groups.binary_search_by(|(w, _, _)| w.as_str().cmp(first)).ok()?;
        let (_, lo, hi) = self.groups[g];
        Some(&self.phrases[lo as usize..hi as usize])
    }

    /// The longest phrase starting at token `at`, as
    /// `(payload, word count)`, or `None`.
    pub fn matches_at<S: AsRef<str>>(&self, tokens: &[S], at: usize) -> Option<(u32, usize)> {
        self.group_at(tokens, at)?.iter().find_map(|(words, payload)| {
            (at + words.len() <= tokens.len() && words.iter().enumerate().all(|(k, w)| tokens[at + k].as_ref() == w))
                .then_some((*payload, words.len()))
        })
    }

    /// Every phrase starting at token `at`, longest first,
    /// appended to `out` as `(payload, word count)`. For callers
    /// where overlapping phrases at one start all count (nested
    /// terms, or one term under several payloads), unlike
    /// `matches_at`, which keeps only the longest.
    pub fn matches_all_at<S: AsRef<str>>(&self, tokens: &[S], at: usize, out: &mut Vec<(u32, usize)>) {
        let Some(group) = self.group_at(tokens, at) else { return };
        for (words, payload) in group {
            if at + words.len() <= tokens.len() && words.iter().enumerate().all(|(k, w)| tokens[at + k].as_ref() == w) {
                out.push((*payload, words.len()));
            }
        }
    }

    /// Whether no phrases are indexed.
    pub fn is_empty(&self) -> bool {
        self.phrases.is_empty()
    }

    /// How many phrases are indexed.
    pub fn len(&self) -> usize {
        self.phrases.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index() -> PhraseIndex {
        PhraseIndex::new([
            (vec!["new".into(), "york".into()], 10),
            (vec!["new".into(), "york".into(), "city".into()], 11),
            (vec!["york".into()], 12),
            (vec![], 99),
        ])
    }

    #[test]
    fn longest_phrase_at_a_token_wins() {
        let ix = index();
        let toks = ["in", "new", "york", "city", "today"];
        assert_eq!(ix.matches_at(&toks, 1), Some((11, 3)));
        assert_eq!(ix.matches_at(&toks, 2), Some((12, 1)));
        assert_eq!(ix.matches_at(&toks, 0), None);
        assert_eq!(ix.matches_at(&toks, 5), None);
        assert_eq!(ix.len(), 3);
    }

    #[test]
    fn all_matches_at_a_start_are_returned_longest_first() {
        let ix = PhraseIndex::new([
            (vec!["african".into()], 20),
            (vec!["african".into(), "american".into()], 21),
            (vec!["x".into()], 30),
            (vec!["x".into()], 31),
        ]);
        let mut out = Vec::new();
        ix.matches_all_at(&["an", "african", "american", "author"], 1, &mut out);
        assert_eq!(out, vec![(21, 2), (20, 1)], "nested phrases at one start both match");
        out.clear();
        ix.matches_all_at(&["x"], 0, &mut out);
        assert_eq!(out.len(), 2, "one term under two payloads matches both");
        assert!(out.contains(&(30, 1)) && out.contains(&(31, 1)));
    }

    #[test]
    fn phrases_ending_past_the_stream_do_not_match() {
        let ix = index();
        assert_eq!(ix.matches_at(&["new"], 0), None);
        assert_eq!(ix.matches_at(&["new", "york"], 0), Some((10, 2)));
    }
}
