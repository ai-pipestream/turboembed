//! Single-term SymSpell lookup over the symmetric delete index.

use std::collections::HashSet;

use super::deletes::Deletes;
use super::distance::Rows;
use super::suggestion::{Suggestion, Verbosity};
use super::symspell::{LookupError, SymSpell, prefix};

impl SymSpell {
    /// The suggestions for `term` within `max_edit_distance`, best first.
    pub fn lookup(
        &self,
        term: &str,
        verbosity: Verbosity,
        max_edit_distance: usize,
    ) -> Result<Vec<Suggestion<'_>>, LookupError> {
        self.check(max_edit_distance)?;
        Ok(self.lookup_within(term, verbosity, max_edit_distance))
    }

    /// `lookup` with `Verbosity::Top` and the largest edit distance.
    pub fn lookup_top(&self, term: &str) -> Vec<Suggestion<'_>> {
        self.lookup_within(term, Verbosity::Top, self.max_edit_distance())
    }

    pub(crate) fn check(&self, max_edit_distance: usize) -> Result<(), LookupError> {
        let max = self.config.max_dictionary_edit_distance();
        if max_edit_distance > max {
            return Err(LookupError { requested: max_edit_distance, max_dictionary_edit_distance: max });
        }
        Ok(())
    }

    /// `lookup` for a distance already checked against the index.
    pub(crate) fn lookup_within(&self, term: &str, verbosity: Verbosity, max_edit: usize) -> Vec<Suggestion<'_>> {
        let mut found = Vec::new();
        let term_chars: Vec<char> = term.chars().collect();
        let term_len = term_chars.len();
        // Longer than the longest word by more than the distance allows.
        if term_len > self.max_length + max_edit {
            return found;
        }
        if let Some(id) = self.word_id(term) {
            let count = self.counts[id as usize];
            if count >= self.config.count_threshold() {
                found.push(Suggestion { term: &self.words[id as usize], edit_distance: 0, count });
                if verbosity != Verbosity::All {
                    return found;
                }
            }
        }
        if max_edit == 0 {
            return found;
        }
        let mut search = Search {
            engine: self,
            term,
            term_chars: &term_chars,
            verbosity,
            max_edit2: max_edit,
            considered: HashSet::new(),
            rows: Rows::default(),
            word_chars: Vec::new(),
        };
        let prefix_length = self.config.prefix_length();
        let term_prefix_len = term_len.min(prefix_length);
        let mut deletes = Deletes::default();
        deletes.push(prefix(term, prefix_length));
        let mut i = 0;
        while i < deletes.len() {
            let (candidate, candidate_len) = deletes.get(i);
            let length_diff = term_prefix_len - candidate_len;
            // Already further from the query than the best so far.
            if length_diff > search.max_edit2 {
                if verbosity == Verbosity::All {
                    i += 1;
                    continue;
                }
                break;
            }
            for &id in self.filed(candidate) {
                if self.has_delete(id, candidate, candidate_len) {
                    search.visit(id, candidate, candidate_len, &mut found);
                }
            }
            if length_diff < max_edit
                && candidate_len <= prefix_length
                && (verbosity == Verbosity::All || length_diff < search.max_edit2)
            {
                deletes.expand(i);
            }
            i += 1;
        }
        if found.len() > 1 {
            found.sort();
        }
        found
    }
}

/// The state of one lookup over the candidates.
struct Search<'e, 't> {
    engine: &'e SymSpell,
    term: &'t str,
    term_chars: &'t [char],
    verbosity: Verbosity,
    /// The largest distance still worth a candidate: the maximum, lowered
    /// to the best distance found unless every suggestion is wanted.
    max_edit2: usize,
    considered: HashSet<u32>,
    rows: Rows,
    word_chars: Vec<char>,
}

impl<'e> Search<'e, '_> {
    /// Verifies the word `id` found under `candidate` and records it.
    fn visit(&mut self, id: u32, candidate: &str, candidate_len: usize, found: &mut Vec<Suggestion<'e>>) {
        let engine = self.engine;
        let word = engine.words[id as usize].as_str();
        if word == self.term {
            return;
        }
        let word_len = engine.lengths[id as usize] as usize;
        let term_len = self.term_chars.len();
        // Length checks before the costly verification.
        if word_len.abs_diff(term_len) > self.max_edit2
            || word_len < candidate_len
            || (word_len == candidate_len && word != candidate)
        {
            return;
        }
        let word_prefix_len = word_len.min(engine.config.prefix_length());
        if word_prefix_len > candidate_len && word_prefix_len - candidate_len > self.max_edit2 {
            return;
        }
        let distance = if candidate_len == 0 {
            // Only the empty delete was shared: the distance is the longer
            // length.
            let d = term_len.max(word_len);
            if d > self.max_edit2 || !self.considered.insert(id) {
                return;
            }
            d
        } else if word_len == 1 {
            let c = word.chars().next().expect("a word of one code point");
            let d = if self.term_chars.contains(&c) { term_len - 1 } else { term_len };
            if d > self.max_edit2 || !self.considered.insert(id) {
                return;
            }
            d
        } else {
            if !self.considered.insert(id) {
                return;
            }
            self.word_chars.clear();
            self.word_chars.extend(word.chars());
            let metric = engine.config.metric();
            match metric.chars(self.term_chars, &self.word_chars, self.max_edit2, &mut self.rows) {
                Some(d) => d,
                None => return,
            }
        };
        if distance > self.max_edit2 {
            return;
        }
        let count = engine.counts[id as usize];
        let item = Suggestion { term: word, edit_distance: distance, count };
        if let Some(best) = found.first() {
            match self.verbosity {
                // Keep only the closest found so far.
                Verbosity::Closest if distance < self.max_edit2 => found.clear(),
                Verbosity::Top => {
                    if distance < best.edit_distance || count > best.count {
                        self.max_edit2 = distance;
                        found[0] = item;
                    }
                    return;
                }
                _ => {}
            }
        }
        if self.verbosity != Verbosity::All {
            self.max_edit2 = distance;
        }
        found.push(item);
    }
}
