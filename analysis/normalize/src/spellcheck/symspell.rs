//! The symmetric delete engine.
//!
//! Every indexed word is reduced to the strings its first `prefix_length`
//! code points become when up to `max_dictionary_edit_distance` of them
//! are deleted, and the index maps each such delete to the words that
//! produce it. A query is reduced the same way, and the words found under
//! its deletes are the candidates, which the configured metric verifies.
//! The lookup is a port of OpenNLP's `SymSpell.lookup`, with the same
//! candidate order, the same pruning and the same tie-breaking, so it
//! returns the same suggestions in the same order.
//!
//! Memory layout. The words are one `Vec<String>` in the order they were
//! first added, with their counts and their lengths in code points in two
//! vectors beside it; a word's position is its id. Words are found by an
//! open-addressing table of ids keyed by the word's hash, so no word is
//! stored twice. The deletes index maps the 64-bit hash of a delete string
//! to its words: one id stored inline when one word produces the delete
//! (most deletes), or the position of a list of ids when several do. The
//! delete strings themselves are not stored, which is most of the memory
//! a map from strings would take (OpenNLP keeps every delete as a Java
//! `String`). Because two delete strings can share a hash, a lookup checks
//! that the delete it probes for really is a delete of each word it finds
//! (a subsequence test on at most `prefix_length` code points), so the
//! candidates are exactly OpenNLP's. A lookup allocates only the arena of
//! its own deletes and the code points of the query.

use std::collections::{HashMap, HashSet};
use std::fmt;

use super::config::{MAX_COUNT, SymSpellConfig};
use super::deletes::Deletes;
use super::distance::Rows;
use super::suggestion::{Suggestion, Verbosity};
use super::table::{BuildPrehashed, IdTable, fmix, hash_str};

/// Marks an index entry that points into `lists` instead of holding a word
/// id itself.
const LIST: u32 = 1 << 31;

/// A symmetric delete spelling corrector over a frequency dictionary.
#[derive(Clone)]
pub struct SymSpell {
    pub(crate) config: SymSpellConfig,
    words: Vec<String>,
    counts: Vec<u64>,
    lengths: Vec<u32>,
    by_word: IdTable,
    deletes: HashMap<u64, u32, BuildPrehashed>,
    lists: Vec<Vec<u32>>,
    pub(crate) bigrams: HashMap<String, u64>,
    pub(crate) min_bigram_count: u64,
    max_bigram_count: u64,
    /// The length in code points of the longest indexed word.
    max_length: usize,
    /// The sum of every count added, the derived corpus size.
    total_count: u64,
    fingerprint: u64,
    /// The deletes of the word being indexed, kept to reuse its memory.
    scratch: Deletes,
}

impl fmt::Debug for SymSpell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SymSpell")
            .field("config", &self.config)
            .field("words", &self.words.len())
            .field("deletes", &self.deletes.len())
            .field("bigrams", &self.bigrams.len())
            .field("fingerprint", &self.fingerprint)
            .finish()
    }
}

impl Default for SymSpell {
    fn default() -> SymSpell {
        SymSpell::new(SymSpellConfig::default())
    }
}

/// A lookup asked for a larger edit distance than the index answers.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct LookupError {
    pub requested: usize,
    pub max_dictionary_edit_distance: usize,
}

impl fmt::Display for LookupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "maxEditDistance ({}) must not exceed maxDictionaryEditDistance ({})",
            self.requested, self.max_dictionary_edit_distance
        )
    }
}

impl std::error::Error for LookupError {}

/// `a + b`, or [`MAX_COUNT`] when that is larger.
pub(crate) fn saturated_add(a: u64, b: u64) -> u64 {
    a.saturating_add(b).min(MAX_COUNT)
}

impl SymSpell {
    pub fn new(config: SymSpellConfig) -> SymSpell {
        SymSpell {
            config,
            words: Vec::new(),
            counts: Vec::new(),
            lengths: Vec::new(),
            by_word: IdTable::default(),
            deletes: HashMap::default(),
            lists: Vec::new(),
            bigrams: HashMap::new(),
            min_bigram_count: MAX_COUNT,
            max_bigram_count: 0,
            max_length: 0,
            total_count: 0,
            fingerprint: hash_str(&format!("{config:?}")),
            scratch: Deletes::default(),
        }
    }

    pub fn config(&self) -> &SymSpellConfig {
        &self.config
    }

    /// The largest edit distance a lookup may ask for.
    pub fn max_edit_distance(&self) -> usize {
        self.config.max_dictionary_edit_distance()
    }

    /// The number of distinct words added, indexed or not.
    pub fn word_count(&self) -> usize {
        self.words.len()
    }

    /// The number of distinct delete hashes in the index (OpenNLP's
    /// `entryCount`, the number of distinct delete strings, unless two of
    /// them share a 64-bit hash).
    pub fn entry_count(&self) -> usize {
        self.deletes.len()
    }

    /// The number of distinct bigrams.
    pub fn bigram_count(&self) -> usize {
        self.bigrams.len()
    }

    /// The count of the bigram `w1 w2`.
    pub fn bigram(&self, w1: &str, w2: &str) -> Option<u64> {
        self.bigrams.get(&format!("{w1} {w2}")).copied()
    }

    /// The largest bigram count added.
    pub fn max_bigram_count(&self) -> u64 {
        self.max_bigram_count
    }

    /// The accumulated count of `word`, whether or not it reached the
    /// threshold.
    pub fn count(&self, word: &str) -> Option<u64> {
        self.word_id(word).map(|id| self.counts[id as usize])
    }

    /// Whether `word` is indexed: its count has reached the threshold.
    pub fn contains(&self, word: &str) -> bool {
        self.count(word).is_some_and(|c| c >= self.config.count_threshold())
    }

    /// The corpus size N of the compound scoring: the fixed value, or the
    /// sum of the counts added (at least 1).
    pub fn corpus_word_count(&self) -> u64 {
        match self.config.corpus_word_count() {
            0 => self.total_count.max(1),
            n => n,
        }
    }

    /// A hash of everything added, in order (words and bigrams with their
    /// counts) and of the configuration. Two engines built by the same
    /// calls have the same fingerprint, so it can stand for the dictionary
    /// in a normalizer's configuration hash.
    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    fn mix(&mut self, parts: &[u64]) {
        for &p in parts {
            self.fingerprint = fmix(self.fingerprint.rotate_left(17) ^ p);
        }
    }

    fn word_id(&self, word: &str) -> Option<u32> {
        self.by_word.find(hash_str(word), |id| self.words[id as usize] == word)
    }

    /// Adds `count` to `word`, adding the word if it is new, and indexes it
    /// once its count reaches the threshold. Every count, even one of a
    /// word below the threshold, adds to the corpus size. A zero count
    /// changes nothing else. Counts saturate at [`MAX_COUNT`]. Returns
    /// whether the word is indexed.
    ///
    /// Panics if more than 2^31 - 1 distinct words are added.
    pub fn add(&mut self, word: &str, count: u64) -> bool {
        let count = count.min(MAX_COUNT);
        self.mix(&[1, hash_str(word), count]);
        self.total_count = saturated_add(self.total_count, count);
        if count == 0 {
            return false;
        }
        let threshold = self.config.count_threshold();
        let hash = hash_str(word);
        let id = match self.word_id(word) {
            Some(id) => {
                let previous = self.counts[id as usize];
                let updated = saturated_add(previous, count);
                self.counts[id as usize] = updated;
                // A word already at the threshold is already indexed.
                if previous >= threshold {
                    return true;
                }
                if updated < threshold {
                    return false;
                }
                id
            }
            None => {
                let id = self.words.len() as u32;
                assert!(id < LIST, "a SymSpell index holds at most 2^31 - 1 words");
                self.words.push(word.to_owned());
                self.counts.push(count);
                self.lengths.push(word.chars().count() as u32);
                self.by_word.insert(hash, id);
                if count < threshold {
                    return false;
                }
                id
            }
        };
        self.index(id);
        true
    }

    /// Adds `count` to the bigram `w1 w2` for compound correction.
    pub fn add_bigram(&mut self, w1: &str, w2: &str, count: u64) {
        let count = count.min(MAX_COUNT);
        self.mix(&[2, hash_str(w1), hash_str(w2), count]);
        let key = format!("{w1} {w2}");
        let entry = self.bigrams.entry(key).or_insert(0);
        *entry = saturated_add(*entry, count);
        let updated = *entry;
        self.max_bigram_count = self.max_bigram_count.max(updated);
        self.min_bigram_count = self.min_bigram_count.min(updated);
    }

    /// Puts word `id` under every delete of its prefix, as OpenNLP's
    /// `editsPrefix` generates them: the prefix itself, the strings one to
    /// `max_dictionary_edit_distance` deletions away (at least one, and
    /// never deleting the last code point), and the empty string when the
    /// whole word is no longer than the maximum edit distance.
    fn index(&mut self, id: u32) {
        let length = self.lengths[id as usize] as usize;
        self.max_length = self.max_length.max(length);
        let max = self.config.max_dictionary_edit_distance();
        let mut deletes = std::mem::take(&mut self.scratch);
        deletes.clear();
        let word = &self.words[id as usize];
        deletes.push(prefix(word, self.config.prefix_length()));
        let prefix_chars = deletes.get(0).1;
        let mut i = 0;
        while i < deletes.len() {
            let chars = deletes.get(i).1;
            if chars > 1 && prefix_chars - chars < max.max(1) {
                deletes.expand(i);
            }
            i += 1;
        }
        if length <= max {
            self.file(hash_str(""), id);
        }
        for i in 0..deletes.len() {
            self.file(hash_str(deletes.get(i).0), id);
        }
        self.scratch = deletes;
    }

    /// Appends word `id` to the entry of a delete hash, once.
    fn file(&mut self, hash: u64, id: u32) {
        match self.deletes.get_mut(&hash) {
            None => {
                self.deletes.insert(hash, id);
            }
            Some(entry) if *entry & LIST == 0 => {
                if *entry != id {
                    let list = self.lists.len() as u32;
                    self.lists.push(vec![*entry, id]);
                    *entry = LIST | list;
                }
            }
            Some(entry) => {
                let list = &mut self.lists[(*entry & !LIST) as usize];
                if list.last() != Some(&id) {
                    list.push(id);
                }
            }
        }
    }

    /// The words filed under the hash of `delete`, in the order they were
    /// indexed. A word filed under another delete with the same hash is
    /// among them; the caller checks with `has_delete`.
    fn filed(&self, delete: &str) -> &[u32] {
        match self.deletes.get(&hash_str(delete)) {
            None => &[],
            Some(entry) if *entry & LIST == 0 => std::slice::from_ref(entry),
            Some(entry) => &self.lists[(*entry & !LIST) as usize],
        }
    }

    /// Whether `delete` (of `chars` code points) is one of the deletes
    /// `index` filed word `id` under.
    fn has_delete(&self, id: u32, delete: &str, chars: usize) -> bool {
        let length = self.lengths[id as usize] as usize;
        let max = self.config.max_dictionary_edit_distance();
        let prefix_chars = length.min(self.config.prefix_length());
        if chars == 0 {
            return length <= max || length == 0;
        }
        if chars > prefix_chars || prefix_chars - chars > max.max(1) {
            return false;
        }
        // A subsequence of the prefix.
        let mut word = self.words[id as usize].chars().take(prefix_chars);
        delete.chars().all(|c| word.any(|w| w == c))
    }

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

/// The first `n` code points of `s`.
pub(crate) fn prefix(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((end, _)) => &s[..end],
        None => s,
    }
}
