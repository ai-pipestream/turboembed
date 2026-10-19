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

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use super::config::{MAX_COUNT, SymSpellConfig};
use super::deletes::Deletes;
use super::table::{BuildPrehashed, IdTable, fmix, hash_str};

/// Marks an index entry that points into `lists` instead of holding a word
/// id itself.
const LIST: u32 = 1 << 31;

/// [`SymSpell`]'s scanned minimum bigram count, 0 when not scanned.
#[derive(Debug, Default)]
struct MinCache(AtomicU64);

impl Clone for MinCache {
    fn clone(&self) -> MinCache {
        MinCache(AtomicU64::new(self.0.load(Ordering::Relaxed)))
    }
}

/// A symmetric delete spelling corrector over a frequency dictionary.
#[derive(Clone)]
pub struct SymSpell {
    pub(crate) config: SymSpellConfig,
    pub(super) words: Vec<String>,
    pub(super) counts: Vec<u64>,
    pub(super) lengths: Vec<u32>,
    by_word: IdTable,
    deletes: HashMap<u64, u32, BuildPrehashed>,
    lists: Vec<Vec<u32>>,
    pub(crate) bigrams: HashMap<String, u64>,
    /// The smallest bigram count as [`min_bigram_count`](Self::min_bigram_count)
    /// last found it, or 0 (never a count) when a bigram was added since.
    /// Finding it when it is read, once, keeps adding a bigram as cheap as
    /// on a map that tracks no minimum.
    min_cache: MinCache,
    max_bigram_count: u64,
    /// The length in code points of the longest indexed word.
    pub(super) max_length: usize,
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
            min_cache: MinCache(AtomicU64::new(0)),
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

    /// The smallest bigram count, [`MAX_COUNT`] before any bigram.
    pub(crate) fn min_bigram_count(&self) -> u64 {
        let cached = self.min_cache.0.load(Ordering::Relaxed);
        if cached != 0 {
            return cached;
        }
        let min = self.bigrams.values().copied().min().unwrap_or(MAX_COUNT);
        self.min_cache.0.store(min, Ordering::Relaxed);
        min
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
        fmix(self.fingerprint ^ self.min_bigram_count())
    }

    fn mix(&mut self, parts: &[u64]) {
        for &p in parts {
            self.fingerprint = fmix(self.fingerprint.rotate_left(17) ^ p);
        }
    }

    pub(super) fn word_id(&self, word: &str) -> Option<u32> {
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

    /// Adds `count` to the bigram `w1 w2` for compound correction. A zero
    /// count records the call in the fingerprint and changes nothing else
    /// (matching [`SymSpell::add`]). Unlike OpenNLP's `SymSpell.addBigram`,
    /// which stores zero counts and leaves `minBigramCount` at its stale
    /// pre-increment value when an existing minimum bigram is updated, this
    /// keeps `min_bigram_count` equal to the minimum positive bigram count.
    /// Counts saturate at [`MAX_COUNT`].
    pub fn add_bigram(&mut self, w1: &str, w2: &str, count: u64) {
        let count = count.min(MAX_COUNT);
        self.mix(&[2, hash_str(w1), hash_str(w2), count]);
        if count == 0 {
            return;
        }
        let key = format!("{w1} {w2}");
        let entry = self.bigrams.entry(key).or_insert(0);
        let updated = saturated_add(*entry, count);
        *entry = updated;
        self.max_bigram_count = self.max_bigram_count.max(updated);
        *self.min_cache.0.get_mut() = 0;
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
    pub(super) fn filed(&self, delete: &str) -> &[u32] {
        match self.deletes.get(&hash_str(delete)) {
            None => &[],
            Some(entry) if *entry & LIST == 0 => std::slice::from_ref(entry),
            Some(entry) => &self.lists[(*entry & !LIST) as usize],
        }
    }

    /// Whether `delete` (of `chars` code points) is one of the deletes
    /// `index` filed word `id` under.
    pub(super) fn has_delete(&self, id: u32, delete: &str, chars: usize) -> bool {
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
}

/// The first `n` code points of `s`.
pub(crate) fn prefix(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((end, _)) => &s[..end],
        None => s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_bigram_updates_min_bigram_count_and_ignores_zero_count() {
        let mut engine = SymSpell::default();
        engine.add_bigram("a", "b", 0);
        assert_eq!(engine.bigram_count(), 0);
        assert_eq!(engine.min_bigram_count(), MAX_COUNT);

        engine.add_bigram("a", "b", 5);
        assert_eq!(engine.min_bigram_count(), 5);
        assert_eq!(engine.max_bigram_count(), 5);

        engine.add_bigram("a", "b", 10);
        assert_eq!(engine.bigram("a", "b"), Some(15));
        assert_eq!(engine.min_bigram_count(), 15);
        assert_eq!(engine.max_bigram_count(), 15);

        engine.add_bigram("c", "d", 8);
        assert_eq!(engine.min_bigram_count(), 8);
        engine.add_bigram("c", "d", 12);
        assert_eq!(engine.min_bigram_count(), 15);
    }

    #[test]
    fn min_bigram_count_is_the_smallest_count_after_every_add() {
        let mut engine = SymSpell::default();
        let mut state = 7u64;
        for _ in 0..2000 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let (a, b, count) = ((state >> 33) % 5, (state >> 40) % 5, (state >> 50) % 4);
            engine.add_bigram(&a.to_string(), &b.to_string(), count);
            let want = engine.bigrams.values().copied().min().unwrap_or(MAX_COUNT);
            assert_eq!(engine.min_bigram_count(), want);
        }
    }
}
