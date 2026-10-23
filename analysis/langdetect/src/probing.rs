//! Incremental chunk-based language detection, ported from OpenNLP's
//! `LanguageDetectorME.probingPredictLanguages`, `LanguageDetectorConfig`,
//! and `ProbingLanguageDetectionResult`.
//!
//! What the JVM does, step by step:
//!
//! 1. Keeps one cumulative `ngramCounts` (`new HashMap<CharSequence, MutableInt>()`)
//!    and a sliding list `predictions` of the last chunk rankings across the
//!    probing loop, starting at code point offset `start = 0`.
//! 2. Each step takes up to `actualChunkSize = min(chunkSize, maxLength - start)`
//!    Unicode code points (`content.codePoints().skip(start).limit(actualChunkSize)`)
//!    into a new string `chunk.s`. When `chunk` has 0 code points, the loop
//!    ends: if no chunk was evaluated yet (`currPredictions == null`), it
//!    evaluates the empty `ngramCounts` map (`predict(ngramCounts)`); otherwise
//!    it returns `(currPredictions, start)`.
//! 3. `start` advances by the number of code points in `chunk`, and
//!    `mContextGenerator.getContext(chunk.s)` normalizes `chunk.s` on its own,
//!    lowercases it into UTF-16 code units, collects its distinct 1..=3 code
//!    unit n-grams into a chunk `NGramCharModel` `HashMap`, and returns them
//!    in that first map's bucket order.
//! 4. `updateCounts` walks those n-grams in that order: an n-gram not yet in
//!    `ngramCounts` is inserted with `put(ngram, new MutableInt(1))`; an n-gram
//!    already in `ngramCounts` has its `MutableInt` incremented in place without
//!    altering `ngramCounts`'s buckets.
//! 5. `predict(ngramCounts)` iterates `ngramCounts.entrySet()` in its current
//!    `HashMap` bucket order, passes each n-gram with weight `1.0` to the
//!    maxent model, and stable-sorts outcomes by descending `Double.compare`.
//! 6. `seenEnough` stops early when at least `minConsecImprovements` chunks
//!    have preceded the current chunk, `p[0] - p[1] >= minDiff` (when
//!    `minDiff > 0.0`), and across the window of the last
//!    `minConsecImprovements + 1` predictions the top language code is
//!    unchanged and its confidence is non-decreasing (`!(lastConf > conf)`).

use turbo_jvm::hash_map::{HashMapOrder, Insert, spread};
use turbo_unicode::modified_utf8::push_modified_utf8;

use crate::context::{ContextGenerator, Gram, OPENNLP_MAX, OPENNLP_MIN, grow, home_slot};
use crate::detector::Ranking;

/// Configuration for [`LanguageDetector::probing_predict_languages_with`](crate::LanguageDetector::probing_predict_languages_with),
/// ported from OpenNLP's `LanguageDetectorConfig`.
///
/// Distances (`max_length` and `chunk_size`) are measured in Unicode code
/// points (`char`s), matching `CharSequence.codePoints()` on the JVM.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbingConfig {
    /// Maximum number of Unicode code points of the input to process
    /// (`DEFAULT_MAX_LENGTH = 10_000`).
    pub max_length: usize,
    /// Number of Unicode code points per chunk (`DEFAULT_CHUNK_SIZE = 200`).
    pub chunk_size: usize,
    /// Number of consecutive improvements required before stopping early
    /// (`DEFAULT_MIN_CONSEC_IMPROVEMENTS = 2`). When positive, stopping
    /// requires a window of `min_consec_improvements + 1` consecutive chunk
    /// evaluations with the same top language and non-decreasing confidence.
    pub min_consec_improvements: usize,
    /// Minimum probability margin between the first and second ranked
    /// languages required to stop early (`DEFAULT_MIN_DIFF = 0.20`). A value
    /// `<= 0.0` disables the margin check, as on the JVM.
    pub min_diff: f64,
}

impl ProbingConfig {
    /// `LanguageDetectorConfig.DEFAULT_MAX_LENGTH`.
    pub const DEFAULT_MAX_LENGTH: usize = 10_000;
    /// `LanguageDetectorConfig.DEFAULT_CHUNK_SIZE`.
    pub const DEFAULT_CHUNK_SIZE: usize = 200;
    /// `LanguageDetectorConfig.DEFAULT_MIN_CONSEC_IMPROVEMENTS`.
    pub const DEFAULT_MIN_CONSEC_IMPROVEMENTS: usize = 2;
    /// `LanguageDetectorConfig.DEFAULT_MIN_DIFF`.
    pub const DEFAULT_MIN_DIFF: f64 = 0.20;
}

impl Default for ProbingConfig {
    /// `LanguageDetectorConfig.DEFAULT_LANGUAGE_DETECTOR_CONFIG`.
    fn default() -> ProbingConfig {
        ProbingConfig {
            max_length: Self::DEFAULT_MAX_LENGTH,
            chunk_size: Self::DEFAULT_CHUNK_SIZE,
            min_consec_improvements: Self::DEFAULT_MIN_CONSEC_IMPROVEMENTS,
            min_diff: Self::DEFAULT_MIN_DIFF,
        }
    }
}

/// Cumulative n-gram accumulator across chunks for
/// `LanguageDetectorME.probingPredictLanguages`.
///
/// Each [`add_chunk`](Self::add_chunk) call normalizes and extracts the
/// distinct n-grams of that chunk alone (in the first `HashMap`'s iteration
/// order, as `DefaultLanguageDetectorContextGenerator.getContext` returns
/// them), inserts any n-gram not yet seen into the cumulative table, and
/// updates the cumulative `HashMap` bucket order (`ngramCounts`).
#[derive(Debug)]
pub struct ProbingContext {
    chunk: ContextGenerator,
    /// Concatenated UTF-16 code units of all distinct n-grams seen across chunks.
    units: Vec<u16>,
    /// Distinct n-grams in the order they were first inserted into `ngramCounts`.
    grams: Vec<Gram>,
    /// Open-addressing set over `grams`: 1-based index into `grams`, or 0.
    slots: Vec<u32>,
    /// `HashMap.hash` of each gram in `grams`.
    spread: Vec<u32>,
    /// Iteration order of the cumulative `ngramCounts` `HashMap`.
    order: Vec<u32>,
    /// Each gram's Java modified UTF-8 bytes, by gram index, and where it ends.
    bytes: Vec<u8>,
    ends: Vec<u32>,
    /// Whether every chunk's first `HashMap` order was JVM-exact.
    chunks_exact: bool,
    /// Whether the cumulative `ngramCounts` `HashMap` order is JVM-exact.
    cumulative_exact: bool,
    order_scratch: HashMapOrder,
}

impl Default for ProbingContext {
    fn default() -> ProbingContext {
        ProbingContext::new(OPENNLP_MIN, OPENNLP_MAX)
    }
}

impl ProbingContext {
    /// A cumulative probing context for n-grams of `min` to `max` UTF-16
    /// code units.
    ///
    /// # Panics
    ///
    /// Unless `1 <= min <= max`.
    pub fn new(min: usize, max: usize) -> ProbingContext {
        ProbingContext {
            chunk: ContextGenerator::new(min, max),
            units: Vec::new(),
            grams: Vec::new(),
            slots: Vec::new(),
            spread: Vec::new(),
            order: Vec::new(),
            bytes: Vec::new(),
            ends: Vec::new(),
            chunks_exact: true,
            cumulative_exact: true,
            order_scratch: HashMapOrder::default(),
        }
    }

    /// Clears the cumulative n-gram table for a new document while retaining
    /// allocated buffer capacities.
    pub fn reset(&mut self) {
        self.units.clear();
        self.grams.clear();
        self.slots.clear();
        self.slots.resize(64, 0);
        self.spread.clear();
        self.order.clear();
        self.bytes.clear();
        self.ends.clear();
        self.chunks_exact = true;
        self.cumulative_exact = true;
    }

    /// Runs `mContextGenerator.getContext` on `chunk` and merges its n-grams
    /// into the cumulative `ngramCounts` map as `LanguageDetectorME.updateCounts`
    /// does.
    pub fn add_chunk(&mut self, chunk: &str) {
        if self.slots.is_empty() {
            self.slots.resize(64, 0);
        }
        self.chunk.generate_chunk(chunk);
        self.chunks_exact &= self.chunk.training_exact_order();
        let prev_len = self.grams.len();
        for (gram, hash) in self.chunk.counted_grams() {
            if self.grams.len() * 2 >= self.slots.len() {
                grow(&mut self.slots, &self.grams);
            }
            if insert_owned(&mut self.slots, &mut self.grams, &mut self.units, gram, hash) {
                self.spread.push(spread(hash));
                for &unit in gram {
                    push_modified_utf8(&mut self.bytes, u32::from(unit));
                }
                self.ends.push(self.bytes.len() as u32);
            }
        }
        if self.grams.len() != prev_len {
            self.order_scratch.reset();
            self.cumulative_exact = self.order_scratch.order(&self.spread, Insert::Put, &mut self.order);
        }
    }

    /// The number of distinct n-grams accumulated so far.
    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// The cumulative n-grams as Java modified UTF-8, in the iteration order
    /// of the JVM's `ngramCounts` `HashMap`.
    pub fn features(&self) -> impl ExactSizeIterator<Item = &[u8]> + Clone + '_ {
        self.order.iter().map(move |&g| self.gram_bytes(g))
    }

    fn gram_bytes(&self, g: u32) -> &[u8] {
        let g = g as usize;
        let start = if g == 0 { 0 } else { self.ends[g - 1] as usize };
        &self.bytes[start..self.ends[g] as usize]
    }

    /// Whether [`features`](Self::features) is guaranteed to be in the exact
    /// JVM order across all chunks added since [`reset`](Self::reset).
    pub fn exact_order(&self) -> bool {
        self.chunks_exact && self.cumulative_exact
    }

    /// Releases oversized buffers when any retention threshold is exceeded.
    pub fn shrink_excess(&mut self) {
        const MAX_RETAINED_UNITS: usize = 16 * 1024;
        const MAX_RETAINED_GRAMS: usize = 16 * 1024;
        const MAX_RETAINED_BYTES: usize = 64 * 1024;

        self.chunk.shrink_excess();
        if self.units.capacity() > MAX_RETAINED_UNITS
            || self.grams.capacity() > MAX_RETAINED_GRAMS
            || self.bytes.capacity() > MAX_RETAINED_BYTES
        {
            *self = Self::new(self.chunk.min_length(), self.chunk.max_length());
        }
    }
}

/// Inserts `gram` (with `CharBuffer.hashCode` `hash`) into `grams` and
/// `units` if not already present, returning `true` when newly inserted.
#[inline]
fn insert_owned(slots: &mut [u32], grams: &mut Vec<Gram>, units: &mut Vec<u16>, gram: &[u16], hash: i32) -> bool {
    let mask = slots.len() - 1;
    let mut slot = home_slot(hash, slots.len());
    loop {
        match slots[slot] {
            0 => {
                let start = u32::try_from(units.len()).expect("probing UTF-16 units fit in u32");
                units.extend_from_slice(gram);
                grams.push(Gram { start, len: gram.len() as u32, hash });
                slots[slot] = grams.len() as u32;
                return true;
            }
            index => {
                let g = grams[index as usize - 1];
                if g.hash == hash && units[g.start as usize..(g.start + g.len) as usize] == *gram {
                    return false;
                }
            }
        }
        slot = (slot + 1) & mask;
    }
}

/// Takes up to `limit` Unicode code points (`char`s) from the front of
/// `*rest`, advancing `*rest` past them and returning the slice and the
/// number of `char`s taken.
pub(crate) fn take_chars<'a>(rest: &mut &'a str, limit: usize) -> (&'a str, usize) {
    if limit == 0 || rest.is_empty() {
        return ("", 0);
    }
    let mut chars = rest.char_indices();
    let mut count = 0;
    for _ in 0..limit {
        if chars.next().is_none() {
            let chunk = *rest;
            *rest = "";
            return (chunk, count);
        }
        count += 1;
    }
    let split = chars.next().map_or(rest.len(), |(idx, _)| idx);
    let (chunk, tail) = rest.split_at(split);
    *rest = tail;
    (chunk, count)
}

/// `LanguageDetectorME.seenEnough`: updates the sliding window `history` of
/// `(top_outcome, top_confidence)` pairs and returns `true` when probing can
/// stop early.
pub(crate) fn seen_enough(history: &mut Vec<(u32, f64)>, current: &Ranking, config: &ProbingConfig) -> bool {
    let top = (current.outcomes()[0], current.probabilities()[0]);
    if history.len() < config.min_consec_improvements {
        history.push(top);
        return false;
    } else if history.len() > config.min_consec_improvements && !history.is_empty() {
        history.remove(0);
    }
    history.push(top);
    if config.min_diff > 0.0 {
        let second = current.probabilities().get(1).copied().unwrap_or(0.0);
        if top.1 - second < config.min_diff {
            return false;
        }
    }
    let mut last = history[0];
    for &p in &history[1..] {
        if last.0 != p.0 || last.1 > p.1 {
            return false;
        }
        last = p;
    }
    true
}
