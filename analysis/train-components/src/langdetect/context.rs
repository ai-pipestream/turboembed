//! The language detector's context: the distinct character n-grams of a
//! text in deterministic first-seen order (shortest n-grams first, then in
//! text order within each length), after the five normalizers
//! `LanguageDetectorFactory` chains (`turbo_normalize::language_detection::prepare`)
//! and code-point lowercasing (`Character.toLowerCase(int)`).
//!
//! Each distinct n-gram is written as Java modified UTF-8 (the form the
//! model keys its predicates by, in which a lone surrogate is three bytes)
//! into one contiguous buffer.

use turbo_normalize::language_detection::prepare;
use turbo_unicode::java::code_point::to_lower_case;
use turbo_unicode::modified_utf8::push_modified_utf8;

/// The shortest and longest n-grams, in UTF-16 code units, of the
/// generator `LanguageDetectorFactory` builds.
pub const OPENNLP_MIN: usize = 1;
pub const OPENNLP_MAX: usize = 3;

/// One distinct n-gram: where its code units start in the lowercased
/// text, how many there are, and a hash of its code units for open-addressing
/// deduplication.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Gram {
    pub(crate) start: u32,
    pub(crate) len: u32,
    pub(crate) hash: i32,
}

/// Builds the context of a text and keeps its buffers for the next one,
/// so a generator reused across documents allocates for the n-grams only
/// while its buffers grow. [`generate`](Self::generate) also allocates a
/// new string whenever the normalizers change the text.
#[derive(Debug)]
pub struct ContextGenerator {
    min: usize,
    max: usize,
    /// The lowercased text as UTF-16 code units.
    units: Vec<u16>,
    /// The distinct n-grams in the order they were first seen.
    grams: Vec<Gram>,
    /// An open addressing set over `grams`: a gram's index plus one, or
    /// zero for an empty slot. Its length is a power of two.
    slots: Vec<u32>,
    /// Each gram's modified UTF-8, by gram index, and where it ends.
    bytes: Vec<u8>,
    ends: Vec<u32>,
}

impl Default for ContextGenerator {
    /// The generator `LanguageDetectorFactory` builds: n-grams of 1 to 3
    /// code units.
    fn default() -> ContextGenerator {
        ContextGenerator::new(OPENNLP_MIN, OPENNLP_MAX)
    }
}

impl ContextGenerator {
    /// A generator of n-grams from `min` to `max` UTF-16 code units long.
    ///
    /// # Panics
    ///
    /// Unless `1 <= min <= max`.
    pub fn new(min: usize, max: usize) -> ContextGenerator {
        assert!(1 <= min && min <= max, "n-gram lengths {min}..={max}: need 1 <= min <= max");
        ContextGenerator {
            min,
            max,
            units: Vec::new(),
            grams: Vec::new(),
            slots: Vec::new(),
            bytes: Vec::new(),
            ends: Vec::new(),
        }
    }

    /// The context of `text` as `getContext` makes it: the text prepared by
    /// the five normalizers, then [`generate_prepared`](Self::generate_prepared).
    pub fn generate(&mut self, text: &str) {
        let prepared = prepare(text);
        self.generate_prepared(&prepared);
    }

    /// The context of a text that is already normalized: the distinct
    /// n-grams of its lowercase form, in deterministic first-seen order.
    pub fn generate_prepared(&mut self, text: &str) {
        self.lowercase(text);
        self.collect_grams();

        self.bytes.clear();
        self.ends.clear();
        for g in &self.grams {
            for &unit in &self.units[g.start as usize..(g.start + g.len) as usize] {
                push_modified_utf8(&mut self.bytes, u32::from(unit));
            }
            self.ends.push(self.bytes.len() as u32);
        }
    }

    /// Generates the chunk's distinct lowercased n-grams in deterministic
    /// first-seen order without materializing modified UTF-8 bytes.
    pub(crate) fn generate_chunk(&mut self, text: &str) {
        let prepared = prepare(text);
        self.lowercase(&prepared);
        self.collect_grams();
    }

    /// Each distinct n-gram of the last [`generate_chunk`](Self::generate_chunk)
    /// call as `(utf16_units, hash)` in deterministic first-seen order.
    pub(crate) fn counted_grams(&self) -> impl ExactSizeIterator<Item = (&[u16], i32)> + '_ {
        self.grams.iter().map(move |g| (&self.units[g.start as usize..(g.start + g.len) as usize], g.hash))
    }

    pub(crate) fn min_length(&self) -> usize {
        self.min
    }

    pub(crate) fn max_length(&self) -> usize {
        self.max
    }

    /// The number of distinct n-grams.
    pub fn len(&self) -> usize {
        self.grams.len()
    }

    pub fn is_empty(&self) -> bool {
        self.grams.is_empty()
    }

    /// The n-grams as Java modified UTF-8, in deterministic first-seen order.
    pub fn features(&self) -> impl ExactSizeIterator<Item = &[u8]> + Clone + '_ {
        (0..self.grams.len() as u32).map(move |g| self.gram_bytes(g))
    }

    /// The n-grams as Java modified UTF-8 in deterministic first-seen order:
    /// the predicates of the event `LanguageDetectorEventStream` makes of
    /// the text for training.
    pub fn training_features(&self) -> impl ExactSizeIterator<Item = &[u8]> + Clone + '_ {
        (0..self.grams.len() as u32).map(move |g| self.gram_bytes(g))
    }

    fn gram_bytes(&self, g: u32) -> &[u8] {
        let g = g as usize;
        let start = if g == 0 { 0 } else { self.ends[g - 1] as usize };
        &self.bytes[start..self.ends[g] as usize]
    }

    /// Whether [`features`](Self::features) is in deterministic order (always
    /// `true` under native first-seen ordering).
    pub fn exact_order(&self) -> bool {
        true
    }

    /// Whether [`training_features`](Self::training_features) is in
    /// deterministic order (always `true` under native first-seen ordering).
    pub fn training_exact_order(&self) -> bool {
        true
    }

    /// Releases an oversized context, including its feature indexes.
    ///
    /// If any limit is exceeded, both feature iterators become empty until
    /// the next generation. Smaller contexts retain their features and capacity.
    /// The configured n-gram range is preserved.
    pub fn shrink_excess(&mut self) {
        const MAX_RETAINED_UNITS: usize = 16 * 1024;
        const MAX_RETAINED_GRAMS: usize = 16 * 1024;
        const MAX_RETAINED_BYTES: usize = 64 * 1024;

        if self.units.capacity() > MAX_RETAINED_UNITS
            || self.grams.capacity() > MAX_RETAINED_GRAMS
            || self.bytes.capacity() > MAX_RETAINED_BYTES
        {
            *self = Self::new(self.min, self.max);
        }
    }

    /// `StringUtil.toLowerCaseCharBuffer`: each code point through
    /// `Character.toLowerCase(int)`, written as UTF-16. No simple
    /// lowercase mapping changes how many code units a character takes
    /// (a test checks every code point), so the buffer is as long as the
    /// text in UTF-16, as the Java allocates it.
    fn lowercase(&mut self, text: &str) {
        self.units.clear();
        for c in text.chars() {
            let lower = to_lower_case(u32::from(c));
            debug_assert_eq!(lower > 0xFFFF, u32::from(c) > 0xFFFF, "U+{:04X}", u32::from(c));
            if lower <= 0xFFFF {
                self.units.push(lower as u16);
            } else {
                let offset = lower - 0x1_0000;
                self.units.push(0xD800 + (offset >> 10) as u16);
                self.units.push(0xDC00 + (offset & 0x3FF) as u16);
            }
        }
        assert!(u32::try_from(self.units.len()).is_ok(), "a text of {} UTF-16 code units", self.units.len());
    }

    /// Every n-gram, shortest first and in text order within a length,
    /// each distinct one recorded the first time it is seen.
    fn collect_grams(&mut self) {
        self.grams.clear();
        self.slots.clear();
        self.slots.resize(64, 0);
        let n = self.units.len();
        for len in self.min..=self.max.min(n) {
            for start in 0..=n - len {
                if self.grams.len() * 2 >= self.slots.len() {
                    grow(&mut self.slots, &self.grams);
                }
                let gram = &self.units[start..start + len];
                insert(&mut self.slots, &mut self.grams, &self.units, gram, start as u32);
            }
        }
    }
}

/// The slot of a hash in a table of `slots` entries (a power of two):
/// the high bits of the hash times 2^64 over the golden ratio, which
/// spreads the small hashes of short n-grams over the whole table.
#[inline]
pub(crate) fn home_slot(hash: i32, slots: usize) -> usize {
    let product = u64::from(hash as u32).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (product >> (64 - slots.trailing_zeros())) as usize
}

/// Doubles the slot table and puts every gram back.
pub(crate) fn grow(slots: &mut Vec<u32>, grams: &[Gram]) {
    let size = slots.len() * 2;
    slots.clear();
    slots.resize(size, 0);
    let mask = size - 1;
    for (i, g) in grams.iter().enumerate() {
        let mut slot = home_slot(g.hash, size);
        while slots[slot] != 0 {
            slot = (slot + 1) & mask;
        }
        slots[slot] = i as u32 + 1;
    }
}

/// Hash of UTF-16 code units for open-addressing n-gram deduplication.
#[inline]
fn gram_hash(units: &[u16]) -> i32 {
    let mut h: i32 = 1;
    for &u in units.iter().rev() {
        h = h.wrapping_mul(31).wrapping_add(i32::from(u));
    }
    h
}

/// Records `gram` (the units at `start` of `units`) unless an equal one
/// is already in `grams`. The caller keeps the table at most half full.
#[inline]
fn insert(slots: &mut [u32], grams: &mut Vec<Gram>, units: &[u16], gram: &[u16], start: u32) {
    let hash = gram_hash(gram);
    let mask = slots.len() - 1;
    let mut slot = home_slot(hash, slots.len());
    loop {
        match slots[slot] {
            0 => {
                grams.push(Gram { start, len: gram.len() as u32, hash });
                slots[slot] = grams.len() as u32;
                return;
            }
            index => {
                let g = grams[index as usize - 1];
                if g.hash == hash && units[g.start as usize..(g.start + g.len) as usize] == *gram {
                    return;
                }
            }
        }
        slot = (slot + 1) & mask;
    }
}
