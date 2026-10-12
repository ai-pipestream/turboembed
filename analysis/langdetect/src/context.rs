//! The language detector's context: the distinct character n-grams of a
//! text, as OpenNLP's `DefaultLanguageDetectorContextGenerator` and
//! `LanguageDetectorME.predictLanguages` produce them.
//!
//! What the JVM does, step by step:
//!
//! 1. The five normalizers `LanguageDetectorFactory` chains run on the
//!    text (`turbo_normalize::language_detection::prepare`).
//! 2. `NGramCharModel.add` lowercases it code point by code point with
//!    `Character.toLowerCase(int)` into a `CharBuffer` of UTF-16 code
//!    units, then takes every run of 1, 2 and 3 code units (all of length
//!    1 first, each length in text order) as an n-gram. An n-gram can hold
//!    one half of a surrogate pair. Each distinct n-gram is a key of a
//!    `HashMap` keyed by `CharBuffer`, put the first time it is seen.
//! 3. `getContext` lists that map's keys in its iteration order, and
//!    `predictLanguages` puts them in that order into a second `HashMap`
//!    (to count them) and hands its keys to the model in its own
//!    iteration order, each with the value 1.
//!
//! The context here is the same n-grams in the same order, each written
//! as Java modified UTF-8 (the form the model keys its predicates by, in
//! which a lone surrogate is three bytes) into one buffer. The order
//! comes from [`crate::java_map`]; `CharBuffer.hashCode` is computed over
//! the code units from the last to the first.

use turbo_normalize::language_detection::prepare;
use turbo_unicode::java::code_point::to_lower_case;
use turbo_unicode::modified_utf8::push_modified_utf8;

use crate::java_map::{self, OrderScratch};

/// The shortest and longest n-grams, in UTF-16 code units, of the
/// generator `LanguageDetectorFactory` builds.
pub const OPENNLP_MIN: usize = 1;
pub const OPENNLP_MAX: usize = 3;

/// One distinct n-gram: where its code units start in the lowercased
/// text, how many there are, and `CharBuffer.hashCode` of them.
#[derive(Copy, Clone, Debug)]
struct Gram {
    start: u32,
    len: u32,
    hash: i32,
}

/// Builds the context of a text and keeps its buffers for the next one,
/// so a generator reused across documents allocates only while its
/// buffers grow.
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
    /// `HashMap.hash` of each gram, by gram index.
    spread: Vec<u32>,
    /// 0, 1, 2, ...: the order the grams are put into the first map.
    first_seen: Vec<u32>,
    /// The first map's iteration order, the put order of the second.
    counted: Vec<u32>,
    /// The second map's iteration order: the order the model gets.
    order: Vec<u32>,
    /// Each gram's modified UTF-8, by gram index, and where it ends.
    bytes: Vec<u8>,
    ends: Vec<u32>,
    exact: bool,
    order_scratch: OrderScratch,
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
            spread: Vec::new(),
            first_seen: Vec::new(),
            counted: Vec::new(),
            order: Vec::new(),
            bytes: Vec::new(),
            ends: Vec::new(),
            exact: true,
            order_scratch: OrderScratch::default(),
        }
    }

    /// The context of `text` as `getContext` makes it: the text prepared by
    /// the five normalizers, then [`generate_prepared`](Self::generate_prepared).
    pub fn generate(&mut self, text: &str) {
        let prepared = prepare(text);
        self.generate_prepared(&prepared);
    }

    /// The context of a text that is already normalized: the distinct
    /// n-grams of its lowercase form, in the order the model receives them.
    pub fn generate_prepared(&mut self, text: &str) {
        self.lowercase(text);
        self.collect_grams();

        let n = self.grams.len() as u32;
        self.spread.clear();
        self.spread.extend(self.grams.iter().map(|g| java_map::spread(g.hash)));
        self.first_seen.clear();
        self.first_seen.extend(0..n);
        let first =
            java_map::iteration_order(&self.spread, &self.first_seen, &mut self.order_scratch, &mut self.counted);
        let second = java_map::iteration_order(&self.spread, &self.counted, &mut self.order_scratch, &mut self.order);
        self.exact = first && second;

        self.bytes.clear();
        self.ends.clear();
        for g in &self.grams {
            for &unit in &self.units[g.start as usize..(g.start + g.len) as usize] {
                push_modified_utf8(&mut self.bytes, u32::from(unit));
            }
            self.ends.push(self.bytes.len() as u32);
        }
    }

    /// The number of distinct n-grams.
    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// The n-grams as Java modified UTF-8, in the order the JVM hands them
    /// to the model.
    pub fn features(&self) -> impl ExactSizeIterator<Item = &[u8]> + Clone + '_ {
        self.order.iter().map(move |&g| {
            let g = g as usize;
            let start = if g == 0 { 0 } else { self.ends[g - 1] as usize };
            &self.bytes[start..self.ends[g] as usize]
        })
    }

    /// Whether [`features`](Self::features) is exactly the JVM's order.
    /// It is not when one of the JVM's maps would have turned a bucket into
    /// a tree (nine or more n-grams in one bucket of a table of 64 or more),
    /// whose order is not reproduced; the n-grams are the same, and the
    /// sums the model makes over them can differ in the last bits.
    pub fn exact_order(&self) -> bool {
        self.exact
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

/// `CharBuffer.hashCode`: 31 times the hash plus each code unit, from the
/// last unit to the first, starting at 1.
#[inline]
pub(crate) fn char_buffer_hash(units: &[u16]) -> i32 {
    units.iter().rev().fold(1i32, |h, &u| h.wrapping_mul(31).wrapping_add(i32::from(u)))
}

/// The slot of a hash in a table of `slots` entries (a power of two):
/// the high bits of the hash times 2^64 over the golden ratio, which
/// spreads the small hashes of short n-grams over the whole table.
#[inline]
fn home_slot(hash: i32, slots: usize) -> usize {
    let product = u64::from(hash as u32).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (product >> (64 - slots.trailing_zeros())) as usize
}

/// Doubles the slot table and puts every gram back.
fn grow(slots: &mut Vec<u32>, grams: &[Gram]) {
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

/// Records `gram` (the units at `start` of `units`) unless an equal one
/// is already in `grams`. The caller keeps the table at most half full.
#[inline]
fn insert(slots: &mut [u32], grams: &mut Vec<Gram>, units: &[u16], gram: &[u16], start: u32) {
    let hash = char_buffer_hash(gram);
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
