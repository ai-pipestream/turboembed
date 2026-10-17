//! The hash codes and iteration order of `java.util.HashMap` (JDK 19 and
//! later, checked against JDK 21) with the default load factor (`0.75f`).
//! JDK 8 to 18 pre-size `putAll` and `new HashMap(Map)` to
//! `tableSizeFor(s / 0.75f + 1)`, which differs from JDK 19+ (JDK-8281631)
//! when `s / 0.75` is a power of two.
//!
//! Several OpenNLP components depend on the order a `HashMap` visits its
//! entries:
//!
//! - `LanguageDetectorME` collects distinct character n-grams in a
//!   `HashMap<CharBuffer, ...>`, then copies them into a second `HashMap`
//!   before summing model parameters in the second map's iteration order.
//! - `BPETokenizerTrainer` counts words (`merge`), builds its vocabulary
//!   (`put`, `clear`, `putAll`), and counts adjacent symbol pairs (`merge`),
//!   breaking frequency ties by the first pair in `pairCounts`'s iteration
//!   order.
//! - `RegexNameFinderFactory` combines default and custom pattern maps with
//!   `put` and `putAll` (`putMapEntries`), and `RegexNameFinder` runs each
//!   entity type in that map's iteration order.
//! - `POSDictionary` (when case-insensitive) walks a `HashMap<String, ...>`
//!   of raw words and inserts their lowercased forms into a new map, so the
//!   last colliding word in `HashMap` order wins.
//! - The MASC and Brat corpus readers index tokens, entities, header files,
//!   and annotations in `HashMap`s and emit or link them in map iteration
//!   order.
//!
//! ## Table model
//!
//! The table is an array of bins whose length (`capacity`) is a power of two,
//! `16` when first allocated by `put` or `merge`, `tableSizeFor(n)` for
//! `new HashMap<>(n)`, or `tableSizeFor(ceil(s / 0.75f))` (when larger) for
//! `putAll` of `s` entries into a map with no table yet. A key with Java
//! `hashCode` `h` lands in bin `spread(h) & (capacity - 1)`, where
//! `spread(h) = (h as u32) ^ ((h as u32) >> 16)`.
//!
//! - `put` appends a new key to the tail of its bin. If the bin already held
//!   8 keys (`TREEIFY_THRESHOLD`), `treeifyBin` doubles the table while
//!   `capacity < 64` (`MIN_TREEIFY_CAPACITY`). Then, if `size > threshold`
//!   (`capacity * 3 / 4`), `resize` doubles the table.
//! - `merge` checks `size > threshold || capacity == 0` **before** looking up
//!   the key, so a `merge` call on a table holding `threshold + 1` keys
//!   doubles the table even when the key is already present. For a new key,
//!   `merge` inserts at the **head** of its bin, and calls `treeifyBin` when
//!   the bin already held 7 keys (8 including the new key).
//! - `resize` splits each bin `j` into `j` (`spread & old_cap == 0`) and
//!   `j + old_cap` (`spread & old_cap != 0`), preserving the relative order
//!   of keys within each half.
//! - Iteration walks bins `0..capacity` in index order, and each bin front to
//!   back.
//!
//! ## Tree bins
//!
//! When a list bin reaches the treeify threshold at `capacity >= 64`, the
//! JDK turns it into a red-black tree (`treeifyBin`), and iteration still
//! follows the nodes' `next` links, which the tree code reorders:
//!
//! - `treeify` inserts the bin's nodes in list order into a red-black tree
//!   ordered by hash (as a signed `int`), then by `compareTo` when the class
//!   is comparable, then by `tieBreakOrder`, and moves the root to the
//!   front of the list.
//! - `putTreeVal` links a new key into the list right after its tree
//!   parent, rebalances, and moves the root to the front again.
//! - `resize` splits a tree bin into two lists in order; a half of at most
//!   6 nodes (`UNTREEIFY_THRESHOLD`) becomes a plain list, and a longer half
//!   is treeified again when the other half is not empty.
//!
//! All of this is reproduced, red-black rotations included. The one thing
//! that is not is `tieBreakOrder` between two keys with equal hash codes
//! whose class `HashMap` cannot compare (`CharBuffer`, `List`, records):
//! the JVM orders them by `System.identityHashCode`, which varies between
//! runs, so no model can match it. [`JavaHashMap::is_jvm_exact`] and
//! [`HashMapOrder::order`] return `false` when that happened.

use std::borrow::Cow;
use std::cmp::Ordering;

use crate::chars::compare;

mod map;
mod order;
#[cfg(test)]
mod tests;
mod tree;

pub use map::JavaHashMap;
pub use order::{HashMapOrder, Insert, put_order};

/// `DEFAULT_INITIAL_CAPACITY`.
pub(super) const DEFAULT_CAPACITY: usize = 16;

/// `MAXIMUM_CAPACITY`.
pub(super) const MAXIMUM_CAPACITY: usize = 1 << 30;

/// `TREEIFY_THRESHOLD`.
pub(super) const TREEIFY_THRESHOLD: usize = 8;

/// `MIN_TREEIFY_CAPACITY`.
pub(super) const MIN_TREEIFY_CAPACITY: usize = 64;

/// A key whose Java `Object.hashCode()` is reproduced bit for bit.
pub trait JavaHash {
    fn java_hash(&self) -> i32;

    /// How a `HashMap` tree bin orders this key against `other`, a different
    /// key with the same hash code: `compareTo` when `HashMap` can compare
    /// the key's class (`comparableClassFor` accepts `String` and classes
    /// such as `Integer` that implement `Comparable` of themselves), and
    /// `None` otherwise (`CharBuffer`, `List`, records), where the JVM
    /// breaks the tie by `System.identityHashCode`.
    fn tree_order(&self, other: &Self) -> Option<Ordering> {
        let _ = other;
        None
    }
}

/// `String.hashCode`: `s[0]*31^(n-1) + ... + s[n-1]` over UTF-16 code units,
/// in wrapping 32-bit signed arithmetic.
#[inline]
pub fn string_hash(s: &str) -> i32 {
    s.encode_utf16().fold(0i32, |h, u| h.wrapping_mul(31).wrapping_add(i32::from(u)))
}

/// `String.hashCode` over a slice of UTF-16 code units.
#[inline]
pub fn utf16_string_hash(units: &[u16]) -> i32 {
    units.iter().fold(0i32, |h, &u| h.wrapping_mul(31).wrapping_add(i32::from(u)))
}

/// `CharBuffer.hashCode`: `31 * h + units[i]` from the last code unit down to
/// the first, starting from `1`.
#[inline]
pub fn char_buffer_hash(units: &[u16]) -> i32 {
    units.iter().rev().fold(1i32, |h, &u| h.wrapping_mul(31).wrapping_add(i32::from(u)))
}

/// `String.hashCode` of `a` concatenated with `b`, from each string's hash
/// code and the length of `b` in UTF-16 code units.
#[inline]
pub fn concat_hash(a: i32, b: i32, b_units: usize) -> i32 {
    // 31 is odd, so its powers modulo 2^32 repeat with a period dividing
    // 2^30, and truncating the length to 32 bits keeps the power.
    a.wrapping_mul(31i32.wrapping_pow(b_units as u32)).wrapping_add(b)
}

/// The hash code of a two-component Java record such as
/// `SymbolPair(String left, String right)`: `31 * left + right`.
#[inline]
pub fn pair_hash(left: i32, right: i32) -> i32 {
    left.wrapping_mul(31).wrapping_add(right)
}

/// `List.hashCode` for a list whose elements have the given `hashCode`s.
#[inline]
pub fn list_hash(items: impl IntoIterator<Item = i32>) -> i32 {
    items.into_iter().fold(1i32, |h, e| h.wrapping_mul(31).wrapping_add(e))
}

/// `HashMap.hash`: folds the upper 16 bits of `hash` into the lower 16 bits.
#[inline]
pub fn spread(hash: i32) -> u32 {
    let h = hash as u32;
    h ^ (h >> 16)
}

impl JavaHash for str {
    #[inline]
    fn java_hash(&self) -> i32 {
        string_hash(self)
    }

    fn tree_order(&self, other: &Self) -> Option<Ordering> {
        Some(compare(self, other))
    }
}

impl JavaHash for String {
    #[inline]
    fn java_hash(&self) -> i32 {
        string_hash(self)
    }

    fn tree_order(&self, other: &Self) -> Option<Ordering> {
        Some(compare(self, other))
    }
}

impl JavaHash for Box<str> {
    #[inline]
    fn java_hash(&self) -> i32 {
        string_hash(self)
    }

    fn tree_order(&self, other: &Self) -> Option<Ordering> {
        Some(compare(self, other))
    }
}

impl JavaHash for Cow<'_, str> {
    #[inline]
    fn java_hash(&self) -> i32 {
        string_hash(self)
    }

    fn tree_order(&self, other: &Self) -> Option<Ordering> {
        Some(compare(self, other))
    }
}

impl JavaHash for i32 {
    #[inline]
    fn java_hash(&self) -> i32 {
        *self
    }

    fn tree_order(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: JavaHash + ?Sized> JavaHash for &T {
    #[inline]
    fn java_hash(&self) -> i32 {
        (**self).java_hash()
    }

    fn tree_order(&self, other: &Self) -> Option<Ordering> {
        (**self).tree_order(*other)
    }
}

impl<T: JavaHash> JavaHash for Option<T> {
    #[inline]
    fn java_hash(&self) -> i32 {
        self.as_ref().map_or(0, JavaHash::java_hash)
    }

    /// `null` sorts before every key with its hash code (0): `tieBreakOrder`
    /// gives `null` the identity hash code 0, below any object's.
    fn tree_order(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Some(a), Some(b)) => a.tree_order(b),
            (a, b) => Some(a.is_some().cmp(&b.is_some())),
        }
    }
}

impl<A: JavaHash, B: JavaHash> JavaHash for (A, B) {
    #[inline]
    fn java_hash(&self) -> i32 {
        pair_hash(self.0.java_hash(), self.1.java_hash())
    }
}

impl<T: JavaHash> JavaHash for [T] {
    #[inline]
    fn java_hash(&self) -> i32 {
        list_hash(self.iter().map(JavaHash::java_hash))
    }
}

impl<T: JavaHash> JavaHash for Vec<T> {
    #[inline]
    fn java_hash(&self) -> i32 {
        self.as_slice().java_hash()
    }
}

/// A borrowed slice of UTF-16 code units hashed as `java.nio.CharBuffer`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CharBuffer<'a>(pub &'a [u16]);

impl JavaHash for CharBuffer<'_> {
    #[inline]
    fn java_hash(&self) -> i32 {
        char_buffer_hash(self.0)
    }
}

/// A borrowed slice of UTF-16 code units hashed as `java.lang.String`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Utf16Str<'a>(pub &'a [u16]);

impl JavaHash for Utf16Str<'_> {
    #[inline]
    fn java_hash(&self) -> i32 {
        utf16_string_hash(self.0)
    }

    fn tree_order(&self, other: &Self) -> Option<Ordering> {
        Some(self.0.cmp(other.0))
    }
}

/// An owned vector of UTF-16 code units hashed as `java.lang.String`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Utf16String(pub Vec<u16>);

impl JavaHash for Utf16String {
    #[inline]
    fn java_hash(&self) -> i32 {
        utf16_string_hash(&self.0)
    }

    fn tree_order(&self, other: &Self) -> Option<Ordering> {
        Some(self.0.cmp(&other.0))
    }
}

/// `HashMap.tableSizeFor`: the least power of two at or above `cap`.
pub(super) fn table_size_for(cap: usize) -> usize {
    cap.max(1).next_power_of_two().min(MAXIMUM_CAPACITY)
}
