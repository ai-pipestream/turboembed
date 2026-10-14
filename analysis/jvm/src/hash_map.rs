//! The hash codes and iteration order of `java.util.HashMap` (JDK 8 and
//! later, as in JDK 21) with the default load factor (`0.75f`).
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
//! `16` when first allocated by `put` or `merge`, or `tableSizeFor(ceil(s /
//! 0.75f))` when pre-sized by `putAll` on an empty map. A key with Java
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
//! ## Treeified bins
//!
//! When a bin reaches the treeify threshold at `capacity >= 64`, the JDK
//! turns that bin into a red-black tree whose order for non-`Comparable` keys
//! with equal hashes depends on `System.identityHashCode` and varies between
//! JVM runs. That order is not modelled: the bin stays in list order and
//! [`JavaHashMap::is_jvm_exact`] / [`HashMapOrder::order`] returns `false`.

use std::borrow::Borrow;
use std::borrow::Cow;

/// `DEFAULT_INITIAL_CAPACITY`.
const DEFAULT_CAPACITY: usize = 16;

/// `MAXIMUM_CAPACITY`.
const MAXIMUM_CAPACITY: usize = 1 << 30;

/// `TREEIFY_THRESHOLD`.
const TREEIFY_THRESHOLD: usize = 8;

/// `MIN_TREEIFY_CAPACITY`.
const MIN_TREEIFY_CAPACITY: usize = 64;

/// A key whose Java `Object.hashCode()` is reproduced bit for bit.
pub trait JavaHash {
    fn java_hash(&self) -> i32;
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
    let mut h = a;
    for _ in 0..b_units {
        h = h.wrapping_mul(31);
    }
    h.wrapping_add(b)
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
}

impl JavaHash for String {
    #[inline]
    fn java_hash(&self) -> i32 {
        string_hash(self)
    }
}

impl JavaHash for Box<str> {
    #[inline]
    fn java_hash(&self) -> i32 {
        string_hash(self)
    }
}

impl JavaHash for Cow<'_, str> {
    #[inline]
    fn java_hash(&self) -> i32 {
        string_hash(self)
    }
}

impl JavaHash for i32 {
    #[inline]
    fn java_hash(&self) -> i32 {
        *self
    }
}

impl<T: JavaHash + ?Sized> JavaHash for &T {
    #[inline]
    fn java_hash(&self) -> i32 {
        (**self).java_hash()
    }
}

impl<T: JavaHash> JavaHash for Option<T> {
    #[inline]
    fn java_hash(&self) -> i32 {
        self.as_ref().map_or(0, JavaHash::java_hash)
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
}

/// An owned vector of UTF-16 code units hashed as `java.lang.String`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Utf16String(pub Vec<u16>);

impl JavaHash for Utf16String {
    #[inline]
    fn java_hash(&self) -> i32 {
        utf16_string_hash(&self.0)
    }
}

/// How distinct keys are inserted into a `HashMapOrder`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Insert {
    /// `put` and `putAll`: appends each new key to the tail of its bin.
    Put,
    /// `merge`: inserts each new key at the head of its bin. Because `merge`
    /// checks `size > threshold` before looking up the key, a call for an
    /// existing key after the `threshold + 1`-th new key also resizes the
    /// table; `again` records whether any duplicate `merge` call occurred
    /// after the last new key was first seen.
    Merge { again: bool },
}

/// Computes the iteration order of a `java.util.HashMap` over pre-spread key
/// hashes without allocating map nodes, keeping its table capacity across
/// [`order`](Self::order) / [`order_keys`](Self::order_keys) calls (as
/// `HashMap.clear()` keeps the table) unless [`reset`](Self::reset) is
/// called.
#[derive(Clone, Debug, Default)]
pub struct HashMapOrder {
    /// The table length; `0` before the table is allocated.
    capacity: usize,
    /// Reusable bin counts during insertion and prefix-sum offsets during the
    /// final bucket sort.
    counts: Vec<u32>,
}

impl HashMapOrder {
    /// A new `HashMapOrder` with no table allocated yet.
    pub fn new() -> HashMapOrder {
        HashMapOrder::default()
    }

    /// The current table length (`0` before any key is inserted).
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Resets the table capacity to `0` (as a fresh `new HashMap<>()`),
    /// retaining the allocated scratch buffer for reuse.
    pub fn reset(&mut self) {
        self.capacity = 0;
        self.counts.clear();
    }

    /// Inserts distinct keys `0..spread.len()` whose spread hashes are
    /// `spread` into the empty map (new or cleared), and writes their indices
    /// into `out` in `HashMap` iteration order.
    ///
    /// Returns `false` if any bin reached the treeify threshold at
    /// `capacity >= 64` (that bin is left in list order).
    pub fn order(&mut self, spread: &[u32], insert: Insert, out: &mut Vec<u32>) -> bool {
        self.order_with(spread, spread.len(), insert, |i| i as u32, out)
    }

    /// Inserts the distinct keys listed in `keys` (indices into `spread`, in
    /// insertion order) into the empty map (new or cleared), and writes those
    /// key indices into `out` in `HashMap` iteration order.
    ///
    /// Returns `false` if any bin reached the treeify threshold at
    /// `capacity >= 64`.
    pub fn order_keys(&mut self, spread: &[u32], keys: &[u32], insert: Insert, out: &mut Vec<u32>) -> bool {
        self.order_with(spread, keys.len(), insert, |i| keys[i], out)
    }

    fn order_with(
        &mut self,
        spread: &[u32],
        n: usize,
        insert: Insert,
        at: impl Fn(usize) -> u32,
        out: &mut Vec<u32>,
    ) -> bool {
        out.clear();
        if n == 0 {
            return true;
        }
        self.counts.clear();
        self.counts.resize(self.capacity, 0);
        let mut exact = true;
        for i in 0..n {
            if matches!(insert, Insert::Merge { .. }) && (self.capacity == 0 || i > self.threshold()) {
                self.grow(spread, i, &at);
            }
            if self.capacity == 0 {
                self.grow(spread, i, &at);
            }
            let bin = spread[at(i) as usize] as usize & (self.capacity - 1);
            let before = self.counts[bin] as usize;
            self.counts[bin] += 1;
            let long = match insert {
                Insert::Put => before >= TREEIFY_THRESHOLD,
                Insert::Merge { .. } => before + 1 >= TREEIFY_THRESHOLD,
            };
            if long {
                if self.capacity < MIN_TREEIFY_CAPACITY {
                    self.grow(spread, i + 1, &at);
                } else {
                    exact = false;
                }
            }
            if insert == Insert::Put && i + 1 > self.threshold() {
                self.grow(spread, i + 1, &at);
            }
        }
        if insert == (Insert::Merge { again: true }) && n > self.threshold() {
            self.grow(spread, n, &at);
        }

        let mask = self.capacity - 1;
        self.counts.clear();
        self.counts.resize(self.capacity + 1, 0);
        for i in 0..n {
            let k = at(i);
            self.counts[(spread[k as usize] as usize & mask) + 1] += 1;
        }
        for b in 1..self.counts.len() {
            self.counts[b] += self.counts[b - 1];
        }
        out.resize(n, 0);
        let mut place = |i: usize| {
            let k = at(i);
            let slot = &mut self.counts[spread[k as usize] as usize & mask];
            out[*slot as usize] = k;
            *slot += 1;
        };
        match insert {
            Insert::Put => (0..n).for_each(&mut place),
            Insert::Merge { .. } => (0..n).rev().for_each(&mut place),
        }
        exact
    }

    fn threshold(&self) -> usize {
        self.capacity / 4 * 3
    }

    fn grow(&mut self, spread: &[u32], count: usize, at: &impl Fn(usize) -> u32) {
        self.capacity = if self.capacity == 0 { DEFAULT_CAPACITY } else { self.capacity * 2 };
        self.counts.clear();
        self.counts.resize(self.capacity, 0);
        let mask = self.capacity - 1;
        for i in 0..count {
            let k = at(i);
            self.counts[spread[k as usize] as usize & mask] += 1;
        }
    }
}

/// Returns the indices `0..keys.len()` in the order a fresh `new HashMap<>()`
/// iterates them after each distinct key is inserted with `put`, together
/// with whether the order is JVM-exact (no treeified bin).
pub fn put_order<I, K>(keys: I) -> (Vec<usize>, bool)
where
    I: IntoIterator<Item = K>,
    K: JavaHash,
{
    let spreads: Vec<u32> = keys.into_iter().map(|k| spread(k.java_hash())).collect();
    let mut order = Vec::with_capacity(spreads.len());
    let exact = HashMapOrder::new().order(&spreads, Insert::Put, &mut order);
    (order.into_iter().map(|i| i as usize).collect(), exact)
}

/// A `java.util.HashMap<K, V>` with the default load factor (`0.75f`),
/// reproducing JDK 21's table sizing, bin splitting, and iteration order for
/// `put`, `merge`, `putAll`, and `clear`.
#[derive(Clone, Debug)]
pub struct JavaHashMap<K, V> {
    capacity: usize,
    threshold: usize,
    size: usize,
    exact: bool,
    entries: Vec<Option<(K, u32, V)>>,
    bins: Vec<Vec<u32>>,
}

impl<K, V> Default for JavaHashMap<K, V> {
    fn default() -> Self {
        JavaHashMap { capacity: 0, threshold: 0, size: 0, exact: true, entries: Vec::new(), bins: Vec::new() }
    }
}

impl<K, V> JavaHashMap<K, V> {
    /// Creates an empty map (`new HashMap<>()`).
    pub fn new() -> JavaHashMap<K, V> {
        JavaHashMap::default()
    }

    /// Number of key-value mappings in the map.
    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Current table length (`0` before the first insertion).
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Current resize threshold.
    pub fn threshold(&self) -> usize {
        self.threshold
    }

    /// Whether no bin has reached the treeify threshold (`8` keys) at a
    /// table capacity of `64` or more.
    pub fn is_jvm_exact(&self) -> bool {
        self.exact
    }

    /// `HashMap.clear()`: removes all entries while keeping the allocated
    /// table capacity and threshold.
    pub fn clear(&mut self) {
        self.entries.clear();
        for bin in &mut self.bins {
            bin.clear();
        }
        self.size = 0;
        self.exact = true;
    }

    /// Iterates `(key, value)` pairs in the map's bucket and bin order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&K, &V)> + '_ {
        Iter { entries: &self.entries, inner: self.bins.iter().flatten(), remaining: self.size }
    }

    /// Iterates keys in the map's bucket and bin order.
    pub fn keys(&self) -> impl ExactSizeIterator<Item = &K> + '_ {
        self.iter().map(|(k, _)| k)
    }

    /// Iterates values in the map's bucket and bin order.
    pub fn values(&self) -> impl ExactSizeIterator<Item = &V> + '_ {
        self.iter().map(|(_, v)| v)
    }

    /// Consumes the map and returns its `(key, value)` pairs in iteration
    /// order.
    pub fn into_ordered(mut self) -> Vec<(K, V)> {
        let mut out = Vec::with_capacity(self.size);
        for bin in self.bins {
            for idx in bin {
                if let Some((k, _, v)) = self.entries[idx as usize].take() {
                    out.push((k, v));
                }
            }
        }
        out
    }
}

struct Iter<'a, K, V, I> {
    entries: &'a [Option<(K, u32, V)>],
    inner: I,
    remaining: usize,
}

impl<'a, K, V, I: Iterator<Item = &'a u32>> Iterator for Iter<'a, K, V, I> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        let &idx = self.inner.next()?;
        self.remaining -= 1;
        let (k, _, v) = self.entries[idx as usize].as_ref().expect("live entry");
        Some((k, v))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a, K, V, I: Iterator<Item = &'a u32>> ExactSizeIterator for Iter<'a, K, V, I> {}

impl<K: JavaHash + Eq, V> JavaHashMap<K, V> {
    /// Looks up the value for `key`.
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: JavaHash + Eq + ?Sized,
    {
        if self.capacity == 0 {
            return None;
        }
        let h = spread(key.java_hash());
        let bin = h as usize & (self.capacity - 1);
        for &idx in &self.bins[bin] {
            let (k, s, v) = self.entries[idx as usize].as_ref().expect("live entry");
            if *s == h && k.borrow() == key {
                return Some(v);
            }
        }
        None
    }

    /// Looks up a mutable reference to the value for `key`.
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: JavaHash + Eq + ?Sized,
    {
        if self.capacity == 0 {
            return None;
        }
        let h = spread(key.java_hash());
        let bin = h as usize & (self.capacity - 1);
        for &idx in &self.bins[bin] {
            let (k, s, _) = self.entries[idx as usize].as_ref().expect("live entry");
            if *s == h && k.borrow() == key {
                return self.entries[idx as usize].as_mut().map(|(_, _, v)| v);
            }
        }
        None
    }

    /// Whether the map contains `key`.
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: JavaHash + Eq + ?Sized,
    {
        self.get(key).is_some()
    }

    /// `HashMap.put(key, value)`: replaces the value in place if `key` is
    /// present, or appends `(key, value)` to the end of its bin and resizes
    /// if needed.
    pub fn put(&mut self, key: K, value: V) {
        if self.capacity == 0 {
            self.resize();
        }
        let h = spread(key.java_hash());
        let bin = h as usize & (self.capacity - 1);
        for &idx in &self.bins[bin] {
            let (k, s, v) = self.entries[idx as usize].as_mut().expect("live entry");
            if *s == h && *k == key {
                *v = value;
                return;
            }
        }
        let before = self.bins[bin].len();
        let idx = self.entries.len() as u32;
        self.entries.push(Some((key, h, value)));
        self.bins[bin].push(idx);
        if before >= TREEIFY_THRESHOLD {
            self.treeify_bin();
        }
        self.size += 1;
        if self.size > self.threshold {
            self.resize();
        }
    }

    /// `HashMap.merge(key, value, remappingFunction)` for a non-null
    /// remapping result: resizes first when `size > threshold || capacity == 0`,
    /// updates an existing entry in place, or inserts a new entry at the head
    /// of its bin.
    pub fn merge(&mut self, key: K, value: V, combine: impl FnOnce(&mut V, V)) {
        if self.size > self.threshold || self.capacity == 0 {
            self.resize();
        }
        let h = spread(key.java_hash());
        let bin = h as usize & (self.capacity - 1);
        for &idx in &self.bins[bin] {
            let (k, s, v) = self.entries[idx as usize].as_mut().expect("live entry");
            if *s == h && *k == key {
                combine(v, value);
                return;
            }
        }
        let before = self.bins[bin].len();
        let idx = self.entries.len() as u32;
        self.entries.push(Some((key, h, value)));
        self.bins[bin].insert(0, idx);
        if before + 1 >= TREEIFY_THRESHOLD {
            self.treeify_bin();
        }
        self.size += 1;
    }

    /// `HashMap.putAll(m)` (`putMapEntries(m, true)`): pre-sizes the table
    /// from `entries.len()` before inserting each `(key, value)` with `put`.
    pub fn put_all(&mut self, entries: impl ExactSizeIterator<Item = (K, V)>) {
        let s = entries.len();
        if s == 0 {
            return;
        }
        if self.capacity == 0 {
            let dt = (s as f64 / f64::from(0.75f32)).ceil();
            let t = if dt < MAXIMUM_CAPACITY as f64 { dt as usize } else { MAXIMUM_CAPACITY };
            if t > self.threshold {
                self.threshold = table_size_for(t);
            }
        } else {
            while s > self.threshold && self.capacity < MAXIMUM_CAPACITY {
                self.resize();
            }
        }
        for (k, v) in entries {
            self.put(k, v);
        }
    }

    fn treeify_bin(&mut self) {
        if self.capacity < MIN_TREEIFY_CAPACITY {
            self.resize();
        } else {
            self.exact = false;
        }
    }

    fn resize(&mut self) {
        let (old_cap, old_thr) = (self.capacity, self.threshold);
        let mut new_thr = 0;
        let new_cap;
        if old_cap > 0 {
            if old_cap >= MAXIMUM_CAPACITY {
                self.threshold = i32::MAX as usize;
                return;
            }
            new_cap = old_cap << 1;
            if new_cap < MAXIMUM_CAPACITY && old_cap >= DEFAULT_CAPACITY {
                new_thr = old_thr << 1;
            }
        } else if old_thr > 0 {
            new_cap = old_thr;
        } else {
            new_cap = DEFAULT_CAPACITY;
            new_thr = 12;
        }
        if new_thr == 0 {
            let ft = new_cap as f32 * 0.75f32;
            new_thr = if new_cap < MAXIMUM_CAPACITY && ft < MAXIMUM_CAPACITY as f32 {
                ft as usize
            } else {
                i32::MAX as usize
            };
        }
        self.capacity = new_cap;
        self.threshold = new_thr;

        if old_cap == 0 {
            self.bins.resize_with(new_cap, Vec::new);
            return;
        }
        self.bins.resize_with(new_cap, Vec::new);
        let bit = old_cap as u32;
        for j in 0..old_cap {
            let mut hi = Vec::new();
            let mut write = 0;
            for read in 0..self.bins[j].len() {
                let idx = self.bins[j][read];
                let h = self.entries[idx as usize].as_ref().expect("live entry").1;
                if h & bit == 0 {
                    self.bins[j][write] = idx;
                    write += 1;
                } else {
                    hi.push(idx);
                }
            }
            self.bins[j].truncate(write);
            self.bins[j + old_cap] = hi;
        }
    }
}

/// `HashMap.tableSizeFor`: the least power of two at or above `cap`.
fn table_size_for(cap: usize) -> usize {
    cap.max(1).next_power_of_two().min(MAXIMUM_CAPACITY)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn order(hashes: &[i32], insert: Insert) -> (Vec<u32>, bool, usize) {
        let spread: Vec<u32> = hashes.iter().map(|&h| spread(h)).collect();
        let mut map = HashMapOrder::new();
        let mut out = Vec::new();
        let exact = map.order(&spread, insert, &mut out);
        (out, exact, map.capacity())
    }

    #[test]
    fn hash_codes_match_java() {
        assert_eq!("".java_hash(), 0);
        assert_eq!("ab".java_hash(), 3105);
        assert_eq!("Aa".java_hash(), 2112);
        assert_eq!("BB".java_hash(), 2112);
        assert_eq!("EMAIL".java_hash(), 66081660);
        assert_eq!("😀".java_hash(), 1_772_899);
        assert_eq!(concat_hash(string_hash("ab"), string_hash("c</w>"), 5), string_hash("abc</w>"));
        assert_eq!(("ab", "c").java_hash(), 96354);
        assert_eq!(vec!["a", "b"].java_hash(), 31 * (31 + 97) + 98);
        assert_eq!(CharBuffer(&[b'a' as u16, b'b' as u16]).java_hash(), 31 * (31 + 98) + 97);
        assert_eq!(None::<&str>.java_hash(), 0);
        assert_eq!(Some("ab").java_hash(), 3105);
        assert_eq!(42i32.java_hash(), 42);
    }

    #[test]
    fn put_keeps_put_order_in_a_bin_and_merge_reverses_it() {
        assert_eq!(order(&[3, 1, 19, 0, 17], Insert::Put).0, [3, 1, 4, 0, 2]);
        assert_eq!(order(&[3, 1, 19, 0, 17], Insert::Merge { again: false }).0, [3, 4, 1, 2, 0]);
    }

    #[test]
    fn put_grows_after_the_thirteenth_key_and_merge_before_the_fourteenth() {
        let hashes: Vec<i32> = (0..13).collect();
        assert_eq!(order(&hashes, Insert::Put).2, 32);
        assert_eq!(order(&hashes, Insert::Merge { again: false }).2, 16);
        let hashes: Vec<i32> = (0..14).collect();
        assert_eq!(order(&hashes, Insert::Merge { again: false }).2, 32);
        let hashes: Vec<i32> = (0..13).collect();
        assert_eq!(order(&hashes, Insert::Merge { again: true }).2, 32);
        let hashes: Vec<i32> = (0..12).collect();
        assert_eq!(order(&hashes, Insert::Merge { again: true }).2, 16);
    }

    #[test]
    fn a_long_bin_grows_a_small_table_and_is_reported_in_a_large_one() {
        let hashes: Vec<i32> = (0..8).map(|i| i * 64).collect();
        assert_eq!(order(&hashes, Insert::Merge { again: false }).2, 32);
        assert_eq!(order(&hashes, Insert::Put).2, 16);
        let hashes: Vec<i32> = (0..9).map(|i| i * 64).collect();
        assert_eq!(order(&hashes, Insert::Put).2, 32);
        let hashes: Vec<i32> = (0..10).map(|i| i * 64).collect();
        let (out, exact, capacity) = order(&hashes, Insert::Merge { again: false });
        assert_eq!((exact, capacity, out.len()), (false, 64, 10));
    }

    #[test]
    fn a_cleared_map_keeps_its_capacity_and_reset_starts_fresh() {
        let mut map = HashMapOrder::new();
        let mut out = Vec::new();
        let spread: Vec<u32> = (0..20).collect();
        map.order(&spread, Insert::Put, &mut out);
        assert_eq!(map.capacity(), 32);
        map.order(&[16, 0], Insert::Put, &mut out);
        assert_eq!(out, [1, 0]);
        map.reset();
        map.order(&[16, 0], Insert::Put, &mut out);
        assert_eq!(map.capacity(), 16);
        assert_eq!(out, [0, 1]);
    }

    #[test]
    fn java_hash_map_put_all_and_replace_match_jdk_rules() {
        let mut m = JavaHashMap::<String, u8>::new();
        m.put_all([("a".into(), 1)].into_iter());
        assert_eq!((m.capacity(), m.threshold()), (2, 1));
        m.put_all([("b".into(), 2)].into_iter());
        assert_eq!((m.capacity(), m.threshold()), (4, 3));
        m.put("c".into(), 3);
        m.put("d".into(), 4);
        assert_eq!((m.capacity(), m.threshold()), (8, 6));
        m.put("a".into(), 9);
        assert_eq!(m.get("a"), Some(&9));
        let keys: Vec<String> = m.into_ordered().into_iter().map(|(k, _)| k).collect();
        assert_eq!(keys, ["a", "b", "c", "d"]);
    }
}
