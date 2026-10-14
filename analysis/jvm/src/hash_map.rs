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

use std::borrow::Borrow;
use std::borrow::Cow;
use std::cmp::Ordering;

use crate::chars::compare;

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

/// How distinct keys are inserted into a `HashMapOrder`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Insert {
    /// `put`: appends each new key to the tail of its bin. This is also
    /// `putAll` into a map whose table already holds the new size, which
    /// does not pre-size it; [`JavaHashMap::put_all`] pre-sizes an empty or
    /// smaller table as the JDK does.
    Put,
    /// `merge`: inserts each new key at the head of its bin. Because `merge`
    /// checks `size > threshold` before looking up the key, a call for an
    /// existing key after the `threshold + 1`-th new key also resizes the
    /// table; `again` records whether any duplicate `merge` call occurred
    /// after the last new key was first seen. Duplicate calls between new
    /// keys need no record: such a resize happens with no insertion
    /// between it and the next new key's call, which would make the same
    /// resize itself.
    Merge { again: bool },
}

/// Computes the iteration order of a `java.util.HashMap` over pre-spread key
/// hashes without allocating map nodes (unless a bin becomes a tree, which
/// it hands to a [`JavaHashMap`]), keeping its table capacity across
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
    /// Returns `false` if the order may not be the JVM's: a tree bin had to
    /// order keys with equal hash codes. This model sees only the hashes,
    /// so it treats every key as one the JVM orders by
    /// `System.identityHashCode` there (see [`JavaHash::tree_order`]).
    pub fn order(&mut self, spread: &[u32], insert: Insert, out: &mut Vec<u32>) -> bool {
        self.order_with(spread, spread.len(), insert, |i| i as u32, out)
    }

    /// Inserts the distinct keys listed in `keys` (indices into `spread`, in
    /// insertion order) into the empty map (new or cleared), and writes those
    /// key indices into `out` in `HashMap` iteration order.
    ///
    /// Returns `false` as [`order`](Self::order) does.
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
        let start = self.capacity;
        self.counts.clear();
        self.counts.resize(self.capacity, 0);
        for i in 0..n {
            // `merge` resizes a full table before it looks up its key; both
            // allocate the table on the first key.
            if self.capacity == 0 || (matches!(insert, Insert::Merge { .. }) && i > self.threshold()) {
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
                    // The bin becomes a tree: run the full model instead.
                    self.capacity = start;
                    return self.order_by_map(spread, n, insert, &at, out);
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
        true
    }

    /// The order [`order_with`](Self::order_with) computes, from a
    /// [`JavaHashMap`] holding the keys, for maps with a tree bin.
    fn order_by_map(
        &mut self,
        spread: &[u32],
        n: usize,
        insert: Insert,
        at: &impl Fn(usize) -> u32,
        out: &mut Vec<u32>,
    ) -> bool {
        let mut map = JavaHashMap::with_table(self.capacity);
        for i in 0..n {
            let index = at(i);
            // `spread` is its own inverse.
            let key = Slot { index, hash: self::spread(spread[index as usize] as i32) as i32 };
            match insert {
                Insert::Put => map.put(key, ()),
                Insert::Merge { .. } => map.merge(key, (), |_, _| {}),
            }
        }
        if insert == (Insert::Merge { again: true }) {
            map.resize_before_merge();
        }
        self.capacity = map.capacity();
        out.extend(map.keys().map(|k| k.index));
        map.is_jvm_exact()
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
/// iterates them after each key, all distinct, is inserted with `put`,
/// together with whether the order is the JVM's (see
/// [`JavaHashMap::is_jvm_exact`]). A tree bin holding keys with equal hash
/// codes is ordered by the keys themselves, so `String` and `Integer` keys
/// always come in the JVM's order.
pub fn put_order<I, K>(keys: I) -> (Vec<usize>, bool)
where
    I: IntoIterator<Item = K>,
    K: JavaHash + Eq,
{
    let keys: Vec<K> = keys.into_iter().collect();
    let spreads: Vec<u32> = keys.iter().map(|k| spread(k.java_hash())).collect();
    let mut order = Vec::with_capacity(spreads.len());
    if HashMapOrder::new().order(&spreads, Insert::Put, &mut order) {
        return (order.into_iter().map(|i| i as usize).collect(), true);
    }
    let mut map = JavaHashMap::new();
    for (i, k) in keys.into_iter().enumerate() {
        map.put(k, i);
    }
    let exact = map.is_jvm_exact();
    (map.into_ordered().into_iter().map(|(_, i)| i).collect(), exact)
}

/// A `java.util.HashMap<K, V>` with the default load factor (`0.75f`),
/// reproducing JDK 21's table sizing, bin splitting, tree bins and
/// iteration order for `new HashMap<>()`, `new HashMap<>(initialCapacity)`,
/// `put`, `merge`, `putAll`, and `clear`.
///
/// A `HashSet` is a `HashMap` of its elements: `add` is [`put`](Self::put),
/// `new HashSet<>(n)` is [`with_capacity(n)`](Self::with_capacity), and
/// `new HashSet<>(c)` is `with_capacity(max((int) (c.size() / .75f) + 1,
/// 16))` followed by a `put` of each element of `c`.
#[derive(Clone, Debug)]
pub struct JavaHashMap<K, V> {
    /// The table length; `0` before the table is allocated.
    capacity: usize,
    /// `threshold`: the size above which the table doubles, or before the
    /// table is allocated, the length to allocate (`0` for the default).
    threshold: usize,
    size: usize,
    exact: bool,
    /// Entries in first insertion order: key, spread hash, value.
    entries: Vec<Option<(K, u32, V)>>,
    /// The red-black tree links of each entry, by entry index; used only
    /// while the entry is in a tree bin.
    links: Vec<Links>,
    bins: Vec<Bin>,
}

/// One bin of the table: its entries in `next` order (the order the map
/// iterates them), and whether it is a tree bin. In a tree bin the first
/// entry is always the root, as `moveRootToFront` keeps it.
#[derive(Clone, Debug, Default)]
struct Bin {
    nodes: Vec<u32>,
    tree: bool,
}

/// `TreeNode`'s `parent`, `left`, `right`, and `red` fields.
#[derive(Clone, Copy, Debug)]
struct Links {
    parent: u32,
    left: u32,
    right: u32,
    red: bool,
}

/// No node (`null`).
const NIL: u32 = u32::MAX;

/// `UNTREEIFY_THRESHOLD`.
const UNTREEIFY_THRESHOLD: usize = 6;

const NO_LINKS: Links = Links { parent: NIL, left: NIL, right: NIL, red: false };

impl<K, V> Default for JavaHashMap<K, V> {
    fn default() -> Self {
        JavaHashMap {
            capacity: 0,
            threshold: 0,
            size: 0,
            exact: true,
            entries: Vec::new(),
            links: Vec::new(),
            bins: Vec::new(),
        }
    }
}

impl<K, V> JavaHashMap<K, V> {
    /// Creates an empty map (`new HashMap<>()`).
    pub fn new() -> JavaHashMap<K, V> {
        JavaHashMap::default()
    }

    /// Creates an empty map with an initial capacity (`new
    /// HashMap<>(initialCapacity)`): the first insertion allocates a table
    /// of `tableSizeFor(initial_capacity)` bins.
    pub fn with_capacity(initial_capacity: usize) -> JavaHashMap<K, V> {
        JavaHashMap { threshold: table_size_for(initial_capacity.min(MAXIMUM_CAPACITY)), ..JavaHashMap::default() }
    }

    /// An empty map whose table of `capacity` bins is allocated, as a map
    /// grown from the default capacity and then cleared. `capacity` is `0`
    /// or a power of two of at least `16`.
    fn with_table(capacity: usize) -> JavaHashMap<K, V> {
        let mut map = JavaHashMap::new();
        if capacity > 0 {
            map.capacity = capacity;
            map.threshold = capacity / 4 * 3;
            map.bins.resize_with(capacity, Bin::default);
        }
        map
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

    /// Whether the iteration order is the JVM's. It is not when a tree bin
    /// had to order two keys with equal hash codes that `HashMap` cannot
    /// compare (see [`JavaHash::tree_order`]): the JVM then breaks the tie
    /// by `System.identityHashCode`, which differs from run to run.
    pub fn is_jvm_exact(&self) -> bool {
        self.exact
    }

    /// The number of bins that are red-black trees.
    pub fn tree_bins(&self) -> usize {
        self.bins.iter().filter(|b| b.tree).count()
    }

    /// `HashMap.clear()`: removes all entries while keeping the allocated
    /// table capacity and threshold.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.links.clear();
        for bin in &mut self.bins {
            bin.nodes.clear();
            bin.tree = false;
        }
        self.size = 0;
        self.exact = true;
    }

    /// Iterates `(key, value)` pairs in the map's bucket and bin order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&K, &V)> + '_ {
        Iter { entries: &self.entries, inner: self.bins.iter().flat_map(|b| &b.nodes), remaining: self.size }
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
            for idx in bin.nodes {
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
    /// The entry index of `key` (whose spread hash is `h`) in bin `bin`.
    fn find<Q>(&self, bin: usize, h: u32, key: &Q) -> Option<u32>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        self.bins[bin].nodes.iter().copied().find(|&idx| {
            let (k, s, _) = self.entries[idx as usize].as_ref().expect("live entry");
            *s == h && k.borrow() == key
        })
    }

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
        let idx = self.find(h as usize & (self.capacity - 1), h, key)?;
        self.entries[idx as usize].as_ref().map(|(_, _, v)| v)
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
        let idx = self.find(h as usize & (self.capacity - 1), h, key)?;
        self.entries[idx as usize].as_mut().map(|(_, _, v)| v)
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
    /// present; otherwise appends `(key, value)` to the end of its list bin
    /// (making the bin a tree, or doubling a table of fewer than 64 bins,
    /// when it already held 8 entries) or adds it to its tree bin, then
    /// doubles the table if the size passes the threshold.
    pub fn put(&mut self, key: K, value: V) {
        if self.capacity == 0 {
            self.resize();
        }
        let h = spread(key.java_hash());
        let bin = h as usize & (self.capacity - 1);
        if let Some(idx) = self.find(bin, h, &key) {
            self.entries[idx as usize].as_mut().expect("live entry").2 = value;
            return;
        }
        let idx = self.push(key, h, value);
        if self.bins[bin].tree {
            self.put_tree_val(bin, idx);
        } else {
            let before = self.bins[bin].nodes.len();
            self.bins[bin].nodes.push(idx);
            if before >= TREEIFY_THRESHOLD {
                self.treeify_bin(bin);
            }
        }
        self.size += 1;
        if self.size > self.threshold {
            self.resize();
        }
    }

    /// `HashMap.merge(key, value, remappingFunction)` for a non-null
    /// remapping result: resizes first when `size > threshold || capacity == 0`,
    /// updates an existing entry in place, or inserts a new entry at the head
    /// of its list bin (making the bin a tree, or doubling a table of fewer
    /// than 64 bins, when it already held 7 entries) or into its tree bin.
    pub fn merge(&mut self, key: K, value: V, combine: impl FnOnce(&mut V, V)) {
        self.resize_before_merge();
        let h = spread(key.java_hash());
        let bin = h as usize & (self.capacity - 1);
        if let Some(idx) = self.find(bin, h, &key) {
            combine(&mut self.entries[idx as usize].as_mut().expect("live entry").2, value);
            return;
        }
        let idx = self.push(key, h, value);
        if self.bins[bin].tree {
            self.put_tree_val(bin, idx);
        } else {
            let before = self.bins[bin].nodes.len();
            self.bins[bin].nodes.insert(0, idx);
            if before + 1 >= TREEIFY_THRESHOLD {
                self.treeify_bin(bin);
            }
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

    fn push(&mut self, key: K, h: u32, value: V) -> u32 {
        let idx = self.entries.len() as u32;
        self.entries.push(Some((key, h, value)));
        self.links.push(NO_LINKS);
        idx
    }

    fn key(&self, idx: u32) -> (&K, u32) {
        let (k, h, _) = self.entries[idx as usize].as_ref().expect("live entry");
        (k, *h)
    }

    /// The side of tree node `p` that the new node `x` goes to: `-1` (left)
    /// or `1` (right). Nodes are ordered by their hash as a signed `int`,
    /// then by [`JavaHash::tree_order`], and the rest by `tieBreakOrder`,
    /// whose `System.identityHashCode` comparison is not reproducible
    /// (`-1` is used and the map is marked not exact).
    fn dir(&mut self, x: u32, p: u32) -> i32 {
        let (xk, xh) = self.key(x);
        let (pk, ph) = self.key(p);
        let (xh, ph) = (xh as i32, ph as i32);
        if ph > xh {
            return -1;
        }
        if ph < xh {
            return 1;
        }
        match xk.tree_order(pk) {
            Some(std::cmp::Ordering::Less) => -1,
            Some(std::cmp::Ordering::Greater) => 1,
            _ => {
                self.exact = false;
                -1
            }
        }
    }

    /// `treeifyBin`: doubles a table of fewer than `MIN_TREEIFY_CAPACITY`
    /// bins, or turns the bin into a tree.
    fn treeify_bin(&mut self, bin: usize) {
        if self.capacity < MIN_TREEIFY_CAPACITY {
            self.resize();
        } else {
            self.bins[bin].tree = true;
            self.treeify(bin);
        }
    }

    /// `TreeNode.treeify`: builds a red-black tree by inserting the bin's
    /// nodes in `next` order, then moves the root to the front.
    fn treeify(&mut self, bin: usize) {
        let mut root = NIL;
        for i in 0..self.bins[bin].nodes.len() {
            let x = self.bins[bin].nodes[i];
            self.links[x as usize].left = NIL;
            self.links[x as usize].right = NIL;
            if root == NIL {
                self.links[x as usize].parent = NIL;
                self.links[x as usize].red = false;
                root = x;
                continue;
            }
            let mut p = root;
            loop {
                let dir = self.dir(x, p);
                let xp = p;
                p = if dir <= 0 { self.links[xp as usize].left } else { self.links[xp as usize].right };
                if p == NIL {
                    self.links[x as usize].parent = xp;
                    if dir <= 0 {
                        self.links[xp as usize].left = x;
                    } else {
                        self.links[xp as usize].right = x;
                    }
                    root = self.balance_insertion(root, x);
                    break;
                }
            }
        }
        self.move_root_to_front(bin, root);
    }

    /// `TreeNode.putTreeVal` for a key not in the tree: links `x` under the
    /// leaf it sorts to and right after that leaf in `next` order, then
    /// rebalances and moves the root to the front.
    fn put_tree_val(&mut self, bin: usize, x: u32) {
        let root = self.bins[bin].nodes[0];
        let mut p = root;
        loop {
            let dir = self.dir(x, p);
            let xp = p;
            p = if dir <= 0 { self.links[xp as usize].left } else { self.links[xp as usize].right };
            if p == NIL {
                let at = self.bins[bin].nodes.iter().position(|&n| n == xp).expect("parent in its bin");
                self.bins[bin].nodes.insert(at + 1, x);
                self.links[x as usize] = Links { parent: xp, ..NO_LINKS };
                if dir <= 0 {
                    self.links[xp as usize].left = x;
                } else {
                    self.links[xp as usize].right = x;
                }
                let root = self.balance_insertion(root, x);
                self.move_root_to_front(bin, root);
                return;
            }
        }
    }

    /// `HashMap.resize` for the call sites that can double a table:
    /// `put`, `merge`, `putAll` and `treeifyBin`.
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
        self.bins.resize_with(new_cap, Bin::default);
        if old_cap == 0 {
            return;
        }
        let bit = old_cap as u32;
        for j in 0..old_cap {
            let Bin { nodes, tree } = std::mem::take(&mut self.bins[j]);
            if nodes.is_empty() {
                continue;
            }
            let (lo, hi): (Vec<u32>, Vec<u32>) = nodes.into_iter().partition(|&idx| self.key(idx).1 & bit == 0);
            let (lo_empty, hi_empty) = (lo.is_empty(), hi.is_empty());
            if !tree {
                self.bins[j].nodes = lo;
                self.bins[j + old_cap].nodes = hi;
                continue;
            }
            // `TreeNode.split`: a half of at most `UNTREEIFY_THRESHOLD`
            // nodes becomes a list again; a longer one stays a tree, rebuilt
            // when the other half took some of its nodes.
            for (at, half, other_empty) in [(j, lo, hi_empty), (j + old_cap, hi, lo_empty)] {
                if half.is_empty() {
                    continue;
                }
                let tree = half.len() > UNTREEIFY_THRESHOLD;
                self.bins[at] = Bin { nodes: half, tree };
                if tree && !other_empty {
                    self.treeify(at);
                }
            }
        }
    }

    /// The resize check `merge` makes before it looks up its key, which
    /// doubles a full table even when the key is already present.
    fn resize_before_merge(&mut self) {
        if self.size > self.threshold || self.capacity == 0 {
            self.resize();
        }
    }

    /// `moveRootToFront`: makes `root` the first node of its bin.
    fn move_root_to_front(&mut self, bin: usize, root: u32) {
        let nodes = &mut self.bins[bin].nodes;
        if nodes[0] != root {
            let at = nodes.iter().position(|&n| n == root).expect("root in its bin");
            nodes.remove(at);
            nodes.insert(0, root);
        }
    }

    /// `TreeNode.rotateLeft`.
    fn rotate_left(&mut self, mut root: u32, p: u32) -> u32 {
        let l = &mut self.links;
        let r = if p == NIL { NIL } else { l[p as usize].right };
        if r != NIL {
            let rl = l[r as usize].left;
            l[p as usize].right = rl;
            if rl != NIL {
                l[rl as usize].parent = p;
            }
            let pp = l[p as usize].parent;
            l[r as usize].parent = pp;
            if pp == NIL {
                root = r;
                l[r as usize].red = false;
            } else if l[pp as usize].left == p {
                l[pp as usize].left = r;
            } else {
                l[pp as usize].right = r;
            }
            l[r as usize].left = p;
            l[p as usize].parent = r;
        }
        root
    }

    /// `TreeNode.rotateRight`.
    fn rotate_right(&mut self, mut root: u32, p: u32) -> u32 {
        let l = &mut self.links;
        let left = if p == NIL { NIL } else { l[p as usize].left };
        if left != NIL {
            let lr = l[left as usize].right;
            l[p as usize].left = lr;
            if lr != NIL {
                l[lr as usize].parent = p;
            }
            let pp = l[p as usize].parent;
            l[left as usize].parent = pp;
            if pp == NIL {
                root = left;
                l[left as usize].red = false;
            } else if l[pp as usize].right == p {
                l[pp as usize].right = left;
            } else {
                l[pp as usize].left = left;
            }
            l[left as usize].right = p;
            l[p as usize].parent = left;
        }
        root
    }

    /// `TreeNode.balanceInsertion`: restores the red-black properties after
    /// `x` was linked as a leaf, and returns the root.
    fn balance_insertion(&mut self, mut root: u32, mut x: u32) -> u32 {
        self.links[x as usize].red = true;
        loop {
            let mut xp = self.links[x as usize].parent;
            if xp == NIL {
                self.links[x as usize].red = false;
                return x;
            }
            let mut xpp = self.links[xp as usize].parent;
            if !self.links[xp as usize].red || xpp == NIL {
                return root;
            }
            let xppl = self.links[xpp as usize].left;
            if xp == xppl {
                let xppr = self.links[xpp as usize].right;
                if xppr != NIL && self.links[xppr as usize].red {
                    self.links[xppr as usize].red = false;
                    self.links[xp as usize].red = false;
                    self.links[xpp as usize].red = true;
                    x = xpp;
                } else {
                    if x == self.links[xp as usize].right {
                        x = xp;
                        root = self.rotate_left(root, x);
                        xp = self.links[x as usize].parent;
                        xpp = if xp == NIL { NIL } else { self.links[xp as usize].parent };
                    }
                    if xp != NIL {
                        self.links[xp as usize].red = false;
                        if xpp != NIL {
                            self.links[xpp as usize].red = true;
                            root = self.rotate_right(root, xpp);
                        }
                    }
                }
            } else if xppl != NIL && self.links[xppl as usize].red {
                self.links[xppl as usize].red = false;
                self.links[xp as usize].red = false;
                self.links[xpp as usize].red = true;
                x = xpp;
            } else {
                if x == self.links[xp as usize].left {
                    x = xp;
                    root = self.rotate_right(root, x);
                    xp = self.links[x as usize].parent;
                    xpp = if xp == NIL { NIL } else { self.links[xp as usize].parent };
                }
                if xp != NIL {
                    self.links[xp as usize].red = false;
                    if xpp != NIL {
                        self.links[xpp as usize].red = true;
                        root = self.rotate_left(root, xpp);
                    }
                }
            }
        }
    }
}

/// `HashMap.tableSizeFor`: the least power of two at or above `cap`.
fn table_size_for(cap: usize) -> usize {
    cap.max(1).next_power_of_two().min(MAXIMUM_CAPACITY)
}

/// A key of [`HashMapOrder`]'s full model: an index with the hash code it
/// stands for, equal only to itself and not comparable, as a key whose
/// class `HashMap` cannot compare.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Slot {
    index: u32,
    hash: i32,
}

impl JavaHash for Slot {
    fn java_hash(&self) -> i32 {
        self.hash
    }
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
    fn a_long_bin_grows_a_small_table_and_becomes_a_tree_in_a_large_one() {
        let hashes: Vec<i32> = (0..8).map(|i| i * 64).collect();
        assert_eq!(order(&hashes, Insert::Merge { again: false }).2, 32);
        assert_eq!(order(&hashes, Insert::Put).2, 16);
        let hashes: Vec<i32> = (0..9).map(|i| i * 64).collect();
        assert_eq!(order(&hashes, Insert::Put).2, 32);
        // The orders JDK 21 gives: the tree's root first, then the rest in
        // list order.
        let hashes: Vec<i32> = (0..10).map(|i| i * 64).collect();
        assert_eq!(order(&hashes, Insert::Merge { again: false }), (vec![6, 9, 8, 7, 5, 4, 3, 2, 1, 0], true, 64));
        let hashes: Vec<i32> = (0..12).map(|i| i * 64).collect();
        assert_eq!(order(&hashes, Insert::Put), (vec![3, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11], true, 64));
        // Equal hash codes in a tree are ordered by identity on the JVM.
        let (out, exact, capacity) = order(&[64; 10], Insert::Merge { again: false });
        assert_eq!((exact, capacity, out.len()), (false, 64, 10));
    }

    #[test]
    fn strings_with_one_hash_code_in_a_tree_are_ordered_by_compare_to() {
        // "Aa" and "BB" share a hash code, so these ten strings all have
        // one. The ninth turns their bin of a table of 64 into a tree.
        let keys = [
            "BBAaAaBB", "BBAaAaAa", "AaBBBBBB", "AaBBBBAa", "AaBBAaBB", "AaBBAaAa", "AaAaBBBB", "AaAaBBAa", "AaAaAaBB",
            "AaAaAaAa",
        ];
        let mut map = JavaHashMap::with_capacity(64);
        for k in keys {
            map.put(k, ());
        }
        assert_eq!((map.tree_bins(), map.is_jvm_exact()), (1, true));
        // The order JDK 21 gives.
        let order: Vec<&str> = map.keys().copied().collect();
        assert_eq!(
            order,
            [
                "AaBBBBAa", "BBAaAaBB", "BBAaAaAa", "AaBBBBBB", "AaBBAaBB", "AaBBAaAa", "AaAaBBBB", "AaAaBBAa",
                "AaAaAaBB", "AaAaAaAa"
            ]
        );
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
