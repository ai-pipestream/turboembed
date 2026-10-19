use super::{
    DEFAULT_CAPACITY, JavaHash, JavaHashMap, MAXIMUM_CAPACITY, MIN_TREEIFY_CAPACITY, TREEIFY_THRESHOLD, spread,
};

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
            // `spread` is its own inverse. Key `Slot` by insertion position `i`
            // so repeated indices in `order_keys` remain distinct entries,
            // matching the non-tree bucket-sort path in `order_with`.
            let key = Slot { index: i as u32, hash: super::spread(spread[index as usize] as i32) as i32 };
            match insert {
                Insert::Put => map.put(key, ()),
                Insert::Merge { .. } => map.merge(key, (), |_, _| {}),
            }
        }
        if insert == (Insert::Merge { again: true }) {
            map.resize_before_merge();
        }
        self.capacity = map.capacity();
        out.extend(map.keys().map(|k| at(k.index as usize)));
        map.is_jvm_exact()
    }

    pub(super) fn threshold(&self) -> usize {
        if self.capacity >= MAXIMUM_CAPACITY { i32::MAX as usize } else { self.capacity / 4 * 3 }
    }

    fn grow(&mut self, spread: &[u32], count: usize, at: &impl Fn(usize) -> u32) {
        if self.capacity >= MAXIMUM_CAPACITY {
            return;
        }
        self.capacity = if self.capacity == 0 { DEFAULT_CAPACITY } else { (self.capacity * 2).min(MAXIMUM_CAPACITY) };
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
