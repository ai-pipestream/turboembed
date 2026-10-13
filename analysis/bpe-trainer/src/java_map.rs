//! The iteration order of a `java.util.HashMap` (JDK 8 and later, as in
//! JDK 21) with the default load factor, for the maps of OpenNLP's
//! `BPETokenizerTrainer`.
//!
//! The table is an array of bins whose length (the capacity) is a power
//! of two, 16 when it is first made. A key goes into bin
//! `spread(hash) & (capacity - 1)`. `put` appends a new key to the end of
//! its bin and then doubles the table when the map holds more than three
//! quarters of its capacity; `merge` doubles the table first when it
//! already holds more than that, then puts a new key at the head of its
//! bin. A bin that grows long (nine keys after a `put`, eight after a
//! `merge`) doubles the table while the capacity is below 64. A resize
//! splits every bin in two and keeps the relative order of its keys.
//! Iteration walks the bins in index order, each front to back. So the
//! order is the keys sorted by bin at the final capacity, and within a
//! bin in the order they were put (`put`) or the reverse (`merge`).
//!
//! A long bin in a table of 64 or more becomes a red-black tree, which
//! iterates in an order of its own and, for keys with equal hashes that
//! are not `Comparable`, one that depends on `System.identityHashCode`
//! and so changes from run to run. That order is not reproduced: the bin
//! keeps the list order here, and [`HashMapOrder::order`] reports it.

/// `TREEIFY_THRESHOLD`.
const TREEIFY_THRESHOLD: u32 = 8;

/// `MIN_TREEIFY_CAPACITY`.
const MIN_TREEIFY_CAPACITY: usize = 64;

/// `DEFAULT_INITIAL_CAPACITY`.
const INITIAL_CAPACITY: usize = 16;

/// `String.hashCode`: over the UTF-16 units of `s`.
pub fn string_hash(s: &str) -> i32 {
    s.encode_utf16().fold(0i32, |h, u| h.wrapping_mul(31).wrapping_add(i32::from(u)))
}

/// `String.hashCode` of `a` followed by `b`, from the hash of each and
/// the number of UTF-16 units of `b`.
pub fn concat_hash(a: i32, b: i32, b_units: usize) -> i32 {
    let mut h = a;
    for _ in 0..b_units {
        h = h.wrapping_mul(31);
    }
    h.wrapping_add(b)
}

/// `List.hashCode` of a list whose elements have the hashes `items`.
pub fn list_hash(items: impl IntoIterator<Item = i32>) -> i32 {
    items.into_iter().fold(1i32, |h, e| h.wrapping_mul(31).wrapping_add(e))
}

/// `HashMap.hash`: the high half of the hash folded into the low half.
#[inline]
pub fn spread(hash: i32) -> u32 {
    let h = hash as u32;
    h ^ (h >> 16)
}

/// How a key is put into the map.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Insert {
    /// `put` and `putAll`: at the end of its bin.
    Put,
    /// `merge`: at the head of its bin. `merge` resizes a full table
    /// before it looks for the key, so a call for a key already there
    /// can resize too; `again` says whether there was such a call after
    /// the last new key.
    Merge { again: bool },
}

/// A `HashMap`'s capacity, kept across [`order`](HashMapOrder::order)
/// calls as `clear` keeps the table.
#[derive(Clone, Debug, Default)]
pub struct HashMapOrder {
    /// The table length; 0 before the table is made.
    capacity: usize,
    /// Keys in each bin at the current capacity.
    counts: Vec<u32>,
}

impl HashMapOrder {
    /// A new map, with no table yet.
    pub fn new() -> HashMapOrder {
        HashMapOrder::default()
    }

    /// The table length.
    #[cfg(test)]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Puts distinct keys into the map, which is empty (new or cleared),
    /// in the order of `spread`, which holds each key's spread hash, and
    /// writes into `out` the keys (indexes into `spread`) in the order the
    /// map then iterates them. Returns false when a bin would have become
    /// a tree, whose order is not reproduced; the keys are then all in
    /// `out`, that bin in list order.
    pub fn order(&mut self, spread: &[u32], insert: Insert, out: &mut Vec<u32>) -> bool {
        out.clear();
        let mut exact = true;
        let n = spread.len();
        for i in 0..n {
            if matches!(insert, Insert::Merge { .. }) && (self.capacity == 0 || i > self.threshold()) {
                self.grow(&spread[..i]);
            }
            if self.capacity == 0 {
                self.grow(&spread[..i]);
            }
            let bin = spread[i] as usize & (self.capacity - 1);
            let before = self.counts[bin];
            self.counts[bin] += 1;
            let long = match insert {
                Insert::Put => before >= TREEIFY_THRESHOLD,
                Insert::Merge { .. } => before + 1 >= TREEIFY_THRESHOLD,
            };
            if long {
                if self.capacity < MIN_TREEIFY_CAPACITY {
                    self.grow(&spread[..=i]);
                } else {
                    exact = false;
                }
            }
            if insert == Insert::Put && i + 1 > self.threshold() {
                self.grow(&spread[..=i]);
            }
        }
        if insert == (Insert::Merge { again: true }) && n > self.threshold() {
            self.grow(spread);
        }

        // A stable sort by bin at the final capacity, of the keys in put
        // order, or in reverse for `merge`.
        let mask = self.capacity.wrapping_sub(1);
        let mut starts = vec![0u32; self.capacity + 1];
        for &s in spread {
            starts[(s as usize & mask) + 1] += 1;
        }
        for i in 1..starts.len() {
            starts[i] += starts[i - 1];
        }
        out.resize(n, 0);
        let mut place = |k: usize| {
            let slot = &mut starts[spread[k] as usize & mask];
            out[*slot as usize] = k as u32;
            *slot += 1;
        };
        match insert {
            Insert::Put => (0..n).for_each(&mut place),
            Insert::Merge { .. } => (0..n).rev().for_each(&mut place),
        }
        // A cleared map keeps its table, but holds no key.
        self.counts.iter_mut().for_each(|c| *c = 0);
        exact
    }

    /// Three quarters of the capacity.
    fn threshold(&self) -> usize {
        self.capacity / 4 * 3
    }

    /// `resize`: the table made, or doubled, holding `keys`.
    fn grow(&mut self, keys: &[u32]) {
        self.capacity = if self.capacity == 0 { INITIAL_CAPACITY } else { self.capacity * 2 };
        self.counts.clear();
        self.counts.resize(self.capacity, 0);
        for &s in keys {
            self.counts[s as usize & (self.capacity - 1)] += 1;
        }
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
    fn hash_codes_are_javas() {
        assert_eq!(string_hash("ab"), 3105);
        assert_eq!(string_hash("😀"), 1_772_899);
        assert_eq!(concat_hash(string_hash("ab"), string_hash("c</w>"), 5), string_hash("abc</w>"));
        assert_eq!(list_hash([97, 98]), 31 * (31 + 97) + 98);
    }

    #[test]
    fn put_keeps_put_order_in_a_bin_and_merge_reverses_it() {
        // Bins 3, 1, 3, 0 and 1 (17 & 15).
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
        // A thirteenth key, then a call for a key already there.
        let hashes: Vec<i32> = (0..13).collect();
        assert_eq!(order(&hashes, Insert::Merge { again: true }).2, 32);
        let hashes: Vec<i32> = (0..12).collect();
        assert_eq!(order(&hashes, Insert::Merge { again: true }).2, 16);
    }

    #[test]
    fn a_long_bin_grows_a_small_table_and_is_reported_in_a_large_one() {
        // Eight keys of one bin: merge grows 16 to 32, put does not.
        let hashes: Vec<i32> = (0..8).map(|i| i * 64).collect();
        assert_eq!(order(&hashes, Insert::Merge { again: false }).2, 32);
        assert_eq!(order(&hashes, Insert::Put).2, 16);
        let hashes: Vec<i32> = (0..9).map(|i| i * 64).collect();
        assert_eq!(order(&hashes, Insert::Put).2, 32);
        // Bin 0 at every capacity up to 64: a tree at 64.
        let hashes: Vec<i32> = (0..10).map(|i| i * 64).collect();
        let (out, exact, capacity) = order(&hashes, Insert::Merge { again: false });
        assert_eq!((exact, capacity, out.len()), (false, 64, 10));
    }

    #[test]
    fn a_cleared_map_keeps_its_capacity() {
        let mut map = HashMapOrder::new();
        let mut out = Vec::new();
        let spread: Vec<u32> = (0..20).collect();
        map.order(&spread, Insert::Put, &mut out);
        assert_eq!(map.capacity(), 32);
        // 16 and 0 share a bin at 16, not at 32.
        map.order(&[16, 0], Insert::Put, &mut out);
        assert_eq!(out, [1, 0]);
    }
}
