use std::borrow::Borrow;

use super::{DEFAULT_CAPACITY, JavaHash, MAXIMUM_CAPACITY, TREEIFY_THRESHOLD, spread, table_size_for};

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
    pub(super) capacity: usize,
    /// `threshold`: the size above which the table doubles, or before the
    /// table is allocated, the length to allocate (`0` for the default).
    pub(super) threshold: usize,
    pub(super) size: usize,
    pub(super) exact: bool,
    /// Entries in first insertion order: key, spread hash, value.
    pub(super) entries: Vec<Option<(K, u32, V)>>,
    /// The red-black tree links of each entry, by entry index; used only
    /// while the entry is in a tree bin.
    pub(super) links: Vec<Links>,
    pub(super) bins: Vec<Bin>,
}

/// One bin of the table: its entries in `next` order (the order the map
/// iterates them), and whether it is a tree bin. In a tree bin the first
/// entry is always the root, as `moveRootToFront` keeps it.
#[derive(Clone, Debug, Default)]
pub(super) struct Bin {
    pub(super) nodes: Vec<u32>,
    pub(super) tree: bool,
}

/// `TreeNode`'s `parent`, `left`, `right`, and `red` fields.
#[derive(Clone, Copy, Debug)]
pub(super) struct Links {
    pub(super) parent: u32,
    pub(super) left: u32,
    pub(super) right: u32,
    pub(super) red: bool,
}

/// No node (`null`).
pub(super) const NIL: u32 = u32::MAX;

/// `UNTREEIFY_THRESHOLD`.
pub(super) const UNTREEIFY_THRESHOLD: usize = 6;

pub(super) const NO_LINKS: Links = Links { parent: NIL, left: NIL, right: NIL, red: false };

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
    pub(super) fn with_table(capacity: usize) -> JavaHashMap<K, V> {
        let mut map = JavaHashMap::new();
        if capacity > 0 {
            let cap = capacity.min(MAXIMUM_CAPACITY);
            map.capacity = cap;
            map.threshold = if cap >= MAXIMUM_CAPACITY { i32::MAX as usize } else { cap / 4 * 3 };
            map.bins.resize_with(cap, Bin::default);
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
        Q: JavaHash + Eq + ?Sized,
    {
        let nodes = &self.bins[bin].nodes;
        if self.bins[bin].tree
            && let Some(&root) = nodes.first()
        {
            return self.find_in_tree(root, h, key);
        }
        nodes.iter().copied().find(|&idx| {
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

    pub(super) fn push(&mut self, key: K, h: u32, value: V) -> u32 {
        let idx = self.entries.len() as u32;
        self.entries.push(Some((key, h, value)));
        self.links.push(NO_LINKS);
        idx
    }

    pub(super) fn key(&self, idx: u32) -> (&K, u32) {
        let (k, h, _) = self.entries[idx as usize].as_ref().expect("live entry");
        (k, *h)
    }

    /// `HashMap.resize` for the call sites that can double a table:
    /// `put`, `merge`, `putAll` and `treeifyBin`.
    pub(super) fn resize(&mut self) {
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
    pub(super) fn resize_before_merge(&mut self) {
        if self.size > self.threshold || self.capacity == 0 {
            self.resize();
        }
    }
}
