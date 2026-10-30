//! Bucket-order hash map reproducing `java.util.HashMap` (load factor 0.75,
//! initial capacity 16) for corpus readers whose output or error order
//! depends on map iteration order.

use std::borrow::Borrow;
use std::collections::{HashMap, VecDeque};
use std::hash::Hash;

const DEFAULT_CAPACITY: usize = 16;
const MAXIMUM_CAPACITY: usize = 1 << 30;
const TREEIFY_THRESHOLD: usize = 8;
const MIN_TREEIFY_CAPACITY: usize = 64;

/// A key with a 32-bit signed hash code matching Java's `Object.hashCode()`.
pub(crate) trait JavaHash {
    fn java_hash(&self) -> i32;
}

#[inline]
fn string_hash(s: &str) -> i32 {
    s.encode_utf16().fold(0i32, |h, u| h.wrapping_mul(31).wrapping_add(i32::from(u)))
}

#[inline]
fn spread(hash: i32) -> u32 {
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

/// Bucket-ordered map matching `java.util.HashMap`'s table growth and bin
/// placement for `put` and `merge`.
#[derive(Clone, Debug)]
pub(crate) struct JavaHashMap<K, V> {
    capacity: usize,
    threshold: usize,
    entries: Vec<Option<(K, u32, V)>>,
    index: HashMap<K, u32>,
    bins: Vec<VecDeque<u32>>,
}

impl<K, V> Default for JavaHashMap<K, V> {
    fn default() -> Self {
        Self { capacity: 0, threshold: 0, entries: Vec::new(), index: HashMap::new(), bins: Vec::new() }
    }
}

impl<K, V> JavaHashMap<K, V> {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (&K, &V)> + '_ {
        self.bins.iter().flat_map(|b| b.iter()).map(|&idx| {
            let (k, _, v) = self.entries[idx as usize].as_ref().expect("live entry");
            (k, v)
        })
    }

    pub(crate) fn into_ordered(mut self) -> Vec<(K, V)> {
        let mut out = Vec::with_capacity(self.entries.len());
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

impl<K: JavaHash + Eq + Hash + Clone, V> JavaHashMap<K, V> {
    pub(crate) fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let &idx = self.index.get(key)?;
        self.entries[idx as usize].as_mut().map(|(_, _, v)| v)
    }

    pub(crate) fn put(&mut self, key: K, value: V) {
        if self.capacity == 0 {
            self.resize();
        }
        if let Some(&idx) = self.index.get(&key) {
            self.entries[idx as usize].as_mut().expect("live entry").2 = value;
            return;
        }
        let h = spread(key.java_hash());
        let idx = self.entries.len() as u32;
        self.index.insert(key.clone(), idx);
        self.entries.push(Some((key, h, value)));
        let bin = h as usize & (self.capacity - 1);
        let before = self.bins[bin].len();
        self.bins[bin].push_back(idx);
        if before >= TREEIFY_THRESHOLD && self.capacity < MIN_TREEIFY_CAPACITY {
            self.resize();
        }
        if self.entries.len() > self.threshold {
            self.resize();
        }
    }

    pub(crate) fn merge(&mut self, key: K, value: V, combine: impl FnOnce(&mut V, V)) {
        if self.entries.len() > self.threshold || self.capacity == 0 {
            self.resize();
        }
        if let Some(&idx) = self.index.get(&key) {
            combine(&mut self.entries[idx as usize].as_mut().expect("live entry").2, value);
            return;
        }
        let h = spread(key.java_hash());
        let idx = self.entries.len() as u32;
        self.index.insert(key.clone(), idx);
        self.entries.push(Some((key, h, value)));
        let bin = h as usize & (self.capacity - 1);
        let before = self.bins[bin].len();
        self.bins[bin].push_front(idx);
        if before + 1 >= TREEIFY_THRESHOLD && self.capacity < MIN_TREEIFY_CAPACITY {
            self.resize();
        }
    }

    fn resize(&mut self) {
        if self.capacity == 0 {
            self.capacity = DEFAULT_CAPACITY;
            self.threshold = 12;
            self.bins.resize_with(DEFAULT_CAPACITY, VecDeque::new);
            return;
        }
        if self.capacity >= MAXIMUM_CAPACITY {
            self.threshold = i32::MAX as usize;
            return;
        }
        let old_cap = self.capacity;
        let new_cap = old_cap << 1;
        self.capacity = new_cap;
        self.threshold = if new_cap < MAXIMUM_CAPACITY { self.threshold << 1 } else { i32::MAX as usize };
        self.bins.resize_with(new_cap, VecDeque::new);
        let bit = old_cap as u32;
        for j in 0..old_cap {
            let nodes = std::mem::take(&mut self.bins[j]);
            for idx in nodes {
                let h = self.entries[idx as usize].as_ref().expect("live entry").1;
                if h & bit == 0 {
                    self.bins[j].push_back(idx);
                } else {
                    self.bins[j + old_cap].push_back(idx);
                }
            }
        }
    }
}

/// Returns the indices `0..keys.len()` in the order a fresh `HashMap`
/// iterates them after each distinct key is inserted with `put`.
pub(crate) fn put_order<I, K>(keys: I) -> (Vec<usize>, bool)
where
    I: IntoIterator<Item = K>,
    K: JavaHash + Eq + Hash + Clone,
{
    let mut map = JavaHashMap::new();
    for (i, k) in keys.into_iter().enumerate() {
        map.put(k, i);
    }
    (map.into_ordered().into_iter().map(|(_, i)| i).collect(), true)
}
