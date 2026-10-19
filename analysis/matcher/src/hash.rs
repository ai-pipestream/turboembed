//! The hashes behind the vocabulary and the phrase trie.
//!
//! The vocabulary hashes strings with the folded multiply wyhash and
//! foldhash use (as `turbo-maxent`'s predicate table does): the inputs
//! are the matcher's own phrases and the document's tokens, so the table
//! needs no defence against chosen collisions, and SipHash would cost
//! several times as much per token.
//!
//! The trie's edges are `(state, token id)` pairs packed into a `u64`, in
//! one open addressing table with linear probing: two arrays, the keys and
//! the targets, so an edge costs twelve bytes plus the free slots.

use std::hash::{BuildHasherDefault, Hasher};

#[derive(Default, Clone, Copy)]
pub(crate) struct FoldHasher(u64);

/// The `BuildHasher` for maps keyed by strings.
pub(crate) type BuildFoldHasher = BuildHasherDefault<FoldHasher>;

impl FoldHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        // The first 64 bits of pi's fraction, as wyhash and foldhash use.
        let product = u128::from(self.0 ^ word) * 0x243f_6a88_85a3_08d3;
        self.0 = product as u64 ^ (product >> 64) as u64;
    }
}

impl Hasher for FoldHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let (words, rest) = bytes.as_chunks::<8>();
        for &w in words {
            self.add(u64::from_le_bytes(w));
        }
        let mut tail = 0u64;
        for (i, &b) in rest.iter().enumerate() {
            tail |= u64::from(b) << (8 * i);
        }
        self.add(tail ^ ((bytes.len() as u64) << 56));
    }

    #[inline]
    fn write_u8(&mut self, b: u8) {
        self.add(u64::from(b));
    }

    #[inline]
    fn write_usize(&mut self, n: usize) {
        self.add(n as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

/// A key no edge has: state `u32::MAX` is never made.
const FREE: u64 = u64::MAX;

/// The edges of a trie, `(state, token id) -> state`, in one open
/// addressing table.
#[derive(Clone, Debug)]
pub(crate) struct EdgeTable {
    keys: Vec<u64>,
    targets: Vec<u32>,
    len: usize,
    /// `64 - log2(capacity)`, the shift that takes a product's top bits.
    shift: u32,
}

impl Default for EdgeTable {
    fn default() -> Self {
        EdgeTable::with_capacity_log2(4)
    }
}

impl EdgeTable {
    fn with_capacity_log2(bits: u32) -> EdgeTable {
        let n = 1usize << bits;
        EdgeTable { keys: vec![FREE; n], targets: vec![0; n], len: 0, shift: 64 - bits }
    }

    #[inline]
    fn key(state: u32, token: u32) -> u64 {
        (u64::from(state) << 32) | u64::from(token)
    }

    #[inline]
    fn slot(&self, key: u64) -> usize {
        // Fibonacci hashing: the product's top bits index the table.
        (key.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> self.shift) as usize
    }

    /// The state the edge `(state, token)` leads to.
    #[inline]
    pub(crate) fn get(&self, state: u32, token: u32) -> Option<u32> {
        let key = EdgeTable::key(state, token);
        let mask = self.keys.len() - 1;
        let mut at = self.slot(key);
        loop {
            let k = self.keys[at];
            if k == key {
                return Some(self.targets[at]);
            }
            if k == FREE {
                return None;
            }
            at = (at + 1) & mask;
        }
    }

    /// Adds the edge `(state, token) -> target`; the edge must be new.
    pub(crate) fn insert(&mut self, state: u32, token: u32, target: u32) {
        // Grow past seven tenths full: linear probing slows sharply above.
        if (self.len + 1) * 10 > self.keys.len() * 7 {
            self.grow();
        }
        self.place(EdgeTable::key(state, token), target);
        self.len += 1;
    }

    fn place(&mut self, key: u64, target: u32) {
        let mask = self.keys.len() - 1;
        let mut at = self.slot(key);
        while self.keys[at] != FREE {
            debug_assert_ne!(self.keys[at], key, "an edge added twice");
            at = (at + 1) & mask;
        }
        self.keys[at] = key;
        self.targets[at] = target;
    }

    fn grow(&mut self) {
        let bits = 64 - self.shift + 1;
        let old = std::mem::replace(self, EdgeTable::with_capacity_log2(bits));
        for (k, t) in old.keys.into_iter().zip(old.targets) {
            if k != FREE {
                self.place(k, t);
            }
        }
        self.len = old.len;
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn heap_bytes(&self) -> usize {
        self.keys.capacity() * 8 + self.targets.capacity() * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edges_survive_growth() {
        let mut t = EdgeTable::default();
        for s in 0..5000u32 {
            t.insert(s, s * 7 + 1, s + 1);
            t.insert(s, s * 7 + 2, s + 2);
        }
        assert_eq!(t.len(), 10_000);
        for s in 0..5000u32 {
            assert_eq!(t.get(s, s * 7 + 1), Some(s + 1));
            assert_eq!(t.get(s, s * 7 + 2), Some(s + 2));
            assert_eq!(t.get(s, s * 7 + 3), None);
        }
        assert!(t.keys.len() * 7 >= t.len() * 10);
    }
}
