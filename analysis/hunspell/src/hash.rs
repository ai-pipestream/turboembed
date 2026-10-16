//! The hash of the word maps: per eight-byte word of the key, an xor into
//! the state and a folded multiply (the 128-bit product of the state and a
//! constant, its halves xored together), as `turbo-maxent` hashes its
//! predicates. A dictionary is fixed once loaded, so the maps need no
//! defence against chosen collisions, and SipHash would cost several times
//! as much per lookup.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

#[derive(Default, Clone, Copy)]
pub(crate) struct FoldHasher(u64);

/// A map with [`FoldHasher`].
pub(crate) type FoldMap<K, V> = HashMap<K, V, BuildHasherDefault<FoldHasher>>;

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
        // The length keeps a key apart from itself padded with NUL bytes.
        self.add(bytes.len() as u64);
        let (words, rest) = bytes.as_chunks::<8>();
        for &w in words {
            self.add(u64::from_le_bytes(w));
        }
        let mut tail = 0u64;
        for (i, &b) in rest.iter().enumerate() {
            tail |= u64::from(b) << (8 * i);
        }
        self.add(tail);
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::BuildHasher;

    #[test]
    fn a_changed_byte_changes_the_hash() {
        let b = BuildHasherDefault::<FoldHasher>::default();
        assert_eq!(b.hash_one("walked"), b.hash_one("walked"));
        assert_ne!(b.hash_one("walked"), b.hash_one("walkes"));
        assert_ne!(b.hash_one("ab"), b.hash_one("ab\0"));
    }
}
