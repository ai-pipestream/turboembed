//! The hash of the vocabulary table: per eight-byte word of the key, an
//! xor into the state and a folded multiply, as `turbo-maxent`'s predicate
//! table hashes. Feature strings share long prefixes (`bow=`, `ng=:`), so
//! the multiply's high half is folded down into the bits a table indexes
//! by. The vocabulary is fixed once loaded and needs no defence against
//! chosen collisions.

use std::hash::{BuildHasherDefault, Hasher};

#[derive(Default, Clone, Copy)]
pub(crate) struct FoldHasher(u64);

pub(crate) type BuildFoldHasher = BuildHasherDefault<FoldHasher>;

impl FoldHasher {
    #[inline]
    fn add(&mut self, word: u64) {
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
        self.add(tail);
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
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
    fn equal_keys_hash_equal_and_a_changed_byte_changes_the_hash() {
        let b = BuildFoldHasher::default();
        assert_eq!(b.hash_one("bow=apple"), b.hash_one("bow=apple"));
        assert_ne!(b.hash_one("bow=apple"), b.hash_one("bow=apply"));
        assert_ne!(b.hash_one("bow=apple"), b.hash_one("bow=apple "));
    }
}
