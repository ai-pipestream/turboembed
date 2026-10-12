//! The hash for the predicate table: per eight-byte word of the key, an
//! xor into the state and a folded multiply (the 128-bit product of the
//! state and a constant, with its high and low halves xored together),
//! the mixing step wyhash and foldhash use.
//!
//! A model's predicates are fixed once it is loaded, so the table needs no
//! defence against chosen collisions, and SipHash, the standard library's
//! default, would cost several times as much per lookup. FxHash, which
//! rustc uses, is as cheap but its plain multiply carries a key's bits only
//! upwards; a table indexes buckets by the low bits, so keys that share a
//! long prefix, as predicates do ("p1f1=", "suffix="), crowded into a few
//! buckets and lookups on a real tokenizer model were eight times slower.
//! The folded multiply brings the high half of the product down.

use std::hash::{BuildHasherDefault, Hasher};

#[derive(Default, Clone, Copy)]
pub(crate) struct FoldHasher(u64);

/// The `BuildHasher` for maps keyed by predicate bytes.
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
    fn equal_bytes_hash_equal_and_a_changed_byte_changes_the_hash() {
        let b = BuildFoldHasher::default();
        assert_eq!(b.hash_one(&b"predicate=abc"[..]), b.hash_one(&b"predicate=abc"[..]));
        assert_ne!(b.hash_one(&b"predicate=abc"[..]), b.hash_one(&b"predicate=abd"[..]));
        // The length prefix the slice hash writes keeps a key and its
        // zero-padded extension apart.
        assert_ne!(b.hash_one(&b"ab"[..]), b.hash_one(&b"ab\0"[..]));
    }

    #[test]
    fn keys_that_differ_late_spread_over_the_low_bits() {
        // Predicates share long prefixes ("p1f1=", "suffix="). A table of 1024
        // buckets indexed by the low bits must still see them spread out.
        let b = BuildFoldHasher::default();
        let mut buckets = std::collections::HashSet::new();
        for i in 0..1000 {
            let key = format!("prefix=abcdefgh{i}");
            buckets.insert(b.hash_one(key.as_bytes()) & 1023);
        }
        assert!(buckets.len() > 550, "{} buckets used", buckets.len());
    }
}
