//! The strings a rule set names, each given a number, so a token's value
//! is looked up once per document and every test after that compares
//! numbers.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// The number of a string the rules name.
pub(crate) type Sym = u32;

/// A token value no rule names: it equals no literal.
pub(crate) const NONE: Sym = u32::MAX;

#[derive(Default, Clone)]
pub(crate) struct Vocab {
    index: HashMap<Box<str>, Sym, BuildHasherDefault<FoldHasher>>,
}

impl Vocab {
    pub(crate) fn intern(&mut self, s: &str) -> Sym {
        if let Some(&i) = self.index.get(s) {
            return i;
        }
        let i = self.index.len() as Sym;
        assert!(i < NONE, "fewer than u32::MAX distinct strings");
        self.index.insert(s.into(), i);
        i
    }

    pub(crate) fn get(&self, s: &str) -> Sym {
        self.index.get(s).copied().unwrap_or(NONE)
    }

    pub(crate) fn len(&self) -> usize {
        self.index.len()
    }

    pub(crate) fn entries(&self) -> impl Iterator<Item = (&str, &Sym)> {
        self.index.iter().map(|(k, v)| (&**k, v))
    }
}

/// The hash of turbo-maxent's predicate table: per eight-byte word, an xor
/// into the state and a folded multiply, which brings the high half of the
/// product down so strings with a long shared prefix still spread over the
/// table's buckets. The keys are the rules' own strings, fixed once the
/// rules are built, so the table needs no defence against chosen keys.
#[derive(Default, Clone, Copy)]
pub(crate) struct FoldHasher(u64);

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
        self.add(tail ^ ((rest.len() as u64) << 56));
    }

    #[inline]
    fn write_u8(&mut self, b: u8) {
        self.add(u64::from(b));
    }

    #[inline]
    fn write_u32(&mut self, n: u32) {
        self.add(u64::from(n));
    }

    #[inline]
    fn write_u64(&mut self, n: u64) {
        self.add(n);
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

/// FNV-1a, for the configuration hash a step records in its provenance:
/// the same rules give the same hash on every platform and Rust release.
pub(crate) struct Fnv(pub(crate) u64);

impl Default for Fnv {
    fn default() -> Fnv {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
}

impl Fnv {
    pub(crate) fn bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    pub(crate) fn str(&mut self, s: &str) {
        self.u64(s.len() as u64);
        self.bytes(s.as_bytes());
    }

    pub(crate) fn u64(&mut self, n: u64) {
        self.bytes(&n.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interned_strings_keep_their_number() {
        let mut v = Vocab::default();
        let a = v.intern("boston");
        let b = v.intern("celtics");
        assert_eq!(v.intern("boston"), a);
        assert_ne!(a, b);
        assert_eq!(v.get("celtics"), b);
        assert_eq!(v.get("lakers"), NONE);
        assert_eq!(v.len(), 2);
    }
}
