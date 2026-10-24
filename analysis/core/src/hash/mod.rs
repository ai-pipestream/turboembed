//! The canonical deterministic FNV-1a hasher for `config_hash`
//! and fingerprint implementations across the analysis crates.
//!
//! One framing everywhere: strings are length-prefixed so
//! adjacent fields cannot collide (`"ab" + "c"` differs from
//! `"a" + "bc"`), integers are little-endian at canonical widths
//! (`usize` as `u64`, never `to_ne_bytes`), and floats are
//! canonicalized (`-0.0` becomes `+0.0`, every NaN becomes one
//! quiet NaN) before their bits are mixed. Hashes are therefore
//! platform-independent and stable across runs.

use std::hash::Hasher;

use crate::layer::{Key, KeyId, Layer};

/// The 64-bit FNV-1a offset basis.
pub const FNV1A_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// The 64-bit FNV-1a prime.
pub const FNV1A_PRIME: u64 = 0x0100_0000_01b3;

/// The FNV-1a hash of `bytes` from the offset basis.
pub const fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = FNV1A_OFFSET_BASIS;
    let mut i = 0;
    while i < bytes.len() {
        h ^= bytes[i] as u64;
        h = h.wrapping_mul(FNV1A_PRIME);
        i += 1;
    }
    h
}

/// A chainable FNV-1a builder; see the module docs for framing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fnv1a(u64);

impl Default for Fnv1a {
    fn default() -> Self {
        Self::new()
    }
}

impl Fnv1a {
    /// A fresh hasher at the offset basis.
    pub const fn new() -> Self {
        Self(FNV1A_OFFSET_BASIS)
    }

    /// A hasher continued from a prior `finish()` (or any seed).
    pub const fn with_seed(seed: u64) -> Self {
        Self(seed)
    }

    /// The current hash value.
    pub const fn finish(&self) -> u64 {
        self.0
    }

    /// Mixes raw bytes, unframed.
    pub fn bytes(&mut self, bytes: &[u8]) -> &mut Self {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(FNV1A_PRIME);
        }
        self
    }

    /// Mixes a string framed by its length as `u64` little-endian.
    pub fn str(&mut self, s: &str) -> &mut Self {
        self.u64(s.len() as u64).bytes(s.as_bytes())
    }

    pub fn u8(&mut self, v: u8) -> &mut Self {
        self.bytes(&[v])
    }

    pub fn u16(&mut self, v: u16) -> &mut Self {
        self.bytes(&v.to_le_bytes())
    }

    pub fn u32(&mut self, v: u32) -> &mut Self {
        self.bytes(&v.to_le_bytes())
    }

    pub fn u64(&mut self, v: u64) -> &mut Self {
        self.bytes(&v.to_le_bytes())
    }

    pub fn i32(&mut self, v: i32) -> &mut Self {
        self.bytes(&v.to_le_bytes())
    }

    pub fn i64(&mut self, v: i64) -> &mut Self {
        self.bytes(&v.to_le_bytes())
    }

    /// Mixes a `usize` canonically as `u64` little-endian.
    pub fn usize(&mut self, v: usize) -> &mut Self {
        self.u64(v as u64)
    }

    pub fn bool(&mut self, v: bool) -> &mut Self {
        self.u8(u8::from(v))
    }

    /// Mixes an `f32` with `-0.0` and NaN canonicalized.
    pub fn f32(&mut self, v: f32) -> &mut Self {
        let bits = if v == 0.0 {
            0
        } else if v.is_nan() {
            f32::NAN.to_bits()
        } else {
            v.to_bits()
        };
        self.bytes(&bits.to_le_bytes())
    }

    /// Mixes an `f64` with `-0.0` and NaN canonicalized.
    pub fn f64(&mut self, v: f64) -> &mut Self {
        let bits = if v == 0.0 {
            0
        } else if v.is_nan() {
            f64::NAN.to_bits()
        } else {
            v.to_bits()
        };
        self.bytes(&bits.to_le_bytes())
    }

    /// Mixes a layer key: its layer name and instance name,
    /// each length-framed.
    pub fn key_id(&mut self, k: KeyId) -> &mut Self {
        self.str(&k.layer.to_string()).str(k.name)
    }

    /// Mixes a typed layer key.
    pub fn key<L: Layer>(&mut self, key: Key<L>) -> &mut Self {
        self.key_id(key.id())
    }
}

impl Hasher for Fnv1a {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        self.bytes(bytes);
    }
}

#[cfg(test)]
mod tests;
