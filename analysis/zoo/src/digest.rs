//! SHA-256 and SHA-512 digests, written as lowercase hex.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::Digest as _;

/// A digest of `N` bytes. `Hex<32>` is a SHA-256, `Hex<64>` a SHA-512.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Hex<const N: usize>(pub [u8; N]);

pub type Sha256 = Hex<32>;
pub type Sha512 = Hex<64>;

impl Sha256 {
    pub fn of(bytes: &[u8]) -> Sha256 {
        Hex(sha2::Sha256::digest(bytes).into())
    }
}

impl Sha512 {
    pub fn of(bytes: &[u8]) -> Sha512 {
        Hex(sha2::Sha512::digest(bytes).into())
    }
}

impl<const N: usize> fmt::Display for Hex<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for b in self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

impl<const N: usize> fmt::Debug for Hex<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl<const N: usize> FromStr for Hex<N> {
    type Err = String;
    /// Exactly `2 * N` lowercase hex digits, the form `sha256sum` and
    /// `sha512sum` print. Uppercase is refused so that one digest has one
    /// spelling in the index.
    fn from_str(s: &str) -> Result<Self, String> {
        let b = s.as_bytes();
        if b.len() != 2 * N {
            return Err(format!("{s:?} is not {} lowercase hex digits", 2 * N));
        }
        let mut out = [0u8; N];
        for (i, pair) in b.as_chunks::<2>().0.iter().enumerate() {
            let (Some(hi), Some(lo)) = (nibble(pair[0]), nibble(pair[1])) else {
                return Err(format!("{s:?} is not {} lowercase hex digits", 2 * N));
            };
            out[i] = hi << 4 | lo;
        }
        Ok(Hex(out))
    }
}

fn nibble(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        _ => None,
    }
}

impl<const N: usize> Serialize for Hex<N> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

impl<'de, const N: usize> Deserialize<'de> for Hex<N> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_digests() {
        assert_eq!(Sha256::of(b"").to_string(), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(
            Sha512::of(b"abc").to_string(),
            "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a\
             2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f"
        );
    }

    #[test]
    fn parse_is_strict() {
        let h = Sha256::of(b"x").to_string();
        assert_eq!(h.parse::<Sha256>().unwrap().to_string(), h);
        assert!(h.to_uppercase().parse::<Sha256>().is_err());
        assert!(h[1..].parse::<Sha256>().is_err());
        assert!(format!("{h}0").parse::<Sha256>().is_err());
        assert!(h.parse::<Sha512>().is_err());
        assert!("g".repeat(64).parse::<Sha256>().is_err());
    }
}
