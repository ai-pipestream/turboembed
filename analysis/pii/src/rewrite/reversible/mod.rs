//! Reversible protection, behind the `reversible` feature: a holder of
//! the key can recover the original value. [`AesGcmEncryptor`] replaces
//! each mention with an authenticated AES-256-GCM token, and
//! [`Ff1Encryptor`] encrypts the digits or letters of each mention with FF1
//! (NIST SP 800-38G), so the replacement keeps the value's shape.
//!
//! Both take a 32-byte key from the caller. The key lives only in the
//! cipher's key schedule, which is zeroised on drop, and never in a
//! `Debug` output. Both rewrite the mention as it is written in the text
//! (its surface), so decryption gives back the text that was there.

mod aead;
mod ff1;

pub use aead::AesGcmEncryptor;
pub use ff1::{Ff1Alphabet, Ff1Encryptor};

use std::fmt;

use crate::checksum::sha256;

/// The key length both encryptors take: an AES-256 key.
pub const KEY_BYTES: usize = 32;

/// Why a value could not be encrypted or a token not decrypted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CryptError {
    /// The key is not 32 bytes long.
    KeyLength(usize),
    /// An empty value.
    EmptyValue,
    /// A value longer than the cipher takes.
    TooLong,
    /// The token is not `KIND:` followed by base64url.
    Malformed,
    /// The token failed authentication: wrong key, wrong context, or the
    /// token was changed.
    Authentication,
    /// A decrypted value that is not UTF-8.
    NotUtf8,
    /// A letter or digit outside the FF1 alphabet. It would be left in
    /// the clear, so the value is refused.
    OutsideAlphabet(char),
    /// Fewer letters or digits of the alphabet than FF1 takes: the domain
    /// must hold at least a million values.
    TooFewNumerals { have: usize, need: usize },
}

impl fmt::Display for CryptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CryptError::KeyLength(n) => write!(f, "the key has {n} bytes, an AES-256 key has {KEY_BYTES}"),
            CryptError::EmptyValue => f.write_str("the value must not be empty"),
            CryptError::TooLong => f.write_str("the value is longer than the cipher takes"),
            CryptError::Malformed => f.write_str("the token is not KIND: followed by base64url"),
            CryptError::Authentication => f.write_str("the token failed authentication"),
            CryptError::NotUtf8 => f.write_str("the decrypted value is not UTF-8"),
            CryptError::OutsideAlphabet(c) => write!(f, "{c:?} is a letter or digit outside the alphabet"),
            CryptError::TooFewNumerals { have, need } => {
                write!(f, "the value has {have} letters or digits of the alphabet, FF1 needs at least {need}")
            }
        }
    }
}

impl std::error::Error for CryptError {}

fn check_key(key: &[u8]) -> Result<(), CryptError> {
    if key.len() == KEY_BYTES { Ok(()) } else { Err(CryptError::KeyLength(key.len())) }
}

const BASE64URL: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

/// Appends `bytes` to `out` as unpadded base64url (RFC 4648 section 5).
fn base64url(bytes: &[u8], out: &mut String) {
    for chunk in bytes.chunks(3) {
        let n = chunk.iter().enumerate().fold(0u32, |n, (i, &b)| n | (u32::from(b) << (16 - 8 * i)));
        for i in 0..=chunk.len() {
            out.push(char::from(BASE64URL[((n >> (18 - 6 * i)) & 63) as usize]));
        }
    }
}

/// The bytes of unpadded base64url `text`; `None` for any other text.
fn from_base64url(text: &str) -> Option<Vec<u8>> {
    let sextet = |c: u8| BASE64URL.iter().position(|&b| b == c).map(|p| p as u32);
    let b = text.as_bytes();
    if b.len() % 4 == 1 {
        return None;
    }
    let mut out = Vec::with_capacity(b.len() * 3 / 4);
    for chunk in b.chunks(4) {
        let mut n = 0u32;
        for (i, &c) in chunk.iter().enumerate() {
            n |= sextet(c)? << (18 - 6 * i);
        }
        let bytes = chunk.len() - 1;
        // The bits past the last byte must be zero, so one token has one
        // spelling.
        if n & ((1 << (24 - 8 * bytes)) - 1) != 0 {
            return None;
        }
        out.extend((0..bytes).map(|i| (n >> (16 - 8 * i)) as u8));
    }
    Some(out)
}

/// 32 bytes from the system's random source: `/dev/urandom` on Unix, and
/// elsewhere std's hasher keys, which std draws from that source, with the
/// time, run through SHA-256.
fn random_seed() -> [u8; 32] {
    let mut out = [0u8; 32];
    #[cfg(unix)]
    {
        use std::io::Read;
        if std::fs::File::open("/dev/urandom").and_then(|mut f| f.read_exact(&mut out)).is_ok() {
            return out;
        }
    }
    use std::hash::{BuildHasher, Hasher};
    let mut seed = sha256::Sha256::default();
    for i in 0..8u64 {
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u64(i);
        seed.update(&h.finish().to_le_bytes());
    }
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
    seed.update(&now.to_le_bytes());
    out.copy_from_slice(&seed.finish());
    out
}

#[cfg(test)]
mod tests {
    use super::{base64url, from_base64url};

    fn enc(b: &[u8]) -> String {
        let mut s = String::new();
        base64url(b, &mut s);
        s
    }

    #[test]
    fn base64url_is_rfc_4648_without_padding_and_round_trips() {
        for (bytes, text) in [
            (&b""[..], ""),
            (b"f", "Zg"),
            (b"fo", "Zm8"),
            (b"foo", "Zm9v"),
            (b"foob", "Zm9vYg"),
            (b"fooba", "Zm9vYmE"),
            (b"foobar", "Zm9vYmFy"),
            (&[0xfb, 0xff], "-_8"),
        ] {
            assert_eq!(enc(bytes), text);
            assert_eq!(from_base64url(text).as_deref(), Some(bytes));
        }
        for bad in ["Z", "Zh", "Zm9", "Zm+v", "Zm9v=", "Zm 9"] {
            assert_eq!(from_base64url(bad), None, "{bad}");
        }
    }
}
