//! SHA-256 and SHA-512 (FIPS 180-4) and HMAC-SHA-256 (RFC 2104), the one
//! implementation every crate hashes with, and SHA-1 for reading the id
//! git gives a file.
//!
//! A layer's provenance names the model that made it by the SHA-256 of
//! the model file, the same value `sha256sum` prints and the golden files
//! record; the zoo checks every file it fetches and serves by its SHA-256
//! and SHA-512; the S3 signer and the PII tokenizers use the HMAC. Models
//! are hundreds of megabytes, so SHA-256 runs on the CPU's SHA
//! instructions where it has them (see `compress`).

mod compress;
mod hmac;
mod sha1;
mod sha512;

pub use hmac::hmac_sha256;
pub use sha1::{git_blob_id, sha1};
pub use sha512::{Sha512, sha512};

/// The initial hash value: the first 32 bits of the fractional parts of
/// the square roots of the first eight primes.
const H0: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];

/// A SHA-256 digest.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Digest(pub [u8; 32]);

impl Digest {
    /// The all-zero 256-bit digest, used as a sentinel when an optional digest is absent.
    pub const ZERO: Digest = Digest([0u8; 32]);

    /// The digest as 64 lower case hex digits, as `sha256sum` prints it.
    pub fn to_hex(&self) -> String {
        to_hex(self.0)
    }

    /// Raw 32-byte digest array.
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// `bytes` as lower case hex, two digits a byte, as `sha256sum` prints a digest.
pub fn to_hex(bytes: impl AsRef<[u8]>) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let bytes = bytes.as_ref();
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[usize::from(b >> 4)] as char);
        s.push(HEX[usize::from(b & 0x0F)] as char);
    }
    s
}

impl AsRef<[u8]> for Digest {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

/// Streaming SHA-256 hasher with length-prefixed framing helpers for
/// structured provenance and content digests.
#[derive(Clone, Debug)]
pub struct Sha256 {
    state: [u32; 8],
    buf: [u8; 64],
    buf_len: usize,
    total_len: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    /// Creates a new streaming SHA-256 state at the FIPS 180-4 initial vector.
    pub const fn new() -> Self {
        Self { state: H0, buf: [0u8; 64], buf_len: 0, total_len: 0 }
    }

    /// Feeds raw bytes into the SHA-256 state.
    pub fn update(&mut self, mut data: &[u8]) -> &mut Self {
        self.total_len = self.total_len.wrapping_add(data.len() as u64);
        if self.buf_len > 0 {
            let take = (64 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 64 {
                compress::blocks(&mut self.state, &[self.buf]);
                self.buf_len = 0;
            }
        }
        let (blocks, rest) = data.as_chunks::<64>();
        compress::blocks(&mut self.state, blocks);
        if !rest.is_empty() {
            self.buf[..rest.len()].copy_from_slice(rest);
            self.buf_len = rest.len();
        }
        self
    }

    /// Feeds a length-prefixed UTF-8 string (`u64` little-endian length followed by bytes).
    pub fn str(&mut self, s: &str) -> &mut Self {
        self.u64(s.len() as u64).update(s.as_bytes())
    }

    /// Feeds a single byte.
    pub fn u8(&mut self, v: u8) -> &mut Self {
        self.update(&[v])
    }

    /// Feeds a `u16` in little-endian byte order.
    pub fn u16(&mut self, v: u16) -> &mut Self {
        self.update(&v.to_le_bytes())
    }

    /// Feeds a `u32` in little-endian byte order.
    pub fn u32(&mut self, v: u32) -> &mut Self {
        self.update(&v.to_le_bytes())
    }

    /// Feeds a `u64` in little-endian byte order.
    pub fn u64(&mut self, v: u64) -> &mut Self {
        self.update(&v.to_le_bytes())
    }

    /// Feeds a `usize` canonically as a little-endian `u64`.
    pub fn usize(&mut self, v: usize) -> &mut Self {
        self.u64(v as u64)
    }

    /// Finalizes padding and returns the 256-bit [`Digest`].
    pub fn finish(mut self) -> Digest {
        let mut tail = [0u8; 128];
        tail[..self.buf_len].copy_from_slice(&self.buf[..self.buf_len]);
        tail[self.buf_len] = 0x80;
        let tail_len = if self.buf_len < 56 { 64 } else { 128 };
        let bits = self.total_len.wrapping_mul(8);
        tail[tail_len - 8..tail_len].copy_from_slice(&bits.to_be_bytes());
        let (tail_blocks, _) = tail[..tail_len].as_chunks::<64>();
        compress::blocks(&mut self.state, tail_blocks);
        let mut out = [0u8; 32];
        for (word, bytes) in self.state.iter().zip(out.as_chunks_mut::<4>().0) {
            *bytes = word.to_be_bytes();
        }
        Digest(out)
    }
}

/// The SHA-256 digest of `data`.
pub fn sha256(data: &[u8]) -> Digest {
    let mut h = Sha256::new();
    h.update(data);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(data: &[u8]) -> String {
        sha256(data).to_hex()
    }

    #[test]
    fn standard_test_vectors() {
        // FIPS 180-4 examples and the NIST short message vectors.
        assert_eq!(hex(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(
            hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        assert_eq!(
            hex(b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"),
            "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1"
        );
        assert_eq!(hex(&vec![b'a'; 1_000_000]), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    }

    #[test]
    fn padding_boundaries() {
        // 55 bytes fit the length in the same block, 56 need a second one,
        // 64 is exactly one block of data.
        assert_eq!(hex(&[b'a'; 55]), "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318");
        assert_eq!(hex(&[b'a'; 56]), "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a");
        assert_eq!(hex(&[b'a'; 64]), "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb");
    }
}
