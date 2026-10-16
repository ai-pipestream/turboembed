//! The checksum a snapshot ends with: a 64-bit hash of every byte before
//! it, so a file damaged on disk or cut short in a copy is refused instead
//! of answering queries wrongly. It finds accidents, not tampering: anyone
//! can write a matching checksum for altered bytes.
//!
//! The bytes are read as little-endian 64-bit words, four at a time, each
//! of four lanes taking every fourth word through a multiply and rotate, so
//! the lanes run side by side; the lanes, the zero-padded last words and
//! the length are mixed at the end. [`Summing`] computes the same value
//! over bytes written in pieces of any size.

use std::io::{self, Write};

const K: u64 = 0x9E37_79B9_7F4A_7C15;
const BLOCK: usize = 32;

fn step(lane: u64, word: u64) -> u64 {
    (lane ^ word).wrapping_mul(K).rotate_left(31)
}

/// The four lanes over whole 32 byte blocks.
#[derive(Clone, Copy)]
struct Lanes([u64; 4]);

impl Lanes {
    fn new() -> Lanes {
        Lanes([1, 2, 3, 4])
    }

    fn blocks(&mut self, blocks: &[[u8; BLOCK]]) {
        for block in blocks {
            let (words, _) = block.as_chunks::<8>();
            for (lane, word) in self.0.iter_mut().zip(words) {
                *lane = step(*lane, u64::from_le_bytes(*word));
            }
        }
    }

    /// Mixes in the last bytes, fewer than a block, and the total length.
    fn finish(mut self, tail: &[u8], len: u64) -> u64 {
        let mut block = [0u8; BLOCK];
        block[..tail.len()].copy_from_slice(tail);
        if !tail.is_empty() {
            self.blocks(&[block]);
        }
        let mut h = len.wrapping_mul(K);
        for lane in self.0 {
            h = step(h, lane);
        }
        h ^= h >> 32;
        h = h.wrapping_mul(0xD6E8_FEB8_6659_FD93);
        h ^ (h >> 32)
    }
}

/// The checksum of `bytes`.
pub(crate) fn checksum(bytes: &[u8]) -> u64 {
    let (blocks, tail) = bytes.as_chunks::<BLOCK>();
    let mut lanes = Lanes::new();
    lanes.blocks(blocks);
    lanes.finish(tail, bytes.len() as u64)
}

/// A writer that passes bytes on and computes their [`checksum`].
pub(crate) struct Summing<W> {
    inner: W,
    lanes: Lanes,
    pending: [u8; BLOCK],
    filled: usize,
    len: u64,
}

impl<W: Write> Summing<W> {
    pub(crate) fn new(inner: W) -> Summing<W> {
        Summing { inner, lanes: Lanes::new(), pending: [0; BLOCK], filled: 0, len: 0 }
    }

    /// The checksum of everything written, and the inner writer.
    pub(crate) fn finish(self) -> (u64, W) {
        (self.lanes.finish(&self.pending[..self.filled], self.len), self.inner)
    }
}

impl<W: Write> Write for Summing<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(buf)?;
        let mut rest = &buf[..n];
        self.len += n as u64;
        if self.filled > 0 {
            let take = rest.len().min(BLOCK - self.filled);
            self.pending[self.filled..self.filled + take].copy_from_slice(&rest[..take]);
            self.filled += take;
            rest = &rest[take..];
            if self.filled < BLOCK {
                return Ok(n);
            }
            self.lanes.blocks(&[self.pending]);
            self.filled = 0;
        }
        let (blocks, tail) = rest.as_chunks::<BLOCK>();
        self.lanes.blocks(blocks);
        self.pending[..tail.len()].copy_from_slice(tail);
        self.filled = tail.len();
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pieces_of_any_size_sum_as_the_whole() {
        let bytes: Vec<u8> = (0..1000u32).map(|i| (i * 7 + i / 13) as u8).collect();
        for piece in [1, 3, 8, 31, 32, 33, 100, 1000] {
            let mut w = Summing::new(Vec::new());
            for chunk in bytes.chunks(piece) {
                w.write_all(chunk).unwrap();
            }
            let (sum, out) = w.finish();
            assert_eq!(out, bytes);
            assert_eq!(sum, checksum(&bytes), "pieces of {piece}");
        }
    }

    #[test]
    fn a_changed_or_missing_byte_changes_the_sum() {
        let bytes = vec![0u8; 100];
        let mut other = bytes.clone();
        other[57] = 1;
        assert_ne!(checksum(&bytes), checksum(&other));
        assert_ne!(checksum(&bytes), checksum(&bytes[..99]));
        assert_ne!(checksum(&[]), checksum(&[0]));
    }
}
