//! Compressed transfers: a file the index serves gzip or zstd compressed
//! carries the compressed bytes' own size and SHA-256 beside the unpacked
//! file's size, SHA-256 and SHA-512. The zoo checks the bytes as they
//! arrive, unpacks them to exactly the unpacked size, and checks the
//! unpacked file as it checks any other, before anything is written.

use serde::{Deserialize, Serialize};

use crate::digest::Sha256;

/// How a file is compressed on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Compression {
    /// RFC 1952, one member.
    Gzip,
    /// RFC 8878, one or more frames.
    Zstd,
}

/// What a compressed file is on the wire.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Download {
    pub compression: Compression,
    /// The compressed bytes' size.
    pub download_bytes: u64,
    pub download_sha256: Sha256,
}

/// The gzip header flags (RFC 1952 section 2.3.1).
const FHCRC: u8 = 0x02;
const FEXTRA: u8 = 0x04;
const FNAME: u8 = 0x08;
const FCOMMENT: u8 = 0x10;
const RESERVED: u8 = 0xE0;

/// The deflate stream of a one-member gzip file, and its trailer's CRC-32
/// and size.
fn gzip_parts(b: &[u8]) -> Result<(&[u8], u32, u32), String> {
    if b.len() < 18 || b[0] != 0x1f || b[1] != 0x8b {
        return Err("not gzip".into());
    }
    if b[2] != 8 {
        return Err(format!("gzip compression method {} is not deflate", b[2]));
    }
    let flags = b[3];
    if flags & RESERVED != 0 {
        return Err("gzip header sets reserved flags".into());
    }
    let mut at = 10;
    let short = || "the gzip header ends early".to_string();
    if flags & FEXTRA != 0 {
        let n = b.get(at..at + 2).ok_or_else(short)?;
        at += 2 + usize::from(u16::from_le_bytes([n[0], n[1]]));
    }
    for flag in [FNAME, FCOMMENT] {
        if flags & flag != 0 {
            let end = b.get(at..).and_then(|r| r.iter().position(|&c| c == 0)).ok_or_else(short)?;
            at += end + 1;
        }
    }
    if flags & FHCRC != 0 {
        at += 2;
    }
    let body = b.get(at..b.len() - 8).ok_or_else(short)?;
    let t = &b[b.len() - 8..];
    Ok((body, u32::from_le_bytes([t[0], t[1], t[2], t[3]]), u32::from_le_bytes([t[4], t[5], t[6], t[7]])))
}

/// Unpacks `bytes` to exactly `size` bytes. A gzip trailer whose CRC-32 or
/// length does not match what was unpacked is refused.
pub fn unpack(compression: Compression, bytes: &[u8], size: u64) -> Result<Vec<u8>, String> {
    let size = usize::try_from(size).map_err(|_| format!("an unpacked size of {size} bytes"))?;
    match compression {
        Compression::Gzip => {
            let (deflate, crc, isize) = gzip_parts(bytes)?;
            let out = turbo_maxent::inflate::inflate(deflate, size).map_err(|e| e.to_string())?;
            if turbo_maxent::crc32(&out) != crc || out.len() as u32 != isize {
                return Err("the gzip trailer does not match the unpacked bytes".into());
            }
            Ok(out)
        }
        Compression::Zstd => turbo_zstd::decompress(bytes, size).map_err(|e| e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HELLO: &[u8] = b"hello hello hello\n";

    /// HELLO as Python's `gzip.compress(..., mtime=0)` writes it.
    const HELLO_GZ: [u8; 29] = [
        0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x57, 0xc8, 0x40,
        0x90, 0x5c, 0x00, 0x3b, 0x7c, 0x8a, 0xdf, 0x12, 0x00, 0x00, 0x00,
    ];

    /// HELLO as one zstd frame of one raw block: the magic, a single
    /// segment header with a one-byte content size, and a last-block
    /// header of type raw.
    fn hello_zst() -> Vec<u8> {
        let n = HELLO.len() as u32;
        let block = (n << 3) | 1;
        let mut out = vec![0x28, 0xb5, 0x2f, 0xfd, 0x20, n as u8];
        out.extend_from_slice(&block.to_le_bytes()[..3]);
        out.extend_from_slice(HELLO);
        out
    }

    #[test]
    fn both_compressions_unpack_to_the_stated_size() {
        assert_eq!(unpack(Compression::Gzip, &HELLO_GZ, HELLO.len() as u64).unwrap(), HELLO);
        assert_eq!(unpack(Compression::Zstd, &hello_zst(), HELLO.len() as u64).unwrap(), HELLO);
        assert!(unpack(Compression::Gzip, &HELLO_GZ, 5).is_err());
        assert!(unpack(Compression::Zstd, &hello_zst(), 5).is_err());
        assert!(unpack(Compression::Zstd, &HELLO_GZ, HELLO.len() as u64).is_err());
    }

    #[test]
    fn a_gzip_trailer_must_match() {
        let mut bad = HELLO_GZ;
        bad[21] ^= 1;
        assert!(unpack(Compression::Gzip, &bad, HELLO.len() as u64).is_err(), "CRC-32");
        let mut bad = HELLO_GZ;
        bad[25] = 0x13;
        assert!(unpack(Compression::Gzip, &bad, HELLO.len() as u64).is_err(), "length");
    }

    #[test]
    fn a_header_that_is_not_gzip_is_refused() {
        assert!(gzip_parts(b"PK\x03\x04 not gzip at all").is_err());
        let mut bad = HELLO_GZ;
        bad[2] = 7;
        assert!(gzip_parts(&bad).is_err());
        bad[2] = 8;
        bad[3] = 0x20;
        assert!(gzip_parts(&bad).is_err());
        bad[3] = FNAME;
        assert!(gzip_parts(&bad[..12]).is_err());
    }
}
