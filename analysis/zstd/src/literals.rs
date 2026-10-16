//! The literals section of a compressed block: its header, and the
//! literals raw, as one repeated byte, or Huffman coded with a new table
//! or the one before (treeless).

use crate::Error;
use crate::error::Part;
use crate::huffman::Table;

/// Room past the literals in the buffer, so short copies out of it can
/// move 16 bytes at once.
const SLACK: usize = 16;

/// A block's literals: the first `len` bytes of `bytes`, which may go on
/// past them.
pub struct Literals<'a> {
    pub bytes: &'a [u8],
    pub len: usize,
}

/// Reads the literals section at the start of `block`, for a block of at
/// most `max` bytes. Raw literals are left in `block`; the others are
/// decoded into `buf`. `table` is the Huffman table of the last
/// compressed literals section, which a new one replaces. Returns the
/// literals and the length of the section.
pub fn read<'a>(
    block: &'a [u8],
    max: usize,
    table: &mut Table,
    buf: &'a mut Vec<u8>,
) -> Result<(Literals<'a>, usize), Error> {
    let header = |n: usize| block.get(..n).ok_or(Error::Truncated(Part::LiteralsHeader));
    let b0 = *block.first().ok_or(Error::Truncated(Part::LiteralsHeader))?;
    let kind = b0 & 3;
    let format = (b0 >> 2) & 3;
    if kind < 2 {
        // Raw or RLE: the size in 5, 12 or 20 bits.
        let (len, at) = match format {
            0 | 2 => (usize::from(b0 >> 3), 1),
            1 => {
                let h = header(2)?;
                (usize::from(h[0] >> 4) | usize::from(h[1]) << 4, 2)
            }
            _ => {
                let h = header(3)?;
                (usize::from(h[0] >> 4) | usize::from(h[1]) << 4 | usize::from(h[2]) << 12, 3)
            }
        };
        if len > max {
            return Err(Error::TooManyLiterals { len, max });
        }
        if kind == 0 {
            if block.len() - at < len {
                return Err(Error::Truncated(Part::Literals));
            }
            return Ok((Literals { bytes: &block[at..], len }, at + len));
        }
        let &byte = block.get(at).ok_or(Error::Truncated(Part::Literals))?;
        let out = room(buf, len);
        out[..len].fill(byte);
        return Ok((Literals { bytes: out, len }, at + 1));
    }
    // Huffman coded: the regenerated and compressed sizes in 10, 14 or
    // 18 bits each; one stream only in the shortest form.
    let (at, width, four) = match format {
        0 => (3, 10, false),
        1 => (3, 10, true),
        2 => (4, 14, true),
        _ => (5, 18, true),
    };
    let h = header(at)?.iter().rev().fold(0u64, |h, &b| h << 8 | u64::from(b));
    let mask = (1u64 << width) - 1;
    let len = (h >> 4 & mask) as usize;
    let size = (h >> (4 + width) & mask) as usize;
    if len > max {
        return Err(Error::TooManyLiterals { len, max });
    }
    let data = block.get(at..at + size).ok_or(Error::Truncated(Part::Literals))?;
    let streams = if kind == 2 {
        &data[table.read(data)?..]
    } else if table.is_empty() {
        return Err(Error::NoHuffmanTable);
    } else {
        data
    };
    let out = room(buf, len);
    if four {
        table.decode_four(streams, &mut out[..len])?;
    } else {
        table.decode_one(streams, &mut out[..len])?;
    }
    Ok((Literals { bytes: out, len }, at + size))
}

/// `buf`, grown to hold `len` literals and the slack after them. It only
/// grows, so the zeros it is filled with are written once.
fn room(buf: &mut Vec<u8>, len: usize) -> &mut [u8] {
    if buf.len() < len + SLACK {
        buf.resize(len + SLACK, 0);
    }
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lits(block: &[u8], max: usize) -> Result<(Vec<u8>, usize), Error> {
        let mut table = Table::default();
        let mut buf = Vec::new();
        let (l, n) = read(block, max, &mut table, &mut buf)?;
        Ok((l.bytes[..l.len].to_vec(), n))
    }

    #[test]
    fn raw_and_rle_sizes_in_each_header_form() {
        // Raw, 1-byte header: the size in the top 5 bits, so the format's
        // second bit is the size's lowest and formats 0 and 2 are one.
        assert_eq!(lits(&[2 << 3, b'a', b'b', 9], 100).unwrap(), (b"ab".to_vec(), 3));
        assert_eq!(lits(&[3 << 3, b'a', b'b', b'c', 9], 100).unwrap(), (b"abc".to_vec(), 4));
        // Raw, 2-byte header (format 1): 20 = 4 + (1 << 4).
        let mut b = vec![0b0100 | (4 << 4), 1];
        b.extend(0..20);
        assert_eq!(lits(&b, 100).unwrap(), ((0..20).collect(), 22));
        // RLE, 3-byte header (format 3): 4097 = 1 + (0 << 4) + (1 << 12).
        let (v, n) = lits(&[0b1101 | (1 << 4), 0, 1, b'z'], 1 << 17).unwrap();
        assert_eq!((v.len(), n), (4097, 4));
        assert!(v.iter().all(|&x| x == b'z'));
        // A size past the block's maximum, and literals past its end.
        assert_eq!(lits(&[0b1101 | (1 << 4), 0, 1, b'z'], 4096), Err(Error::TooManyLiterals { len: 4097, max: 4096 }));
        assert_eq!(lits(&[3 << 3, b'a'], 100), Err(Error::Truncated(Part::Literals)));
        assert_eq!(lits(&[0b0100 | (4 << 4)], 100), Err(Error::Truncated(Part::LiteralsHeader)));
        assert_eq!(lits(&[1 << 3], 100), Err(Error::Truncated(Part::Literals)));
    }

    #[test]
    fn huffman_header_forms() {
        // Weights 1, 1 (direct, header 129 and one byte) and the last
        // symbol, 2, at weight 2: symbols 0 and 1 have the codes 00 and
        // 01, symbol 2 the code 1. Format 0, one stream: regenerated 3,
        // compressed 3 (the two table bytes and a stream of "2, 0, 1",
        // which is 1, 00, 01 under the mark).
        let table = [129, 0x11];
        let stream = 0b0011_0001;
        let mut b = Vec::new();
        let h: u32 = 0b10 | (3 << 4) | (3 << 14);
        b.extend(&h.to_le_bytes()[..3]);
        b.extend(table);
        b.push(stream);
        assert_eq!(lits(&b, 100).unwrap(), (vec![2, 0, 1], 6));
        // More literals than the block's maximum.
        assert_eq!(lits(&b, 2), Err(Error::TooManyLiterals { len: 3, max: 2 }));
        // A treeless section with no table before it.
        b[0] |= 1;
        assert_eq!(lits(&b, 100), Err(Error::NoHuffmanTable));
        // Compressed size past the block.
        b[0] &= !1;
        assert_eq!(lits(&b[..5], 100), Err(Error::Truncated(Part::Literals)));
    }

    #[test]
    fn treeless_reuses_the_last_table() {
        let mut table = Table::default();
        let mut buf = Vec::new();
        let h: u32 = 0b10 | (3 << 4) | (3 << 14);
        let mut b = h.to_le_bytes()[..3].to_vec();
        b.extend([129, 0x11, 0b0011_0001]);
        assert_eq!(read(&b, 100, &mut table, &mut buf).unwrap().1, 6);
        // Treeless, regenerated 2, compressed 1: "0, 2" = 00, 1.
        let h: u32 = 0b11 | (2 << 4) | (1 << 14);
        let mut t = h.to_le_bytes()[..3].to_vec();
        t.push(0b0000_1001);
        let (l, n) = read(&t, 100, &mut table, &mut buf).unwrap();
        assert_eq!((&l.bytes[..l.len], n), (&[0, 2][..], 4));
    }
}
