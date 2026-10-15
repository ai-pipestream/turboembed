//! Snappy decompression of the raw format (not the framed one), which is
//! what Parquet's SNAPPY codec stores in a page: a varint of the
//! uncompressed length, then literals and back references (Snappy's
//! `format_description.txt`).

use crate::Result;

/// Decompresses `src`, which must expand to exactly `want` bytes.
pub fn decompress(src: &[u8], want: usize) -> Result<Vec<u8>> {
    let mut pos = 0;
    let mut len: u64 = 0;
    for shift in (0..35).step_by(7) {
        let b = *src.get(pos).ok_or("snappy: no length")?;
        pos += 1;
        len |= u64::from(b & 0x7f) << shift;
        if b & 0x80 == 0 {
            break;
        }
        if shift == 28 {
            return Err("snappy: length longer than 5 bytes".into());
        }
    }
    if len != want as u64 {
        return Err(format!("snappy: the stream holds {len} bytes, the page header says {want}"));
    }
    let mut out = Vec::with_capacity(want);
    while pos < src.len() {
        let tag = src[pos];
        pos += 1;
        let (n, offset) = match tag & 3 {
            0 => {
                let mut n = usize::from(tag >> 2);
                if n >= 60 {
                    let extra = n - 59;
                    let b = src.get(pos..pos + extra).ok_or("snappy: ends inside a literal's length")?;
                    n = b.iter().rev().fold(0usize, |a, &x| a << 8 | usize::from(x));
                    pos += extra;
                }
                let n = n + 1;
                let lit = src.get(pos..pos + n).ok_or("snappy: ends inside a literal")?;
                if out.len() + n > want {
                    return Err("snappy: expands past its length".into());
                }
                out.extend_from_slice(lit);
                pos += n;
                continue;
            }
            1 => {
                let b = *src.get(pos).ok_or("snappy: ends inside a copy")?;
                pos += 1;
                (4 + usize::from((tag >> 2) & 7), usize::from(tag >> 5) << 8 | usize::from(b))
            }
            2 => {
                let b = src.get(pos..pos + 2).ok_or("snappy: ends inside a copy")?;
                pos += 2;
                (1 + usize::from(tag >> 2), usize::from(u16::from_le_bytes([b[0], b[1]])))
            }
            _ => {
                let b = src.get(pos..pos + 4).ok_or("snappy: ends inside a copy")?;
                pos += 4;
                (1 + usize::from(tag >> 2), u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
            }
        };
        if offset == 0 || offset > out.len() {
            return Err(format!("snappy: a copy from {offset} bytes back at byte {} of the output", out.len()));
        }
        if out.len() + n > want {
            return Err("snappy: expands past its length".into());
        }
        let from = out.len() - offset;
        if offset >= n {
            out.extend_from_within(from..from + n);
        } else {
            // The copy overlaps what it writes, so it repeats the last
            // `offset` bytes.
            for i in 0..n {
                out.push(out[from + i]);
            }
        }
    }
    if out.len() != want {
        return Err(format!("snappy: expands to {} bytes, not {want}", out.len()));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_and_copies() {
        // "abcd" as a literal, then a 1-byte-offset copy of 8 from 4 back
        // (overlapping), a 2-byte-offset copy of 3 from 12 back, and a
        // 4-byte-offset copy of 2 from 1 back.
        let src =
            [17, 0x0c, b'a', b'b', b'c', b'd', 0x01 | (4 << 2), 4, 0x02 | (2 << 2), 12, 0, 0x03 | (1 << 2), 1, 0, 0, 0];
        assert_eq!(decompress(&src, 17).unwrap(), b"abcdabcdabcdabccc");
    }

    #[test]
    fn long_literal_length() {
        // A literal of 100 bytes takes tag 60 and one length byte, 99.
        let mut src = vec![100, 60 << 2, 99];
        src.extend(std::iter::repeat_n(b'x', 100));
        assert_eq!(decompress(&src, 100).unwrap(), vec![b'x'; 100]);
    }

    #[test]
    fn damage_is_an_error() {
        // The length disagrees with the page.
        assert!(decompress(&[3, 0x08, b'a', b'b', b'c'], 4).is_err());
        // A copy from before the start.
        let e = decompress(&[8, 0x00, b'a', 0x01, 2], 8).unwrap_err();
        assert!(e.contains("copy from 2 bytes back"), "{e}");
        // A literal past the end of the input.
        assert!(decompress(&[4, 0x0c, b'a'], 4).is_err());
        // Too short.
        assert!(decompress(&[4, 0x04, b'a', b'b'], 4).is_err());
    }
}
