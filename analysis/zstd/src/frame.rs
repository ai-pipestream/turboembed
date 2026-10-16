//! Frames: a Zstandard frame's header, its blocks and its content
//! checksum, and skippable frames, which are stepped over.

use crate::Error;
use crate::block::{self, State};
use crate::error::Part;
use crate::output::Output;
use crate::sequences::History;
use crate::xxhash::Xxh64;

pub const MAGIC: u32 = 0xFD2F_B528;

/// Skippable frames have the magic numbers 0x184D2A50 to 0x184D2A5F.
pub const SKIPPABLE: u32 = 0x184D_2A50;
pub const SKIPPABLE_MASK: u32 = 0xFFFF_FFF0;

/// The largest window a frame header may ask for, the reference
/// decoder's default limit. In single-segment mode the window is the
/// content size, which the size the caller asks for bounds instead.
pub const MAX_WINDOW: u64 = 1 << 27;

/// The most any block holds.
const MAX_BLOCK: u64 = 128 << 10;

#[derive(Debug, PartialEq, Eq)]
struct Header {
    window: u64,
    content: Option<u64>,
    checksum: bool,
    /// The header's length in bytes.
    len: usize,
}

/// A little-endian number of up to 8 bytes.
fn le(b: &[u8]) -> u64 {
    b.iter().rev().fold(0, |v, &x| v << 8 | u64::from(x))
}

/// Reads the frame header at the start of `src`, just after the magic
/// number.
fn read_header(src: &[u8]) -> Result<Header, Error> {
    let fhd = *src.first().ok_or(Error::Truncated(Part::FrameHeader))?;
    let single = fhd & 0x20 != 0;
    if fhd & 0x08 != 0 {
        return Err(Error::ReservedBit);
    }
    let id_len = [0, 1, 2, 4][usize::from(fhd & 3)];
    // With no content size flag the size is there only in single-segment
    // mode, in one byte.
    let content_len = [usize::from(single), 2, 4, 8][usize::from(fhd >> 6)];
    let len = 1 + usize::from(!single) + id_len + content_len;
    let b = src.get(..len).ok_or(Error::Truncated(Part::FrameHeader))?;
    let mut at = 1;
    let mut window = 0;
    if !single {
        let exponent = u32::from(b[1] >> 3);
        let base = 1u64 << (10 + exponent);
        window = base + base / 8 * u64::from(b[1] & 7);
        at = 2;
    }
    let id = le(&b[at..at + id_len]) as u32;
    if id != 0 {
        return Err(Error::Dictionary(id));
    }
    at += id_len;
    let content = match content_len {
        0 => None,
        2 => Some(le(&b[at..at + 2]) + 256),
        _ => Some(le(&b[at..at + content_len])),
    };
    if let Some(c) = content.filter(|_| single) {
        window = c;
    } else if window > MAX_WINDOW {
        return Err(Error::WindowTooLarge { window, limit: MAX_WINDOW });
    }
    Ok(Header { window, content, checksum: fhd & 0x04 != 0, len })
}

/// What a frame's headers say about it, read without decoding it.
pub struct Extent {
    /// The bytes the frame takes, magic number included.
    pub len: usize,
    /// The most the frame can decode to: what its blocks can hold, or its
    /// content size if the header gives a smaller one.
    pub max: u64,
    /// The content size the header gives, if it does.
    pub content: Option<u64>,
}

/// Steps over the Zstandard frame at the start of `src`, magic number
/// included, reading its header and each block's 3-byte header but not
/// the blocks. A raw or RLE block holds its size, a compressed one at
/// most the frame's block maximum.
pub fn measure(src: &[u8]) -> Result<Extent, Error> {
    let h = read_header(&src[4..])?;
    let block_max = h.window.min(MAX_BLOCK) as usize;
    let mut at = 4 + h.len;
    let mut max = 0u64;
    loop {
        let b = block::Header::read(&src[at..], block_max)?;
        at += 3;
        if src.len() - at < b.body_len() {
            return Err(Error::Truncated(Part::Block));
        }
        at += b.body_len();
        max += if b.kind == block::Kind::Compressed { block_max } else { b.size } as u64;
        if b.last {
            break;
        }
    }
    if h.checksum {
        if src.len() - at < 4 {
            return Err(Error::Truncated(Part::Checksum));
        }
        at += 4;
    }
    Ok(Extent { len: at, max: h.content.map_or(max, |c| c.min(max)), content: h.content })
}

/// Decodes the Zstandard frame at the start of `src`, magic number
/// included, into `out`. Returns the bytes it takes.
pub fn decode(src: &[u8], state: &mut State, out: &mut Output) -> Result<usize, Error> {
    let h = read_header(&src[4..])?;
    let start = out.pos;
    let want = out.buf.len();
    let end = match h.content {
        None => want,
        Some(c) if c <= (want - start) as u64 => start + c as usize,
        Some(_) => return Err(Error::TooLong { want }),
    };
    let frame = block::Frame {
        history: History { start, window: h.window },
        block_max: h.window.min(MAX_BLOCK) as usize,
        end,
        content: h.content,
    };
    state.reset();
    // Each block's output is hashed just after it is decoded, while it is
    // still in cache.
    let mut hash = h.checksum.then(|| Xxh64::new(0));
    let mut at = 4 + h.len;
    loop {
        let from = out.pos;
        let (n, last) = block::decode(&src[at..], &frame, state, out)?;
        if let Some(hash) = &mut hash {
            hash.update(&out.buf[from..out.pos]);
        }
        at += n;
        if last {
            break;
        }
    }
    let got = (out.pos - start) as u64;
    if let Some(header) = h.content.filter(|&c| c != got) {
        return Err(Error::ContentTooShort { header, got });
    }
    if let Some(hash) = hash {
        let stored = src.get(at..at + 4).ok_or(Error::Truncated(Part::Checksum))?;
        let stored = le(stored) as u32;
        let computed = hash.finish() as u32;
        if stored != computed {
            return Err(Error::Checksum { stored, computed });
        }
        at += 4;
    }
    Ok(at)
}

/// The length of the skippable frame at the start of `src`, magic number
/// included.
pub fn skip(src: &[u8]) -> Result<usize, Error> {
    let size = src.get(4..8).ok_or(Error::Truncated(Part::SkippableFrame))?;
    let size = le(size) as usize;
    if src.len() - 8 < size {
        return Err(Error::Truncated(Part::SkippableFrame));
    }
    Ok(8 + size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_size_widths() {
        // Single segment, no flag: one byte, and the window is the size.
        let h = read_header(&[0x20, 200]).unwrap();
        assert_eq!(h, Header { window: 200, content: Some(200), checksum: false, len: 2 });
        // Flag 1: two bytes, plus 256.
        let h = read_header(&[0x60, 0xff, 0xff]).unwrap();
        assert_eq!((h.content, h.len), (Some(65791), 3));
        // Flag 2: four bytes; flag 3: eight.
        let h = read_header(&[0xa0, 1, 2, 3, 4]).unwrap();
        assert_eq!((h.content, h.len), (Some(0x0403_0201), 5));
        let h = read_header(&[0xe0, 1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        assert_eq!((h.content, h.len), (Some(0x0807_0605_0403_0201), 9));
        // No flag and not single segment: no size, a window byte.
        let h = read_header(&[0x00, 0x00]).unwrap();
        assert_eq!(h, Header { window: 1024, content: None, checksum: false, len: 2 });
        // The checksum flag; the unused bit (4) is not read.
        assert!(read_header(&[0x04, 0]).unwrap().checksum);
        assert_eq!(read_header(&[0x10, 0]).unwrap(), read_header(&[0x00, 0]).unwrap());
    }

    #[test]
    fn window_descriptor() {
        // Exponent 7, mantissa 3: 2^17 + 3 * 2^14.
        assert_eq!(read_header(&[0x00, 7 << 3 | 3]).unwrap().window, (1 << 17) + 3 * (1 << 14));
        // 2^27 is the limit: exponent 17 with mantissa 0 is accepted, with
        // mantissa 1 refused.
        assert_eq!(read_header(&[0x00, 17 << 3]).unwrap().window, 1 << 27);
        assert_eq!(
            read_header(&[0x00, 17 << 3 | 1]).err(),
            Some(Error::WindowTooLarge { window: (1 << 27) + (1 << 24), limit: 1 << 27 })
        );
        // The largest descriptor.
        assert_eq!(
            read_header(&[0x00, 0xff]).err(),
            Some(Error::WindowTooLarge { window: (1 << 41) + 7 * (1 << 38), limit: 1 << 27 })
        );
    }

    #[test]
    fn dictionary_ids_and_reserved_bit() {
        // One, two and four byte ids; an id of 0 needs no dictionary.
        assert_eq!(read_header(&[0x01, 0, 7]).err(), Some(Error::Dictionary(7)));
        assert_eq!(read_header(&[0x02, 0, 1, 2]).err(), Some(Error::Dictionary(0x0201)));
        assert_eq!(read_header(&[0x23, 1, 2, 3, 4, 9]).err(), Some(Error::Dictionary(0x0403_0201)));
        assert_eq!(read_header(&[0x23, 0, 0, 0, 0, 9]).unwrap().len, 6);
        assert_eq!(read_header(&[0x08, 0]).err(), Some(Error::ReservedBit));
        assert_eq!(read_header(&[0xe0, 1, 2]).err(), Some(Error::Truncated(Part::FrameHeader)));
    }
}
