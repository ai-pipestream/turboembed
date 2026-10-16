//! A Zstandard decoder, as RFC 8878 defines the format (and the zstd
//! project's `doc/zstd_compression_format.md`, which it follows): one or
//! more frames, skippable frames stepped over, raw, RLE and compressed
//! blocks, Huffman coded literals in one or four streams, FSE coded
//! sequences in every table mode, and the content checksum.
//!
//! Dictionaries are not supported: a frame that names one is refused.
//! A frame header may ask for a window of at most 2^27 bytes (128 MiB),
//! the reference decoder's default limit. The output is decoded into one
//! buffer of exactly the size the caller asks for, allocated once, after
//! the frame and block headers have been read and only if the frames can
//! come to that size; a frame that would write past it is refused, as is
//! any other input that breaks a rule of the format the decoder checks.
//!
//! The decoder refuses some inputs zstd 1.5.5 decodes, where it checks a
//! rule of RFC 8878 that zstd does not:
//!
//! - A sequences bitstream must end exactly where the last sequence's
//!   bits do. zstd also reads the state updates after the last sequence,
//!   which the RFC leaves out, and refuses only bits left over after
//!   them, so it takes a stream that runs short and one with bits to
//!   spare.
//! - A Huffman table may be at most 11 bits deep; zstd takes 12.
//! - An FSE description of Huffman weights may not give a probability to
//!   a symbol above 11, the largest weight. zstd refuses only weights
//!   above 12, and only those it decodes.
//! - The reserved bits of the sequence compression modes must be 0; zstd
//!   does not read them.
//! - A match may not reach further back than the frame's window; zstd
//!   checks only that it stays within the content decoded so far.
//! - A block, its literals and its output may each be at most the
//!   frame's block maximum, the window or 128 KiB, whichever is smaller.
//!   zstd's one-pass decoder, `ZSTD_decompress`, does not check a raw or
//!   RLE block's size, and checks a compressed block and its literals
//!   only against 128 KiB.
//! - A compressed block of size 0 is refused, as it has no literals
//!   section; zstd's streaming decoder takes it as an empty block.
//! - A frame whose content size differs from its output is refused even
//!   when its last block is empty, a case zstd's streaming decoder does
//!   not check.
//!
//! A sequence count of 0 written in two bytes does not end the sequences
//! section as the 1-byte 0 does: as the RFC lays the section out, the
//! compression modes and the tables they call for follow, and they
//! replace the frame's tables. The RFC does not say what the bitstream
//! of no sequences holds; the bytes after the tables are not read, as
//! zstd 1.5.5 does not read them.

mod bits;
mod block;
mod error;
mod frame;
mod fse;
mod huffman;
mod literals;
mod output;
mod sequences;
mod xxhash;

pub use error::{Error, Part, Symbol};

use output::Output;

/// The two kinds of frame, told apart by the magic number.
enum Kind {
    Zstd,
    Skippable,
}

/// The kind of the frame at the start of `src`.
fn kind(src: &[u8]) -> Result<Kind, Error> {
    let magic = src.first_chunk::<4>().ok_or(Error::Truncated(Part::Magic))?;
    match u32::from_le_bytes(*magic) {
        frame::MAGIC => Ok(Kind::Zstd),
        m if m & frame::SKIPPABLE_MASK == frame::SKIPPABLE => Ok(Kind::Skippable),
        m => Err(Error::BadMagic(m)),
    }
}

/// Decodes every frame in `src`, which must come to exactly `want` bytes.
///
/// The frame headers and the 3-byte block headers are read first, the
/// blocks stepped over, and the output is allocated only once `want` is
/// known to be within what the frames can decode to. A frame can decode
/// to what its blocks hold, a raw or RLE block its size and a compressed
/// block the frame's block maximum (its window or 128 KiB, whichever is
/// smaller), or to its content size if the header gives a smaller one. A
/// larger `want` is refused, and so is one that differs from the sum of
/// the content sizes when every frame gives one. An input whose headers
/// break a rule, or that ends inside a frame, is refused then too.
pub fn decompress(src: &[u8], want: usize) -> Result<Vec<u8>, Error> {
    if src.is_empty() {
        return Err(Error::NoFrames);
    }
    let mut max = 0u64;
    // The sum of the content sizes, while every frame gives one.
    let mut total = Some(0u64);
    let mut at = 0;
    while at < src.len() {
        at += match kind(&src[at..])? {
            Kind::Zstd => {
                let e = frame::measure(&src[at..])?;
                max = max.saturating_add(e.max);
                total = total.zip(e.content).map(|(t, c)| t.saturating_add(c));
                e.len
            }
            Kind::Skippable => frame::skip(&src[at..])?,
        };
    }
    match total {
        Some(t) if t < want as u64 => return Err(Error::TooShort { got: t as usize, want }),
        Some(t) if t > want as u64 => return Err(Error::TooLong { want }),
        _ => {}
    }
    if want as u64 > max {
        return Err(Error::CannotExpand { max, want });
    }
    let mut buf = vec![0u8; want];
    let mut out = Output { buf: &mut buf, pos: 0 };
    let mut state = block::State::new();
    let mut at = 0;
    while at < src.len() {
        at += match kind(&src[at..])? {
            Kind::Zstd => frame::decode(&src[at..], &mut state, &mut out)?,
            Kind::Skippable => frame::skip(&src[at..])?,
        };
    }
    if out.pos != want {
        return Err(Error::TooShort { got: out.pos, want });
    }
    Ok(buf)
}
