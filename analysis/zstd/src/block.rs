//! Blocks: the 3-byte header, and raw, RLE and compressed blocks, with
//! what a frame's compressed blocks pass on to the next one.

use crate::Error;
use crate::error::Part;
use crate::huffman;
use crate::literals;
use crate::output::{Limit, Output};
use crate::sequences::{self, History, Tables};

/// What carries from one compressed block to the next in a frame: the
/// last Huffman table, the three sequence tables and the repeated
/// offsets. The literals buffer is only reused, so it is not reset.
pub struct State {
    huffman: huffman::Table,
    tables: Tables,
    reps: [usize; 3],
    lits: Vec<u8>,
}

impl State {
    pub fn new() -> State {
        State { huffman: huffman::Table::default(), tables: Tables::new(), reps: [1, 4, 8], lits: Vec::new() }
    }

    /// The state a frame starts in: no tables, and repeated offsets 1, 4
    /// and 8.
    pub fn reset(&mut self) {
        self.huffman.clear();
        self.tables.clear();
        self.reps = [1, 4, 8];
    }
}

/// What a frame's blocks need from its header.
pub struct Frame {
    pub history: History,
    /// The most a block holds, compressed or not: the window or 128 KiB,
    /// whichever is smaller.
    pub block_max: usize,
    /// Where the frame's content must end in the output.
    pub end: usize,
    /// The content size the header gives, if it does.
    pub content: Option<u64>,
}

/// The three block types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Raw,
    Rle,
    Compressed,
}

/// A block's 3-byte header.
pub struct Header {
    pub kind: Kind,
    /// For an RLE block the bytes it stands for; for the others the bytes
    /// it takes, which bound what a raw block stands for.
    pub size: usize,
    pub last: bool,
}

impl Header {
    /// Reads the block header at the start of `src`, in a frame whose
    /// blocks hold at most `block_max` bytes.
    pub fn read(src: &[u8], block_max: usize) -> Result<Header, Error> {
        let h = src.get(..3).ok_or(Error::Truncated(Part::BlockHeader))?;
        let h = u32::from(h[0]) | u32::from(h[1]) << 8 | u32::from(h[2]) << 16;
        let kind = match h >> 1 & 3 {
            0 => Kind::Raw,
            1 => Kind::Rle,
            2 => Kind::Compressed,
            _ => return Err(Error::ReservedBlockType),
        };
        let size = (h >> 3) as usize;
        if size > block_max {
            return Err(Error::BlockTooLarge { size, max: block_max });
        }
        Ok(Header { kind, size, last: h & 1 == 1 })
    }

    /// The bytes the block takes after its header: an RLE block's one
    /// byte, or its size.
    pub fn body_len(&self) -> usize {
        if self.kind == Kind::Rle { 1 } else { self.size }
    }
}

/// Decodes the block at the start of `src` into `out`. Returns the bytes
/// it takes in `src` and whether it is the frame's last.
pub fn decode(src: &[u8], frame: &Frame, state: &mut State, out: &mut Output) -> Result<(usize, bool), Error> {
    let Header { kind, size, last } = Header::read(src, frame.block_max)?;
    let limit = Limit::new(out.pos, frame.block_max, frame.end, frame.content, out.buf.len());
    let body = &src[3..];
    match kind {
        Kind::Raw => {
            let data = body.get(..size).ok_or(Error::Truncated(Part::Block))?;
            if out.pos + size > limit.end {
                return Err(limit.error(out.pos + size));
            }
            out.extend(data);
            Ok((3 + size, last))
        }
        Kind::Rle => {
            let &byte = body.first().ok_or(Error::Truncated(Part::Block))?;
            if out.pos + size > limit.end {
                return Err(limit.error(out.pos + size));
            }
            out.fill(byte, size);
            Ok((4, last))
        }
        Kind::Compressed => {
            let data = body.get(..size).ok_or(Error::Truncated(Part::Block))?;
            let (lits, n) = literals::read(data, frame.block_max, &mut state.huffman, &mut state.lits)?;
            sequences::execute(&data[n..], &mut state.tables, &mut state.reps, &lits, &frame.history, &limit, out)?;
            Ok((3 + size, last))
        }
    }
}
