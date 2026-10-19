//! RFC 1951 inflate: stored, fixed Huffman and dynamic Huffman blocks,
//! decoded into a buffer whose final size is known in advance (a zip
//! directory record states it).
//!
//! Huffman codes are decoded through a table indexed by the next
//! `TABLE_BITS` bits of input, which resolves every code of up to that
//! length in one lookup. Longer codes, which are rare, fall back to the
//! canonical decoding loop of RFC 1951 section 3.2.2, reading one bit at
//! a time. Nothing is allocated per symbol: the output grows in a vector
//! that starts with a capacity bounded by the input's length, and the
//! tables live in one struct per call.

use crate::error::{Error, Result};

/// Bits resolved by one table lookup. Literal and length codes in real
/// streams are almost all 10 bits or shorter.
const TABLE_BITS: u32 = 10;
const TABLE_SIZE: usize = 1 << TABLE_BITS;
const TABLE_MASK: u64 = TABLE_SIZE as u64 - 1;
/// The longest code RFC 1951 allows.
const MAX_BITS: usize = 15;

/// The most output one input byte can produce: a length code of 258
/// bytes and a distance code can be one bit each in a dynamic block, so
/// two bits give 258 bytes.
const MAX_RATIO: usize = 1032;

/// The largest stated size `inflate` accepts, 1 GiB. The largest OpenNLP
/// models are a few hundred megabytes uncompressed; a stated size above
/// this is refused before anything is allocated.
pub const MAX_SIZE: usize = 1 << 30;

/// The output's initial capacity is at most this many times the input's
/// length, so a stated size the stream does not back with data costs no
/// more than this before decoding fails. Real model entries compress by
/// less than this, so they are not reallocated.
const INITIAL_RATIO: usize = 64;

const LENGTH_BASE: [u16; 29] =
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LENGTH_EXTRA: [u8; 29] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DISTANCE_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145,
    8193, 12289, 16385, 24577,
];
const DISTANCE_EXTRA: [u8; 30] =
    [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];

/// The order in which a dynamic block lists the code length code's
/// lengths (RFC 1951 section 3.2.7).
const CODE_LENGTH_ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

/// Inflates a raw deflate stream (no zlib or gzip header) that must
/// produce exactly `size` bytes. Bytes after the final block are ignored.
pub fn inflate(input: &[u8], size: usize) -> Result<Vec<u8>> {
    if size > MAX_SIZE {
        return Err(Error::Inflate("the stated size is above the 1 GiB limit"));
    }
    if size > input.len().saturating_mul(MAX_RATIO).saturating_add(MAX_RATIO) {
        return Err(Error::Inflate("the stated size is more than the input can produce"));
    }
    let mut out = Vec::new();
    let initial = size.min(input.len().saturating_mul(INITIAL_RATIO));
    if out.try_reserve_exact(initial).is_err() {
        return Err(Error::Inflate("the output buffer could not be allocated"));
    }
    let mut inflater = Inflater {
        bits: Bits { data: input, pos: 0, buf: 0, count: 0 },
        out,
        size,
        literals: Huffman::new(),
        distances: Huffman::new(),
    };
    inflater.run()?;
    Ok(inflater.out)
}

mod bits;
mod huffman;
use bits::Bits;
use huffman::{CodeKind, Huffman};
struct Inflater<'a> {
    bits: Bits<'a>,
    /// The bytes produced so far.
    out: Vec<u8>,
    /// The size the stream must produce.
    size: usize,
    literals: Box<Huffman>,
    distances: Box<Huffman>,
}

impl Inflater<'_> {
    fn run(&mut self) -> Result<()> {
        loop {
            let last = self.bits.take(1)? == 1;
            match self.bits.take(2)? {
                0 => self.stored()?,
                1 => {
                    self.fixed_codes()?;
                    self.codes()?;
                }
                2 => {
                    self.dynamic_codes()?;
                    self.codes()?;
                }
                _ => return Err(Error::Inflate("a block has the reserved type 3")),
            }
            if last {
                break;
            }
        }
        if self.out.len() != self.size {
            return Err(Error::Inflate("the stream ends before the stated size"));
        }
        Ok(())
    }

    fn stored(&mut self) -> Result<()> {
        let start = self.bits.align_to_byte();
        let data = self.bits.data;
        let Some(header) = data.get(start..start + 4) else {
            return Err(Error::Inflate("the input ends inside a stored block header"));
        };
        let len = usize::from(u16::from_le_bytes([header[0], header[1]]));
        let check = u16::from_le_bytes([header[2], header[3]]);
        if check != !(len as u16) {
            return Err(Error::Inflate("a stored block's length and its complement disagree"));
        }
        let Some(bytes) = data.get(start + 4..start + 4 + len) else {
            return Err(Error::Inflate("the input ends inside a stored block"));
        };
        if len > self.size - self.out.len() {
            return Err(Error::Inflate("the stream produces more than the stated size"));
        }
        self.out.extend_from_slice(bytes);
        self.bits.pos = start + 4 + len;
        Ok(())
    }

    fn fixed_codes(&mut self) -> Result<()> {
        let mut lengths = [0u8; 288];
        lengths[..144].fill(8);
        lengths[144..256].fill(9);
        lengths[256..280].fill(7);
        lengths[280..].fill(8);
        self.literals.build(&lengths, CodeKind::LiteralsOrDistances)?;
        // Thirty-two five-bit codes, so the code is complete; symbols 30
        // and 31 are refused when they appear.
        self.distances.build(&[5; 32], CodeKind::LiteralsOrDistances)
    }

    fn dynamic_codes(&mut self) -> Result<()> {
        let literal_count = self.bits.take(5)? as usize + 257;
        let distance_count = self.bits.take(5)? as usize + 1;
        let code_length_count = self.bits.take(4)? as usize + 4;
        if literal_count > 286 || distance_count > 30 {
            return Err(Error::Inflate("a dynamic block declares more than 286 length or 30 distance codes"));
        }
        let mut code_lengths = [0u8; 19];
        for &symbol in &CODE_LENGTH_ORDER[..code_length_count] {
            code_lengths[symbol] = self.bits.take(3)? as u8;
        }
        // The code length code is decoded with the literal table, which is
        // rebuilt below.
        self.literals.build(&code_lengths, CodeKind::CodeLengths)?;

        let total = literal_count + distance_count;
        let mut lengths = [0u8; 286 + 30];
        let mut i = 0;
        while i < total {
            let symbol = self.literals.decode(&mut self.bits)?;
            let (value, repeat) = match symbol {
                0..=15 => (symbol as u8, 1),
                16 => {
                    if i == 0 {
                        return Err(Error::Inflate("a repeat code has no previous length"));
                    }
                    (lengths[i - 1], 3 + self.bits.take(2)? as usize)
                }
                17 => (0, 3 + self.bits.take(3)? as usize),
                _ => (0, 11 + self.bits.take(7)? as usize),
            };
            if i + repeat > total {
                return Err(Error::Inflate("code lengths repeat past the end of the list"));
            }
            lengths[i..i + repeat].fill(value);
            i += repeat;
        }
        if lengths[256] == 0 {
            return Err(Error::Inflate("a dynamic block has no end-of-block code"));
        }
        self.literals.build(&lengths[..literal_count], CodeKind::LiteralsOrDistances)?;
        self.distances.build(&lengths[literal_count..total], CodeKind::LiteralsOrDistances)
    }

    /// Decodes literals and copies up to the end-of-block code.
    fn codes(&mut self) -> Result<()> {
        loop {
            let symbol = self.literals.decode(&mut self.bits)?;
            if symbol < 256 {
                if self.out.len() == self.size {
                    return Err(Error::Inflate("the stream produces more than the stated size"));
                }
                self.out.push(symbol as u8);
                continue;
            }
            if symbol == 256 {
                return Ok(());
            }
            let code = usize::from(symbol - 257);
            if code >= LENGTH_BASE.len() {
                return Err(Error::Inflate("a length symbol is 286 or 287"));
            }
            let len = usize::from(LENGTH_BASE[code]) + self.bits.take(u32::from(LENGTH_EXTRA[code]))? as usize;
            let code = usize::from(self.distances.decode(&mut self.bits)?);
            if code >= DISTANCE_BASE.len() {
                return Err(Error::Inflate("a distance symbol is 30 or 31"));
            }
            let distance = usize::from(DISTANCE_BASE[code]) + self.bits.take(u32::from(DISTANCE_EXTRA[code]))? as usize;
            self.copy(distance, len)?;
        }
    }

    fn copy(&mut self, distance: usize, len: usize) -> Result<()> {
        let pos = self.out.len();
        if distance > pos {
            return Err(Error::Inflate("a distance reaches before the start of the output"));
        }
        if len > self.size - pos {
            return Err(Error::Inflate("the stream produces more than the stated size"));
        }
        let from = pos - distance;
        if distance >= len {
            self.out.extend_from_within(from..from + len);
        } else if distance == 1 {
            let byte = self.out[from];
            self.out.resize(pos + len, byte);
        } else {
            // The source overlaps the bytes being written: the copy repeats
            // the last `distance` bytes, so it goes one byte at a time.
            self.out.reserve(len);
            for i in 0..len {
                let byte = self.out[from + i];
                self.out.push(byte);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod reference_tests;
