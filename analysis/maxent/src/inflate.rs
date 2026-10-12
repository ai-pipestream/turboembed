//! RFC 1951 inflate: stored, fixed Huffman and dynamic Huffman blocks,
//! decoded into a buffer whose final size is known in advance (a zip
//! directory record states it).
//!
//! Huffman codes are decoded through a table indexed by the next
//! `TABLE_BITS` bits of input, which resolves every code of up to that
//! length in one lookup. Longer codes, which are rare, fall back to the
//! canonical decoding loop of RFC 1951 section 3.2.2, reading one bit at
//! a time. Nothing is allocated per symbol: the output is allocated once
//! at its stated size and the tables live in one struct per call.

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
    if size > input.len().saturating_mul(MAX_RATIO).saturating_add(MAX_RATIO) {
        return Err(Error::Inflate("the stated size is more than the input can produce"));
    }
    let mut inflater = Inflater {
        bits: Bits { data: input, pos: 0, buf: 0, count: 0 },
        out: vec![0; size],
        pos: 0,
        literals: Huffman::new(),
        distances: Huffman::new(),
    };
    inflater.run()?;
    Ok(inflater.out)
}

/// A little-endian bit reader over a byte slice, as deflate packs its
/// bits: the first bit of the stream is the lowest bit of the first byte.
/// `buf` holds `count` valid bits; every bit above them is zero, so a
/// table lookup near the end of the input sees zeros where input is
/// missing and the decoder then finds the code longer than `count`.
struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    buf: u64,
    count: u32,
}

impl Bits<'_> {
    /// Tops the buffer up to at least 56 bits, or to the end of the input.
    #[inline]
    fn refill(&mut self) {
        if let Some(chunk) = self.data.get(self.pos..self.pos + 8) {
            let word = u64::from_le_bytes(chunk.try_into().expect("an eight byte slice"));
            let take = (63 - self.count) / 8;
            self.buf |= word << self.count;
            self.pos += take as usize;
            self.count += take * 8;
            self.buf &= (1u64 << self.count) - 1;
        } else {
            while self.count <= 56 {
                let Some(&byte) = self.data.get(self.pos) else { break };
                self.buf |= u64::from(byte) << self.count;
                self.pos += 1;
                self.count += 8;
            }
        }
    }

    /// Takes the next `n` bits (at most 32), lowest first.
    #[inline]
    fn take(&mut self, n: u32) -> Result<u32> {
        if self.count < n {
            self.refill();
            if self.count < n {
                return Err(Error::Inflate("the input ends inside a block"));
            }
        }
        let value = (self.buf & ((1u64 << n) - 1)) as u32;
        self.consume(n);
        Ok(value)
    }

    #[inline]
    fn consume(&mut self, n: u32) {
        self.buf >>= n;
        self.count -= n;
    }

    /// Drops the bits up to the next byte boundary and hands back the
    /// position of the next unread byte, emptying the buffer.
    fn align_to_byte(&mut self) -> usize {
        let drop = self.count % 8;
        self.consume(drop);
        let pos = self.pos - (self.count / 8) as usize;
        self.buf = 0;
        self.count = 0;
        self.pos = pos;
        pos
    }
}

/// A canonical Huffman code. `table` maps the next `TABLE_BITS` input bits
/// to `symbol | length << 16` for codes of up to `TABLE_BITS` bits; an
/// entry of zero sends the decoder to the canonical loop over `count` and
/// `symbols`.
struct Huffman {
    table: [u32; TABLE_SIZE],
    count: [u16; MAX_BITS + 1],
    symbols: [u16; 288],
}

/// Which code a set of lengths describes. Only literal/length and
/// distance codes may be incomplete, and then only when they use a single
/// one-bit code (RFC 1951 allows one distance code; zlib accepts the same
/// shapes, so this follows zlib).
#[derive(Clone, Copy, PartialEq, Eq)]
enum CodeKind {
    CodeLengths,
    LiteralsOrDistances,
}

impl Huffman {
    fn new() -> Box<Huffman> {
        Box::new(Huffman { table: [0; TABLE_SIZE], count: [0; MAX_BITS + 1], symbols: [0; 288] })
    }

    fn build(&mut self, lengths: &[u8], kind: CodeKind) -> Result<()> {
        self.count = [0; MAX_BITS + 1];
        for &len in lengths {
            self.count[len as usize] += 1;
        }
        self.count[0] = 0;
        let mut left: i32 = 1;
        let mut max_len = 0;
        for len in 1..=MAX_BITS {
            left <<= 1;
            left -= i32::from(self.count[len]);
            if left < 0 {
                return Err(Error::Inflate("a Huffman code is over-subscribed"));
            }
            if self.count[len] != 0 {
                max_len = len;
            }
        }
        if left > 0 && max_len != 0 && (kind == CodeKind::CodeLengths || max_len != 1) {
            return Err(Error::Inflate("a Huffman code is incomplete"));
        }

        // Symbols sorted by code length, then by symbol value: the order
        // in which canonical codes are assigned.
        let mut offsets = [0u16; MAX_BITS + 2];
        for len in 1..=MAX_BITS {
            offsets[len + 1] = offsets[len] + self.count[len];
        }
        for (symbol, &len) in lengths.iter().enumerate() {
            if len != 0 {
                self.symbols[offsets[len as usize] as usize] = symbol as u16;
                offsets[len as usize] += 1;
            }
        }

        self.table.fill(0);
        let mut next_code = [0u32; MAX_BITS + 1];
        let mut code = 0u32;
        for (len, next) in next_code.iter_mut().enumerate().skip(1) {
            code = (code + u32::from(self.count[len - 1])) << 1;
            *next = code;
        }
        for (symbol, &len) in lengths.iter().enumerate() {
            let len = u32::from(len);
            if len == 0 {
                continue;
            }
            let code = next_code[len as usize];
            next_code[len as usize] += 1;
            if len > TABLE_BITS {
                continue;
            }
            let entry = symbol as u32 | len << 16;
            let mut index = (code.reverse_bits() >> (32 - len)) as usize;
            while index < TABLE_SIZE {
                self.table[index] = entry;
                index += 1 << len;
            }
        }
        Ok(())
    }

    /// Decodes one symbol.
    #[inline]
    fn decode(&self, bits: &mut Bits<'_>) -> Result<u16> {
        if bits.count < MAX_BITS as u32 {
            bits.refill();
        }
        let entry = self.table[(bits.buf & TABLE_MASK) as usize];
        let len = entry >> 16;
        if len != 0 {
            if len > bits.count {
                return Err(Error::Inflate("the input ends inside a block"));
            }
            bits.consume(len);
            return Ok(entry as u16);
        }
        self.decode_slow(bits)
    }

    /// The canonical decoding loop: codes of one length are consecutive
    /// integers, so a code read most significant bit first is found by
    /// comparing it with the first code of each length.
    fn decode_slow(&self, bits: &mut Bits<'_>) -> Result<u16> {
        let mut code: i32 = 0;
        let mut first: i32 = 0;
        let mut index: i32 = 0;
        for len in 1..=MAX_BITS as u32 {
            if len > bits.count {
                return Err(Error::Inflate("the input ends inside a block"));
            }
            code |= ((bits.buf >> (len - 1)) & 1) as i32;
            let count = i32::from(self.count[len as usize]);
            if code - first < count {
                bits.consume(len);
                return Ok(self.symbols[(index + code - first) as usize]);
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        Err(Error::Inflate("an input code is not in the Huffman code"))
    }
}

struct Inflater<'a> {
    bits: Bits<'a>,
    out: Vec<u8>,
    pos: usize,
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
        if self.pos != self.out.len() {
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
        let Some(dest) = self.out.get_mut(self.pos..self.pos + len) else {
            return Err(Error::Inflate("the stream produces more than the stated size"));
        };
        dest.copy_from_slice(bytes);
        self.pos += len;
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
                let Some(slot) = self.out.get_mut(self.pos) else {
                    return Err(Error::Inflate("the stream produces more than the stated size"));
                };
                *slot = symbol as u8;
                self.pos += 1;
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
        if distance > self.pos {
            return Err(Error::Inflate("a distance reaches before the start of the output"));
        }
        if len > self.out.len() - self.pos {
            return Err(Error::Inflate("the stream produces more than the stated size"));
        }
        let from = self.pos - distance;
        if distance >= len {
            self.out.copy_within(from..from + len, self.pos);
        } else if distance == 1 {
            let byte = self.out[from];
            self.out[self.pos..self.pos + len].fill(byte);
        } else {
            // The source overlaps the bytes being written: the copy repeats
            // the last `distance` bytes, so it goes one byte at a time.
            for i in 0..len {
                self.out[self.pos + i] = self.out[from + i];
            }
        }
        self.pos += len;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(result: Result<Vec<u8>>) -> &'static str {
        match result {
            Err(Error::Inflate(message)) => message,
            other => panic!("expected an inflate error, got {other:?}"),
        }
    }

    /// Writes bits least significant first, as deflate packs them, for
    /// building small streams by hand.
    struct BitWriter {
        bytes: Vec<u8>,
        bit: u32,
    }

    impl BitWriter {
        fn new() -> BitWriter {
            BitWriter { bytes: Vec::new(), bit: 0 }
        }

        fn put(&mut self, value: u32, n: u32) {
            for i in 0..n {
                if self.bit == 0 {
                    self.bytes.push(0);
                }
                let b = (value >> i) & 1;
                *self.bytes.last_mut().unwrap() |= (b as u8) << self.bit;
                self.bit = (self.bit + 1) % 8;
            }
        }

        /// Writes a Huffman code most significant bit first.
        fn code(&mut self, code: u32, len: u32) {
            for i in (0..len).rev() {
                self.put((code >> i) & 1, 1);
            }
        }

        /// A fixed Huffman literal or length symbol.
        fn fixed(&mut self, symbol: u32) {
            match symbol {
                0..=143 => self.code(0x30 + symbol, 8),
                144..=255 => self.code(0x190 + symbol - 144, 9),
                256..=279 => self.code(symbol - 256, 7),
                _ => self.code(0xC0 + symbol - 280, 8),
            }
        }
    }

    #[test]
    fn a_fixed_block_with_an_overlapping_copy() {
        // "ab" then a copy of length 6 at distance 2: "abababab".
        let mut w = BitWriter::new();
        w.put(1, 1);
        w.put(1, 2);
        w.fixed(u32::from(b'a'));
        w.fixed(u32::from(b'b'));
        w.fixed(260); // length 6
        w.code(1, 5); // distance 2
        w.fixed(256);
        assert_eq!(inflate(&w.bytes, 8).unwrap(), b"abababab");
    }

    #[test]
    fn a_run_at_distance_one() {
        let mut w = BitWriter::new();
        w.put(1, 1);
        w.put(1, 2);
        w.fixed(u32::from(b'z'));
        w.fixed(285); // length 258
        w.code(0, 5); // distance 1
        w.fixed(256);
        assert_eq!(inflate(&w.bytes, 259).unwrap(), vec![b'z'; 259]);
    }

    #[test]
    fn stored_blocks_including_an_empty_one() {
        let mut bytes = vec![0b000, 3, 0, !3, 0xFF, b'a', b'b', b'c'];
        bytes.extend_from_slice(&[0b001, 0, 0, 0xFF, 0xFF]);
        assert_eq!(inflate(&bytes, 3).unwrap(), b"abc");
    }

    #[test]
    fn a_distance_before_the_start_of_the_output() {
        let mut w = BitWriter::new();
        w.put(1, 1);
        w.put(1, 2);
        w.fixed(u32::from(b'a'));
        w.fixed(257); // length 3
        w.code(1, 5); // distance 2, but only one byte is out
        w.fixed(256);
        assert_eq!(message(inflate(&w.bytes, 4)), "a distance reaches before the start of the output");
    }

    #[test]
    fn a_stored_length_that_disagrees_with_its_complement() {
        let bytes = [0b001, 3, 0, 0, 0, b'a', b'b', b'c'];
        assert_eq!(message(inflate(&bytes, 3)), "a stored block's length and its complement disagree");
    }

    #[test]
    fn the_reserved_block_type() {
        assert_eq!(message(inflate(&[0b111], 0)), "a block has the reserved type 3");
    }

    #[test]
    fn truncated_input() {
        let mut w = BitWriter::new();
        w.put(1, 1);
        w.put(1, 2);
        w.fixed(u32::from(b'a'));
        // No end-of-block code follows; the remaining bits of the byte are
        // zero, which is the start of a seven-bit code with no input left.
        assert_eq!(message(inflate(&w.bytes, 1)), "the input ends inside a block");
        assert_eq!(message(inflate(&[0b000, 3, 0], 3)), "the input ends inside a stored block header");
        assert_eq!(message(inflate(&[0b001, 3, 0, !3, 0xFF, b'a'], 3)), "the input ends inside a stored block");
    }

    #[test]
    fn output_longer_or_shorter_than_stated() {
        let bytes = [0b001, 3, 0, !3, 0xFF, b'a', b'b', b'c'];
        assert_eq!(message(inflate(&bytes, 2)), "the stream produces more than the stated size");
        assert_eq!(message(inflate(&bytes, 4)), "the stream ends before the stated size");
        assert_eq!(message(inflate(&bytes, 1 << 30)), "the stated size is more than the input can produce");
    }

    #[test]
    fn an_over_subscribed_code_length_code() {
        // A dynamic block whose code length code gives four symbols one bit.
        let mut w = BitWriter::new();
        w.put(1, 1);
        w.put(2, 2);
        w.put(0, 5);
        w.put(0, 5);
        w.put(0, 4); // four code length lengths follow: symbols 16, 17, 18, 0
        for _ in 0..4 {
            w.put(1, 3);
        }
        assert_eq!(message(inflate(&w.bytes, 0)), "a Huffman code is over-subscribed");
    }

    #[test]
    fn an_incomplete_code_length_code() {
        let mut w = BitWriter::new();
        w.put(1, 1);
        w.put(2, 2);
        w.put(0, 5);
        w.put(0, 5);
        w.put(0, 4);
        w.put(2, 3); // symbol 16 gets two bits, nothing else: incomplete
        w.put(0, 3);
        w.put(0, 3);
        w.put(0, 3);
        assert_eq!(message(inflate(&w.bytes, 0)), "a Huffman code is incomplete");
    }

    #[test]
    fn codes_longer_than_the_table_use_the_canonical_loop() {
        // A code where symbol 0 has one bit, then 1, 2, ... down to two
        // fifteen-bit codes: lengths 1..=14 for symbols 0..=13 and 15 for
        // symbols 14 and 15. That is complete.
        let mut lengths = [0u8; 16];
        for (i, len) in lengths.iter_mut().enumerate() {
            *len = (i as u8 + 1).min(15);
        }
        let mut h = Huffman::new();
        h.build(&lengths, CodeKind::LiteralsOrDistances).unwrap();
        // Canonical codes: symbol k < 14 is k ones then a zero; symbols 14
        // and 15 are fourteen ones then 0 and 1.
        for symbol in 0..16u32 {
            let len = u32::from(lengths[symbol as usize]);
            let code = match symbol {
                0..=13 => ((1u32 << symbol) - 1) << 1,
                14 => ((1u32 << 14) - 1) << 1,
                _ => (1u32 << 15) - 1,
            };
            let mut w = BitWriter::new();
            w.code(code, len);
            let mut bits = Bits { data: &w.bytes, pos: 0, buf: 0, count: 0 };
            assert_eq!(u32::from(h.decode(&mut bits).unwrap()), symbol, "symbol {symbol}");
        }
    }
}
