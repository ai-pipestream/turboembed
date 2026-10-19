use super::*;
/// A canonical Huffman code. `table` maps the next `TABLE_BITS` input bits
/// to `symbol | length << 16` for codes of up to `TABLE_BITS` bits; an
/// entry of zero sends the decoder to the canonical loop over `count` and
/// `symbols`.
pub(super) struct Huffman {
    table: [u32; TABLE_SIZE],
    count: [u16; MAX_BITS + 1],
    symbols: [u16; 288],
}

/// Which code a set of lengths describes. Only literal/length and
/// distance codes may be incomplete, and then only when they use a single
/// one-bit code (RFC 1951 allows one distance code; zlib accepts the same
/// shapes, so this follows zlib).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum CodeKind {
    CodeLengths,
    LiteralsOrDistances,
}

impl Huffman {
    pub(super) fn new() -> Box<Huffman> {
        Box::new(Huffman { table: [0; TABLE_SIZE], count: [0; MAX_BITS + 1], symbols: [0; 288] })
    }

    pub(super) fn build(&mut self, lengths: &[u8], kind: CodeKind) -> Result<()> {
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
    pub(super) fn decode(&self, bits: &mut Bits<'_>) -> Result<u16> {
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
    pub(super) fn decode_slow(&self, bits: &mut Bits<'_>) -> Result<u16> {
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
