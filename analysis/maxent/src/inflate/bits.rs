use super::*;
/// bits: the first bit of the stream is the lowest bit of the first byte.
/// `buf` holds `count` valid bits; every bit above them is zero, so a
/// table lookup near the end of the input sees zeros where input is
/// missing and the decoder then finds the code longer than `count`.
pub(super) struct Bits<'a> {
    pub(super) data: &'a [u8],
    pub(super) pos: usize,
    pub(super) buf: u64,
    pub(super) count: u32,
}

impl Bits<'_> {
    /// Tops the buffer up to at least 56 bits, or to the end of the input.
    #[inline]
    pub(super) fn refill(&mut self) {
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
    pub(super) fn take(&mut self, n: u32) -> Result<u32> {
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
    pub(super) fn consume(&mut self, n: u32) {
        self.buf >>= n;
        self.count -= n;
    }

    /// Drops the bits up to the next byte boundary and hands back the
    /// position of the next unread byte, emptying the buffer.
    pub(super) fn align_to_byte(&mut self) -> usize {
        let drop = self.count % 8;
        self.consume(drop);
        let pos = self.pos - (self.count / 8) as usize;
        self.buf = 0;
        self.count = 0;
        self.pos = pos;
        pos
    }
}
