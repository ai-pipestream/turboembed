//! The bitstreams of Huffman and FSE coded data, which are read backwards:
//! the writer appends a 1 bit after its last bit and pads the byte with
//! zeros, so the last byte holds the first bits to read, under that mark.
//! Each field is read as a little-endian number, the first bit read being
//! its highest.

use crate::Error;

/// A backward reader over one bitstream. It keeps the eight bytes at
/// `pos` in `container` and counts the bits read from the top of them;
/// [`reload`](Self::reload) moves the eight bytes back over what has been
/// read. Past the stream's start it reads zeros, and
/// [`finished`](Self::finished) tells whether exactly every bit was read.
pub struct BackBits<'a> {
    src: &'a [u8],
    pos: usize,
    container: u64,
    consumed: u32,
}

/// The eight bytes at `pos`. Every caller keeps `pos + 8` within `src`;
/// the zero is never used.
#[inline(always)]
fn load(src: &[u8], pos: usize) -> u64 {
    src.get(pos..).and_then(|s| s.first_chunk::<8>()).map_or(0, |b| u64::from_le_bytes(*b))
}

impl<'a> BackBits<'a> {
    /// A reader at the first bit of `src`, under its end mark.
    pub fn new(src: &'a [u8]) -> Result<BackBits<'a>, Error> {
        let last = *src.last().ok_or(Error::NoEndMark)?;
        if last == 0 {
            return Err(Error::NoEndMark);
        }
        let mark = last.leading_zeros() + 1;
        if src.len() >= 8 {
            let pos = src.len() - 8;
            return Ok(BackBits { src, pos, container: load(src, pos), consumed: mark });
        }
        // A short stream sits in the low bytes, its missing high bytes
        // counted as read.
        let container = src.iter().rev().fold(0u64, |c, &b| c << 8 | u64::from(b));
        Ok(BackBits { src, pos: 0, container, consumed: mark + 8 * (8 - src.len() as u32) })
    }

    /// The next `n` bits without reading them, `n` at most 56.
    #[inline(always)]
    pub fn peek(&self, n: u32) -> u64 {
        // The shift in two steps gives 0 for `n` = 0. Past the stream's
        // start `consumed` is over 64 and the bits are meaningless, which
        // `finished` reports.
        (self.container << (self.consumed & 63)) >> 1 >> (63 - n)
    }

    #[inline(always)]
    pub fn skip(&mut self, n: u32) {
        self.consumed += n;
    }

    /// Reads `n` bits, at most 56.
    #[inline(always)]
    pub fn read(&mut self, n: u32) -> u64 {
        let v = self.peek(n);
        self.consumed += n;
        v
    }

    /// Moves the container back over the bytes read. After it at least 57
    /// unread bits are in the container, or all that are left.
    #[inline(always)]
    pub fn reload(&mut self) {
        if self.pos == 0 {
            return;
        }
        let back = ((self.consumed >> 3) as usize).min(self.pos);
        self.pos -= back;
        self.consumed -= 8 * back as u32;
        self.container = load(self.src, self.pos);
    }

    /// Whether more bits were read than the stream holds.
    pub fn overflowed(&mut self) -> bool {
        self.reload();
        self.pos == 0 && self.consumed > 64
    }

    /// Whether every bit of the stream was read, and no more.
    pub fn finished(&mut self) -> bool {
        self.reload();
        self.pos == 0 && self.consumed == 64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fields_come_from_the_end_highest_bit_first() {
        // 0b0000_0101 0b1011_0001: the mark is bit 2 of the last byte;
        // under it 01, then the first byte 1011_0001.
        let mut b = BackBits::new(&[0b1011_0001, 0b0000_0101]).unwrap();
        assert_eq!(b.read(2), 0b01);
        assert_eq!(b.read(3), 0b101);
        assert_eq!(b.read(0), 0);
        assert_eq!(b.read(5), 0b10001);
        assert!(b.finished());
        // Past the start, zeros, and the stream reports it.
        assert_eq!(b.read(3), 0);
        assert!(b.overflowed() && !b.finished());
    }

    #[test]
    fn long_streams_reload() {
        // 20 bytes: a 1 bit as the mark in the last byte, then 152 bits
        // of the pattern 1100 read back to the start.
        let mut src = vec![0b1100_1100u8; 20];
        src[19] = 1;
        let mut b = BackBits::new(&src).unwrap();
        for _ in 0..19 {
            b.reload();
            assert_eq!(b.read(4), 0b1100);
            assert_eq!(b.read(4), 0b1100);
        }
        b.reload();
        assert!(b.finished());
    }

    #[test]
    fn a_last_byte_of_zero_has_no_mark() {
        assert_eq!(BackBits::new(&[1, 0]).err(), Some(Error::NoEndMark));
        assert_eq!(BackBits::new(&[]).err(), Some(Error::NoEndMark));
        // A mark alone is an empty stream.
        assert!(BackBits::new(&[1]).unwrap().finished());
    }
}
