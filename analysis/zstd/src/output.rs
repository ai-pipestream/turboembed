//! The output buffer the content is decoded into, sized once to what the
//! caller asks for, the copies that fill it, and the limit a block's
//! output must stay within.

use crate::Error;

/// Where a block's output must end, and which rule output past it
/// breaks: the block's maximum size, its frame's content size, or the
/// size the caller asks for.
pub struct Limit {
    pub end: usize,
    block_start: usize,
    block_max: usize,
    content: Option<u64>,
    want: usize,
}

impl Limit {
    /// The limit of a block starting at `block_start` in a frame whose
    /// content must end by `frame_end`.
    pub fn new(block_start: usize, block_max: usize, frame_end: usize, content: Option<u64>, want: usize) -> Limit {
        Limit { end: frame_end.min(block_start + block_max), block_start, block_max, content, want }
    }

    /// The error for output reaching `end`, past the limit.
    pub fn error(&self, end: usize) -> Error {
        let size = end - self.block_start;
        if size > self.block_max {
            Error::BlockTooLarge { size, max: self.block_max }
        } else if let Some(header) = self.content {
            Error::ContentTooLong { header }
        } else {
            Error::TooLong { want: self.want }
        }
    }
}

/// The output so far: `buf[..pos]` is decoded content. Bytes past `pos`
/// may hold leftovers of copies that move more than they need to; every
/// one is written again before `pos` passes it.
pub struct Output<'a> {
    pub buf: &'a mut [u8],
    pub pos: usize,
}

/// Copies of up to this many bytes move this many at once, when the
/// buffers have room.
const WIDE: usize = 16;

impl Output<'_> {
    /// Appends `src`, which the caller has checked fits.
    pub fn extend(&mut self, src: &[u8]) {
        self.buf[self.pos..self.pos + src.len()].copy_from_slice(src);
        self.pos += src.len();
    }

    /// Appends `n` copies of `byte`, which the caller has checked fit.
    pub fn fill(&mut self, byte: u8, n: usize) {
        self.buf[self.pos..self.pos + n].fill(byte);
        self.pos += n;
    }

    /// Appends the `n` literals at `at` in `lits`, which the caller has
    /// checked are there and fit. `lits` may go on past them, which lets
    /// a short run move 16 bytes at once.
    #[inline(always)]
    pub fn literals(&mut self, lits: &[u8], at: usize, n: usize) {
        let pos = self.pos;
        if n <= WIDE && pos + WIDE <= self.buf.len() && at + WIDE <= lits.len() {
            self.buf[pos..pos + WIDE].copy_from_slice(&lits[at..at + WIDE]);
        } else {
            self.buf[pos..pos + n].copy_from_slice(&lits[at..at + n]);
        }
        self.pos = pos + n;
    }

    /// Appends `n` bytes copied from `offset` back, which the caller has
    /// checked are in the output and fit. When the offset is shorter than
    /// the match the copy reads what it writes, repeating the last
    /// `offset` bytes.
    #[inline(always)]
    pub fn copy_match(&mut self, offset: usize, n: usize) {
        let pos = self.pos;
        let from = pos - offset;
        if offset >= WIDE && pos + n + WIDE <= self.buf.len() {
            // Each 16 bytes are read before they are written, as the
            // offset is at least 16; the last may write past the match.
            let mut i = 0;
            while i < n {
                self.buf.copy_within(from + i..from + i + WIDE, pos + i);
                i += WIDE;
            }
        } else if offset >= n {
            self.buf.copy_within(from..from + n, pos);
        } else if offset == 1 {
            let b = self.buf[from];
            self.buf[pos..pos + n].fill(b);
        } else {
            // The pattern repeats every `offset` bytes, so what is written
            // can be copied again: each copy from `from` doubles the run.
            let mut done = 0;
            while done < n {
                let len = (n - done).min(offset + done);
                self.buf.copy_within(from..from + len, pos + done);
                done += len;
            }
        }
        self.pos = pos + n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `copy_match` against a byte at a time, for offsets and lengths on
    /// both sides of 16 and of each other, near the end of the buffer
    /// and away from it.
    #[test]
    fn matches_copy_as_byte_loops_do() {
        for offset in 1..40 {
            for n in 0..70 {
                for room in [0, 5, 100] {
                    let start = 40;
                    let mut buf: Vec<u8> = (0..start + n + room).map(|i| (i * 7 + 3) as u8).collect();
                    let mut want = buf.clone();
                    for i in 0..n {
                        want[start + i] = want[start + i - offset];
                    }
                    let mut out = Output { buf: &mut buf, pos: start };
                    out.copy_match(offset, n);
                    assert_eq!(out.pos, start + n);
                    assert_eq!(&buf[..start + n], &want[..start + n], "offset {offset} length {n} room {room}");
                }
            }
        }
    }

    #[test]
    fn literals_copy_only_what_is_asked() {
        let lits: Vec<u8> = (0..40).collect();
        for n in 0..20 {
            for room in [0, 30] {
                let mut buf = vec![0xff; 10 + n + room];
                let mut out = Output { buf: &mut buf, pos: 10 };
                out.literals(&lits, 3, n);
                assert_eq!(out.pos, 10 + n);
                assert_eq!(&buf[10..10 + n], &lits[3..3 + n]);
            }
        }
    }
}
