//! FSE, the tabled asymmetric numeral system the format codes sequence
//! symbols and Huffman weights with: reading a table description (a
//! normalized distribution) and building the decoding table from it.

use crate::Error;
use crate::error::Part;

/// The largest accuracy log any use allows (9, for literal and match
/// lengths), so the largest table.
pub const MAX_LOG: u32 = 9;

/// The largest symbol any use allows (52, the last match length code).
const MAX_SYMBOLS: usize = 53;

/// A normalized distribution: each symbol's probability, -1 standing for
/// "less than 1", summing to `1 << log` with -1 counted as 1.
pub struct Distribution {
    pub log: u32,
    pub probs: [i16; MAX_SYMBOLS],
    /// The symbols described, 0 to the last one with a probability.
    pub len: usize,
}

impl Distribution {
    /// A distribution of `probs`, which must be normalized to `log`.
    pub fn of(log: u32, probs: &[i16]) -> Distribution {
        let mut d = Distribution { log, probs: [0; MAX_SYMBOLS], len: probs.len() };
        d.probs[..probs.len()].copy_from_slice(probs);
        d
    }

    /// Reads a table description at the start of `src`, for symbols up
    /// to `max_symbol` and an accuracy log up to `max_log`. Returns the
    /// distribution and the bytes the description takes.
    pub fn read(src: &[u8], max_symbol: usize, max_log: u32) -> Result<(Distribution, usize), Error> {
        let mut bits = Forward { src, bit: 0 };
        let log = bits.read(4)? + 5;
        if log > max_log {
            return Err(Error::AccuracyLog { log, max: max_log });
        }
        let mut d = Distribution { log, probs: [0; MAX_SYMBOLS], len: 0 };
        // The points not yet given; every value read is at most this plus
        // one, so it never goes below 0.
        let mut remaining = 1i32 << log;
        while remaining > 0 {
            if d.len > max_symbol {
                return Err(Error::SymbolPastMax { symbol: d.len, max: max_symbol });
            }
            // Values 0 to `remaining + 1` are possible. The field has the
            // fewest bits that can hold them; the values below `low` take
            // one bit less.
            let top = (remaining + 1) as u32;
            let n = 32 - top.leading_zeros();
            let low = (1 << n) - 1 - top;
            let peek = bits.peek(n);
            let half = peek & ((1 << (n - 1)) - 1);
            let value = if half < low {
                bits.skip(n - 1)?;
                half
            } else {
                bits.skip(n)?;
                if peek > half { peek - low } else { peek }
            };
            let prob = value as i32 - 1;
            remaining -= prob.abs();
            d.probs[d.len] = prob as i16;
            d.len += 1;
            if prob == 0 {
                // Repeat flags: each says how many more zeros follow, and
                // a 3 says another flag follows.
                loop {
                    let repeat = bits.read(2)? as usize;
                    if d.len + repeat > max_symbol + 1 {
                        return Err(Error::SymbolPastMax { symbol: d.len + repeat - 1, max: max_symbol });
                    }
                    d.len += repeat;
                    if repeat < 3 {
                        break;
                    }
                }
            }
        }
        Ok((d, bits.bit.div_ceil(8)))
    }
}

/// A table description is read forward, low bits first.
struct Forward<'a> {
    src: &'a [u8],
    bit: usize,
}

impl Forward<'_> {
    /// The next `n` bits (at most 16), zeros past the end.
    fn peek(&self, n: u32) -> u32 {
        let at = self.bit / 8;
        let b = |i: usize| u32::from(self.src.get(at + i).copied().unwrap_or(0));
        let word = b(0) | b(1) << 8 | b(2) << 16;
        (word >> (self.bit % 8)) & ((1 << n) - 1)
    }

    fn skip(&mut self, n: u32) -> Result<(), Error> {
        self.bit += n as usize;
        if self.bit > 8 * self.src.len() {
            return Err(Error::Truncated(Part::FseTable));
        }
        Ok(())
    }

    fn read(&mut self, n: u32) -> Result<u32, Error> {
        let v = self.peek(n);
        self.skip(n)?;
        Ok(v)
    }
}

/// One state of a decoding table: its symbol, and the next state, which
/// is `base` plus the next `bits` bits of the stream.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Cell {
    pub symbol: u8,
    pub bits: u8,
    pub base: u16,
}

/// Fills `cells[..1 << d.log]` with the decoding table of `d`, as the
/// format's "from normalized distribution to decoding tables" builds it.
pub fn build(d: &Distribution, cells: &mut [Cell; 1 << MAX_LOG]) {
    let size = 1usize << d.log;
    let probs = &d.probs[..d.len];
    // Each symbol's next state descriptor, which counts up from its
    // probability as its cells are given out in state order.
    let mut next = [0u16; MAX_SYMBOLS];
    // "Less than 1" symbols take one cell each from the end.
    let mut high = size;
    for (s, &p) in probs.iter().enumerate() {
        if p == -1 {
            high -= 1;
            cells[high].symbol = s as u8;
            next[s] = 1;
        }
    }
    // The others spread over the rest, stepping by a stride that is odd,
    // so it visits every cell, and skipping the cells at the end.
    let step = (size >> 1) + (size >> 3) + 3;
    let mask = size - 1;
    let mut pos = 0;
    for (s, &p) in probs.iter().enumerate() {
        if p <= 0 {
            continue;
        }
        next[s] = p as u16;
        for _ in 0..p {
            cells[pos].symbol = s as u8;
            pos = (pos + step) & mask;
            while pos >= high {
                pos = (pos + step) & mask;
            }
        }
    }
    // In state order, a symbol's first cells read one bit more than its
    // last, so that together they cover the whole range of states.
    for cell in &mut cells[..size] {
        let s = usize::from(cell.symbol);
        let desc = u32::from(next[s]);
        next[s] += 1;
        let bits = d.log - (31 - desc.leading_zeros());
        cell.bits = bits as u8;
        cell.base = ((desc << bits) - size as u32) as u16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variable_width_values() {
        // The format's example: accuracy log 8 with 100 points given, so
        // values 0 to 157 in an 8 bit field, of which 0 to 97 take 7 bits.
        // Symbol 0 gives the 100 points; the field under test is symbol
        // 1's; symbol 2 takes what is left.
        for (field, width, value) in [(97, 7, 97), (98, 8, 98), (127, 8, 127), (226, 8, 128), (255, 8, 157)] {
            let mut w = Writer::default();
            w.put(3, 4);
            w.value(101, 256);
            w.put(field, width);
            let left = 156 - (value - 1);
            if left > 0 {
                w.value(left + 1, left);
            }
            let (d, _) = Distribution::read(&w.bytes(), 52, 9).unwrap();
            assert_eq!(&d.probs[..d.len], &[100, value as i16 - 1, left as i16][..d.len], "field {field}");
        }
        // A 7 bit field whose next bit is 1 reads as 128 to 225 in 8 bits
        // and is still 0 to 97: here 97, then symbol 2's field starting
        // with a 1 bit.
        let mut w = Writer::default();
        w.put(3, 4);
        w.value(101, 256);
        w.put(97, 7);
        w.value(61, 60);
        assert!(w.bits[4 + 8 + 7]);
        let (d, _) = Distribution::read(&w.bytes(), 52, 9).unwrap();
        assert_eq!(&d.probs[..d.len], &[100, 96, 60]);
    }

    /// Writes fields low bit first, as a table description is laid out.
    #[derive(Default)]
    struct Writer {
        bits: Vec<bool>,
    }

    impl Writer {
        fn put(&mut self, v: u32, n: u32) {
            self.bits.extend((0..n).map(|i| v >> i & 1 == 1));
        }

        /// `v` in the field for `remaining` points left, as the format
        /// lays it out.
        fn value(&mut self, v: u32, remaining: u32) {
            let top = remaining + 1;
            let n = 32 - top.leading_zeros();
            let low = (1 << n) - 1 - top;
            if v < low {
                self.put(v, n - 1);
            } else if v < 1 << (n - 1) {
                self.put(v, n);
            } else {
                self.put(v + low, n);
            }
        }

        fn bytes(&self) -> Vec<u8> {
            self.bits.chunks(8).map(|c| c.iter().rev().fold(0, |b, &x| b << 1 | u8::from(x))).collect()
        }
    }

    #[test]
    fn zero_repeats_and_less_than_one() {
        // Accuracy log 5 (32 points): symbol 0 at -1 (value 0), symbol 1
        // at 0 (value 1) followed by repeat flags 3 then 1 (four more
        // zeros: symbols 2 to 5), then symbol 6 with the other 31.
        let mut w = Writer::default();
        w.put(0, 4);
        w.value(0, 32);
        w.value(1, 31);
        w.put(3, 2);
        w.put(1, 2);
        w.value(32, 31);
        let (d, used) = Distribution::read(&w.bytes(), 52, 9).unwrap();
        assert_eq!(&d.probs[..d.len], &[-1, 0, 0, 0, 0, 0, 31]);
        assert_eq!(used, 3);
    }

    #[test]
    fn limits_are_enforced() {
        // Accuracy log 10 where 9 is the most.
        assert_eq!(Distribution::read(&[5, 0], 52, 9).err(), Some(Error::AccuracyLog { log: 10, max: 9 }));
        // Symbol 0 at 0 with repeat flags of 3 that run past symbol 35.
        let mut w = Writer::default();
        w.put(0, 4);
        w.value(1, 32);
        for _ in 0..13 {
            w.put(3, 2);
        }
        assert_eq!(Distribution::read(&w.bytes(), 35, 9).err(), Some(Error::SymbolPastMax { symbol: 36, max: 35 }));
        // The description ends before its points are given.
        assert_eq!(Distribution::read(&[0x00], 52, 9).err(), Some(Error::Truncated(Part::FseTable)));
        // Symbols 0 and 1 at probability 1, and points left for a symbol 2
        // where 1 is the last allowed.
        let mut w = Writer::default();
        w.put(0, 4);
        w.value(2, 32);
        w.value(2, 31);
        w.value(31, 30);
        assert_eq!(Distribution::read(&w.bytes(), 1, 9).err(), Some(Error::SymbolPastMax { symbol: 2, max: 1 }));
        assert_eq!(Distribution::read(&w.bytes(), 2, 9).unwrap().0.len, 3);
    }

    #[test]
    fn the_formats_example_symbol() {
        // A symbol of probability 5 at accuracy log 7 whose first cell is
        // state 1 gets states 1, 84, 39, 122, 77, read with 5, 5, 5, 4, 4
        // bits from bases 32, 64, 96, 0, 16. Symbol 0 takes 91 cells, so
        // symbol 1 starts at state 1, and the next symbol at 32.
        let d = Distribution::of(7, &[91, 5, 32]);
        let mut cells = [Cell::default(); 1 << MAX_LOG];
        build(&d, &mut cells);
        let got: Vec<_> =
            (0..128).filter(|&i| cells[i].symbol == 1).map(|i| (i, cells[i].bits, cells[i].base)).collect();
        assert_eq!(got, [(1, 5, 32), (39, 5, 64), (77, 5, 96), (84, 4, 0), (122, 4, 16)]);
        assert_eq!(cells[32].symbol, 2);
        // A "less than 1" symbol takes the last cell, and a probability of
        // 1 reads the whole accuracy log from base 0.
        let d = Distribution::of(5, &[-1, 1, 30]);
        build(&d, &mut cells);
        assert_eq!(cells[31], Cell { symbol: 0, bits: 5, base: 0 });
        assert_eq!(cells[0], Cell { symbol: 1, bits: 5, base: 0 });
    }
}
