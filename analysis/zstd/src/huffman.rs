//! Huffman coded literals: the tree description (weights, written
//! directly or FSE coded), the decoding table built from the weights, and
//! the one or four streams the literals are coded in.

use crate::Error;
use crate::bits::BackBits;
use crate::error::Part;
use crate::fse::{self, Cell, Distribution};

/// The deepest table the format allows.
const MAX_BITS: u32 = 11;

/// The most weights a description lists; the last symbol's weight is
/// left out, so up to 256 symbols.
const MAX_WEIGHTS: usize = 255;

#[derive(Debug, Clone, Copy, Default)]
struct Entry {
    symbol: u8,
    bits: u8,
}

/// A decoding table: indexed by the next `log` bits of a stream, the
/// symbol they start with and the length of its code.
pub struct Table {
    log: u32,
    entries: [Entry; 1 << MAX_BITS],
}

impl Table {
    /// Whether no table has been read since the last [`clear`](Self::clear).
    pub fn is_empty(&self) -> bool {
        self.log == 0
    }

    /// Forgets the table, as a new frame starts with none.
    pub fn clear(&mut self) {
        self.log = 0;
    }

    /// Reads the tree description at the start of `src` into a table.
    /// Returns the bytes the description takes.
    pub fn read(&mut self, src: &[u8]) -> Result<usize, Error> {
        let header = usize::from(*src.first().ok_or(Error::Truncated(Part::HuffmanTable))?);
        let mut weights = [0u8; MAX_WEIGHTS];
        let (n, used) = if header >= 128 {
            // Direct: 4 bits each, two to a byte, the first in the high
            // half.
            let n = header - 127;
            let bytes = src.get(1..1 + n.div_ceil(2)).ok_or(Error::Truncated(Part::HuffmanTable))?;
            for (i, w) in weights[..n].iter_mut().enumerate() {
                *w = if i % 2 == 0 { bytes[i / 2] >> 4 } else { bytes[i / 2] & 15 };
            }
            (n, 1 + bytes.len())
        } else {
            let stream = src.get(1..1 + header).ok_or(Error::Truncated(Part::HuffmanTable))?;
            (fse_weights(stream, &mut weights)?, 1 + header)
        };
        self.build(&weights[..n])?;
        Ok(used)
    }

    /// Builds the table from the weights of every symbol but the last,
    /// whose weight is what completes the others to a power of 2.
    fn build(&mut self, weights: &[u8]) -> Result<(), Error> {
        // Each weight w stands for 2^(w-1) of the table's 2^bits entries.
        let sum: u32 = weights.iter().map(|&w| (1u32 << w) >> 1).sum();
        if sum == 0 {
            return Err(Error::NoWeights);
        }
        let bits = 32 - sum.leading_zeros();
        if bits > MAX_BITS {
            return Err(Error::HuffmanTooDeep(bits));
        }
        let rest = (1u32 << bits) - sum;
        if !rest.is_power_of_two() {
            return Err(Error::WeightsIncomplete);
        }
        let last = (rest.trailing_zeros() + 1) as u8;
        let all = || weights.iter().copied().chain(std::iter::once(last));
        // Codes go out from the lowest weight up, each weight's symbols
        // in their order, so a weight's entries start after those of all
        // lower weights.
        let mut count = [0u32; MAX_BITS as usize + 1];
        for w in all() {
            count[usize::from(w)] += 1;
        }
        if count[1] == 0 {
            return Err(Error::NoWeightOne);
        }
        let mut start = [0u32; MAX_BITS as usize + 1];
        for w in 2..=bits as usize {
            start[w] = start[w - 1] + (count[w - 1] << (w - 2));
        }
        for (symbol, w) in all().enumerate() {
            if w == 0 {
                continue;
            }
            let at = start[usize::from(w)] as usize;
            let len = 1 << (w - 1);
            let e = Entry { symbol: symbol as u8, bits: (bits + 1 - u32::from(w)) as u8 };
            self.entries[at..at + len].fill(e);
            start[usize::from(w)] += len as u32;
        }
        self.log = bits;
        Ok(())
    }

    #[inline(always)]
    fn decode(&self, b: &mut BackBits) -> u8 {
        let e = self.entries[b.peek(self.log) as usize & ((1 << MAX_BITS) - 1)];
        b.skip(u32::from(e.bits));
        e.symbol
    }

    /// Decodes one stream into `out`, which it must fill exactly.
    pub fn decode_one(&self, src: &[u8], out: &mut [u8]) -> Result<(), Error> {
        let mut b = BackBits::new(src)?;
        let (chunks, tail) = out.as_chunks_mut::<4>();
        // Four codes of at most 11 bits fit in what a reload leaves.
        for c in chunks {
            b.reload();
            for x in c {
                *x = self.decode(&mut b);
            }
        }
        for x in tail {
            b.reload();
            *x = self.decode(&mut b);
        }
        if !b.finished() {
            return Err(Error::StreamNotConsumed);
        }
        Ok(())
    }

    /// Decodes four streams, `src` starting with their jump table, into
    /// `out`: each stream a quarter of it rounded up, the last what is
    /// left.
    pub fn decode_four(&self, src: &[u8], out: &mut [u8]) -> Result<(), Error> {
        if out.len() < 6 {
            return Err(Error::TooFewLiteralsForFourStreams(out.len()));
        }
        let jump = src.get(..6).ok_or(Error::Truncated(Part::JumpTable))?;
        let size = |i: usize| usize::from(u16::from_le_bytes([jump[2 * i], jump[2 * i + 1]]));
        let (s1, s2, s3) = (size(0), size(1), size(2));
        // The fourth stream takes the rest, and is at least one byte.
        if 6 + s1 + s2 + s3 >= src.len() {
            return Err(Error::JumpTable);
        }
        let (e1, e2, e3) = (6 + s1, 6 + s1 + s2, 6 + s1 + s2 + s3);
        let mut b = [
            BackBits::new(&src[6..e1])?,
            BackBits::new(&src[e1..e2])?,
            BackBits::new(&src[e2..e3])?,
            BackBits::new(&src[e3..])?,
        ];
        let seg = out.len().div_ceil(4);
        let (o1, rest) = out.split_at_mut(seg);
        let (o2, rest) = rest.split_at_mut(seg);
        let (o3, o4) = rest.split_at_mut(seg);
        // The streams side by side, four symbols each per reload, as far
        // as the shortest (the last) goes; then each stream's tail.
        let rounds = o4.len() / 4;
        {
            let (c1, _) = o1.as_chunks_mut::<4>();
            let (c2, _) = o2.as_chunks_mut::<4>();
            let (c3, _) = o3.as_chunks_mut::<4>();
            let (c4, _) = o4.as_chunks_mut::<4>();
            for (((c1, c2), c3), c4) in c1.iter_mut().zip(c2.iter_mut()).zip(c3.iter_mut()).zip(c4.iter_mut()) {
                for b in &mut b {
                    b.reload();
                }
                for k in 0..4 {
                    c1[k] = self.decode(&mut b[0]);
                    c2[k] = self.decode(&mut b[1]);
                    c3[k] = self.decode(&mut b[2]);
                    c4[k] = self.decode(&mut b[3]);
                }
            }
        }
        for (b, o) in b.iter_mut().zip([o1, o2, o3, o4]) {
            for x in &mut o[rounds * 4..] {
                b.reload();
                *x = self.decode(b);
            }
            if !b.finished() {
                return Err(Error::StreamNotConsumed);
            }
        }
        Ok(())
    }
}

impl Default for Table {
    fn default() -> Table {
        Table { log: 0, entries: [Entry::default(); 1 << MAX_BITS] }
    }
}

/// Decodes FSE coded weights from `src` into `weights`, returning how
/// many. Two states share one table and take turns; the weights end when
/// a state update reads past the stream's start, and the other state's
/// symbol is the last.
fn fse_weights(src: &[u8], weights: &mut [u8; MAX_WEIGHTS]) -> Result<usize, Error> {
    let (d, used) = Distribution::read(src, MAX_BITS as usize, 6)?;
    let mut cells = [Cell::default(); 1 << fse::MAX_LOG];
    fse::build(&d, &mut cells);
    let mut b = BackBits::new(&src[used.min(src.len())..])?;
    let mut states = [b.read(d.log) as usize, b.read(d.log) as usize];
    if b.overflowed() {
        return Err(Error::Truncated(Part::HuffmanTable));
    }
    let mut n = 0;
    loop {
        for turn in 0..2 {
            if n == MAX_WEIGHTS {
                return Err(Error::TooManyWeights);
            }
            let c = cells[states[turn] & (cells.len() - 1)];
            weights[n] = c.symbol;
            n += 1;
            states[turn] = usize::from(c.base) + b.read(u32::from(c.bits)) as usize;
            if b.overflowed() {
                if n == MAX_WEIGHTS {
                    return Err(Error::TooManyWeights);
                }
                weights[n] = cells[states[1 - turn] & (cells.len() - 1)].symbol;
                return Ok(n + 1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(weights: &[u8]) -> Result<Table, Error> {
        let mut t = Table::default();
        t.build(weights)?;
        Ok(t)
    }

    #[test]
    fn the_formats_example_tree() {
        // Weights A 4, B 3, C 2, D 0, E 1, F 1 (F's left out), so 4 bits
        // deep, and the prefix codes A 1, B 01, C 001, E 0000, F 0001.
        let mut w = vec![0u8; 65];
        w.extend([4, 3, 2, 0, 1]);
        let t = table(&w).unwrap();
        assert_eq!(t.log, 4);
        let code = |sym: u8| {
            let i = t.entries.iter().position(|e| e.symbol == sym && e.bits > 0).unwrap();
            let bits = t.entries[i].bits;
            (i >> (4 - bits), bits)
        };
        assert_eq!(code(b'A'), (0b1, 1));
        assert_eq!(code(b'B'), (0b01, 2));
        assert_eq!(code(b'C'), (0b001, 3));
        assert_eq!(code(b'E'), (0b0000, 4));
        assert_eq!(code(b'F'), (0b0001, 4));
        // "ABEF" written last symbol first, the mark above it: 0000_0001
        // (F under E) then 0000_1_1_01 (the mark, A, B).
        let mut out = [0u8; 4];
        t.decode_one(&[0b0000_0001, 0b0000_1101], &mut out).unwrap();
        assert_eq!(&out, b"ABEF");
        // The stream must end with the last symbol: one symbol short, and
        // one more than the bits hold.
        assert_eq!(t.decode_one(&[0b0000_0001, 0b0000_1101], &mut [0; 3]), Err(Error::StreamNotConsumed));
        assert_eq!(t.decode_one(&[0b0000_0001, 0b0000_1101], &mut [0; 5]), Err(Error::StreamNotConsumed));
    }

    #[test]
    fn weight_rules() {
        // All zero.
        assert_eq!(table(&[0, 0]).err(), Some(Error::NoWeights));
        // 1 + 2 = 3, and 4 - 3 = 1 completes it: weights 1, 2, 1.
        assert!(table(&[1, 2]).is_ok());
        // 2 + 1 + 1 = 4 needs 4 more, weight 3: fine. 2 + 2 + 1 = 5 needs
        // 3, not a power of 2.
        assert!(table(&[2, 1, 1]).is_ok());
        assert_eq!(table(&[2, 2, 1]).err(), Some(Error::WeightsIncomplete));
        // A single weight 2 completes with another 2: two 1-bit codes but
        // no weight 1.
        assert_eq!(table(&[2]).err(), Some(Error::NoWeightOne));
        // 11 bits deep is the most: weights 1 and 11 down to 2 sum to
        // 2047, and a last weight of 1 fills the 2048 entries. A weight of
        // 12 makes the table 12 bits deep.
        assert_eq!(table(&[1, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]).unwrap().log, 11);
        assert_eq!(table(&[1, 12]).err(), Some(Error::HuffmanTooDeep(12)));
    }

    #[test]
    fn direct_weights() {
        // Header 128 + 4: four weights in 0x21 0x10, so 2, 1, 1, 0, which
        // sum to 4; the last symbol (4) gets weight 3 to make 8, so the
        // table is 3 bits deep and a weight w takes 2^(w-1) entries, the
        // lowest weights first.
        let mut t = Table::default();
        assert_eq!(t.read(&[131, 0x21, 0x10, 0xff]).unwrap(), 3);
        assert_eq!(t.log, 3);
        let syms: Vec<u8> = t.entries[..8].iter().map(|e| e.symbol).collect();
        assert_eq!(syms, [1, 2, 0, 0, 4, 4, 4, 4]);
        // An odd count still takes whole bytes, the last low half unused.
        assert_eq!(t.read(&[129, 0x11]).unwrap(), 2);
        assert_eq!(t.read(&[130, 0x11]).err(), Some(Error::Truncated(Part::HuffmanTable)));
    }

    #[test]
    fn fse_weights_go_up_to_11() {
        // A 3-byte description at accuracy log 5: symbol 0 at probability
        // 0, repeat flags 3, 3, 3 and 2 for 11 more zeros, then all 32
        // points on symbol 12, which no weight can be.
        let mut t = Table::default();
        assert_eq!(t.read(&[3, 0x10, 0x7E, 0x7F]).err(), Some(Error::SymbolPastMax { symbol: 12, max: 11 }));
        // Flags 3, 3, 3 and 1 put the points on symbol 11 instead, and the
        // description is taken; the bitstream after it is missing, so has
        // no end mark.
        assert_eq!(t.read(&[3, 0x10, 0xFE, 0x7E]).err(), Some(Error::NoEndMark));
    }

    #[test]
    fn four_streams_split_the_literals() {
        // Two symbols of 1 bit: 0 is "a", 1 is "b". Ten literals give
        // streams of 3, 3, 3 and 1 symbols.
        let t = table(&[1]).unwrap();
        // A stream of k symbols, the first in the highest bit under the
        // mark.
        let stream = |syms: &[u8]| {
            let mut v = 1u32;
            for &s in syms {
                v = v << 1 | u32::from(s);
            }
            let n = (32 - v.leading_zeros()).div_ceil(8) as usize;
            v.to_le_bytes()[..n].to_vec()
        };
        let parts = [stream(&[0, 1, 1]), stream(&[1, 0, 0]), stream(&[1, 1, 1]), stream(&[0])];
        let mut src = Vec::new();
        for p in &parts[..3] {
            src.extend((p.len() as u16).to_le_bytes());
        }
        for p in &parts {
            src.extend(p);
        }
        let mut out = [9u8; 10];
        t.decode_four(&src, &mut out).unwrap();
        assert_eq!(out, [0, 1, 1, 1, 0, 0, 1, 1, 1, 0]);
        // The fourth stream must have a byte.
        let short = &src[..src.len() - 1];
        assert_eq!(t.decode_four(short, &mut out), Err(Error::JumpTable));
        // Fewer than 6 literals cannot be split in four.
        assert_eq!(t.decode_four(&src, &mut out[..5]), Err(Error::TooFewLiteralsForFourStreams(5)));
    }
}
