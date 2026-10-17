//! The sequences section of a compressed block: its header, the three
//! FSE tables (literal lengths, offsets, match lengths) in predefined,
//! RLE, FSE coded or repeat mode, and the sequences themselves, decoded
//! from one backward bitstream and executed as they are decoded.

use std::sync::OnceLock;

use crate::Error;
use crate::bits::BackBits;
use crate::error::{Part, Symbol};
use crate::fse::{self, Cell, Distribution};
use crate::literals::Literals;
use crate::output::{Limit, Output};

/// Each literal length code's baseline and the bits added to it.
const LL_BASE: [u32; 36] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 28, 32, 40, 48, 64, 128, 256, 512, 1024,
    2048, 4096, 8192, 16384, 32768, 65536,
];
const LL_BITS: [u8; 36] =
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

/// Each match length code's baseline and the bits added to it.
const ML_BASE: [u32; 53] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 37, 39, 41, 43, 47, 51, 59, 67, 83, 99, 131, 259, 515, 1027, 2051, 4099, 8195, 16387, 32771, 65539,
];
const ML_BITS: [u8; 53] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2,
    3, 3, 4, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
];

/// The predefined distributions, at accuracy logs 6, 5 and 6.
const LL_DEFAULT: [i16; 36] =
    [4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1, -1, -1, -1, -1];
const OF_DEFAULT: [i16; 29] =
    [1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1];
const ML_DEFAULT: [i16; 53] = [
    1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1,
];

impl Symbol {
    /// The last code. Offset codes are the number of bits of the offset;
    /// 31 is the most this decoder takes, as the reference decoder does.
    fn max_code(self) -> u8 {
        match self {
            Symbol::LiteralLength => 35,
            Symbol::Offset => 31,
            Symbol::MatchLength => 52,
        }
    }

    fn max_log(self) -> u32 {
        match self {
            Symbol::Offset => 8,
            _ => 9,
        }
    }

    fn predefined(self) -> Distribution {
        match self {
            Symbol::LiteralLength => Distribution::of(6, &LL_DEFAULT),
            Symbol::Offset => Distribution::of(5, &OF_DEFAULT),
            Symbol::MatchLength => Distribution::of(6, &ML_DEFAULT),
        }
    }

    /// The baseline of `code` and the bits added to it. An offset code
    /// `c` stands for `(1 << c)` plus `c` bits.
    fn code(self, code: u8) -> (u32, u8) {
        let c = usize::from(code);
        match self {
            Symbol::LiteralLength => (LL_BASE[c], LL_BITS[c]),
            Symbol::Offset => (1 << code, code),
            Symbol::MatchLength => (ML_BASE[c], ML_BITS[c]),
        }
    }
}

/// One state of a sequence table, with its code's baseline and extra
/// bits in place of the code, so a sequence needs no second lookup.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct SeqCell {
    base: u32,
    extra: u8,
    bits: u8,
    next: u16,
}

struct SeqTable {
    log: u32,
    /// Whether the table holds one set in this frame, which repeat mode
    /// needs.
    set: bool,
    cells: [SeqCell; 1 << fse::MAX_LOG],
}

impl SeqTable {
    fn new() -> SeqTable {
        SeqTable { log: 0, set: false, cells: [SeqCell::default(); 1 << fse::MAX_LOG] }
    }

    fn fse(&mut self, kind: Symbol, d: &Distribution) {
        let mut cells = [Cell::default(); 1 << fse::MAX_LOG];
        fse::build(d, &mut cells);
        for (out, c) in self.cells.iter_mut().zip(&cells[..1 << d.log]) {
            let (base, extra) = kind.code(c.symbol);
            *out = SeqCell { base, extra, bits: c.bits, next: c.base };
        }
        self.log = d.log;
        self.set = true;
    }

    /// The predefined table of `kind`, built the first time it is asked
    /// for and kept for the life of the process.
    fn predefined(kind: Symbol) -> &'static SeqTable {
        static TABLES: OnceLock<[SeqTable; 3]> = OnceLock::new();
        let tables = TABLES.get_or_init(|| {
            [Symbol::LiteralLength, Symbol::Offset, Symbol::MatchLength].map(|kind| {
                let mut t = SeqTable::new();
                t.fse(kind, &kind.predefined());
                t
            })
        });
        match kind {
            Symbol::LiteralLength => &tables[0],
            Symbol::Offset => &tables[1],
            Symbol::MatchLength => &tables[2],
        }
    }

    /// Copies the table `from` holds into this one, which repeat mode may
    /// then use.
    fn copy(&mut self, from: &SeqTable) {
        let n = 1 << from.log;
        self.cells[..n].copy_from_slice(&from.cells[..n]);
        self.log = from.log;
        self.set = true;
    }

    /// One state, one code, reading no bits.
    fn rle(&mut self, kind: Symbol, code: u8) {
        let (base, extra) = kind.code(code);
        self.cells[0] = SeqCell { base, extra, bits: 0, next: 0 };
        self.log = 0;
        self.set = true;
    }

    #[inline(always)]
    fn cell(&self, state: usize) -> SeqCell {
        self.cells[state & ((1 << fse::MAX_LOG) - 1)]
    }
}

/// The three tables, kept from block to block of a frame for repeat
/// mode.
pub struct Tables {
    ll: SeqTable,
    of: SeqTable,
    ml: SeqTable,
}

impl Tables {
    pub fn new() -> Tables {
        Tables { ll: SeqTable::new(), of: SeqTable::new(), ml: SeqTable::new() }
    }

    /// Forgets the tables, as a new frame starts with none.
    pub fn clear(&mut self) {
        for t in [&mut self.ll, &mut self.of, &mut self.ml] {
            t.set = false;
        }
    }

    /// Reads the sequences section header at the start of `src`, setting
    /// the tables by its modes. Returns the number of sequences and the
    /// header's length. Not inlined: it runs once a block, and inlined
    /// into [`execute`] it made the sequence loop there measurably slower.
    #[inline(never)]
    fn read_header(&mut self, src: &[u8]) -> Result<(usize, usize), Error> {
        let byte = |i: usize| src.get(i).copied().map(usize::from).ok_or(Error::Truncated(Part::SequencesHeader));
        let b0 = byte(0)?;
        let (n, mut at) = match b0 {
            0 => {
                // No sequences: the section ends here, and the tables stay
                // as they were.
                if src.len() != 1 {
                    return Err(Error::BytesAfterNoSequences);
                }
                return Ok((0, 1));
            }
            1..128 => (b0, 1),
            // This form can give 0 too, and the modes still follow.
            128..255 => ((b0 - 128) << 8 | byte(1)?, 2),
            _ => ((byte(1)? | byte(2)? << 8) + 0x7f00, 3),
        };
        let modes = byte(at)?;
        at += 1;
        if modes & 3 != 0 {
            return Err(Error::ReservedModeBits);
        }
        let parts = [(Symbol::LiteralLength, 6), (Symbol::Offset, 4), (Symbol::MatchLength, 2)];
        for (kind, shift) in parts {
            let table = match kind {
                Symbol::LiteralLength => &mut self.ll,
                Symbol::Offset => &mut self.of,
                Symbol::MatchLength => &mut self.ml,
            };
            match modes >> shift & 3 {
                0 => table.copy(SeqTable::predefined(kind)),
                1 => {
                    let code = byte(at)? as u8;
                    at += 1;
                    if code > kind.max_code() {
                        return Err(Error::CodePastMax { symbol: kind, code });
                    }
                    table.rle(kind, code);
                }
                2 => {
                    let (d, used) = Distribution::read(&src[at..], usize::from(kind.max_code()), kind.max_log())?;
                    at += used;
                    table.fse(kind, &d);
                }
                _ => {
                    if !table.set {
                        return Err(Error::NoTableToRepeat(kind));
                    }
                }
            }
        }
        Ok((n, at))
    }
}

/// The offset a sequence's offset value stands for, updating the repeated
/// offsets `reps`, the most recent first. Values 1 to 3 pick a repeated
/// offset, shifted by one when the literal length is 0; larger values
/// are the offset plus 3.
fn resolve(value: usize, ll: usize, reps: &mut [usize; 3]) -> Result<usize, Error> {
    if value > 3 {
        let o = value - 3;
        *reps = [o, reps[0], reps[1]];
        return Ok(o);
    }
    match value - 1 + usize::from(ll == 0) {
        0 => {}
        1 => reps.swap(0, 1),
        2 => *reps = [reps[2], reps[0], reps[1]],
        _ => {
            let o = reps[0].checked_sub(1).filter(|&o| o > 0).ok_or(Error::ZeroOffset)?;
            *reps = [o, reps[0], reps[1]];
        }
    }
    Ok(reps[0])
}

/// A mask of the `n` low bits, `n` at most 56.
#[inline(always)]
fn low(n: u32) -> u64 {
    (1 << n) - 1
}

/// What a block's matches may reach back to: the start of its frame's
/// content in the output, and the frame's window.
pub struct History {
    pub start: usize,
    pub window: u64,
}

/// Decodes the sequences section `src` of a block and executes it: each
/// sequence's literals, then its match, then the literals left after the
/// last sequence.
pub fn execute(
    src: &[u8],
    tables: &mut Tables,
    reps: &mut [usize; 3],
    lits: &Literals,
    history: &History,
    limit: &Limit,
    out: &mut Output,
) -> Result<(), Error> {
    let (n, at) = tables.read_header(src)?;
    let mut used = 0;
    // A count of 0 in two bytes is followed by tables; whatever comes
    // after them is not read.
    if n > 0 {
        let mut b = BackBits::new(&src[at..])?;
        let (ll_t, of_t, ml_t) = (&tables.ll, &tables.of, &tables.ml);
        let mut ll_s = b.read(ll_t.log) as usize;
        let mut of_s = b.read(of_t.log) as usize;
        let mut ml_s = b.read(ml_t.log) as usize;
        if b.overflowed() {
            return Err(Error::StreamNotConsumed);
        }
        for i in 0..n {
            // A reload leaves at least 57 bits: up to 31 offset bits and
            // 16 match length bits, then the literal length's 16 and 26
            // for the three state updates, with a reload between when
            // the extra bits come to more than 31.
            b.reload();
            let (llc, ofc, mlc) = (ll_t.cell(ll_s), of_t.cell(of_s), ml_t.cell(ml_s));
            // The offset's and the match length's extra bits in one read,
            // the offset's first, so highest.
            let (of_n, ml_n) = (u32::from(ofc.extra), u32::from(mlc.extra));
            let both = b.read(of_n + ml_n);
            let value = ofc.base as usize + (both >> ml_n) as usize;
            let ml = mlc.base as usize + (both & low(ml_n)) as usize;
            if of_n + ml_n + u32::from(llc.extra) > 31 {
                b.reload();
            }
            let ll = llc.base as usize + b.read(u32::from(llc.extra)) as usize;
            if i + 1 < n {
                // The three state updates in one read: literal length,
                // match length, offset.
                let (ll_n, ml_n, of_n) = (u32::from(llc.bits), u32::from(mlc.bits), u32::from(ofc.bits));
                let all = b.read(ll_n + ml_n + of_n);
                ll_s = usize::from(llc.next) + (all >> (ml_n + of_n)) as usize;
                ml_s = usize::from(mlc.next) + (all >> of_n & low(ml_n)) as usize;
                of_s = usize::from(ofc.next) + (all & low(of_n)) as usize;
            }
            if b.overflowed() {
                return Err(Error::StreamNotConsumed);
            }
            let offset = resolve(value, ll, reps)?;
            if ll > lits.len - used {
                return Err(Error::LiteralsOverrun);
            }
            let end = out.pos + ll + ml;
            if end > limit.end {
                return Err(limit.error(end));
            }
            out.literals(lits.bytes, used, ll);
            used += ll;
            let available = out.pos - history.start;
            if offset > available {
                return Err(Error::OffsetBeforeStart { offset, available });
            }
            if offset as u64 > history.window {
                return Err(Error::OffsetPastWindow { offset, window: history.window });
            }
            out.copy_match(offset, ml);
        }
        if !b.finished() {
            return Err(Error::StreamNotConsumed);
        }
    }
    let end = out.pos + (lits.len - used);
    if end > limit.end {
        return Err(limit.error(end));
    }
    out.extend(&lits.bytes[used..lits.len]);
    Ok(())
}

#[cfg(test)]
mod tests;
