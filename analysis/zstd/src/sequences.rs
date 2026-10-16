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
mod tests {
    use super::*;

    #[test]
    fn repeated_offsets_as_the_formats_table() {
        // The format's table: offset value, literal length, then the
        // three repeated offsets after the sequence.
        let rows = [
            (1114, 11, [1111, 1, 4]),
            (1, 22, [1111, 1, 4]),
            (2225, 22, [2222, 1111, 1]),
            (1114, 111, [1111, 2222, 1111]),
            (3336, 33, [3333, 1111, 2222]),
            (2, 22, [1111, 3333, 2222]),
            (3, 33, [2222, 1111, 3333]),
            (3, 0, [2221, 2222, 1111]),
            (1, 0, [2222, 2221, 1111]),
        ];
        let mut reps = [1, 4, 8];
        for (value, ll, after) in rows {
            let offset = resolve(value, ll, &mut reps).unwrap();
            assert_eq!(reps, after, "value {value} literal length {ll}");
            assert_eq!(offset, reps[0]);
        }
        // With a literal length of 0, value 2 picks the third offset.
        let mut reps = [10, 20, 30];
        assert_eq!(resolve(2, 0, &mut reps), Ok(30));
        assert_eq!(reps, [30, 10, 20]);
        // Value 3 with a literal length of 0 is the first offset less 1,
        // which may not be 0.
        let mut reps = [1, 20, 30];
        assert_eq!(resolve(3, 0, &mut reps), Err(Error::ZeroOffset));
    }

    /// Appendix A of the format: each state's symbol, bits and base.
    #[test]
    fn predefined_tables_as_appendix_a() {
        #[rustfmt::skip]
        let ll = [
            (0, 4, 0), (0, 4, 16), (1, 5, 32), (3, 5, 0), (4, 5, 0), (6, 5, 0), (7, 5, 0), (9, 5, 0), (10, 5, 0),
            (12, 5, 0), (14, 6, 0), (16, 5, 0), (18, 5, 0), (19, 5, 0), (21, 5, 0), (22, 5, 0), (24, 5, 0),
            (25, 5, 32), (26, 5, 0), (27, 6, 0), (29, 6, 0), (31, 6, 0), (0, 4, 32), (1, 4, 0), (2, 5, 0),
            (4, 5, 32), (5, 5, 0), (7, 5, 32), (8, 5, 0), (10, 5, 32), (11, 5, 0), (13, 6, 0), (16, 5, 32),
            (17, 5, 0), (19, 5, 32), (20, 5, 0), (22, 5, 32), (23, 5, 0), (25, 4, 0), (25, 4, 16), (26, 5, 32),
            (28, 6, 0), (30, 6, 0), (0, 4, 48), (1, 4, 16), (2, 5, 32), (3, 5, 32), (5, 5, 32), (6, 5, 32),
            (8, 5, 32), (9, 5, 32), (11, 5, 32), (12, 5, 32), (15, 6, 0), (17, 5, 32), (18, 5, 32), (20, 5, 32),
            (21, 5, 32), (23, 5, 32), (24, 5, 32), (35, 6, 0), (34, 6, 0), (33, 6, 0), (32, 6, 0),
        ];
        #[rustfmt::skip]
        let ml = [
            (0, 6, 0), (1, 4, 0), (2, 5, 32), (3, 5, 0), (5, 5, 0), (6, 5, 0), (8, 5, 0), (10, 6, 0), (13, 6, 0),
            (16, 6, 0), (19, 6, 0), (22, 6, 0), (25, 6, 0), (28, 6, 0), (31, 6, 0), (33, 6, 0), (35, 6, 0),
            (37, 6, 0), (39, 6, 0), (41, 6, 0), (43, 6, 0), (45, 6, 0), (1, 4, 16), (2, 4, 0), (3, 5, 32),
            (4, 5, 0), (6, 5, 32), (7, 5, 0), (9, 6, 0), (12, 6, 0), (15, 6, 0), (18, 6, 0), (21, 6, 0),
            (24, 6, 0), (27, 6, 0), (30, 6, 0), (32, 6, 0), (34, 6, 0), (36, 6, 0), (38, 6, 0), (40, 6, 0),
            (42, 6, 0), (44, 6, 0), (1, 4, 32), (1, 4, 48), (2, 4, 16), (4, 5, 32), (5, 5, 32), (7, 5, 32),
            (8, 5, 32), (11, 6, 0), (14, 6, 0), (17, 6, 0), (20, 6, 0), (23, 6, 0), (26, 6, 0), (29, 6, 0),
            (52, 6, 0), (51, 6, 0), (50, 6, 0), (49, 6, 0), (48, 6, 0), (47, 6, 0), (46, 6, 0),
        ];
        #[rustfmt::skip]
        let of = [
            (0, 5, 0), (6, 4, 0), (9, 5, 0), (15, 5, 0), (21, 5, 0), (3, 5, 0), (7, 4, 0), (12, 5, 0), (18, 5, 0),
            (23, 5, 0), (5, 5, 0), (8, 4, 0), (14, 5, 0), (20, 5, 0), (2, 5, 0), (7, 4, 16), (11, 5, 0),
            (17, 5, 0), (22, 5, 0), (4, 5, 0), (8, 4, 16), (13, 5, 0), (19, 5, 0), (1, 5, 0), (6, 4, 16),
            (10, 5, 0), (16, 5, 0), (28, 5, 0), (27, 5, 0), (26, 5, 0), (25, 5, 0), (24, 5, 0),
        ];
        let check = |kind: Symbol, want: &[(u8, u8, u16)]| {
            let d = kind.predefined();
            assert_eq!(1 << d.log, want.len());
            let mut cells = [Cell::default(); 1 << fse::MAX_LOG];
            fse::build(&d, &mut cells);
            let got: Vec<_> = cells[..want.len()].iter().map(|c| (c.symbol, c.bits, c.base)).collect();
            assert_eq!(got, want, "{kind}");
        };
        check(Symbol::LiteralLength, &ll);
        check(Symbol::MatchLength, &ml);
        check(Symbol::Offset, &of);
    }

    /// The predefined tables built once are the ones built from their
    /// distributions, and copying one over a larger table takes its log.
    #[test]
    fn predefined_tables_are_kept_as_built() {
        for kind in [Symbol::LiteralLength, Symbol::Offset, Symbol::MatchLength] {
            let mut built = SeqTable::new();
            built.fse(kind, &kind.predefined());
            let kept = SeqTable::predefined(kind);
            assert_eq!(kept.log, built.log, "{kind}");
            assert_eq!(kept.cells[..1 << kept.log], built.cells[..1 << built.log], "{kind}");
            let mut t = SeqTable::new();
            t.fse(kind, &Distribution::of(9, &[256, 256]));
            t.copy(kept);
            assert_eq!((t.log, t.set), (kept.log, true), "{kind}");
            assert_eq!(t.cells[..1 << t.log], kept.cells[..1 << kept.log], "{kind}");
        }
    }

    #[test]
    fn codes_as_the_formats_tables() {
        // Literal lengths 0 to 15 are their own code; then baselines and
        // bits from the format's tables, ending at 65536 + 16 bits.
        for c in 0..16 {
            assert_eq!(Symbol::LiteralLength.code(c), (u32::from(c), 0));
        }
        assert_eq!(Symbol::LiteralLength.code(16), (16, 1));
        assert_eq!(Symbol::LiteralLength.code(24), (48, 4));
        assert_eq!(Symbol::LiteralLength.code(25), (64, 6));
        assert_eq!(Symbol::LiteralLength.code(35), (65536, 16));
        // Match lengths 3 to 34 for codes 0 to 31.
        for c in 0..32 {
            assert_eq!(Symbol::MatchLength.code(c), (u32::from(c) + 3, 0));
        }
        assert_eq!(Symbol::MatchLength.code(32), (35, 1));
        assert_eq!(Symbol::MatchLength.code(42), (99, 5));
        assert_eq!(Symbol::MatchLength.code(43), (131, 7));
        assert_eq!(Symbol::MatchLength.code(52), (65539, 16));
        // Each code's range ends where the next one's starts.
        for (base, bits) in [(&LL_BASE[..], &LL_BITS[..]), (&ML_BASE[..], &ML_BITS[..])] {
            for c in 0..base.len() - 1 {
                assert_eq!(base[c] + (1 << bits[c]), base[c + 1], "code {c}");
            }
        }
        // Offset code c: 2^c plus c bits.
        assert_eq!(Symbol::Offset.code(0), (1, 0));
        assert_eq!(Symbol::Offset.code(31), (1 << 31, 31));
    }

    #[test]
    fn header_forms_and_modes() {
        let mut t = Tables::new();
        // One byte: 0 sequences, nothing after it.
        assert_eq!(t.read_header(&[0]), Ok((0, 1)));
        assert_eq!(t.read_header(&[0, 0]), Err(Error::BytesAfterNoSequences));
        // Two bytes: (0x81 - 0x80) << 8 + 2 = 258; three bytes:
        // 0x7f00 + 0x0102. All RLE: codes 1, 2, 3.
        assert_eq!(t.read_header(&[0x81, 2, 0x54, 1, 2, 3, 1]), Ok((258, 6)));
        assert_eq!(t.read_header(&[0xff, 2, 1, 0x54, 1, 2, 3, 1]), Ok((0x8002, 7)));
        assert_eq!(t.ll.cells[0].base, 1);
        assert_eq!((t.of.cells[0].base, t.of.cells[0].extra), (4, 2));
        assert_eq!(t.ml.cells[0].base, 6);
        // Reserved bits, and an RLE code past the last.
        assert_eq!(t.read_header(&[1, 0x55]), Err(Error::ReservedModeBits));
        assert_eq!(t.read_header(&[1, 0x40, 36]), Err(Error::CodePastMax { symbol: Symbol::LiteralLength, code: 36 }));
        assert_eq!(t.read_header(&[1, 0x10, 32]), Err(Error::CodePastMax { symbol: Symbol::Offset, code: 32 }));
        assert_eq!(t.read_header(&[1, 0x04, 53]), Err(Error::CodePastMax { symbol: Symbol::MatchLength, code: 53 }));
        // Repeat mode needs a table set in the frame.
        t.clear();
        assert_eq!(t.read_header(&[1, 0xfc]), Err(Error::NoTableToRepeat(Symbol::LiteralLength)));
        assert_eq!(t.read_header(&[1, 0x3c]), Err(Error::NoTableToRepeat(Symbol::Offset)));
        assert_eq!(t.read_header(&[1, 0x0c]), Err(Error::NoTableToRepeat(Symbol::MatchLength)));
        // Predefined then repeat: the predefined tables are kept.
        assert_eq!(t.read_header(&[1, 0x00, 1]), Ok((1, 2)));
        assert_eq!(t.read_header(&[1, 0xfc, 1]), Ok((1, 2)));
        assert_eq!((t.ll.log, t.of.log, t.ml.log), (6, 5, 6));
        // A count of 0 in two bytes is followed by the modes and tables,
        // which replace the ones before.
        t.clear();
        assert_eq!(t.read_header(&[0x80, 0, 0x54, 7, 5, 9]), Ok((0, 6)));
        assert_eq!((t.ll.cells[0].base, t.of.cells[0].base, t.ml.cells[0].base), (7, 32, 12));
        assert!(t.ll.set && t.of.set && t.ml.set);
        assert_eq!(t.read_header(&[0x80, 0]), Err(Error::Truncated(Part::SequencesHeader)));
        assert_eq!(t.read_header(&[0x80, 0, 0x55]), Err(Error::ReservedModeBits));
        // Truncated inside the header.
        assert_eq!(t.read_header(&[0x81]), Err(Error::Truncated(Part::SequencesHeader)));
        assert_eq!(t.read_header(&[1]), Err(Error::Truncated(Part::SequencesHeader)));
        assert_eq!(t.read_header(&[1, 0x40]), Err(Error::Truncated(Part::SequencesHeader)));
    }
}
