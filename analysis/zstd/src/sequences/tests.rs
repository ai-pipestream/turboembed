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

#[test]
fn truncated_sequence_bitstream_stops_before_executing_phantom_sequences() {
    // Predefined tables need 6 + 5 + 6 = 17 bits for the initial states,
    // but a 1-byte stream `[1]` has 0 unread bits under its end mark.
    let mut tables = Tables::new();
    let mut reps = [1, 4, 8];
    let lits = Literals { bytes: b"abcd", len: 4 };
    let history = History { start: 0, window: 1024 };
    let limit = Limit::new(0, 64, 64, None, 64);
    let mut buf = [0u8; 64];
    let mut out = Output { buf: &mut buf, pos: 0 };
    assert_eq!(
        execute(&[1, 0x00, 1], &mut tables, &mut reps, &lits, &history, &limit, &mut out),
        Err(Error::StreamNotConsumed)
    );
    assert_eq!(out.pos, 0);
}
