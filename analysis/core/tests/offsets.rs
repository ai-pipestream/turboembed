//! UTF-16 and code point offsets at the edge, checked against counts made
//! plainly with `encode_utf16` and `chars`.

use turbo_analysis::{
    CodePointOffset, Document, Error, Key, OffsetCursor, OffsetError, OffsetTable, OffsetUnit, Provenance, Span,
    Utf16Offset, View, ViewBuilder, ViewId,
};

/// The UTF-16 and code point offsets of byte `b`, counted from the start.
fn plain(text: &str, b: usize) -> (u32, u32) {
    let head = &text[..b];
    (head.encode_utf16().count() as u32, head.chars().count() as u32)
}

/// Every character boundary of `text`, in bytes, in order.
fn boundaries(text: &str) -> Vec<u32> {
    text.char_indices().map(|(i, _)| i as u32).chain([text.len() as u32]).collect()
}

/// A small deterministic generator (Knuth's MMIX constants), so the long
/// tests need no crate and always see the same text.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }
}

/// Mixed text of one to four byte characters, `n` characters long.
fn mixed(n: usize, seed: u64) -> String {
    const CHARS: [char; 8] = ['a', ' ', '\u{E9}', '\u{3B1}', '\u{20AC}', '\u{4E2D}', '\u{1D11E}', '\u{1F600}'];
    let mut r = Lcg(seed);
    (0..n).map(|_| CHARS[r.below(CHARS.len() as u32) as usize]).collect()
}

#[test]
fn ascii_text_is_the_identity() {
    let text = "The quick brown fox.";
    let t = OffsetTable::build(text);
    assert!(t.is_ascii());
    assert_eq!(t.checkpoints(), 0);
    for b in 0..=text.len() as u32 {
        assert_eq!(t.to_utf16(text, b), Utf16Offset(b));
        assert_eq!(t.to_code_point(text, b), CodePointOffset(b));
        assert_eq!(t.from_utf16(text, Utf16Offset(b)), Ok(b));
        assert_eq!(t.from_code_point(text, CodePointOffset(b)), Ok(b));
    }
}

#[test]
fn one_to_four_byte_characters_at_every_boundary() {
    // a (1 byte), é (2), € (3), 𝄞 (4, a surrogate pair in UTF-16), z.
    let text = "a\u{E9}\u{20AC}\u{1D11E}z";
    let t = OffsetTable::build(text);
    assert!(!t.is_ascii());
    let expect: [(u32, u32, u32); 6] = [(0, 0, 0), (1, 1, 1), (3, 2, 2), (6, 3, 3), (10, 5, 4), (11, 6, 5)];
    assert_eq!(boundaries(text), expect.map(|e| e.0));
    for (b, u, c) in expect {
        assert_eq!(t.to_utf16(text, b), Utf16Offset(u), "byte {b}");
        assert_eq!(t.to_code_point(text, b), CodePointOffset(c), "byte {b}");
        assert_eq!(t.from_utf16(text, Utf16Offset(u)), Ok(b), "UTF-16 {u}");
        assert_eq!(t.from_code_point(text, CodePointOffset(c)), Ok(b), "code point {c}");
    }
    assert_eq!(t.utf16_len(), Utf16Offset(6));
    assert_eq!(t.code_point_len(), CodePointOffset(5));
}

#[test]
fn an_offset_inside_a_surrogate_pair_is_refused() {
    let text = "x\u{1F600}y";
    let t = OffsetTable::build(text);
    // UTF-16: x is 0, the emoji is 1..3, y is 3.
    let e = t.from_utf16(text, Utf16Offset(2)).unwrap_err();
    assert_eq!(e, Error::Offset { unit: OffsetUnit::Utf16, at: 2, reason: OffsetError::InsideCharacter });
    assert!(e.to_string().contains("UTF-16 offset 2"), "{e}");
    assert_eq!(t.from_utf16(text, Utf16Offset(1)), Ok(1));
    assert_eq!(t.from_utf16(text, Utf16Offset(3)), Ok(5));
    // In code points the emoji is one character, so every offset is a place.
    assert_eq!(t.from_code_point(text, CodePointOffset(2)), Ok(5));
}

#[test]
fn offsets_past_the_end_are_refused() {
    for text in ["abc", "a\u{1D11E}"] {
        let t = OffsetTable::build(text);
        let (u, c) = plain(text, text.len());
        assert_eq!(t.from_utf16(text, Utf16Offset(u)), Ok(text.len() as u32));
        assert_eq!(t.from_code_point(text, CodePointOffset(c)), Ok(text.len() as u32));
        let e = t.from_utf16(text, Utf16Offset(u + 1)).unwrap_err();
        assert_eq!(e, Error::Offset { unit: OffsetUnit::Utf16, at: u + 1, reason: OffsetError::PastEnd { len: u } });
        let e = t.from_code_point(text, CodePointOffset(c + 1)).unwrap_err();
        assert_eq!(
            e,
            Error::Offset { unit: OffsetUnit::CodePoint, at: c + 1, reason: OffsetError::PastEnd { len: c } }
        );
        assert!(e.to_string().contains("past the end"), "{e}");
    }
}

#[test]
fn an_empty_text() {
    let t = OffsetTable::build("");
    assert!(t.is_empty());
    assert_eq!(t.to_utf16("", 0), Utf16Offset(0));
    assert_eq!(t.to_code_point("", 0), CodePointOffset(0));
    assert_eq!(t.from_utf16("", Utf16Offset(0)), Ok(0));
    assert_eq!(t.from_code_point("", CodePointOffset(0)), Ok(0));
    assert!(t.from_utf16("", Utf16Offset(1)).is_err());
    assert_eq!(t.span_to_utf16("", Span::new(0, 0)), (Utf16Offset(0), Utf16Offset(0)));
}

#[test]
fn a_long_text_across_many_checkpoints() {
    let text = mixed(4000, 7);
    assert!(text.len() > 9000, "{} bytes", text.len());
    let t = OffsetTable::build(&text);
    assert!(t.checkpoints() > 30);
    let bounds = boundaries(&text);
    let (u_len, c_len) = plain(&text, text.len());
    assert_eq!((t.utf16_len(), t.code_point_len()), (Utf16Offset(u_len), CodePointOffset(c_len)));
    let mut r = Lcg(42);
    for _ in 0..2000 {
        let b = bounds[r.below(bounds.len() as u32) as usize];
        let (u, c) = plain(&text, b as usize);
        assert_eq!(t.to_utf16(&text, b), Utf16Offset(u), "byte {b}");
        assert_eq!(t.to_code_point(&text, b), CodePointOffset(c), "byte {b}");
        assert_eq!(t.from_utf16(&text, Utf16Offset(u)), Ok(b), "UTF-16 {u}");
        assert_eq!(t.from_code_point(&text, CodePointOffset(c)), Ok(b), "code point {c}");
    }
    // Every UTF-16 offset either is a boundary or splits a surrogate pair,
    // and the split ones are exactly the four byte characters.
    let mut split = 0;
    for u in 0..=u_len {
        match t.from_utf16(&text, Utf16Offset(u)) {
            Ok(b) => assert_eq!(plain(&text, b as usize).0, u),
            Err(Error::Offset { reason: OffsetError::InsideCharacter, at, .. }) => {
                assert_eq!(at, u);
                split += 1;
            }
            Err(e) => panic!("{e}"),
        }
    }
    assert_eq!(split, text.chars().filter(|c| c.len_utf8() == 4).count());
}

#[test]
fn the_cursor_matches_the_point_conversions() {
    let text = mixed(2500, 99);
    let t = OffsetTable::build(&text);
    let bounds = boundaries(&text);
    // A sorted column with repeats and long gaps, so the cursor both walks
    // and jumps to checkpoints.
    let mut r = Lcg(5);
    let mut column: Vec<u32> = (0..600).map(|_| bounds[r.below(bounds.len() as u32) as usize]).collect();
    column.sort_unstable();
    let mut cursor = OffsetCursor::new(&t, &text);
    for &b in &column {
        assert_eq!(cursor.to_utf16(b), t.to_utf16(&text, b), "byte {b}");
        assert_eq!(cursor.to_code_point(b), t.to_code_point(&text, b), "byte {b}");
    }
    let mut out = Vec::new();
    t.column_to_utf16(&text, &column, &mut out);
    assert_eq!(out, column.iter().map(|&b| t.to_utf16(&text, b)).collect::<Vec<_>>());
    let mut out = Vec::new();
    t.column_to_code_point(&text, &column, &mut out);
    assert_eq!(out, column.iter().map(|&b| t.to_code_point(&text, b)).collect::<Vec<_>>());
    // A column out of order (the ends of overlapping entities) still
    // converts, offset by offset.
    let shuffled: Vec<u32> = column.iter().rev().copied().collect();
    let mut out = Vec::new();
    t.column_to_utf16(&text, &shuffled, &mut out);
    assert_eq!(out, shuffled.iter().map(|&b| t.to_utf16(&text, b)).collect::<Vec<_>>());
    // Spans, through the same cursor.
    let (a, b) = (column[100], column[500]);
    assert_eq!(t.span_to_utf16(&text, Span::new(a, b)), (t.to_utf16(&text, a), t.to_utf16(&text, b)));
    assert_eq!(t.span_to_code_point(&text, Span::new(a, b)), (t.to_code_point(&text, a), t.to_code_point(&text, b)));
}

#[test]
fn a_derived_view_has_its_own_table() {
    // The source is ASCII; the derived view writes a ligature and an
    // emoji, so its table differs.
    let src = "fine :) day";
    let mut doc = Document::new(src);
    let (out, alignment) = {
        let mut b = ViewBuilder::new(src);
        b.replace(2, "\u{FB01}");
        b.keep(5);
        b.replace(7, "\u{1F642}");
        b.finish()
    };
    assert_eq!(out, "\u{FB01}ne \u{1F642} day");
    let view = View::derived(doc.get::<View>().unwrap(), out.clone(), alignment);
    doc.add(&Key::named("folded"), view, Provenance::by("t")).unwrap();
    let folded = ViewId(1);
    assert!(doc.offsets(ViewId::SOURCE).is_ascii());
    let t = doc.offsets(folded);
    assert!(!t.is_ascii());
    assert!(std::ptr::eq(t, doc.offsets(folded)), "built once and kept");
    // "day" in the view: bytes 11..14, UTF-16 7..10, code points 6..9.
    let day = Span::new(11, 14);
    assert_eq!(&doc.text(folded)[day.range()], "day");
    assert_eq!(t.span_to_utf16(doc.text(folded), day), (Utf16Offset(7), Utf16Offset(10)));
    assert_eq!(t.span_to_code_point(doc.text(folded), day), (CodePointOffset(6), CodePointOffset(9)));
    // The same words in the source are at the same offsets in all units.
    let src_day = doc.to_source(folded, day);
    assert_eq!(doc.offsets(ViewId::SOURCE).span_to_utf16(src, src_day), (Utf16Offset(8), Utf16Offset(11)));
}
