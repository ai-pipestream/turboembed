//! Offset conversion around stride and surrogate boundaries, checked against
//! counts made plainly with `char_indices`, `len_utf16`, `encode_utf16` and
//! `chars`.

use turbo_analysis::{
    CodePointOffset, Document, Error, Key, OffsetCursor, OffsetError, OffsetTable, OffsetUnit, Provenance, Span,
    Utf16Offset, View, ViewBuilder, ViewId,
};

use turbo_analysis::offsets::STRIDE;

const STRIDE_BYTES: usize = STRIDE as usize;

/// Characters of 1, 2, 3 and 4 bytes.
const W1: char = 'a';
const W2: char = '\u{E9}';
const W3: char = '\u{20AC}';
const W4: char = '\u{1D11E}';

/// Every character boundary of `text` as (byte, UTF-16, code point), counted
/// one character at a time.
fn boundaries(text: &str) -> Vec<(u32, u32, u32)> {
    let mut out = vec![(0, 0, 0)];
    let (mut u, mut c) = (0u32, 0u32);
    for (i, ch) in text.char_indices() {
        u += ch.len_utf16() as u32;
        c += 1;
        out.push(((i + ch.len_utf8()) as u32, u, c));
    }
    out
}

/// The same counts by re-encoding the head of the text.
fn plain(text: &str, b: usize) -> (u32, u32) {
    let head = &text[..b];
    (head.encode_utf16().count() as u32, head.chars().count() as u32)
}

fn offset_error(unit: OffsetUnit, at: u32, reason: OffsetError) -> Error {
    Error::Offset { unit, at, reason }
}

/// The number of checkpoints the module describes: one for the first
/// character boundary at or after each multiple of the stride below the
/// text's length, where that boundary is itself before the end and not the
/// same place as the previous checkpoint.
fn expected_checkpoints(text: &str) -> usize {
    let mut places: Vec<usize> = (0..text.len())
        .step_by(STRIDE_BYTES)
        .map(|m| (m..=text.len()).find(|&b| text.is_char_boundary(b)).unwrap())
        .filter(|&b| b < text.len())
        .collect();
    places.dedup();
    places.len()
}

/// Checks all four conversions at every boundary, the lengths, both
/// PastEnd cases, every UTF-16 index inside a surrogate pair, and the cursor
/// over all boundaries, against the plain computation.
fn check_all(text: &str) {
    let t = OffsetTable::build(text);
    let all = boundaries(text);
    let &(len, ulen, clen) = all.last().unwrap();
    assert_eq!(t.len(), len);
    assert_eq!(t.utf16_len(), Utf16Offset(ulen));
    assert_eq!(t.code_point_len(), CodePointOffset(clen));
    assert_eq!(t.is_ascii(), text.is_ascii());
    if text.is_ascii() {
        assert_eq!(t.checkpoints(), 0);
    } else {
        assert_eq!(t.checkpoints(), expected_checkpoints(text));
    }

    let mut cursor = OffsetCursor::new(&t, text);
    let mut cursor_c = OffsetCursor::new(&t, text);
    for &(b, u, c) in &all {
        assert_eq!(plain(text, b as usize), (u, c));
        assert_eq!(t.to_utf16(text, b), Utf16Offset(u), "to_utf16 at byte {b}");
        assert_eq!(t.to_code_point(text, b), CodePointOffset(c), "to_code_point at byte {b}");
        assert_eq!(t.from_utf16(text, Utf16Offset(u)), Ok(b), "from_utf16 at {u}");
        assert_eq!(t.from_code_point(text, CodePointOffset(c)), Ok(b), "from_code_point at {c}");
        assert_eq!(cursor.to_utf16(b), Utf16Offset(u), "cursor utf16 at byte {b}");
        assert_eq!(cursor_c.to_code_point(b), CodePointOffset(c), "cursor code point at byte {b}");
    }

    // Every UTF-16 index that is not a boundary is inside a surrogate pair.
    let on_boundary: std::collections::HashSet<u32> = all.iter().map(|x| x.1).collect();
    for u in 0..=ulen {
        if !on_boundary.contains(&u) {
            assert_eq!(
                t.from_utf16(text, Utf16Offset(u)),
                Err(offset_error(OffsetUnit::Utf16, u, OffsetError::InsideCharacter)),
                "UTF-16 index {u}"
            );
        }
    }

    assert_eq!(t.from_utf16(text, Utf16Offset(ulen)), Ok(len));
    assert_eq!(t.from_code_point(text, CodePointOffset(clen)), Ok(len));
    assert_eq!(
        t.from_utf16(text, Utf16Offset(ulen + 1)),
        Err(offset_error(OffsetUnit::Utf16, ulen + 1, OffsetError::PastEnd { len: ulen }))
    );
    assert_eq!(
        t.from_code_point(text, CodePointOffset(clen + 1)),
        Err(offset_error(OffsetUnit::CodePoint, clen + 1, OffsetError::PastEnd { len: clen }))
    );

    // The whole column in one call, and the whole text as one span.
    let bytes: Vec<u32> = all.iter().map(|x| x.0).collect();
    let mut out16 = Vec::new();
    let mut outc = Vec::new();
    t.column_to_utf16(text, &bytes, &mut out16);
    t.column_to_code_point(text, &bytes, &mut outc);
    assert_eq!(out16, all.iter().map(|x| Utf16Offset(x.1)).collect::<Vec<_>>());
    assert_eq!(outc, all.iter().map(|x| CodePointOffset(x.2)).collect::<Vec<_>>());
    let whole = Span::new(0, len);
    assert_eq!(t.span_to_utf16(text, whole), (Utf16Offset(0), Utf16Offset(ulen)));
    assert_eq!(t.span_to_code_point(text, whole), (CodePointOffset(0), CodePointOffset(clen)));
}

fn ch(width: usize) -> char {
    match width {
        1 => W1,
        2 => W2,
        3 => W3,
        4 => W4,
        _ => unreachable!(),
    }
}

/// `start` ASCII bytes, then one character of `width` bytes, then ASCII up to
/// `total` bytes (or no padding when the character already reaches it).
fn char_at(start: usize, width: usize, total: usize) -> String {
    let mut s = "a".repeat(start);
    s.push(ch(width));
    let used = start + width;
    s.push_str(&"b".repeat(total.saturating_sub(used)));
    s
}

#[test]
fn stride_is_256_bytes() {
    assert_eq!(STRIDE, 256);
}

#[test]
fn character_straddling_every_stride_multiple() {
    for m in [256usize, 512, 768, 1024] {
        for width in 2..=4 {
            // The character starts from 5 bytes before the multiple to 1
            // after, so it ends before, on, and after the multiple and
            // straddles it by 1, 2 and 3 bytes where the width allows.
            for start in m - 5..=m + 1 {
                check_all(&char_at(start, width, m + 8));
                // With nothing after the character as well.
                check_all(&char_at(start, width, 0));
            }
        }
    }
}

#[test]
fn conversions_right_around_a_straddling_character() {
    for width in 2..=4usize {
        for k in 1..width {
            // The character starts k bytes before byte 256, so byte 256 is
            // inside it and the next boundary is width - k bytes after.
            let start = STRIDE_BYTES - k;
            let text = char_at(start, width, 600);
            let t = OffsetTable::build(&text);
            let extra_u = if width == 4 { 2 } else { 1 };
            let end = (start + width) as u32;
            assert_eq!(t.to_utf16(&text, start as u32), Utf16Offset(start as u32));
            assert_eq!(t.to_code_point(&text, start as u32), CodePointOffset(start as u32));
            assert_eq!(t.to_utf16(&text, end), Utf16Offset(start as u32 + extra_u));
            assert_eq!(t.to_code_point(&text, end), CodePointOffset(start as u32 + 1));
            assert_eq!(t.from_utf16(&text, Utf16Offset(start as u32)), Ok(start as u32));
            assert_eq!(t.from_utf16(&text, Utf16Offset(start as u32 + extra_u)), Ok(end));
            assert_eq!(t.from_code_point(&text, CodePointOffset(start as u32)), Ok(start as u32));
            assert_eq!(t.from_code_point(&text, CodePointOffset(start as u32 + 1)), Ok(end));
            if width == 4 {
                assert_eq!(
                    t.from_utf16(&text, Utf16Offset(start as u32 + 1)),
                    Err(offset_error(OffsetUnit::Utf16, start as u32 + 1, OffsetError::InsideCharacter))
                );
            }
        }
    }
}

#[test]
fn repeated_character_of_each_width_around_stride_multiples() {
    for width in 1..=4usize {
        let unit = ch(width).to_string();
        for n in 1..=(1300 / width) {
            // Only lengths within 5 bytes of a multiple of the stride, to
            // keep the run short while covering every boundary there.
            let bytes = n * width;
            let r = bytes % STRIDE_BYTES;
            if r <= 5 || r >= STRIDE_BYTES - 5 {
                check_all(&unit.repeat(n));
            }
        }
    }
}

#[test]
fn texts_of_exact_sizes() {
    for size in [255usize, 256, 257, 512, 513] {
        check_all(&"a".repeat(size));
        for width in 2..=4usize {
            // Non-ASCII text of exactly `size` bytes: a run of the wide
            // character, then ASCII to fill.
            let n = size / width;
            let mut s = ch(width).to_string().repeat(n);
            s.push_str(&"z".repeat(size - n * width));
            assert_eq!(s.len(), size);
            check_all(&s);
            // The wide character first, then ASCII.
            let mut s = ch(width).to_string();
            s.push_str(&"z".repeat(size - width));
            assert_eq!(s.len(), size);
            check_all(&s);
            // ASCII, then the wide character last.
            let mut s = "z".repeat(size - width);
            s.push(ch(width));
            assert_eq!(s.len(), size);
            check_all(&s);
        }
    }
}

#[test]
fn non_ascii_only_in_the_last_character() {
    for width in 2..=4usize {
        for n in [0usize, 1, 255, 256, 257, 700] {
            let mut s = "x".repeat(n);
            s.push(ch(width));
            let t = OffsetTable::build(&s);
            assert!(!t.is_ascii());
            check_all(&s);
        }
    }
}

#[test]
fn non_ascii_only_in_the_first_character() {
    for width in 2..=4usize {
        for n in [0usize, 1, 255, 256, 257, 700] {
            let mut s = ch(width).to_string();
            s.push_str(&"x".repeat(n));
            let t = OffsetTable::build(&s);
            assert!(!t.is_ascii());
            check_all(&s);
        }
    }
}

#[test]
fn empty_text() {
    check_all("");
    let t = OffsetTable::build("");
    assert!(t.is_empty());
    assert_eq!(
        t.from_utf16("", Utf16Offset(1)),
        Err(offset_error(OffsetUnit::Utf16, 1, OffsetError::PastEnd { len: 0 }))
    );
}

#[test]
fn from_utf16_at_every_index_of_four_byte_characters() {
    // 200 characters: 800 bytes, 400 UTF-16 units, over three strides.
    let text = W4.to_string().repeat(200);
    let t = OffsetTable::build(&text);
    for u in 0..=400u32 {
        let got = t.from_utf16(&text, Utf16Offset(u));
        if u % 2 == 0 {
            assert_eq!(got, Ok(u * 2), "UTF-16 index {u}");
        } else {
            assert_eq!(
                got,
                Err(Error::Offset { unit: OffsetUnit::Utf16, at: u, reason: OffsetError::InsideCharacter }),
                "UTF-16 index {u}"
            );
        }
    }
    assert_eq!(
        t.from_utf16(&text, Utf16Offset(401)),
        Err(Error::Offset { unit: OffsetUnit::Utf16, at: 401, reason: OffsetError::PastEnd { len: 400 } })
    );
    for c in 0..=200u32 {
        assert_eq!(t.from_code_point(&text, CodePointOffset(c)), Ok(c * 4));
    }
}

#[test]
fn from_utf16_and_from_code_point_at_length_and_past_it() {
    for text in ["", "abc", "a\u{E9}\u{1F600}", &"\u{20AC}".repeat(300), &"a".repeat(256), &"\u{E9}".repeat(256)] {
        let t = OffsetTable::build(text);
        let (u, c) = plain(text, text.len());
        assert_eq!(t.from_utf16(text, Utf16Offset(u)), Ok(text.len() as u32));
        assert_eq!(t.from_code_point(text, CodePointOffset(c)), Ok(text.len() as u32));
        assert_eq!(
            t.from_utf16(text, Utf16Offset(u + 1)),
            Err(offset_error(OffsetUnit::Utf16, u + 1, OffsetError::PastEnd { len: u }))
        );
        assert_eq!(
            t.from_code_point(text, CodePointOffset(c + 1)),
            Err(offset_error(OffsetUnit::CodePoint, c + 1, OffsetError::PastEnd { len: c }))
        );
        assert_eq!(
            t.from_utf16(text, Utf16Offset(u32::MAX)),
            Err(offset_error(OffsetUnit::Utf16, u32::MAX, OffsetError::PastEnd { len: u }))
        );
        assert_eq!(
            t.from_code_point(text, CodePointOffset(u32::MAX)),
            Err(offset_error(OffsetUnit::CodePoint, u32::MAX, OffsetError::PastEnd { len: c }))
        );
    }
}

#[test]
fn cursor_given_the_same_offset_twice() {
    let text = format!("{}{}{}", "a".repeat(300), W4, "\u{E9}".repeat(300));
    let t = OffsetTable::build(&text);
    for &(b, u, c) in &boundaries(&text) {
        let mut cur = OffsetCursor::new(&t, &text);
        assert_eq!(cur.to_utf16(b), Utf16Offset(u));
        assert_eq!(cur.to_utf16(b), Utf16Offset(u));
        let mut cur = OffsetCursor::new(&t, &text);
        assert_eq!(cur.to_code_point(b), CodePointOffset(c));
        assert_eq!(cur.to_code_point(b), CodePointOffset(c));
        // Mixed units on one cursor, same offset.
        let mut cur = OffsetCursor::new(&t, &text);
        assert_eq!(cur.to_utf16(b), Utf16Offset(u));
        assert_eq!(cur.to_code_point(b), CodePointOffset(c));
    }
}

/// A long text of mixed widths, about 2 KiB.
fn long_mixed() -> String {
    let pattern = [W1, W2, W3, W4, W1, W1, W3, W2, W4];
    (0..700).map(|i| pattern[i % pattern.len()]).collect()
}

#[test]
fn cursor_over_every_boundary_of_a_long_mixed_text() {
    let text = long_mixed();
    assert!(text.len() > 4 * STRIDE_BYTES);
    check_all(&text);
}

#[test]
fn cursor_jumping_over_several_checkpoints_at_once() {
    let text = long_mixed();
    let t = OffsetTable::build(&text);
    let all = boundaries(&text);
    // Pick the first boundary at or after each of these byte positions.
    let wanted = [0usize, 1, 3 * STRIDE_BYTES + 1, 3 * STRIDE_BYTES + 2, 6 * STRIDE_BYTES, text.len()];
    let picks: Vec<(u32, u32, u32)> = wanted
        .iter()
        .filter(|&&w| w <= text.len())
        .map(|&w| *all.iter().find(|x| x.0 as usize >= w).unwrap())
        .collect();
    assert!(picks.windows(2).all(|w| w[0].0 <= w[1].0));
    let mut a = OffsetCursor::new(&t, &text);
    let mut b = OffsetCursor::new(&t, &text);
    for &(byte, u, c) in &picks {
        assert_eq!(a.to_utf16(byte), Utf16Offset(u));
        assert_eq!(b.to_code_point(byte), CodePointOffset(c));
    }
    // A fresh cursor whose first offset skips every checkpoint but the last.
    let &(byte, u, c) = all.last().unwrap();
    assert_eq!(OffsetCursor::new(&t, &text).to_utf16(byte), Utf16Offset(u));
    assert_eq!(OffsetCursor::new(&t, &text).to_code_point(byte), CodePointOffset(c));
}

#[test]
fn cursor_with_a_single_offset_at_the_very_end() {
    for text in [long_mixed(), "a".repeat(1000), format!("{}{}", "a".repeat(255), W4), String::from(W3)] {
        let t = OffsetTable::build(&text);
        let (u, c) = plain(&text, text.len());
        assert_eq!(OffsetCursor::new(&t, &text).to_utf16(text.len() as u32), Utf16Offset(u));
        assert_eq!(OffsetCursor::new(&t, &text).to_code_point(text.len() as u32), CodePointOffset(c));
        let mut o16 = Vec::new();
        let mut oc = Vec::new();
        t.column_to_utf16(&text, &[text.len() as u32], &mut o16);
        t.column_to_code_point(&text, &[text.len() as u32], &mut oc);
        assert_eq!(o16, vec![Utf16Offset(u)]);
        assert_eq!(oc, vec![CodePointOffset(c)]);
    }
}

#[test]
fn columns_that_are_not_sorted() {
    let text = long_mixed();
    let t = OffsetTable::build(&text);
    let all: Vec<u32> = boundaries(&text).iter().map(|x| x.0).collect();
    let mut state = 12345u64;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    let mut columns: Vec<Vec<u32>> = vec![
        all.iter().rev().copied().collect(),
        vec![*all.last().unwrap(), 0, *all.last().unwrap(), 0],
        vec![all[all.len() / 2], all[all.len() / 2], 0, all[all.len() / 2]],
    ];
    for _ in 0..20 {
        columns.push((0..100).map(|_| all[next() % all.len()]).collect());
    }
    for column in &columns {
        let mut o16 = vec![Utf16Offset(999)];
        let mut oc = vec![CodePointOffset(999)];
        t.column_to_utf16(&text, column, &mut o16);
        t.column_to_code_point(&text, column, &mut oc);
        // The column is appended to what `out` already holds.
        assert_eq!(o16[0], Utf16Offset(999));
        assert_eq!(oc[0], CodePointOffset(999));
        let want: Vec<(u32, u32)> = column.iter().map(|&b| plain(&text, b as usize)).collect();
        assert_eq!(o16[1..], want.iter().map(|x| Utf16Offset(x.0)).collect::<Vec<_>>()[..]);
        assert_eq!(oc[1..], want.iter().map(|x| CodePointOffset(x.1)).collect::<Vec<_>>()[..]);
    }
}

#[test]
fn empty_column_appends_nothing() {
    let text = long_mixed();
    let t = OffsetTable::build(&text);
    let mut o16 = Vec::new();
    let mut oc = Vec::new();
    t.column_to_utf16(&text, &[], &mut o16);
    t.column_to_code_point(&text, &[], &mut oc);
    assert!(o16.is_empty() && oc.is_empty());
}

#[test]
fn spans_empty_whole_and_across_strides() {
    let text = long_mixed();
    let t = OffsetTable::build(&text);
    let all = boundaries(&text);
    let at = |w: usize| *all.iter().find(|x| x.0 as usize >= w).unwrap();

    // Empty spans at the start, in the middle, across a stride, and at the end.
    for w in [0, 1, STRIDE_BYTES, STRIDE_BYTES + 1, 2 * STRIDE_BYTES - 1, text.len()] {
        let (b, u, c) = at(w);
        assert_eq!(t.span_to_utf16(&text, Span::new(b, b)), (Utf16Offset(u), Utf16Offset(u)));
        assert_eq!(t.span_to_code_point(&text, Span::new(b, b)), (CodePointOffset(c), CodePointOffset(c)));
    }

    // The whole text.
    let (len, ulen, clen) = *all.last().unwrap();
    assert_eq!(t.span_to_utf16(&text, Span::new(0, len)), (Utf16Offset(0), Utf16Offset(ulen)));
    assert_eq!(t.span_to_code_point(&text, Span::new(0, len)), (CodePointOffset(0), CodePointOffset(clen)));

    // Start and end several strides apart, and in neighbouring strides.
    for (s, e) in [(10, 3 * STRIDE_BYTES + 7), (STRIDE_BYTES - 1, STRIDE_BYTES + 1), (0, STRIDE_BYTES), (700, 1500)] {
        let (sb, su, sc) = at(s);
        let (eb, eu, ec) = at(e);
        assert_eq!(plain(&text, sb as usize), (su, sc));
        assert_eq!(plain(&text, eb as usize), (eu, ec));
        assert_eq!(t.span_to_utf16(&text, Span::new(sb, eb)), (Utf16Offset(su), Utf16Offset(eu)));
        assert_eq!(t.span_to_code_point(&text, Span::new(sb, eb)), (CodePointOffset(sc), CodePointOffset(ec)));
    }

    // The same on ASCII text and on an empty text.
    let ascii = "a".repeat(700);
    let ta = OffsetTable::build(&ascii);
    assert_eq!(ta.span_to_utf16(&ascii, Span::new(5, 600)), (Utf16Offset(5), Utf16Offset(600)));
    assert_eq!(ta.span_to_code_point(&ascii, Span::new(0, 0)), (CodePointOffset(0), CodePointOffset(0)));
    let te = OffsetTable::build("");
    assert_eq!(te.span_to_utf16("", Span::new(0, 0)), (Utf16Offset(0), Utf16Offset(0)));
}

#[test]
fn document_views_each_have_a_table_for_their_own_text() {
    // A source with a ligature-able "fi" and a long tail, so the derived
    // views can be shorter and longer than it.
    let src = format!("fine {} caf\u{E9} {}", W4, "word ".repeat(120));
    let mut doc = Document::new(src.as_str());
    check_all(&src);

    // Shorter: two ASCII bytes become the 3 byte ligature (longer in bytes,
    // shorter in characters), and the long tail is dropped.
    let (shorter, alignment) = {
        let mut b = ViewBuilder::new(&src);
        b.replace(2, "\u{FB01}");
        b.keep(src.find("caf").unwrap());
        b.replace(src.len(), "");
        b.finish()
    };
    assert!(shorter.len() < src.len());
    let view = View::derived(doc.get::<View>().unwrap(), shorter.clone(), alignment);
    doc.add(&Key::named("shorter"), view, Provenance::by("t")).unwrap();

    // Longer: every ASCII space becomes a 4 byte character.
    let (longer, alignment) = {
        let mut b = ViewBuilder::new(&src);
        let mut at = 0;
        while let Some(i) = src[at..].find(' ') {
            let i = at + i;
            b.keep(i);
            b.replace(i + 1, "\u{1F600}");
            at = i + 1;
        }
        b.finish()
    };
    assert!(longer.len() > src.len() + 300);
    let view = View::derived(doc.get::<View>().unwrap(), longer.clone(), alignment);
    doc.add(&Key::named("longer"), view, Provenance::by("t")).unwrap();

    for (id, text) in [(ViewId::SOURCE, src.as_str()), (ViewId(1), shorter.as_str()), (ViewId(2), longer.as_str())] {
        assert_eq!(doc.text(id), text);
        let t = doc.offsets(id);
        assert_eq!(t.len() as usize, text.len());
        assert_eq!(t.utf16_len(), Utf16Offset(text.encode_utf16().count() as u32));
        assert_eq!(t.code_point_len(), CodePointOffset(text.chars().count() as u32));
        for (b, u, c) in boundaries(text) {
            assert_eq!(t.to_utf16(text, b), Utf16Offset(u));
            assert_eq!(t.to_code_point(text, b), CodePointOffset(c));
            assert_eq!(t.from_utf16(text, Utf16Offset(u)), Ok(b));
            assert_eq!(t.from_code_point(text, CodePointOffset(c)), Ok(b));
        }
        assert!(std::ptr::eq(t, doc.offsets(id)), "built once and kept");
    }
    // The three tables differ from each other.
    assert_ne!(doc.offsets(ViewId::SOURCE), doc.offsets(ViewId(1)));
    assert_ne!(doc.offsets(ViewId::SOURCE), doc.offsets(ViewId(2)));
    assert_ne!(doc.offsets(ViewId(1)), doc.offsets(ViewId(2)));
}

#[test]
fn document_source_view_matches_its_own_text() {
    let text = format!("{}{}", W3.to_string().repeat(100), "a".repeat(400));
    let doc = Document::new(text.as_str());
    let t = doc.offsets(ViewId::SOURCE);
    assert!(!t.is_ascii());
    assert_eq!(t.len() as usize, text.len());
    check_all(&text);
    assert_eq!(t, &OffsetTable::build(&text));
}

/// A small deterministic generator, so the test needs no crate.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

#[test]
fn two_hundred_pseudo_random_texts() {
    const ALPHABET: [char; 8] = ['a', 'Z', '\u{E9}', '\u{3B1}', '\u{20AC}', '\u{4E2D}', '\u{1D11E}', '\u{1F600}'];
    let mut r = Lcg(0x5EED);
    for _ in 0..200 {
        let limit = (r.next() % 3001) as usize;
        let mut text = String::new();
        while text.len() < limit {
            // Mostly mixed, with occasional ASCII-only runs so some
            // strides hold no wide characters.
            let c = if r.next().is_multiple_of(4) { 'q' } else { ALPHABET[(r.next() % 8) as usize] };
            if text.len() + c.len_utf8() > limit {
                break;
            }
            text.push(c);
        }
        assert!(text.len() <= 3000);
        check_all(&text);
    }
}
