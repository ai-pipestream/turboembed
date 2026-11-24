//! Cursors, whole columns of offsets, and spans.

use super::*;

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
