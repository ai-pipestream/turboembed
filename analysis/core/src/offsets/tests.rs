use super::*;

/// Every character boundary of `text`, with its offset in all three
/// units, counted plainly.
fn boundaries(text: &str) -> Vec<(u32, u32, u32)> {
    let mut out = vec![(0, 0, 0)];
    let (mut u, mut c) = (0, 0);
    for (i, ch) in text.char_indices() {
        u += ch.len_utf16() as u32;
        c += 1;
        out.push(((i + ch.len_utf8()) as u32, u, c));
    }
    out
}

#[test]
fn ascii_keeps_no_checkpoints() {
    let text = "plain ASCII text";
    let t = OffsetTable::build(text);
    assert!(t.is_ascii());
    assert_eq!(t.checkpoints(), 0);
    assert_eq!(t.to_utf16(text, 6), Utf16Offset(6));
    assert_eq!(t.from_code_point(text, CodePointOffset(16)), Ok(16));
}

#[test]
fn checkpoints_land_on_boundaries_one_stride_apart() {
    // 3 byte characters do not divide 256, so most checkpoints are past
    // their stride's start.
    let text = "\u{20AC}".repeat(400);
    let t = OffsetTable::build(&text);
    assert_eq!(t.checkpoints(), text.len().div_ceil(STRIDE as usize));
    for (k, &b) in t.byte.iter().enumerate() {
        assert!(text.is_char_boundary(b as usize));
        assert!(b >= k as u32 * STRIDE && b < k as u32 * STRIDE + 4);
        assert_eq!(t.utf16[k], b / 3);
        assert_eq!(t.code_point[k], b / 3);
    }
}

#[test]
fn every_boundary_both_ways() {
    let text = "a\u{E9}\u{20AC}\u{1D11E}b\u{1F600}\u{4E2D}";
    let t = OffsetTable::build(text);
    for (b, u, c) in boundaries(text) {
        assert_eq!(t.to_utf16(text, b), Utf16Offset(u));
        assert_eq!(t.to_code_point(text, b), CodePointOffset(c));
        assert_eq!(t.from_utf16(text, Utf16Offset(u)), Ok(b));
        assert_eq!(t.from_code_point(text, CodePointOffset(c)), Ok(b));
    }
}

#[test]
fn default_table_matches_empty_text() {
    let t = OffsetTable::default();
    assert_eq!(t, OffsetTable::build(""));
    assert!(t.is_ascii());
    assert_eq!(t.to_utf16("", 0), Utf16Offset(0));
    assert_eq!(t.to_code_point("", 0), CodePointOffset(0));
    assert_eq!(t.from_utf16("", Utf16Offset(0)), Ok(0));
    assert_eq!(t.from_code_point("", CodePointOffset(0)), Ok(0));
}

#[test]
fn typed_unit_spans_round_trip_and_check_bounds() {
    let text = "a\u{E9}\u{1F600}b"; // 'a' (1B, 1u16, 1cp), 'é' (2B, 1u16, 1cp), '😀' (4B, 2u16, 1cp), 'b' (1B, 1u16, 1cp)
    let t = OffsetTable::build(text);

    let emoji_bytes = Span::new(3, 7);
    let u16_span = t.span_to_utf16_typed(text, emoji_bytes);
    assert_eq!(u16_span, Utf16Span::new(Utf16Offset(2), Utf16Offset(4)));
    assert_eq!(u16_span.len(), 2);
    assert!(!u16_span.is_empty());
    assert_eq!(t.span_from_utf16(text, u16_span), Ok(emoji_bytes));

    let cp_span = t.span_to_code_point_typed(text, emoji_bytes);
    assert_eq!(cp_span, CodePointSpan::new(CodePointOffset(2), CodePointOffset(3)));
    assert_eq!(cp_span.len(), 1);
    assert!(!cp_span.is_empty());
    assert_eq!(t.span_from_code_point(text, cp_span), Ok(emoji_bytes));

    assert!(Utf16Span::try_new(Utf16Offset(4), Utf16Offset(2)).is_none());
    assert!(CodePointSpan::try_new(CodePointOffset(3), CodePointOffset(1)).is_none());

    // Surrogate interior is refused.
    let split_surrogate = Utf16Span::new(Utf16Offset(2), Utf16Offset(3));
    assert_eq!(
        t.span_from_utf16(text, split_surrogate),
        Err(Error::Offset { unit: OffsetUnit::Utf16, at: 3, reason: OffsetError::InsideCharacter })
    );

    // Past end is refused.
    let past_u16 = Utf16Span::new(Utf16Offset(0), Utf16Offset(99));
    assert_eq!(
        t.span_from_utf16(text, past_u16),
        Err(Error::Offset { unit: OffsetUnit::Utf16, at: 99, reason: OffsetError::PastEnd { len: 5 } })
    );
    let past_cp = CodePointSpan::new(CodePointOffset(0), CodePointOffset(99));
    assert_eq!(
        t.span_from_code_point(text, past_cp),
        Err(Error::Offset { unit: OffsetUnit::CodePoint, at: 99, reason: OffsetError::PastEnd { len: 4 } })
    );

    // Reversed raw struct literals are refused without panicking.
    let rev_u16 = Utf16Span { start: Utf16Offset(4), end: Utf16Offset(1) };
    assert!(t.span_from_utf16(text, rev_u16).is_err());
    let rev_cp = CodePointSpan { start: CodePointOffset(3), end: CodePointOffset(1) };
    assert!(t.span_from_code_point(text, rev_cp).is_err());

    // Reversed spans on a derived view report the derived ViewId, not ViewId(0).
    let mut doc = crate::Document::new(text);
    let v0 = doc.get::<crate::View>().unwrap().clone();
    let mut b = crate::ViewBuilder::new(v0.text());
    b.keep(text.len());
    let (derived_text, align) = b.finish();
    let v1_key = crate::Key::<crate::View>::named("v1");
    doc.add(&v1_key, crate::View::derived(&v0, derived_text, align), crate::Provenance::by("test")).unwrap();
    let v1 = doc.get_key(&v1_key).unwrap();
    let v1_id = v1.id();
    assert_eq!(v1_id, crate::ViewId(1));
    let rev_u16_v1 = Utf16Span { start: Utf16Offset(2), end: Utf16Offset(1) };
    let rev_cp_v1 = CodePointSpan { start: CodePointOffset(2), end: CodePointOffset(1) };
    assert!(matches!(v1.span_from_utf16(rev_u16_v1), Err(Error::Bounds { view, .. }) if view == v1_id));
    assert!(matches!(v1.span_from_code_point(rev_cp_v1), Err(Error::Bounds { view, .. }) if view == v1_id));
    assert!(matches!(doc.view_span_from_utf16(v1_id, rev_u16_v1), Err(Error::Bounds { view, .. }) if view == v1_id));
    assert!(
        matches!(doc.view_span_from_code_point(v1_id, rev_cp_v1), Err(Error::Bounds { view, .. }) if view == v1_id)
    );
}
