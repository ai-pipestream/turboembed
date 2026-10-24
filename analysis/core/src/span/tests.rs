use super::text::check_no_crossing;
use crate::text::ViewId;

use super::*;

struct Dummy;

fn spans_of(list: &[(u32, u32)]) -> Spans<Dummy> {
    let mut s = Spans::with_capacity(ViewId::SOURCE, list.len());
    for &(a, b) in list {
        s.push(Span::new(a, b));
    }
    s
}

#[test]
fn overlap_none_nesting_and_any_check_invariants() {
    let ordered = spans_of(&[(0, 4), (4, 8), (10, 14)]);
    assert_eq!(ordered.check(Overlap::None), Ok(()));
    assert_eq!(ordered.check(Overlap::Nesting), Ok(()));
    assert_eq!(ordered.check(Overlap::Any), Ok(()));

    // Nested spans: "Bank of New York" (0..16) and "New York" (8..16),
    // plus an inner "Bank" (0..4) and identical span (8..16).
    let nested = spans_of(&[(0, 16), (0, 4), (8, 16), (8, 16), (17, 22)]);
    assert!(nested.check(Overlap::None).is_err());
    assert_eq!(nested.check(Overlap::Nesting), Ok(()));
    assert_eq!(nested.check(Overlap::Any), Ok(()));

    // Out-of-order nested and disjoint spans fail Overlap::None and
    // pass Overlap::Nesting and Overlap::Any (e.g. RegexNameFinder emits
    // entities grouped by type rather than text start order).
    let unsorted_nested = spans_of(&[(8, 16), (17, 22), (0, 16), (0, 4)]);
    assert!(!unsorted_nested.is_start_ordered());
    assert!(unsorted_nested.check(Overlap::None).is_err());
    assert_eq!(unsorted_nested.check(Overlap::Nesting), Ok(()));
    assert_eq!(unsorted_nested.check(Overlap::Any), Ok(()));

    // Two identical spans (0..5, 0..5) and 0..10 containing 0..5 and 5..10
    // satisfy Overlap::Nesting; 0..6 and 4..10 partially cross.
    assert_eq!(spans_of(&[(0, 5), (0, 5)]).check(Overlap::Nesting), Ok(()));
    assert_eq!(spans_of(&[(0, 10), (0, 5), (5, 10)]).check(Overlap::Nesting), Ok(()));
    assert!(spans_of(&[(0, 6), (4, 10)]).check(Overlap::Nesting).unwrap_err().contains("partially crosses"));

    // Partially crossing spans: 0..10 and 6..14 fail Nesting and pass Any.
    let crossing = spans_of(&[(0, 10), (6, 14)]);
    assert!(crossing.check(Overlap::None).is_err());
    assert!(crossing.check(Overlap::Nesting).unwrap_err().contains("partially crosses"));
    assert_eq!(crossing.check(Overlap::Any), Ok(()));
}

#[test]
fn discontinuous_spans_keep_envelopes_and_fragments_by_column() {
    let mut s = Spans::<Dummy>::new(ViewId::SOURCE);
    let i0 = s.push(Span::new(0, 3));
    assert!(!s.has_discontinuous());
    assert!(s.fragment_columns().is_none());

    // "take ... off" at 4..8 and 14..17, with "this coat" in 9..13.
    let i1 = s.push_fragments(&[Span::new(4, 8), Span::new(14, 17)]);
    let i2 = s.push(Span::new(9, 13));
    assert!(s.has_discontinuous());
    assert!(!s.is_discontinuous(i0));
    assert!(s.is_discontinuous(i1));
    assert!(!s.is_discontinuous(i2));
    assert_eq!(s.fragment_count(i0), 1);
    assert_eq!(s.fragment_count(i1), 2);
    assert_eq!(s.fragment_count(i2), 1);
    assert_eq!(s.get(i1), Span::new(4, 17));
    assert_eq!(s.fragments(i0).collect::<Vec<_>>(), [Span::new(0, 3)]);
    assert_eq!(s.fragments(i1).collect::<Vec<_>>(), [Span::new(4, 8), Span::new(14, 17)]);
    assert_eq!(s.fragments(i2).collect::<Vec<_>>(), [Span::new(9, 13)]);

    // Discontinuous spans fail Overlap::None, and pass Nesting when
    // neither envelopes nor fragments cross (here 9..13 is nested
    // inside the envelope 4..17 and disjoint from its fragments).
    assert!(s.check(Overlap::None).is_err());
    assert_eq!(s.check(Overlap::Nesting), Ok(()));
    assert_eq!(s.check(Overlap::Any), Ok(()));

    // Refuses empty input, backwards fragment, empty fragment in
    // multi-fragment, and touching/overlapping/out-of-order fragments.
    assert_eq!(s.try_push_fragments(&[]), Err(FragmentError::Empty));
    assert_eq!(
        s.try_push_fragments(&[Span { start: 25, end: 20 }]),
        Err(FragmentError::EndsBeforeStart { start: 25, end: 20 }),
    );
    assert_eq!(
        s.try_push_fragments(&[Span::new(20, 20), Span::new(24, 28)]),
        Err(FragmentError::EmptyFragment { start: 20 }),
    );
    assert_eq!(
        s.try_push_fragments(&[Span::new(20, 24), Span::new(24, 28)]),
        Err(FragmentError::NotSeparated { first: Span::new(20, 24), second: Span::new(24, 28) }),
    );
    assert_eq!(
        s.try_push_fragments(&[Span::new(20, 25), Span::new(24, 28)]),
        Err(FragmentError::NotSeparated { first: Span::new(20, 25), second: Span::new(24, 28) }),
    );
    assert_eq!(
        s.try_push_fragments(&[Span::new(24, 28), Span::new(20, 22)]),
        Err(FragmentError::NotSeparated { first: Span::new(24, 28), second: Span::new(20, 22) }),
    );
}

#[test]
fn with_capacity_followed_by_push_fragments_preserves_first_item_offsets() {
    let mut s = Spans::<Dummy>::with_capacity(ViewId::SOURCE, 4);
    let i0 = s.push_fragments(&[Span::new(2, 5), Span::new(10, 14)]);
    let i1 = s.push(Span::new(15, 20));
    assert_eq!(s.get(i0), Span::new(2, 14));
    assert_eq!(s.fragments(i0).collect::<Vec<_>>(), [Span::new(2, 5), Span::new(10, 14)]);
    assert_eq!(s.fragments(i1).collect::<Vec<_>>(), [Span::new(15, 20)]);
    assert_eq!(s.check(Overlap::Nesting), Ok(()));
}

#[test]
fn discontinuous_fragment_nesting_rejects_crossing_fragments_and_gap_splits() {
    // Valid: outer 0..20 contains discontinuous 2..5; 10..15.
    let mut ok = Spans::<Dummy>::new(ViewId::SOURCE);
    ok.push(Span::new(0, 20));
    ok.push_fragments(&[Span::new(2, 5), Span::new(10, 15)]);
    assert_eq!(ok.check(Overlap::Nesting), Ok(()));

    // Valid: two discontinuous items whose fragments are completely
    // disjoint (0..4; 10..13 and 5..8; 14..16), even though their
    // bounding envelopes (0..13 and 5..16) cross.
    let mut disjoint_interleaved = Spans::<Dummy>::new(ViewId::SOURCE);
    disjoint_interleaved.push_fragments(&[Span::new(0, 4), Span::new(10, 13)]);
    disjoint_interleaved.push_fragments(&[Span::new(5, 8), Span::new(14, 16)]);
    assert_eq!(disjoint_interleaved.check(Overlap::Nesting), Ok(()));

    // Invalid: item 0 has fragments 0..6 and 15..20 (envelope 0..20),
    // item 1 is 4..10 (inside envelope 0..20, but partially crosses
    // fragment 0..6 and is not contained by item 0's fragments).
    let mut bad = Spans::<Dummy>::new(ViewId::SOURCE);
    bad.push_fragments(&[Span::new(0, 6), Span::new(15, 20)]);
    bad.push(Span::new(4, 10));
    let err = bad.check(Overlap::Nesting).unwrap_err();
    assert!(err.contains("partially crosses"), "{err}");
    // Under Overlap::Any, the same spans are accepted.
    assert_eq!(bad.check(Overlap::Any), Ok(()));

    // Invalid, and only the item-level check sees it: no fragment
    // crosses another, but item 1 (2..4; 10..12) is neither inside
    // item 0 (0..6; 15..20) nor outside it.
    let mut bad = Spans::<Dummy>::new(ViewId::SOURCE);
    bad.push_fragments(&[Span::new(0, 6), Span::new(15, 20)]);
    bad.push_fragments(&[Span::new(2, 4), Span::new(10, 12)]);
    let (frag_start, frag_end, _) = bad.fragment_columns().unwrap();
    assert_eq!(check_no_crossing(frag_start, frag_end, "fragment"), Ok(()));
    let err = bad.check(Overlap::Nesting).unwrap_err();
    assert!(err.contains("across fragments"), "{err}");

    // Invalid: item 0 has fragments 0..5 and 10..20, item 1 is 2..15
    // (shares bytes with both fragments of item 0 without either item
    // containing the other).
    let mut bad2 = Spans::<Dummy>::new(ViewId::SOURCE);
    bad2.push_fragments(&[Span::new(0, 5), Span::new(10, 20)]);
    bad2.push(Span::new(2, 15));
    assert!(bad2.check(Overlap::Nesting).is_err());
}

#[test]
fn text_spans_check_detects_corrupted_fragment_columns() {
    let start = [0u32, 10];
    let end = [8u32, 20];
    let frag_start = [0u32, 5, 10];
    let frag_end = [3u32, 8, 20];
    let frag_item_end = [2u32, 3];
    let base = TextSpans {
        view: ViewId::SOURCE,
        start: &start,
        end: &end,
        frag_start: &frag_start,
        frag_end: &frag_end,
        frag_item_end: &frag_item_end,
    };
    assert_eq!(base.check(Overlap::Nesting), Ok(()));

    // 1. Unequal frag_start / frag_end length.
    let bad_frag_len = TextSpans { frag_end: &frag_end[..2], ..base };
    assert!(bad_frag_len.check(Overlap::Any).unwrap_err().contains("fragment starts"));

    // 2. frag_item_end.len() != start.len().
    let bad_item_end_len = TextSpans { frag_item_end: &frag_item_end[..1], ..base };
    assert!(bad_item_end_len.check(Overlap::Any).unwrap_err().contains("fragment item ends"));

    // 3. frag_item_end.last() != Some(frag_start.len()).
    let bad_total = [1u32, 2];
    let bad_last = TextSpans { frag_item_end: &bad_total, ..base };
    assert!(bad_last.check(Overlap::Any).unwrap_err().contains("fragment item ends reach"));

    // 4. 0-fragment item / non-monotonic frag_item_end.
    let zero_frag = [0u32, 3];
    let bad_zero = TextSpans { frag_item_end: &zero_frag, ..base };
    assert!(bad_zero.check(Overlap::Any).unwrap_err().contains("empty or out-of-bounds fragment range"));

    // 5. Envelope mismatch with first/last fragment.
    let wrong_env_start = [1u32, 10];
    let bad_env = TextSpans { start: &wrong_env_start, ..base };
    assert!(bad_env.check(Overlap::Any).unwrap_err().contains("envelope"));
}

#[test]
fn covering_overlapping_and_in_range_handle_nesting_and_gaps() {
    let mut s = Spans::<Dummy>::new(ViewId::SOURCE);
    let outer = s.push(Span::new(0, 16)); // "Bank of New York"
    let inner = s.push(Span::new(8, 16)); // "New York"
    let disc = s.push_fragments(&[Span::new(20, 24), Span::new(30, 33)]); // "take ... off"
    let gap_word = s.push(Span::new(25, 29)); // inside the gap of `disc`

    assert_eq!(s.covering(2), [outer]);
    assert_eq!(s.covering(10), [outer, inner]);
    assert_eq!(s.covering(16), []);
    // Byte 22 is in the first fragment of `disc`; byte 26 is in the gap
    // between its fragments, so `disc` does not cover 26.
    assert_eq!(s.covering(22), [disc]);
    assert_eq!(s.covering(26), [gap_word]);
    assert_eq!(s.covering(31), [disc]);

    assert_eq!(s.overlapping(Span::new(12, 21)), [outer, inner, disc]);
    assert_eq!(s.overlapping(Span::new(25, 29)), [gap_word]);
    assert_eq!(s.overlapping(Span::new(24, 30)), [gap_word]);

    assert_eq!(s.in_range(Span::new(0, 16)), [outer, inner]);
    assert_eq!(s.in_range(Span::new(6, 18)), [inner]);
    assert_eq!(s.in_range(Span::new(20, 33)), [disc, gap_word]);

    // Empty Spans, out-of-order Spans, zero-width spans at query boundary,
    // and duplicate spans.
    let empty = Spans::<Dummy>::new(ViewId::SOURCE);
    assert_eq!(empty.covering(0), []);
    assert_eq!(empty.overlapping(Span::new(0, 5)), []);
    assert_eq!(empty.in_range(Span::new(0, 5)), []);

    let unsorted = spans_of(&[(10, 15), (5, 5), (0, 8), (0, 8)]);
    assert!(!unsorted.is_start_ordered());
    assert_eq!(unsorted.covering(5), [Ix::new(1), Ix::new(2), Ix::new(3)]);
    assert_eq!(unsorted.overlapping(Span::new(5, 10)), [Ix::new(1), Ix::new(2), Ix::new(3)]);
    assert_eq!(unsorted.in_range(Span::new(0, 10)), [Ix::new(1), Ix::new(2), Ix::new(3)]);

    let boundary = spans_of(&[(0, 5), (5, 5), (5, 10), (10, 10)]);
    assert_eq!(boundary.overlapping(Span::new(5, 10)), [Ix::new(1), Ix::new(2)]);
    assert_eq!(boundary.overlapping(Span::new(5, 5)), [Ix::new(1), Ix::new(2)]);
}

#[test]
fn filter_spans_resolves_conflicts_by_rule_into_ordered_spans() {
    // 0: 0..16 ("Bank of New York", score 0.6)
    // 1: 0..4  ("Bank", score 0.95)
    // 2: 8..16 ("New York", score 0.9)
    // 3: 12..20 ("York City", score 0.85, crosses 0 and 2)
    // 4: 21..27 ("Boston", score 0.7)
    let spans = spans_of(&[(0, 16), (0, 4), (8, 16), (12, 20), (21, 27)]);
    let scores = [0.6, 0.95, 0.9, 0.85, 0.7];

    // LongestFirst: 0..16 (len 16) wins over 0..4, 8..16, and 12..20;
    // 21..27 stays.
    let longest = filter_spans(&spans, Some(&scores), FilterRule::LongestFirst);
    assert_eq!(longest, [Ix::new(0), Ix::new(4)]);

    // HighestScore: 0..4 (0.95) and 8..16 (0.9) win; 0..16 and 12..20
    // conflict with them and are dropped; 21..27 stays.
    let best = filter_spans(&spans, Some(&scores), FilterRule::HighestScore);
    assert_eq!(best, [Ix::new(1), Ix::new(2), Ix::new(4)]);

    // LeftmostLongest: at start 0, 0..16 beats 0..4 and blocks 8..16
    // and 12..20; 21..27 stays.
    let leftmost = filter_spans(&spans, Some(&scores), FilterRule::LeftmostLongest);
    assert_eq!(leftmost, [Ix::new(0), Ix::new(4)]);

    // Difference between LongestFirst and LeftmostLongest when an earlier
    // shorter span crosses a later longer span: 0..6 (len 6) vs 4..14 (len 10).
    let cross = spans_of(&[(0, 6), (4, 14)]);
    assert_eq!(filter_spans(&cross, None, FilterRule::LongestFirst), [Ix::new(1)]);
    assert_eq!(filter_spans(&cross, None, FilterRule::LeftmostLongest), [Ix::new(0)]);

    for rule in [FilterRule::LongestFirst, FilterRule::HighestScore, FilterRule::LeftmostLongest] {
        let kept = spans.filter(Some(&scores), rule);
        let mut resolved = Spans::<Dummy>::new(ViewId::SOURCE);
        for ix in kept {
            resolved.push(spans.get(ix));
        }
        assert_eq!(resolved.check(Overlap::None), Ok(()));
    }
}

/// Parity fixture against spaCy's `spacy.util.filter_spans`: sorts by
/// `(span.end - span.start)` descending, then `span.start` ascending,
/// keeping non-overlapping spans and returning them in text order.
#[test]
fn filter_spans_longest_first_matches_spacy_util_filter_spans() {
    type Case<'a> = (&'a [(u32, u32)], &'a [u32]);
    let fixtures: &[Case<'_>] = &[
        // Nested and crossing spans from spaCy's test_filter_spans.
        (&[(1, 4), (2, 3), (5, 8), (6, 9), (1, 2)], &[0, 2]),
        // Equal-length overlapping spans: earlier start wins (2..6 over 4..8).
        (&[(4, 8), (2, 6)], &[1]),
        // Identical duplicate spans: earlier input index wins (index 0 over index 1).
        (&[(2, 6), (2, 6), (6, 10)], &[0, 2]),
        // Touching spans do not overlap and are all kept.
        (&[(0, 3), (3, 7), (7, 10)], &[0, 1, 2]),
        // Chain of crossing spans where the middle longest span unblocks both ends.
        (&[(0, 4), (2, 10), (8, 12)], &[1]),
        (&[(0, 3), (2, 5), (4, 10), (9, 12)], &[0, 2]),
    ];
    for &(input, expected) in fixtures {
        let s = spans_of(input);
        let got: Vec<u32> =
            filter_spans(&s, None, FilterRule::LongestFirst).into_iter().map(|ix| ix.get() as u32).collect();
        assert_eq!(got, expected, "input: {input:?}");
    }
}

#[test]
fn overlapping_range_and_covering_at_match_the_general_queries() {
    let spans = spans_of(&[(0, 5), (5, 9), (12, 12), (12, 20), (25, 30)]);
    let range = |a: u32, b: u32| spans.overlapping_range(Span::new(a, b));
    let idx = |r: IxRange<Dummy>| (r.start.get(), r.end.get());
    assert_eq!(idx(range(0, 30)), (0, 5));
    assert_eq!(idx(range(4, 13)), (0, 4));
    assert_eq!(idx(range(9, 12)), (2, 2), "the gap overlaps nothing");
    assert_eq!(idx(range(20, 25)), (4, 4), "touching ends do not overlap");
    assert_eq!(idx(range(12, 12)), (2, 4), "point query: empty item and covering item");
    assert_eq!(idx(range(5, 5)), (1, 2), "point at a boundary belongs to the next item");
    assert_eq!(idx(range(30, 30)), (5, 5));
    for (span, want) in [((0, 30), 5usize), ((4, 13), 4), ((12, 12), 2)] {
        assert_eq!(spans.overlapping(Span::new(span.0, span.1)).len(), want);
    }
    assert_eq!(spans.covering_at(0), Some(Ix::new(0)));
    assert_eq!(spans.covering_at(4), Some(Ix::new(0)));
    assert_eq!(spans.covering_at(5), Some(Ix::new(1)));
    assert_eq!(spans.covering_at(10), None, "gap");
    assert_eq!(spans.covering_at(12), Some(Ix::new(3)), "covering item wins over the empty one at its start");
    assert_eq!(spans.covering_at(30), None);
    let empty = Spans::<Dummy>::new(ViewId::SOURCE);
    assert_eq!(empty.covering_at(0), None);
    assert_eq!(empty.overlapping_range(Span::new(0, 0)).len(), 0);
    let only_empty = spans_of(&[(7, 7)]);
    assert_eq!(only_empty.covering_at(7), Some(Ix::new(0)), "an empty item covers its own point");
    assert_eq!(only_empty.covering_at(6), None);
}
