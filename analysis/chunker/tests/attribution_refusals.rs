//! Inputs from another view, unit ids past the end of `Levels`, and a
//! `Chunks` other than the one an attribution was opened over are
//! refused, never clipped into wrong spans or a panic.

use turbo_analysis::{Ix, IxRange, Span, Spans, ViewId};
use turbo_analysis_layers::{Chunks, Cut, Levels, Strs, Tokens, Unit};
use turbo_chunker::attribution::{ChunkAttribution, LateMatch, LexicalTerm, MatchSignal};
use turbo_chunker::cascade::{FilterMode, UnitFilter};
use turbo_chunker::terms::{IndexedTerms, Vocabulary};

fn spans<L>(view: ViewId, pairs: &[(u32, u32)]) -> Spans<L> {
    let mut s = Spans::new(view);
    for &(a, b) in pairs {
        s.push(Span::new(a, b));
    }
    s
}

fn chunks(pairs: &[(u32, u32)]) -> Chunks {
    let n = pairs.len();
    Chunks {
        spans: spans(ViewId::SOURCE, pairs),
        level: vec![0; n],
        parent: vec![None; n],
        sentences: (0..n).map(|i| IxRange::new(i as u32, i as u32 + 1)).collect(),
        cut: vec![Cut::Sentence; n],
        cost: vec![0; n],
        model_tokens: vec![1; n],
        over_budget: vec![false; n],
        prompt_overhead: 0,
    }
}

fn levels(view: ViewId) -> Levels {
    Levels {
        spans: spans(view, &[(0, 11), (0, 5), (3, 11)]),
        unit: vec![Unit::Document, Unit::Chunk(Ix::new(0)), Unit::Chunk(Ix::new(1))],
        parent: vec![None, Some(Ix::new(0)), Some(Ix::new(0))],
    }
}

fn filter(units: &[u32]) -> UnitFilter {
    UnitFilter {
        name: "place".into(),
        mode: FilterMode::Boost(2.0),
        units: units.iter().map(|&u| Ix::new(u)).collect(),
    }
}

#[test]
fn lexical_tokens_from_another_view_are_refused() {
    let ch = chunks(&[(0, 5), (3, 11)]);
    let tokens = Tokens { spans: spans(ViewId(1), &[(0, 2), (3, 5)]), sentence: None };
    let indexed = IndexedTerms { term: vec![Some(Ix::new(0)), Some(Ix::new(0))] };
    let mut text = Strs::default();
    text.push("aa");
    let vocab = Vocabulary { text };
    let mut a = ChunkAttribution::new(&ch, Ix::new(0)).unwrap();
    let q = [LexicalTerm { term: Ix::new(0), weight: 1.0, signal: MatchSignal::ExactTerm, query_span: None }];
    let err = a.add_lexical(&ch, &tokens, &indexed, &vocab, &q).unwrap_err();
    assert!(err.contains("view"), "{err}");
    assert_eq!((a.lexical_score, a.matched_spans.len()), (None, 0));
}

#[test]
fn unit_filter_levels_from_another_view_are_refused() {
    let ch = chunks(&[(0, 5), (3, 11)]);
    let mut a = ChunkAttribution::new(&ch, Ix::new(1)).unwrap();
    let err = a.apply_unit_filter(&ch, &levels(ViewId(1)), &filter(&[1])).unwrap_err();
    assert!(err.contains("view"), "{err}");
    assert_eq!((a.filter_boost, a.matched_spans.len()), (None, 0));
    // The same filter over same-view levels clips unit 1 [0,5) to [3,5).
    assert_eq!(a.apply_unit_filter(&ch, &levels(ViewId::SOURCE), &filter(&[1])).unwrap(), 1);
    assert_eq!((a.filter_boost, a.matched_spans[0].span), (Some(2.0), Span::new(3, 5)));
}

#[test]
fn unit_filter_units_past_the_levels_are_refused() {
    let ch = chunks(&[(0, 5), (3, 11)]);
    let mut a = ChunkAttribution::new(&ch, Ix::new(1)).unwrap();
    let err = a.apply_unit_filter(&ch, &levels(ViewId::SOURCE), &filter(&[1, 10])).unwrap_err();
    assert!(err.contains("unit 10"), "{err}");
    assert!(a.matched_spans.is_empty());
}

#[test]
fn builders_refuse_chunks_other_than_the_attributed_ones() {
    let five = chunks(&[(0, 2), (2, 4), (4, 6), (6, 8), (8, 11)]);
    let two = chunks(&[(0, 5), (3, 11)]);
    let mut a = ChunkAttribution::new(&five, Ix::new(4)).unwrap();
    let late = [LateMatch { query_token: 0, span: Span::new(8, 10), score: 1.0, term: "x".into(), query_span: None }];
    let err = a.add_late(&two, &late).unwrap_err();
    assert!(err.contains("chunk 4"), "{err}");
    let err = a.add_filter_spans(&two, "x", &[Span::new(0, 11)], None).unwrap_err();
    assert!(err.contains("chunk 4"), "{err}");
    // Same index, different envelope: chunk 1 is [2,4) in `five`, [3,11) in `two`.
    let mut b = ChunkAttribution::new(&five, Ix::new(1)).unwrap();
    assert!(b.add_late(&two, &late).is_err());
    assert_eq!(a.add_late(&five, &late).unwrap(), 1);
}
