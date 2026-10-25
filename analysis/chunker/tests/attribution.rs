use std::collections::BTreeSet;
use std::marker::PhantomData;

use turbo_analysis::{Document, Error, Ix, IxRange, Key, ModelRef, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{Chunks, Cut, Sentences, SparseTerms, Strs, Tokens};
use turbo_chunker::attribution::{ChunkAttribution, DocumentHeat, LateMatch, LexicalTerm, MatchSignal};
use turbo_chunker::terms::{IndexedTerms, Vocabulary};

fn spans<L>(pairs: &[(u32, u32)]) -> Spans<L> {
    let mut s = Spans::new(ViewId::SOURCE);
    for &(a, b) in pairs {
        s.push(Span::new(a, b));
    }
    s
}

fn chunks(pairs: &[(u32, u32)]) -> Chunks {
    let n = pairs.len();
    Chunks {
        spans: spans(pairs),
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

/// Text "aa bb cc dd" (11 bytes): tokens at 0..2, 3..5, 6..8, 9..11;
/// vocab ids aa=0, bb=1, cc=2, dd=3; chunks [0,5) and [3,11) overlap.
fn fixture() -> (Document, Chunks, Tokens, IndexedTerms, Vocabulary) {
    let mut doc = Document::new("aa bb cc dd");
    doc.add(&Key::DEFAULT, Sentences { spans: spans(&[(0, 11)]) }, Provenance::by("t")).unwrap();
    let tokens = Tokens { spans: spans(&[(0, 2), (3, 5), (6, 8), (9, 11)]), sentence: None };
    let indexed = IndexedTerms { term: vec![Some(Ix::new(0)), Some(Ix::new(1)), Some(Ix::new(2)), Some(Ix::new(1))] };
    let mut text = Strs::default();
    for w in ["aa", "bb", "cc", "dd"] {
        text.push(w);
    }
    (doc, chunks(&[(0, 5), (3, 11)]), tokens, indexed, Vocabulary { text })
}

#[test]
fn lexical_sparse_late_and_filter_attribution() {
    let (_doc, ch, tokens, indexed, vocab) = fixture();
    let mut a = ChunkAttribution::new(&ch, Ix::new(1)).unwrap();
    // bb occurs at tokens 1 and 3, both inside chunk 1 [3,11).
    let added = a
        .add_lexical(
            &ch,
            &tokens,
            &indexed,
            &vocab,
            &[LexicalTerm {
                term: Ix::new(1),
                weight: 0.5,
                signal: MatchSignal::ExactTerm,
                query_span: Some(Span::new(0, 2)),
            }],
        )
        .unwrap();
    assert_eq!(added, 2);
    assert_eq!(a.lexical_score, Some(1.0));
    assert_eq!(a.matched_spans[0].span, Span::new(3, 5));
    assert_eq!(a.matched_spans[1].span, Span::new(9, 11));
    assert_eq!(a.matched_spans[0].term, "bb");
    // Sparse: entry for chunk 1 with one occurrence over cc; chunk 0
    // has an expanded entry (no occurrences).
    let mut sparse = SparseTerms::<Chunks> {
        row: vec![0, 1, 2],
        term: vec![7, 2],
        weight: vec![0.25, 0.75],
        span: vec![None, Some(Span::new(6, 8))],
        occ: vec![Span::new(6, 8)],
        occ_at: vec![0, 1],
        model: ModelRef { id: "m".into(), revision: "0".into(), sha256: "0".into() },
        _of: PhantomData,
    };
    sparse.span[0] = None;
    let added = a.add_sparse(&ch, &sparse, |id| (id == 2).then_some("cc"), None).unwrap();
    assert_eq!(added, 1);
    assert_eq!(a.sparse_score, Some(0.75));
    assert!(matches!(a.matched_spans[2].signal, MatchSignal::SparseTerm { term_id: 2, expanded: false }));
    // Wanted filter excludes it on a fresh attribution.
    let mut b = ChunkAttribution::new(&ch, Ix::new(1)).unwrap();
    assert_eq!(b.add_sparse(&ch, &sparse, |_| Some("x"), Some(&BTreeSet::from([9]))).unwrap(), 0);
    assert_eq!(b.sparse_score, None);
    // Late + filter + total.
    let added = a
        .add_late(
            &ch,
            &[LateMatch { query_token: 4, span: Span::new(9, 11), score: 0.25, term: "dd".into(), query_span: None }],
        )
        .unwrap();
    assert_eq!(added, 1);
    let added = a.add_filter_spans(&ch, "place", &[Span::new(0, 4)], Some(2.0)).unwrap();
    assert_eq!(added, 1, "filter span clips to the chunk fragment");
    assert_eq!(a.matched_spans.last().unwrap().span, Span::new(3, 4));
    a.dense_score = Some(0.5);
    a.recompute_total();
    assert_eq!(a.total_score, (1.0 + 0.75 + 0.25 + 0.5) * 2.0);
}

#[test]
fn attribution_refusals() {
    let (_doc, ch, tokens, indexed, vocab) = fixture();
    assert!(ChunkAttribution::new(&ch, Ix::new(9)).is_err());
    let mut a = ChunkAttribution::new(&ch, Ix::new(0)).unwrap();
    assert!(
        a.add_lexical(
            &ch,
            &tokens,
            &indexed,
            &vocab,
            &[LexicalTerm { term: Ix::new(0), weight: f32::NAN, signal: MatchSignal::ExactTerm, query_span: None }],
        )
        .is_err()
    );
    assert!(
        a.add_lexical(
            &ch,
            &tokens,
            &indexed,
            &vocab,
            &[LexicalTerm { term: Ix::new(9), weight: 1.0, signal: MatchSignal::ExactTerm, query_span: None }],
        )
        .unwrap_err()
        .contains("outside")
    );
    assert!(a.add_filter_spans(&ch, "x", &[Span::new(0, 2)], Some(0.0)).is_err());
    assert!(
        a.add_late(
            &ch,
            &[LateMatch {
                query_token: 0,
                span: Span::new(0, 2),
                score: f32::INFINITY,
                term: "x".into(),
                query_span: None
            }]
        )
        .is_err()
    );
}

#[test]
fn document_heat_partitions_and_resolves_overlaps() {
    let (doc, ch, tokens, indexed, vocab) = fixture();
    // Chunk 0 [0,5) scores 1.0, chunk 1 [3,11) scores 2.0: the overlap
    // [3,5) belongs to chunk 1. Chunk 0 attributed over aa token.
    let mut a0 = ChunkAttribution::new(&ch, Ix::new(0)).unwrap();
    a0.add_lexical(
        &ch,
        &tokens,
        &indexed,
        &vocab,
        &[LexicalTerm { term: Ix::new(0), weight: 1.0, signal: MatchSignal::ExactTerm, query_span: None }],
    )
    .unwrap();
    a0.total_score = 1.0;
    let mut a1 = ChunkAttribution::new(&ch, Ix::new(1)).unwrap();
    a1.add_lexical(
        &ch,
        &tokens,
        &indexed,
        &vocab,
        &[LexicalTerm { term: Ix::new(1), weight: 1.0, signal: MatchSignal::ExactTerm, query_span: None }],
    )
    .unwrap();
    a1.total_score = 2.0;
    let heat = DocumentHeat::build(&doc, ViewId::SOURCE, &ch, &[a0, a1]).unwrap();
    type HeatRow = (u32, u32, Option<f32>, Option<u32>, usize);
    let rows: Vec<HeatRow> = heat
        .segments
        .iter()
        .map(|s| (s.span.start, s.span.end, s.score, s.chunk.map(|c| c.get() as u32), s.matched_spans.len()))
        .collect();
    assert_eq!(rows, vec![(0, 3, Some(1.0), Some(0), 1), (3, 11, Some(2.0), Some(1), 2)]);
    // The aa matched span rides with chunk 0 at its exact offsets.
    assert_eq!(heat.segments[0].matched_spans[0].span, Span::new(0, 2));
    // No attributions: one unscored gap over the whole view.
    let heat = DocumentHeat::build(&doc, ViewId::SOURCE, &ch, &[]).unwrap();
    assert_eq!(heat.segments.len(), 1);
    assert_eq!((heat.segments[0].score, heat.segments[0].chunk), (None, None));
    assert_eq!(heat.segments[0].span, Span::new(0, 11));
    // Refusals: duplicate attribution, wrong view.
    let a0 = ChunkAttribution::new(&ch, Ix::new(0)).unwrap();
    let dup = ChunkAttribution::new(&ch, Ix::new(0)).unwrap();
    assert!(matches!(DocumentHeat::build(&doc, ViewId::SOURCE, &ch, &[a0, dup]), Err(Error::Invalid { .. })));
    assert!(matches!(DocumentHeat::build(&doc, ViewId(7), &ch, &[]), Err(Error::Bounds { .. })));
}
