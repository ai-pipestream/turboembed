use std::marker::PhantomData;

use turbo_analysis::{Document, Error, Ix, IxRange, Key, ModelRef, Provenance, Span, Spans, View, ViewId};
use turbo_analysis_layers::{Chunks, Cut, Levels, SparseTerms, Strs, Tokens, Unit};
use turbo_chunker::attribution::{ChunkAttribution, DocumentHeat, LateMatch, LexicalTerm, MatchSignal};
use turbo_chunker::cascade::{FilterMode, UnitFilter};
use turbo_chunker::terms::{IndexedTerms, Vocabulary};
use turbo_normalize::{CaseFold, normalize};

fn chunks(view: ViewId, rows: &[&[(u32, u32)]]) -> Chunks {
    let mut spans = Spans::new(view);
    for row in rows {
        let fragments: Vec<_> = row.iter().map(|&(a, b)| Span::new(a, b)).collect();
        spans.push_fragments(&fragments);
    }
    let n = rows.len();
    Chunks {
        spans,
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

fn token_data(view: ViewId, span: Span) -> (Tokens, IndexedTerms, Vocabulary) {
    let mut spans = Spans::new(view);
    spans.push(span);
    let mut text = Strs::default();
    text.push("xx");
    (Tokens { spans, sentence: None }, IndexedTerms { term: vec![Some(Ix::new(0))] }, Vocabulary { text })
}

fn late(span: Span) -> LateMatch {
    LateMatch { query_token: 0, span, score: 1.0, term: "xx".into(), query_span: None }
}

fn accepted_builders(original: &Chunks, supplied: &Chunks, span: Span) -> Vec<String> {
    let base = ChunkAttribution::new(original, Ix::new(0)).unwrap();
    let (tokens, indexed, vocab) = token_data(supplied.spans.view(), span);
    let query = [LexicalTerm { term: Ix::new(0), weight: 1.0, signal: MatchSignal::ExactTerm, query_span: None }];
    let mut accepted = Vec::new();
    let mut a = base.clone();
    if let Ok(n) = a.add_lexical(supplied, &tokens, &indexed, &vocab, &query) {
        accepted.push(format!("lexical={n}"));
    } else {
        assert_eq!(a, base);
    }
    let sparse = SparseTerms::<Chunks> {
        row: vec![0, 1],
        term: vec![0],
        weight: vec![1.0],
        span: vec![Some(span)],
        occ: Vec::new(),
        occ_at: Vec::new(),
        model: ModelRef { id: "span-fixture".into(), revision: "1".into(), sha256: "0".into() },
        _of: PhantomData,
    };
    let mut a = base.clone();
    if let Ok(n) = a.add_sparse(supplied, &sparse, |id| (id == 0).then(|| vocab.get(Ix::new(id))), None) {
        accepted.push(format!("sparse={n}"));
    } else {
        assert_eq!(a, base);
    }
    let mut a = base.clone();
    if let Ok(n) = a.add_late(supplied, &[late(span)]) {
        accepted.push(format!("late={n}"));
    } else {
        assert_eq!(a, base);
    }
    let mut a = base.clone();
    if let Ok(n) = a.add_filter_spans(supplied, "place", &[span], Some(1.0)) {
        accepted.push(format!("spans={n}"));
    } else {
        assert_eq!(a, base);
    }
    let mut spans = Spans::new(supplied.spans.view());
    spans.push(span);
    let levels = Levels { spans, unit: vec![Unit::Document], parent: vec![None] };
    let filter =
        UnitFilter { name: "place".into(), mode: FilterMode::Boost(1.0), units: [Ix::new(0)].into_iter().collect() };
    let mut a = base.clone();
    if let Ok(n) = a.apply_unit_filter(supplied, &levels, &filter) {
        accepted.push(format!("levels={n}"));
    } else {
        assert_eq!(a, base);
    }
    accepted
}

#[test]
fn same_envelope_different_view_is_not_the_original_chunk() {
    let original = chunks(ViewId::SOURCE, &[&[(0, 6)]]);
    let other = chunks(ViewId(1), &[&[(0, 6)]]);
    let accepted = accepted_builders(&original, &other, Span::new(0, 2));
    assert!(accepted.is_empty(), "foreign-view chunks accepted by {accepted:?}");
}

#[test]
fn same_envelope_different_fragments_cannot_credit_a_gap() {
    let original = chunks(ViewId::SOURCE, &[&[(0, 2), (4, 6)]]);
    for replacement in [&[(0, 6)][..], &[(0, 1), (4, 6)][..]] {
        let other = chunks(ViewId::SOURCE, &[replacement]);
        let accepted = accepted_builders(&original, &other, Span::new(2, 4));
        assert!(accepted.is_empty(), "different geometry {replacement:?} accepted by {accepted:?}");
    }
}

#[test]
fn equivalent_geometry_in_separate_allocations_and_storage_is_accepted() {
    let doc = Document::new("aaxxbb");
    for geometry in [&[(0, 6)][..], &[(0, 2), (4, 6)][..]] {
        let original = chunks(ViewId::SOURCE, &[geometry]);
        // Adding an unrelated discontinuous row gives even a continuous row
        // a fragment table instead of its original compact representation.
        let equivalent = chunks(ViewId::SOURCE, &[geometry, &[(0, 1), (5, 6)]]);
        assert_eq!(
            accepted_builders(&original, &equivalent, Span::new(0, 2)),
            ["lexical=1", "sparse=1", "late=1", "spans=1", "levels=1"]
        );
        let mut a = ChunkAttribution::new(&original, Ix::new(0)).unwrap();
        a.total_score = 1.0;
        assert_eq!(
            DocumentHeat::build(&doc, ViewId::SOURCE, &original, &[a.clone()]).unwrap(),
            DocumentHeat::build(&doc, ViewId::SOURCE, &equivalent, &[a]).unwrap()
        );
    }
}

#[test]
fn heat_cannot_transfer_attribution_to_another_view_or_fragment_layout() {
    let mut doc = Document::new("AAxxBB");
    let (text, alignment) = normalize(&CaseFold::lowercase(), doc.source());
    let view = View::derived(doc.get::<View>().unwrap(), text, alignment);
    doc.add(&Key::named("lower"), view, Provenance::by("case-fold")).unwrap();
    let child = doc.get_key(&Key::<View>::named("lower")).unwrap().id();
    assert_eq!(doc.text(child), "aaxxbb");
    let original = chunks(ViewId::SOURCE, &[&[(0, 2), (4, 6)]]);
    let continuous = chunks(ViewId::SOURCE, &[&[(0, 6)]]);
    let foreign_view = chunks(child, &[&[(0, 2), (4, 6)]]);
    let mut a = ChunkAttribution::new(&original, Ix::new(0)).unwrap();
    a.dense_score = Some(1.0);
    a.recompute_total();
    let mut accepted = Vec::new();
    for (name, view, supplied) in
        [("different fragments", ViewId::SOURCE, &continuous), ("different view", child, &foreign_view)]
    {
        if let Ok(heat) = DocumentHeat::build(&doc, view, supplied, &[a.clone()]) {
            accepted.push(format!("{name}: {:?}", heat.segments));
        }
    }
    assert!(accepted.is_empty(), "attribution transferred: {}", accepted.join("; "));
}

#[test]
fn heat_checks_chunk_endpoints_even_without_matched_spans() {
    let mut accepted = Vec::new();
    for (name, text, range) in [
        ("past view end", "abcdef", (0, 7)),
        ("splits a scalar", "éx", (0, 1)),
        ("empty inside a scalar", "éx", (1, 1)),
    ] {
        let doc = Document::new(text);
        let ch = chunks(ViewId::SOURCE, &[&[range]]);
        let mut a = ChunkAttribution::new(&ch, Ix::new(0)).unwrap();
        a.total_score = 1.0;
        match DocumentHeat::build(&doc, ViewId::SOURCE, &ch, &[a]) {
            Err(Error::Bounds { view: ViewId::SOURCE, .. }) => {}
            result => accepted.push(format!("{name}: {result:?}")),
        }
    }
    assert!(accepted.is_empty(), "invalid view endpoints accepted: {}", accepted.join("; "));
}

#[test]
fn heat_refuses_invalid_internal_fragments_and_unattributed_chunks_with_bounds_errors() {
    let doc = Document::new("aéxbb");
    // The envelope is valid; only the interior fragment endpoint cuts é.
    let discontinuous = chunks(ViewId::SOURCE, &[&[(0, 2), (4, 6)]]);
    let a = ChunkAttribution::new(&discontinuous, Ix::new(0)).unwrap();
    assert!(matches!(DocumentHeat::build(&doc, ViewId::SOURCE, &discontinuous, &[a]), Err(Error::Bounds { .. })));
    let unscored = chunks(ViewId::SOURCE, &[&[(0, 7)]]);
    assert!(matches!(DocumentHeat::build(&doc, ViewId::SOURCE, &unscored, &[]), Err(Error::Bounds { .. })));
    let empty = Document::new("");
    let outside = chunks(ViewId::SOURCE, &[&[(1, 1)]]);
    assert!(matches!(DocumentHeat::build(&empty, ViewId::SOURCE, &outside, &[]), Err(Error::Bounds { .. })));
}

#[test]
fn valid_fragment_edges_and_zero_width_matches_follow_the_documented_policy() {
    let doc = Document::new("aaxxbb");
    let ch = chunks(ViewId::SOURCE, &[&[(0, 2), (4, 6)]]);
    let mut a = ChunkAttribution::new(&ch, Ix::new(0)).unwrap();
    assert_eq!(
        a.add_late(
            &ch,
            &[
                late(Span::new(1, 5)),
                late(Span::new(2, 2)),
                late(Span::new(3, 3)),
                late(Span::new(4, 4)),
                late(Span::new(6, 6))
            ]
        )
        .unwrap(),
        5
    );
    assert_eq!(a.late_score, Some(4.0));
    a.recompute_total();
    let heat = DocumentHeat::build(&doc, ViewId::SOURCE, &ch, &[a]).unwrap();
    assert_eq!(
        heat.segments.iter().map(|s| (s.span, s.chunk)).collect::<Vec<_>>(),
        [(Span::new(0, 2), Some(Ix::new(0))), (Span::new(2, 4), None), (Span::new(4, 6), Some(Ix::new(0)))]
    );
    let matched: Vec<_> = heat.segments.iter().flat_map(|s| s.matched_spans.iter().map(|m| m.span)).collect();
    assert_eq!(matched, [Span::new(1, 2), Span::new(4, 5), Span::new(4, 4)]);
    for (text, at) in [("", 0), ("éx", 2), ("éx", 3)] {
        let doc = Document::new(text);
        let ch = chunks(ViewId::SOURCE, &[&[(at, at)]]);
        let mut a = ChunkAttribution::new(&ch, Ix::new(0)).unwrap();
        a.add_late(&ch, &[late(Span::new(at, at))]).unwrap();
        a.recompute_total();
        let heat = DocumentHeat::build(&doc, ViewId::SOURCE, &ch, &[a]).unwrap();
        assert!(heat.segments.iter().all(|s| s.score.is_none() && s.matched_spans.is_empty()));
    }
}

#[test]
fn losing_starts_are_dropped_but_winning_starts_keep_exact_offsets() {
    let doc = Document::new("abcdef");
    let ch = chunks(ViewId::SOURCE, &[&[(0, 4)], &[(2, 6)]]);
    let mut a = ChunkAttribution::new(&ch, Ix::new(0)).unwrap();
    a.add_late(&ch, &[late(Span::new(1, 4)), late(Span::new(3, 4))]).unwrap();
    a.total_score = 1.0;
    let mut b = ChunkAttribution::new(&ch, Ix::new(1)).unwrap();
    b.add_late(&ch, &[late(Span::new(2, 3))]).unwrap();
    b.total_score = 2.0;
    let heat = DocumentHeat::build(&doc, ViewId::SOURCE, &ch, &[a, b]).unwrap();
    assert_eq!(heat.segments[0].matched_spans.iter().map(|m| m.span).collect::<Vec<_>>(), [Span::new(1, 4)]);
    assert_eq!(heat.segments[1].matched_spans.iter().map(|m| m.span).collect::<Vec<_>>(), [Span::new(2, 3)]);
}

#[test]
fn signed_zero_and_equal_scores_have_the_documented_winners() {
    let doc = Document::new("abcdef");
    let ch = chunks(ViewId::SOURCE, &[&[(0, 6)], &[(0, 6)]]);
    for (left, right, winner, negative) in
        [(-0.0f32, 0.0, 1, false), (0.0, -0.0, 0, false), (0.0, 0.0, 0, false), (-0.0, -0.0, 0, true)]
    {
        let mut a = ChunkAttribution::new(&ch, Ix::new(0)).unwrap();
        let mut b = ChunkAttribution::new(&ch, Ix::new(1)).unwrap();
        a.total_score = left;
        b.total_score = right;
        let heat = DocumentHeat::build(&doc, ViewId::SOURCE, &ch, &[a, b]).unwrap();
        assert_eq!(heat.segments[0].chunk, Some(Ix::new(winner)));
        assert_eq!(heat.segments[0].score.unwrap().is_sign_negative(), negative);
    }
}
