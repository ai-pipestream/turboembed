use std::marker::PhantomData;

use turbo_analysis::{Ix, IxRange, ModelRef, Span, Spans, ViewId};
use turbo_analysis_layers::{Chunks, Cut, Levels, SparseTerms, Strs, Tokens, Unit};
use turbo_chunker::attribution::{ChunkAttribution, LateMatch, LexicalTerm, MatchSignal};
use turbo_chunker::cascade::{FilterMode, UnitFilter};
use turbo_chunker::terms::{IndexedTerms, Vocabulary};

fn spans<L>() -> Spans<L> {
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 2));
    spans
}

fn chunks() -> Chunks {
    Chunks {
        spans: spans(),
        level: vec![0],
        parent: vec![None],
        sentences: vec![IxRange::new(0, 1)],
        cut: vec![Cut::Sentence],
        cost: vec![0],
        model_tokens: vec![1],
        over_budget: vec![false],
        prompt_overhead: 0,
    }
}

fn seeded(ch: &Chunks) -> ChunkAttribution {
    let mut a = ChunkAttribution::new(ch, Ix::new(0)).unwrap();
    a.add_filter_spans(ch, "existing", &[Span::new(0, 1)], Some(2.0)).unwrap();
    a.dense_score = Some(3.0);
    a.recompute_total();
    a
}

fn late(score: f32) -> LateMatch {
    LateMatch { query_token: 0, span: Span::new(0, 2), score, term: "aa".into(), query_span: None }
}

fn lexical(weight: f32) -> LexicalTerm {
    LexicalTerm { term: Ix::new(0), weight, signal: MatchSignal::ExactTerm, query_span: None }
}

fn sparse() -> SparseTerms<Chunks> {
    SparseTerms {
        row: vec![0, 2],
        term: vec![0, 1],
        weight: vec![1.0, 2.0],
        span: vec![Some(Span::new(0, 1)), Some(Span::new(1, 2))],
        occ: vec![],
        occ_at: vec![],
        model: ModelRef { id: "terms".into(), revision: "1".into(), sha256: "fixture".into() },
        _of: PhantomData,
    }
}

fn words() -> (Tokens, IndexedTerms, Vocabulary) {
    let mut text = Strs::default();
    text.push("aa");
    (Tokens { spans: spans(), sentence: None }, IndexedTerms { term: vec![Some(Ix::new(0))] }, Vocabulary { text })
}

#[test]
fn boost_narrowing_refuses_infinite_or_zero_results_before_mutation() {
    let ch = chunks();
    let levels = Levels { spans: spans(), unit: vec![Unit::Chunk(Ix::new(0))], parent: vec![None] };
    for factor in [1e300, 1e-300, f64::MAX, f64::from_bits(1), f64::NAN, -1.0, 0.0] {
        for units in [vec![Ix::new(0)], vec![]] {
            let mut a = seeded(&ch);
            let before = a.clone();
            let f = UnitFilter {
                name: "boost".into(),
                mode: FilterMode::Boost(factor),
                units: units.into_iter().collect(),
            };
            assert!(a.apply_unit_filter(&ch, &levels, &f).is_err(), "accepted {factor}");
            assert_eq!(a, before);
        }
    }
}

#[test]
fn representable_boosts_preserve_the_existing_rounding_contract() {
    let ch = chunks();
    let levels = Levels { spans: spans(), unit: vec![Unit::Chunk(Ix::new(0))], parent: vec![None] };
    for factor in [f32::from_bits(1) as f64, f32::MIN_POSITIVE as f64, 0.1, 1.0, f32::MAX as f64] {
        let mut a = seeded(&ch);
        let f = UnitFilter {
            name: "boost".into(),
            mode: FilterMode::Boost(factor),
            units: [Ix::new(0)].into_iter().collect(),
        };
        assert_eq!(a.apply_unit_filter(&ch, &levels, &f).unwrap(), 1);
        assert_eq!(a.filter_boost, Some(factor as f32));
        assert_eq!(a.matched_spans.last().unwrap().weight.to_bits(), (factor as f32).to_bits());
    }
}

#[test]
fn late_refusal_preserves_existing_state_and_retry_does_not_duplicate_matches() {
    let ch = chunks();
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut a = seeded(&ch);
        a.add_late(&ch, &[late(0.5)]).unwrap();
        let before = a.clone();
        assert!(a.add_late(&ch, &[late(1.0), late(invalid)]).is_err());
        assert_eq!(a, before);
        assert_eq!(a.add_late(&ch, &[late(1.0), late(2.0)]).unwrap(), 2);
        assert_eq!(a.late_score, Some(3.5));
        assert_eq!(a.matched_spans.len(), before.matched_spans.len() + 2);
    }
}

#[test]
fn lexical_refusal_rolls_back_a_valid_prefix() {
    let ch = chunks();
    let (tokens, indexed, vocab) = words();
    let bad = [
        lexical(f32::NAN),
        LexicalTerm { term: Ix::new(99), ..lexical(1.0) },
        LexicalTerm { signal: MatchSignal::StructuredFilter { kind: "bad".into() }, ..lexical(1.0) },
    ];
    for bad in bad {
        let mut a = seeded(&ch);
        let before = a.clone();
        assert!(a.add_lexical(&ch, &tokens, &indexed, &vocab, &[lexical(1.0), bad]).is_err());
        assert_eq!(a, before);
        assert_eq!(a.add_lexical(&ch, &tokens, &indexed, &vocab, &[lexical(1.0), lexical(2.0)]).unwrap(), 2);
        assert_eq!(a.lexical_score, Some(3.0));
    }
}

#[test]
fn sparse_lookup_refusal_rolls_back_a_valid_prefix() {
    let ch = chunks();
    let mut a = seeded(&ch);
    let before = a.clone();
    assert!(a.add_sparse(&ch, &sparse(), |id| (id == 0).then_some("aa"), None).is_err());
    assert_eq!(a, before);
    assert_eq!(a.add_sparse(&ch, &sparse(), |_| Some("aa"), None).unwrap(), 2);
    assert_eq!(a.sparse_score, Some(3.0));
    assert_eq!(a.matched_spans.len(), before.matched_spans.len() + 2);
}

#[test]
fn sparse_row_count_must_match_all_chunks_not_only_the_selected_row() {
    let ch = chunks();
    let mut wrong = sparse();
    wrong.row.push(2);
    let mut a = seeded(&ch);
    let before = a.clone();
    assert!(a.add_sparse(&ch, &wrong, |_| Some("aa"), None).is_err());
    assert_eq!(a, before);
}

#[test]
fn finite_inputs_cannot_overflow_a_signal_score_or_leave_partial_matches() {
    let ch = chunks();
    let (tokens, indexed, vocab) = words();
    let mut a = seeded(&ch);
    let before = a.clone();
    assert!(a.add_late(&ch, &[late(f32::MAX), late(f32::MAX)]).is_err());
    assert_eq!(a, before);
    assert!(a.add_lexical(&ch, &tokens, &indexed, &vocab, &[lexical(f32::MAX), lexical(f32::MAX)]).is_err());
    assert_eq!(a, before);
    let mut huge = sparse();
    huge.weight.fill(f32::MAX);
    assert!(a.add_sparse(&ch, &huge, |_| Some("aa"), None).is_err());
    assert_eq!(a, before);
}
