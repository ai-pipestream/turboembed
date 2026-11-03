use std::sync::Arc;

use turbo_analysis::{Annotator, Error, Layer, ModelRef, Span, Spans, Tag, TagSet, ViewId};
use turbo_analysis_layers::{IncivilityLabel, Sentences, Vectors};

use super::*;

fn test_model() -> ModelRef {
    ModelRef { id: "tiny-bert".into(), revision: "rev1".into(), sha256: "0123456789abcdef".into() }
}

#[test]
fn linear_and_mlp_heads_round_trip_and_score_deterministically() {
    let tags = TagSet::new("sentiment", &["positive", "negative", "neutral"]);
    let linear = ClassifierHead::linear(
        test_model(),
        3,
        Arc::clone(&tags),
        HeadMode::SingleLabel,
        vec![
            2.0, -1.0, 0.0, // positive
            -1.0, 2.0, 0.0, // negative
            0.0, 0.0, 1.5, // neutral
        ],
        vec![0.1, -0.1, 0.0],
    )
    .unwrap();

    let bytes = linear.to_bytes();
    let loaded = ClassifierHead::from_bytes(&bytes).unwrap();
    assert_eq!(linear, loaded);

    let mut probs1 = [0.0f64; 3];
    let mut probs2 = [0.0f64; 3];
    let (t1, p1) = linear.score(&[1.0, -0.5, 0.0], &mut probs1).unwrap();
    let (t2, p2) = loaded.score(&[1.0, -0.5, 0.0], &mut probs2).unwrap();
    assert_eq!(t1, Tag(0));
    assert_eq!(t1, t2);
    assert_eq!(p1.to_bits(), p2.to_bits());
    assert_eq!(probs1, probs2);
    assert!((probs1.iter().sum::<f64>() - 1.0).abs() < 1e-12);

    // 2-layer MLP with GELU activation and MultiLabel sigmoid mode.
    let inciv_names = IncivilityLabel::ALL.map(|l| l.name());
    let inciv_tags = TagSet::new("incivility", &inciv_names);
    let mlp = ClassifierHead::mlp(
        test_model(),
        2,
        3,
        inciv_tags,
        HeadMode::MultiLabel,
        HiddenActivation::Gelu,
        vec![1.0, 0.5, -0.5, 1.0, 0.2, -0.2],
        vec![0.0, 0.1, -0.1],
        vec![
            1.2, -0.4, 0.0, // insult
            0.8, 0.2, -0.1, // profanity
            -0.5, 0.9, 0.3, // threat
            0.1, -0.2, 0.7, // identity_attack
            -0.3, 0.4, -0.2, // intolerance
            0.9, 0.9, 0.1, // severe_toxicity
        ],
        vec![-0.2, -0.5, -1.0, -0.8, -0.9, -1.2],
    )
    .unwrap();

    let mlp_bytes = mlp.to_bytes();
    let mlp_loaded = ClassifierHead::from_bytes(&mlp_bytes).unwrap();
    assert_eq!(mlp, mlp_loaded);

    let mut s1 = [0.0f64; 6];
    let mut s2 = [0.0f64; 6];
    mlp.score(&[1.5, -0.5], &mut s1).unwrap();
    mlp_loaded.score(&[1.5, -0.5], &mut s2).unwrap();
    assert_eq!(s1, s2);
    assert!(s1.iter().all(|&v| (0.0..=1.0).contains(&v)));
}

#[test]
fn fit_single_and_multi_label_converge_deterministically() {
    let tags = TagSet::new("sentiment", &["positive", "negative"]);
    let examples: &[([f32; 2], Tag)] =
        &[([1.0, 0.1], Tag(0)), ([0.9, 0.2], Tag(0)), ([-1.0, -0.1], Tag(1)), ([-0.8, -0.2], Tag(1))];
    let h1 = ClassifierHead::fit_single_label(test_model(), 2, Arc::clone(&tags), examples, 1e-3, 150, 0.5).unwrap();
    let h2 = ClassifierHead::fit_single_label(test_model(), 2, Arc::clone(&tags), examples, 1e-3, 150, 0.5).unwrap();
    assert_eq!(h1, h2);

    let mut probs = [0.0f64; 2];
    let (tag_pos, p_pos) = h1.score(&[0.95, 0.15], &mut probs).unwrap();
    assert_eq!(tag_pos, Tag(0));
    assert!(p_pos > 0.9, "p_pos={p_pos}");

    let (tag_neg, p_neg) = h1.score(&[-0.95, -0.15], &mut probs).unwrap();
    assert_eq!(tag_neg, Tag(1));
    assert!(p_neg > 0.9, "p_neg={p_neg}");
}

#[test]
fn vector_head_annotator_rejects_mismatched_scope_and_vector_counts() {
    let tags = TagSet::new("sentiment", &["positive", "negative"]);
    let head =
        ClassifierHead::linear(test_model(), 2, tags, HeadMode::SingleLabel, vec![1.0, 0.0, 0.0, 1.0], vec![0.0, 0.0])
            .unwrap();
    let annotator = VectorHeadAnnotator::<Sentences>::new(head);

    let mut sent_spans = Spans::new(ViewId::SOURCE);
    sent_spans.push(Span::new(0, 5));
    sent_spans.push(Span::new(6, 11));
    let sentences = Sentences { spans: sent_spans };

    let bad_vectors = Vectors::<Sentences>::new(2, vec![1.0, 0.0], Vec::new(), test_model());
    let err = annotator.annotate((&sentences, &bad_vectors)).unwrap_err();
    assert!(matches!(err, Error::Annotator { annotator: VectorHeadAnnotator::<Sentences>::ID, .. }));

    let ok_vectors = Vectors::<Sentences>::new(2, vec![1.0, 0.0, 0.0, 1.0], Vec::new(), test_model());
    let (cats,) = annotator.annotate((&sentences, &ok_vectors)).unwrap();
    assert_eq!(cats.len(), 2);
}

#[test]
fn incivility_and_pragmatics_scoring_reject_tokens_from_another_view() {
    use turbo_analysis::IxRange;
    use turbo_analysis_layers::{Chunks, Cut, Tokens};
    let inciv_names = IncivilityLabel::ALL.map(|l| l.name());
    let inciv_tags = TagSet::new("incivility", &inciv_names);
    let inciv_head =
        ClassifierHead::linear(test_model(), 2, inciv_tags, HeadMode::MultiLabel, vec![0.0; 12], vec![0.0; 6]).unwrap();
    let prag_tags = TagSet::new("pragmatics", &["sarcastic", "literal"]);
    let prag_head =
        ClassifierHead::linear(test_model(), 2, prag_tags, HeadMode::SingleLabel, vec![0.0; 4], vec![0.0; 2]).unwrap();

    let mut sent_spans = Spans::new(ViewId::SOURCE);
    sent_spans.push(Span::new(0, 5));
    let sentences = Sentences { spans: sent_spans };

    let mut chunk_spans = Spans::new(ViewId::SOURCE);
    chunk_spans.push(Span::new(0, 5));
    let chunks = Chunks {
        spans: chunk_spans,
        level: vec![0],
        parent: vec![None],
        sentences: vec![IxRange::new(0, 1)],
        cut: vec![Cut::Sentence],
        cost: vec![0],
        model_tokens: vec![0],
        over_budget: vec![false],
        prompt_overhead: 0,
    };

    let mut tok_spans = Spans::new(ViewId(1));
    tok_spans.push(Span::new(0, 5));
    let bad_tokens = Tokens { spans: tok_spans, sentence: None };

    let vecs = Vectors::<Sentences>::new(2, vec![0.1, 0.2], Vec::new(), test_model());
    let err = inciv_head.score_incivility_sentences("test", &bad_tokens, &sentences, &vecs).unwrap_err();
    assert!(err.to_string().contains("the tokens are in view 1, the sentences in view 0"), "got: {err}");
    let err = prag_head.score_pragmatics_sentences("test", &sentences, &bad_tokens, &vecs).unwrap_err();
    assert!(err.to_string().contains("the tokens are in view 1, the sentences in view 0"), "got: {err}");

    let chunk_vecs = Vectors::<Chunks>::new(2, vec![0.1, 0.2], Vec::new(), test_model());
    let err = inciv_head.score_incivility_chunks("test", &bad_tokens, &chunks, &chunk_vecs).unwrap_err();
    assert!(err.to_string().contains("the tokens are in view 1, the chunks in view 0"), "got: {err}");
}

#[test]
fn classifier_head_is_a_pragmatics_vector_scorer() {
    let head = ClassifierHead::linear(
        test_model(),
        1,
        TagSet::new("pragmatics", &["literal", "sarcastic"]),
        HeadMode::SingleLabel,
        vec![0.0, 2.0],
        vec![0.0, 0.0],
    )
    .unwrap();
    let expected = 1.0 / (1.0 + (-2.0f64).exp());
    assert!((head.score_pragmatics_vector(&[1.0]).unwrap() - expected).abs() < 1e-12);
    let scorer: &dyn turbo_doccat::sentiment::pragmatics::PragmaticsVectorScorer = &head;
    assert!((scorer.score_sentence_vector(&[1.0]).unwrap() - expected).abs() < 1e-12);
    assert_eq!(scorer.config_hash(), head.config_hash());
    assert_eq!(scorer.model_ref(), Some(test_model()));
    assert!(scorer.score_sentence_vector(&[1.0, 0.0]).is_err());
}
