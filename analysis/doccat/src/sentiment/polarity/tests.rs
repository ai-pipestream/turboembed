use std::sync::Arc;

use turbo_analysis::{Analyzer, Annotator, Document, Key, Layer, ModelRef, Provenance, Span, Spans, View, ViewId};
use turbo_analysis_layers::{Opinions, Polarity, Sentences, Tokens, Vectors};

use crate::scorers::{PolarityStage, PolarityVectorScorer};
use crate::sentiment::lexicon::{LexiconSentiment, SENTENCE_SENTIMENT};

use super::*;

fn encoder_ref(id: &str) -> ModelRef {
    ModelRef { id: id.into(), revision: "1".into(), sha256: "a".repeat(64) }
}

fn synthetic_polarity_dataset() -> (Vec<PolarityExample>, Vec<PolarityExample>) {
    let mut train = Vec::new();
    let mut dev = Vec::new();
    for i in 0..24 {
        let offset = (i as f32) * 0.02;
        train.push(PolarityExample::new(vec![1.2 + offset, -0.3, 0.1, 0.4], Polarity::Positive));
        train.push(PolarityExample::new(vec![-0.4, 1.3 + offset, -0.2, 0.1], Polarity::Negative));
        train.push(PolarityExample::new(vec![0.0, -0.1, 1.1 + offset, -0.3], Polarity::Neutral));
    }
    for i in 0..8 {
        let offset = (i as f32) * 0.03;
        dev.push(PolarityExample::new(vec![1.1 + offset, -0.2, 0.0, 0.3], Polarity::Positive));
        dev.push(PolarityExample::new(vec![-0.3, 1.2 + offset, -0.1, 0.0], Polarity::Negative));
        dev.push(PolarityExample::new(vec![0.1, 0.0, 1.0 + offset, -0.2], Polarity::Neutral));
    }
    (train, dev)
}

#[test]
fn polarity_vector_head_round_trips_bytes_and_validates_inputs() {
    let enc = encoder_ref("encoder:modernbert");
    let head = PolarityVectorHead::new(
        3,
        vec![1.5, -0.5, 0.0, -0.5, 1.5, 0.0, 0.0, 0.0, 1.5],
        [0.1, -0.1, 0.0],
        Some(enc.clone()),
    )
    .unwrap();
    let bytes = head.to_bytes();
    let decoded = PolarityVectorHead::from_bytes(&bytes).unwrap();
    assert_eq!(decoded.dim(), 3);
    assert_eq!(decoded.weights(), head.weights());
    assert_eq!(decoded.biases(), head.biases());
    assert_eq!(decoded.encoder(), Some(&enc));
    assert_eq!(decoded.model_ref(), head.model_ref());

    let (label, scores) = decoded.predict(&[2.0, -1.0, 0.0]).unwrap();
    assert_eq!(label, Polarity::Positive);
    assert!(scores.positive > scores.negative && scores.positive > scores.neutral);
    assert!(((scores.positive + scores.negative + scores.neutral) - 1.0).abs() < 1e-12);

    assert!(PolarityVectorHead::new(0, vec![], [0.0; 3], None).is_err());
    assert!(PolarityVectorHead::new(2, vec![1.0; 5], [0.0; 3], None).is_err());
    assert!(PolarityVectorHead::new(2, vec![f32::NAN; 6], [0.0; 3], None).is_err());
    assert!(decoded.score_polarity(&[1.0, 2.0]).is_err());
    assert!(decoded.score_polarity(&[1.0, f32::NAN, 0.0]).is_err());
}

#[test]
fn turbo_trainer_fits_polarity_head_and_beats_majority_baseline() {
    let (train, dev) = synthetic_polarity_dataset();
    let config = PolarityFitConfig {
        epochs: 30,
        batch_size: 12,
        learning_rate: 0.08,
        weight_decay: 1e-3,
        warmup_steps: 2,
        clip_norm: 5.0,
        patience: 6,
        seed: 42,
    };
    let enc = encoder_ref("encoder:test");
    let (head, report) = PolarityVectorHead::fit_with_dev(&train, &dev, &config, Some(enc.clone())).unwrap();
    assert!(report.epochs_run >= 1 && report.steps_run >= 1);
    assert!(report.final_train_loss.is_finite());
    assert_eq!(head.encoder(), Some(&enc));

    let train_gold: Vec<Polarity> = train.iter().map(|e| e.polarity).collect();
    let dev_gold: Vec<Polarity> = dev.iter().map(|e| e.polarity).collect();
    let (_, baseline) = majority_baseline(&train_gold, &dev_gold).unwrap();
    let metrics = evaluate_polarity(&head, &dev).unwrap();

    assert!(metrics.accuracy >= 0.95, "expected trained polarity head accuracy >= 0.95, got {:.4}", metrics.accuracy);
    assert!(
        metrics.macro_f1 > baseline.macro_f1 + 0.50,
        "trained macro_f1 {:.4} must beat majority baseline {:.4} by > 0.50",
        metrics.macro_f1,
        baseline.macro_f1
    );
    for polarity in [Polarity::Positive, Polarity::Negative, Polarity::Neutral] {
        assert!(metrics.class(polarity).f1 >= 0.95);
    }
}

#[test]
fn polarity_tsv_parser_handles_canonical_labels_and_rejects_bad_rows() {
    let tsv = "\
# comment line
positive\tThis film was wonderful and deeply moving.
neg\tA tedious, poorly written disaster.
0\tThe screening starts at seven o'clock.
";
    let parsed = parse_polarity_tsv(tsv).unwrap();
    assert_eq!(parsed.len(), 3);
    assert_eq!(parsed[0].0, Polarity::Positive);
    assert_eq!(parsed[1].0, Polarity::Negative);
    assert_eq!(parsed[2].0, Polarity::Neutral);

    assert!(parse_polarity_tsv("unknown\ttext").is_err());
    assert!(parse_polarity_tsv("positive\t   ").is_err());
    assert!(parse_polarity_tsv("no_tab_here").is_err());
}

fn tokenize_whitespace(text: &str) -> (Sentences, Tokens) {
    let mut sent_spans = Spans::new(ViewId::SOURCE);
    let mut tok_spans = Spans::new(ViewId::SOURCE);
    let mut sent_start = 0usize;
    for (idx, ch) in text.char_indices() {
        if matches!(ch, '.' | '!' | '?') {
            let end = idx + ch.len_utf8();
            while sent_start < end && text.as_bytes()[sent_start].is_ascii_whitespace() {
                sent_start += 1;
            }
            if sent_start < end {
                sent_spans.push(Span::new(sent_start as u32, end as u32));
            }
            sent_start = end;
        }
    }
    if sent_start < text.len() {
        while sent_start < text.len() && text.as_bytes()[sent_start].is_ascii_whitespace() {
            sent_start += 1;
        }
        if sent_start < text.len() {
            sent_spans.push(Span::new(sent_start as u32, text.len() as u32));
        }
    }
    let mut in_tok = None;
    for (idx, ch) in text.char_indices() {
        if ch.is_whitespace() {
            if let Some(s) = in_tok.take() {
                tok_spans.push(Span::new(s as u32, idx as u32));
            }
        } else if in_tok.is_none() {
            in_tok = Some(idx);
        }
    }
    if let Some(s) = in_tok {
        tok_spans.push(Span::new(s as u32, text.len() as u32));
    }
    (Sentences { spans: sent_spans }, Tokens { spans: tok_spans, sentence: None })
}

#[test]
fn document_polarity_annotator_scores_lexicon_learned_and_hybrid_modes() {
    let (train, _) = synthetic_polarity_dataset();
    let enc = encoder_ref("encoder:doc");
    let head = Arc::new(
        PolarityVectorHead::fit(&train, &PolarityFitConfig { epochs: 25, ..Default::default() }, Some(enc.clone()))
            .unwrap(),
    );

    let text = "The acting was wonderful! Truly a great masterpiece.";
    let (sents, toks) = tokenize_whitespace(text);
    let mut doc = Document::new(text);
    doc.add(&Key::DEFAULT, sents, Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, toks, Provenance::by("test")).unwrap();
    doc.add(
        &Key::DEFAULT,
        Vectors::<Sentences>::new(4, vec![1.3, -0.3, 0.1, 0.4, 1.4, -0.2, 0.1, 0.5], vec![], enc.clone()),
        Provenance::by("test"),
    )
    .unwrap();

    for ann in [
        PolarityAnnotator::new(),
        PolarityAnnotator::from_head(head.clone()),
        PolarityAnnotator::hybrid(LexiconSentiment::new(), head.clone(), 0.6),
    ] {
        let analyzer = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Tokens>::DEFAULT)
            .given(Key::<Vectors<Sentences>>::DEFAULT)
            .step(ann.bound())
            .build()
            .unwrap();
        let mut docs = [doc.clone()];
        analyzer.run(&mut docs)[0].as_ref().unwrap();
        let cats = docs[0].get_key(&SENTIMENT).unwrap();
        assert_eq!(cats.len(), 1);
        assert_eq!(cats.best_category(0), "positive");
        let sum: f64 = cats.row_scores(0).iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    // Wrong encoder ModelRef is rejected.
    let wrong_vecs = Vectors::<Sentences>::new(
        4,
        vec![1.3, -0.3, 0.1, 0.4, 1.4, -0.2, 0.1, 0.5],
        vec![],
        encoder_ref("encoder:wrong"),
    );
    let learned = PolarityAnnotator::from_head(head.clone());
    assert!(
        learned
            .annotate((
                doc.get::<View>().unwrap(),
                doc.get::<Sentences>(),
                doc.get::<Tokens>(),
                None,
                None,
                Some(&wrong_vecs),
            ))
            .is_err()
    );

    // Mismatched Vectors<Sentences> length vs Sentences length is rejected.
    let one_row_vecs = Vectors::<Sentences>::new(4, vec![1.3, -0.3, 0.1, 0.4], vec![], enc);
    assert!(
        PolarityAnnotator::from_head(head)
            .annotate((
                doc.get::<View>().unwrap(),
                doc.get::<Sentences>(),
                doc.get::<Tokens>(),
                None,
                None,
                Some(&one_row_vecs),
            ))
            .is_err()
    );
}

#[test]
fn sentence_polarity_annotator_writes_opinions_and_sentence_categories() {
    let (train, _) = synthetic_polarity_dataset();
    let enc = encoder_ref("encoder:sent");
    let head = Arc::new(
        PolarityVectorHead::fit(&train, &PolarityFitConfig { epochs: 25, ..Default::default() }, Some(enc.clone()))
            .unwrap(),
    );

    let text = "The service was excellent. The food was terrible.";
    let (sents, toks) = tokenize_whitespace(text);
    assert_eq!(sents.len(), 2);
    let sent_vecs = Vectors::<Sentences>::new(4, vec![1.3, -0.3, 0.1, 0.4, -0.4, 1.4, -0.2, 0.1], vec![], enc);

    let mut doc = Document::new(text);
    doc.add(&Key::DEFAULT, sents, Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, toks, Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, sent_vecs, Provenance::by("test")).unwrap();

    for stage in [PolarityStage::Lexicon, PolarityStage::Learned, PolarityStage::Hybrid] {
        let ann = SentencePolarityAnnotator::new()
            .with_vector_scorer(head.clone())
            .with_head_weight(0.6)
            .with_stage(stage)
            .bound();
        let analyzer = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Tokens>::DEFAULT)
            .given(Key::<Vectors<Sentences>>::DEFAULT)
            .step(ann)
            .build()
            .unwrap();
        let mut docs = [doc.clone()];
        analyzer.run(&mut docs)[0].as_ref().unwrap();
        let cats = docs[0].get_key(&SENTENCE_SENTIMENT).unwrap();
        assert_eq!(cats.len(), 2);
        assert_eq!(cats.best_category(0), "positive");
        assert_eq!(cats.best_category(1), "negative");
        let ops = docs[0].get::<Opinions>().unwrap();
        assert!(ops.len() >= 2, "expected opinion spans from lexicon tokens in stage {stage:?}");
    }
}

#[test]
fn hybrid_zero_head_weight_preserves_vader_compound_decision_on_neutral_heavy_sentence() {
    let enc = encoder_ref("encoder:zero-weight");
    let head = Arc::new(
        PolarityVectorHead::new(2, vec![-2.0, 0.0, 2.0, 0.0, 0.0, 2.0], [0.0, 0.0, 0.0], Some(enc.clone())).unwrap(),
    );
    let text = "The table in the room is good.";
    let (sents, toks) = tokenize_whitespace(text);
    let sent_vecs = Vectors::<Sentences>::new(2, vec![1.0, 0.0], vec![], enc);

    let mut doc = Document::new(text);
    doc.add(&Key::DEFAULT, sents, Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, toks, Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, sent_vecs, Provenance::by("test")).unwrap();

    let (lex_pol, lex_scores) = PolarityAnnotator::new()
        .score_document(
            doc.get::<View>().unwrap(),
            doc.get::<Sentences>(),
            doc.get::<Tokens>(),
            None,
            None,
            doc.get::<Vectors<Sentences>>(),
        )
        .unwrap();
    assert_eq!(lex_pol, Polarity::Positive);
    assert_eq!(lex_scores.label(), Polarity::Positive);

    let (hyb_pol, hyb_scores) = PolarityAnnotator::hybrid(LexiconSentiment::new(), head.clone(), 0.0)
        .score_document(
            doc.get::<View>().unwrap(),
            doc.get::<Sentences>(),
            doc.get::<Tokens>(),
            None,
            None,
            doc.get::<Vectors<Sentences>>(),
        )
        .unwrap();
    assert_eq!(hyb_pol, Polarity::Positive);
    assert_eq!(hyb_scores.label(), Polarity::Positive);

    let sent_ann = SentencePolarityAnnotator::new()
        .with_vector_scorer(head)
        .with_head_weight(0.0)
        .with_stage(PolarityStage::Hybrid);
    let (_, sent_cats) = sent_ann
        .annotate((
            doc.get::<View>().unwrap(),
            doc.get::<Sentences>().unwrap(),
            doc.get::<Tokens>(),
            None,
            None,
            doc.get::<Vectors<Sentences>>(),
        ))
        .unwrap();
    assert_eq!(sent_cats.best_category(0), "positive");
}

#[test]
fn fit_handles_oversized_and_usize_max_batch_size_without_capacity_overflow() {
    let (train, dev) = synthetic_polarity_dataset();
    for batch_size in [usize::MAX, usize::MAX / 2, 1_000_000] {
        let config = PolarityFitConfig { epochs: 2, batch_size, warmup_steps: 0, ..PolarityFitConfig::default() };
        let (head, report) = PolarityVectorHead::fit_with_dev(&train, &dev, &config, None).unwrap();
        assert_eq!(head.dim(), 4);
        assert_eq!(report.epochs_run, 2);
        assert_eq!(report.steps_run, 2);
        assert!(report.final_train_loss.is_finite());
    }
}

#[test]
fn polarity_zoo_loader_resolves_bound_head_and_validates_builtin_modernbert_entry() {
    use turbo_zoo::{Acceptances, Index, Offline, Policy, Sha256, Sha512, UserId, Zoo};
    let dir = std::env::temp_dir().join(format!("turbo-polarity-zoo-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let enc = encoder_ref("answerdotai/ModernBERT-base");
    let head =
        PolarityVectorHead::new(3, vec![1.5, -0.5, 0.0, -0.5, 1.5, 0.0, 0.0, 0.0, 1.5], [0.1, -0.1, 0.0], Some(enc))
            .unwrap();
    let bytes = head.to_bytes();
    let sha256 = Sha256::of(&bytes);
    let sha512 = Sha512::of(&bytes);
    let art_dir = dir.join(MASC_MODERNBERT_POLARITY_HEAD);
    std::fs::create_dir_all(&art_dir).unwrap();
    std::fs::write(art_dir.join("head.tpvh"), &bytes).unwrap();
    let index_json = format!(
        r#"{{
  "zoo_version": 1,
  "artifacts": [
    {{
      "id": "{id}",
      "kind": "KIND_HEAD",
      "component": "COMPONENT_DOCUMENT_CATEGORIZER",
      "language": "en",
      "format": "FORMAT_HEAD",
      "version": "1",
      "license": "Apache-2.0",
      "license_terms": "https://www.apache.org/licenses/LICENSE-2.0",
      "trained_on": [
        {{
          "name": "MASC 3.0.0",
          "license": "CC-BY-3.0-US",
          "license_terms": "https://anc.org/data/masc/"
        }}
      ],
      "class": "CLASS_PERMISSIVE",
      "attribution": "Trained on MASC 3.0.0 (CC BY 3.0 US).",
      "files": [
        {{
          "path": "head.tpvh",
          "url": "https://example.org/models/{id}.tpvh",
          "size": {len},
          "sha256": "{sha256}",
          "sha512": "{sha512}"
        }}
      ]
    }}
  ]
}}"#,
        id = MASC_MODERNBERT_POLARITY_HEAD,
        len = bytes.len(),
        sha256 = sha256,
        sha512 = sha512,
    );
    let index = Index::parse(index_json.as_bytes()).unwrap();
    let zoo = Zoo::new(index, dir.clone(), Policy::default(), &Offline);
    let user: UserId = "test".parse().unwrap();
    let loaded =
        PolarityVectorHead::from_zoo(&zoo, MASC_MODERNBERT_POLARITY_HEAD, &user, || Ok(Acceptances::default()))
            .unwrap();
    assert_eq!(loaded, head);

    let builtin_index = Index::builtin();
    let entry = builtin_index.get(MASC_MODERNBERT_POLARITY_HEAD).unwrap();
    assert_eq!(entry.kind, turbo_zoo::Kind::Head);
    assert_eq!(entry.format, turbo_zoo::Format::Head);
    assert_eq!(entry.class, turbo_zoo::LicenceClass::Permissive);
    let _ = std::fs::remove_dir_all(&dir);
}
