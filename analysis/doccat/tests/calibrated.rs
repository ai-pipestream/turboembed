use std::sync::Arc;

use turbo_analysis::{Annotator, Document, Layer, ModelRef, Provenance, Span, Spans, View, ViewId};
use turbo_analysis_layers::{Polarity, Sentences, Tokens, Vectors};
use turbo_calibration::{Calibration, DataIdentity, Policy, Predictor, Transform, digest, metrics};
use turbo_doccat::calibrated::{CalibratableCategorizer, CalibratedCategorizer, identity};
use turbo_doccat::sentiment::{
    LexiconSentiment, NegationMode, PolarityAnnotator, PolarityExample, PolarityFitConfig, PolarityVectorHead,
};

fn polarity_categorizer() -> PolarityAnnotator {
    PolarityAnnotator::new()
}

fn polarity_head_categorizer(weight: f64) -> PolarityAnnotator {
    let examples = vec![
        PolarityExample::new(vec![1.5, -0.5, 0.0], Polarity::Positive),
        PolarityExample::new(vec![-0.5, 1.5, 0.0], Polarity::Negative),
        PolarityExample::new(vec![0.0, 0.0, 1.5], Polarity::Neutral),
    ];
    let head = PolarityVectorHead::fit(
        &examples,
        &PolarityFitConfig { epochs: 20, batch_size: 3, ..Default::default() },
        Some(ModelRef { id: "enc".into(), revision: "1".into(), sha256: "a".repeat(64) }),
    )
    .unwrap();
    PolarityAnnotator::hybrid(LexiconSentiment::new(), Arc::new(head), weight)
}

fn tokens(text: &str) -> Tokens {
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, text.len() as u32));
    Tokens { spans, sentence: None }
}

fn artifact<C: CalibratableCategorizer>(cat: &C) -> Calibration {
    Calibration::new(
        identity(cat, "reviews").unwrap(),
        Transform::Temperature(2.0),
        DataIdentity { split: "fit".into(), sha256: digest(b"fit") },
        DataIdentity { split: "conf".into(), sha256: digest(b"conf") },
        vec![0.1; 20],
    )
    .unwrap()
}

#[test]
fn raw_polarity_outputs_survive_abstention() {
    let cat = polarity_categorizer();
    let doc = Document::new("wonderful");
    let view = doc.get::<View>().unwrap();
    let t = tokens("wonderful");
    let (raw,) = cat.annotate((view, None, Some(&t), None, None, None)).unwrap();
    let a = artifact(&cat);
    let loaded = Calibration::from_bytes(&a.to_bytes(), &identity(&cat, "reviews").unwrap()).unwrap();
    let wrap = CalibratedCategorizer::new(
        cat,
        loaded,
        "reviews",
        Policy { coverage: 0.9, max_set_size: 1, min_confidence: 1.0 },
    )
    .unwrap();
    let (unchanged, side) = wrap.annotate((view, None, Some(&t), None, None, None)).unwrap();
    side.check().unwrap();
    assert_eq!(raw.iter().next().unwrap().scores, unchanged.iter().next().unwrap().scores);
    assert_eq!(side.row(0).raw, raw.iter().next().unwrap().scores);
    assert_ne!(side.row(0).raw, side.row(0).calibrated);
    assert!(side.row(0).abstention.is_some());
    assert_eq!(unchanged.len(), 1);
    assert_eq!(wrap.config_hash(), wrap.predictor().config_hash());
}

#[test]
fn wrong_domain_or_producer_cannot_bind() {
    let cat = polarity_categorizer();
    let a = artifact(&cat);
    assert!(
        CalibratedCategorizer::new(
            cat,
            a.clone(),
            "different",
            Policy { coverage: 0.9, max_set_size: 3, min_confidence: 0.0 }
        )
        .err()
        .unwrap()
        .contains(&a.id())
    );
    let window_cat =
        PolarityAnnotator::new().with_lexicon(LexiconSentiment::new().with_negation_mode(NegationMode::Window));
    assert!(
        CalibratedCategorizer::new(
            window_cat,
            a,
            "reviews",
            Policy { coverage: 0.9, max_set_size: 3, min_confidence: 0.0 }
        )
        .is_err()
    );
}

#[test]
fn heldout_receipts_for_categorizer() {
    let cat = polarity_categorizer();
    let id = identity(&cat, "synthetic-reviews").unwrap();
    let mut rows = Vec::new();
    let mut gold = Vec::new();
    let mut seed = 998u64;
    for i in 0..300 {
        let word = match i % 3 {
            0 => "wonderful",
            1 => "terrible",
            _ => "table",
        };
        let doc = Document::new(word);
        let t = tokens(word);
        let (raw,) = cat.annotate((doc.get::<View>().unwrap(), None, Some(&t), None, None, None)).unwrap();
        let p = raw.iter().next().unwrap().scores.to_vec();
        let best = i % 3;
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let noisy = if (seed >> 32) % 10 >= 7 { (best + 1) % 3 } else { best };
        gold.push(noisy);
        rows.push(p);
    }

    for transform in [
        Transform::Temperature(2.5),
        Transform::Platt { slope: 0.45, intercept: 0.0 },
        Transform::Isotonic { upper: vec![0.3, 0.7, 1.0], value: vec![0.25, 0.35, 0.65] },
    ] {
        let a = Calibration::new(
            id.clone(),
            transform,
            DataIdentity { split: "fit".into(), sha256: digest(b"fit") },
            DataIdentity { split: "conformal".into(), sha256: digest(b"conformal") },
            vec![0.85; 100],
        )
        .unwrap();
        let p = Predictor::new(a, &id, Policy { coverage: 0.95, max_set_size: 3, min_confidence: 0.0 }).unwrap();
        let predictions: Vec<_> = rows.iter().map(|r| p.predict(r).unwrap()).collect();
        let before = metrics::reliability(&rows, &gold, 10).unwrap();
        let after =
            metrics::reliability(&predictions.iter().map(|r| r.calibrated.clone()).collect::<Vec<_>>(), &gold, 10)
                .unwrap();
        let coverage = metrics::coverage(&predictions, &gold).unwrap();
        assert!(after.ece < before.ece, "{before:?} {after:?}");
        assert!(coverage.set_coverage >= 0.90);
    }
}

#[test]
fn producer_feature_configuration_is_bound() {
    let cat = polarity_head_categorizer(0.4);
    let original = identity(&cat, "reviews").unwrap();
    let changed = polarity_head_categorizer(0.8);
    let expected = identity(&changed, "reviews").unwrap();
    assert_eq!(original.model, expected.model);
    assert_eq!(original.revision, expected.revision);
    assert_eq!(original.labels, expected.labels);
    assert_ne!(original.sha256, expected.sha256);

    let doc = Document::new("wonderful");
    let view = doc.get::<View>().unwrap();
    let t = tokens("wonderful");
    let vecs = Vectors::<Sentences>::new(
        3,
        vec![-0.5, 1.5, 0.0],
        vec![],
        ModelRef { id: "enc".into(), revision: "1".into(), sha256: "a".repeat(64) },
    );
    let (scores_a,) = cat.annotate((view, None, Some(&t), None, None, Some(&vecs))).unwrap();
    let (scores_b,) = changed.annotate((view, None, Some(&t), None, None, Some(&vecs))).unwrap();
    assert_ne!(
        scores_a.row_scores(0),
        scores_b.row_scores(0),
        "changed hybrid head weight must affect producer probabilities"
    );
    assert!(artifact(&cat).bind(&expected).is_err());
    assert!(
        CalibratedCategorizer::new(
            changed,
            artifact(&cat),
            "reviews",
            Policy { coverage: 0.9, max_set_size: 3, min_confidence: 0.0 },
        )
        .is_err()
    );
}

#[test]
fn analyzer_provenance_and_document_graph_keep_the_calibrated_sidecar() {
    use turbo_analysis::export::{export, import};
    use turbo_analysis::{Analyzer, Key};
    use turbo_analysis_layers::{Calibrated, Categories, export::codecs};
    let cat = polarity_categorizer();
    let a = artifact(&cat);
    let wrapped =
        CalibratedCategorizer::new(cat, a, "reviews", Policy { coverage: 0.9, max_set_size: 1, min_confidence: 0.99 })
            .unwrap();
    let analyzer = Analyzer::builder().given(Key::<Tokens>::DEFAULT).add(wrapped).build().unwrap();
    let mut doc = Document::new("wonderful");
    doc.add(&Key::DEFAULT, tokens("wonderful"), Provenance::by("test")).unwrap();
    let mut docs = [doc];
    analyzer.run(&mut docs)[0].as_ref().unwrap();
    let side = docs[0].get::<Calibrated<Categories>>().unwrap();
    assert!(side.row(0).abstention.is_some());
    let graph = export(&docs[0], &codecs()).unwrap();
    let back = import(&graph, &codecs()).unwrap();
    assert_eq!(
        back.get::<Calibrated<Categories>>().unwrap().rows().collect::<Vec<_>>(),
        side.rows().collect::<Vec<_>>()
    );
}
