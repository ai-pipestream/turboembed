use std::sync::Arc;
use turbo_analysis::{Annotator, Document, Layer, Span, Spans, View, ViewId};
use turbo_analysis_layers::{Sentences, Tokens};
use turbo_calibration::{Calibration, DataIdentity, Policy, Transform, digest, metrics};
use turbo_maxent::ModelFile;
use turbo_namefind::calibrated::{CalibratedNameAnnotator, identity};
use turbo_sequence::{
    PatternMode,
    namefind::{MODEL_ENTRY_NAME, NameFinder},
};
use turbo_train::{
    Event, Key, Manifest, ModelPackage, TrainingConfiguration, TrainingParameters,
    calibration::{Method, Samples, fit},
};

fn file() -> Arc<ModelFile> {
    let events = [
        Event::from_strs("person-start", &["w=Alice"]),
        Event::from_strs("person-start", &["w=Bob"]),
        Event::from_strs("other", &["w=walked"]),
        Event::from_strs("other", &["w=home"]),
    ];
    let mut params = TrainingParameters::new();
    params.put(Key::ITERATIONS, 50);
    params.put(Key::CUTOFF, 0);
    let (trained, _) = turbo_train::gis::train_events(&events, &params, &mut TrainingConfiguration::default()).unwrap();
    let mut package = ModelPackage::new(Manifest::new("TokenNameFinder", "en", 1700000000000));
    package.add_entry(MODEL_ENTRY_NAME, turbo_train::write_gis(&trained).unwrap());
    Arc::new(ModelFile::open(package.to_bytes()).unwrap())
}
fn artifact(file: &ModelFile) -> Calibration {
    Calibration::new(
        identity(file, "news").unwrap(),
        Transform::Temperature(2.0),
        DataIdentity { split: "fit".into(), sha256: digest(b"fit") },
        DataIdentity { split: "conf".into(), sha256: digest(b"conf") },
        vec![0.1; 20],
    )
    .unwrap()
}
fn inputs() -> (Document, Sentences, Tokens) {
    let doc = Document::new("Alice walked");
    let mut s = Spans::new(ViewId::SOURCE);
    s.push(Span::new(0, 12));
    let mut t = Spans::new(ViewId::SOURCE);
    t.push(Span::new(0, 5));
    t.push(Span::new(6, 12));
    (doc, Sentences { spans: s }, Tokens { spans: t, sentence: None })
}
#[test]
fn scored_search_preserves_names_probabilities_and_adaptive_state() {
    let f = file();
    let model = f.maxent(MODEL_ENTRY_NAME).unwrap();
    let mut raw = NameFinder::new(&f, &model, PatternMode::Utf16Units).unwrap();
    let mut scored = NameFinder::new(&f, &model, PatternMode::Utf16Units).unwrap();
    for words in [vec!["Alice", "walked"], vec!["Bob", "Alice", "home"], vec![], vec!["Alice"]] {
        let a = raw.find(&words);
        let (b, rows) = scored.find_scored(&words);
        assert_eq!(a, b);
        assert_eq!(raw.probs(), scored.probs());
        assert_eq!(rows.len(), words.len());
        for row in rows {
            turbo_calibration::check_distribution(&row).unwrap();
        }
    }
}
#[test]
fn name_annotator_preserves_raw_entities_and_validates_boundaries() {
    let f = file();
    let a = artifact(&f);
    let model = f.maxent(MODEL_ENTRY_NAME).unwrap();
    let mut raw = NameFinder::new(&f, &model, PatternMode::Utf16Units).unwrap();
    let expected = raw.find(&["Alice", "walked"]);
    let ann = CalibratedNameAnnotator::new(
        f.clone(),
        a.clone(),
        "news",
        Policy { coverage: 0.9, max_set_size: 1, min_confidence: 1.0 },
    )
    .unwrap();
    let (doc, s, mut t) = inputs();
    let v = doc.get::<View>().unwrap();
    let (entities, side) = ann.annotate((v, &s, &t)).unwrap();
    side.check().unwrap();
    assert_eq!(side.len(), 2);
    assert_eq!(entities.len(), expected.len());
    for (i, n) in expected.iter().enumerate() {
        assert_eq!(entities.score[i], n.prob);
        assert_eq!(entities.tokens[i].start.get(), n.span.start);
    }
    assert!(side.rows.iter().all(|r| r.abstention.is_some()));
    t.spans = Spans::new(ViewId(99));
    assert!(ann.annotate((v, &s, &t)).is_err());
    let bad = Document::new("é");
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 1));
    let mut sent = Spans::new(ViewId::SOURCE);
    sent.push(Span::new(0, 2));
    assert!(
        ann.annotate((bad.get::<View>().unwrap(), &Sentences { spans: sent }, &Tokens { spans, sentence: None }))
            .is_err()
    );
    assert!(
        CalibratedNameAnnotator::new(
            f,
            a,
            "other-domain",
            Policy { coverage: 0.9, max_set_size: 1, min_confidence: 0.0 }
        )
        .is_err()
    );
}
#[test]
fn heldout_token_calibration_receipt() {
    let f = file();
    let model = f.maxent(MODEL_ENTRY_NAME).unwrap();
    let generate = |mut seed: u64, n: usize| {
        let mut finder = NameFinder::new(&f, &model, PatternMode::Utf16Units).unwrap();
        let mut rows = Vec::new();
        let mut gold = Vec::new();
        for i in 0..n {
            finder.clear_adaptive_data();
            let words = if i % 2 == 0 { ["Alice"] } else { ["walked"] };
            let (_, mut scored) = finder.find_scored(&words);
            let row = scored.pop().unwrap();
            let best = usize::from(row[1] > row[0]);
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            gold.push(best ^ usize::from((seed >> 32) % 10 >= 7));
            rows.push(row);
        }
        (rows, gold)
    };
    let (fitting, fy) = generate(1, 300);
    let (conf, cy) = generate(123, 301);
    let (test, ty) = generate(998, 1000);
    let id = identity(&f, "synthetic-tokens").unwrap();
    let a = fit(
        id.clone(),
        Method::Temperature,
        Samples { split: "fit", probabilities: &fitting, gold: &fy },
        Samples { split: "conformal", probabilities: &conf, gold: &cy },
    )
    .unwrap();
    let p = turbo_calibration::Predictor::new(a, &id, Policy { coverage: 0.9, max_set_size: 2, min_confidence: 0.0 })
        .unwrap();
    let predictions: Vec<_> = test.iter().map(|r| p.predict(r).unwrap()).collect();
    let before = metrics::reliability(&test, &ty, 10).unwrap();
    let after =
        metrics::reliability(&predictions.iter().map(|r| r.calibrated.clone()).collect::<Vec<_>>(), &ty, 10).unwrap();
    let c = metrics::coverage(&predictions, &ty).unwrap();
    assert!(after.ece < before.ece);
    assert!(c.set_coverage >= 0.87);
    println!(
        "NameFinderME: ECE {:.6} -> {:.6}; Brier {:.6} -> {:.6}; coverage {:.6}",
        before.ece, after.ece, before.brier, after.brier, c.set_coverage
    );
}
