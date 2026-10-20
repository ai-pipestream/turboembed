use turbo_analysis::export::{DocumentGraph, export, import};
use turbo_analysis::{Document, Key, Layer, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{Calibrated, Tokens, export::codecs};
use turbo_calibration::*;
fn predictor() -> Predictor {
    let id = Identity {
        model: "m".into(),
        revision: "1".into(),
        sha256: digest(b"model"),
        task: Task::TokenTagging,
        domain: "news".into(),
        labels: vec!["name".into(), "other".into()],
        inference_hash: 1,
    };
    let a = Calibration::new(
        id.clone(),
        Transform::Temperature(2.0),
        DataIdentity { split: "fit".into(), sha256: digest(b"fit") },
        DataIdentity { split: "conf".into(), sha256: digest(b"conf") },
        vec![0.5; 10],
    )
    .unwrap();
    Predictor::new(a, &id, Policy { coverage: 0.9, max_set_size: 1, min_confidence: 0.9 }).unwrap()
}
#[test]
fn graph_roundtrip_retains_scores_policy_identity_and_reason() {
    let p = predictor();
    let mut doc = Document::new("Alice");
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 5));
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("test")).unwrap();
    let mut side = Calibrated::<Tokens>::new(&p);
    side.push(&p, &[0.8, 0.2]).unwrap();
    doc.add(&Key::DEFAULT, side.clone(), Provenance::by("test")).unwrap();
    let g = export(&doc, &codecs()).unwrap();
    let bytes = g.to_bytes().unwrap();
    let back = import(&DocumentGraph::from_bytes(&bytes).unwrap(), &codecs()).unwrap();
    let b = back.get::<Calibrated<Tokens>>().unwrap();
    assert_eq!(b.metadata(), side.metadata());
    assert_eq!(b.rows().collect::<Vec<_>>(), side.rows().collect::<Vec<_>>());
    assert_eq!(export(&back, &codecs()).unwrap().to_bytes().unwrap(), bytes);
}
#[test]
fn inconsistent_decisions_and_nonfinite_metadata_are_rejected() {
    let p = predictor();
    let mut side = Calibrated::<Tokens>::new(&p);
    side.push(&p, &[0.8, 0.2]).unwrap();
    side.check().unwrap();
    let reject = |row| assert!(Calibrated::<Tokens>::from_rows(p.shared_metadata(), [row]).is_err());
    let row = p.predict(&[0.8, 0.2]).unwrap();
    let mut bad = row.clone();
    bad.abstention = None;
    reject(bad);
    let mut bad = row.clone();
    bad.set = vec![1];
    reject(bad);
    let mut bad = row;
    bad.calibrated[0] = f64::NAN;
    reject(bad);
    let m = p.metadata();
    let make = |identity, policy, cutoff| DecisionMetadata::new(identity, m.calibration_id().into(), policy, cutoff);
    assert!(make(std::sync::Arc::new(m.identity().clone()), m.policy(), f64::NAN).is_err());
    let mut policy = m.policy();
    policy.min_confidence = f64::INFINITY;
    assert!(make(std::sync::Arc::new(m.identity().clone()), policy, m.cutoff()).is_err());
    let mut identity = m.identity().clone();
    identity.labels[1] = identity.labels[0].clone();
    assert!(make(std::sync::Arc::new(identity), m.policy(), m.cutoff()).is_err());
}

#[test]
fn flat_rows_preserve_small_row_and_failed_append_is_atomic() {
    let p = predictor();
    let mut side = Calibrated::<Tokens>::new(&p);
    let inputs = [[0.8, 0.2], [0.5, 0.5], [1.0, 0.0]];
    for raw in inputs {
        side.push(&p, &raw).unwrap();
        let n = side.len();
        assert!(side.push(&p, &[f64::NAN, 0.0]).is_err());
        assert_eq!(side.len(), n);
    }
    for (row, raw) in side.rows().zip(inputs) {
        let expected = p.predict(&raw).unwrap();
        assert_eq!(row.raw, expected.raw);
        assert_eq!(row.calibrated, expected.calibrated);
        assert_eq!(row.set, expected.set);
        assert_eq!(row.abstention, expected.abstention);
    }
}
