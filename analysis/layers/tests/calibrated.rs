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
    side.rows.push(p.predict(&[0.8, 0.2]).unwrap());
    doc.add(&Key::DEFAULT, side.clone(), Provenance::by("test")).unwrap();
    let g = export(&doc, &codecs()).unwrap();
    let bytes = g.to_bytes().unwrap();
    let back = import(&DocumentGraph::from_bytes(&bytes).unwrap(), &codecs()).unwrap();
    let b = back.get::<Calibrated<Tokens>>().unwrap();
    assert_eq!(b.identity, side.identity);
    assert_eq!(b.calibration_id, side.calibration_id);
    assert_eq!(b.rows, side.rows);
    assert_eq!(b.policy, side.policy);
    assert_eq!(b.cutoff, side.cutoff);
    assert_eq!(export(&back, &codecs()).unwrap().to_bytes().unwrap(), bytes);
}
#[test]
fn inconsistent_decisions_and_nonfinite_metadata_are_rejected() {
    let p = predictor();
    let mut side = Calibrated::<Tokens>::new(&p);
    side.rows.push(p.predict(&[0.8, 0.2]).unwrap());
    side.check().unwrap();
    let mut bad = side.clone();
    bad.rows[0].abstention = None;
    assert!(bad.check().is_err());
    let mut bad = side.clone();
    bad.rows[0].set = vec![1];
    assert!(bad.check().is_err());
    let mut bad = side.clone();
    bad.cutoff = f64::NAN;
    assert!(bad.check().is_err());
    let mut bad = side.clone();
    bad.policy.min_confidence = f64::INFINITY;
    assert!(bad.check().is_err());
    let mut bad = side.clone();
    std::sync::Arc::make_mut(&mut bad.identity).labels.reverse(); // Labels can be different but remain unique.
    std::sync::Arc::make_mut(&mut bad.identity).labels[1] = bad.identity.labels[0].clone();
    assert!(bad.check().is_err());
    let mut bad = side;
    bad.rows[0].calibrated[0] = f64::NAN;
    assert!(bad.check().is_err());
}
