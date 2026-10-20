use std::sync::Arc;

use super::*;
use crate::layer::{Key, Provenance};
use crate::text::ViewId;
use crate::{Span, Spans};

/// A layer of marks over the source, to support evidence.
struct Marks {
    spans: Spans<Marks>,
}

impl Layer for Marks {
    const ID: &'static str = "test:marks";
    fn anchor() -> Anchor {
        Anchor::Document
    }
    fn len(&self) -> usize {
        self.spans.len()
    }
}

fn model() -> ModelRef {
    ModelRef { id: "tagger".into(), revision: "3".into(), sha256: "ab".into() }
}

/// "héllo wörld" with two marks, made by a step that ran `model()`.
fn doc() -> Document {
    let mut doc = Document::new("héllo wörld");
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 6));
    spans.push(Span::new(7, 13));
    let p = Provenance { model: Some(model()), ..Provenance::by("test:marker") };
    doc.add(&Key::DEFAULT, Marks { spans }, p).unwrap();
    doc
}

/// One record per span, produced by `producer`, supported by mark `i`.
fn evidence(producer: Producer, spans: &[(u32, u32)]) -> Evidence<Marks> {
    let n = spans.len() as u32;
    Evidence {
        producers: Arc::new(vec![producer]),
        producer: vec![0; spans.len()],
        span_row: (0..=n).collect(),
        spans: spans.iter().map(|&(a, b)| Span::new(a, b)).collect(),
        raw: (0..n).map(|i| (i, 0.5)).collect(),
        calibrated: Vec::new(),
        abstention: Vec::new(),
        layers: Arc::new(vec![LayerRef { layer: "test:marks".into(), instance: String::new() }]),
        support_row: (0..=n).collect(),
        support: (0..n).map(|i| Support { layer: 0, start: i, end: i + 1 }).collect(),
        _of: PhantomData,
    }
}

fn marker() -> Producer {
    Producer { annotator: "test:marker".into(), model: Some(model()) }
}

#[test]
fn evidence_that_describes_its_document_verifies() {
    let ev = evidence(marker(), &[(0, 6), (7, 13)]);
    assert_eq!(ev.verify(&doc()), Ok(()));
    assert_eq!((ev.records(), ev.spans_of(1), ev.producer_of(0)), (2, &[Span::new(7, 13)][..], &marker()));
    // A rule-backed producer names no model and needs none to have run.
    let rule = Producer { annotator: "test:rule".into(), model: None };
    assert_eq!(evidence(rule, &[(1, 3)]).verify(&doc()), Ok(()));
}

#[test]
fn a_model_that_did_not_run_is_refused() {
    let other = ModelRef { revision: "4".into(), ..model() };
    let ev = evidence(Producer { annotator: "test:marker".into(), model: Some(other.clone()) }, &[(0, 6)]);
    assert_eq!(ev.verify(&doc()), Err(EvidenceError::UnknownModel { record: Some(0), model: other }));
}

#[test]
fn spans_outside_the_source_or_inside_a_character_are_refused() {
    // The source is 13 bytes: "é" is bytes 1 and 2, "ö" bytes 8 and 9.
    let ev = evidence(marker(), &[(0, 6), (7, 14)]);
    assert_eq!(ev.verify(&doc()), Err(EvidenceError::SpanOutsideSource { record: 1, span: Span::new(7, 14), len: 13 }));
    let ev = evidence(marker(), &[(2, 6)]);
    assert_eq!(ev.verify(&doc()), Err(EvidenceError::SpanOffCharBoundary { record: 0, span: Span::new(2, 6) }));
}

#[test]
fn support_outside_the_documents_layers_is_refused() {
    let mut ev = evidence(marker(), &[(0, 6), (7, 13)]);
    ev.support[1] = Support { layer: 0, start: 1, end: 3 };
    let marks = LayerRef { layer: "test:marks".into(), instance: String::new() };
    assert_eq!(ev.verify(&doc()), Err(EvidenceError::SupportOutOfRange { record: 1, layer: marks, end: 3, len: 2 }));
    let mut ev = evidence(marker(), &[(0, 6)]);
    Arc::make_mut(&mut ev.layers)[0].instance = "gold".into();
    let gold = LayerRef { layer: "test:marks".into(), instance: "gold".into() };
    assert_eq!(ev.verify(&doc()), Err(EvidenceError::UnknownSupport { record: 0, layer: gold }));
}

/// A change that breaks a column.
type Breaks = fn(&mut Evidence<Marks>);

#[test]
fn columns_that_do_not_hold_together_are_refused() {
    let broken: [(&str, Breaks); 7] = [
        ("raw scores out of record order", |e| e.raw.swap(0, 1)),
        ("span rows past the spans", |e| e.span_row[2] = 3),
        ("span rows out of order", |e| e.span_row = vec![0, 3, 2]),
        ("a producer past its table", |e| e.producer[1] = 1),
        ("support past its layer table", |e| e.support[0].layer = 1),
        ("a NaN raw score", |e| e.raw[0].1 = f64::NAN),
        ("a calibrated score past one", |e| e.calibrated.push((1, 1.5))),
    ];
    for (what, f) in broken {
        let mut ev = evidence(marker(), &[(0, 6), (7, 13)]);
        f(&mut ev);
        assert!(ev.validate().is_err(), "{what}");
        assert!(matches!(ev.verify(&doc()), Err(EvidenceError::Invalid(_))), "{what}");
    }
    let mut ev = evidence(marker(), &[(0, 6), (7, 13)]);
    ev.calibrated.push((1, 1.0));
    ev.abstention.push((1, "below the threshold".into()));
    assert_eq!(ev.validate(), Ok(()));
    assert_eq!((ev.raw_of(0), ev.calibrated_of(0), ev.calibrated_of(1)), (Some(0.5), None, Some(1.0)));
    assert_eq!((ev.abstention_of(0), ev.abstention_of(1)), (None, Some("below the threshold")));
    ev.abstention.push((2, "past the records".into()));
    assert!(ev.validate().is_err());
}
