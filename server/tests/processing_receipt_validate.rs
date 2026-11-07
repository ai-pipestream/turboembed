//! `RoundTripTypedDocument` refuses what the receipt contract forbids
//! (issue #1908): an UNSPECIFIED or unknown layer or view status, an error
//! on an OK layer or none on a skipped one, a missing or malformed receipt,
//! one naming another input or result, and assessed ranges that are not
//! sorted, disjoint and in bounds. The builder normalizes what producers
//! report into that shape.

use turbo_analysis::{Document, Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{Pii, Sentences};
use turbo_kserve::offsets::OffsetMode;
use turbo_kserve::proto::result::{
    ByteRange, LayerError, LayerErrorKind, LayerWireStatus, ProcessingStatus, TruncationReason, TypedDocumentResult,
};
use turbo_kserve::results::receipt::{self, Coverage, Operation};
use turbo_kserve::results::typed::identity::input_identity;
use turbo_kserve::results::typed::{document_to_typed_bound, round_trip_typed_document};

const TEXT: &str = "Nothing sensitive in this sentence.";

fn doc() -> Document {
    let mut d = Document::new(TEXT);
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, TEXT.len() as u32));
    d.add(&Key::DEFAULT, Sentences { spans }, Provenance::by("test:sent")).unwrap();
    d.add(&Key::DEFAULT, Pii::new(ViewId::SOURCE), Provenance::by("test:pii")).unwrap();
    d
}

fn typed() -> TypedDocumentResult {
    let t = document_to_typed_bound("d", &doc(), ViewId::SOURCE, OffsetMode::Utf8, None, "op-1").unwrap();
    round_trip_typed_document(&t).expect("the server's own answer validates");
    t
}

fn refused(t: &TypedDocumentResult, what: &str) {
    assert!(round_trip_typed_document(t).is_err(), "accepted: {what}");
}

fn error(kind: LayerErrorKind) -> Option<LayerError> {
    Some(LayerError { kind: kind as i32, ..LayerError::default() })
}

#[test]
fn layer_and_view_status_presence_is_enforced() {
    let base = typed();
    let ix = base.layers.iter().position(|l| l.id == "turbo:pii").unwrap();
    for status in [0, 99] {
        let mut t = base.clone();
        t.layers[ix].status = status;
        refused(&t, "layer status 0 or unknown");
        let mut t = base.clone();
        t.views[0].status = status;
        refused(&t, "view status 0 or unknown");
    }
    let mut t = base.clone();
    t.layers[ix].error = error(LayerErrorKind::Annotator);
    refused(&t, "an OK layer with an error");
    let mut t = base.clone();
    t.views[0].error = error(LayerErrorKind::Declined);
    refused(&t, "an OK view with an error");
    for e in [None, error(LayerErrorKind::Unspecified), Some(LayerError { kind: 99, ..LayerError::default() })] {
        let mut t = base.clone();
        t.layers[ix].status = LayerWireStatus::Skipped as i32;
        t.layers[ix].error = e;
        refused(&t, "a skipped layer without a known error");
    }
}

#[test]
fn the_receipt_is_required_and_bound_to_the_envelope() {
    let base = typed();
    let mut cases: Vec<(&str, TypedDocumentResult)> = Vec::new();
    let mut push = |what, f: &dyn Fn(&mut TypedDocumentResult)| {
        let mut t = base.clone();
        f(&mut t);
        cases.push((what, t));
    };
    push("no receipt", &|t| t.receipt = None);
    push("schema version", &|t| t.receipt.as_mut().unwrap().receipt_schema_version = 2);
    push("operation key", &|t| t.receipt.as_mut().unwrap().operation_key = "has space".into());
    push("attempt id", &|t| t.receipt.as_mut().unwrap().attempt_id = "XYZ".into());
    push("upper-case attempt id", &|t| {
        let r = t.receipt.as_mut().unwrap();
        r.attempt_id = r.attempt_id.to_uppercase().replace(|c: char| c.is_ascii_digit(), "A");
    });
    push("status 0", &|t| t.receipt.as_mut().unwrap().status = 0);
    push("status unknown", &|t| t.receipt.as_mut().unwrap().status = 99);
    push("status disagrees", &|t| t.receipt.as_mut().unwrap().status = ProcessingStatus::Partial as i32);
    push("config digest", &|t| t.receipt.as_mut().unwrap().config_digest = vec![0; 31]);
    push("other input", &|t| t.receipt.as_mut().unwrap().input.as_mut().unwrap().source_text_len += 1);
    push("other result", &|t| t.receipt.as_mut().unwrap().result.as_mut().unwrap().content_digest[0] ^= 1);
    push("no result", &|t| t.receipt.as_mut().unwrap().result = None);
    push("cancelled with a result", &|t| t.receipt.as_mut().unwrap().status = ProcessingStatus::Cancelled as i32);
    push("item count", &|t| t.receipt.as_mut().unwrap().item_count = 5);
    push("layer status 0", &|t| t.receipt.as_mut().unwrap().layers[1].status = 0);
    push("layer truncation 0", &|t| t.receipt.as_mut().unwrap().layers[1].truncations = vec![0]);
    push("layer digest", &|t| t.receipt.as_mut().unwrap().layers[1].provenance_digest = Some(vec![1; 3]));
    push("complete with an error", &|t| t.receipt.as_mut().unwrap().layers[1].error = error(LayerErrorKind::Pipeline));
    push("not assessed with ranges", &|t| {
        t.receipt.as_mut().unwrap().layers[1].status = ProcessingStatus::NotAssessed as i32
    });
    for (what, t) in &cases {
        refused(t, what);
    }
}

#[test]
fn assessed_ranges_must_be_sorted_disjoint_and_in_bounds() {
    let len = TEXT.len() as u32;
    let base = typed();
    for bad in [
        vec![ByteRange { start: 10, end: len }, ByteRange { start: 0, end: 10 }],
        vec![ByteRange { start: 0, end: 20 }, ByteRange { start: 10, end: len }],
        vec![ByteRange { start: 0, end: len + 1 }],
        vec![ByteRange { start: 0, end: 0 }, ByteRange { start: 0, end: len }],
    ] {
        let mut t = base.clone();
        t.receipt.as_mut().unwrap().layers[1].assessed = bad.clone();
        refused(&t, &format!("assessed {bad:?}"));
    }

    // What a producer reports is normalized: clamped, sorted and merged.
    let d = doc();
    let messy = Coverage {
        layer: "turbo:pii".into(),
        instance: String::new(),
        assessed: vec![
            ByteRange { start: 20, end: len + 50 },
            ByteRange { start: 0, end: 12 },
            ByteRange { start: 5, end: 20 },
            ByteRange { start: 3, end: 3 },
        ],
        truncations: vec![TruncationReason::Unspecified],
    };
    let r = receipt::build(&Operation::start("k"), &d, input_identity(&d, None), None, &[messy]);
    let pii = r.layers.iter().find(|l| l.layer == "turbo:pii").unwrap();
    assert_eq!(pii.assessed, vec![ByteRange { start: 0, end: len }]);
    assert!(pii.truncations.is_empty());
    assert_eq!(pii.status, ProcessingStatus::Complete as i32, "overlaps that cover the text are whole");
    let gap = Coverage {
        layer: "turbo:pii".into(),
        instance: String::new(),
        assessed: vec![
            ByteRange { start: 0, end: 10 },
            ByteRange { start: 0, end: 5 },
            ByteRange { start: 12, end: len },
        ],
        truncations: Vec::new(),
    };
    let r = receipt::build(&Operation::start("k"), &d, input_identity(&d, None), None, &[gap]);
    let pii = r.layers.iter().find(|l| l.layer == "turbo:pii").unwrap();
    assert_eq!(pii.assessed, vec![ByteRange { start: 0, end: 10 }, ByteRange { start: 12, end: len }]);
    assert_eq!(pii.status, ProcessingStatus::Partial as i32, "a gap is partial even with an end at the text's end");
}
