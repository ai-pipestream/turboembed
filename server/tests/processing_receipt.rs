//! The processing receipt's lifecycle (issue #1908): no findings, not
//! assessed, partial, failed and cancelled are told apart; a retry keeps
//! the operation key and gets a new attempt id; and no receipt or layer
//! error carries a planted value or source text.

use prost::Message as _;
use turbo_analysis::export::export;
use turbo_analysis::{Document, Error, Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{ChecksumAlgorithm, Pii, PiiEvidence, PiiKind, PiiMention, Sentences, Tokens};
use turbo_kserve::offsets::OffsetMode;
use turbo_kserve::proto::result::{
    ByteRange, InspectDocumentGraphRequest, LayerErrorKind, LayerReceipt, LayerWireStatus, ProcessingReceipt,
    ProcessingStatus, TruncationReason, TypedDocumentResult,
};
use turbo_kserve::results::disclosure::Disclosure;
use turbo_kserve::results::document::codecs;
use turbo_kserve::results::receipt::{self, Coverage, Operation};
use turbo_kserve::results::typed::identity::input_identity;
use turbo_kserve::results::typed::{document_to_typed_bound, inspect_document_graph};

/// A value no receipt or error may carry: it is in the source text, a
/// PII value, a skip reason and an annotator's failure detail.
const PLANTED: &str = "4111-1111-1111-1111";
const TEXT: &str = "Card 4111-1111-1111-1111 was used. Nothing else.";

fn typed(doc: &Document, key: &str) -> TypedDocumentResult {
    document_to_typed_bound("doc", doc, ViewId::SOURCE, OffsetMode::Utf8, None, key, &Disclosure::FULL).unwrap()
}

fn receipt_of(t: &TypedDocumentResult) -> &ProcessingReceipt {
    t.receipt.as_ref().expect("every typed answer carries its receipt")
}

fn layer<'a>(r: &'a ProcessingReceipt, id: &str) -> &'a LayerReceipt {
    r.layers.iter().find(|l| l.layer == id).unwrap_or_else(|| panic!("no {id} on the receipt"))
}

fn status(r: &ProcessingReceipt) -> ProcessingStatus {
    ProcessingStatus::try_from(r.status).unwrap()
}

fn with_sentences(text: &str) -> Document {
    let mut doc = Document::new(text);
    let len = text.len() as u32;
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, len));
    doc.add(&Key::DEFAULT, Sentences { spans }, Provenance::by("test:sentences")).unwrap();
    doc
}

#[test]
fn complete_with_no_findings_is_explicit() {
    let mut doc = with_sentences(TEXT);
    doc.add(&Key::DEFAULT, Pii::new(ViewId::SOURCE), Provenance::by("test:pii")).unwrap();
    let t = typed(&doc, "op-1");
    let r = receipt_of(&t);
    assert_eq!(status(r), ProcessingStatus::Complete);
    let pii = layer(r, "turbo:pii");
    assert_eq!(pii.status, ProcessingStatus::Complete as i32);
    assert_eq!(pii.item_count, 0, "assessed and nothing found");
    assert_eq!(pii.assessed, vec![ByteRange { start: 0, end: TEXT.len() as u32 }]);
    assert_eq!(pii.executed.len(), 1);
    assert_eq!(pii.executed[0].id, "test:pii");
    assert!(pii.provenance_digest.as_ref().is_some_and(|d| d.len() == 32));
    assert_eq!(r.result, t.result, "the receipt names the result it was issued with");
    assert_eq!(r.input, t.input);
    assert_eq!(r.receipt_schema_version, receipt::RECEIPT_SCHEMA_VERSION);
}

#[test]
fn a_declined_layer_is_not_assessed_and_no_layers_is_not_assessed() {
    let bare = typed(&Document::new(TEXT), "");
    assert_eq!(status(receipt_of(&bare)), ProcessingStatus::NotAssessed);

    let mut doc = Document::new(TEXT);
    let reason = format!("declined: saw {PLANTED}");
    doc.skip(Key::<Pii>::DEFAULT.id(), reason, Provenance::by("test:pii")).unwrap();
    let t = typed(&doc, "");
    let r = receipt_of(&t);
    assert_eq!(status(r), ProcessingStatus::NotAssessed);
    let pii = layer(r, "turbo:pii");
    assert_eq!(pii.status, ProcessingStatus::NotAssessed as i32);
    assert!(pii.assessed.is_empty() && pii.executed.is_empty());
    assert_eq!(pii.requested.len(), 1);
    assert_eq!(pii.error.as_ref().unwrap().kind, LayerErrorKind::Declined as i32);
    let wire = t.layers.iter().find(|l| l.id == "turbo:pii").unwrap();
    assert_eq!(wire.status, LayerWireStatus::Skipped as i32);
}

#[test]
fn a_skip_after_an_incomplete_read_is_upstream_and_the_document_partial() {
    let mut doc = with_sentences(TEXT);
    doc.skip(Key::<Pii>::DEFAULT.id(), "no tokens".into(), Provenance::by("test:pii")).unwrap();
    let on_pii = Provenance { reads: vec![Key::<Pii>::DEFAULT.id()], ..Provenance::by("test:tokens") };
    doc.skip(Key::<Tokens>::DEFAULT.id(), "needs pii".into(), on_pii).unwrap();
    let t = typed(&doc, "");
    let r = receipt_of(&t);
    assert_eq!(status(r), ProcessingStatus::Partial, "sentences assessed, the rest not");
    let tokens = layer(r, "turbo:tokens");
    assert_eq!(tokens.error.as_ref().unwrap().kind, LayerErrorKind::Upstream as i32);
}

#[test]
fn a_truncated_layer_is_partial() {
    let mut doc = with_sentences(TEXT);
    doc.add(&Key::DEFAULT, Pii::new(ViewId::SOURCE), Provenance::by("test:pii")).unwrap();
    let cut = Coverage {
        layer: "turbo:pii".into(),
        instance: String::new(),
        assessed: vec![ByteRange { start: 0, end: 10 }],
        truncations: vec![TruncationReason::MaxSequence],
    };
    let r = receipt::build(&Operation::start("k"), &doc, input_identity(&doc, None), None, &[cut]);
    assert_eq!(status(&r), ProcessingStatus::Partial);
    let pii = layer(&r, "turbo:pii");
    assert_eq!(pii.status, ProcessingStatus::Partial as i32);
    assert_eq!(pii.assessed, vec![ByteRange { start: 0, end: 10 }]);
    assert_eq!(pii.truncations, vec![TruncationReason::MaxSequence as i32]);
    // A coverage that reaches the end without a truncation is complete.
    let whole = Coverage {
        layer: "turbo:pii".into(),
        instance: String::new(),
        assessed: vec![ByteRange { start: 0, end: 20 }, ByteRange { start: 20, end: TEXT.len() as u32 }],
        truncations: Vec::new(),
    };
    let r = receipt::build(&Operation::start("k"), &doc, input_identity(&doc, None), None, &[whole]);
    assert_eq!(status(&r), ProcessingStatus::Complete);
}

#[test]
fn a_model_error_is_failed_with_a_sanitized_error() {
    let mut doc = with_sentences(TEXT);
    let err = Error::Annotator { annotator: "test:ner", detail: format!("model refused {PLANTED}") };
    doc.fail(Key::<Pii>::DEFAULT.id(), err, Provenance::by("test:ner")).unwrap();
    let t = typed(&doc, "");
    let r = receipt_of(&t);
    assert_eq!(status(r), ProcessingStatus::Failed);
    let e = layer(r, "turbo:pii").error.as_ref().unwrap();
    assert_eq!(e.kind, LayerErrorKind::Annotator as i32);
    assert_eq!(e.annotator, "test:ner");
    assert_eq!(e.turbo_code, turbo_kserve::api::TURBO_E_INTERNAL);
    assert!(layer(r, "turbo:pii").executed.is_empty());
}

#[test]
fn a_cancelled_operation_is_cancelled_and_names_the_steps_that_did_not_run() {
    let doc = with_sentences(TEXT);
    let mut op = Operation::start("k");
    op.requested_steps = vec!["sentences".into(), "pii".into()];
    op.executed_steps = vec!["sentences".into()];
    op.cancelled = true;
    let r = receipt::build(&op, &doc, input_identity(&doc, None), None, &[]);
    assert_eq!(status(&r), ProcessingStatus::Cancelled);
    assert!(r.result.is_none(), "a cancelled attempt hands over no result");
    // Not cancelled, the step that did not run makes the document partial.
    op.cancelled = false;
    let r = receipt::build(&op, &doc, input_identity(&doc, None), None, &[]);
    assert_eq!(status(&r), ProcessingStatus::Partial);
}

#[test]
fn a_retry_keeps_the_key_and_gets_a_new_attempt_id() {
    let doc = with_sentences(TEXT);
    let graph = export(&doc, codecs()).unwrap().to_bytes().unwrap();
    let req = InspectDocumentGraphRequest {
        doc_id: "doc".into(),
        graph,
        view: 0,
        offsets: 0,
        binding: None,
        operation_key: "ingest/42".into(),
        pii_output: None,
    };
    let (a, b) = (inspect_document_graph(&req).unwrap(), inspect_document_graph(&req).unwrap());
    let (ra, rb) = (receipt_of(&a), receipt_of(&b));
    assert_eq!((ra.operation_key.as_str(), rb.operation_key.as_str()), ("ingest/42", "ingest/42"));
    assert_ne!(ra.attempt_id, rb.attempt_id);
    assert_eq!(ra.attempt_id.len(), 32);
    assert!(ra.attempt_id.bytes().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    assert_eq!(a.result, b.result, "a new receipt never changes the result identity");
    assert_eq!(ra.result, rb.result);

    for bad in ["has space", "\u{e9}t\u{e9}", &"k".repeat(receipt::MAX_OPERATION_KEY_BYTES + 1)] {
        let err = inspect_document_graph(&InspectDocumentGraphRequest { operation_key: bad.into(), ..req.clone() })
            .unwrap_err();
        assert!(!err.message.contains(bad), "the refusal does not quote the key");
    }
}

#[test]
fn no_receipt_or_layer_error_carries_a_planted_value() {
    let mut doc = with_sentences(TEXT);
    let mut pii = Pii::new(ViewId::SOURCE);
    let (start, end) = (5, 5 + PLANTED.len() as u32);
    let detector = pii.detector("test", "1").unwrap();
    pii.push(PiiMention {
        span: Span::new(start, end),
        kind: PiiKind::Card,
        detector,
        evidence: &[PiiEvidence::Checksum { algorithm: ChecksumAlgorithm::Luhn, passed: true }],
        score: None,
        normalized: PLANTED,
    });
    doc.add(&Key::DEFAULT, pii, Provenance::by("test:pii")).unwrap();
    doc.skip(Key::<Tokens>::DEFAULT.id(), format!("skipped near {PLANTED}"), Provenance::by("test:tok")).unwrap();
    let bounds = Error::Bounds {
        layer: Key::<Sentences>::named("b").id(),
        view: ViewId::SOURCE,
        detail: format!("span over {PLANTED}"),
    };
    doc.fail(Key::<Sentences>::named("b").id(), bounds, Provenance::by("test:b")).unwrap();
    let t = typed(&doc, "op");
    let r = receipt_of(&t);
    assert_eq!(layer(r, "turbo:pii").item_count, 1);
    let bytes = r.encode_to_vec();
    assert!(!contains(&bytes, PLANTED.as_bytes()), "the receipt carries the planted value");
    assert!(!contains(&bytes, b"Card"), "the receipt carries source text");
    for l in &t.layers {
        let e = l.error.as_ref().map(|e| e.encode_to_vec()).unwrap_or_default();
        assert!(!contains(&e, PLANTED.as_bytes()), "{} error carries the planted value", l.id);
    }
    for v in &t.views {
        assert!(v.error.is_none(), "the source view is OK");
    }
    assert!(format!("{r:?}").find(PLANTED).is_none());
}

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    hay.windows(needle.len()).any(|w| w == needle)
}
