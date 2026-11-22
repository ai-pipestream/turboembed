//! Every reader of a result identity holds a shown one to the same check
//! (#2494): a 32 byte `content_digest`, a `result_id` that is its
//! lowercase hexadecimal, and the served graph format and envelope
//! schema versions. The typed envelope, the search projection, the
//! receipt's own check and the terminal receipt of a stream each refuse
//! a result that breaks it. The search projection also holds its listed
//! views to its input identity: the source view carries the input's own
//! digest when the source is shown and none when it is withheld.

use turbo_analysis::{Analyzer, Document, ViewId};
use turbo_kserve::offsets::OffsetMode;
use turbo_kserve::proto::result::analyze_progressive_event::Event;
use turbo_kserve::proto::result::{
    AnalyzeProgressiveEvent, ProcessingReceipt, ResultIdentity, SearchDocumentProjection, TypedDocumentResult,
};
use turbo_kserve::results::disclosure::Disclosure;
use turbo_kserve::results::document::export_document_response;
use turbo_kserve::results::receipt::check::check_receipt_shape;
use turbo_kserve::results::receipt::stream::{Incomplete, terminal_receipt};
use turbo_kserve::results::receipt::{self, Operation};
use turbo_kserve::results::search_projection::{document_to_search_projection_bound, round_trip_search_projection};
use turbo_kserve::results::typed::identity::{input_identity_with, result_identity_with};
use turbo_kserve::results::typed::{document_to_typed_bound, round_trip_typed_document};
use turbo_pii::PiiDetector;
use turbo_pii::rewrite::MaskedView;

const EMAIL: &str = "alice.planted@example.com";
const MASKED: ViewId = ViewId(1);
const REWRITTEN: Disclosure = Disclosure { rewritten_text: true, full_text: false, detail: false };

/// A document that is only `text`, with its findings and its masked view.
fn masked(text: &str) -> Document {
    let analyzer =
        Analyzer::builder().add(PiiDetector::cursor()).step(MaskedView::by_kind('*').unwrap().bound()).build().unwrap();
    let mut docs = vec![Document::new(text.to_owned())];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    docs.pop().unwrap()
}

fn typed(doc: &Document, view: ViewId, disclosure: &Disclosure) -> TypedDocumentResult {
    let t = document_to_typed_bound("d", doc, view, OffsetMode::Utf8, None, "", disclosure).unwrap();
    round_trip_typed_document(&t).unwrap();
    t
}

fn search(doc: &Document, view: ViewId, disclosure: &Disclosure) -> SearchDocumentProjection {
    let s = document_to_search_projection_bound("d", doc, view, OffsetMode::Utf8, None, &[], disclosure).unwrap();
    round_trip_search_projection(&s).unwrap();
    s
}

/// The progressive stream of an answer about `doc` under `disclosure`,
/// ending with the receipt that answer makes, and that receipt.
fn stream(doc: &Document, disclosure: &Disclosure) -> (Vec<AnalyzeProgressiveEvent>, ProcessingReceipt) {
    let complete = export_document_response(doc, disclosure).unwrap();
    let shown = !complete.graph.is_empty();
    let (input, result) = (input_identity_with(doc, None, shown), result_identity_with(&complete.graph, shown));
    let r = receipt::build(&Operation::start("k"), doc, input, Some(result), &[]);
    check_receipt_shape(&r).unwrap();
    let events = vec![
        AnalyzeProgressiveEvent { sequence: 1, event: Some(Event::Complete(complete)) },
        AnalyzeProgressiveEvent { sequence: 2, event: Some(Event::Receipt(r.clone())) },
    ];
    terminal_receipt(&events, "k", doc.text(ViewId::SOURCE)).unwrap();
    (events, r)
}

fn with_receipt(events: &[AnalyzeProgressiveEvent], r: ProcessingReceipt) -> Vec<AnalyzeProgressiveEvent> {
    let mut events = events.to_vec();
    events.last_mut().unwrap().event = Some(Event::Receipt(r));
    events
}

type Change = (&'static str, fn(&mut ResultIdentity));

/// What each reader refuses on a shown result. Only the id, or only a
/// version, is changed: the digest and length still name the graph.
const SHOWN: [Change; 7] = [
    ("another id", |r| r.result_id = "0".repeat(64)),
    ("an id one digit short", |r| {
        r.result_id.pop();
    }),
    ("an id one digit long", |r| r.result_id.push('0')),
    ("an upper-case id", |r| r.result_id = r.result_id.to_uppercase()),
    ("no id", |r| r.result_id.clear()),
    ("graph format version 0", |r| r.graph_format_version = 0),
    ("envelope schema version 0", |r| r.envelope_schema_version = 0),
];

/// What each reader refuses on a withheld result: the versions are the
/// one thing it still carries, and they are the served ones.
const WITHHELD: [Change; 2] = [
    ("graph format version 0", |r| r.graph_format_version = 0),
    ("envelope schema version 9", |r| r.envelope_schema_version = 9),
];

/// Applies `change` to every reader's copy of the result identity of
/// `doc` under `disclosure` and asserts each one refuses it.
fn every_reader_refuses(doc: &Document, disclosure: &Disclosure, view: ViewId, changes: &[Change]) {
    let (t, s) = (typed(doc, view, disclosure), search(doc, view, disclosure));
    let (events, r) = stream(doc, disclosure);
    let text = doc.text(ViewId::SOURCE);
    for (what, change) in changes {
        let mut m = t.clone();
        change(m.result.as_mut().unwrap());
        // The envelope's receipt names the same result, so only the
        // identity check can refuse it.
        m.receipt.as_mut().unwrap().result = m.result.clone();
        let why = round_trip_typed_document(&m).expect_err(&format!("typed: {what}")).message;
        assert!(why.starts_with("result"), "typed, {what}: {why}");

        let mut m = s.clone();
        change(m.provenance.as_mut().unwrap().result.as_mut().unwrap());
        let why = round_trip_search_projection(&m).expect_err(&format!("search: {what}")).message;
        assert!(why.starts_with("provenance.result"), "search, {what}: {why}");

        let mut m = r.clone();
        change(m.result.as_mut().unwrap());
        let why = check_receipt_shape(&m).map(|_| ()).expect_err(&format!("receipt: {what}")).message;
        assert!(why.starts_with("receipt.result"), "receipt, {what}: {why}");
        assert_eq!(
            terminal_receipt(&with_receipt(&events, m), "k", text).unwrap_err(),
            Incomplete::Malformed,
            "{what}"
        );
    }
}

#[test]
fn every_reader_refuses_a_shown_result_whose_id_or_versions_are_not_the_served_ones() {
    let pii = masked(EMAIL);
    let plain = Document::new("nothing to find here".to_owned());
    every_reader_refuses(&pii, &Disclosure::FULL, ViewId::SOURCE, &SHOWN);
    every_reader_refuses(&plain, &Disclosure::FINDINGS, ViewId::SOURCE, &SHOWN);
}

#[test]
fn every_reader_refuses_a_withheld_result_whose_versions_are_not_the_served_ones() {
    every_reader_refuses(&masked(EMAIL), &REWRITTEN, MASKED, &WITHHELD);
}

/// A search projection whose input withholds its digests and whose source
/// view still lists the digest of the source text gives back what the
/// input left out: refused, like the typed envelope's mapping.
#[test]
fn a_withheld_search_projection_lists_no_source_view_digest() {
    let doc = masked(EMAIL);
    let full = search(&doc, MASKED, &Disclosure::FULL);
    let source_digest = full.provenance.as_ref().unwrap().input.as_ref().unwrap().source_text_digest.clone();
    let withheld = search(&doc, MASKED, &REWRITTEN);
    let source = |m: &mut SearchDocumentProjection| {
        let views = &mut m.provenance.as_mut().unwrap().views;
        views.iter_mut().find(|v| v.view_id == 0).unwrap().text_sha256 = source_digest.clone();
    };
    let mut m = withheld.clone();
    source(&mut m);
    let why = round_trip_search_projection(&m).unwrap_err().message;
    assert!(why.contains("source view") && why.contains("withheld"), "{why}");
    // With its length back too, it is the shown view whole.
    let mut m = withheld;
    source(&mut m);
    m.provenance.as_mut().unwrap().views.iter_mut().find(|v| v.view_id == 0).unwrap().utf8_bytes = EMAIL.len() as u32;
    assert!(round_trip_search_projection(&m).unwrap_err().message.contains("source view"));
}

/// With the source shown, every view is listed with its digest and the
/// source view's is the input's own.
#[test]
fn a_shown_search_projection_lists_each_view_digest_and_the_source_view_is_the_inputs() {
    let doc = masked(EMAIL);
    let s = search(&doc, MASKED, &Disclosure::FULL);
    let refused = |change: &dyn Fn(&mut SearchDocumentProjection)| {
        let mut m = s.clone();
        change(&mut m);
        round_trip_search_projection(&m).unwrap_err().message
    };
    let views = |m: &mut SearchDocumentProjection| m.provenance.as_mut().unwrap().views.clone();
    let masked_ix = views(&mut s.clone()).iter().position(|v| v.view_id == u32::from(MASKED.0)).unwrap();
    let source_ix = views(&mut s.clone()).iter().position(|v| v.view_id == 0).unwrap();
    let masked_digest = views(&mut s.clone())[masked_ix].text_sha256.clone();

    let why = refused(&|m| m.provenance.as_mut().unwrap().views[source_ix].text_sha256 = masked_digest.clone());
    assert!(why.contains("source view") && why.contains("source_text_digest"), "{why}");
    let why = refused(&|m| {
        let v = &mut m.provenance.as_mut().unwrap().views[masked_ix];
        (v.text_sha256, v.utf8_bytes) = (Vec::new(), 0);
    });
    assert!(why.contains("no text_sha256") && why.contains("shown"), "{why}");
    let why = refused(&|m| {
        m.provenance.as_mut().unwrap().views.remove(source_ix);
    });
    assert!(why.contains("no source view"), "{why}");
}
