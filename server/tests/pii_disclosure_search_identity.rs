//! `ProjectSearchDocument` on a PII rewrite view under `rewritten_text`
//! alone carries no unkeyed digest of what it withholds (#2403). The
//! SHA-256 of the source text, of the raw bytes or of the whole graph
//! lets a reader recover a short value by hashing guesses, so each is
//! left out with the text it is the digest of. The masked view's own
//! digest, and every digest of an answer that may carry the source,
//! are unchanged.

use prost::Message;
use turbo_analysis::{Analyzer, Document, ViewId, ViewTextDigest};
use turbo_kserve::offsets::OffsetMode;
use turbo_kserve::proto::result::{ProjectSearchDocumentRequest, SearchDocumentProjection};
use turbo_kserve::results::disclosure::{Disclosure, PiiOutputPolicy};
use turbo_kserve::results::document::codecs;
use turbo_kserve::results::search_projection::{
    document_to_search_projection_bound, project_search_document_under, round_trip_search_projection,
    validate_search_projection,
};
use turbo_pii::PiiDetector;
use turbo_pii::rewrite::MaskedView;

const EMAIL: &str = "alice.planted@example.com";
/// Another address of the same length and shape.
const OTHER: &str = "carol.guessed@example.org";
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

fn graph_bytes(doc: &Document) -> Vec<u8> {
    turbo_analysis::export::export(doc, codecs()).unwrap().to_bytes().unwrap()
}

fn projected(doc: &Document, disclosure: &Disclosure) -> SearchDocumentProjection {
    let p = document_to_search_projection_bound("d", doc, MASKED, OffsetMode::Utf8, None, &[], disclosure).unwrap();
    validate_search_projection(&p).unwrap();
    p
}

/// The same projection through the request handler, with the server's
/// policy capped at `rewritten_text`.
fn handled(doc: &Document) -> SearchDocumentProjection {
    let request = ProjectSearchDocumentRequest {
        doc_id: "d".into(),
        graph: graph_bytes(doc),
        view: u32::from(MASKED.0),
        pii_output: Some(REWRITTEN.to_options()),
        ..Default::default()
    };
    let policy = PiiOutputPolicy { full_text: false, detail: false, ..PiiOutputPolicy::ALL };
    project_search_document_under(&request, policy).unwrap()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn holds(m: &SearchDocumentProjection, needle: &[u8]) -> bool {
    m.encode_to_vec().windows(needle.len()).any(|w| w == needle)
}

/// Every unkeyed digest that names the source of `doc`: of its text, and
/// of the graph bytes that hold that text.
fn source_digests(doc: &Document) -> Vec<(&'static str, Vec<u8>)> {
    let text = ViewTextDigest::of_text(doc.text(ViewId::SOURCE)).as_bytes().to_vec();
    let graph = turbo_analysis::export::graph_content_digest(&graph_bytes(doc)).as_bytes().to_vec();
    vec![("source text", text), ("graph bytes", graph)]
}

#[test]
fn a_masked_answer_carries_no_digest_of_the_source() {
    let doc = masked(EMAIL);
    assert!(!doc.text(MASKED).contains("alice"), "the fixture masks the address");
    for (path, p) in [("function", projected(&doc, &REWRITTEN)), ("handler", handled(&doc))] {
        for (what, digest) in source_digests(&doc) {
            assert!(!holds(&p, &digest), "{path}: the answer carries the SHA-256 of the {what}");
            assert!(!holds(&p, hex(&digest).as_bytes()), "{path}: the answer carries the hex SHA-256 of the {what}");
        }
        let prov = p.provenance.as_ref().unwrap();
        let input = prov.input.as_ref().unwrap();
        assert!(input.raw_bytes_digest.is_empty() && input.source_text_digest.is_empty(), "{path}: {input:?}");
        let result = prov.result.as_ref().unwrap();
        assert!(result.content_digest.is_empty() && result.result_id.is_empty(), "{path}: {result:?}");
        let source = prov.views.iter().find(|v| v.view_id == 0).unwrap();
        assert!(source.text_sha256.is_empty(), "{path}: the source view's digest left");
    }
}

/// The property behind the digests: two documents that differ only in the
/// masked value answer with the same bytes, so nothing in the answer can
/// be compared against a guess.
#[test]
fn two_values_of_one_shape_give_the_same_masked_answer() {
    let (a, b) = (masked(EMAIL), masked(OTHER));
    assert_eq!(a.text(MASKED), b.text(MASKED), "the fixture's two addresses mask alike");
    assert_eq!(projected(&a, &REWRITTEN).encode_to_vec(), projected(&b, &REWRITTEN).encode_to_vec());
    assert_eq!(handled(&a).encode_to_vec(), handled(&b).encode_to_vec());
}

#[test]
fn the_masked_view_keeps_its_own_digest() {
    let doc = masked(EMAIL);
    for p in [projected(&doc, &REWRITTEN), handled(&doc)] {
        let prov = p.provenance.as_ref().unwrap();
        let view = prov.views.iter().find(|v| v.view_id == u32::from(MASKED.0)).unwrap();
        assert_eq!(view.text_sha256, ViewTextDigest::of_text(doc.text(MASKED)).as_bytes());
        assert_eq!(view.utf8_bytes as usize, doc.text(MASKED).len());
    }
}

/// An answer that may carry the source names it by digest as before: with
/// `full_text`, and for a document with no findings under any disclosure.
#[test]
fn an_answer_that_may_carry_the_source_keeps_every_digest() {
    let pii = masked(EMAIL);
    let plain = Document::new("nothing to find here".to_owned());
    for (doc, disclosure) in [(&pii, Disclosure::FULL), (&plain, REWRITTEN), (&plain, Disclosure::FINDINGS)] {
        let p = document_to_search_projection_bound("d", doc, ViewId::SOURCE, OffsetMode::Utf8, None, &[], &disclosure)
            .unwrap();
        validate_search_projection(&p).unwrap();
        let prov = p.provenance.as_ref().unwrap();
        let text = ViewTextDigest::of_text(doc.text(ViewId::SOURCE));
        let input = prov.input.as_ref().unwrap();
        assert_eq!(input.source_text_digest, text.as_bytes());
        assert_eq!(input.raw_bytes_digest, text.as_bytes());
        let graph = graph_bytes(doc);
        let result = prov.result.as_ref().unwrap();
        assert_eq!(result.content_digest, turbo_analysis::export::graph_content_digest(&graph).as_bytes());
        assert_eq!(result.result_id, hex(&result.content_digest));
        assert_eq!(result.graph_bytes_len, graph.len() as u64);
        assert!(prov.views.iter().all(|v| v.text_sha256.len() == 32), "{:?}", prov.views);
    }
}

/// A consumer can still read a masked answer back: the validator accepts
/// the digests it leaves out, and still refuses one of a wrong length.
#[test]
fn a_masked_answer_round_trips_and_a_short_digest_is_still_refused() {
    let p = projected(&masked(EMAIL), &REWRITTEN);
    assert_eq!(round_trip_search_projection(&p).unwrap(), p);
    let mut short = p.clone();
    short.provenance.as_mut().unwrap().input.as_mut().unwrap().source_text_digest = vec![7; 31];
    assert!(round_trip_search_projection(&short).unwrap_err().message.contains("source_text_digest"));
    let mut short = p;
    short.provenance.as_mut().unwrap().views[0].text_sha256 = vec![7; 5];
    assert!(round_trip_search_projection(&short).unwrap_err().message.contains("text_sha256"));
}
