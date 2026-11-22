//! The typed projection (`InspectDocumentGraph`), the search projection
//! (`ProjectSearchDocument`) and the processing receipt name a withheld
//! source by one rule (#2403, #2408): a digest leaves only with the text
//! it is a digest of. The two projections answer alike but for the one
//! documented difference, the source lengths, and every reader refuses
//! the same half identities. A receipt that withholds its digests and
//! still names the graph by its id, which is the digest in hexadecimal,
//! or by its length, is not whole.

use prost::Message;
use turbo_analysis::{Analyzer, Document, ViewId, ViewTextDigest};
use turbo_kserve::offsets::OffsetMode;
use turbo_kserve::proto::result::analyze_progressive_event::Event;
use turbo_kserve::proto::result::{
    AnalyzeProgressiveEvent, InputBinding, InputIdentity, ProcessingReceipt, ResultIdentity, SearchDocumentProjection,
    SearchReplayProvenance, TypedDocumentResult, ViewChain, mapping_payload,
};
use turbo_kserve::results::disclosure::Disclosure;
use turbo_kserve::results::document::{codecs, export_document_response};
use turbo_kserve::results::receipt::check::check_receipt_shape;
use turbo_kserve::results::receipt::stream::{Incomplete, terminal_receipt};
use turbo_kserve::results::receipt::{self, Operation};
use turbo_kserve::results::search_projection::{
    document_to_search_projection_bound, round_trip_search_projection, validate_search_projection,
};
use turbo_kserve::results::typed::identity::{input_identity_with, result_identity, result_identity_with};
use turbo_kserve::results::typed::{document_to_typed_bound, round_trip_typed_document, validate_typed_document};
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

fn graph_bytes(doc: &Document) -> Vec<u8> {
    turbo_analysis::export::export(doc, codecs()).unwrap().to_bytes().unwrap()
}

fn binding() -> InputBinding {
    InputBinding {
        source_revision: "rev-7".into(),
        parse_artifact_id: "parse-3".into(),
        parse_artifact_digest: vec![9; 8],
    }
}

fn typed(doc: &Document, view: ViewId, disclosure: &Disclosure) -> TypedDocumentResult {
    let t = document_to_typed_bound("d", doc, view, OffsetMode::Utf8, Some(&binding()), "", disclosure).unwrap();
    validate_typed_document(&t).unwrap();
    t
}

fn search(doc: &Document, view: ViewId, disclosure: &Disclosure) -> SearchDocumentProjection {
    let binding = binding();
    let p =
        document_to_search_projection_bound("d", doc, view, OffsetMode::Utf8, Some(&binding), &[], disclosure).unwrap();
    validate_search_projection(&p).unwrap();
    p
}

fn replay(p: &SearchDocumentProjection) -> &SearchReplayProvenance {
    p.provenance.as_ref().unwrap()
}

fn mapping(t: &TypedDocumentResult) -> &[ViewChain] {
    let Some(mapping_payload::Payload::Inline(mapping)) = &t.mapping.as_ref().unwrap().payload else {
        panic!("a mapping")
    };
    &mapping.views
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The one deliberate difference, and nothing else: the typed answer
/// keeps the source lengths its finding spans are counted in, and the
/// search answer, which carries no source span, zeroes them.
#[test]
fn both_projections_name_a_withheld_source_alike_but_for_its_lengths() {
    let doc = masked(EMAIL);
    let (t, s) = (typed(&doc, MASKED, &REWRITTEN), search(&doc, MASKED, &REWRITTEN));
    let (ti, si) = (t.input.as_ref().unwrap(), replay(&s).input.as_ref().unwrap());
    assert!(ti.raw_bytes_digest.is_empty() && ti.source_text_digest.is_empty(), "{ti:?}");
    assert_eq!((ti.raw_bytes_len, ti.source_text_len), (EMAIL.len() as u64, EMAIL.len() as u32));
    assert_eq!((si.raw_bytes_len, si.source_text_len), (0, 0));
    assert_eq!(&InputIdentity { raw_bytes_len: 0, source_text_len: 0, ..ti.clone() }, si);
    assert_eq!(si.binding, Some(binding()), "the caller's binding is copied either way");
    assert_eq!(t.result, replay(&s).result);
    assert_eq!(
        t.result.as_ref().unwrap(),
        &ResultIdentity {
            result_id: String::new(),
            content_digest: Vec::new(),
            graph_bytes_len: 0,
            ..result_identity(&graph_bytes(&doc))
        }
    );

    // Each view: one digest decision on both, and the length by contract.
    assert_eq!(mapping(&t).len(), replay(&s).views.len());
    for v in mapping(&t) {
        let other = replay(&s).views.iter().find(|o| o.view_id == v.view).unwrap();
        let text = doc.text(ViewId(v.view as u16));
        assert_eq!(v.text_digest, other.text_sha256, "view {}", v.view);
        assert_eq!(v.text_len as usize, text.len(), "view {}: the typed mapping keeps every length", v.view);
        if ViewId(v.view as u16) == MASKED {
            assert_eq!(v.text_digest, ViewTextDigest::of_text(text).as_bytes(), "the rewrite view is shown");
            assert_eq!(other.utf8_bytes, v.text_len);
        } else {
            assert!(v.text_digest.is_empty(), "view {}: the source's digest left", v.view);
            assert_eq!(other.utf8_bytes, 0, "view {}: the search answer gives a withheld length", v.view);
        }
    }
}

/// With the source shown there is no difference at all.
#[test]
fn both_projections_name_a_shown_source_alike() {
    let pii = masked(EMAIL);
    let plain = Document::new("nothing to find here".to_owned());
    for (doc, disclosure) in [(&pii, Disclosure::FULL), (&plain, REWRITTEN), (&plain, Disclosure::FINDINGS)] {
        let (t, s) = (typed(doc, ViewId::SOURCE, &disclosure), search(doc, ViewId::SOURCE, &disclosure));
        assert_eq!(t.input, replay(&s).input);
        assert_eq!(t.result, replay(&s).result);
        assert_eq!(t.result.as_ref().unwrap(), &result_identity(&graph_bytes(doc)));
        let text = doc.text(ViewId::SOURCE);
        let input = t.input.as_ref().unwrap();
        assert_eq!(input.source_text_digest, ViewTextDigest::of_text(text).as_bytes());
        assert_eq!((input.raw_bytes_len, input.source_text_len), (text.len() as u64, text.len() as u32));
        for v in mapping(&t) {
            let other = replay(&s).views.iter().find(|o| o.view_id == v.view).unwrap();
            assert_eq!((v.text_len, &v.text_digest), (other.utf8_bytes, &other.text_sha256), "view {}", v.view);
            assert_eq!(v.text_digest.len(), 32);
        }
    }
}

/// A receipt of `doc` that withholds its digests, with its result.
fn withheld_receipt(doc: &Document) -> ProcessingReceipt {
    let (input, result) = (input_identity_with(doc, None, false), result_identity_with(&graph_bytes(doc), false));
    receipt::build(&Operation::start("k"), doc, input, Some(result), &[])
}

/// The stream of a withheld answer about `doc` that ends with `r`.
fn stream(doc: &Document, r: &ProcessingReceipt) -> Vec<AnalyzeProgressiveEvent> {
    let complete = export_document_response(doc, &Disclosure::FINDINGS).unwrap();
    assert!(complete.graph.is_empty(), "the fixture's graph is withheld");
    vec![
        AnalyzeProgressiveEvent { sequence: 1, event: Some(Event::Complete(complete)) },
        AnalyzeProgressiveEvent { sequence: 2, event: Some(Event::Receipt(r.clone())) },
    ]
}

/// The id of a result is its graph digest in hexadecimal, so a receipt
/// that leaves the digest out and keeps the id has left nothing out. The
/// receipt's own check refuses it, and so a stream that ends with it is
/// not complete.
#[test]
fn a_receipt_that_withholds_its_digests_names_no_graph_by_id_or_length() {
    let doc = masked(EMAIL);
    let whole = result_identity(&graph_bytes(&doc));
    let r = withheld_receipt(&doc);
    // Positive control: the withheld receipt is whole and terminal.
    check_receipt_shape(&r).unwrap();
    terminal_receipt(&stream(&doc, &r), "k", EMAIL).unwrap();
    assert!(!r.encode_to_vec().windows(64).any(|w| w == whole.result_id.as_bytes()));

    let mut by_id = r.clone();
    by_id.result.as_mut().unwrap().result_id = whole.result_id.clone();
    assert_eq!(by_id.result.as_ref().unwrap().result_id, hex(&whole.content_digest), "the id is the digest");
    let mut by_length = r.clone();
    by_length.result.as_mut().unwrap().graph_bytes_len = whole.graph_bytes_len;
    for (what, named) in [("id", by_id), ("length", by_length)] {
        let why = check_receipt_shape(&named).map(|_| ()).expect_err(&format!("a graph named by {what} is whole"));
        assert!(why.message.contains("no content_digest"), "{what}: {}", why.message);
        assert_eq!(terminal_receipt(&stream(&doc, &named), "k", EMAIL).unwrap_err(), Incomplete::Malformed, "{what}");
    }
}

/// One rule, three readers: each half of an identity that the typed
/// validator refuses, the search validator and the receipt check refuse
/// too.
#[test]
fn every_reader_refuses_the_same_half_identities() {
    let doc = masked(EMAIL);
    let whole = result_identity(&graph_bytes(&doc));
    let text = ViewTextDigest::of_text(EMAIL).as_bytes().to_vec();
    type Half<'a> = (&'a str, Box<dyn Fn(&mut InputIdentity, &mut ResultIdentity) + 'a>);
    let halves: Vec<Half> = vec![
        ("source_text_digest", Box::new(|i, _| i.source_text_digest = text.clone())),
        ("raw_bytes_digest", Box::new(|i, _| i.raw_bytes_digest = text.clone())),
        ("result_id", Box::new(|_, r| r.result_id = whole.result_id.clone())),
        ("graph_bytes_len", Box::new(|_, r| r.graph_bytes_len = whole.graph_bytes_len)),
        ("content_digest", Box::new(|_, r| r.content_digest = whole.content_digest.clone())),
    ];
    let (t, s, r) = (typed(&doc, MASKED, &REWRITTEN), search(&doc, MASKED, &REWRITTEN), withheld_receipt(&doc));
    round_trip_typed_document(&t).unwrap();
    round_trip_search_projection(&s).unwrap();
    for (what, half) in &halves {
        let mut m = t.clone();
        half(m.input.as_mut().unwrap(), m.result.as_mut().unwrap());
        // The envelope's receipt names the same identities.
        let (input, result) = (m.input.clone(), m.result.clone());
        (m.receipt.as_mut().unwrap().input, m.receipt.as_mut().unwrap().result) = (input, result);
        assert!(round_trip_typed_document(&m).is_err(), "the typed validator takes a lone {what}");

        let mut m = s.clone();
        let prov = m.provenance.as_mut().unwrap();
        half(prov.input.as_mut().unwrap(), prov.result.as_mut().unwrap());
        assert!(round_trip_search_projection(&m).is_err(), "the search validator takes a lone {what}");

        let mut m = r.clone();
        half(m.input.as_mut().unwrap(), m.result.as_mut().unwrap());
        assert!(check_receipt_shape(&m).is_err(), "the receipt check takes a lone {what}");
    }
}
