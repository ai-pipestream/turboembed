//! `InspectDocumentGraph`, and every answer made by the same projection,
//! carries no unkeyed digest of what it withholds (#2403). The SHA-256 of
//! a short source (an address, a phone number, an id) gives the value back
//! by hashing guesses, and so does every digest that chains it: the raw
//! bytes, the graph bytes, a view's text, and each layer's provenance,
//! content and input snapshot digests. A masked answer leaves them out
//! with the text; an answer that may carry the source keeps them all.

use prost::Message;
use turbo_analysis::{Analyzer, Document, ViewId, ViewTextDigest};
use turbo_kserve::offsets::OffsetMode;
use turbo_kserve::proto::result::{InspectDocumentGraphRequest, TypedDocumentResult, mapping_payload};
use turbo_kserve::results::disclosure::{Disclosure, PiiOutputPolicy};
use turbo_kserve::results::document::{codecs, export_document_response};
use turbo_kserve::results::typed::scoped::{resolve, scoped_ref};
use turbo_kserve::results::typed::{
    document_to_typed_with, inspect_document_graph_under, round_trip_typed_document, validate_typed_document,
};
use turbo_pii::PiiDetector;
use turbo_pii::rewrite::MaskedView;

const EMAIL: &str = "alice.planted@example.com";
/// Another address of the same length and shape.
const OTHER: &str = "carol.guessed@example.org";
const MASKED: ViewId = ViewId(1);
const REWRITTEN: Disclosure = Disclosure { rewritten_text: true, full_text: false, detail: false };
/// Every disclosure that withholds the source, with the views it is asked on.
const WITHHELD: [(Disclosure, ViewId); 4] = [
    (Disclosure::FINDINGS, ViewId::SOURCE),
    (Disclosure::FINDINGS, MASKED),
    (REWRITTEN, ViewId::SOURCE),
    (REWRITTEN, MASKED),
];

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

fn projected(doc: &Document, view: ViewId, disclosure: &Disclosure) -> TypedDocumentResult {
    let typed = document_to_typed_with("d", doc, view, OffsetMode::Utf8, disclosure).unwrap();
    validate_typed_document(&typed).unwrap();
    typed
}

/// The same answer through the `InspectDocumentGraph` handler, with the
/// server's policy capped at what `disclosure` asks.
fn handled(doc: &Document, view: ViewId, disclosure: &Disclosure) -> TypedDocumentResult {
    let request = InspectDocumentGraphRequest {
        doc_id: "d".into(),
        graph: graph_bytes(doc),
        view: u32::from(view.0),
        offsets: 1,
        binding: None,
        pii_output: Some(disclosure.to_options()),
        operation_key: String::new(),
        pii_policy: None,
    };
    let policy = PiiOutputPolicy { rewritten_text: disclosure.rewritten_text, ..PiiOutputPolicy::NONE };
    inspect_document_graph_under(&request, policy).unwrap()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn holds(bytes: &[u8], needle: &[u8]) -> bool {
    bytes.windows(needle.len()).any(|w| w == needle)
}

/// What a reader of `answer` learns from a list of guesses: it hashes each
/// one and looks for the digest, raw or in hexadecimal.
fn confirmed<'a>(answer: &TypedDocumentResult, guesses: &[&'a str]) -> Vec<&'a str> {
    let bytes = answer.encode_to_vec();
    let found = |g: &&str| {
        let d = ViewTextDigest::of_text(g);
        holds(&bytes, d.as_bytes()) || holds(&bytes, d.to_hex().as_bytes())
    };
    guesses.iter().copied().filter(found).collect()
}

/// Every 32-byte digest the full answer about `doc` carries that is made
/// from the source text: the input and result identities, each view's text
/// digest but the masked view's own, and each entry's provenance, content
/// and input snapshot digests.
fn source_digests(doc: &Document) -> Vec<(String, Vec<u8>)> {
    let full = projected(doc, ViewId::SOURCE, &Disclosure::FULL);
    let (input, result) = (full.input.unwrap(), full.result.unwrap());
    let mut out = vec![
        ("input.raw_bytes_digest".to_owned(), input.raw_bytes_digest),
        ("input.source_text_digest".to_owned(), input.source_text_digest),
        ("result.content_digest".to_owned(), result.content_digest),
    ];
    let Some(mapping_payload::Payload::Inline(mapping)) = full.mapping.unwrap().payload else { panic!("a mapping") };
    out.push(("mapping source text_digest".to_owned(), mapping.views[0].text_digest.clone()));
    for p in full.provenance {
        let at = format!("provenance {:?} {:?}", p.layer, p.instance);
        out.push((format!("{at} provenance_digest"), p.provenance_digest));
        out.extend(p.content_digest.map(|d| (format!("{at} content_digest"), d)));
        out.extend(p.input_snapshot_digest.map(|d| (format!("{at} input_snapshot_digest"), d)));
    }
    assert!(out.iter().all(|(_, d)| d.len() == 32), "positive control: the full answer carries every digest");
    out
}

/// `typed` without what differs between two attempts at one answer.
fn settled(mut typed: TypedDocumentResult) -> Vec<u8> {
    typed.receipt.as_mut().unwrap().attempt_id.clear();
    typed.encode_to_vec()
}

#[test]
fn a_guess_is_not_confirmed_by_a_masked_answer() {
    let doc = masked(EMAIL);
    assert!(!doc.text(MASKED).contains("alice"), "the fixture masks the address");
    let guesses = ["bob.wrong@example.com", OTHER, EMAIL, "dave.other@example.net"];
    // Positive control: an answer that may carry the source names it.
    assert_eq!(confirmed(&projected(&doc, ViewId::SOURCE, &Disclosure::FULL), &guesses), [EMAIL]);
    for (disclosure, view) in WITHHELD {
        for (path, answer) in
            [("function", projected(&doc, view, &disclosure)), ("handler", handled(&doc, view, &disclosure))]
        {
            let at = format!("{path}, view {}, {disclosure:?}", view.0);
            assert_eq!(confirmed(&answer, &guesses), [""; 0], "{at}: the answer confirms a guess of the source");
            let bytes = answer.encode_to_vec();
            for (what, digest) in source_digests(&doc) {
                assert!(!holds(&bytes, &digest), "{at}: the answer carries {what}");
                assert!(!holds(&bytes, hex(&digest).as_bytes()), "{at}: the answer carries {what} in hexadecimal");
            }
        }
    }
}

/// The property behind the digests: two documents that differ only in the
/// masked value answer with the same bytes, so nothing in the answer can
/// be compared against a guess.
#[test]
fn two_values_of_one_shape_give_the_same_masked_answer() {
    let (a, b) = (masked(EMAIL), masked(OTHER));
    assert_eq!(a.text(MASKED), b.text(MASKED), "the fixture's two addresses mask alike");
    for (disclosure, view) in WITHHELD {
        let at = format!("view {}, {disclosure:?}", view.0);
        assert_eq!(settled(projected(&a, view, &disclosure)), settled(projected(&b, view, &disclosure)), "{at}");
        assert_eq!(settled(handled(&a, view, &disclosure)), settled(handled(&b, view, &disclosure)), "handler, {at}");
    }
    // The findings that stand in for a withheld graph (RoundTripDocumentGraph,
    // the progressive answer, a sink item) are the same projection.
    let findings = |doc: &Document| export_document_response(doc, &Disclosure::FINDINGS).unwrap().findings.unwrap();
    assert_eq!(settled(findings(&a)), settled(findings(&b)));
}

#[test]
fn a_masked_answer_names_its_input_and_result_by_length_and_version_only() {
    let doc = masked(EMAIL);
    for (disclosure, view) in WITHHELD {
        let typed = projected(&doc, view, &disclosure);
        let input = typed.input.as_ref().unwrap();
        assert!(input.raw_bytes_digest.is_empty() && input.source_text_digest.is_empty(), "{input:?}");
        assert_eq!((input.raw_bytes_len, input.source_text_len), (EMAIL.len() as u64, EMAIL.len() as u32));
        let result = typed.result.as_ref().unwrap();
        assert!(result.content_digest.is_empty() && result.result_id.is_empty(), "{result:?}");
        assert_eq!(result.graph_bytes_len, 0);
        assert!(result.graph_format_version != 0 && result.envelope_schema_version != 0);
        for p in &typed.provenance {
            assert!(p.provenance_digest.is_empty(), "{p:?}");
            assert!(p.content_digest.is_none() && p.input_snapshot_digest.is_none(), "{p:?}");
            assert!(!p.layer.is_empty(), "the entry still names its layer");
        }
        let receipt = typed.receipt.as_ref().unwrap();
        assert_eq!(receipt.input.as_ref(), typed.input.as_ref());
        assert_eq!(receipt.result.as_ref(), typed.result.as_ref());
        assert!(receipt.layers.iter().all(|l| l.provenance_digest.is_none()), "{:?}", receipt.layers);
        // The configuration digest is of no text and stays.
        assert_eq!(receipt.config_digest.len(), 32);
        let Some(mapping_payload::Payload::Inline(mapping)) = &typed.mapping.as_ref().unwrap().payload else {
            panic!("a mapping")
        };
        assert!(mapping.views[0].text_digest.is_empty(), "the source view's digest left");
        // The masked view's digest leaves with its text, and only then.
        let own = &mapping.views[usize::from(MASKED.0)];
        assert_eq!(own.text_len as usize, doc.text(MASKED).len());
        if disclosure.rewritten_text {
            assert_eq!(own.text_digest, ViewTextDigest::of_text(doc.text(MASKED)).as_bytes());
        } else {
            assert!(own.text_digest.is_empty(), "a withheld view's digest left");
        }
    }
}

/// An answer that may carry the source names it by digest as before: with
/// `full_text`, and for a document with no findings under any disclosure.
#[test]
fn an_answer_that_may_carry_the_source_keeps_every_digest() {
    let pii = masked(EMAIL);
    let plain = Document::new("nothing to find here".to_owned());
    for (doc, disclosure) in [(&pii, Disclosure::FULL), (&plain, REWRITTEN), (&plain, Disclosure::FINDINGS)] {
        let typed = projected(doc, ViewId::SOURCE, &disclosure);
        let text = ViewTextDigest::of_text(doc.text(ViewId::SOURCE));
        let input = typed.input.as_ref().unwrap();
        assert_eq!(input.source_text_digest, text.as_bytes());
        assert_eq!(input.raw_bytes_digest, text.as_bytes());
        let graph = graph_bytes(doc);
        let result = typed.result.as_ref().unwrap();
        assert_eq!(result.content_digest, turbo_analysis::export::graph_content_digest(&graph).as_bytes());
        assert_eq!(result.result_id, hex(&result.content_digest));
        assert_eq!(result.graph_bytes_len, graph.len() as u64);
        assert!(typed.provenance.iter().all(|p| p.provenance_digest.len() == 32), "{:?}", typed.provenance);
        let receipt = typed.receipt.as_ref().unwrap();
        assert!(receipt.layers.iter().all(|l| l.provenance_digest.is_some()), "{:?}", receipt.layers);
        let Some(mapping_payload::Payload::Inline(mapping)) = &typed.mapping.as_ref().unwrap().payload else {
            panic!("a mapping")
        };
        assert!(mapping.views.iter().all(|v| v.text_digest.len() == 32), "{:?}", mapping.views);
    }
}

/// `RoundTripTypedDocument` reads a masked answer back, and refuses one
/// that carries a digest of what it withholds, or half an identity.
#[test]
fn the_validator_refuses_a_digest_of_withheld_text() {
    let doc = masked(EMAIL);
    let full = projected(&doc, MASKED, &Disclosure::FULL);
    let refused = |change: &dyn Fn(&mut TypedDocumentResult), on: &TypedDocumentResult| {
        let mut m = on.clone();
        change(&mut m);
        round_trip_typed_document(&m).expect_err("the changed message validates").message
    };
    let receipt_too = |m: &mut TypedDocumentResult| {
        let (input, result) = (m.input.clone(), m.result.clone());
        let r = m.receipt.as_mut().unwrap();
        (r.input, r.result) = (input, result);
    };
    for (disclosure, view) in WITHHELD {
        let typed = projected(&doc, view, &disclosure);
        assert_eq!(round_trip_typed_document(&typed).unwrap(), typed);
        // The whole identity of the full answer on a message that withholds.
        let why = refused(
            &|m| {
                (m.input, m.result) = (full.input.clone(), full.result.clone());
                receipt_too(m);
            },
            &typed,
        );
        assert!(why.contains("withh"), "{why}");
        let why = refused(
            &|m| {
                m.input.as_mut().unwrap().source_text_digest = full.input.as_ref().unwrap().source_text_digest.clone();
                receipt_too(m);
            },
            &typed,
        );
        assert!(why.contains("source_text_digest"), "{why}");
        let why = refused(
            &|m| {
                m.result.as_mut().unwrap().result_id = full.result.as_ref().unwrap().result_id.clone();
                receipt_too(m);
            },
            &typed,
        );
        assert!(why.contains("result"), "{why}");
        let why =
            refused(&|m| m.provenance[0].provenance_digest = full.provenance[0].provenance_digest.clone(), &typed);
        assert!(why.contains("provenance"), "{why}");
        let why = refused(&|m| m.provenance[0].content_digest = full.provenance[0].content_digest.clone(), &typed);
        assert!(why.contains("content_digest"), "{why}");
        let why = refused(
            &|m| {
                let Some(mapping_payload::Payload::Inline(mapping)) = &mut m.mapping.as_mut().unwrap().payload else {
                    panic!("a mapping")
                };
                mapping.views[0].text_digest = ViewTextDigest::of_text(EMAIL).as_bytes().to_vec();
            },
            &typed,
        );
        assert!(why.contains("source view"), "{why}");
        let why = refused(
            &|m| {
                m.input.as_mut().unwrap().source_text_digest = vec![7; 31];
                receipt_too(m);
            },
            &typed,
        );
        assert!(why.contains("source_text_digest"), "{why}");
    }
    // An answer that shows the source cannot leave its digests out.
    let shown = projected(&doc, ViewId::SOURCE, &Disclosure::FULL);
    let bare = projected(&doc, ViewId::SOURCE, &Disclosure::FINDINGS);
    let why = refused(
        &|m| {
            (m.input, m.result, m.provenance) = (bare.input.clone(), bare.result.clone(), bare.provenance.clone());
            (m.mapping, m.receipt) = (bare.mapping.clone(), bare.receipt.clone());
        },
        &shown,
    );
    assert!(why.contains("digest"), "{why}");
}

/// A scoped reference names its result by the graph digest, so a masked
/// answer gives none, and none resolves into it.
#[test]
fn a_masked_answer_gives_and_resolves_no_scoped_reference() {
    let doc = masked(EMAIL);
    let full = projected(&doc, MASKED, &Disclosure::FULL);
    let pii = |m: &TypedDocumentResult| m.layers.iter().position(|l| l.id == "turbo:pii").unwrap();
    let mut made = scoped_ref(&full, &full.layers[pii(&full)], 0).unwrap();
    resolve(&full, &made).unwrap();
    let typed = projected(&doc, MASKED, &REWRITTEN);
    assert!(scoped_ref(&typed, &typed.layers[pii(&typed)], 0).is_err());
    assert!(resolve(&typed, &made).is_err());
    made.result_id.clear();
    made.view_text_digest.clear();
    assert!(resolve(&typed, &made).unwrap_err().message.contains("no result identity"));
}
