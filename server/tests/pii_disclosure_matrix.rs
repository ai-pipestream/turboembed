//! The disclosure matrix for stored strings (#1932): a source keyword
//! whose phrase is the planted value, searched for in the encoded bytes of
//! every answer under findings only, rewritten text only, rewritten text
//! with detail, and full text. Detail is decided on its own, whatever the
//! text options, and a server that denies it refuses it. A keyword on a
//! sibling view that is not a PII rewrite stays bare under rewritten text;
//! one on the masked view itself leaves (positive control). Offsets are
//! UTF-16 over a text with non-ASCII before the value.

use prost::Message;
use turbo_analysis::{Analyzer, Document, Key, Provenance, Span, Tag, View, ViewBuilder, ViewId};
use turbo_keywords::Keywords;
use turbo_kserve::offsets::OffsetMode;
use turbo_kserve::proto::result::{InspectDocumentGraphRequest, PiiOutputOptions, TypedDocumentResult};
use turbo_kserve::results::disclosure::{Disclosure, PiiOutputPolicy};
use turbo_kserve::results::document::{codecs, export_document_response};
use turbo_kserve::results::typed::{document_to_typed_with, inspect_document_graph_under, validate_typed_document};
use turbo_pii::PiiDetector;
use turbo_pii::rewrite::MaskedView;

const PLANTED: &str = "alice.planted@example.com";
const PREFIX: &str = "Grüße, café 東京: write to ";
const MASKED: ViewId = ViewId(1);
const PLAIN: Key<View> = Key::named("plain");

fn at() -> u32 {
    PREFIX.len() as u32
}

fn keyword(doc: &mut Document, view: ViewId, name: &'static str, phrase: &str) {
    let mut k = Keywords::new(view);
    k.push(Span::new(at(), at() + PLANTED.len() as u32), phrase, 0.9, Tag(0), 1);
    doc.add(&Key::named(name), k, Provenance::by("test:keywords")).unwrap();
}

/// The text scanned, masked, with a keyword holding the value on the
/// source, one holding the masked text on the masked view, and a sibling
/// view of the source (not a rewrite) with a keyword holding the value.
fn doc() -> Document {
    let analyzer =
        Analyzer::builder().add(PiiDetector::cursor()).step(MaskedView::by_kind('*').unwrap().bound()).build().unwrap();
    let mut docs = vec![Document::new(format!("{PREFIX}{PLANTED} before noon."))];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    let mut doc = docs.pop().unwrap();
    let masked_phrase = doc.text(MASKED)[at() as usize..at() as usize + PLANTED.len()].to_owned();
    keyword(&mut doc, ViewId::SOURCE, "source", PLANTED);
    keyword(&mut doc, MASKED, "masked", &masked_phrase);
    let source = doc.get::<View>().unwrap().clone();
    let mut b = ViewBuilder::new(source.text());
    b.keep(source.text().len());
    let (text, alignment) = b.finish();
    doc.add(&PLAIN, View::derived(&source, text, alignment), Provenance::by("test:plain")).unwrap();
    let plain = doc.get_key(&PLAIN).unwrap().id();
    keyword(&mut doc, plain, "plain", PLANTED);
    doc
}

fn holds(m: &impl Message) -> bool {
    m.encode_to_vec().windows(PLANTED.len()).any(|w| w == PLANTED.as_bytes())
}

fn decide(rewritten_text: bool, full_text: bool, detail: bool) -> Disclosure {
    let o = PiiOutputOptions { rewritten_text, full_text, detail, ..Default::default() };
    Disclosure::decide(Some(&o), PiiOutputPolicy::ALL).unwrap()
}

fn typed(doc: &Document, d: &Disclosure) -> TypedDocumentResult {
    let t = document_to_typed_with("d", doc, MASKED, OffsetMode::Utf16, d).unwrap();
    validate_typed_document(&t).unwrap();
    t
}

fn annotations(t: &TypedDocumentResult, id: &str, instance: &str) -> usize {
    t.layers.iter().find(|l| l.id == id && l.instance == instance).map_or(0, |l| l.annotations.len())
}

#[test]
fn stored_strings_follow_the_text_options_and_detail_stands_alone() {
    let doc = doc();
    assert!(!doc.text(MASKED).contains(PLANTED));
    // Findings only: no text, no keyword, no candidates, findings kept.
    let findings = typed(&doc, &Disclosure::FINDINGS);
    assert!(findings.raw_text.is_empty() && !holds(&findings));
    assert_eq!(annotations(&findings, "turbo:keywords", "masked"), 0);
    assert_eq!(annotations(&findings, "turbo:pii-candidates", ""), 0);
    assert!(annotations(&findings, "turbo:pii", "") > 0);
    // Rewritten only: the masked text and its own keyword, nothing of the
    // source or the sibling view, no candidates.
    let rewritten = typed(&doc, &decide(true, false, false));
    assert_eq!(rewritten.raw_text, doc.text(MASKED));
    assert!(!holds(&rewritten), "rewritten-only releases a stored source string");
    assert_eq!(annotations(&rewritten, "turbo:keywords", "masked"), 1, "positive control");
    assert_eq!(annotations(&rewritten, "turbo:keywords", "source"), 0);
    assert_eq!(annotations(&rewritten, "turbo:keywords", "plain"), 0);
    assert_eq!(annotations(&rewritten, "turbo:pii-candidates", ""), 0, "detail came with rewritten text");
    // Detail alone, and with rewritten text: candidates, still no value.
    for d in [decide(false, false, true), decide(true, false, true)] {
        let t = typed(&doc, &d);
        assert!(annotations(&t, "turbo:pii-candidates", "") > 0, "{d:?} drops the candidates");
        assert!(!holds(&t), "{d:?} releases the value");
    }
    // Full text: the source keyword comes back (positive control).
    let full = typed(&doc, &decide(false, true, false));
    assert_eq!(annotations(&full, "turbo:keywords", "source"), 1);
    assert!(holds(&full));
}

#[test]
fn a_server_that_denies_detail_refuses_it_beside_rewritten_text() {
    let no_detail = PiiOutputPolicy { detail: false, full_text: false, ..PiiOutputPolicy::ALL };
    let o = PiiOutputOptions { rewritten_text: true, detail: true, ..Default::default() };
    assert!(Disclosure::decide(Some(&o), no_detail).is_err());
    let doc = doc();
    let graph = turbo_analysis::export::export(&doc, codecs()).unwrap().to_bytes().unwrap();
    let request =
        InspectDocumentGraphRequest { graph: graph.clone(), view: 1, pii_output: Some(o), ..Default::default() };
    assert!(inspect_document_graph_under(&request, no_detail).is_err());
    // Rewritten text alone under that policy, through export and import.
    let request = InspectDocumentGraphRequest {
        graph,
        view: 1,
        pii_output: Some(decide(true, false, false).to_options()),
        ..Default::default()
    };
    let shown = inspect_document_graph_under(&request, no_detail).unwrap();
    assert_eq!(shown.raw_text, doc.text(MASKED));
    assert!(!holds(&shown));
    assert_eq!(annotations(&shown, "turbo:pii-candidates", ""), 0);
}

#[test]
fn the_graph_response_under_rewritten_text_carries_no_value() {
    let doc = doc();
    let r = export_document_response(&doc, &decide(true, false, false)).unwrap();
    assert!(r.graph.is_empty() && r.source_text.is_empty());
    assert!(!holds(&r), "the graph response releases the value");
    let full = export_document_response(&doc, &decide(false, true, false)).unwrap();
    assert!(holds(&full), "positive control");
}
