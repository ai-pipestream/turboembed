//! A typed result shown on a PII rewrite view under `rewritten_text`
//! alone (#1904): the masked view's own text leaves, but a layer native to
//! the source (its lemmas, here one per token, the address's lemma being
//! the address) carries the source's words, so it is listed bare, and the
//! result still validates. With `full_text` the lemmas come back.

use prost::Message;
use turbo_analysis::{Analyzer, Document, Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{Lemmas, Strs, Tokens};
use turbo_kserve::offsets::OffsetMode;
use turbo_kserve::proto::result::typed_annotation::Value;
use turbo_kserve::proto::result::{PiiOutputOptions, TypedDocumentResult};
use turbo_kserve::results::disclosure::{Disclosure, PiiOutputPolicy};
use turbo_kserve::results::typed::{document_to_typed_with, validate_typed_document};
use turbo_pii::PiiDetector;
use turbo_pii::rewrite::MaskedView;

const PLANTED: &str = "alice.planted@example.com";
const MASKED: ViewId = ViewId(1);

/// "Write to <PLANTED> now", scanned and masked, with tokens and lemmas on
/// the source; the address's lemma is the address.
fn doc() -> Document {
    let text = format!("Write to {PLANTED} now");
    let analyzer =
        Analyzer::builder().add(PiiDetector::cursor()).step(MaskedView::by_kind('*').unwrap().bound()).build().unwrap();
    let mut docs = vec![Document::new(text.clone())];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    let mut doc = docs.pop().unwrap();
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut lemma = Strs::default();
    let mut at = 0u32;
    for word in text.split(' ') {
        spans.push(Span::new(at, at + word.len() as u32));
        lemma.push(&word.to_lowercase());
        at += word.len() as u32 + 1;
    }
    doc.add(&Key::<Tokens>::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("test:tok")).unwrap();
    let on_tok = Provenance { on: vec![Key::<Tokens>::DEFAULT.id()], ..Provenance::by("test:lemma") };
    doc.add(&Key::<Lemmas>::DEFAULT, Lemmas { lemma }, on_tok).unwrap();
    doc
}

fn holds(m: &TypedDocumentResult) -> bool {
    m.encode_to_vec().windows(PLANTED.len()).any(|w| w == PLANTED.as_bytes())
}

fn lemmas(m: &TypedDocumentResult) -> usize {
    let l = m.layers.iter().find(|l| l.id == "turbo:lemmas").expect("the lemmas layer is listed");
    l.annotations.iter().filter(|a| matches!(a.value, Some(Value::Lemma(_)))).count()
}

fn decide(full_text: bool, rewritten_text: bool) -> Disclosure {
    let o = PiiOutputOptions { full_text, rewritten_text, ..Default::default() };
    Disclosure::decide(Some(&o), PiiOutputPolicy::ALL).unwrap()
}

#[test]
fn rewritten_only_projection_must_not_restore_source_lemma_value() {
    let doc = doc();
    assert!(!doc.text(MASKED).contains(PLANTED), "view 1 is the masked view");
    // Positive control: with the source disclosed the lemma is the value.
    let full = document_to_typed_with("d", &doc, MASKED, OffsetMode::Utf8, &decide(true, false)).unwrap();
    assert_eq!(lemmas(&full), 4);
    assert!(holds(&full), "positive control: a source lemma holds the value");
    // rewritten_text alone: the masked text, the findings, no source words.
    let masked = document_to_typed_with("d", &doc, MASKED, OffsetMode::Utf8, &decide(false, true)).unwrap();
    assert_eq!(masked.raw_text, doc.text(MASKED));
    assert_eq!(lemmas(&masked), 0, "the source's lemmas are listed bare");
    assert!(!holds(&masked), "the rewritten-only result holds the value");
    let pii = masked.layers.iter().find(|l| l.id == "turbo:pii").expect("the findings are kept");
    assert!(!pii.annotations.is_empty());
    validate_typed_document(&masked).unwrap();
    // A bare layer from another view is accepted only beside a rewrite view.
    let mut source = document_to_typed_with("d", &doc, ViewId::SOURCE, OffsetMode::Utf8, &decide(true, false)).unwrap();
    source.layers.iter_mut().find(|l| l.id == "turbo:lemmas").unwrap().annotations.clear();
    assert!(validate_typed_document(&source).is_err(), "a bare lemma layer on the source view validates");
}
