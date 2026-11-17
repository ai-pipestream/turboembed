//! `ProjectSearchDocument` on a PII rewrite view under `rewritten_text`
//! alone (#1932, #2345): the masked view's own text and positions may
//! leave, but a term, an entity name or a value column made from the source
//! view carries the source's words, so it is left out unless its whole
//! lineage may leave. Layers made from the masked view are kept.

use prost::Message;
use turbo_analysis::{
    Analyzer, Bound, Document, IxRange, Key, KeyId, Provenance, Span, Spans, Tag, TagSet, View, ViewId,
};
use turbo_analysis_layers::{Entities, NumberNotation, TokenGraph, TokenGraphBuilder, Tokens};
use turbo_kserve::offsets::OffsetMode;
use turbo_kserve::proto::result::{ProjectSearchDocumentRequest, SearchDocumentProjection};
use turbo_kserve::results::disclosure::{Disclosure, PiiOutputPolicy};
use turbo_kserve::results::document::codecs;
use turbo_kserve::results::search_projection::{
    document_to_search_projection_bound, project_search_document_under, validate_search_projection,
};
use turbo_numeric::money::MoneyExtractor;
use turbo_pii::PiiDetector;
use turbo_pii::rewrite::{MASKED as MASKED_KEY, MaskedView};

const EMAIL: &str = "alice.planted@example.com";
const MASKED: ViewId = ViewId(1);
const REWRITTEN: Disclosure = Disclosure { rewritten_text: true, full_text: false, detail: false };

fn masked() -> Document {
    let analyzer =
        Analyzer::builder().add(PiiDetector::cursor()).step(MaskedView::by_kind('*').unwrap().bound()).build().unwrap();
    let mut docs = vec![Document::new(format!("Write to {EMAIL} about Acme today"))];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    docs.pop().unwrap()
}

/// The spans of the space-separated words of `view`.
fn word_spans(doc: &Document, view: ViewId) -> Vec<Span> {
    let mut spans = Vec::new();
    let mut at = 0u32;
    for word in doc.text(view).split(' ') {
        spans.push(Span::new(at, at + word.len() as u32));
        at += word.len() as u32 + 1;
    }
    spans
}

fn spans_on<L>(view: ViewId, list: &[Span]) -> Spans<L> {
    let mut s = Spans::new(view);
    for &sp in list {
        s.push(sp);
    }
    s
}

/// Tokens under `name` on `view`, read from the view under `view_key`.
fn add_tokens(doc: &mut Document, name: &'static str, view: ViewId, view_key: KeyId) {
    let tokens = Tokens { spans: spans_on(view, &word_spans(doc, view)), sentence: None };
    doc.add(&Key::<Tokens>::named(name), tokens, Provenance { reads: vec![view_key], ..Provenance::by("test:tok") })
        .unwrap();
}

/// A token graph under `name` whose terms are the words of `view`.
fn add_graph(doc: &mut Document, name: &'static str, view: ViewId, view_key: KeyId) {
    let text = doc.text(view).to_owned();
    let spans = word_spans(doc, view);
    let terms = spans.iter().map(|s| text[s.range()].to_lowercase()).collect();
    let graph = TokenGraphBuilder::from_base_tokens(view, spans, terms).finish().unwrap();
    doc.add(
        &Key::<TokenGraph>::named(name),
        graph,
        Provenance { reads: vec![view_key], ..Provenance::by("test:graph") },
    )
    .unwrap();
}

/// Tokens and one entity over the word at `word` of `view`, both under `name`.
fn add_entity(doc: &mut Document, name: &'static str, view: ViewId, view_key: KeyId, word: usize) {
    add_tokens(doc, name, view, view_key);
    let ents = Entities {
        spans: spans_on(view, &[word_spans(doc, view)[word]]),
        tokens: vec![IxRange::new(word as u32, word as u32 + 1)],
        kind: vec![Tag(0)],
        kinds: TagSet::new("test", &["CONTACT"]),
        score: vec![0.9],
    };
    doc.add(
        &Key::<Entities>::named(name),
        ents,
        Provenance { on: vec![Key::<Tokens>::named(name).id()], ..Provenance::by("test:ner") },
    )
    .unwrap();
}

fn holds(m: &SearchDocumentProjection, needle: &str) -> bool {
    m.encode_to_vec().windows(needle.len()).any(|w| w == needle.as_bytes())
}

fn projected(doc: &Document, disclosure: &Disclosure) -> SearchDocumentProjection {
    let p = document_to_search_projection_bound("d", doc, MASKED, OffsetMode::Utf8, None, &[], disclosure).unwrap();
    validate_search_projection(&p).unwrap();
    p
}

/// The same projection through the request handler, with the server's
/// policy capped at `rewritten_text`.
fn handled(doc: &Document) -> SearchDocumentProjection {
    let graph = turbo_analysis::export::export(doc, codecs()).unwrap().to_bytes().unwrap();
    let request = ProjectSearchDocumentRequest {
        doc_id: "d".into(),
        graph,
        view: u32::from(MASKED.0),
        pii_output: Some(REWRITTEN.to_options()),
        ..Default::default()
    };
    let policy = PiiOutputPolicy { full_text: false, detail: false, ..PiiOutputPolicy::ALL };
    project_search_document_under(&request, policy).unwrap()
}

#[test]
fn source_tokens_do_not_release_the_masked_value() {
    let mut doc = masked();
    add_tokens(&mut doc, "", ViewId::SOURCE, Key::<View>::DEFAULT.id());
    assert!(!doc.text(MASKED).contains(EMAIL), "the fixture masks the email");
    for p in [projected(&doc, &REWRITTEN), handled(&doc)] {
        assert!(!holds(&p, EMAIL), "a source token releases the masked value: {:?}", p.tokens);
        assert!(p.tokens.is_empty(), "{:?}", p.tokens);
    }
    // With the full text asked for and allowed, the terms are the source's.
    let full = projected(&doc, &Disclosure::FULL);
    assert!(full.tokens.iter().any(|t| t.term == EMAIL), "{:?}", full.tokens);
}

#[test]
fn a_source_token_graph_and_source_entities_do_not_release_the_masked_value() {
    let mut doc = masked();
    let source = Key::<View>::DEFAULT.id();
    add_graph(&mut doc, "body", ViewId::SOURCE, source);
    add_entity(&mut doc, "", ViewId::SOURCE, source, 2);
    for p in [projected(&doc, &REWRITTEN), handled(&doc)] {
        assert!(!holds(&p, EMAIL), "source text left: {p:?}");
        assert!(p.tokens.is_empty(), "{:?}", p.tokens);
        let entities = p.entities.unwrap();
        assert!(entities.affirmed.is_empty() && entities.negated.is_empty() && entities.uncertain.is_empty());
    }
    let full = projected(&doc, &Disclosure::FULL);
    assert!(full.tokens.iter().any(|t| t.term == EMAIL));
    assert_eq!(full.entities.unwrap().affirmed[0].canonical_name, EMAIL);
}

#[test]
fn layers_made_from_the_masked_view_are_kept() {
    let mut doc = masked();
    let source = Key::<View>::DEFAULT.id();
    // The source's graph is dropped, the masked view's graph beside it kept.
    add_graph(&mut doc, "source_body", ViewId::SOURCE, source);
    add_graph(&mut doc, "masked_body", MASKED, MASKED_KEY.id());
    add_entity(&mut doc, "masked", MASKED, MASKED_KEY.id(), 4);
    for p in [projected(&doc, &REWRITTEN), handled(&doc)] {
        assert!(!holds(&p, EMAIL));
        let terms: Vec<&str> = p.tokens.iter().map(|t| t.term.as_str()).collect();
        assert_eq!(p.tokens.iter().filter(|t| t.instance == "masked_body").count(), terms.len(), "{terms:?}");
        assert_eq!(terms.len(), 6, "{terms:?}");
        assert_eq!((terms[0], terms[1], terms[4]), ("write", "to", "acme"));
        let entities = p.entities.unwrap();
        assert_eq!(entities.affirmed.len(), 1);
        assert_eq!(entities.affirmed[0].canonical_name, "Acme");
    }
}

/// The fixture's text with a price in it, and a money layer read from the
/// source view or from the masked one.
fn priced(from_masked: bool) -> Document {
    let money = Bound::new(MoneyExtractor::new(NumberNotation::LatinUs));
    let money = if from_masked { money.read(0, "masked") } else { money };
    let analyzer = Analyzer::builder()
        .add(PiiDetector::cursor())
        .step(MaskedView::by_kind('*').unwrap().bound())
        .step(money)
        .build()
        .unwrap();
    let mut docs = vec![Document::new(format!("Write to {EMAIL} about the $5,000 fee"))];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    docs.pop().unwrap()
}

#[test]
fn a_value_column_read_from_the_source_is_left_out() {
    let doc = priced(false);
    for p in [projected(&doc, &REWRITTEN), handled(&doc)] {
        assert!(p.numeric_temporal.unwrap().money.is_empty(), "a column made from the source view left");
    }
    let full = projected(&doc, &Disclosure::FULL).numeric_temporal.unwrap();
    assert_eq!(full.money.len(), 1, "{:?}", full.money);
    assert_eq!(full.money[0].plain_amount, "5000");
}

#[test]
fn a_value_column_read_from_the_masked_view_is_kept() {
    let doc = priced(true);
    for p in [projected(&doc, &REWRITTEN), handled(&doc)] {
        let money = p.numeric_temporal.unwrap().money;
        assert_eq!(money.len(), 1, "{money:?}");
        assert_eq!((money[0].plain_amount.as_str(), money[0].currency.as_str()), ("5000", "USD"));
    }
}
