//! A layer is judged by what it was made from, not by the view its spans
//! sit on (#1932, review 6726736067 on #1955). Under `rewritten_text`
//! alone: a `ProjectView` copy of the source's keywords sits on the masked
//! view but holds the source's phrases, so it is bare; linked names from
//! a linker that read the source are not joined into entities on the
//! masked view; a lemma column aligned to masked tokens has no view of its
//! own and is kept. A second rewrite over the masked view that read the
//! late finding (directly, or as a projected copy) is releasable.

use prost::Message;
use turbo_analysis::{Analyzer, Document, IxRange, Key, ProjectView, Provenance, Span, Spans, Tag, TagSet, View, ViewId};
use turbo_analysis_layers::{Entities, Lemmas, Pii, PiiEvidence, PiiKind, PiiMention, Strs, Tokens};
use turbo_entity_linker::LinkedEntities;
use turbo_keywords::Keywords;
use turbo_kserve::offsets::OffsetMode;
use turbo_kserve::proto::result::typed_annotation::Value;
use turbo_kserve::proto::result::{InspectDocumentGraphRequest, TypedDocumentResult};
use turbo_kserve::results::disclosure::{Disclosure, PiiOutputPolicy};
use turbo_kserve::results::document::codecs;
use turbo_kserve::results::typed::{document_to_typed_with, inspect_document_graph_under, validate_typed_document};
use turbo_pii::PiiDetector;
use turbo_pii::rewrite::{MASKED as MASKED_KEY, MaskedView};

const EMAIL: &str = "alice.planted@example.com";
const LATE: &str = "ZQPLANTED77";
const MASKED: ViewId = ViewId(1);
const REWRITTEN: Disclosure = Disclosure { rewritten_text: true, full_text: false, detail: false };

fn masked() -> Document {
    let analyzer =
        Analyzer::builder().add(PiiDetector::cursor()).step(MaskedView::by_kind('*').unwrap().bound()).build().unwrap();
    let mut docs = vec![Document::new(format!("Write to {EMAIL} with code {LATE} now"))];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    docs.pop().unwrap()
}

fn run(doc: Document, analyzer: &Analyzer) -> Document {
    let mut docs = vec![doc];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    docs.pop().unwrap()
}

fn holds(m: &TypedDocumentResult, needle: &str) -> bool {
    m.encode_to_vec().windows(needle.len()).any(|w| w == needle.as_bytes())
}

fn typed(doc: &Document) -> TypedDocumentResult {
    let t = document_to_typed_with("d", doc, MASKED, OffsetMode::Utf8, &REWRITTEN).unwrap();
    validate_typed_document(&t).unwrap();
    t
}

fn inspected(doc: &Document) -> TypedDocumentResult {
    let graph = turbo_analysis::export::export(doc, codecs()).unwrap().to_bytes().unwrap();
    let request = InspectDocumentGraphRequest {
        graph,
        view: u32::from(MASKED.0),
        pii_output: Some(REWRITTEN.to_options()),
        ..Default::default()
    };
    let policy = PiiOutputPolicy { full_text: false, detail: false, ..PiiOutputPolicy::ALL };
    inspect_document_graph_under(&request, policy).unwrap()
}

fn annotations(t: &TypedDocumentResult, id: &str, instance: &str) -> usize {
    t.layers.iter().find(|l| l.id == id && l.instance == instance).map_or(0, |l| l.annotations.len())
}

/// A mention of the late value on `view` of `doc`.
fn late_on(doc: &Document, view: ViewId) -> Pii {
    let mut pii = Pii::new(view);
    let at = doc.text(view).find(LATE).unwrap() as u32;
    let detector = pii.detector("test:late", "1").unwrap();
    pii.push(PiiMention {
        span: Span::new(at, at + LATE.len() as u32),
        kind: PiiKind::GithubToken,
        detector,
        evidence: &[PiiEvidence::Shape],
        score: None,
        normalized: LATE,
    });
    pii
}

#[test]
fn a_projected_copy_of_source_keywords_is_bare_under_rewritten_text() {
    let mut doc = masked();
    let at = doc.text(ViewId::SOURCE).find(EMAIL).unwrap() as u32;
    let mut kw = Keywords::new(ViewId::SOURCE);
    kw.push(Span::new(at, at + EMAIL.len() as u32), EMAIL, 0.9, Tag(0), 1);
    doc.add(&Key::<Keywords>::DEFAULT, kw, Provenance::by("test:keywords")).unwrap();
    let project = Analyzer::builder()
        .given(Key::<Keywords>::DEFAULT)
        .given(MASKED_KEY)
        .step(ProjectView::<Keywords>::to_view(Key::DEFAULT, Key::named("on_masked"), MASKED_KEY).unwrap())
        .build()
        .unwrap();
    let doc = run(doc, &project);
    for t in [typed(&doc), inspected(&doc)] {
        assert_eq!(t.raw_text, doc.text(MASKED), "the masked text still leaves");
        assert!(!holds(&t, EMAIL), "the projected keyword releases the source value");
        assert_eq!(annotations(&t, "turbo:keywords", "on_masked"), 0);
    }
}

/// Tokens on the masked view, read from it, and lemmas aligned to them.
fn masked_tokens_and_lemmas(doc: &mut Document) -> Key<Tokens> {
    let text = doc.text(MASKED).to_owned();
    let mut spans = Spans::new(MASKED);
    let mut lemma = Strs::default();
    let mut at = 0u32;
    for word in text.split(' ') {
        spans.push(Span::new(at, at + word.len() as u32));
        lemma.push(&word.to_lowercase());
        at += word.len() as u32 + 1;
    }
    let tok = Key::<Tokens>::named("masked");
    let reads = vec![MASKED_KEY.id()];
    doc.add(&tok, Tokens { spans, sentence: None }, Provenance { reads, ..Provenance::by("test:tok") }).unwrap();
    let on = vec![tok.id()];
    doc.add(&Key::<Lemmas>::named("masked"), Lemmas { lemma }, Provenance { on, ..Provenance::by("test:lemma") })
        .unwrap();
    tok
}

#[test]
fn layers_made_from_the_masked_view_are_kept_and_source_links_are_not_joined() {
    let mut doc = masked();
    let tok = masked_tokens_and_lemmas(&mut doc);
    let words = doc.text(MASKED).split(' ').count() as u32;
    let mut spans = Spans::new(MASKED);
    spans.push(Span::new(0, 5));
    let entities = Key::<Entities>::named("masked");
    doc.add(
        &entities,
        Entities {
            spans,
            tokens: vec![IxRange::new(0, 1)],
            kind: vec![Tag(0)],
            kinds: TagSet::new("ner", &["THING"]),
            score: vec![0.9],
        },
        Provenance { reads: vec![tok.id()], ..Provenance::by("test:ner") },
    )
    .unwrap();
    let shown = typed(&doc);
    assert_eq!(annotations(&shown, "turbo:lemmas", "masked") as u32, words, "masked lemmas are kept");
    assert_eq!(annotations(&shown, "turbo:entities", "masked"), 1, "masked entities are kept");
    // A linker that read the source names the entity by the source value.
    let links = LinkedEntities {
        kb_id: vec![None],
        provider_id: vec!["fixture".into()],
        canonical_name: vec![EMAIL.into()],
        matched_alias: vec![EMAIL.into()],
        score: vec![0.9],
        nil_reason: vec![None],
        nil_cluster: vec![None],
        candidates: vec![vec![]],
    };
    let source = doc.layers().find(|k| k.ty == std::any::TypeId::of::<View>() && k.name.is_empty()).unwrap();
    let p = Provenance { on: vec![entities.id()], reads: vec![source], ..Provenance::by("test:linker") };
    doc.add(&Key::<LinkedEntities>::DEFAULT, links, p).unwrap();
    for t in [typed(&doc), inspected(&doc)] {
        assert!(!holds(&t, EMAIL), "a linked name read from the source is joined into a masked entity");
        assert_eq!(annotations(&t, "turbo:entities", "masked"), 0);
        assert!(annotations(&t, "turbo:lemmas", "masked") > 0, "the lemmas do not depend on the links");
    }
    let full = document_to_typed_with("d", &doc, MASKED, OffsetMode::Utf8, &Disclosure::FULL).unwrap();
    let ent = full.layers.iter().find(|l| l.id == "turbo:entities" && l.instance == "masked").unwrap();
    assert!(matches!(&ent.annotations[0].value, Some(Value::Entity(e)) if e.canonical == EMAIL), "positive control");
}

#[test]
fn a_second_rewrite_that_read_the_late_finding_is_releasable() {
    // Directly: the late detector ran on the masked view.
    let mut doc = masked();
    let late = late_on(&doc, MASKED);
    doc.add(&Key::<Pii>::named("late"), late, Provenance::by("test:late-pii")).unwrap();
    assert!(!REWRITTEN.view_text(&doc, MASKED), "the first rewrite missed the late finding");
    let second = Analyzer::builder()
        .given(MASKED_KEY)
        .given(Key::<Pii>::named("late"))
        .step(MaskedView::by_kind('*').unwrap().bound().read(0, "masked").read(1, "late").write(0, "masked2"))
        .build()
        .unwrap();
    let doc = run(doc, &second);
    let masked2 = doc.get_key(&Key::<View>::named("masked2")).unwrap().id();
    assert!(!doc.text(masked2).contains(LATE) && !doc.text(masked2).contains(EMAIL));
    assert!(REWRITTEN.view_text(&doc, masked2), "a rewrite that covered every finding is withheld");
    // Projected: the late detector ran on the source and was copied over.
    let mut doc = masked();
    let late = late_on(&doc, ViewId::SOURCE);
    doc.add(&Key::<Pii>::named("late"), late, Provenance::by("test:late-pii")).unwrap();
    let second = Analyzer::builder()
        .given(Key::<Pii>::named("late"))
        .given(MASKED_KEY)
        .step(ProjectView::<Pii>::to_view(Key::named("late"), Key::named("late_on_masked"), MASKED_KEY).unwrap())
        .step(MaskedView::by_kind('*').unwrap().bound().read(0, "masked").read(1, "late_on_masked").write(0, "masked2"))
        .build()
        .unwrap();
    let doc = run(doc, &second);
    let masked2 = doc.get_key(&Key::<View>::named("masked2")).unwrap().id();
    assert!(!doc.text(masked2).contains(LATE));
    assert!(REWRITTEN.view_text(&doc, masked2), "a rewrite of a projected copy does not cover its source");
    assert!(!REWRITTEN.view_text(&doc, MASKED), "the first rewrite still missed it");
}
