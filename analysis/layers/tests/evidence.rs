//! Evidence for result layers made on an anchor their step wrote: the
//! records follow the instance the document bound the result to (the
//! provenance's `on` before its `reads`), and a score the step must not
//! record fails the step.

use turbo_analysis::{
    Analyzer, Document, Evidence, EvidenceStep, IxRange, Key, ModelRef, Provenance, Span, Spans, Status, Tag, TagSet,
    ViewId,
};
use turbo_analysis_layers::{Categories, Entities, Sentences, Tokens};

/// "old new" with two named instances of `T`, `old` at 0..3 and `new` at
/// 4..7, each made by `make`.
fn document<T: turbo_analysis::Layer>(make: impl Fn(Span) -> T) -> Document {
    let mut doc = Document::new("old new");
    for (name, span) in [("old", Span::new(0, 3)), ("new", Span::new(4, 7))] {
        doc.add(&Key::named(name), make(span), Provenance::by("test")).unwrap();
    }
    doc
}

fn spans<L>(s: Span) -> Spans<L> {
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(s);
    spans
}

/// A result read from `old` and made on `new`, its step's own output.
fn provenance<T: turbo_analysis::Layer>(model: Option<ModelRef>) -> Provenance {
    Provenance {
        reads: vec![Key::<T>::named("old").id()],
        on: vec![Key::<T>::named("new").id()],
        model,
        ..Provenance::by("test:finder")
    }
}

fn run<L: turbo_analysis::Evidenced>(doc: Document, given: Key<L>) -> Document {
    let analyzer = Analyzer::builder().given(given).step(EvidenceStep::new(given)).build().unwrap();
    let mut docs = vec![doc];
    let _ = analyzer.run(&mut docs);
    docs.pop().unwrap()
}

fn entity(score: f64) -> Entities {
    Entities {
        spans: spans(Span::new(4, 7)),
        tokens: vec![IxRange::new(0, 1)],
        kind: vec![Tag(0)],
        kinds: TagSet::new("kinds", &["thing"]),
        score: vec![score],
    }
}

fn model() -> ModelRef {
    ModelRef { id: "finder".into(), revision: "1".into(), sha256: "ab".into() }
}

#[test]
fn entity_support_names_the_tokens_the_entity_was_made_on() {
    let mut doc = document(|s| Tokens { spans: spans(s), sentence: None });
    doc.add(&Key::DEFAULT, entity(0.5), provenance::<Tokens>(None)).unwrap();
    let doc = run(doc, Key::<Entities>::DEFAULT);
    let ev = doc.get::<Evidence<Entities>>().unwrap();
    assert_eq!((ev.layers[0].layer.as_str(), ev.layers[0].instance.as_str()), ("opennlp:tokens", "new"));
    assert_eq!(ev.spans_of(0), [Span::new(4, 7)]);
    assert_eq!(ev.verify(&doc), Ok(()));
}

#[test]
fn sentence_categories_take_the_spans_of_the_sentences_they_were_made_on() {
    let mut doc = document(|s| Sentences { spans: spans(s) });
    let mut categories = Categories::<Sentences>::new(TagSet::new("cats", &["a", "b"]));
    categories.push(Tag(1), &[0.25, 0.75]);
    doc.add(&Key::DEFAULT, categories, provenance::<Sentences>(None)).unwrap();
    let doc = run(doc, Key::<Categories<Sentences>>::DEFAULT);
    let ev = doc.get::<Evidence<Categories<Sentences>>>().unwrap();
    assert_eq!(ev.spans_of(0), [Span::new(4, 7)]);
    assert_eq!(ev.raw_of(0), Some(0.75));
}

#[test]
fn a_score_that_is_not_finite_fails_the_step() {
    let mut doc = document(|s| Tokens { spans: spans(s), sentence: None });
    doc.add(&Key::DEFAULT, entity(f64::INFINITY), provenance::<Tokens>(Some(model()))).unwrap();
    let doc = run(doc, Key::<Entities>::DEFAULT);
    assert!(doc.get::<Evidence<Entities>>().is_none());
    let status = doc.status(Key::<Evidence<Entities>>::DEFAULT.id());
    assert!(matches!(status, Some(Status::Failed(_))), "{status:?}");
}
