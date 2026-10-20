//! Conditional branch selection and annotation filtering, ported from
//! OpenNLP's `PredicateAnnotatorsTest` (opennlp.tools.document).

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use turbo_analysis::annotate::Step;
use turbo_analysis::{
    Annotator, Bound, ConditionalAnnotator, Document, Error, FilterAnnotator, Filterable, Ix, Key, Layer, Overlap,
    Provenance, Refs, Span, SpanItem, Spans, Status, TextSpans, View, ViewId,
};
use turbo_analysis_layers::{Confidence, Strs, Tokens};

const WORDS: Key<Tokens> = Key::DEFAULT;
const LONG_WORDS: Key<Tokens> = Key::named("long");

/// A minimal producing annotator: every whitespace-delimited run is a
/// word, like the Java fixture's `PRODUCER`.
struct Producer;

impl Annotator for Producer {
    const ID: &'static str = "test:words";
    type Reads = (View,);
    type Writes = (Tokens,);

    fn annotate(&self, (view,): Refs<'_, Self>) -> Result<(Tokens,), Error> {
        let text = view.text();
        let mut spans = Spans::new(view.id());
        let mut at = 0;
        for word in text.split(' ') {
            if !word.is_empty() {
                spans.push(Span::new(at as u32, (at + word.len()) as u32));
            }
            at += word.len() + 1;
        }
        Ok((Tokens { spans, sentence: None },))
    }
}

/// A delegate that must never run: its outputs are what the assertions
/// read, like the Java fixture's two-layer delegate.
struct TwoLayers;

impl Annotator for TwoLayers {
    const ID: &'static str = "test:two-layers";
    type Reads = (View,);
    type Writes = (Tokens, Confidence<Tokens>);

    fn annotate(&self, _: Refs<'_, Self>) -> Result<(Tokens, Confidence<Tokens>), Error> {
        panic!("the delegate must not run when the condition fails")
    }
}

/// A span-anchored layer whose items may be discontinuous.
struct Mentions {
    spans: Spans<Mentions>,
}

impl Layer for Mentions {
    const ID: &'static str = "test:mentions";
    const OVERLAP: Overlap = Overlap::Nesting;

    fn len(&self) -> usize {
        self.spans.len()
    }

    fn text_spans(&self) -> Option<TextSpans<'_>> {
        Some(self.spans.text_spans())
    }
}

impl Filterable for Mentions {
    type Item = SpanItem;

    fn item(&self, i: usize) -> SpanItem {
        SpanItem { view: self.spans.view(), span: self.spans.get(Ix::new(i as u32)) }
    }

    fn filter(&self, keep: &mut dyn FnMut(&SpanItem) -> bool) -> Mentions {
        let kept: Vec<usize> = (0..self.spans.len()).filter(|&i| keep(&self.item(i))).collect();
        Mentions { spans: self.spans.subset(&kept) }
    }
}

/// A document-scoped label layer, for the document-scoped filter port.
struct Labels {
    labels: Strs,
}

impl Layer for Labels {
    const ID: &'static str = "test:labels";

    fn anchor() -> turbo_analysis::Anchor {
        turbo_analysis::Anchor::Document
    }

    fn len(&self) -> usize {
        self.labels.len()
    }
}

impl Filterable for Labels {
    type Item = String;

    fn item(&self, i: usize) -> String {
        self.labels.get(i).into()
    }

    fn filter(&self, keep: &mut dyn FnMut(&String) -> bool) -> Labels {
        let mut labels = Strs::default();
        for i in 0..self.len() {
            let label = self.item(i);
            if keep(&label) {
                labels.push(&label);
            }
        }
        Labels { labels }
    }
}

const CATEGORIES: Key<Labels> = Key::named("categories");
const SELECTED: Key<Labels> = Key::named("selected");

/// Runs one step over one document, as `Analyzer::run` would.
fn run_step(step: &dyn Step, doc: Document) -> (Document, Result<(), Error>) {
    let mut docs = vec![doc];
    let mut status = vec![Ok(())];
    let out = step.compute(&docs);
    step.store(out, &mut docs, &mut status);
    (docs.pop().unwrap(), status.pop().unwrap())
}

fn construction_err<T>(built: Result<T, Error>) -> Error {
    match built {
        Ok(_) => panic!("the construction must be refused"),
        Err(e) => e,
    }
}

/// The document `Producer` makes for `text`.
fn produced(text: &str) -> Document {
    let (doc, status) = run_step(&Bound::new(Producer), Document::new(text));
    assert!(status.is_ok(), "{status:?}");
    doc
}

/// The words a tokens layer covers, as text.
fn words_of(doc: &Document, key: &Key<Tokens>) -> Vec<String> {
    let tokens = doc.get_key(key).unwrap();
    let view = tokens.spans.view();
    tokens.spans.iter().map(|(_, s)| doc.text(view)[s.range()].to_string()).collect()
}

/// The word-length filter: the document-aware form reads the covered
/// text, the annotation-only form uses the envelope span.
fn long_enough(min: u32) -> FilterAnnotator<Tokens> {
    FilterAnnotator::with_document(WORDS, LONG_WORDS, move |d: &Document, t: &SpanItem| {
        d.text(t.view)[t.span.range()].len() as u32 >= min
    })
    .unwrap()
}

#[test]
fn the_condition_runs_the_delegate() {
    let conditional = ConditionalAnnotator::when(|d: &Document| d.source().len() > 3, Bound::new(Producer)).unwrap();
    let (doc, status) = run_step(&conditional, Document::new("two words"));
    assert!(status.is_ok(), "{status:?}");
    assert_eq!(doc.get_key(&WORDS).unwrap().len(), 2);
}

#[test]
fn a_skipped_delegate_still_declares_its_layers() {
    let guarded = ConditionalAnnotator::when(|_: &Document| false, Bound::new(Producer)).unwrap();
    let (doc, status) = run_step(&guarded, Document::new("two words"));
    assert!(status.is_ok(), "a declined branch is not a failure: {status:?}");
    // The typed pipeline records the provided layers skipped, its form of
    // the Java form's empty layers: never mistaken for "found nothing".
    assert_eq!(doc.status(WORDS.id()), Some(&Status::Skipped("condition is false".into())));
    assert!(doc.get_key(&WORDS).is_none());
    assert_eq!(guarded.writes(), Bound::new(Producer).writes());
    assert_eq!(guarded.reads(), Bound::new(Producer).reads());
}

#[test]
fn a_skipped_delegate_declares_every_one_of_its_layers() {
    let guarded = ConditionalAnnotator::when(|_: &Document| false, Bound::new(TwoLayers)).unwrap();
    let (doc, status) = run_step(&guarded, Document::new("two words"));
    assert!(status.is_ok(), "{status:?}");
    let skipped = Status::Skipped("condition is false".into());
    assert_eq!(doc.status(WORDS.id()), Some(&skipped));
    let confidences = Key::<Confidence<Tokens>>::DEFAULT.id();
    assert_eq!(doc.status(confidences), Some(&skipped));
}

#[test]
fn the_filter_writes_survivors_and_keeps_the_source() {
    let filter = long_enough(5);
    let (doc, status) = run_step(&filter, produced("a lengthy pair of words"));
    assert!(status.is_ok(), "{status:?}");
    assert_eq!(doc.get_key(&WORDS).unwrap().len(), 5, "the source layer must be unchanged");
    assert_eq!(words_of(&doc, &LONG_WORDS), ["lengthy", "words"]);
    assert_eq!(filter.reads(), vec![WORDS.id()]);
    assert_eq!(filter.writes(), vec![LONG_WORDS.id()]);
}

#[test]
fn a_filter_on_an_empty_source_provides_an_empty_target() {
    let filter = FilterAnnotator::with_document(WORDS, LONG_WORDS, |_: &Document, _: &SpanItem| true).unwrap();
    let (doc, status) = run_step(&filter, produced(""));
    assert!(status.is_ok(), "{status:?}");
    assert!(doc.get_key(&LONG_WORDS).is_some());
    assert_eq!(doc.get_key(&LONG_WORDS).unwrap().len(), 0);
}

#[test]
fn a_filter_fails_loud_when_the_source_layer_is_absent() {
    let filter = FilterAnnotator::with_document(WORDS, LONG_WORDS, |_: &Document, _: &SpanItem| true).unwrap();
    let (doc, status) = run_step(&filter, Document::new("no words layer"));
    assert_eq!(status, Err(Error::Missing(WORDS.id())));
    assert_eq!(doc.status(LONG_WORDS.id()), Some(&Status::Failed(Error::Missing(WORDS.id()))));
}

#[test]
fn a_skipped_conditional_still_requires_the_delegates_layers() {
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = {
        let calls = calls.clone();
        move |_: &Document| {
            calls.fetch_add(1, Ordering::Relaxed);
            true
        }
    };
    let filter = FilterAnnotator::with_document(WORDS, LONG_WORDS, |_: &Document, _: &SpanItem| true).unwrap();
    let guarded = ConditionalAnnotator::when(counted, filter).unwrap();
    let (doc, status) = run_step(&guarded, Document::new("no words layer"));
    assert_eq!(calls.load(Ordering::Relaxed), 0, "the inputs are verified before the condition is tested");
    assert_eq!(status, Err(Error::Missing(WORDS.id())));
    assert_eq!(doc.status(LONG_WORDS.id()), Some(&Status::Failed(Error::Missing(WORDS.id()))));
}

#[test]
fn the_skip_path_still_collides_with_a_pre_existing_provided_layer() {
    let guarded = ConditionalAnnotator::when(|_: &Document| false, Bound::new(Producer)).unwrap();
    let (doc, status) = run_step(&guarded, produced("two words"));
    // Recording the skipped branch's layer refuses, like the Java form's
    // duplicate-layer rejection, and the layer that was there is untouched.
    assert_eq!(status, Err(Error::Duplicate(WORDS.id())));
    assert_eq!(doc.status(WORDS.id()), Some(&Status::Complete));
}

#[test]
fn a_filter_rejecting_every_annotation_provides_an_empty_target() {
    let filter = FilterAnnotator::with_document(WORDS, LONG_WORDS, |_: &Document, _: &SpanItem| false).unwrap();
    let (doc, status) = run_step(&filter, produced("a lengthy pair of words"));
    assert!(status.is_ok(), "{status:?}");
    assert!(doc.get_key(&LONG_WORDS).is_some(), "an all-rejecting filter must still provide its target");
    assert_eq!(doc.get_key(&LONG_WORDS).unwrap().len(), 0);
}

#[test]
fn the_condition_runs_a_delegate_with_requirements_when_its_layer_is_present() {
    let filter = long_enough(5);
    let guarded = ConditionalAnnotator::when(|_: &Document| true, filter).unwrap();
    let (doc, status) = run_step(&guarded, produced("a lengthy pair of words"));
    assert!(status.is_ok(), "{status:?}");
    assert!(doc.get_key(&LONG_WORDS).is_some(), "the delegate must have run");
    assert_eq!(words_of(&doc, &LONG_WORDS), ["lengthy", "words"]);
}

#[test]
fn the_condition_is_tested_exactly_once_with_the_document_it_is_deciding() {
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = Arc::new(Mutex::new(String::new()));
    let (count, record) = (calls.clone(), seen.clone());
    let conditional = ConditionalAnnotator::when(
        move |d: &Document| {
            count.fetch_add(1, Ordering::Relaxed);
            *record.lock().unwrap() = d.source().into();
            true
        },
        Bound::new(Producer),
    )
    .unwrap();
    let (doc, status) = run_step(&conditional, Document::new("two words"));
    assert!(status.is_ok(), "{status:?}");
    assert_eq!(calls.load(Ordering::Relaxed), 1, "the condition must be tested once");
    assert_eq!(*seen.lock().unwrap(), "two words");
    assert_eq!(doc.get_key(&WORDS).unwrap().len(), 2);
}

#[test]
fn the_filter_keeps_the_surviving_items_in_order() {
    let doc = produced("a lengthy pair of words");
    let source: Vec<Span> = doc.get_key(&WORDS).unwrap().spans.iter().map(|(_, s)| s).collect();
    let (doc, status) = run_step(&long_enough(5), doc);
    assert!(status.is_ok(), "{status:?}");
    // A columnar layer has no per-item object identity: a kept item is the
    // same span and text as its source item, in source order.
    let matches: Vec<Span> = doc.get_key(&LONG_WORDS).unwrap().spans.iter().map(|(_, s)| s).collect();
    let expected: Vec<Span> = source.iter().copied().filter(|s| s.len() >= 5).collect();
    assert_eq!(matches, expected);
}

#[test]
fn the_filter_can_inspect_the_original_document_text() {
    let mut doc = Document::new("Alice met bob");
    let mut spans = Spans::new(ViewId::SOURCE);
    for (start, end) in [(0, 5), (6, 9), (10, 13)] {
        spans.push(Span::new(start, end));
    }
    doc.add(&WORDS, Tokens { spans, sentence: None }, Provenance::by("test")).unwrap();
    let filter = FilterAnnotator::with_document(WORDS, LONG_WORDS, |d: &Document, t: &SpanItem| {
        d.source()[t.span.range()].chars().next().is_some_and(char::is_uppercase)
    })
    .unwrap();
    let (doc, status) = run_step(&filter, doc);
    assert!(status.is_ok(), "{status:?}");
    assert_eq!(words_of(&doc, &LONG_WORDS), ["Alice"]);
    assert_eq!(doc.get_key(&WORDS).unwrap().len(), 3);
}

#[test]
fn the_conditional_runs_the_selected_branch() {
    for condition in [true, false] {
        let long_words = long_enough(5);
        let short_words = FilterAnnotator::with_document(WORDS, LONG_WORDS, move |d: &Document, t: &SpanItem| {
            d.text(t.view)[t.span.range()].len() < 5
        })
        .unwrap();
        let guarded = ConditionalAnnotator::new(move |_: &Document| condition, long_words, short_words).unwrap();
        let (doc, status) = run_step(&guarded, produced("a lengthy pair of words"));
        assert!(status.is_ok(), "{status:?}");
        let expected: &[&str] = if condition { &["lengthy", "words"] } else { &["a", "pair", "of"] };
        assert_eq!(words_of(&doc, &LONG_WORDS), expected, "condition {condition}");
    }
}

#[test]
fn the_conditional_rejects_branches_with_different_outputs() {
    let filter = FilterAnnotator::with_document(WORDS, LONG_WORDS, |_: &Document, _: &SpanItem| true).unwrap();
    let e = construction_err(ConditionalAnnotator::new(|_: &Document| true, Bound::new(Producer), filter));
    assert_eq!(e, Error::Pipeline("turbo:conditional: branches must provide the same layers".into()));
}

#[test]
fn construction_rejects_contract_violations() {
    // target equal to source, in both predicate forms
    let e = construction_err(FilterAnnotator::new(WORDS, WORDS, |_: &SpanItem| true));
    assert_eq!(e, Error::Pipeline("turbo:filter: target must differ from source".into()));
    let e = construction_err(FilterAnnotator::with_document(WORDS, WORDS, |_: &Document, _: &SpanItem| true));
    assert_eq!(e, Error::Pipeline("turbo:filter: target must differ from source".into()));
    // a named instance is still the same target
    let e = construction_err(FilterAnnotator::new(LONG_WORDS, LONG_WORDS, |_: &SpanItem| true));
    assert_eq!(e, Error::Pipeline("turbo:filter: target must differ from source".into()));
}

#[test]
fn a_document_scoped_filter_selects_labels() {
    let mut labels = Strs::default();
    for label in ["news", "sports", "ads"] {
        labels.push(label);
    }
    let document = || {
        let mut doc = Document::new("Sports results");
        doc.add(&CATEGORIES, Labels { labels: labels.clone() }, Provenance::by("test")).unwrap();
        doc
    };
    let kept = |doc: &Document| -> Vec<String> {
        let selected = doc.get_key(&SELECTED).unwrap();
        (0..selected.len()).map(|i| selected.labels.get(i).into()).collect()
    };

    // The predicate that sees only an annotation.
    let filter = FilterAnnotator::new(CATEGORIES, SELECTED, |label: &String| label != "ads").unwrap();
    let (doc, status) = run_step(&filter, document());
    assert!(status.is_ok(), "{status:?}");
    assert_eq!(kept(&doc), ["news", "sports"]);
    assert_eq!(doc.get_key(&CATEGORIES).unwrap().len(), 3, "the source layer must be unchanged");

    // The predicate that can also inspect the document: the label is still
    // in the source layer and passes the value predicate.
    let filter = FilterAnnotator::with_document(CATEGORIES, SELECTED, |doc: &Document, label: &String| {
        doc.get_key(&CATEGORIES).is_some_and(|src| (0..src.len()).any(|i| src.labels.get(i) == label)) && label != "ads"
    })
    .unwrap();
    let (doc, status) = run_step(&filter, document());
    assert!(status.is_ok(), "{status:?}");
    assert_eq!(kept(&doc), ["news", "sports"]);
    assert_eq!(doc.get_key(&CATEGORIES).unwrap().len(), 3);
}

#[test]
fn filtering_a_span_layer_preserves_discontinuous_fragments() {
    const MENTION: Key<Mentions> = Key::DEFAULT;
    const KEPT: Key<Mentions> = Key::named("kept");
    const WIDE: Key<Mentions> = Key::named("wide");
    let mut doc = Document::new("aa bb cc dd ee");
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 2));
    spans.push_fragments(&[Span::new(3, 5), Span::new(9, 11)]);
    spans.push(Span::new(12, 14));
    doc.add(&MENTION, Mentions { spans }, Provenance::by("test")).unwrap();

    // Keep every item: the fragment table survives, one fragment for the
    // continuous items, two for the discontinuous one.
    let filter = FilterAnnotator::new(MENTION, KEPT, |_: &SpanItem| true).unwrap();
    let (doc, status) = run_step(&filter, doc);
    assert!(status.is_ok(), "{status:?}");
    let kept = doc.get_key(&KEPT).unwrap();
    assert_eq!(kept.spans.len(), 3);
    assert!(kept.spans.fragment_columns().is_some());
    assert!(kept.spans.is_discontinuous(Ix::new(1)));
    assert_eq!(kept.spans.fragments(Ix::new(1)).collect::<Vec<_>>(), [Span::new(3, 5), Span::new(9, 11)]);

    // Keep only the wide item: its fragments are preserved, not flattened
    // into the bounding envelope.
    let filter = FilterAnnotator::new(MENTION, WIDE, |item: &SpanItem| item.span.len() > 2).unwrap();
    let (doc, status) = run_step(&filter, doc);
    assert!(status.is_ok(), "{status:?}");
    let kept = doc.get_key(&WIDE).unwrap();
    assert_eq!(kept.spans.len(), 1);
    assert!(kept.spans.is_discontinuous(Ix::new(0)));
    assert_eq!(kept.spans.get(Ix::new(0)), Span::new(3, 11));
    assert_eq!(kept.spans.fragments(Ix::new(0)).collect::<Vec<_>>(), [Span::new(3, 5), Span::new(9, 11)]);
}
