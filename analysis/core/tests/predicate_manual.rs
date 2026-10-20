//! The manual's predicate-annotator examples over the chapter's
//! token-length fixture, ported from OpenNLP's
//! `PredicateManualExampleTest` (opennlp.tools.document). Offsets are
//! UTF-8 byte offsets, this library's unit; the Java cases that cut into a
//! UTF-16 surrogate pair cannot be written as Rust spans and are left out.

use turbo_analysis::annotate::Step;
use turbo_analysis::{
    Annotator, Bound, ConditionalAnnotator, Document, Error, FilterAnnotator, Filterable, Ix, Key, Layer, Overlap,
    Provenance, Refs, Span, SpanItem, Spans, Status, TextSpans, View, ViewId,
};
use turbo_analysis_layers::Strs;

/// The chapter's custom layer: one integer per space-delimited token.
struct Lengths {
    spans: Spans<Lengths>,
    value: Vec<i32>,
}

impl Layer for Lengths {
    const ID: &'static str = "test:token-lengths";

    fn len(&self) -> usize {
        self.spans.len()
    }

    fn text_spans(&self) -> Option<TextSpans<'_>> {
        Some(self.spans.text_spans())
    }
}

impl Filterable for Lengths {
    type Item = i32;

    fn item(&self, i: usize) -> i32 {
        self.value[i]
    }

    fn filter(&self, keep: &mut dyn FnMut(&i32) -> bool) -> Lengths {
        let mut spans = Spans::with_capacity(self.spans.view(), self.len());
        let mut value = Vec::new();
        for i in 0..self.len() {
            if keep(&self.item(i)) {
                spans.push(self.spans.get(Ix::new(i as u32)));
                value.push(self.value[i]);
            }
        }
        Lengths { spans, value }
    }
}

const TOKEN_LENGTHS: Key<Lengths> = Key::DEFAULT;
const SELECTED_LENGTHS: Key<Lengths> = Key::named("selected");

/// The chapter's producer of the token-length layer, like the Java
/// fixture's `LENGTHS`.
struct LengthProducer;

impl Annotator for LengthProducer {
    const ID: &'static str = "test:lengths";
    type Reads = (View,);
    type Writes = (Lengths,);

    fn annotate(&self, (view,): Refs<'_, Self>) -> Result<(Lengths,), Error> {
        let text = view.text();
        let mut spans = Spans::new(view.id());
        let mut value = Vec::new();
        let mut at = 0;
        for word in text.split(' ') {
            if !word.is_empty() {
                spans.push(Span::new(at as u32, (at + word.len()) as u32));
                value.push(word.len() as i32);
            }
            at += word.len() + 1;
        }
        Ok((Lengths { spans, value },))
    }
}

/// Lowercased words mapped to their original surface text.
struct NormWords {
    spans: Spans<NormWords>,
    word: Strs,
}

impl Layer for NormWords {
    const ID: &'static str = "test:normalized-words";
    const OVERLAP: Overlap = Overlap::Nesting;

    fn len(&self) -> usize {
        self.spans.len()
    }

    fn text_spans(&self) -> Option<TextSpans<'_>> {
        Some(self.spans.text_spans())
    }
}

impl Filterable for NormWords {
    type Item = SpanItem;

    fn item(&self, i: usize) -> SpanItem {
        SpanItem { view: self.spans.view(), span: self.spans.get(Ix::new(i as u32)) }
    }

    fn filter(&self, keep: &mut dyn FnMut(&SpanItem) -> bool) -> NormWords {
        let mut kept = Vec::new();
        let mut word = Strs::default();
        for i in 0..self.len() {
            if keep(&self.item(i)) {
                kept.push(i);
                word.push(self.word.get(i));
            }
        }
        NormWords { spans: self.spans.subset(&kept), word }
    }
}

const NORMALIZED_WORDS: Key<NormWords> = Key::named("normalized");
const CAPITALIZED_WORDS: Key<NormWords> = Key::named("capitalized");

/// The chapter's example text, with token lengths 3, 3, 6, 2, 5.
const TEXT: &str = "The dog barks. It naps.";

/// Runs one step over one document, as `Analyzer::run` would.
fn run_step(step: &dyn Step, doc: Document) -> (Document, Result<(), Error>) {
    let mut docs = vec![doc];
    let mut status = vec![Ok(())];
    let out = step.compute(&docs);
    step.store(out, &mut docs, &mut status);
    (docs.pop().unwrap(), status.pop().unwrap())
}

/// The document `LengthProducer` makes for `text`.
fn with_lengths(text: &str) -> Document {
    let (doc, status) = run_step(&Bound::new(LengthProducer), Document::new(text));
    assert!(status.is_ok(), "{status:?}");
    doc
}

fn values(doc: &Document, key: &Key<Lengths>) -> Vec<i32> {
    doc.get_key(key).unwrap().value.clone()
}

/// The manual's filter example: `SELECTED_LENGTHS` holds 6 and 5, the
/// source still five.
#[test]
fn the_filter_example_states_the_survivors() {
    let filter = FilterAnnotator::new(TOKEN_LENGTHS, SELECTED_LENGTHS, |v: &i32| *v >= 5).unwrap();
    let (doc, status) = run_step(&filter, with_lengths(TEXT));
    assert!(status.is_ok(), "{status:?}");
    assert_eq!(values(&doc, &SELECTED_LENGTHS), [6, 5]);
    assert_eq!(doc.get_key(&TOKEN_LENGTHS).unwrap().len(), 5);
}

/// A document without the source layer is refused with a message naming
/// the layer.
#[test]
fn the_missing_layer_example_states_the_exact_message() {
    let filter = FilterAnnotator::new(TOKEN_LENGTHS, SELECTED_LENGTHS, |v: &i32| *v >= 5).unwrap();
    let (_, status) = run_step(&filter, Document::new(TEXT));
    assert_eq!(status.unwrap_err().to_string(), "the document has no test:token-lengths layer");
}

/// The conditional example: a long document passes with the layer
/// produced, a short one passes with the layer skipped.
#[test]
fn the_conditional_example_keeps_the_contract_on_short_documents() {
    let guarded =
        ConditionalAnnotator::when(|d: &Document| d.source().len() >= 20, Bound::new(LengthProducer)).unwrap();
    let (ran, status) = run_step(&guarded, Document::new(TEXT));
    assert!(status.is_ok(), "{status:?}");
    assert_eq!(ran.get_key(&TOKEN_LENGTHS).unwrap().len(), 5);

    let (skipped, status) = run_step(&guarded, Document::new("It naps."));
    assert!(status.is_ok(), "{status:?}");
    assert_eq!(skipped.status(TOKEN_LENGTHS.id()), Some(&Status::Skipped("condition is false".into())));
}

/// The explicit branches select long or short token lengths.
#[test]
fn the_conditional_else_example_selects_one_branch() {
    let selected = ConditionalAnnotator::new(
        |d: &Document| d.source().len() >= 20,
        FilterAnnotator::new(TOKEN_LENGTHS, SELECTED_LENGTHS, |v: &i32| *v >= 5).unwrap(),
        FilterAnnotator::new(TOKEN_LENGTHS, SELECTED_LENGTHS, |v: &i32| *v < 5).unwrap(),
    )
    .unwrap();

    let (long_text, status) = run_step(&selected, with_lengths(TEXT));
    assert!(status.is_ok(), "{status:?}");
    assert_eq!(values(&long_text, &SELECTED_LENGTHS), [6, 5]);

    let (short_text, status) = run_step(&selected, with_lengths("It naps."));
    assert!(status.is_ok(), "{status:?}");
    assert_eq!(values(&short_text, &SELECTED_LENGTHS), [2]);
}

/// The document-aware filter compares the normalized word with its
/// original surface text.
#[test]
fn the_document_aware_filter_example_reads_the_original_text() {
    let mut doc = Document::new("Alice met bob");
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut word = Strs::default();
    for ((start, end), w) in [((0, 5), "alice"), ((6, 9), "met"), ((10, 13), "bob")] {
        spans.push(Span::new(start, end));
        word.push(w);
    }
    doc.add(&NORMALIZED_WORDS, NormWords { spans, word }, Provenance::by("test")).unwrap();

    let (doc, status) = run_step(&capitalized(), doc);
    assert!(status.is_ok(), "{status:?}");
    let kept = doc.get_key(&CAPITALIZED_WORDS).unwrap();
    let kept: Vec<&str> = (0..kept.len()).map(|i| kept.word.get(i)).collect();
    assert_eq!(kept, ["alice"]);
}

/// The manual filter selects by the complete initial code point inside an
/// annotation. `selected` is whether the initial code point is uppercase.
#[test]
fn the_capitalization_example_handles_utf8_offsets() {
    for (text, start, end, selected) in [
        ("", 0, 0, false),
        ("Alice", 0, 0, false),
        ("Alice", 5, 5, false),
        ("a A", 2, 2, false),
        ("Alice", 0, 5, true),
        ("alice", 0, 5, false),
        ("xAlice", 1, 6, true),
        ("\u{10400}lice", 0, 8, true),
        ("\u{10428}lice", 0, 8, false),
        ("\u{1D400}lice", 0, 8, true),
        ("\u{10330}", 0, 4, false),
        ("Αλφα", 0, 8, true),
        ("ǅuro", 0, 5, false),
        ("\u{0301}Alice", 0, 7, false),
        ("\u{10400}", 0, 4, true),
        ("x\u{10400}", 1, 5, true),
        ("😀Alice", 4, 9, true),
        ("😀Alice", 0, 4, false),
    ] {
        let mut doc = Document::new(text);
        let mut spans = Spans::new(ViewId::SOURCE);
        spans.push(Span::new(start, end));
        let mut word = Strs::default();
        word.push("normalized");
        doc.add(&NORMALIZED_WORDS, NormWords { spans, word }, Provenance::by("test")).unwrap();

        let (doc, status) = run_step(&capitalized(), doc);
        assert!(status.is_ok(), "{status:?}");
        let kept = doc.get_key(&CAPITALIZED_WORDS).unwrap();
        let source = doc.get_key(&NORMALIZED_WORDS).unwrap();
        assert_eq!(kept.len() == 1, selected, "{text:?} {start}..{end}");
        assert_eq!(source.len(), 1, "the source layer must be unchanged");
        assert_eq!(doc.source(), text);
        if selected {
            assert_eq!(kept.spans.get(Ix::new(0)), Span::new(start, end));
        }
    }
}

/// The document-aware capitalization filter shown in the manual: the
/// initial code point must be complete inside the annotation and
/// uppercase.
fn capitalized() -> FilterAnnotator<NormWords> {
    FilterAnnotator::with_document(NORMALIZED_WORDS, CAPITALIZED_WORDS, |source: &Document, word: &SpanItem| {
        let covered = &source.text(word.view)[word.span.range()];
        let Some(initial) = covered.chars().next() else { return false };
        u32::try_from(initial.len_utf8()).unwrap() <= word.span.len() && initial.is_uppercase()
    })
    .unwrap()
}
