use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::analyzer::Analyzer;
use crate::annotate::{Annotator, Bound, Refs, Step};
use crate::doc::{Document, LayerStatus, Status, View};
use crate::error::Error;
use crate::layer::{Anchor, Key, Layer, Provenance};
use crate::predicate::{ConditionalAnnotator, FilterAnnotator, FilterableLayer, SpanFilter};
use crate::span::{Ix, Overlap, Span, Spans, TextSpans};
use crate::text::ViewId;

#[derive(Clone, Debug)]
struct Words {
    spans: Spans<Words>,
    values: Vec<String>,
}

impl Layer for Words {
    const ID: &'static str = "test:words";
    const OVERLAP: Overlap = Overlap::Any;

    fn len(&self) -> usize {
        self.spans.len()
    }

    fn text_spans(&self) -> Option<TextSpans<'_>> {
        Some(self.spans.text_spans())
    }

    fn check(&self) -> Result<(), String> {
        if self.values.len() != self.spans.len() {
            return Err("a value for every word span".into());
        }
        self.spans.check(Self::OVERLAP)
    }
}

impl FilterableLayer for Words {
    fn retain_indices(&self, kept: &[Ix<Self>]) -> Self {
        Words {
            spans: self.spans.retain_indices(kept),
            values: kept.iter().map(|&ix| self.values[ix.get()].clone()).collect(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DocCategories {
    items: Vec<String>,
}

impl Layer for DocCategories {
    const ID: &'static str = "test:doc-categories";

    fn anchor() -> Anchor {
        Anchor::Document
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

impl FilterableLayer for DocCategories {
    fn retain_indices(&self, kept: &[Ix<Self>]) -> Self {
        DocCategories { items: kept.iter().map(|&ix| self.items[ix.get()].clone()).collect() }
    }
}

const WORDS: Key<Words> = Key::DEFAULT;
const LONG_WORDS: Key<Words> = Key::named("long");
const OTHER_WORDS: Key<Words> = Key::named("other");

struct WordProducer;

impl Annotator for WordProducer {
    const ID: &'static str = "test:word-producer";
    type Reads = (View,);
    type Writes = (Words,);

    fn annotate(&self, (view,): Refs<'_, Self>) -> Result<(Words,), Error> {
        let text = view.text();
        let mut spans = Spans::new(view.id());
        let mut values = Vec::new();
        let mut at = 0usize;
        for part in text.split(' ') {
            if !part.is_empty() {
                spans.push(Span::new(at as u32, (at + part.len()) as u32));
                values.push(part.to_string());
            }
            at += part.len() + 1;
        }
        Ok((Words { spans, values },))
    }
}

struct TwoLayerPanicker;

impl Annotator for TwoLayerPanicker {
    const ID: &'static str = "test:two-layers";
    type Reads = (View,);
    type Writes = (Words, DocCategories);

    fn annotate(&self, _: Refs<'_, Self>) -> Result<(Words, DocCategories), Error> {
        panic!("the delegate must not run when the condition fails");
    }
}

#[test]
fn filter_rejects_equal_source_and_target_and_filters_survivors() {
    let err = FilterAnnotator::new(WORDS, WORDS, |_, _| true).err().unwrap();
    assert_eq!(err, Error::Pipeline("target must differ from source".into()));
    let err_doc = FilterAnnotator::with_document(LONG_WORDS, LONG_WORDS, |_, _, _| true).err().unwrap();
    assert_eq!(err_doc, Error::Pipeline("target must differ from source".into()));

    let mut doc = Document::new("a lengthy pair of words");
    Bound::new(WordProducer).compute_and_store_one(&mut doc).unwrap();

    let filter = FilterAnnotator::new(WORDS, LONG_WORDS, |w, ix| w.values[ix.get()].len() > 4).unwrap();
    assert_eq!(filter.reads(), vec![WORDS.id()]);
    assert_eq!(filter.writes(), vec![LONG_WORDS.id()]);
    filter.annotate_doc(&mut doc).unwrap();

    assert_eq!(doc.get_key(&WORDS).unwrap().len(), 5, "source layer must be unchanged");
    assert_eq!(doc.get_key(&LONG_WORDS).unwrap().values, vec!["lengthy".to_string(), "words".to_string()]);

    // Empty source layer produces an empty target layer.
    let mut empty_doc = Document::new("");
    Bound::new(WordProducer).compute_and_store_one(&mut empty_doc).unwrap();
    FilterAnnotator::new(WORDS, LONG_WORDS, |_, _| true).unwrap().annotate_doc(&mut empty_doc).unwrap();
    assert_eq!(empty_doc.get_key(&LONG_WORDS).unwrap().len(), 0);
    assert_eq!(empty_doc.status(LONG_WORDS.id()), Some(&Status::Complete));

    // All-rejecting filter produces an empty target layer.
    let mut reject_doc = Document::new("a lengthy pair of words");
    Bound::new(WordProducer).compute_and_store_one(&mut reject_doc).unwrap();
    FilterAnnotator::new(WORDS, LONG_WORDS, |_, _| false).unwrap().annotate_doc(&mut reject_doc).unwrap();
    assert_eq!(reject_doc.get_key(&LONG_WORDS).unwrap().len(), 0);
    assert_eq!(reject_doc.status(LONG_WORDS.id()), Some(&Status::Complete));

    // Missing source layer fails with Error::Missing.
    let mut missing_doc = Document::new("no words layer");
    let missing_err = FilterAnnotator::new(WORDS, LONG_WORDS, |_, _| true).unwrap().annotate_doc(&mut missing_doc);
    assert_eq!(missing_err, Err(Error::Missing(WORDS.id())));
}

#[test]
fn filter_preserves_discontinuous_fragments_and_document_scoped_items() {
    let mut doc = Document::new("pick the rug right up now");
    let mut spans = Spans::new(ViewId::SOURCE);
    // Item 0: discontinuous particle verb "pick" (0..4) + "up" (19..21)
    spans.push_fragments(&[Span::new(0, 4), Span::new(19, 21)]);
    // Item 1: continuous noun phrase "the rug" (5..12)
    spans.push(Span::new(5, 12));
    // Item 2: discontinuous "the" (5..8) + "now" (22..25)
    spans.push_fragments(&[Span::new(5, 8), Span::new(22, 25)]);
    doc.add(
        &WORDS,
        Words { spans, values: vec!["pick_up".into(), "the_rug".into(), "the_now".into()] },
        Provenance::by("test"),
    )
    .unwrap();

    let filter = SpanFilter::new(WORDS, LONG_WORDS, |w, ix| w.spans.is_discontinuous(ix)).unwrap();
    filter.annotate_doc(&mut doc).unwrap();

    let kept = doc.get_key(&LONG_WORDS).unwrap();
    assert_eq!(kept.len(), 2);
    assert_eq!(kept.values, vec!["pick_up".to_string(), "the_now".to_string()]);
    assert!(kept.spans.has_discontinuous());
    assert!(kept.spans.is_discontinuous(Ix::new(0)));
    assert_eq!(kept.spans.fragments(Ix::new(0)).collect::<Vec<_>>(), vec![Span::new(0, 4), Span::new(19, 21)]);
    assert_eq!(kept.spans.fragments(Ix::new(1)).collect::<Vec<_>>(), vec![Span::new(5, 8), Span::new(22, 25)]);

    // Document-scoped filter (Anchor::Document, no spans).
    let cats: Key<DocCategories> = Key::DEFAULT;
    let selected: Key<DocCategories> = Key::named("selected");
    doc.add(&cats, DocCategories { items: vec!["news".into(), "sports".into(), "ads".into()] }, Provenance::by("test"))
        .unwrap();
    FilterAnnotator::with_document(cats, selected, move |d, layer, ix| {
        d.get_key(&cats).is_some() && layer.items[ix.get()] != "ads"
    })
    .unwrap()
    .annotate_doc(&mut doc)
    .unwrap();
    assert_eq!(doc.get_key(&selected).unwrap().items, vec!["news".to_string(), "sports".to_string()]);
}

#[test]
fn document_aware_filter_inspects_original_text_and_unicode_capitalization() {
    let cases = [
        ("", 0u32, 0u32, false),
        ("Alice", 0, 0, false),
        ("Alice", 5, 5, false),
        ("a A", 2, 2, false),
        ("Alice", 0, 5, true),
        ("alice", 0, 5, false),
        ("xAlice", 1, 6, true),
        ("𐐀lice", 0, 8, true),
        ("𐐨lice", 0, 8, false),
        ("𝐀lice", 0, 8, true),
        ("𐌰", 0, 4, false),
        ("Αλφα", 0, 8, true),
        ("ǅuro", 0, 5, false),
        ("\u{0301}Alice", 0, 7, false),
        ("𐐀", 0, 4, true),
        ("x𐐀", 1, 5, true),
        ("😀Alice", 4, 9, true),
        ("😀Alice", 0, 4, false),
    ];

    let cap_filter = FilterAnnotator::with_document(WORDS, LONG_WORDS, |doc, words, ix| {
        let span = words.spans.get(ix);
        if span.is_empty() {
            return false;
        }
        let slice = &doc.source()[span.range()];
        slice.chars().next().is_some_and(|ch| ch.is_uppercase())
    })
    .unwrap();

    for (text, start, end, expected) in cases {
        let mut doc = Document::new(text);
        let mut spans = Spans::new(ViewId::SOURCE);
        spans.push(Span::new(start, end));
        doc.add(&WORDS, Words { spans, values: vec!["normalized".into()] }, Provenance::by("test")).unwrap();
        cap_filter.annotate_doc(&mut doc).unwrap();
        let kept = doc.get_key(&LONG_WORDS).unwrap();
        assert_eq!(kept.len(), usize::from(expected), "failed for text={text:?} {start}..{end}");
    }
}

#[test]
fn conditional_annotator_branches_skips_and_checks_requirements_and_provenance() {
    // 1. Condition runs delegate when true, and skips all output layers when false.
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_ref = Arc::clone(&calls);
    let guarded = ConditionalAnnotator::from_annotator(
        "len>3",
        move |d| {
            calls_ref.fetch_add(1, Ordering::Relaxed);
            d.source().len() > 3
        },
        WordProducer,
    );
    let mut doc_true = Document::new("two words");
    guarded.annotate_doc(&mut doc_true).unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 1, "condition must be tested exactly once");
    assert_eq!(doc_true.get_key(&WORDS).unwrap().len(), 2);
    assert_eq!(doc_true.provenance(WORDS.id()).unwrap().config, guarded.config_hash());

    let mut doc_false = Document::new("hi");
    guarded.annotate_doc(&mut doc_false).unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 2);
    assert!(doc_false.get_key(&WORDS).is_none());
    assert!(matches!(doc_false.status(WORDS.id()), Some(LayerStatus::Skipped(_))));
    assert_eq!(doc_false.provenance(WORDS.id()).unwrap().config, guarded.config_hash());

    // 2. Multi-output step skipped when false without invoking delegate.
    let two_guarded = ConditionalAnnotator::from_annotator("never", |_| false, TwoLayerPanicker);
    let mut doc_two = Document::new("two words");
    two_guarded.annotate_doc(&mut doc_two).unwrap();
    assert!(matches!(doc_two.status(WORDS.id()), Some(Status::Skipped(_))));
    assert!(matches!(doc_two.status(Key::<DocCategories>::DEFAULT.id()), Some(Status::Skipped(_))));

    // 3. Skipped conditional still requires delegate input layers before testing condition.
    let filter_step = FilterAnnotator::new(WORDS, LONG_WORDS, |_, _| true).unwrap();
    let req_guarded = ConditionalAnnotator::new("never", |_| false, filter_step);
    let mut no_words = Document::new("no words layer");
    assert_eq!(req_guarded.annotate_doc(&mut no_words), Err(Error::Missing(WORDS.id())));

    // 4. Skip path collides with pre-existing provided layer.
    let mut dup_doc = Document::new("two words");
    Bound::new(WordProducer).compute_and_store_one(&mut dup_doc).unwrap();
    let skip_dup = ConditionalAnnotator::from_annotator("never", |_| false, WordProducer);
    assert_eq!(skip_dup.annotate_doc(&mut dup_doc), Err(Error::Duplicate(WORDS.id())));

    // 5. Rejects branches with different outputs.
    let mismatch = ConditionalAnnotator::if_else(
        "cond",
        |_| true,
        Bound::new(WordProducer),
        FilterAnnotator::new(WORDS, LONG_WORDS, |_, _| true).unwrap(),
    );
    assert_eq!(mismatch.err().unwrap(), Error::Pipeline("branches must provide the same layers".into()));

    // 6. Declares requirements from both branches and checks union before testing condition.
    let branch_a = FilterAnnotator::new(WORDS, LONG_WORDS, |_, _| true).unwrap().named("a");
    let branch_b = FilterAnnotator::new(OTHER_WORDS, LONG_WORDS, |_, _| true).unwrap().named("b");
    let union_cond = ConditionalAnnotator::if_else("pick", |_| true, branch_a, branch_b).unwrap();
    assert_eq!(union_cond.reads(), vec![WORDS.id(), OTHER_WORDS.id()]);
    let mut only_words = Document::new("two words");
    Bound::new(WordProducer).compute_and_store_one(&mut only_words).unwrap();
    assert_eq!(union_cond.annotate_doc(&mut only_words), Err(Error::Missing(OTHER_WORDS.id())));

    // 7. Deterministic config_hash changes when predicate name or branch changes.
    let f_long =
        FilterAnnotator::new(WORDS, LONG_WORDS, |w, ix| w.values[ix.get()].len() >= 5).unwrap().named("len>=5");
    let f_short = FilterAnnotator::new(WORDS, LONG_WORDS, |w, ix| w.values[ix.get()].len() < 5).unwrap().named("len<5");
    let c1 = ConditionalAnnotator::if_else("doc_len>=20", |d| d.source().len() >= 20, f_long, f_short).unwrap();
    let c2 = ConditionalAnnotator::if_else(
        "doc_len>=20",
        |d| d.source().len() >= 20,
        FilterAnnotator::new(WORDS, LONG_WORDS, |w, ix| w.values[ix.get()].len() >= 5).unwrap().named("len>=5"),
        FilterAnnotator::new(WORDS, LONG_WORDS, |w, ix| w.values[ix.get()].len() < 5).unwrap().named("len<5"),
    )
    .unwrap();
    let c3 = ConditionalAnnotator::if_else(
        "other_pred",
        |d| d.source().len() >= 20,
        FilterAnnotator::new(WORDS, LONG_WORDS, |w, ix| w.values[ix.get()].len() >= 5).unwrap().named("len>=5"),
        FilterAnnotator::new(WORDS, LONG_WORDS, |w, ix| w.values[ix.get()].len() < 5).unwrap().named("len<5"),
    )
    .unwrap();
    assert_eq!(c1.config_hash(), c2.config_hash());
    assert_ne!(c1.config_hash(), c3.config_hash());
}

#[test]
fn concurrent_and_batched_analyzer_pipelines_with_conditional_and_filter() {
    for document_aware in [false, true] {
        for explicit_false in [false, true] {
            let cond_calls = Arc::new(AtomicUsize::new(0));
            let cond_calls_ref = Arc::clone(&cond_calls);
            let make_filter = |long_words: bool| {
                let tag = if long_words { "long" } else { "short" };
                if document_aware {
                    FilterAnnotator::with_document(WORDS, LONG_WORDS, move |doc, w, ix| {
                        let span = w.spans.get(ix);
                        doc.source()[span.range()] == w.values[ix.get()]
                            && if long_words { w.values[ix.get()].len() >= 4 } else { w.values[ix.get()].len() < 4 }
                    })
                    .unwrap()
                    .named(tag)
                } else {
                    FilterAnnotator::new(WORDS, LONG_WORDS, move |w, ix| {
                        if long_words { w.values[ix.get()].len() >= 4 } else { w.values[ix.get()].len() < 4 }
                    })
                    .unwrap()
                    .named(tag)
                }
            };

            let conditional = if explicit_false {
                ConditionalAnnotator::if_else(
                    "starts_with_L",
                    move |d| {
                        cond_calls_ref.fetch_add(1, Ordering::Relaxed);
                        d.source().starts_with('L')
                    },
                    make_filter(true),
                    make_filter(false),
                )
                .unwrap()
            } else {
                ConditionalAnnotator::new(
                    "starts_with_L",
                    move |d| {
                        cond_calls_ref.fetch_add(1, Ordering::Relaxed);
                        d.source().starts_with('L')
                    },
                    make_filter(true),
                )
            };

            let analyzer = Analyzer::builder().add(WordProducer).step(conditional).build().unwrap();
            let mut docs: Vec<Document> = (0..60)
                .map(|i| {
                    let prefix = if i % 2 == 0 { "Long lengthy phrase " } else { "Short a " };
                    Document::new(format!("{prefix}{i}"))
                })
                .collect();

            let status = analyzer.run(&mut docs);
            assert!(status.iter().all(|s| s.is_ok()), "{status:?}");
            assert_eq!(cond_calls.load(Ordering::Relaxed), 60);

            for (i, d) in docs.iter().enumerate() {
                if i % 2 == 0 {
                    assert_eq!(
                        d.get_key(&LONG_WORDS).unwrap().values,
                        vec!["Long".to_string(), "lengthy".to_string(), "phrase".to_string()]
                    );
                } else if explicit_false {
                    assert_eq!(d.get_key(&LONG_WORDS).unwrap().values, vec!["a".to_string(), i.to_string()]);
                } else {
                    assert!(d.get_key(&LONG_WORDS).is_none());
                    assert!(matches!(d.status(LONG_WORDS.id()), Some(Status::Skipped(_))));
                }
            }
        }
    }
}

trait StepSingleExt: Step {
    fn compute_and_store_one(&self, doc: &mut Document) -> Result<(), Error> {
        let out = self.compute(std::slice::from_ref(doc));
        let mut status = [Ok(())];
        self.store(out, std::slice::from_mut(doc), &mut status);
        status.into_iter().next().unwrap()
    }
}

impl<S: Step> StepSingleExt for S {}
