//! Pipeline-level ports from OpenNLP's predicate-annotator tests
//! (`opennlp.tools.document`): concurrent pipelines over an analyzer, and
//! the provenance and `config_hash` contract of `ConditionalAnnotator`.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use turbo_analysis::annotate::Step;
use turbo_analysis::{
    Analyzer, Annotator, Bound, ConditionalAnnotator, Document, Error, FilterAnnotator, Key, Layer, Refs, Span,
    SpanItem, Spans, Status, View,
};
use turbo_analysis_layers::Tokens;

const WORDS: Key<Tokens> = Key::DEFAULT;
const LONG_WORDS: Key<Tokens> = Key::named("long");
const OTHER_WORDS: Key<Tokens> = Key::named("other");

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

/// The document `Producer` makes for `text`.
fn produced(text: &str) -> Document {
    let mut docs = vec![Document::new(text)];
    let mut status = vec![Ok(())];
    let step = Bound::new(Producer);
    let out = step.compute(&docs);
    step.store(out, &mut docs, &mut status);
    assert!(status[0].is_ok(), "{:?}", status[0]);
    docs.pop().unwrap()
}

/// The words a tokens layer covers, as text.
fn words_of(doc: &Document, key: &Key<Tokens>) -> Vec<String> {
    let tokens = doc.get_key(key).unwrap();
    let view = tokens.spans.view();
    tokens.spans.iter().map(|(_, s)| doc.text(view)[s.range()].to_string()).collect()
}

fn aware_pred(long: bool) -> impl Fn(&Document, &SpanItem) -> bool {
    move |d: &Document, t: &SpanItem| {
        let covered = &d.text(t.view)[t.span.range()];
        !covered.is_empty() && (covered.len() >= 4) == long
    }
}

fn plain_pred(long: bool) -> impl Fn(&SpanItem) -> bool {
    move |t: &SpanItem| (t.span.len() >= 4) == long
}

fn word_filter(aware: bool, long: bool) -> FilterAnnotator<Tokens> {
    if aware {
        FilterAnnotator::with_document(WORDS, LONG_WORDS, aware_pred(long)).unwrap()
    } else {
        FilterAnnotator::new(WORDS, LONG_WORDS, plain_pred(long)).unwrap()
    }
}

fn document_text(i: usize) -> String {
    format!("{}{i}", if i.is_multiple_of(2) { "Long lengthy phrase " } else { "Short a " })
}

#[test]
fn concurrent_pipelines_retain_each_documents_annotations() {
    for aware in [false, true] {
        for explicit in [false, true] {
            let condition_calls = Arc::new(AtomicUsize::new(0));
            let counted = {
                let calls = condition_calls.clone();
                move |d: &Document| {
                    calls.fetch_add(1, Ordering::Relaxed);
                    d.source().starts_with('L')
                }
            };
            let conditional = {
                let long_words = word_filter(aware, true);
                if explicit {
                    ConditionalAnnotator::new(counted, long_words, word_filter(aware, false)).unwrap()
                } else {
                    ConditionalAnnotator::when(counted, long_words).unwrap()
                }
            };
            let analyzer = Analyzer::builder().add(Producer).step(conditional).build().unwrap();
            let task_count = 200;
            let mut batches: Vec<Vec<Document>> = (0..4).map(|_| Vec::new()).collect();
            for i in 0..task_count {
                batches[i % 4].push(Document::new(document_text(i)));
            }
            std::thread::scope(|s| {
                let analyzer = &analyzer;
                let mut handles = Vec::new();
                for batch in &mut batches {
                    handles.push(s.spawn(move || analyzer.run(batch)));
                }
                for h in handles {
                    let status = h.join().expect("a pipeline task panicked");
                    assert!(status.iter().all(|s| s.is_ok()), "{status:?}");
                }
            });
            for i in 0..task_count {
                let doc = &batches[i % 4][i / 4];
                let long_document = i % 2 == 0;
                assert_eq!(doc.source(), document_text(i));
                assert!(doc.has(LONG_WORDS.id()), "doc {i}");
                let skipped = Status::Skipped("condition is false".into());
                if !explicit && !long_document {
                    // No false branch: the provided layers are recorded
                    // skipped, the typed pipeline's empty layers.
                    assert_eq!(doc.status(LONG_WORDS.id()), Some(&skipped), "doc {i}");
                    continue;
                }
                assert_eq!(doc.status(LONG_WORDS.id()), Some(&Status::Complete), "doc {i}");
                let got = words_of(doc, &LONG_WORDS);
                if long_document {
                    assert_eq!(got, vec!["Long".to_string(), "lengthy".to_string(), "phrase".to_string()], "doc {i}");
                } else {
                    assert_eq!(got, vec!["a".to_string(), i.to_string()], "doc {i}");
                }
                let words = doc.get_key(&WORDS).unwrap().len();
                assert_eq!(words, if long_document { 4 } else { 3 });
                for (_, span) in doc.get_key(&LONG_WORDS).unwrap().spans.iter() {
                    let covered = &doc.source()[span.range()];
                    assert!(!covered.is_empty());
                    assert!(doc.get_key(&WORDS).unwrap().spans.iter().any(|(_, s)| s == span));
                }
            }
            assert_eq!(condition_calls.load(Ordering::Relaxed), task_count);
        }
    }
}

#[test]
fn the_conditional_requires_the_union_and_preserves_provenance_and_config_hash() {
    // The union of the branches' inputs, each required.
    let words = FilterAnnotator::with_document(WORDS, LONG_WORDS, |_: &Document, _: &SpanItem| true).unwrap();
    let other = FilterAnnotator::with_document(OTHER_WORDS, LONG_WORDS, |_: &Document, _: &SpanItem| true).unwrap();
    let guarded = ConditionalAnnotator::new(|_: &Document| true, words, other).unwrap();
    assert_eq!(guarded.reads(), vec![WORDS.id(), OTHER_WORDS.id()]);
    assert_eq!(guarded.required(), vec![true, true]);
    // A document lacking one branch's input is refused before the
    // condition is tested, whichever branch the condition would select.
    let step = &guarded as &dyn Step;
    let mut docs = vec![produced("two words")];
    let mut status = vec![Ok(())];
    let out = step.compute(&docs);
    step.store(out, &mut docs, &mut status);
    assert_eq!(status[0], Err(Error::Missing(OTHER_WORDS.id())));
    assert_eq!(docs[0].status(LONG_WORDS.id()), Some(&Status::Failed(Error::Missing(OTHER_WORDS.id()))));

    // The layer a branch wrote says the branch made it; provenance is
    // deterministic across runs.
    let run = || {
        let filter = FilterAnnotator::with_document(WORDS, LONG_WORDS, |_: &Document, _: &SpanItem| true).unwrap();
        let want = <FilterAnnotator<Tokens> as Step>::config_hash(&filter);
        let guarded = ConditionalAnnotator::when(|_: &Document| true, filter).unwrap();
        let analyzer = Analyzer::builder().add(Producer).step(guarded).build().unwrap();
        let mut docs = vec![Document::new("two words")];
        assert!(analyzer.run(&mut docs)[0].is_ok());
        (docs.pop().unwrap(), want)
    };
    let (first, want) = run();
    let (second, want_again) = run();
    assert_eq!(want, want_again, "the same construction hashes the same");
    let key = LONG_WORDS.id();
    assert_eq!(first.provenance(key), second.provenance(key));
    let p = first.provenance(key).unwrap();
    assert_eq!(p.annotator, "turbo:filter");
    assert_eq!(p.reads, vec![WORDS.id()]);
    assert_eq!(p.config, want);

    // The declined branch's layers name the conditional and record its
    // config hash.
    let skipped = {
        let filter = FilterAnnotator::with_document(WORDS, LONG_WORDS, |_: &Document, _: &SpanItem| true).unwrap();
        let guarded = ConditionalAnnotator::when(|_: &Document| false, filter).unwrap();
        let want = guarded.config_hash();
        let analyzer = Analyzer::builder().add(Producer).step(guarded).build().unwrap();
        let mut docs = vec![Document::new("two words")];
        assert!(analyzer.run(&mut docs)[0].is_ok());
        let doc = docs.pop().unwrap();
        let p = doc.provenance(key).unwrap().clone();
        (doc, p, want)
    };
    let (doc, p, want) = skipped;
    assert_eq!(doc.status(key), Some(&Status::Skipped("condition is false".into())));
    assert_eq!(p.annotator, "turbo:conditional");
    assert_eq!(p.config, want);

    // config_hash tells the configurations apart: the condition's
    // identity, and either branch's configuration.
    let condition = |_: &Document| true;
    let a = ConditionalAnnotator::when(condition, Bound::new(Producer)).unwrap();
    let b = ConditionalAnnotator::when(condition, Bound::new(Producer)).unwrap();
    assert_eq!(a.config_hash(), b.config_hash(), "the same construction hashes the same");
    assert_ne!(a.config_hash(), 0);
    let other_condition = ConditionalAnnotator::when(|_: &Document| false, Bound::new(Producer)).unwrap();
    assert_ne!(a.config_hash(), other_condition.config_hash(), "the condition's identity hashes in");
    let with_false = ConditionalAnnotator::new(condition, Bound::new(Producer), Bound::new(Producer)).unwrap();
    assert_ne!(a.config_hash(), with_false.config_hash(), "the false branch hashes in");

    let keep_a = |_: &Document, _: &SpanItem| true;
    let keep_b = |_: &Document, _: &SpanItem| false;
    let fa = FilterAnnotator::with_document(WORDS, LONG_WORDS, keep_a).unwrap();
    let fb = FilterAnnotator::with_document(WORDS, LONG_WORDS, keep_b).unwrap();
    assert_ne!(fa.config_hash(), fb.config_hash(), "the predicate's identity hashes in");
    let ga = ConditionalAnnotator::when(condition, fa).unwrap();
    let gb = ConditionalAnnotator::when(condition, fb).unwrap();
    assert_ne!(ga.config_hash(), gb.config_hash(), "a branch's configuration hashes in");
}
