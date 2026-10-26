//! Text longer than one encoder pass, on the reduced model: the annotator
//! cuts it into sentence-aligned, overlapping windows, and what it writes
//! is what the windows give when run one by one as documents of their own,
//! with values found twice merged, every span a span of the source text.

mod common;

use std::collections::BTreeMap;
use std::ops::Range;

use turbo_relation::gliner2::words::split;

use turbo_analysis::{Analyzer, Document, Key, Layer, Provenance, Span, Spans, View, ViewId};
use turbo_analysis_layers::{Categories, Entities, Sentences, Tokens};
use turbo_schema_extract::layer::Records;
use turbo_schema_extract::{
    Cardinality, ClassificationTask, EntityType, ExtractionSchema, FieldSpec, Label, Limits, SchemaAnnotator,
    StructureSpec,
};

const TEXT: &str = "Zoë ordered two crêpes from Café São Paulo. Ana paid twelve euros for them. \
The order left Lisbon on Monday with Bruno. It reached Porto late, and Zoë signed for it. \
Ana wrote to the café about the delay. Bruno answered the next morning.";

fn schema() -> ExtractionSchema {
    let field = |name: &str| FieldSpec {
        name: name.into(),
        description: None,
        threshold: None,
        cardinality: Cardinality::Repeated,
        choices: None,
        checks: Vec::new(),
    };
    let entity = |name: &str, cardinality| EntityType {
        name: name.into(),
        description: None,
        threshold: None,
        cardinality,
        checks: Vec::new(),
    };
    ExtractionSchema {
        entities: vec![entity("person", Cardinality::Repeated), entity("place", Cardinality::Single)],
        classifications: vec![ClassificationTask {
            name: "tone".into(),
            labels: vec![
                Label { name: "calm".into(), description: None },
                Label { name: "urgent".into(), description: None },
            ],
            multi_label: false,
            threshold: 0.0,
            top_k: None,
        }],
        structures: vec![StructureSpec {
            name: "order".into(),
            fields: vec![
                field("buyer"),
                field("item"),
                FieldSpec {
                    choices: Some(vec!["paid".into(), "open".into()]),
                    cardinality: Cardinality::Single,
                    ..field("status")
                },
            ],
        }],
    }
}

fn limits() -> Limits {
    Limits { max_words: 20, ..Limits::default() }
}

fn annotator() -> SchemaAnnotator {
    SchemaAnnotator::new(common::offline(limits()), schema(), 0.0).unwrap()
}

/// `text` with whitespace tokens and, when `sentences`, a sentence per `. `.
fn document(text: &str, sentences: bool) -> Document {
    let mut doc = Document::new(text);
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut at = 0;
    for w in text.split_whitespace() {
        let start = text[at..].find(w).unwrap() + at;
        spans.push(Span::new(start as u32, (start + w.len()) as u32));
        at = start + w.len();
    }
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("test")).unwrap();
    if sentences {
        let mut s = Spans::new(ViewId::SOURCE);
        let mut start = 0;
        for (i, _) in text.match_indices(". ") {
            s.push(Span::new(start as u32, (i + 1) as u32));
            start = i + 2;
        }
        s.push(Span::new(start as u32, text.len() as u32));
        doc.add(&Key::DEFAULT, Sentences { spans: s }, Provenance::by("test")).unwrap();
    }
    doc
}

fn run(doc: Document, given_sentences: bool) -> Result<Document, String> {
    run_with(doc, given_sentences, annotator())
}

fn run_with(doc: Document, given_sentences: bool, annotator: SchemaAnnotator) -> Result<Document, String> {
    let mut b = Analyzer::builder().given(Key::<Tokens>::DEFAULT);
    if given_sentences {
        b = b.given(Key::<Sentences>::DEFAULT);
    }
    let analyzer = b.add(annotator).build().unwrap();
    let mut docs = vec![doc];
    for r in analyzer.run(&mut docs) {
        r.map_err(|e| e.to_string())?;
    }
    Ok(docs.pop().unwrap())
}

#[test]
fn windowed_categories_keep_every_labels_best_probability() {
    let task = |name: &str, names: &[&str], multi_label| ClassificationTask {
        name: name.into(),
        labels: names.iter().map(|n| Label { name: (*n).into(), description: None }).collect(),
        multi_label,
        threshold: 0.0,
        top_k: Some(1),
    };
    let schema = ExtractionSchema {
        classifications: vec![
            task("tone", &["calm", "urgent", "angry"], false),
            task("topic", &["billing", "shipping", "refund"], true),
        ],
        ..Default::default()
    };
    let make = || SchemaAnnotator::new(common::offline(limits()), schema.clone(), 0.0).unwrap();
    let doc = document(TEXT, true);
    let windows = make().windows(doc.get::<View>().unwrap(), doc.get::<Tokens>().unwrap(), doc.get()).unwrap();
    assert!(windows.len() >= 3, "exercise aggregation across several real model passes");
    let mut expected = vec![0.0_f64; 6];
    let mut first = None;
    for at in windows {
        let part = run_with(document(&TEXT[at], true), true, make()).unwrap();
        let row = part.get::<Categories>().unwrap().get(0).unwrap();
        first.get_or_insert_with(|| row.scores.to_vec());
        assert_eq!(row.scores.len(), expected.len());
        assert_eq!(row.selected.len(), 2, "one softmax winner and one top-k sigmoid winner");
        for (best, &score) in expected.iter_mut().zip(row.scores) {
            assert!(score > 0.0 && score < 1.0, "the real model also scores unselected labels");
            *best = best.max(score);
        }
    }
    assert!(expected.iter().zip(first.unwrap()).any(|(&best, first)| best > first), "later windows improve a score");
    let whole = run_with(doc, true, make()).unwrap();
    let categories = whole.get::<Categories>().unwrap();
    assert_eq!(categories.check(), Ok(()));
    let row = categories.get(0).unwrap();
    assert_eq!(row.scores, expected, "every label keeps its maximum window probability, even if unselected");
    assert_eq!(row.selected.len(), 2);
    for range in [0..3, 3..6] {
        let selected = row.selected.iter().map(|t| t.0 as usize).find(|i| range.contains(i)).unwrap();
        assert!(range.clone().all(|i| row.scores[i] <= row.scores[selected]));
    }
}

fn windows(doc: &Document) -> Vec<std::ops::Range<usize>> {
    annotator().windows(doc.get::<View>().unwrap(), doc.get::<Tokens>().unwrap(), doc.get::<Sentences>()).unwrap()
}

type Rows = BTreeMap<(u32, u32, String), u64>;

/// Entities as (start, end, kind) to score bits, moved by `by`.
fn entity_rows(e: &Entities, by: u32, into: &mut Rows) {
    for (i, s) in e.spans.iter() {
        let key = (s.start + by, s.end + by, e.kinds.label(e.kind[i.get()]).to_string());
        let score = e.score[i.get()];
        let kept = into.entry(key).or_insert(score.to_bits());
        if f64::from_bits(*kept) < score {
            *kept = score.to_bits();
        }
    }
}

/// Record span values as (start, end, path), moved by `by`.
fn value_rows(r: &Records, by: u32) -> Vec<(u32, u32, String)> {
    r.spans.iter().map(|(i, s)| (s.start + by, s.end + by, r.paths.label(r.path[i.get()]).to_string())).collect()
}

#[test]
fn long_text_is_read_in_sentence_windows_that_overlap_by_one() {
    let doc = document(TEXT, true);
    // The whole text is over max_words: on main this document is refused.
    let w = windows(&doc);
    assert!(w.len() >= 3, "{w:?}");
    let sentences = doc.get::<Sentences>().unwrap();
    let (starts, ends) = sentences.spans.columns();
    for (k, r) in w.iter().enumerate() {
        assert!(starts.contains(&(r.start as u32)) && ends.contains(&(r.end as u32)), "window {k} off sentences");
        if k > 0 {
            // The window repeats exactly the last sentence of the one before.
            let last = ends.iter().position(|&e| e as usize == w[k - 1].end).unwrap();
            assert_eq!(r.start, starts[last] as usize, "window {k} does not overlap by one sentence");
        }
    }
    assert_eq!((w[0].start, w[w.len() - 1].end), (0, TEXT.len()));
}

#[test]
fn a_long_document_gives_what_its_windows_give_one_by_one_merged() {
    let doc = run(document(TEXT, true), true).unwrap();
    let w = windows(&document(TEXT, true));
    let (mut want, mut values) = (Rows::new(), Vec::new());
    let mut places = Vec::new();
    for r in &w {
        let part = run(document(&TEXT[r.clone()], true), true).unwrap();
        let mut rows = Rows::new();
        entity_rows(part.get::<Entities>().unwrap(), r.start as u32, &mut rows);
        for (k, v) in rows {
            if k.2 == "place" {
                places.push((f64::from_bits(v), k));
            } else {
                let kept = want.entry(k).or_insert(v);
                if f64::from_bits(*kept) < f64::from_bits(v) {
                    *kept = v;
                }
            }
        }
        values.extend(value_rows(part.get::<Records>().unwrap(), r.start as u32));
    }
    // A single-cardinality type keeps its best value over all windows.
    if let Some((score, k)) = places.into_iter().reduce(|b, v| if v.0 > b.0 { v } else { b }) {
        want.insert(k, score.to_bits());
    }
    let mut got = Rows::new();
    let e = doc.get::<Entities>().unwrap();
    assert_eq!(e.check(), Ok(()));
    entity_rows(e, 0, &mut got);
    assert!(!got.is_empty(), "a threshold of 0 keeps entities");
    assert_eq!(got, want);
    assert_eq!(got.len(), e.len(), "no entity written twice");
    // Every record value is a window's value, at the same text, and every
    // window's value is in the document (records found twice are one).
    let r = doc.get::<Records>().unwrap();
    assert_eq!(r.check(), Ok(()));
    let mut got = value_rows(r, 0);
    got.sort();
    got.dedup();
    values.sort();
    values.dedup();
    assert_eq!(got, values);
    for (start, end, _) in &got {
        assert!(TEXT.is_char_boundary(*start as usize) && TEXT.is_char_boundary(*end as usize));
    }
    // One label for the single-label task.
    let c = doc.get::<Categories>().unwrap();
    assert_eq!(c.check(), Ok(()));
}

#[test]
fn without_sentences_windows_fall_on_tokens() {
    let doc = document(TEXT, false);
    let w = windows(&doc);
    assert!(w.len() >= 3, "{w:?}");
    let tokens = doc.get::<Tokens>().unwrap();
    let (starts, ends) = tokens.spans.columns();
    for r in &w {
        assert!(starts.contains(&(r.start as u32)) && ends.contains(&(r.end as u32)));
    }
    let out = run(doc, false).unwrap();
    let e = out.get::<Entities>().unwrap();
    assert_eq!(e.check(), Ok(()));
    assert!(
        e.spans.iter().all(|(_, s)| TEXT.is_char_boundary(s.start as usize) && TEXT.is_char_boundary(s.end as usize))
    );
}

#[test]
fn a_sentence_over_the_limit_alone_is_refused_naming_it() {
    let long = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen \
sixteen seventeen eighteen nineteen twenty twentyone twentytwo";
    let text = format!("Ana paid. {long}. Bruno left.");
    let err = run(document(&text, true), true).err().expect("refused");
    assert!(err.contains("window unit 1"), "{err}");
}

#[test]
fn text_that_fits_is_one_window() {
    let doc = document("Zoë ordered two crêpes. Ana paid.", true);
    let windows = windows(&doc);
    assert_eq!(windows.len(), 1);
    assert_eq!(windows[0], 0..doc.get::<View>().unwrap().text().len());
}

#[test]
fn text_with_space_around_it_that_fits_is_the_whole_text() {
    let text = "  Ana paid. Bruno left.  ";
    let w = windows(&document(text, true));
    assert_eq!(w.len(), 1);
    assert_eq!(w[0], 0..text.len());
}

/// The window of `whole` at `at` against the input built over that text alone.
fn same_as_alone(m: &turbo_schema_extract::Gliner2Extractor, s: &ExtractionSchema, text: &str, at: Range<usize>) {
    use turbo_relation::gliner2::words::split;
    let whole = m.build_words(text, s, split(text)).unwrap();
    assert!(std::ptr::eq(whole.text, text), "the prompt borrows the text, it does not copy it");
    let cut = whole.window(at.clone());
    let alone = m.build_words(&text[at.clone()], s, split(&text[at.clone()])).unwrap();
    assert!(std::ptr::eq(cut.text, &text[at.clone()]));
    assert_eq!(cut.ids, alone.ids, "{text:?} {at:?} ids");
    assert_eq!(cut.words, alone.words, "{text:?} {at:?} words");
    assert_eq!(cut.token_first, alone.token_first, "{text:?} {at:?} token_first");
    assert_eq!((&cut.prefix, &cut.choice_at, &cut.tasks), (&alone.prefix, &alone.choice_at, &alone.tasks));
    assert_eq!(cut.period, alone.period);
    let (c, a) = (cut.measure(at.len()), alone.measure(at.len()));
    assert_eq!((c.overhead, c.words, c.endings), (a.overhead, a.words, a.endings), "{text:?} {at:?} measure");
    // A window cut again is the same as one cut once.
    let inner = cut.window(0..at.len());
    assert_eq!((inner.ids, inner.words), (cut.ids, cut.words));
}

#[test]
fn a_window_cut_from_the_document_input_is_the_window_built_alone() {
    let (m, s) = (common::offline(limits()), schema());
    let mut cuts = windows(&document(TEXT, true));
    // A window that does not end on `.`, so the added `.` is cut too.
    cuts.push(0..TEXT.find(" twelve").unwrap());
    for at in cuts {
        same_as_alone(&m, &s, TEXT, at.clone());
        assert_eq!(m.check(&m.build_words(TEXT, &s, split(TEXT)).unwrap().window(at)), Ok(()), "fits once cut");
    }
    // URLs take the added `.` into themselves; every run of words is cut.
    for text in ["see https://a.b now", "WWW.Ex.com x https://a.b", "https://a.b. x www.c.d", "", "Ana  "] {
        let words = split(text);
        same_as_alone(&m, &s, text, 0..text.len());
        for a in 0..words.len() {
            for b in a..words.len() {
                same_as_alone(&m, &s, text, words[a].start..words[b].end.min(text.len()));
            }
        }
    }
}
