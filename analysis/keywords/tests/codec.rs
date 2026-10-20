//! The keywords layer crosses the document graph boundary
//! byte-identically, and a corrupted payload is refused, never
//! clamped: truncation, trailing bytes, a tag outside the tag set, a
//! score outside `[0, 1]`, an empty phrase, a term count of zero, a
//! row count that disagrees with the spans, a span outside the text.

use turbo_analysis::export::{DocumentGraph, ExportError, Writer, export, import};
use turbo_analysis::{Analyzer, Bound, Document, Key, Layer, Provenance, Span, Spans, Tag, TagSet, ViewId};
use turbo_analysis_layers::{PosTags, Sentences, Strs, Terms, Tokens};
use turbo_keywords::export::codecs;
use turbo_keywords::{Keywords, Stopwords, TextRankAnnotator, YAKE, YakeAnnotator};

/// The fish fixture with POS tags, so both extractors write one
/// document.
fn document() -> Document {
    let sentences: Vec<Vec<&str>> =
        vec![vec!["red", "Fish", "blue", "fish"], vec!["swim", "deep", "red", "swim"], vec!["Fish", "blue", "café"]];
    let mut text = String::new();
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut surface = Strs::default();
    let mut folded = Strs::default();
    let mut s_spans = Spans::new(ViewId::SOURCE);
    let mut cursor = 0usize;
    for sent in &sentences {
        let start = cursor;
        for token in sent {
            if !text.is_empty() {
                text.push(' ');
            }
            let s = text.len() as u32;
            text.push_str(token);
            spans.push(Span::new(s, text.len() as u32));
            surface.push(token);
            folded.push(&token.to_lowercase());
            cursor = text.len();
        }
        s_spans.push(Span::new(start as u32, cursor as u32));
    }
    let mut doc = Document::new(text);
    let by = Provenance::by("test");
    doc.add(&Key::DEFAULT, Sentences { spans: s_spans }, by.clone()).unwrap();
    let n = spans.len();
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, by.clone()).unwrap();
    doc.add(
        &Key::DEFAULT,
        PosTags { tagset: TagSet::new("universal", &["NOUN"]), tags: vec![Tag(0); n], prob: None },
        by.clone(),
    )
    .unwrap();
    doc.add(&Key::DEFAULT, Terms { surface, folded, stem: None, lemma: None }, by).unwrap();
    doc
}

/// Both annotators on one document: the yake instance is the default
/// one, the textrank instance is named.
fn analyzed() -> Document {
    let mut docs = vec![document()];
    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<PosTags>::DEFAULT)
        .given(Key::<Terms>::DEFAULT)
        .step(Bound::new(YakeAnnotator::new(Stopwords::custom(["qwerty"]).unwrap()).with_window_size(2)))
        .step(Bound::new(TextRankAnnotator::new()).write(0, "textrank"))
        .build()
        .unwrap();
    let results = analyzer.run(&mut docs);
    results[0].as_ref().unwrap();
    docs.pop().unwrap()
}

#[test]
fn keywords_cross_the_boundary_and_back_byte_identically() {
    let doc = analyzed();
    let registry = codecs();
    let graph = export(&doc, &registry).unwrap();
    let back = import(&graph, &registry).unwrap();

    let yake = back.get::<Keywords>().expect("the default keywords instance");
    let textrank = back.get_key(&Key::<Keywords>::named("textrank")).expect("the named instance");
    assert!(!yake.is_empty() && !textrank.is_empty());
    for layer in [yake, textrank] {
        layer.check().unwrap();
        assert!(layer.iter().all(|(_, item)| (0.0..=1.0).contains(&item.score)));
        for (span, item) in layer.iter() {
            assert!(!item.phrase.is_empty() && item.terms > 0);
            assert_eq!(span.start, span.start, "span is finite");
        }
    }
    assert!(yake.iter().all(|(_, item)| item.algorithm == YAKE));
    assert!(textrank.iter().all(|(_, item)| item.algorithm == "textrank"));

    // Byte-identical on the way back out, layers and all.
    let again = export(&back, &registry).unwrap();
    assert_eq!(again.to_bytes(), graph.to_bytes(), "re-export is byte-identical");
}

/// Replaces every keywords instance body with `make`'s bytes.
fn splice(graph: &mut DocumentGraph, make: impl Fn() -> Vec<u8>) {
    for layer in &mut graph.layers {
        if layer.layer == "turbo:keywords" && layer.body.is_some() {
            let body = make();
            layer.len = 1;
            layer.body = Some(body);
        }
    }
}

/// One valid body: one span in the default view, the shared tag set,
/// one row.
fn valid_body(phrase: &str, score: f64, tag: u16, terms: u16, span: Span) -> Vec<u8> {
    let mut w = Writer::new();
    w.str("").unwrap();
    w.usize(1).unwrap();
    w.usize(1).unwrap();
    w.span(span);
    w.str("keyword-algorithm").unwrap();
    w.usize(2).unwrap();
    w.str("yake").unwrap();
    w.str("textrank").unwrap();
    w.usize(1).unwrap();
    w.str(phrase).unwrap();
    w.f64(score);
    w.u16(tag);
    w.u16(terms);
    w.finish()
}

fn import_spliced(make: impl Fn() -> Vec<u8>) -> Result<Document, ExportError> {
    let doc = analyzed();
    let registry = codecs();
    let mut graph = export(&doc, &registry).unwrap();
    splice(&mut graph, make);
    import(&graph, &registry)
}

fn assert_refused(result: Result<Document, ExportError>, what: &str) {
    match result {
        Err(e) => {
            let text = e.to_string();
            assert!(
                text.contains("keywords")
                    || text.contains("trailing")
                    || text.contains("truncated")
                    || text.contains("score")
                    || text.contains("algorithm")
                    || text.contains("term count")
                    || text.contains("empty keyphrase")
                    || text.contains("rows for")
                    || text.contains("UTF-8"),
                "{what}: unexpected error {text}"
            );
        }
        Ok(_) => panic!("{what}: a corrupted payload imported"),
    }
}

#[test]
fn corruption_is_refused_not_clamped() {
    // Truncated: drop the last byte of a valid body.
    assert_refused(import_spliced(|| valid_body("red", 0.5, 0, 1, Span::new(0, 3))[..8].to_vec()), "truncated");
    // Trailing bytes.
    assert_refused(
        import_spliced(|| {
            let mut b = valid_body("red", 0.5, 0, 1, Span::new(0, 3));
            b.push(0);
            b
        }),
        "trailing bytes",
    );
    // A tag outside the tag set.
    assert_refused(import_spliced(|| valid_body("red", 0.5, 9, 1, Span::new(0, 3))), "tag outside");
    // Scores that are not a salience in [0, 1].
    assert_refused(import_spliced(|| valid_body("red", f64::NAN, 0, 1, Span::new(0, 3))), "NaN score");
    assert_refused(import_spliced(|| valid_body("red", 1.5, 0, 1, Span::new(0, 3))), "score above 1");
    assert_refused(import_spliced(|| valid_body("red", -0.1, 0, 1, Span::new(0, 3))), "score below 0");
    // An empty phrase.
    assert_refused(import_spliced(|| valid_body("", 0.5, 0, 1, Span::new(0, 3))), "empty phrase");
    // A term count of zero.
    assert_refused(import_spliced(|| valid_body("red", 0.5, 0, 0, Span::new(0, 3))), "term count zero");
    // Two rows for one span.
    assert_refused(
        import_spliced(|| {
            let mut w = Writer::new();
            w.str("").unwrap();
            w.usize(1).unwrap();
            w.usize(1).unwrap();
            w.span(Span::new(0, 3));
            w.str("keyword-algorithm").unwrap();
            w.usize(2).unwrap();
            w.str("yake").unwrap();
            w.str("textrank").unwrap();
            w.usize(2).unwrap();
            for _ in 0..2 {
                w.str("red").unwrap();
                w.f64(0.5);
                w.u16(0);
                w.u16(1);
            }
            w.finish()
        }),
        "row count mismatch",
    );
    // A span outside the text and one splitting a multi-byte character
    // (the document holds "café" at the end).
    assert_refused(import_spliced(|| valid_body("red", 0.5, 0, 1, Span::new(0, 99))), "span past the text");
    let cafe = document().source().find("café").unwrap() as u32;
    assert_refused(
        import_spliced(move || valid_body("café", 0.5, 0, 1, Span::new(cafe + 4, cafe + 5))),
        "a span off a character boundary",
    );
}
