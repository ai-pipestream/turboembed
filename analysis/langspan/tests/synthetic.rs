//! The span detector and its annotator on a small model written here,
//! where `x` is evidence for deu and `y` for eng, and the model also has
//! ell and rus, each the only language of its script. No published model
//! is needed.

mod common;

use std::slice;

use turbo_analysis::annotate::Bound;
use turbo_analysis::{Analyzer, Annotator, Document, Error, Key, Provenance, Span, Spans, View, ViewBuilder, ViewId};
use turbo_analysis_layers::{Language, LanguageSpans, Sentences, Tokens};
use turbo_langdetect::NO_EVIDENCE;
use turbo_langspan::{Settings, SpanDetector};
use turbo_unicode::Script;

use common::small_detector;

fn detector() -> SpanDetector {
    let detector = small_detector(&["deu", "eng", "ell", "rus"], &[("x", &[(0, 2.0)]), ("y", &[(1, 2.0)])]);
    SpanDetector::new(detector, Settings::default())
}

/// Runs the step on `doc` with whichever of sentences and tokens it has.
fn run(doc: Document) -> (Document, Result<(), Error>) {
    let mut b = Analyzer::builder();
    if doc.get::<Sentences>().is_some() {
        b = b.given(Key::<Sentences>::DEFAULT);
    }
    if doc.get::<Tokens>().is_some() {
        b = b.given(Key::<Tokens>::DEFAULT);
    }
    let analyzer = b.add(detector()).build().unwrap();
    let mut doc = doc;
    let result = analyzer.run(slice::from_mut(&mut doc)).remove(0);
    (doc, result)
}

/// The spans as (text, code, script, confidence).
fn spans(doc: &Document) -> Vec<(&str, &str, Script, f64)> {
    let layer = doc.get::<LanguageSpans>().expect("language spans");
    layer.iter().map(|s| (&doc.source()[s.span.range()], s.code, s.script, s.confidence)).collect()
}

fn summary(doc: &Document) -> Vec<(&str, f64)> {
    doc.get::<Language>().expect("a language layer").iter().collect()
}

fn sentences(doc: &mut Document, spans: &[(u32, u32)]) {
    let mut s = Spans::new(ViewId::SOURCE);
    for &(a, b) in spans {
        s.push(Span::new(a, b));
    }
    doc.add(&Key::DEFAULT, Sentences { spans: s }, Provenance::by("test")).unwrap();
}

#[test]
fn two_sentences_in_two_languages_are_two_spans_that_touch() {
    let (doc, result) = run(Document::new("xxxx xxxx xxxx. yyyy yyyy yyyy."));
    result.unwrap();
    let got = spans(&doc);
    assert_eq!(
        got.iter().map(|s| (s.0, s.1, s.2)).collect::<Vec<_>>(),
        [("xxxx xxxx xxxx. ", "deu", Script::Latin), ("yyyy yyyy yyyy.", "eng", Script::Latin)]
    );
    // Each run's probability over the Latin languages of the model.
    assert!(got.iter().all(|s| s.3 > 0.5 && s.3 <= 1.0), "{got:?}");
    // Twelve letters each: a tie, which goes to the language first in the text.
    assert_eq!(summary(&doc), [("deu", 0.5), ("eng", 0.5)]);
}

#[test]
fn a_quotation_takes_its_opening_mark_and_leaves_the_colon() {
    let (doc, result) = run(Document::new("xxxx xxxx xxxx: \"yyyy yyyy yyyy\" xxxx xxxx xxxx."));
    result.unwrap();
    let got: Vec<(&str, &str)> = spans(&doc).iter().map(|s| (s.0, s.1)).collect();
    assert_eq!(got, [("xxxx xxxx xxxx: ", "deu"), ("\"yyyy yyyy yyyy\" ", "eng"), ("xxxx xxxx xxxx.", "deu")]);
    assert_eq!(summary(&doc), [("deu", 2.0 / 3.0), ("eng", 1.0 / 3.0)]);
}

#[test]
fn with_a_sentence_layer_a_span_starts_where_its_sentence_does() {
    let text = "xxxx xxxx xxxx.  - yyyy yyyy yyyy";
    let (doc, result) = run(Document::new(text));
    result.unwrap();
    assert_eq!(spans(&doc)[1].0, "yyyy yyyy yyyy");

    let mut doc = Document::new(text);
    sentences(&mut doc, &[(0, 15), (17, 33)]);
    let (doc, result) = run(doc);
    result.unwrap();
    assert_eq!(spans(&doc)[1].0, "- yyyy yyyy yyyy");
    let p = doc.provenance(Key::<LanguageSpans>::DEFAULT.id()).unwrap();
    assert!(p.reads.contains(&Key::<Sentences>::DEFAULT.id()), "{:?}", p.reads);
}

#[test]
fn with_a_token_layer_words_stay_inside_tokens() {
    // Without tokens "xxxxyyyy" is one word of both languages; tokens cut
    // it in two, and the switch can fall between them.
    let text = "xxxx xxxx xxxxyyyy, yyyy yyyy yyyy";
    let mut doc = Document::new(text);
    let mut t = Spans::new(ViewId::SOURCE);
    for (a, b) in [(0, 4), (5, 9), (10, 14), (14, 18), (18, 19), (20, 24), (25, 29), (30, 34)] {
        t.push(Span::new(a, b));
    }
    doc.add(&Key::DEFAULT, Tokens { spans: t, sentence: None }, Provenance::by("test")).unwrap();
    let (doc, result) = run(doc);
    result.unwrap();
    let got: Vec<(&str, &str)> = spans(&doc).iter().map(|s| (s.0, s.1)).collect();
    assert_eq!(got, [("xxxx xxxx xxxx", "deu"), ("yyyy, yyyy yyyy yyyy", "eng")]);
}

#[test]
fn scripts_with_one_language_need_no_scoring_and_scripts_with_none_are_und() {
    let (doc, result) = run(Document::new("xxxx xxxx xxxx. Καλημέρα κόσμε! Привет мир. สวัสดี"));
    result.unwrap();
    let got = spans(&doc);
    assert_eq!(
        got.iter().map(|s| (s.0, s.1, s.2)).collect::<Vec<_>>(),
        [
            ("xxxx xxxx xxxx. ", "deu", Script::Latin),
            ("Καλημέρα κόσμε! ", "ell", Script::Greek),
            ("Привет мир. ", "rus", Script::Cyrillic),
            ("สวัสดี", "und", Script::Thai),
        ]
    );
    assert_eq!((got[1].3, got[2].3, got[3].3), (1.0, 1.0, 0.0));
    // 13 Greek letters, 12 Latin, 9 Cyrillic; the Thai run has no language.
    let s = summary(&doc);
    assert_eq!(s.iter().map(|l| l.0).collect::<Vec<_>>(), ["ell", "deu", "rus"]);
    assert_eq!(s[0].1, 13.0 / 34.0);
    assert!((s.iter().map(|l| l.1).sum::<f64>() - 1.0).abs() < 1e-12);
}

#[test]
fn a_text_with_no_language_evidence_is_skipped_and_one_of_unknown_scripts_is_und() {
    for text in ["", "   ", "1234 5678", "https://example.com/a #tag @someone", "!!! ... ???"] {
        let (doc, result) = run(Document::new(text));
        result.unwrap();
        assert!(doc.get::<LanguageSpans>().is_none(), "{text:?}");
        let status = format!("{:?}", doc.status(Key::<LanguageSpans>::DEFAULT.id()));
        assert!(status.contains(NO_EVIDENCE), "{text:?}: {status}");
    }
    let (doc, result) = run(Document::new("สวัสดี ครับ"));
    result.unwrap();
    assert_eq!(spans(&doc).iter().map(|s| s.1).collect::<Vec<_>>(), ["und"]);
    assert_eq!(doc.get::<Language>().unwrap().iter().count(), 0);
}

#[test]
fn sentences_or_tokens_of_another_view_fail_the_document() {
    let mut doc = Document::new("xxxx yyyy");
    let mut b = ViewBuilder::new(doc.source());
    b.replace(1, "X");
    b.keep(9);
    let (text, alignment) = b.finish();
    let view = View::derived(doc.get::<View>().unwrap(), text, alignment);
    doc.add(&Key::named("upper"), view, Provenance::by("test")).unwrap();
    sentences(&mut doc, &[(0, 9)]);
    let analyzer = Analyzer::builder()
        .given(Key::<View>::named("upper"))
        .given(Key::<Sentences>::DEFAULT)
        .step(Bound::new(detector()).read(0, "upper"))
        .build()
        .unwrap();
    let e = analyzer.run(slice::from_mut(&mut doc)).remove(0).unwrap_err();
    assert!(e.to_string().contains("the sentences are in another view"), "{e}");
}

#[test]
fn the_step_records_its_model_and_settings() {
    let d = detector();
    assert_eq!(d.model().unwrap().id, "opennlp:LanguageDetectorME:und");
    let other =
        SpanDetector::new(small_detector(&["deu", "eng"], &[]), Settings { clause_letters: 11, ..Settings::default() });
    assert_ne!(d.config_hash(), other.config_hash());
    assert_eq!(d.languages().label(turbo_analysis::Tag(d.undetermined())), "und");
    assert_eq!(d.languages().len(), 5);
}

#[test]
fn a_restricted_detector_answers_only_the_languages_it_keeps() {
    let detector = small_detector(&["deu", "eng", "ell", "rus"], &[("x", &[(0, 2.0)]), ("y", &[(1, 2.0)])]);
    let d = SpanDetector::restricted(detector, Settings::default(), |c| c == "deu" || c == "rus");
    let analyzer = Analyzer::builder().add(d).build().unwrap();
    let mut doc = Document::new("yyyy yyyy yyyy. Καλημέρα κόσμε! Привет мир.");
    analyzer.run(slice::from_mut(&mut doc)).remove(0).unwrap();
    let got: Vec<(&str, &str)> = spans(&doc).iter().map(|s| (s.0, s.1)).collect();
    assert_eq!(got, [("yyyy yyyy yyyy. ", "deu"), ("Καλημέρα κόσμε! ", "und"), ("Привет мир.", "rus")]);
}

#[test]
fn tokens_out_of_text_order_fail_the_document() {
    let mut doc = Document::new("xxxx yyyy");
    let mut t = Spans::new(ViewId::SOURCE);
    t.push(Span::new(5, 9));
    t.push(Span::new(0, 4));
    doc.add(&Key::DEFAULT, Tokens { spans: t, sentence: None }, Provenance::by("test")).unwrap();
    let (_, result) = run(doc);
    let e = result.unwrap_err();
    assert!(e.to_string().contains("not in text order"), "{e}");
}

/// Texts with marks before any letter, emoji, mixed scripts inside one
/// chunk, line breaks and stray punctuation: the spans are in order,
/// touch, start and end on character boundaries and cover every word.
#[test]
fn odd_texts_give_spans_that_tile_the_words() {
    let texts = [
        "\u{301}xxxx yyyy",
        "xxxx😀yyyy 😀😀 ...",
        "«xxxx» (yyyy) — ¿xxxx? ¡yyyy!",
        "x\r\ny\r\n\r\nΚαλη\u{301}μερα",
        "iPhone是Привет",
        "xx-yy x'y x.y. 'x' -y- .",
        "\u{200D}\u{200C}xxxx\u{200D}",
        "ab\u{0}cd",
    ];
    let d = detector();
    let mut scratch = turbo_langspan::SpanScratch::new();
    let mut runs = Vec::new();
    for text in texts {
        d.detect(text, None, None, &mut scratch, &mut runs);
        assert!(!runs.is_empty(), "{text:?}");
        for w in runs.windows(2) {
            assert_eq!(w[0].span.end, w[1].span.start, "{text:?}");
        }
        for r in &runs {
            assert!(r.span.start < r.span.end, "{text:?}");
            assert!(text.is_char_boundary(r.span.start as usize) && text.is_char_boundary(r.span.end as usize));
            assert!((0.0..=1.0).contains(&r.confidence));
        }
        let (doc, result) = run(Document::new(text));
        result.unwrap();
        assert_eq!(doc.get::<LanguageSpans>().unwrap().spans.len(), runs.len(), "{text:?}");
    }
}
