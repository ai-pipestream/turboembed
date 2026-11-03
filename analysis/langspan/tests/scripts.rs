use turbo_analysis::{Analyzer, Annotator, Document, Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{Language, LanguageSpans, Tokens};
use turbo_langspan::ScriptDetector;
use turbo_unicode::Script;

fn analyze(text: &str) -> Document {
    let analyzer = Analyzer::builder().add(ScriptDetector).build().unwrap();
    let mut document = Document::new(text);
    analyzer.run(std::slice::from_mut(&mut document)).remove(0).unwrap();
    document
}

#[test]
fn scripts_are_evidence_not_language_predictions() {
    let text = "hello Привет 中文 مرحبا";
    let document = analyze(text);
    let spans = document.get::<LanguageSpans>().unwrap();
    assert_eq!(spans.script, [Script::Latin, Script::Cyrillic, Script::Han, Script::Arabic]);
    for span in spans.iter() {
        assert_eq!(span.code, "und");
        assert_eq!(span.confidence, 0.0);
        assert!(text.get(span.span.range()).is_some());
    }
    assert_eq!(document.get::<Language>().unwrap().best(), Some(("und", 0.0)));
    assert!(ScriptDetector.model().is_none());
}

#[test]
fn combining_marks_and_supplementary_letters_retain_source_offsets() {
    let text = "é e\u{301} 𐐀";
    let document = analyze(text);
    let spans = document.get::<LanguageSpans>().unwrap();
    let pieces: Vec<_> = spans.iter().map(|s| (&text[s.span.range()], s.script)).collect();
    assert_eq!(pieces, [("é e\u{301}", Script::Latin), ("𐐀", Script::Deseret)]);
}

#[test]
fn empty_and_non_letter_inputs_produce_no_language_evidence() {
    for text in ["", "123 ! 😀", "\u{301}"] {
        let document = analyze(text);
        assert!(document.get::<LanguageSpans>().unwrap().spans.is_empty());
        assert!(document.get::<Language>().unwrap().best().is_none());
    }
}

#[test]
fn existing_token_boundaries_are_preserved() {
    let text = "COVID対策";
    let mut document = Document::new(text);
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, text.len() as u32));
    document.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("test-input")).unwrap();
    let analyzer = Analyzer::builder().given(Key::<Tokens>::DEFAULT).add(ScriptDetector).build().unwrap();
    analyzer.run(std::slice::from_mut(&mut document)).remove(0).unwrap();
    let spans = document.get::<LanguageSpans>().unwrap();
    assert_eq!(spans.spans.len(), 1);
    assert_eq!(spans.iter().next().unwrap().span, Span::new(0, text.len() as u32));
    assert_eq!(spans.script, [Script::Latin]);
}

#[test]
fn malformed_sentences_and_tokens_are_errors() {
    use turbo_analysis::View;
    use turbo_analysis_layers::Sentences;
    let document = Document::new("café language");
    let view = document.get::<View>().unwrap();
    for spans in [
        vec![Span { start: 5, end: 2 }],
        vec![Span::new(0, 99)],
        vec![Span::new(4, 5)],
        vec![Span::new(0, 4)],
        vec![Span::new(0, 6), Span::new(5, 7)],
        vec![Span::new(6, 7), Span::new(0, 2)],
    ] {
        let mut sentences = Sentences { spans: Spans::new(view.id()) };
        let mut tokens = Tokens { spans: Spans::new(view.id()), sentence: None };
        for span in spans {
            sentences.spans.push(span);
            tokens.spans.push(span);
        }
        assert!(ScriptDetector.annotate((view, Some(&sentences), None)).is_err());
        assert!(ScriptDetector.annotate((view, None, Some(&tokens))).is_err());
    }
}
