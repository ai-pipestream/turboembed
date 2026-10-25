use turbo_abbrev::AbbreviationAnnotator;
use turbo_analysis::{Annotator, Document, Span, Spans, View};
use turbo_analysis_layers::{Sentences, Tokens};

#[test]
fn malformed_sentences_and_tokens_are_errors() {
    let doc = Document::new("Café World Health Organization (WHO). WHO");
    let view = doc.get::<View>().unwrap();
    let mut valid = Sentences { spans: Spans::new(view.id()) };
    valid.spans.push(Span::new(0, view.text().len() as u32));
    assert!(AbbreviationAnnotator.annotate((view, &valid, None)).is_ok());
    for spans in [
        vec![Span { start: 5, end: 2 }],
        vec![Span::new(0, 999)],
        vec![Span::new(4, 5)],
        vec![Span::new(0, 4)],
        vec![Span::new(0, 36), Span::new(0, 36)],
        vec![Span::new(6, 7), Span::new(0, 2)],
    ] {
        let mut s = Sentences { spans: Spans::new(view.id()) };
        let mut t = Tokens { spans: Spans::new(view.id()), sentence: None };
        for span in spans {
            s.spans.push(span);
            t.spans.push(span);
        }
        assert!(AbbreviationAnnotator.annotate((view, &s, None)).is_err());
        assert!(AbbreviationAnnotator.annotate((view, &valid, Some(&t))).is_err());
    }
}
