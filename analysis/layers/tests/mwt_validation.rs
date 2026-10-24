use turbo_analysis::{Annotator, Document, Span, Spans, View};
use turbo_analysis_layers::{MwtLanguage, RuleMwtExpander, Tokens};

#[test]
fn malformed_tokens_are_errors_before_expansion() {
    let doc = Document::new("café del");
    let view = doc.get::<View>().unwrap();
    let expander = RuleMwtExpander::new(MwtLanguage::Spanish);
    for spans in [
        vec![Span { start: 5, end: 2 }],
        vec![Span::new(0, 999)],
        vec![Span::new(4, 5)],
        vec![Span::new(0, 4)],
        vec![Span::new(0, 6), Span::new(5, 8)],
        vec![Span::new(6, 8), Span::new(0, 2)],
    ] {
        let mut t = Tokens { spans: Spans::new(view.id()), sentence: None };
        for span in spans {
            t.spans.push(span);
        }
        assert!(expander.annotate((view, &t)).is_err());
    }
}

#[test]
fn validation_checks_view_fragments_and_sentence_columns() {
    use turbo_analysis::{Ix, ViewId};
    use turbo_analysis_layers::Sentences;
    let doc = Document::new("del al");
    let view = doc.get::<View>().unwrap();
    let expander = RuleMwtExpander::new(MwtLanguage::Spanish);
    let mut t = Tokens { spans: Spans::new(ViewId(7)), sentence: None };
    assert!(expander.annotate((view, &t)).is_err());
    let s = Sentences { spans: Spans::new(ViewId(7)) };
    assert!(s.check_contiguous_in(view).is_err());
    t.spans = Spans::new(view.id());
    t.spans.push_fragments(&[Span::new(0, 3), Span::new(4, 6)]);
    assert!(expander.annotate((view, &t)).is_err());
    let mut s = Sentences { spans: Spans::new(view.id()) };
    s.spans.push_fragments(&[Span::new(0, 3), Span::new(4, 6)]);
    assert!(s.check_contiguous_in(view).is_err());
    t.spans = Spans::new(view.id());
    t.spans.push(Span::new(0, 3));
    t.spans.push(Span::new(4, 6));
    t.sentence = Some(vec![]);
    assert!(expander.annotate((view, &t)).is_err());
    t.sentence = Some(vec![Ix::new(1), Ix::new(0)]);
    assert!(expander.annotate((view, &t)).is_err());
    t.sentence = Some(vec![Ix::new(0), Ix::new(0)]);
    assert!(expander.annotate((view, &t)).is_ok());
}
