mod common;
use turbo_analysis::{Annotator, Document, Span, Spans, View};
use turbo_analysis_layers::{Sentences, Tokens};
use turbo_langspan::{Settings, SpanDetector};

fn detector(keep: impl Fn(&str) -> bool) -> SpanDetector {
    SpanDetector::restricted(common::small_detector(&["eng", "fra", "xyz"], &[]), Settings::default(), keep)
}

#[test]
fn retained_languages_are_part_of_configuration_identity() {
    let english = detector(|c| c == "eng");
    let french = detector(|c| c == "fra");
    let all = detector(|_| true);
    let unknown = detector(|c| c == "xyz");
    assert_eq!(english.model(), french.model());
    let hashes = [
        english.config_hash(),
        french.config_hash(),
        all.config_hash(),
        unknown.config_hash(),
        detector(|_| false).config_hash(),
    ];
    for (i, a) in hashes.iter().enumerate() {
        assert!(!hashes[i + 1..].contains(a));
    }
    assert_eq!(english.config_hash(), detector(|c| c == "eng").config_hash());
    let doc = Document::new("bonjour monde");
    let view = doc.get::<View>().unwrap();
    assert_eq!(english.annotate((view, None, None)).unwrap().0.iter().next().unwrap().code, "eng");
    assert_eq!(french.annotate((view, None, None)).unwrap().0.iter().next().unwrap().code, "fra");
}

#[test]
fn malformed_sentences_and_tokens_are_errors() {
    let detector = detector(|_| true);
    let doc = Document::new("café language");
    let view = doc.get::<View>().unwrap();
    for spans in [
        vec![Span { start: 5, end: 2 }],
        vec![Span::new(0, 99)],
        vec![Span::new(4, 5)],
        vec![Span::new(0, 4)],
        vec![Span::new(0, 6), Span::new(5, 7)],
        vec![Span::new(6, 7), Span::new(0, 2)],
    ] {
        let mut s = Sentences { spans: Spans::new(view.id()) };
        let mut t = Tokens { spans: Spans::new(view.id()), sentence: None };
        for span in spans {
            s.spans.push(span);
            t.spans.push(span);
        }
        assert!(detector.annotate((view, Some(&s), None)).is_err());
        assert!(detector.annotate((view, None, Some(&t))).is_err());
    }
}
