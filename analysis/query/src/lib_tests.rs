use turbo_analysis::{Analyzer, Document};
use turbo_analysis_layers::Sentences;
use turbo_chunker::terms::TermForm;

fn sentences_of(text: &str) -> Vec<(u32, u32)> {
    let analyzer = Analyzer::builder()
        .add(crate::QueryTermAnnotator::new(
            "eng",
            "eng",
            TermForm::Surface,
            crate::annotators::QuerySpanDetector::new(crate::default_span_detector(), false),
        ))
        .build()
        .unwrap();
    let mut docs = vec![Document::new(text)];
    analyzer.run(&mut docs).remove(0).unwrap();
    docs[0].get::<Sentences>().unwrap().spans.iter().map(|(_, s)| (s.start, s.end)).collect()
}

#[test]
fn unicode_edge_whitespace_takes_the_detector_path() {
    // "hello" is bytes 0..5 once any edge whitespace is trimmed.
    assert_eq!(sentences_of("hello"), vec![(0, 5)]);
    assert_eq!(sentences_of("hello\u{3000}"), vec![(0, 5)]);
    assert_eq!(sentences_of("\u{00A0}hello"), vec![(2, 7)]);
    assert_eq!(sentences_of("hello\u{000B}"), vec![(0, 5)]);
    assert_eq!(sentences_of("\u{2003}hello\u{2003}"), vec![(3, 8)]);
}

#[test]
fn query_quantity_extractor_accepts_the_long_units() {
    // Guards the silent fallback in query_quantity_extractor: if
    // with_units ever refuses the constant list, the spelled-out
    // units stop extracting instead of falling back unnoticed.
    let extractor = crate::query_quantity_extractor(turbo_analysis_layers::NumberNotation::LatinUs);
    let units: Vec<&str> = extractor.units().collect();
    assert!(units.contains(&"kilometers") && units.contains(&"milliseconds"), "{units:?}");
    assert!(units.len() > turbo_numeric::DEFAULT_UNITS.len());
}
