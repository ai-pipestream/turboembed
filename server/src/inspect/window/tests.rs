use super::*;
use crate::api::TURBO_E_CAPACITY;
use crate::offsets::OffsetMode;
use turbo_analysis::{Key, Provenance};
use turbo_analysis_layers::Tokens;

#[test]
fn annotation_total_is_checked_before_cloning_another_segment() {
    let mut doc = Document::new("abcdef");
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 6));
    spans.push(Span::new(1, 5));
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("test")).unwrap();
    let codecs = crate::results::document::codecs();
    let bytes = turbo_analysis::export::export(&doc, codecs).unwrap().to_bytes().unwrap();
    let graph = turbo_analysis::export::DocumentGraph::from_bytes(&bytes).unwrap();
    let doc = turbo_analysis::export::import(&graph, codecs).unwrap();
    let text = doc.source();
    let offsetter = Offsetter::new(text, OffsetMode::Utf8);
    let span = Span::new(0, text.len() as u32);
    let filter = vec![Tokens::ID.to_owned()];

    // Three intervals carry 1 + 2 + 1 annotations. The small budget exercises
    // the production sweep's capacity check without constructing millions.
    let error = combine_with_budget(&doc, ViewId::SOURCE, text, span, &offsetter, &filter, 3).unwrap_err();
    assert_eq!(error.code, TURBO_E_CAPACITY);
    assert!(error.message.contains("annotation memberships"));
    let segments = combine_with_budget(&doc, ViewId::SOURCE, text, span, &offsetter, &filter, 4).unwrap();
    assert_eq!(segments.len(), 3);
    assert_eq!(segments.iter().map(|s| s.annotations.len()).sum::<usize>(), 4);
}
