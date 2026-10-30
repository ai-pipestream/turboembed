//! Sentence clearing must preserve the original document partitions.
use turbo_analysis::Span;
use turbo_formats::{Field, Name, NameSample};
use turbo_sequence::namefind::NameCodec;
use turbo_train_components::namefind::cross_validate::documents;
use turbo_train_components::namefind::{NameEventStream, TokenNameFinderFactory};

#[test]
fn sentence_clearing_changes_features_not_document_partitions() {
    // Two documents repeat their entity across sentences, so adaptive
    // features differ when cleared at each sentence instead of each document.
    let source = "Mira arrived .\nMira stayed .\n\nOren arrived .\nOren stayed .\nOren left .\n";
    let sample = |spans: [(u32, u32); 3], clear, line| {
        let tokens = spans.map(|(a, b)| Field::of(source, Span::new(a, b))).to_vec();
        let name = Name { start: 0, end: 1, kind: Some("person"), source: tokens[0].span };
        NameSample::new(tokens, vec![name], clear, Span::new(spans[0].0, spans[2].1), line).unwrap()
    };
    let samples = vec![
        sample([(0, 4), (5, 12), (13, 14)], false, 1),
        sample([(15, 19), (20, 26), (27, 28)], false, 2),
        sample([(30, 34), (35, 42), (43, 44)], true, 4),
        sample([(45, 49), (50, 56), (57, 58)], false, 5),
        sample([(59, 63), (64, 68), (69, 70)], false, 6),
    ];
    let original = samples.clone();
    let partitions = documents(&samples);
    assert_eq!(partitions, [0..2, 2..5]);
    let isolated: Vec<_> = samples
        .iter()
        .map(|s| NameSample::new(s.tokens().to_vec(), s.names().to_vec(), true, s.source(), 0).unwrap())
        .collect();
    assert_eq!(documents(&isolated), [0..1, 1..2, 2..3, 3..4, 4..5], "pre-clearing before partitioning is incorrect");
    let factory = TokenNameFinderFactory::new("en", None, NameCodec::Bio).unwrap();
    let events = |input: &[NameSample<'_>]| {
        let mut stream = NameEventStream::new(&factory, None).unwrap();
        let mut events = Vec::new();
        for partition in &partitions {
            for i in partition.clone() {
                stream.events(&input[i], &mut events).unwrap();
            }
        }
        events
    };
    let document = events(&samples);
    let sentence = events(&isolated);
    assert_eq!(sentence, events(&isolated));
    assert_eq!(document.len(), sentence.len());
    assert_eq!(document.len(), 15);
    assert_eq!(document[..3], sentence[..3], "the first sentence has no adaptive history");
    assert_ne!(document[3..6], sentence[3..6], "the second sentence retains its document's entity history");
    assert_eq!(document[6..9], sentence[6..9], "the next document clears the preceding document's history");
    assert_ne!(turbo_train::event::checksum(&document), turbo_train::event::checksum(&sentence));
    assert_eq!(samples, original);
}
