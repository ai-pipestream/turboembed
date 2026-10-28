//! Sentence clearing must preserve the original document partitions.
use turbo_formats::NameSample;
use turbo_formats::native::NameReader;
use turbo_sequence::namefind::NameCodec;
use turbo_train_components::namefind::cross_validate::documents;
use turbo_train_components::namefind::{NameEventStream, TokenNameFinderFactory};

#[test]
fn sentence_clearing_changes_features_not_document_partitions() {
    let text = std::fs::read_to_string(turbo_goldens::data_dir().join("inputs/tagged/names.txt")).unwrap();
    let samples: Vec<_> = NameReader::new(&text).unwrap().map(Result::unwrap).collect();
    let original = samples.clone();
    let partitions = documents(&samples);
    assert_eq!(partitions, [0..4, 4..9, 9..13, 13..19, 19..23]);
    let isolated: Vec<_> = samples
        .iter()
        .map(|s| NameSample::new(s.tokens().to_vec(), s.names().to_vec(), true, s.source(), 0).unwrap())
        .collect();
    assert_eq!(documents(&isolated).len(), 23, "pre-clearing before partitioning is incorrect");
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
    assert_ne!(turbo_train::event::checksum(&document), turbo_train::event::checksum(&sentence));
    assert_eq!(samples, original);
}
