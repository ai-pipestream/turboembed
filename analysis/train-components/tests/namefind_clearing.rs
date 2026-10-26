//! The #427 sentence-clear option must not repartition MASC documents.
use turbo_formats::evaluate::{CrossValidation, NameFinderEvaluator};
use turbo_formats::native::NameReader;
use turbo_formats::{Name, NameSample};
use turbo_maxent::ModelFile;
use turbo_sequence::PatternMode;
use turbo_sequence::namefind::{NameCodec, NameFinder};
use turbo_train::{Key, TrainingConfiguration, TrainingParameters};
use turbo_train_components::Stamp;
use turbo_train_components::namefind::cross_validate::documents;
use turbo_train_components::namefind::{
    MODEL_ENTRY, NameEventStream, TokenNameFinderCrossValidator, TokenNameFinderFactory, train_name_model_from_events,
};
use turbo_unicode::WhitespaceMode;

fn params() -> TrainingParameters {
    let mut params = TrainingParameters::new();
    params.put(Key::ITERATIONS, 5);
    params.put(Key::CUTOFF, 0);
    params
}

#[test]
fn sentence_clearing_changes_features_not_document_partitions() {
    let text = std::fs::read_to_string(turbo_goldens::data_dir().join("inputs/tagged/names.txt")).unwrap();
    let samples: Vec<_> = NameReader::new(&text, WhitespaceMode::Unicode).unwrap().map(Result::unwrap).collect();
    let original = samples.clone();
    let docs = documents(&samples);
    assert_eq!(docs, [0..4, 4..9, 9..13, 13..19, 19..23]);
    let factory = TokenNameFinderFactory::new("en", None, NameCodec::Bio).unwrap();
    let evaluate = |sentence_clear| {
        let cv = TokenNameFinderCrossValidator::new(&factory, None, params());
        let cv = if sentence_clear { cv.with_sentence_clearing() } else { cv };
        cv.evaluate(&samples, 3, &mut TrainingConfiguration::default(), Stamp::at(0)).unwrap()
    };
    let document = evaluate(false);
    let sentence = evaluate(true);
    assert_eq!(sentence, evaluate(true), "deterministic events, scores and predictions");
    assert_eq!(samples, original, "source fields and document flags remain intact");
    assert!(document.folds.iter().zip(&sentence.folds).any(|(a, b)| a.event_checksum != b.event_checksum));
    for ((partition, expected), actual) in CrossValidation::new(docs.len(), 3).zip(&document.folds).zip(&sentence.folds)
    {
        assert_eq!(actual.training_documents, expected.training_documents);
        assert_eq!(actual.test_documents, expected.test_documents);
        assert_eq!(actual.training_samples, expected.training_samples);
        assert_eq!(actual.test_samples, expected.test_samples);
        assert_eq!(actual.events, expected.events);
        assert_eq!(actual.training_documents + actual.test_documents, 5);
        assert_eq!(actual.training_samples + actual.test_samples, 23);
        assert!(partition.training.iter().all(|d| !partition.test.contains(d)));

        // Independent construction: retain document partitions, then replace
        // flags only in samples passed to the actual event stream and model.
        let isolated: Vec<_> = samples
            .iter()
            .map(|s| NameSample::new(s.tokens().to_vec(), s.names().to_vec(), true, s.source(), 0).unwrap())
            .collect();
        assert_eq!(documents(&isolated).len(), 23, "pre-clearing before partitioning is incorrect");
        let mut stream = NameEventStream::new(&factory, None).unwrap();
        let mut events = Vec::new();
        for &d in &partition.training {
            for i in docs[d].clone() {
                stream.events(&isolated[i], &mut events).unwrap();
            }
        }
        assert_eq!(actual.event_checksum, turbo_train::event::checksum(&events));
        let trained = train_name_model_from_events(
            &events,
            &factory,
            &params(),
            &mut TrainingConfiguration::default(),
            Stamp::at(0),
        )
        .unwrap();
        let file = ModelFile::open(trained.package.to_bytes()).unwrap();
        let maxent = file.maxent(MODEL_ENTRY).unwrap();
        let mut finder = NameFinder::new(&file, &maxent, PatternMode::Utf16Units).unwrap();
        let mut evaluator = NameFinderEvaluator::default();
        let mut position = 0;
        for &d in &partition.test {
            for i in docs[d].clone() {
                finder.clear_adaptive_data();
                let tokens: Vec<_> = isolated[i].tokens().iter().map(|t| t.as_str()).collect();
                let found = finder.find(&tokens);
                let predicted = &actual.predictions[position];
                assert_eq!(found.len(), predicted.len());
                for (a, b) in found.iter().zip(predicted) {
                    assert_eq!((a.span.start, a.span.end, &a.span.kind), (b.start, b.end, &b.kind));
                }
                let names: Vec<_> = found
                    .iter()
                    .map(|n| Name::new(n.span.start as u32, n.span.end as u32, n.span.kind.as_deref()).unwrap())
                    .collect();
                evaluator.evaluate(&isolated[i], &names).unwrap();
                position += 1;
            }
        }
        assert_eq!(actual.fmeasure, *evaluator.fmeasure());
    }
}
