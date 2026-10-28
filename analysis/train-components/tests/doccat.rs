//! The document trainer rejects samples whose events do not survive cutoff.

use turbo_analysis::Span;
use turbo_doccat::DoccatFactory;
use turbo_formats::{DocumentSample, Field};
use turbo_train::{Key as ParamKey, TrainingConfiguration, TrainingParameters};
use turbo_train_components::Stamp;
use turbo_train_components::doccat::train_doccat;

#[test]
fn insufficient_training_data_is_rejected() {
    let source = "1 a b c";
    let samples = [DocumentSample {
        category: Field::of(source, Span::new(0, 1)),
        tokens: [(2, 3), (4, 5), (6, 7)].map(|(a, b)| Field::of(source, Span::new(a, b))).to_vec(),
        source: Span::new(0, 7),
    }];
    let mut params = TrainingParameters::new();
    params.put(ParamKey::ITERATIONS, "100");
    params.put(ParamKey::CUTOFF, "5");

    let factory = DoccatFactory::new();
    let mut config = TrainingConfiguration::default();
    let error = train_doccat("x-unspecified", &samples, &params, &factory, &mut config, Stamp::at(0)).unwrap_err();
    assert!(matches!(
        error,
        turbo_train_components::Error::Train(turbo_train::Error::InsufficientTrainingData(
            turbo_train::error::NO_EVENTS
        ))
    ));
}
