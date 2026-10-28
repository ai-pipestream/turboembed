//! Training the document categorizer (`DocumentCategorizerME.train`) and
//! running the trained model: the categories of OpenNLP's
//! `DocumentCategorizerMETest` and the annotators on views, sentences,
//! structure blocks and chunks.

use turbo_doccat::DoccatFactory;
use turbo_formats::DocumentSample;
use turbo_formats::native::DocumentReader;
use turbo_train::{Key as ParamKey, TrainingConfiguration, TrainingParameters};
use turbo_train_components::Stamp;
use turbo_train_components::doccat::train_doccat;

fn parse_samples(corpus: &str) -> Vec<DocumentSample<'_>> {
    DocumentReader::new(corpus).unwrap().map(|r| r.expect("valid sample")).collect()
}

#[test]
fn insufficient_training_data_is_rejected() {
    let corpus = "1\ta b c\n";
    let samples = parse_samples(corpus);
    let mut params = TrainingParameters::new();
    params.put(ParamKey::ITERATIONS, "100");
    params.put(ParamKey::CUTOFF, "5");

    let factory = DoccatFactory::new();
    let mut config = TrainingConfiguration::default();
    assert!(train_doccat("x-unspecified", &samples, &params, &factory, &mut config, Stamp::at(0)).is_err());
}
