use std::path::Path;
use turbo_train::TrainingConfiguration;
use turbo_train_components::{Error, ud::RECIPES};

#[test]
fn published_model_reproduction_stops_before_input_io() {
    let result = RECIPES[0].reproduce(
        Path::new("missing-treebank.conllu"),
        Path::new("missing-published.bin"),
        &mut TrainingConfiguration::default(),
    );
    assert!(matches!(result, Err(Error::Model(turbo_maxent::Error::Disabled))));
}
