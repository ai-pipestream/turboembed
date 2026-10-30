use std::path::Path;

use turbo_train_recipes::release::{MODELS_RELEASES, ModelsRelease, entries};

#[test]
fn classical_release_entries_stop_before_reading_cards_or_models() {
    for release in MODELS_RELEASES {
        let error = entries(Path::new("missing-retired-release"), &release).unwrap_err();
        assert!(error.to_string().contains("OpenNLP model loading is disabled"), "{error}");
    }
    let empty = ModelsRelease { tag: "empty", models: &[] };
    assert!(entries(Path::new("missing-retired-release"), &empty).is_err());
}
