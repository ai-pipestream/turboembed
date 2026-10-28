use turbo_maxent::ModelFile;

#[test]
fn opennlp_archives_are_refused_in_memory() {
    // An existing OpenNLP archive accepted before #1511.
    let bytes = include_bytes!("fixtures/models/tokenizer.bin").to_vec();
    let error = ModelFile::open(bytes).expect_err("OpenNLP loading must be disabled");
    assert!(error.to_string().starts_with("OpenNLP model loading is disabled"), "{error}");
}

#[test]
fn disabled_loader_does_not_read_the_requested_path() {
    let error = ModelFile::read("model-loading-disabled-do-not-open.bin").unwrap_err();
    assert!(error.to_string().starts_with("OpenNLP model loading is disabled"), "{error}");
}
