use turbo_zoo::{Format, Index, Kind, LicenceClass};

#[test]
fn the_masc_aspect_head_is_pinned_experimental_and_local_only() {
    let index = Index::builtin();
    let artifact = index.get("turbo-en-masc-3.0.0-aspect-sentiment-1").unwrap();
    assert_eq!(artifact.kind, Kind::Model);
    assert_eq!(artifact.format, Format::Zip);
    assert_eq!(artifact.class, LicenceClass::LocalOnly);
    assert_eq!(artifact.trained_on.len(), 3);
    assert!(artifact.attribution.as_ref().unwrap().contains("not the default extractor"));
    assert_eq!(artifact.files.len(), 1);
    let file = &artifact.files[0];
    assert_eq!(file.size, 25_106);
    assert_eq!(file.sha256.to_string(), "109c0eea8f2f0d5aa979b9f819820504456d236b372222a9ad9e34362d241be3");
    assert!(file.url.to_string().ends_with("/models-sentiment-924-1/aspect-sentiment-1.zip"));
}

#[test]
fn modernbert_sentiment_heads_are_permissive_and_legacy_heads_are_local_only() {
    let index = Index::builtin();
    for (id, expected_size, expected_sha256) in [
        (
            "turbo-en-masc-3.0.0-modernbert-base-polarity-head-1",
            9_349,
            "5fb54814d92530072e90fa9d0e701792a4cbb7487353ae1a3ddb4cb60b196909",
        ),
        (
            "turbo-en-goemotions-modernbert-base-head-1",
            86_248,
            "e5e54e034221c36b5f89715e09c7c5ef08488ecac71151f2b43acdc4b3913919",
        ),
        (
            "turbo-en-civil-comments-modernbert-base-head-1",
            18_576,
            "dff23f93bc24e36e48f0da8cb3cca7201cb3a34f7124492bb02137cf963a9260",
        ),
        (
            "turbo-en-civil-comments-modernbert-base-context-head-1",
            37_008,
            "465cb98642303556687fd397ea508cf92b9f001c33701bc842b0ba33f45a3330",
        ),
    ] {
        let artifact = index.get(id).unwrap();
        assert_eq!(artifact.kind, Kind::Head, "{id}");
        assert_eq!(artifact.format, Format::Head, "{id}");
        assert_eq!(artifact.class, LicenceClass::Permissive, "{id}");
        assert_eq!(artifact.trained_on.len(), 2, "{id}");
        assert_eq!(artifact.trained_on[1].name, "answerdotai/ModernBERT-base", "{id}");
        assert_eq!(artifact.files.len(), 1, "{id}");
        let file = &artifact.files[0];
        assert_eq!(file.size, expected_size, "{id}");
        assert_eq!(file.sha256.to_string(), expected_sha256, "{id}");
        assert!(file.url.to_string().contains("/models-sentiment-1732-1/"), "{id}");
    }

    for legacy_id in [
        "turbo-en-goemotions-minilm-l6-v2-head-1",
        "turbo-en-goemotions-bge-small-en-v1.5-head-1",
        "turbo-en-civil-comments-minilm-l6-v2-head-1",
        "turbo-en-civil-comments-minilm-l6-v2-context-head-1",
        "turbo-en-civil-comments-bge-small-en-v1.5-head-1",
        "turbo-en-civil-comments-bge-small-en-v1.5-context-head-1",
    ] {
        let artifact = index.get(legacy_id).unwrap();
        assert_eq!(artifact.class, LicenceClass::LocalOnly, "{legacy_id}");
    }
}
