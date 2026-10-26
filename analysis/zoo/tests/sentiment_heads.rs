use turbo_zoo::{Format, Index, Kind, LicenceClass};

#[test]
fn the_masc_aspect_head_is_pinned_and_experimental() {
    let index = Index::builtin();
    let artifact = index.get("turbo-en-masc-3.0.0-aspect-sentiment-1").unwrap();
    assert_eq!(artifact.kind, Kind::Model);
    assert_eq!(artifact.format, Format::Zip);
    assert_eq!(artifact.class, LicenceClass::Permissive);
    assert_eq!(artifact.trained_on.len(), 2);
    assert!(artifact.attribution.as_ref().unwrap().contains("not the default extractor"));
    assert_eq!(artifact.files.len(), 1);
    let file = &artifact.files[0];
    assert_eq!(file.size, 25_106);
    assert_eq!(file.sha256.to_string(), "109c0eea8f2f0d5aa979b9f819820504456d236b372222a9ad9e34362d241be3");
    assert!(file.url.to_string().ends_with("/models-sentiment-924-1/aspect-sentiment-1.zip"));
}
