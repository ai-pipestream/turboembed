use turbo_zoo::{Format, Index, Kind, Licence, LicenceClass, Policy};

#[test]
fn csc_and_its_pragmatics_head_are_pinned_and_permissive() {
    let index = Index::builtin();
    let csc = index.get("csc-conversational-sarcasm-2024").unwrap();
    assert_eq!(
        (csc.kind, csc.format, csc.license, csc.class),
        (Kind::Dataset, Format::Csv, Licence::Mit, LicenceClass::Permissive)
    );
    assert_eq!(csc.version, "f11c92b8d54686f64cf1d92766cef07201efb986");
    assert_eq!(csc.files.len(), 2);
    assert_eq!(csc.files[0].size, 9_966_538);
    assert_eq!(csc.files[0].sha256.to_string(), "fd8ca7531887823a315e91f5d0edbcc9ec685a557c333c37f485a15a9e55ad7a");
    assert!(csc.licences_to_accept(Policy::default()).is_empty());

    let head = index.get("turbo-en-csc-pragmatics-1").unwrap();
    assert_eq!(
        (head.kind, head.format, head.license, head.class),
        (Kind::Model, Format::Zip, Licence::Apache2, LicenceClass::Permissive)
    );
    assert_eq!(head.trained_on.len(), 2);
    assert_eq!(head.files.len(), 1);
    assert_eq!(head.files[0].size, 5_326);
    assert_eq!(head.files[0].sha256.to_string(), "d4e7057f280deacd3cd4228afc342225d1a5ab054e96883057809453178741bc");
    assert!(head.files[0].url.to_string().ends_with("/models-sentiment-925-1/pragmatics-1.zip"));
    assert!(head.licences_to_accept(Policy::default()).is_empty());
}
