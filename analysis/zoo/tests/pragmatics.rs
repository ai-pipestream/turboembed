use turbo_zoo::{Format, Index, Kind, Licence, LicenceClass, Policy};

#[test]
fn csc_is_pinned_and_permissive_and_no_csc_head_is_registered() {
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

    // The models-sentiment-925-1 head was evaluated on a split that leaked
    // scenarios, and its retrained replacement does not beat the majority
    // baseline, so no CSC pragmatics head is registered (#925).
    assert!(index.list().iter().all(|entry| !entry.id.to_string().contains("csc-pragmatics")));
}
