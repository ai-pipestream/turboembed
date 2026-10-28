use super::compare::{ulps_apart, update_parameter_distance};
use super::*;
use turbo_zoo::LicenceClass;

#[test]
fn ulps_count_the_numbers_between_two_values_across_zero() {
    assert_eq!(ulps_apart(1.0, 1.0), 0);
    assert_eq!(ulps_apart(1.0, f64::from_bits(1.0f64.to_bits() + 3)), 3);
    assert_eq!(ulps_apart(0.0, -0.0), 0);
    let tiny = f64::from_bits(2);
    assert_eq!(ulps_apart(tiny, -tiny), 4);
    assert_eq!(ulps_apart(-1.0, -f64::from_bits(1.0f64.to_bits() + 1)), 1);
}

fn comparison_with(parameters: Option<std::result::Result<ParameterDistance, String>>) -> Comparison {
    let digest = EntryDigest { len: 1, sha256: String::new() };
    Comparison {
        model: String::new(),
        events: 0,
        event_hash: Agreement::Same(1),
        model_entry: Agreement::Differ { published: digest.clone(), trained: digest },
        first_difference: Some(0),
        parameters,
        manifest: Agreement::Same(String::new()),
        entry_names: Agreement::Same(Vec::new()),
    }
}

#[test]
fn a_nan_distance_is_no_agreement_and_the_report_shows_every_manifest_line() {
    let close = ParameterDistance { total: 2, differing: 1, max_ulps: 1, max_abs: 1e-15 };
    assert!(comparison_with(Some(Ok(close.clone()))).agrees_within(1e-12));
    let nan = ParameterDistance { max_abs: f64::NAN, ..close.clone() };
    assert!(!comparison_with(Some(Ok(nan))).agrees_within(1e-12));
    assert!(!comparison_with(Some(Err("shapes".to_owned()))).agrees_within(1e-12));
    assert!(!comparison_with(None).agrees_within(1e-12));
    let mut c = comparison_with(Some(Ok(close)));
    c.manifest = Agreement::Differ { published: "a=1\nb=2\nc=3".to_owned(), trained: "a=1\nb=x".to_owned() };
    let report = c.report();
    assert!(report.contains("    published: b=2\n    trained:   b=x\n"), "{report}");
    assert!(report.contains("    published: c=3\n    trained:   (no line)\n"), "{report}");
}

#[test]
fn agrees_within_rejects_negative_and_nan_tolerances_even_when_model_entries_match() {
    let digest = EntryDigest { len: 1, sha256: String::new() };
    let identical = Comparison {
        model: String::new(),
        events: 0,
        event_hash: Agreement::Same(1),
        model_entry: Agreement::Same(digest),
        first_difference: None,
        parameters: None,
        manifest: Agreement::Same(String::new()),
        entry_names: Agreement::Same(Vec::new()),
    };
    assert!(identical.agrees_within(0.0));
    assert!(!identical.agrees_within(-1.0));
    assert!(!identical.agrees_within(f64::NAN));
}

#[test]
fn equal_bit_nan_parameters_record_nan_distance_and_do_not_agree() {
    let mut d = ParameterDistance { total: 0, differing: 0, max_ulps: 0, max_abs: 0.0 };
    update_parameter_distance(&mut d, f64::NAN, f64::NAN);
    assert_eq!(d.total, 1);
    assert_eq!(d.differing, 1);
    assert!(d.max_abs.is_nan());
    assert!(!comparison_with(Some(Ok(d))).agrees_within(1e-12));
}

#[test]
fn the_datasets_make_a_valid_share_alike_index() {
    let index = datasets();
    assert_eq!(index.list().len(), 5);
    let ewt = index.get("ud-english-ewt-r2.16-train").unwrap();
    assert_eq!(ewt.class, LicenceClass::ShareAlike);
    for a in index.list() {
        assert_eq!(a.class, a.computed_class(), "{}", a.id);
    }
    // The class follows the licence: a CC BY 4.0 treebank is permissive.
    let permissive = Treebank { licence: Licence::CcBy40, ..ENGLISH_EWT }.dataset();
    assert_eq!(permissive.class, LicenceClass::Permissive);
    assert_eq!(ewt.files[0].size, 14203025);
    assert_eq!(
        ewt.files[0].url.as_str(),
        "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/r2.16/en_ewt-ud-train.conllu"
    );
    assert_eq!(
        ewt.license_terms.as_str(),
        "https://github.com/UniversalDependencies/UD_English-EWT/blob/r2.16/README.md"
    );
    // Fetching one without an acceptance is refused before any transfer.
    let zoo = Zoo::new(datasets(), PathBuf::from("/nonexistent"), Policy::default(), &turbo_zoo::Offline);
    let user: UserId = "someone".parse().unwrap();
    assert!(zoo.permit("ud-english-ewt-r2.16-train", &user, &Acceptances::default()).is_err());
}
