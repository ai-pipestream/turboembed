//! The frequency lexicons in the built-in index, and the misspelling sets
//! spell correction is measured on (#935).

use turbo_zoo::{Format, Index, Kind, Licence, LicenceClass, Policy};

#[test]
fn the_symspell_frequency_lists_are_pinned_and_permissive() {
    let index = Index::builtin();
    for (id, file) in [
        ("symspell-frequency-dictionary-en-82-765", "frequency_dictionary_en_82_765.txt"),
        ("symspell-frequency-bigramdictionary-en-243-342", "frequency_bigramdictionary_en_243_342.txt"),
    ] {
        let a = index.get(id).unwrap();
        assert_eq!(
            (a.kind, a.format, a.license, a.class),
            (Kind::Lexicon, Format::Text, Licence::Mit, LicenceClass::Permissive),
            "{id}"
        );
        let revision = "c239062ae02961df18ab7da1671d01b4388204e0";
        assert_eq!(a.version, revision, "{id}");
        assert_eq!(a.files.len(), 1, "{id}");
        let f = &a.files[0];
        assert_eq!(f.path.as_str(), file, "{id}");
        assert_eq!(
            f.url.as_str(),
            format!("https://raw.githubusercontent.com/wolfgarbe/SymSpell/{revision}/SymSpell/{file}"),
            "{id}"
        );
        assert_eq!(a.trained_on.len(), 1, "{id}");
        assert_eq!(a.trained_on[0].license, Licence::CcBy30, "{id}");
        let notice = a.attribution.as_deref().unwrap();
        assert!(notice.contains(revision), "{id}");
        assert!(notice.contains("Google Books Ngram data"), "{id}");
        assert!(notice.contains("SCOWL"), "{id}");
        // The README names the Ngram data and SCOWL for the unigram list
        // only; the bigram list's source is inferred.
        assert_eq!(notice.contains("unconfirmed"), id.contains("bigram"), "{id}");
        assert!(a.licences_to_accept(Policy::default()).is_empty(), "{id}");
    }
}

#[test]
fn the_symspell_noisy_queries_are_pinned_and_permissive() {
    let index = Index::builtin();
    let a = index.get("symspell-noisy-query-en-1000").unwrap();
    assert_eq!(
        (a.kind, a.format, a.license, a.class),
        (Kind::Dataset, Format::Text, Licence::Mit, LicenceClass::Permissive)
    );
    let revision = "c239062ae02961df18ab7da1671d01b4388204e0";
    assert_eq!(a.version, revision);
    assert!(a.trained_on.is_empty());
    assert_eq!(a.files.len(), 1);
    assert_eq!(
        a.files[0].url.as_str(),
        format!(
            "https://raw.githubusercontent.com/wolfgarbe/SymSpell/{revision}/SymSpell.Benchmark/test_data/noisy_query_en_1000.txt"
        )
    );
    assert!(a.attribution.as_deref().unwrap().contains(revision));
    assert!(a.licences_to_accept(Policy::default()).is_empty());
}

#[test]
fn the_codespell_misspellings_are_pinned_and_share_alike() {
    let index = Index::builtin();
    let a = index.get("codespell-dictionary-en").unwrap();
    assert_eq!(
        (a.kind, a.format, a.license, a.class),
        (Kind::Dataset, Format::Text, Licence::CcBySa30, LicenceClass::ShareAlike)
    );
    let revision = "f185580744421e1d44e30ce795bc27f9f0ad5584";
    assert_eq!(a.version, revision);
    assert_eq!(
        a.files[0].url.as_str(),
        format!(
            "https://raw.githubusercontent.com/codespell-project/codespell/{revision}/codespell_lib/data/dictionary.txt"
        )
    );
    let notice = a.attribution.as_deref().unwrap();
    assert!(notice.contains(revision) && notice.contains("Wikipedia") && notice.contains("CC BY-SA 3.0"));
    // Fetched only after the user accepts CC BY-SA 3.0.
    assert_eq!(a.licences_to_accept(Policy::default()), [turbo_zoo::Requirement::of(Licence::CcBySa30).unwrap()]);
}
