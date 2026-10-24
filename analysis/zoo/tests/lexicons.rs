//! The frequency lexicons in the built-in index (#935).

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
        assert!(a.licences_to_accept(Policy::default()).is_empty(), "{id}");
    }
}
