//! The training datasets in the built-in index (#945): Civil Comments,
//! GoEmotions (filtered split and raw rater annotations), MASC 3.0.0,
//! the four settled CC BY 4.0 UD 2.16 treebanks, and the Tatoeba export
//! of 2026-10-10.

use turbo_zoo::{Format, Index, Kind, Licence, LicenceClass, Policy};

#[test]
fn civil_comments_is_pinned_cc0_and_permissive() {
    let index = Index::builtin();
    let a = index.get("civil-comments").unwrap();
    assert_eq!(
        (a.kind, a.format, a.license, a.class),
        (Kind::Dataset, Format::Parquet, Licence::Cc0, LicenceClass::Permissive)
    );
    assert_eq!(a.component, None);
    assert_eq!(a.language.as_ref().map(|l| l.as_str()), Some("en"));
    let revision = "f2970eb3a55777454c94069077cc8d9b5866312d";
    assert_eq!(a.version, revision);
    assert_eq!(
        a.license_terms.as_str(),
        format!("https://huggingface.co/datasets/google/civil_comments/blob/{revision}/README.md")
    );
    assert!(a.trained_on.is_empty());
    let notice = a.attribution.as_deref().unwrap();
    assert!(notice.contains("google/civil_comments") && notice.contains(revision) && notice.contains("CC0 1.0"));
    let paths: Vec<&str> = a.files.iter().map(|f| f.path.as_str()).collect();
    assert_eq!(
        paths,
        [
            "README.md",
            "data/train-00000-of-00002.parquet",
            "data/train-00001-of-00002.parquet",
            "data/validation-00000-of-00001.parquet",
            "data/test-00000-of-00001.parquet",
        ]
    );
    for f in &a.files {
        assert_eq!(
            f.url.as_str(),
            format!("https://huggingface.co/datasets/google/civil_comments/resolve/{revision}/{}", f.path.as_str())
        );
    }
    assert!(a.licences_to_accept(Policy::default()).is_empty());

    let ts_rev = "beaf170538df8275c3447b2e26070f2f82e55272";
    let ts = index.get("toxic-spans").unwrap();
    assert_eq!(
        (ts.kind, ts.format, ts.license, ts.class),
        (Kind::Dataset, Format::Csv, Licence::Cc0, LicenceClass::Permissive)
    );
    assert_eq!(ts.component, None);
    assert_eq!(ts.language.as_ref().map(|l| l.as_str()), Some("en"));
    assert_eq!(ts.version, ts_rev);
    assert_eq!(ts.license_terms.as_str(), format!("https://github.com/ipavlopoulos/toxic_spans/blob/{ts_rev}/LICENSE"));
    let ts_paths: Vec<&str> = ts.files.iter().map(|f| f.path.as_str()).collect();
    assert_eq!(
        ts_paths,
        [
            "LICENSE",
            "README.md",
            "ACL2022/data/toxic_spans.csv",
            "data/comments.csv",
            "data/spans.csv",
            "data/annotations.csv",
        ]
    );
    assert!(ts.licences_to_accept(Policy::default()).is_empty());
}

#[test]
fn goemotions_filtered_and_raw_are_pinned_cc_by_4_0_and_permissive() {
    let index = Index::builtin();
    let revision = "7de3624754e2b0f2f505ae496db3ba194bb40be9";
    let terms = format!("https://github.com/google-research/google-research/blob/{revision}/README.md");

    let filtered = index.get("goemotions").unwrap();
    assert_eq!(
        (filtered.kind, filtered.format, filtered.license, filtered.class),
        (Kind::Dataset, Format::Tsv, Licence::CcBy40, LicenceClass::Permissive)
    );
    assert_eq!(filtered.component, None);
    assert_eq!(filtered.language.as_ref().map(|l| l.as_str()), Some("en"));
    assert_eq!(filtered.version, revision);
    assert_eq!(filtered.license_terms.as_str(), terms);
    assert!(filtered.trained_on.is_empty());
    let notice = filtered.attribution.as_deref().unwrap();
    assert!(
        notice.contains("filtered train/dev/test split") && notice.contains(revision) && notice.contains("CC BY 4.0")
    );
    let paths: Vec<&str> = filtered.files.iter().map(|f| f.path.as_str()).collect();
    assert_eq!(
        paths,
        [
            "README.md",
            "data/emotions.txt",
            "data/ekman_mapping.json",
            "data/sentiment_mapping.json",
            "data/train.tsv",
            "data/dev.tsv",
            "data/test.tsv",
        ]
    );
    for f in &filtered.files {
        assert_eq!(
            f.url.as_str(),
            format!(
                "https://raw.githubusercontent.com/google-research/google-research/{revision}/goemotions/{}",
                f.path.as_str()
            )
        );
    }
    assert!(filtered.licences_to_accept(Policy::default()).is_empty());

    let raw = index.get("goemotions-raw").unwrap();
    assert_eq!(
        (raw.kind, raw.format, raw.license, raw.class),
        (Kind::Dataset, Format::Csv, Licence::CcBy40, LicenceClass::Permissive)
    );
    assert_eq!(raw.component, None);
    assert_eq!(raw.language.as_ref().map(|l| l.as_str()), Some("en"));
    assert_eq!(raw.version, revision);
    assert_eq!(raw.license_terms.as_str(), terms);
    assert!(raw.trained_on.is_empty());
    let raw_notice = raw.attribution.as_deref().unwrap();
    assert!(
        raw_notice.contains("raw rater annotations")
            && raw_notice.contains(revision)
            && raw_notice.contains("CC BY 4.0")
    );
    let raw_paths: Vec<&str> = raw.files.iter().map(|f| f.path.as_str()).collect();
    assert_eq!(raw_paths, ["goemotions_1.csv", "goemotions_2.csv", "goemotions_3.csv"]);
    for f in &raw.files {
        assert_eq!(
            f.url.as_str(),
            format!("https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/{}", f.path.as_str())
        );
    }
    assert!(raw.licences_to_accept(Policy::default()).is_empty());
}

#[test]
fn recipe_datasets_are_pinned_and_permissive() {
    let index = Index::builtin();

    let masc = index.get("masc-3.0.0").unwrap();
    assert_eq!(
        (masc.kind, masc.format, masc.license, masc.class),
        (Kind::Dataset, Format::Zip, Licence::CcBy30Us, LicenceClass::Permissive)
    );
    assert_eq!(masc.language.as_ref().map(|l| l.as_str()), Some("en"));
    assert_eq!(masc.version, "3.0.0");
    assert_eq!(masc.files.len(), 1);
    assert_eq!(masc.files[0].path.as_str(), "MASC-3.0.0.zip");
    assert!(masc.licences_to_accept(Policy::default()).is_empty());

    for (id, repo, lang, tb) in [
        ("ud-spanish-ancora-r2.16", "UD_Spanish-AnCora", "es", "ancora"),
        ("ud-catalan-ancora-r2.16", "UD_Catalan-AnCora", "ca", "ancora"),
        ("ud-finnish-ftb-r2.16", "UD_Finnish-FTB", "fi", "ftb"),
        ("ud-italian-markit-r2.16", "UD_Italian-MarkIT", "it", "markit"),
    ] {
        let a = index.get(id).unwrap();
        assert_eq!(
            (a.kind, a.format, a.license, a.class),
            (Kind::Dataset, Format::Conllu, Licence::CcBy40, LicenceClass::Permissive),
            "{id}"
        );
        assert_eq!(a.language.as_ref().map(|l| l.as_str()), Some(lang), "{id}");
        assert_eq!(a.version, "r2.16", "{id}");
        let paths: Vec<&str> = a.files.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(
            paths,
            [
                "LICENSE.txt",
                "README.md",
                &format!("{lang}_{tb}-ud-train.conllu"),
                &format!("{lang}_{tb}-ud-dev.conllu"),
                &format!("{lang}_{tb}-ud-test.conllu"),
            ],
            "{id}"
        );
        for f in &a.files {
            assert_eq!(
                f.url.as_str(),
                format!("https://raw.githubusercontent.com/UniversalDependencies/{repo}/r2.16/{}", f.path.as_str()),
                "{id}"
            );
        }
        assert!(a.licences_to_accept(Policy::default()).is_empty(), "{id}");
    }
    assert!(index.get("ud-portuguese-porttinari-r2.16").is_err(), "Porttinari text rights are not settled");

    let tatoeba = index.get("tatoeba-2026-10-10").unwrap();
    assert_eq!(
        (tatoeba.kind, tatoeba.format, tatoeba.license, tatoeba.class),
        (Kind::Dataset, Format::TarBz2, Licence::CcBy20Fr, LicenceClass::Permissive)
    );
    assert_eq!(tatoeba.language.as_ref().map(|l| l.as_str()), Some("mul"));
    assert_eq!(tatoeba.version, "2026-10-10");
    let paths: Vec<&str> = tatoeba.files.iter().map(|f| f.path.as_str()).collect();
    assert_eq!(paths, ["sentences_detailed.tar.bz2", "sentences_CC0.tar.bz2"]);
    for f in &tatoeba.files {
        assert_eq!(
            f.url.as_str(),
            format!(
                "https://github.com/ai-pipestream/turboembed/releases/download/data-tatoeba-2026-10-10/{}",
                f.path.as_str()
            )
        );
    }
    assert!(tatoeba.licences_to_accept(Policy::default()).is_empty());
}
