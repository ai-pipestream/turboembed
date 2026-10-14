//! The index the crate carries: canonical, nothing refused, every class
//! the one docs/licences.md gives, and in agreement with the model files
//! the repository checks in.

use std::collections::BTreeSet;
use std::path::Path;

use turbo_zoo::{Component, Format, Index, Kind, Licence, LicenceClass, Policy, verify};

#[test]
fn the_builtin_index_parses_and_is_canonical() {
    let index = Index::builtin();
    assert_eq!(Index::to_canonical_json(index.list()), turbo_zoo::index::BUILTIN);
}

#[test]
fn nothing_in_the_builtin_index_is_refused() {
    let raw: serde_json::Value = serde_json::from_str(turbo_zoo::index::BUILTIN).unwrap();
    let listed = Index::builtin().list().len();
    assert_eq!(raw["artifacts"].as_array().unwrap().len(), listed);
    for a in Index::builtin().list() {
        assert_ne!(a.class, LicenceClass::Refused, "{}", a.id);
    }
}

/// The 28 treebanks of the UD 1.3 models the zoo may fetch, by language
/// and treebank, and their licences at UD r2.16 (docs/licences.md).
const TREEBANKS: &[(&str, &str, Licence)] = &[
    ("af", "afribooms", Licence::CcBySa40),
    ("ca", "ancora", Licence::CcBy40),
    ("da", "ddt", Licence::CcBySa40),
    ("de", "gsd", Licence::CcBySa40),
    ("en", "ewt", Licence::CcBySa40),
    ("es", "gsd", Licence::CcBySa40),
    ("fa", "perdt", Licence::CcBySa40),
    ("fi", "tdt", Licence::CcBySa40),
    ("fr", "gsd", Licence::CcBySa40),
    ("ga", "idt", Licence::CcBySa30),
    ("hr", "set", Licence::CcBySa40),
    ("hy", "bsut", Licence::CcBySa40),
    ("id", "gsd", Licence::CcBySa40),
    ("is", "icepahc", Licence::CcBySa40),
    ("ka", "glc", Licence::CcBySa40),
    ("kk", "ktb", Licence::CcBySa40),
    ("ko", "kaist", Licence::CcBySa40),
    ("lv", "lvtb", Licence::CcBySa40),
    ("nl", "alpino", Licence::CcBySa40),
    ("no", "bokmaal", Licence::CcBySa40),
    ("pt", "gsd", Licence::CcBySa40),
    ("ro", "rrt", Licence::CcBySa40),
    ("ru", "gsd", Licence::CcBySa40),
    ("sk", "snk", Licence::CcBySa40),
    ("sl", "ssj", Licence::CcBySa40),
    ("sr", "set", Licence::CcBySa40),
    ("sv", "talbanken", Licence::CcBySa40),
    ("tr", "boun", Licence::CcBySa40),
];

/// The UD 1.3 treebanks OpenNLP publishes models for whose licences are
/// non-commercial: none of their models may be in the index.
const REFUSED_TREEBANKS: &[(&str, &str)] = &[
    ("bg", "btb"),
    ("cs", "pdtc"),
    ("el", "gdt"),
    ("et", "edt"),
    ("eu", "bdt"),
    ("it", "vit"),
    ("pl", "pdb"),
    ("uk", "iu"),
];

const MODELS: &[(&str, Component)] = &[
    ("sentence", Component::SentenceDetector),
    ("tokens", Component::Tokenizer),
    ("pos", Component::Pos),
    ("lemmas", Component::Lemmatizer),
];

const BASE: &str = "https://archive.apache.org/dist/opennlp/models/ud-models-1.3/";

#[test]
fn the_ud_models_carry_their_treebank_licences() {
    let index = Index::builtin();
    let mut expected = BTreeSet::new();
    for (lang, tb, licence) in TREEBANKS {
        for (model, component) in MODELS {
            let id = format!("opennlp-{lang}-ud-{tb}-{model}-1.3-2.5.4");
            expected.insert(id.clone());
            let a = index.get(&id).unwrap();
            assert_eq!(a.kind, Kind::Model);
            assert_eq!(a.format, Format::OpennlpModel);
            assert_eq!(a.component, Some(*component));
            assert_eq!(a.language.as_ref().unwrap().as_str(), *lang);
            assert_eq!(a.license, Licence::Apache2);
            assert_eq!(a.license_terms.as_str(), format!("{BASE}README"));
            assert_eq!(a.trained_on.len(), 1);
            let t = &a.trained_on[0];
            assert_eq!(t.license, *licence, "{id}");
            assert!(t.name.ends_with(" r2.16"), "{id}");
            assert!(t.license_terms.as_str().contains("/blob/r2.16/README."), "{id}");
            let class = if *licence == Licence::CcBy40 { LicenceClass::Permissive } else { LicenceClass::ShareAlike };
            assert_eq!(a.class, class, "{id}");
            assert_eq!(a.files.len(), 1);
            assert_eq!(a.files[0].path.as_str(), format!("{id}.bin"));
            assert_eq!(a.files[0].url.as_str(), format!("{BASE}{id}.bin"));
            let found: Vec<_> = index.find(lang, *component).map(|a| a.id.as_str()).collect();
            assert_eq!(found, [id.as_str()]);
        }
    }
    let listed: BTreeSet<String> = index.list().iter().map(|a| a.id.to_string()).collect();
    assert_eq!(listed, expected);
    for (lang, tb) in REFUSED_TREEBANKS {
        assert!(!listed.iter().any(|id| id.starts_with(&format!("opennlp-{lang}-ud-{tb}-"))), "{lang}-{tb}");
    }
}

#[test]
fn only_catalan_needs_no_acceptance() {
    let index = Index::builtin();
    for a in index.list() {
        let needs = a.licences_to_accept(Policy::default());
        if a.id.as_str().starts_with("opennlp-ca-ud-ancora-") {
            assert!(needs.is_empty(), "{}", a.id);
        } else if a.id.as_str().starts_with("opennlp-ga-") {
            assert_eq!(needs, [turbo_zoo::Requirement::of(Licence::CcBySa30).unwrap()], "{}", a.id);
        } else {
            assert_eq!(needs, [turbo_zoo::Requirement::of(Licence::CcBySa40).unwrap()], "{}", a.id);
        }
    }
}

/// The OpenNLP models the repository checks in as test fixtures are the
/// bytes the index describes.
#[test]
fn checked_in_models_match_the_index() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let index = Index::builtin();
    let mut seen = 0;
    for dir in ["sentdetect/tests/models", "tokenize/tests/models"] {
        for entry in std::fs::read_dir(root.join(dir)).unwrap() {
            let path = entry.unwrap().path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else { continue };
            let Some(id) = name.strip_suffix(".bin") else { continue };
            let a = index.get(id).unwrap();
            verify(a, &a.files[0], &std::fs::read(&path).unwrap()).unwrap();
            seen += 1;
        }
    }
    assert_eq!(seen, 8);
}

/// Fetches every artifact of the built-in index from its address and
/// checks it: `cargo test -p turbo-zoo -- --ignored`. Needs the network.
#[test]
#[ignore]
#[cfg(feature = "http")]
fn every_builtin_artifact_fetches_and_verifies() {
    let cache = std::env::temp_dir().join(format!("turbo-zoo-builtin-{}", std::process::id()));
    let user: turbo_zoo::UserId = "turbo-zoo-test".parse().unwrap();
    let mut acc = turbo_zoo::Acceptances::default();
    acc.accept(&user, Licence::CcBySa40, 0).unwrap();
    acc.accept(&user, Licence::CcBySa30, 0).unwrap();
    let index = Index::builtin();
    let ids: Vec<String> = index.list().iter().map(|a| a.id.to_string()).collect();
    let zoo = turbo_zoo::Zoo::new(index, cache.clone(), Policy::default(), &turbo_zoo::Http);
    for id in &ids {
        let f = zoo.fetch(id, &user, &acc).unwrap();
        let bytes = std::fs::read(f.single().unwrap()).unwrap();
        assert_eq!(turbo_zoo::Sha256::of(&bytes), f.artifact.files[0].sha256);
        assert_eq!(turbo_zoo::Sha512::of(&bytes), f.artifact.files[0].sha512);
    }
    std::fs::remove_dir_all(&cache).unwrap();
}
