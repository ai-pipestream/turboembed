//! The index the crate carries: canonical, nothing refused, every class
//! the one docs/licences.md gives, and in agreement with the model files
//! the repository checks in. Beside OpenNLP's UD models it lists the models
//! the MASC, UD and language detector recipes train, which
//! turbo-train-recipes checks (`release`), and the permissive datasets
//! (`datasets`).

use std::collections::BTreeSet;

use turbo_zoo::{Component, Format, Index, Kind, Licence, LicenceClass, Policy, verify};

#[path = "builtin/checkpoints.rs"]
mod checkpoints;
#[path = "builtin/components.rs"]
mod components;

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
        let attr = a.attribution.as_deref().unwrap_or("");
        assert!(!attr.trim().is_empty() && !attr.chars().any(char::is_control), "{}", a.id);
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
/// non-commercial. A model trained on one is local-only (docs/licences.md),
/// and the built-in index lists none of them.
const NON_COMMERCIAL_TREEBANKS: &[(&str, &str)] = &[
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

/// The models the MASC recipes train (analysis/train-recipes), listed at
/// their addresses in a release of this repository.
const MASC_MODELS: [&str; 4] = [
    "turbo-en-masc-3.0.0-sentence-1",
    "turbo-en-masc-3.0.0-tokens-1",
    "turbo-en-masc-3.0.0-pos-1",
    "turbo-en-masc-3.0.0-chunks-1",
];

/// The models the UD recipes train (analysis/train-recipes) that passed
/// their gates and may ship, listed at their addresses in another release
/// of this repository.
const UD_RECIPE_MODELS: [&str; 10] = [
    "turbo-es-ud-ancora-2.16-sentence-1",
    "turbo-es-ud-ancora-2.16-tokens-1",
    "turbo-es-ud-ancora-2.16-pos-1",
    "turbo-es-ud-ancora-2.16-lemmas-1",
    "turbo-ca-ud-ancora-2.16-sentence-1",
    "turbo-ca-ud-ancora-2.16-tokens-1",
    "turbo-ca-ud-ancora-2.16-pos-1",
    "turbo-ca-ud-ancora-2.16-lemmas-1",
    "turbo-fi-ud-ftb-2.16-pos-1",
    "turbo-fi-ud-ftb-2.16-lemmas-1",
];

/// The language detector the Tatoeba recipe trains (analysis/train-recipes),
/// listed with the ids of its training sentences at their addresses in a
/// third release of this repository.
const TATOEBA_MODEL: &str = "turbo-mul-tatoeba-langdetect-1";
/// The language detector the web recipe trains on Tatoeba's sentences and
/// FineWeb-2 and FineWeb text, listed with the ids of its training
/// sentences and documents at their addresses in a fourth release.
const WEB_MODEL: &str = "turbo-mul-tatoeba-fineweb-langdetect-1";

/// A model this repository's recipes train.
fn trained_here(id: &str) -> bool {
    MASC_MODELS.contains(&id) || UD_RECIPE_MODELS.contains(&id) || id == TATOEBA_MODEL || id == WEB_MODEL
}

/// The published neural checkpoints the index lists (docs/licences.md,
/// "Neural annotator checkpoints"): permissive, so nothing to accept.
pub(crate) const CHECKPOINTS: [&str; 8] = [
    "fastino-gliner2-base-v1",
    "microsoft-deberta-v3-large-tokenizer",
    "urchade-gliner-small-v2.1",
    "microsoft-deberta-v3-small-tokenizer",
    "urchade-gliner-multi-v2.1",
    "microsoft-mdeberta-v3-base-tokenizer",
    "segment-any-text-sat-3l",
    "turbo-en-csc-pragmatics-1",
];

/// The share-alike neural checkpoints the index lists: fetched after the user
/// accepts their training data's CC BY-SA terms, and never bundled.
pub(crate) const SHARE_ALIKE_CHECKPOINTS: [&str; 2] =
    ["cross-encoder-nli-minilm2-l6-h768", "deepset-roberta-base-squad2"];

/// The local-only checkpoints the index lists: fetched only for a user
/// who has accepted their licence.
pub(crate) const LOCAL_CHECKPOINTS: [&str; 1] = ["jackboyla-glirel-large-v0"];

/// The frequency lexicons the spell checker loads (#935): MIT lists, so
/// permissive and nothing to accept. The README gives CC BY Ngram data
/// as the unigram list's source; the bigram list's is unconfirmed.
const LEXICONS: [&str; 2] =
    ["symspell-frequency-dictionary-en-82-765", "symspell-frequency-bigramdictionary-en-243-342"];

/// The spelling evaluation set the spell checker is measured on (#935):
/// SymSpell's MIT benchmark queries, so permissive and nothing to accept.
const EVAL_SETS: [&str; 1] = ["symspell-noisy-query-en-1000"];

/// The human misspelling set spell correction is also measured on (#935):
/// codespell's list, derived from English Wikipedia under CC BY-SA 3.0, so
/// fetched only after the user accepts that licence.
const SHARE_ALIKE_EVAL_SETS: [&str; 1] = ["codespell-dictionary-en"];

/// The retrieval checkpoints the SPLATE study lists (#913): permissive
/// model licences over MS MARCO, so local-only through their training
/// data and fetched only after accepting MS MARCO's terms.
const RETRIEVAL_CHECKPOINTS: [&str; 2] = ["opensearch-neural-sparse-encoding-v1", "colbert-ir-colbertv2.0"];

/// The permissive training datasets in the built-in index (#945): MASC,
/// the four settled CC BY 4.0 UD 2.16 treebanks, the Tatoeba export,
/// Civil Comments, and GoEmotions (filtered split and raw annotations).
const DATASETS: [&str; 10] = [
    "csc-conversational-sarcasm-2024",
    "masc-3.0.0",
    "ud-spanish-ancora-r2.16",
    "ud-catalan-ancora-r2.16",
    "ud-finnish-ftb-r2.16",
    "ud-italian-markit-r2.16",
    "tatoeba-2026-10-10",
    "civil-comments",
    "goemotions",
    "goemotions-raw",
];

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
            assert!(a.attribution.as_deref().unwrap().contains(&t.name), "{id}");
            assert_eq!(a.files.len(), 1);
            assert_eq!(a.files[0].path.as_str(), format!("{id}.bin"));
            assert_eq!(a.files[0].url.as_str(), format!("{BASE}{id}.bin"));
            let found: Vec<_> =
                index.find(lang, *component).map(|a| a.id.as_str()).filter(|id| id.starts_with("opennlp-")).collect();
            assert_eq!(found, [id.as_str()]);
        }
    }
    expected.extend(MASC_MODELS.map(str::to_owned));
    expected.extend(UD_RECIPE_MODELS.map(str::to_owned));
    expected.insert(TATOEBA_MODEL.to_owned());
    expected.insert(WEB_MODEL.to_owned());
    expected.extend(CHECKPOINTS.map(str::to_owned));
    expected.extend(SHARE_ALIKE_CHECKPOINTS.map(str::to_owned));
    expected.extend(LOCAL_CHECKPOINTS.map(str::to_owned));
    expected.extend(LEXICONS.map(str::to_owned));
    expected.extend(EVAL_SETS.map(str::to_owned));
    expected.extend(SHARE_ALIKE_EVAL_SETS.map(str::to_owned));
    expected.extend(RETRIEVAL_CHECKPOINTS.map(str::to_owned));
    expected.extend(DATASETS.map(str::to_owned));
    let listed: BTreeSet<String> = index.list().iter().map(|a| a.id.to_string()).collect();
    assert_eq!(listed, expected);
    for (lang, tb) in NON_COMMERCIAL_TREEBANKS {
        assert!(!listed.iter().any(|id| id.starts_with(&format!("opennlp-{lang}-ud-{tb}-"))), "{lang}-{tb}");
    }
}

#[test]
fn only_catalan_and_the_models_trained_here_need_no_acceptance() {
    let index = Index::builtin();
    for a in index.list() {
        let needs = a.licences_to_accept(Policy::default());
        if a.id.as_str().starts_with("opennlp-ca-ud-ancora-")
            || trained_here(a.id.as_str())
            || CHECKPOINTS.contains(&a.id.as_str())
            || LEXICONS.contains(&a.id.as_str())
            || EVAL_SETS.contains(&a.id.as_str())
            || DATASETS.contains(&a.id.as_str())
        {
            assert!(needs.is_empty(), "{}", a.id);
        } else if a.id.as_str() == SHARE_ALIKE_CHECKPOINTS[0] {
            assert_eq!(
                needs,
                [
                    turbo_zoo::Requirement::of(Licence::CcBySa30).unwrap(),
                    turbo_zoo::Requirement::of(Licence::CcBySa40).unwrap(),
                ],
                "{}",
                a.id
            );
        } else if RETRIEVAL_CHECKPOINTS.contains(&a.id.as_str()) {
            assert_eq!(
                needs,
                [turbo_zoo::Requirement {
                    licence: Licence::RestrictedPublic,
                    text: "https://microsoft.github.io/msmarco/".into(),
                }],
                "{}",
                a.id
            );
        } else if LOCAL_CHECKPOINTS.contains(&a.id.as_str()) {
            assert_eq!(needs, [turbo_zoo::Requirement::of(Licence::CcByNcSa40).unwrap()], "{}", a.id);
        } else if a.id.as_str().starts_with("opennlp-ga-") || SHARE_ALIKE_EVAL_SETS.contains(&a.id.as_str()) {
            assert_eq!(needs, [turbo_zoo::Requirement::of(Licence::CcBySa30).unwrap()], "{}", a.id);
        } else {
            assert_eq!(needs, [turbo_zoo::Requirement::of(Licence::CcBySa40).unwrap()], "{}", a.id);
        }
    }
}

/// The UD models the sentence detector and tokenizer tests run, which
/// `turbo-goldens fetch-models` downloads (they are trained on CC BY-SA
/// treebanks and not checked in), are the bytes the index describes. They
/// are read from the directory in `TURBO_GOLDENS_MODELS`, or
/// `target/turbo-goldens/models` under the repository root; without them
/// the test says so and passes, unless
/// `TURBO_TEST_REQUIRE_OPENNLP_MODELS=1`, which makes a missing model fail
/// it, as CI sets it.
#[test]
fn fetched_test_models_match_the_index() {
    const MODELS: [&str; 8] = [
        "opennlp-de-ud-gsd-sentence-1.3-2.5.4",
        "opennlp-en-ud-ewt-sentence-1.3-2.5.4",
        "opennlp-pt-ud-gsd-sentence-1.3-2.5.4",
        "opennlp-en-ud-ewt-tokens-1.3-2.5.4",
        "opennlp-de-ud-gsd-tokens-1.3-2.5.4",
        "opennlp-es-ud-gsd-tokens-1.3-2.5.4",
        "opennlp-fr-ud-gsd-tokens-1.3-2.5.4",
        "opennlp-pt-ud-gsd-tokens-1.3-2.5.4",
    ];
    let index = Index::builtin();
    for id in MODELS {
        let Some(path) = turbo_goldens::published_model(&format!("{id}.bin")) else { continue };
        let a = index.get(id).unwrap();
        verify(a, &a.files[0], &std::fs::read(&path).unwrap()).unwrap();
    }
}

/// Fetches every artifact of the built-in index from its address and
/// checks it: `cargo test -p turbo-zoo -- --ignored`. Needs the network.
/// The addresses of the models trained here answer only once the `models`,
/// `models-ud` and `models-tatoeba` workflows have published their releases
/// (analysis/train-recipes/README.md, "The models releases"); until then
/// this test fails on them.
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
        assert_eq!(f.files.len(), f.artifact.files.len(), "{id}");
        for (path, file) in f.files.iter().zip(&f.artifact.files) {
            let bytes = std::fs::read(path).unwrap();
            assert_eq!(turbo_zoo::Sha256::of(&bytes), file.sha256, "{id}");
            assert_eq!(turbo_zoo::Sha512::of(&bytes), file.sha512, "{id}");
        }
    }
    std::fs::remove_dir_all(&cache).unwrap();
}
