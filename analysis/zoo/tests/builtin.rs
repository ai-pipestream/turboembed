//! The catalogue preserves neural checkpoints, learned heads, datasets and their licences.

use std::collections::BTreeSet;

use turbo_zoo::{Http, Index, Licence, LicenceClass, Policy};

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

/// Default-tier sentiment, emotion, and incivility heads trained with
/// `turbo-trainer` on `answerdotai/ModernBERT-base` (#1732).
const MODERNBERT_SENTIMENT_HEADS: [&str; 4] = [
    "turbo-en-masc-3.0.0-modernbert-base-polarity-head-1",
    "turbo-en-goemotions-modernbert-base-head-1",
    "turbo-en-civil-comments-modernbert-base-head-1",
    "turbo-en-civil-comments-modernbert-base-context-head-1",
];

/// Legacy `all-MiniLM-L6-v2` and `bge-small-en-v1.5` sentiment heads (#924,
/// #926, #888), reclassified to `CLASS_LOCAL_ONLY` via MS MARCO in their
/// encoder training lineages (#1732).
const LEGACY_MSMARCO_SENTIMENT_HEADS: [&str; 7] = [
    "turbo-en-masc-3.0.0-aspect-sentiment-1",
    "turbo-en-goemotions-minilm-l6-v2-head-1",
    "turbo-en-goemotions-bge-small-en-v1.5-head-1",
    "turbo-en-civil-comments-minilm-l6-v2-head-1",
    "turbo-en-civil-comments-minilm-l6-v2-context-head-1",
    "turbo-en-civil-comments-bge-small-en-v1.5-head-1",
    "turbo-en-civil-comments-bge-small-en-v1.5-context-head-1",
];

const OPEN_SPAN_HEAD: &str = "turbo-en-wnut17-modernbert-base-l12-open-span-1";

/// The MASC 3.0.0 Penn POS head over ModernBERT-base (#1649), assembled
/// into one bundle with its encoder.
const MASC_POS: &str = "turbo-en-masc-pos-modernbert-base-1";

/// A permissive model this repository's recipes train.
fn trained_here_permissive(id: &str) -> bool {
    id == OPEN_SPAN_HEAD || id == MASC_POS || MODERNBERT_SENTIMENT_HEADS.contains(&id)
}

/// The published neural checkpoints the index lists (docs/licences.md,
/// "Neural annotator checkpoints"): permissive, so nothing to accept.
pub(crate) const CHECKPOINTS: [&str; 4] = [
    "microsoft-deberta-v3-large-tokenizer",
    "microsoft-deberta-v3-small-tokenizer",
    "microsoft-mdeberta-v3-base-tokenizer",
    "segment-any-text-sat-3l",
];

/// The Apache-2.0 checkpoints whose training data is unreleased or
/// LLM-labelled web text (Kristian, 2026-10-28): local-only, fetched from
/// their publisher after the user accepts the terms on its card.
pub(crate) const UNRELEASED_DATA_CHECKPOINTS: [(&str, &str); 3] = [
    (
        "fastino-gliner2-base-v1",
        "https://huggingface.co/fastino/gliner2-base-v1/blob/f9634218e53580c56edf0de97ca1a7d3f1c2354e/README.md",
    ),
    ("urchade-gliner-small-v2.1", "https://huggingface.co/datasets/urchade/pile-mistral-v0.1"),
    ("urchade-gliner-multi-v2.1", "https://huggingface.co/datasets/urchade/pile-mistral-v0.1"),
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

/// The permissive training and evaluation datasets in the built-in index (#945, #890, #925):
/// CSC, MASC, the four settled CC BY 4.0 UD 2.16 treebanks, the Tatoeba export,
/// Civil Comments, GoEmotions (filtered split and raw annotations), and Toxic Spans.
const DATASETS: [&str; 12] = [
    "csc-conversational-sarcasm-2024",
    "masc-3.0.0",
    "ud-spanish-ancora-r2.16",
    "ud-catalan-ancora-r2.16",
    "ud-finnish-ftb-r2.16",
    "ud-italian-markit-r2.16",
    "tatoeba-2026-10-10",
    "civil-comments",
    "civil-comments-wilds",
    "goemotions",
    "goemotions-raw",
    "toxic-spans",
];

#[test]
fn the_builtin_catalogue_preserves_supported_artifacts() {
    let index = Index::builtin();
    let mut expected = BTreeSet::from([OPEN_SPAN_HEAD.to_owned(), MASC_POS.to_owned()]);
    expected.extend(MODERNBERT_SENTIMENT_HEADS.map(str::to_owned));
    expected.extend(LEGACY_MSMARCO_SENTIMENT_HEADS.map(str::to_owned));
    expected.extend(CHECKPOINTS.map(str::to_owned));
    expected.extend(SHARE_ALIKE_CHECKPOINTS.map(str::to_owned));
    expected.extend(LOCAL_CHECKPOINTS.map(str::to_owned));
    expected.extend(UNRELEASED_DATA_CHECKPOINTS.map(|(id, _)| id.to_owned()));
    expected.extend(LEXICONS.map(str::to_owned));
    expected.extend(EVAL_SETS.map(str::to_owned));
    expected.extend(SHARE_ALIKE_EVAL_SETS.map(str::to_owned));
    expected.extend(RETRIEVAL_CHECKPOINTS.map(str::to_owned));
    expected.extend(DATASETS.map(str::to_owned));
    let listed: BTreeSet<String> = index.list().iter().map(|a| a.id.to_string()).collect();
    assert_eq!(listed, expected);
}

#[test]
fn the_remaining_artifacts_keep_their_licence_requirements() {
    let index = Index::builtin();
    for a in index.list() {
        let needs = a.licences_to_accept(Policy::default());
        if trained_here_permissive(a.id.as_str())
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
        } else if RETRIEVAL_CHECKPOINTS.contains(&a.id.as_str())
            || LEGACY_MSMARCO_SENTIMENT_HEADS.contains(&a.id.as_str())
        {
            assert_eq!(
                needs,
                [turbo_zoo::Requirement {
                    licence: Licence::RestrictedPublic,
                    text: "https://microsoft.github.io/msmarco/".into(),
                }],
                "{}",
                a.id
            );
        } else if let Some((_, terms)) = UNRELEASED_DATA_CHECKPOINTS.iter().find(|(id, _)| *id == a.id.as_str()) {
            assert_eq!(
                needs,
                [turbo_zoo::Requirement { licence: Licence::RestrictedPublic, text: (*terms).into() }],
                "{}",
                a.id
            );
        } else if LOCAL_CHECKPOINTS.contains(&a.id.as_str()) {
            assert_eq!(needs, [turbo_zoo::Requirement::of(Licence::CcByNcSa40).unwrap()], "{}", a.id);
        } else if SHARE_ALIKE_EVAL_SETS.contains(&a.id.as_str()) {
            assert_eq!(needs, [turbo_zoo::Requirement::of(Licence::CcBySa30).unwrap()], "{}", a.id);
        } else {
            assert_eq!(needs, [turbo_zoo::Requirement::of(Licence::CcBySa40).unwrap()], "{}", a.id);
        }
    }
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
    let transport = Http::new();
    let zoo = turbo_zoo::Zoo::new(index, cache.clone(), Policy::default(), &transport);
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
