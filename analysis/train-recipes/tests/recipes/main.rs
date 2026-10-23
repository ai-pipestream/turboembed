//! The MASC recipes end to end on corpora written in MASC's layout: the
//! split, training, scoring both models on the held-out samples, the
//! gate, the card, the model's directory and the zoo entries of a
//! release; and the part-of-speech recipe's samples on the MASC corpora
//! checked in for the goldens.

/// The bundles the MASC goldens' inputs are written in, read as the
/// `turbo-formats` tests read them.
#[path = "../../../formats/tests/bundle/mod.rs"]
mod bundle;
#[path = "../common/mod.rs"]
mod common;

mod binary;
mod cards;
mod references;
mod skipped;
mod split;
mod tagger;
mod zoo;

use std::path::PathBuf;

use crate::common::SENTENCES;
use turbo_formats::masc::MascCorpus;
use turbo_train_recipes::masc::{self, RELEASE, SPLIT, SplitCorpus};
use turbo_train_recipes::{Recipe, Reference};

/// Whether CI asks for the published models to be present.
fn models_required() -> bool {
    std::env::var("TURBO_TEST_REQUIRE_OPENNLP_MODELS").is_ok_and(|v| v == "1")
}

/// The split of the corpus at `root`, read with its chunk files, as the
/// binary reads it for the chunker, and its digest.
fn split_of(root: &std::path::Path) -> (SplitCorpus, masc::CorpusDigest) {
    let digest = masc::check_corpus(root, RELEASE, true).unwrap();
    let corpus = MascCorpus::read_with_chunks(root).unwrap();
    (SPLIT.apply(corpus, root).unwrap(), digest)
}

/// [`SENTENCES`] with two sentences the recipes' readers skip: one with a
/// token without a tag, which the tagger's reader skips, and one with no
/// Penn token, which both readers skip.
fn with_skipped_sentences() -> Vec<&'static str> {
    let mut pool = SENTENCES.to_vec();
    pool.extend(["Dogs/NNS ran/ home/NN ~./.", "^Not ^tokenized ~^."]);
    pool
}

/// The document's sentences: `first`, then the corpus's sentences.
fn sentences_after(first: &[usize]) -> Vec<usize> {
    first.iter().copied().chain((0..SENTENCES.len()).cycle()).take(common::SENTENCES_PER_DOCUMENT).collect()
}

/// Where a recipe's reference is, or `None` when it is not there and the
/// test may skip. The OpenNLP UD models come from `TURBO_GOLDENS_MODELS`
/// (`cargo run -p turbo-goldens-tool -- fetch-models <dir>` writes them),
/// and are required when `TURBO_TEST_REQUIRE_OPENNLP_MODELS=1`, as CI sets
/// it. The OpenNLP 1.5 tagger, chunker and name finders come from
/// `TURBO_TEST_OPENNLP_MODELS` and are local-only, as in turbo-sequence's
/// tests (their training data is not published, so CI does not fetch
/// them): without that variable they are skipped even when the models are
/// required, and with it a missing file fails.
fn reference_path(recipe: &Recipe) -> Option<PathBuf> {
    let file = recipe.reference_file().unwrap();
    match recipe.reference() {
        Reference::Zoo(_) => {
            let Some(dir) = std::env::var_os("TURBO_GOLDENS_MODELS").map(PathBuf::from) else {
                assert!(!models_required(), "TURBO_GOLDENS_MODELS is not set and TURBO_TEST_REQUIRE_OPENNLP_MODELS=1");
                eprintln!("TURBO_GOLDENS_MODELS is not set: skipping {file}");
                return None;
            };
            let path = dir.join(&file);
            if path.is_file() {
                return Some(path);
            }
            assert!(!models_required(), "{} not found and TURBO_TEST_REQUIRE_OPENNLP_MODELS=1", path.display());
            eprintln!("{} not found: skipping", path.display());
            None
        }
        Reference::Recorded(_) => {
            let Some(dir) = std::env::var_os("TURBO_TEST_OPENNLP_MODELS").map(PathBuf::from) else {
                eprintln!("TURBO_TEST_OPENNLP_MODELS is not set: skipping {file}");
                return None;
            };
            let path = dir.join(&file);
            assert!(path.is_file(), "TURBO_TEST_OPENNLP_MODELS is set and {} is not there", path.display());
            Some(path)
        }
    }
}
