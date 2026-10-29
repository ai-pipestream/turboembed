//! Corpus validation and deterministic MASC split behavior.

#[path = "../common/mod.rs"]
mod common;

mod split;

use turbo_formats::masc::MascCorpus;
use turbo_train_recipes::masc::{self, RELEASE, SPLIT, SplitCorpus};

/// The split of the corpus at `root`, read with its chunk files, as the
/// binary reads it for the chunker, and its digest.
fn split_of(root: &std::path::Path) -> (SplitCorpus, masc::CorpusDigest) {
    let digest = masc::check_corpus(root, RELEASE, true).unwrap();
    let corpus = MascCorpus::read_with_chunks(root).unwrap();
    (SPLIT.apply(corpus, root).unwrap(), digest)
}
