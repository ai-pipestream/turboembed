//! Sentences the recipes' readers skip, and how many they may.

use crate::common::{SENTENCES, temp_dir, write_corpus};
use turbo_formats::masc::MascCorpus;
use turbo_train_recipes::chunk::CHUNKS;
use turbo_train_recipes::masc::SPLIT;
use turbo_train_recipes::{Error, Recipe};

/// A corpus read without its chunk files has no chunk sample: the
/// chunker's reader refuses it, and so does the check made before
/// training.
#[test]
fn the_chunker_refuses_a_corpus_read_without_its_chunk_files() {
    let root = temp_dir("chunk-files");
    write_corpus(&root, 20, SENTENCES);
    let split = SPLIT.apply(MascCorpus::read(&root).unwrap(), &root).unwrap();
    let err = CHUNKS.held_out(&split.held_out).unwrap_err();
    assert!(matches!(err, Error::Masc(turbo_formats::masc::MascError::NoChunks)), "{err:?}");
    assert!(Recipe::Chunk(CHUNKS).check_data(&split, false, "reference", None).is_err());
    assert!(CHUNKS.train(&split.train).is_err());
    std::fs::remove_dir_all(&root).unwrap();
}
