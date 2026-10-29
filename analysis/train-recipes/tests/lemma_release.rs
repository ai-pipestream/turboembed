//! Published recipe ids retain their original on-disk class format.
use turbo_train_recipes::ud::UdRecipe;

#[test]
fn native_training_refuses_non_lemma_or_modified_sources_before_reading_samples() {
    use turbo_train_recipes::ud::{UD_RECIPES, native};
    let pos = UdRecipe::trains("turbo-es-ud-ancora-2.16-pos-1").unwrap();
    assert!(matches!(native::train(&pos, "invalid CoNLL-U"), Err(turbo_train_recipes::Error::Model(_))));
    let mut modified = UdRecipe::trains("turbo-es-ud-ancora-2.16-lemmas-1").unwrap();
    modified.treebank = UD_RECIPES[7].treebank;
    assert!(matches!(native::train(&modified, "invalid CoNLL-U"), Err(turbo_train_recipes::Error::Model(_))));
}
