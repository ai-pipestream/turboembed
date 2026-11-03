//! Native lemma training refuses non-lemma or modified source recipes.
use turbo_train_recipes::ud::UdRecipe;

#[test]
fn native_training_refuses_non_lemma_or_modified_sources_before_reading_samples() {
    use turbo_train_recipes::ud::{UD_RECIPES, native};
    let pos = UdRecipe::trains("turbo-es-ud-ancora-2.16-pos-1").unwrap();
    assert_eq!(
        native::model_id(&pos).unwrap_err().to_string(),
        "model: native lemma training requires a lemma source recipe"
    );
    let mut modified = UdRecipe::trains("turbo-es-ud-ancora-2.16-lemmas-1").unwrap();
    modified.treebank = UD_RECIPES[7].treebank;
    assert_eq!(
        native::model_id(&modified).unwrap_err().to_string(),
        "model: native lemma training requires a canonical UD source recipe"
    );
}
