//! Native UD training keeps no retired lemma recipe sources.

#[test]
fn native_training_has_no_retired_lemma_sources() {
    use turbo_train_recipes::ud::{Component, UD_RECIPES, UdRecipe, native};
    assert!(!UD_RECIPES.iter().any(|recipe| matches!(recipe.component, Component::Lemmas)));
    let pos = UdRecipe::trains("turbo-es-ud-ancora-2.16-pos-1").unwrap();
    assert!(matches!(native::train(&pos, "invalid CoNLL-U"), Err(turbo_train_recipes::Error::Model(_))));
}
