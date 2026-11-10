//! UD CoNLL-U training using syntactic words, including multi-word-token rows.
use turbo_analysis::ViewId;
pub use turbo_depparse::{
    Activation, FeatureLayout, FeedforwardDependencyModel, FeedforwardEpoch, FeedforwardSettings,
};
use turbo_depparse::{DependencySample, FeedforwardDependencyTrainer};
use turbo_formats::conllu::ConlluDependencySample;
pub use turbo_formats::conllu::Tagset;

/// Reads separate CoNLL-U sentences, preserving the checked MWT/offset carrier
/// until training derives oracle events. Non-projective trees are pseudo-projectivized
/// by the trainer. Malformed input returns an error with its sentence number.
pub fn train_feedforward_conllu(
    sentences: &[&str],
    tagset: Tagset,
    settings: &FeedforwardSettings,
) -> Result<FeedforwardDependencyModel, String> {
    train_feedforward_conllu_observed(sentences, tagset, settings, &mut |_| true)
}

/// [`train_feedforward_conllu`], with `observe` called after each epoch;
/// training stops with an error when it answers false.
pub fn train_feedforward_conllu_observed(
    sentences: &[&str],
    tagset: Tagset,
    settings: &FeedforwardSettings,
    observe: &mut dyn FnMut(&FeedforwardEpoch) -> bool,
) -> Result<FeedforwardDependencyModel, String> {
    let samples = sentences
        .iter()
        .enumerate()
        .map(|(i, source)| {
            let sample = ConlluDependencySample::read_sentence(source, tagset, ViewId::SOURCE)
                .map_err(|e| format!("sentence {}: {e}", i + 1))?;
            DependencySample::<turbo_analysis_layers::Words>::from_conllu(sample)
                .map_err(|e| format!("sentence {}: {e}", i + 1))
        })
        .collect::<Result<Vec<_>, _>>()?;
    FeedforwardDependencyTrainer::train_observed(&samples, settings, observe).map_err(|e| e.to_string())
}
