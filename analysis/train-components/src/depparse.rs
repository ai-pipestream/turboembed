//! UD CoNLL-U training using syntactic words, including multi-word-token rows.
use turbo_analysis::ViewId;
use turbo_depparse::{DependencySample, FeedforwardDependencyModel, FeedforwardDependencyTrainer, FeedforwardSettings};
use turbo_formats::conllu::{ConlluDependencySample, Tagset};
use turbo_unicode::WhitespaceMode;

/// Reads separate CoNLL-U sentences, preserving the checked MWT/offset carrier
/// until training derives oracle events. Non-projective trees are skipped by
/// the trainer. Malformed input returns an error with its sentence number.
pub fn train_feedforward_conllu(
    sentences: &[&str],
    tagset: Tagset,
    settings: &FeedforwardSettings,
) -> Result<FeedforwardDependencyModel, String> {
    let samples = sentences
        .iter()
        .enumerate()
        .map(|(i, source)| {
            let sample = ConlluDependencySample::read_sentence(source, tagset, WhitespaceMode::Unicode, ViewId::SOURCE)
                .map_err(|e| format!("sentence {}: {e}", i + 1))?;
            DependencySample::<turbo_analysis_layers::Words>::from_conllu(sample)
                .map_err(|e| format!("sentence {}: {e}", i + 1))
        })
        .collect::<Result<Vec<_>, _>>()?;
    FeedforwardDependencyTrainer::train(&samples, settings).map_err(|e| e.to_string())
}
