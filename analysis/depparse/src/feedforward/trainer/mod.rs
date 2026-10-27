mod initialize;
mod optimize;
use super::{Activation, FeatureLayout, FeedforwardContext, FeedforwardDependencyModel};
use crate::{ArcStandardState, DependencySample, Result, error};
use turbo_analysis_layers::WordOrTokenAnchor;

#[derive(Clone, Debug)]
pub struct FeedforwardSettings {
    pub embedding_size: usize,
    pub hidden_size: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub l2: f64,
    pub dropout: f64,
    pub word_cutoff: usize,
    pub seed: i64,
    pub layout: FeatureLayout,
    pub activation: Activation,
}
impl Default for FeedforwardSettings {
    fn default() -> Self {
        Self {
            embedding_size: 50,
            hidden_size: 200,
            epochs: 10,
            batch_size: 256,
            learning_rate: 0.02,
            l2: 1e-8,
            dropout: 0.5,
            word_cutoff: 2,
            seed: 17,
            layout: FeatureLayout::ChenManning18,
            activation: Activation::Cube,
        }
    }
}
impl FeedforwardSettings {
    pub fn check(&self) -> Result<()> {
        if self.embedding_size == 0
            || self.embedding_size > 4096
            || self.hidden_size == 0
            || self.hidden_size > 65536
            || self.epochs == 0
            || self.batch_size == 0
            || !self.learning_rate.is_finite()
            || self.learning_rate <= 0.0
            || !self.l2.is_finite()
            || self.l2 < 0.0
            || !(0.0..1.0).contains(&self.dropout)
        {
            return Err(error("invalid feedforward training settings"));
        }
        self.check_shape(8, 2)
    }
    fn check_shape(&self, rows: usize, outputs: usize) -> Result<()> {
        let e = self.embedding_size as u128;
        let h = self.hidden_size as u128;
        let o = outputs as u128;
        let count = rows as u128 * e + h * self.layout.feature_count() as u128 * e + h + o * h + o;
        if count > 100_000_000 || rows > 2_000_000 || outputs > 100_000 {
            return Err(error("training model exceeds allocation limit"));
        }
        Ok(())
    }
}
pub struct FeedforwardDependencyTrainer;
impl FeedforwardDependencyTrainer {
    pub fn train<A: WordOrTokenAnchor>(
        samples: &[DependencySample<A>],
        settings: &FeedforwardSettings,
    ) -> Result<FeedforwardDependencyModel> {
        Self::train_with_mode(samples, settings, crate::training::TrainingMode::PseudoProjective)
    }
    /// Train plain arc-standard labels from projective samples only.
    /// Non-projective or already encoded trees produce an error.
    pub fn train_projective<A: WordOrTokenAnchor>(
        samples: &[DependencySample<A>],
        settings: &FeedforwardSettings,
    ) -> Result<FeedforwardDependencyModel> {
        Self::train_with_mode(samples, settings, crate::training::TrainingMode::Projective)
    }
    /// Train native SWAP transitions with ordinary labels and no tree filtering.
    pub fn train_swap<A: WordOrTokenAnchor>(
        samples: &[DependencySample<A>],
        settings: &FeedforwardSettings,
    ) -> Result<FeedforwardDependencyModel> {
        Self::train_with_mode(samples, settings, crate::training::TrainingMode::Swap)
    }
    fn train_with_mode<A: WordOrTokenAnchor>(
        samples: &[DependencySample<A>],
        settings: &FeedforwardSettings,
        mode: crate::training::TrainingMode,
    ) -> Result<FeedforwardDependencyModel> {
        settings.check()?;
        let mut model = initialize::initialize(samples, settings, mode)?;
        let mut examples = Vec::new();
        for sample in samples {
            let graph = mode.graph(sample.graph())?;
            let mut state = ArcStandardState::<A>::new(sample.tokens().len())?;
            let words: Vec<_> = sample.tokens().iter().map(String::as_str).collect();
            let tags: Vec<_> = sample.tags().iter().map(String::as_str).collect();
            for transition in mode.transitions(&graph)? {
                let context = FeedforwardContext::extract(&state, &words, &tags, settings.layout)?;
                let outcome = model
                    .transitions
                    .iter()
                    .position(|t| *t == transition)
                    .ok_or_else(|| error("oracle transition missing"))?;
                examples.push((model.feature_ids(&context)?, outcome));
                state.apply(&transition)?;
            }
        }
        if examples.is_empty() {
            return Err(error("no projective training examples"));
        }
        optimize::optimize(&mut model, &examples, settings)?;
        let mut counts = std::collections::BTreeMap::<String, usize>::new();
        for sample in samples {
            for word in sample.tokens() {
                *counts.entry(super::normalize::normalize(word)).or_default() += 1;
            }
        }
        // Cache only after the final update so no stale weights can be used.
        // One million doubles bounds the returned model's cache at 8 MB of values.
        model.precompute_words(&counts.into_iter().collect::<Vec<_>>(), 1_000_000);
        model.label_encoding = mode.encoding();
        Ok(model)
    }
}
