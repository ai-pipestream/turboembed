//! Feedforward trainer hyperparameters (`FeedforwardPOSTrainer.Settings`).

use crate::neural::NeuralError;

/// Hyperparameters for [`FeedforwardPosTrainer`](super::FeedforwardPosTrainer), matching `FeedforwardPOSTrainer.Settings`.
#[derive(Clone, Debug, PartialEq)]
pub struct FeedforwardSettings {
    pub embedding_size: usize,
    pub hidden_size: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub l2: f64,
    pub dropout: f64,
    pub word_cutoff: usize,
    pub suffix_cutoff: usize,
    pub seed: i64,
}

impl FeedforwardSettings {
    /// Creates and validates `FeedforwardSettings`, matching `FeedforwardPOSTrainer.Settings`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        embedding_size: i32,
        hidden_size: i32,
        epochs: i32,
        batch_size: i32,
        learning_rate: f64,
        l2: f64,
        dropout: f64,
        word_cutoff: i32,
        suffix_cutoff: i32,
        seed: i64,
    ) -> Result<Self, NeuralError> {
        if embedding_size <= 0 {
            return Err(NeuralError::InvalidArgument(format!("embeddingSize must be positive: {embedding_size}")));
        }
        if hidden_size <= 0 {
            return Err(NeuralError::InvalidArgument(format!("hiddenSize must be positive: {hidden_size}")));
        }
        if epochs <= 0 {
            return Err(NeuralError::InvalidArgument(format!("epochs must be positive: {epochs}")));
        }
        if batch_size <= 0 {
            return Err(NeuralError::InvalidArgument(format!("batchSize must be positive: {batch_size}")));
        }
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(NeuralError::InvalidArgument(format!(
                "learningRate must be finite and positive: {learning_rate}"
            )));
        }
        if !l2.is_finite() || l2 < 0.0 {
            return Err(NeuralError::InvalidArgument(format!("l2 must be finite and non-negative: {l2}")));
        }
        if !dropout.is_finite() || !(0.0..1.0).contains(&dropout) {
            return Err(NeuralError::InvalidArgument(format!("dropout must be finite and in [0, 1): {dropout}")));
        }
        if word_cutoff < 0 {
            return Err(NeuralError::InvalidArgument(format!("wordCutoff must be non-negative: {word_cutoff}")));
        }
        if suffix_cutoff < 0 {
            return Err(NeuralError::InvalidArgument(format!("suffixCutoff must be non-negative: {suffix_cutoff}")));
        }
        Ok(Self {
            embedding_size: embedding_size as usize,
            hidden_size: hidden_size as usize,
            epochs: epochs as usize,
            batch_size: batch_size as usize,
            learning_rate,
            l2,
            dropout,
            word_cutoff: word_cutoff as usize,
            suffix_cutoff: suffix_cutoff as usize,
            seed,
        })
    }

    /// Default hyperparameters matching `FeedforwardPOSTrainer.Settings.defaults()`.
    pub fn defaults() -> Self {
        Self::new(32, 128, 10, 64, 0.05, 1e-6, 0.2, 2, 2, 42).unwrap()
    }
}
