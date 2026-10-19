//! Training sample and hyperparameter settings for [`super::BilstmPosTrainer`].

use crate::neural::NeuralError;

/// A training sentence annotated with primary POS tags and optional auxiliary XPOS and morphological feature tags.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiTaskSample {
    /// Sentence tokens.
    pub sentence: Vec<String>,
    /// Primary POS tags (one per token).
    pub tags: Vec<String>,
    /// Optional fine-grained XPOS tags (one per token).
    pub xpos: Option<Vec<String>>,
    /// Optional morphological feature bundles (one per token).
    pub feats: Option<Vec<String>>,
}

impl MultiTaskSample {
    /// Creates a new `MultiTaskSample`, validating that all present tag arrays match `sentence.len()`.
    pub fn new(
        sentence: impl IntoIterator<Item = impl Into<String>>,
        tags: impl IntoIterator<Item = impl Into<String>>,
        xpos: Option<Vec<String>>,
        feats: Option<Vec<String>>,
    ) -> Result<Self, NeuralError> {
        let sample = Self {
            sentence: sentence.into_iter().map(Into::into).collect(),
            tags: tags.into_iter().map(Into::into).collect(),
            xpos,
            feats,
        };
        sample.validate()?;
        Ok(sample)
    }

    /// Validates that `tags`, `xpos`, and `feats` all match `sentence.len()`.
    pub fn validate(&self) -> Result<(), NeuralError> {
        if self.sentence.len() != self.tags.len() {
            return Err(NeuralError::InvalidArgument("tags must have the same length as sentence".into()));
        }
        if let Some(ref x) = self.xpos
            && x.len() != self.sentence.len()
        {
            return Err(NeuralError::InvalidArgument("xpos must have the same length as sentence".into()));
        }
        if let Some(ref f) = self.feats
            && f.len() != self.sentence.len()
        {
            return Err(NeuralError::InvalidArgument("feats must have the same length as sentence".into()));
        }
        Ok(())
    }
}

/// Hyperparameters for [`super::BilstmPosTrainer`], matching OpenNLP's `BilstmPOSTrainer.Settings`.
#[derive(Clone, Debug, PartialEq)]
pub struct BilstmSettings {
    pub word_dim: usize,
    pub char_dim: usize,
    pub char_hidden: usize,
    pub word_hidden: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub max_grad_norm: f64,
    pub dropout: f64,
    pub word_cutoff: usize,
    pub max_word_length: usize,
    pub seed: i64,
    pub threads: usize,
    pub word_dropout: f64,
    pub learning_rate_half_life: usize,
    pub crf: bool,
    pub encoder_layers: usize,
    pub pretrained_dropout: f64,
    pub encoder_dropout: f64,
    pub aux_loss_weight: f64,
    pub pretrained_tuning: f64,
    pub pretrained_adapter: bool,
}

impl BilstmSettings {
    /// Full 22-parameter constructor with validation matching OpenNLP's `BilstmPOSTrainer.Settings`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        word_dim: i32,
        char_dim: i32,
        char_hidden: i32,
        word_hidden: i32,
        epochs: i32,
        batch_size: i32,
        learning_rate: f64,
        max_grad_norm: f64,
        dropout: f64,
        word_cutoff: i32,
        max_word_length: i32,
        seed: i64,
        threads: i32,
        word_dropout: f64,
        learning_rate_half_life: i32,
        crf: bool,
        encoder_layers: i32,
        pretrained_dropout: f64,
        encoder_dropout: f64,
        aux_loss_weight: f64,
        pretrained_tuning: f64,
        pretrained_adapter: bool,
    ) -> Result<Self, NeuralError> {
        if word_dim <= 0 || char_dim <= 0 || char_hidden <= 0 || word_hidden <= 0 {
            return Err(NeuralError::InvalidArgument("all dimensions must be > 0".into()));
        }
        if epochs <= 0 || batch_size <= 0 {
            return Err(NeuralError::InvalidArgument("epochs and batchSize must be > 0".into()));
        }
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(NeuralError::InvalidArgument("learningRate must be finite and > 0".into()));
        }
        if !max_grad_norm.is_finite() || max_grad_norm <= 0.0 {
            return Err(NeuralError::InvalidArgument("clipNorm must be finite and > 0".into()));
        }
        if !dropout.is_finite() || !(0.0..1.0).contains(&dropout) {
            return Err(NeuralError::InvalidArgument("dropout must be in [0, 1)".into()));
        }
        if !word_dropout.is_finite() || !(0.0..1.0).contains(&word_dropout) {
            return Err(NeuralError::InvalidArgument("wordDropout must be in [0, 1)".into()));
        }
        if !pretrained_dropout.is_finite() || !(0.0..1.0).contains(&pretrained_dropout) {
            return Err(NeuralError::InvalidArgument("pretrainedDropout must be in [0, 1)".into()));
        }
        if !encoder_dropout.is_finite() || !(0.0..1.0).contains(&encoder_dropout) {
            return Err(NeuralError::InvalidArgument("encoderDropout must be in [0, 1)".into()));
        }
        if !aux_loss_weight.is_finite() || aux_loss_weight < 0.0 {
            return Err(NeuralError::InvalidArgument("auxLossWeight must be finite and >= 0".into()));
        }
        if !pretrained_tuning.is_finite() || pretrained_tuning < 0.0 {
            return Err(NeuralError::InvalidArgument("pretrainedTuning must be finite and >= 0".into()));
        }
        if pretrained_adapter && pretrained_tuning > 0.0 {
            return Err(NeuralError::InvalidArgument(
                "pretrainedAdapter and pretrainedTuning must not be combined".into(),
            ));
        }
        if word_cutoff <= 0 || max_word_length <= 0 {
            return Err(NeuralError::InvalidArgument("wordCutoff and maxWordLength must be > 0".into()));
        }
        if threads <= 0 {
            return Err(NeuralError::InvalidArgument("threads must be >= 1".into()));
        }
        if learning_rate_half_life < 0 {
            return Err(NeuralError::InvalidArgument("learningRateHalfLife must be >= 0".into()));
        }
        if !(1..=2).contains(&encoder_layers) {
            return Err(NeuralError::InvalidArgument("encoderLayers must be 1 or 2".into()));
        }
        Ok(Self {
            word_dim: word_dim as usize,
            char_dim: char_dim as usize,
            char_hidden: char_hidden as usize,
            word_hidden: word_hidden as usize,
            epochs: epochs as usize,
            batch_size: batch_size as usize,
            learning_rate,
            max_grad_norm,
            dropout,
            word_cutoff: word_cutoff as usize,
            max_word_length: max_word_length as usize,
            seed,
            threads: threads as usize,
            word_dropout,
            learning_rate_half_life: learning_rate_half_life as usize,
            crf,
            encoder_layers: encoder_layers as usize,
            pretrained_dropout,
            encoder_dropout,
            aux_loss_weight,
            pretrained_tuning,
            pretrained_adapter,
        })
    }

    /// Default hyperparameters matching `BilstmPOSTrainer.Settings.defaults()`.
    pub fn defaults() -> Self {
        Self::new(100, 25, 50, 128, 20, 16, 1e-3, 5.0, 0.33, 2, 40, 17, 1, 0.0, 0, false, 1, 0.0, 0.0, 1.0, 0.0, false)
            .unwrap()
    }
}
