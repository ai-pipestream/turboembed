//! BiLSTM(-CRF) trainer (`BilstmPosTrainer`, `BilstmSettings`, `MultiTaskSample`).

mod context;
mod gradients;
mod settings;
mod step;
#[cfg(test)]
mod tests;

pub use context::{TrainingContext, Worker};
pub use settings::{BilstmSettings, MultiTaskSample};

use std::collections::HashMap;

use crate::neural::bilstm_model::BilstmPosModel;
use crate::neural::bilstm_tagger::BilstmPosTagger;
use crate::neural::lstm::LstmLayer;
use crate::neural::{JavaRng, NeuralError, PosSample};

impl TrainingContext {
    pub fn testing_word_embeddings_mut(&mut self) -> &mut [Vec<f64>] {
        self.optimizer.params_mut(self.idx_word_emb)
    }

    pub fn testing_char_embeddings_mut(&mut self) -> &mut [Vec<f64>] {
        self.optimizer.params_mut(self.idx_char_emb)
    }

    pub fn testing_output_weights_mut(&mut self) -> &mut [Vec<f64>] {
        self.optimizer.params_mut(self.idx_out_w)
    }

    pub fn testing_output_bias_mut(&mut self) -> &mut [f64] {
        &mut self.optimizer.params_mut(self.idx_out_b)[0]
    }

    pub fn testing_char_forward_mut(&mut self) -> &mut LstmLayer {
        &mut self.char_forward
    }

    pub fn testing_char_backward_mut(&mut self) -> &mut LstmLayer {
        &mut self.char_backward
    }

    pub fn testing_word_forward_mut(&mut self) -> &mut LstmLayer {
        &mut self.word_forward
    }

    pub fn testing_word_backward_mut(&mut self) -> &mut LstmLayer {
        &mut self.word_backward
    }

    pub fn testing_word_forward2_mut(&mut self) -> Option<&mut LstmLayer> {
        self.word_forward2.as_mut()
    }

    pub fn testing_word_backward2_mut(&mut self) -> Option<&mut LstmLayer> {
        self.word_backward2.as_mut()
    }

    pub fn testing_transition_weights_mut(&mut self) -> Option<&mut [Vec<f64>]> {
        self.idx_crf_tr.map(|idx| self.optimizer.params_mut(idx))
    }

    pub fn testing_start_weights_mut(&mut self) -> Option<&mut [f64]> {
        self.idx_crf_st.map(|idx| &mut self.optimizer.params_mut(idx)[0][..])
    }

    pub fn testing_end_weights_mut(&mut self) -> Option<&mut [f64]> {
        self.idx_crf_en.map(|idx| &mut self.optimizer.params_mut(idx)[0][..])
    }

    pub fn testing_xpos_weights_mut(&mut self) -> Option<&mut [Vec<f64>]> {
        self.idx_xpos_w.map(|idx| self.optimizer.params_mut(idx))
    }

    pub fn testing_xpos_bias_mut(&mut self) -> Option<&mut [f64]> {
        self.idx_xpos_b.map(|idx| &mut self.optimizer.params_mut(idx)[0][..])
    }

    pub fn testing_feats_weights_mut(&mut self) -> Option<&mut [Vec<f64>]> {
        self.idx_feats_w.map(|idx| self.optimizer.params_mut(idx))
    }

    pub fn testing_feats_bias_mut(&mut self) -> Option<&mut [f64]> {
        self.idx_feats_b.map(|idx| &mut self.optimizer.params_mut(idx)[0][..])
    }

    pub fn testing_pretrained_trainable_mut(&mut self) -> Option<&mut [Vec<f64>]> {
        self.pretrained_trainable.as_deref_mut()
    }

    pub fn testing_adapter_weights_mut(&mut self) -> Option<&mut [Vec<f64>]> {
        self.idx_adapter_w.map(|idx| self.optimizer.params_mut(idx))
    }

    pub fn testing_adapter_bias_mut(&mut self) -> Option<&mut [f64]> {
        self.idx_adapter_b.map(|idx| &mut self.optimizer.params_mut(idx)[0][..])
    }
}

/// Trainer for [`BilstmPosModel`], ported from `BilstmPOSTrainer`.
pub struct BilstmPosTrainer;

impl BilstmPosTrainer {
    /// Trains a [`BilstmPosModel`] on single-task [`PosSample`]s without pretrained word vectors.
    pub fn train(samples: &[PosSample], settings: BilstmSettings) -> Result<BilstmPosModel, NeuralError> {
        Self::train_with_vectors(samples, settings, None::<fn(&str) -> Option<Vec<f32>>>, None, None)
    }

    /// Trains a [`BilstmPosModel`] on single-task [`PosSample`]s with optional pretrained word vectors, lexicon, and validation samples.
    pub fn train_with_vectors<F>(
        samples: &[PosSample],
        settings: BilstmSettings,
        pretrained_vectors: Option<F>,
        additional_vocabulary: Option<&[String]>,
        validation_samples: Option<&[PosSample]>,
    ) -> Result<BilstmPosModel, NeuralError>
    where
        F: Fn(&str) -> Option<Vec<f32>>,
    {
        let mut corpus = Vec::with_capacity(samples.len());
        for s in samples {
            if s.sentence.len() != s.tags.len() {
                return Err(NeuralError::InvalidArgument("tags must have the same length as sentence".into()));
            }
            if !s.sentence.is_empty() {
                corpus.push(MultiTaskSample::new(s.sentence.clone(), s.tags.clone(), None, None)?);
            }
        }
        Self::train_multi_task(&corpus, settings, pretrained_vectors, additional_vocabulary, validation_samples)
    }

    /// Trains a [`BilstmPosModel`] on [`MultiTaskSample`]s with optional auxiliary XPOS/FEATS heads, pretrained vectors, and early stopping.
    pub fn train_multi_task<F>(
        samples: &[MultiTaskSample],
        settings: BilstmSettings,
        pretrained_vectors: Option<F>,
        additional_vocabulary: Option<&[String]>,
        validation_samples: Option<&[PosSample]>,
    ) -> Result<BilstmPosModel, NeuralError>
    where
        F: Fn(&str) -> Option<Vec<f32>>,
    {
        for s in samples {
            s.validate()?;
        }
        let corpus: Vec<MultiTaskSample> = samples.iter().filter(|s| !s.sentence.is_empty()).cloned().collect();
        if corpus.is_empty() {
            return Err(NeuralError::InvalidArgument("Training stream produced no samples".into()));
        }
        for s in validation_samples.unwrap_or(&[]) {
            if s.sentence.len() != s.tags.len() {
                return Err(NeuralError::InvalidArgument("tags must have the same length as sentence".into()));
            }
        }
        let validation: Vec<PosSample> =
            validation_samples.unwrap_or(&[]).iter().filter(|s| !s.sentence.is_empty()).cloned().collect();

        let mut ctx = TrainingContext::build(&corpus, settings.clone(), pretrained_vectors, additional_vocabulary)?;
        let mut worker = ctx.new_worker();
        let mut touched_tuning_rows: HashMap<usize, Vec<f64>> = HashMap::new();

        let mut order: Vec<usize> = (0..corpus.len()).collect();
        let mut shuffle_rng = JavaRng::new(settings.seed);
        let mut step_num = 0i32;

        let mut best_model: Option<BilstmPosModel> = None;
        let mut best_dev_acc = -1.0f64;

        for epoch in 1..=settings.epochs {
            let lr = if settings.learning_rate_half_life > 0 {
                settings.learning_rate * 0.5_f64.powf((epoch - 1) as f64 / settings.learning_rate_half_life as f64)
            } else {
                settings.learning_rate
            };
            shuffle_rng.shuffle(&mut order);

            let mut start = 0;
            while start < order.len() {
                let end = (start + settings.batch_size).min(order.len());
                ctx.optimizer.zero();
                touched_tuning_rows.clear();

                let mut batch_tokens = 0usize;
                for (pos, &sample_idx) in order.iter().enumerate().take(end).skip(start) {
                    let sample = &corpus[sample_idx];
                    batch_tokens += sample.sentence.len();
                    ctx.reset_worker(&mut worker);
                    let mask_seed =
                        settings.seed.wrapping_add((epoch as i64).wrapping_mul(1_000_003)).wrapping_add(pos as i64);
                    let mut sentence_rng = JavaRng::new(mask_seed);
                    ctx.sentence_gradients(sample, &mut sentence_rng, &mut worker)?;
                    ctx.absorb_worker(&worker, &mut touched_tuning_rows);
                }

                if batch_tokens > 0 {
                    step_num += 1;
                    ctx.step_batch(1.0, lr, step_num, &mut touched_tuning_rows)?;
                }
                start = end;
            }

            if !validation.is_empty() {
                let candidate = ctx.to_model()?;
                let acc = evaluate(&candidate, &validation)?;
                if acc > best_dev_acc {
                    best_dev_acc = acc;
                    best_model = Some(candidate);
                }
            }
        }

        match best_model {
            Some(m) => Ok(m),
            None => ctx.to_model(),
        }
    }
}

fn evaluate(model: &BilstmPosModel, dev: &[PosSample]) -> Result<f64, NeuralError> {
    let tagger = BilstmPosTagger::new(model.clone());
    let mut correct = 0usize;
    let mut total = 0usize;
    for s in dev {
        if s.sentence.len() != s.tags.len() {
            return Err(NeuralError::InvalidArgument("tags must have the same length as sentence".into()));
        }
        let pred = tagger.tag(&s.sentence)?;
        for i in 0..pred.len() {
            if pred[i] == s.tags[i] {
                correct += 1;
            }
            total += 1;
        }
    }
    Ok(if total == 0 { 0.0 } else { correct as f64 / total as f64 })
}
