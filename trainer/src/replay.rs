//! Bounded externally streamed warm-start updates for a linear token head.
//! Inference does not use this module.

use turbo::status::{Result, invalid};

use crate::adamw::{AdamW, Settings};
use crate::batch::Batch;
use crate::fit::{Draw, Example};
use crate::tagger::Tagger;

/// Holds only the optimizer state and current reusable gradient batch.
pub struct Replay {
    model: Tagger<f32>,
    adam: AdamW<f32>,
    batch: Batch<Tagger<f32>>,
    gradients: Vec<Vec<f32>>,
    decay: Vec<bool>,
    words: u64,
    steps: u64,
}

impl Replay {
    /// A warm start retains every supplied parameter, without a random reset.
    pub fn new(model: Tagger<f32>, settings: Settings, threads: usize) -> Result<Self> {
        let shapes: Vec<_> = model.params().iter().map(Vec::len).collect();
        let adam = AdamW::new(settings, &shapes)?;
        let batch = Batch::new(threads, 32);
        let gradients = model.zero_grads();
        let decay = model.decays();
        Ok(Self { model, adam, batch, gradients, decay, words: 0, steps: 0 })
    }

    /// Consumes one batch's exact word-weighted gradient, then drops the
    /// borrowed rows before the caller loads the next batch.
    pub fn update(&mut self, examples: &[Example]) -> Result<f64> {
        if examples.is_empty() || examples.len() > 128 {
            return Err(invalid("streamed replay requires one to 128 nonempty rows"));
        }
        let draws: Vec<_> = examples
            .iter()
            .enumerate()
            .map(|(i, e)| (e, Draw { at: self.steps * 128 + i as u64, scale: 1.0 }))
            .collect();
        let (loss, words) = self.batch.gradients(&self.model, &draws, &mut self.gradients)?;
        if words == 0 {
            return Err(invalid("streamed replay refuses a zero-word update"));
        }
        let inv = 1.0 / words as f32;
        self.gradients.iter_mut().flatten().for_each(|g| *g *= inv);
        self.model.step(&mut self.adam, &mut self.gradients, &self.decay)?;
        self.words += words as u64;
        self.steps += 1;
        Ok(loss)
    }

    pub fn steps(&self) -> u64 {
        self.steps
    }
    pub fn words(&self) -> u64 {
        self.words
    }
    pub fn into_head(self) -> Tagger<f32> {
        self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{batch_gradients, rng::Rng};
    use turbo::manifest::{Decode, HeadKind, HeadSpec, Scheme, WordPooling};

    #[test]
    fn one_streamed_update_matches_the_existing_batch_optimizer() {
        let spec = HeadSpec {
            name: "test".into(),
            layer: "turboembed:test".into(),
            kind: HeadKind::TokenTagger,
            labels: ["O", "B-X", "I-X"].map(str::to_owned).to_vec(),
            scheme: Scheme::Bio,
            word_pooling: WordPooling::First,
            artifact: "test".into(),
            tensor_prefix: "test".into(),
            decode: Decode::Argmax,
            adapter: None,
            biaffine: None,
            pos_input: None,
            seen_lookup: None,
            hash_embed: None,
            language: None,
            seen: None,
            layer_mix: None,
            span: None,
            relation: None,
            coref: None,
            license_class: None,
        };
        let model = || Tagger::new(&spec, 2, &mut Rng::new(17)).unwrap();
        let batch = [
            Example { features: vec![0.2, 0.4], labels: vec![0], forms: Vec::new(), language: None },
            Example { features: vec![-0.3, 0.1, 0.8, -0.2], labels: vec![1, 2], forms: Vec::new(), language: None },
        ];
        let settings = Settings {
            lr: 0.00008,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            clip: 5.0,
            warmup: 0,
            total: 1,
        };
        let mut eager = model();
        let mut opt = AdamW::new(settings, &eager.params().iter().map(Vec::len).collect::<Vec<_>>()).unwrap();
        let (mut grads, loss) = batch_gradients(&eager, &batch, 2).unwrap();
        grads.iter_mut().flatten().for_each(|g| *g *= 1.0 / 3.0);
        let decay = eager.decays();
        eager.step(&mut opt, &mut grads, &decay).unwrap();
        let mut stream = Replay::new(model(), settings, 2).unwrap();
        assert_eq!(stream.update(&batch).unwrap(), loss);
        assert_eq!(stream.steps(), 1);
        assert_eq!(stream.words(), 3);
        assert_eq!(stream.into_head().params(), eager.params());
    }
}
