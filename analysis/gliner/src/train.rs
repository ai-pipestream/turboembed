//! Native fine-tuning of a markerV0 output MLP on frozen GLiNER features.

use std::collections::BTreeSet;

use turbo::safetensors::{Dtype, File};
use turbo_trainer::adamw::{AdamW, Settings};

use crate::{Candidate, Config, ExtractionSchema, Model};

#[derive(Clone, Debug)]
pub struct TrainingInput {
    pub(crate) spans: Vec<f32>,
    pub(crate) prompts: Vec<f32>,
    pub(crate) pairs: Vec<(usize, usize)>,
}

impl Model {
    /// Runs the production forward pass once and retains the output MLP's
    /// inputs. Gold spans are `(start, end, label)`, with an exclusive end.
    pub fn training_input(
        &self,
        words: &[&str],
        schema: &ExtractionSchema,
    ) -> Result<(TrainingInput, Vec<Candidate>), String> {
        let packed = self.pack(words, schema)?;
        let states = self.states(&packed, schema)?;
        let prompts: Vec<usize> = (0..packed.prompts.len()).collect();
        let words: Vec<usize> = (packed.prompts.len()..packed.prompts.len() + packed.words.len()).collect();
        self.heads.training(&states, self.encoder_config().hidden, &prompts, &words, self.config.max_width)
    }

    pub fn apply_output(&mut self, trainer: &OutputTrainer) -> Result<(), String> {
        if self.config.hidden != trainer.hidden || self.config.projection_factor != trainer.factor {
            return Err("gliner output trainer does not match the model".into());
        }
        self.heads.set_output(&trainer.params)
    }
}

pub struct OutputTrainer {
    hidden: usize,
    factor: usize,
    params: Vec<Vec<f32>>,
    base: Vec<Vec<f32>>,
    anchor: f32,
}

impl OutputTrainer {
    pub fn load(config: &Config, bytes: &[u8]) -> Result<Self, String> {
        if !config.out_project_mlp {
            return Err("gliner output training currently requires the markerV0 MLP".into());
        }
        let file = File::parse("gliner heads", bytes).map_err(|e| e.message)?;
        let (h, f) = (config.hidden, config.projection_factor);
        let mut params = Vec::new();
        for (name, shape) in [
            ("span_rep_layer.span_rep_layer.out_project.0.weight", vec![f * h, 2 * h]),
            ("span_rep_layer.span_rep_layer.out_project.0.bias", vec![f * h]),
            ("span_rep_layer.span_rep_layer.out_project.3.weight", vec![h, f * h]),
            ("span_rep_layer.span_rep_layer.out_project.3.bias", vec![h]),
        ] {
            let tensor = file.get(name, Dtype::F32, shape.len()).map_err(|e| e.message)?;
            let expected: Vec<u64> = shape.iter().map(|&x| x as u64).collect();
            if tensor.shape != expected {
                return Err(format!("gliner {name}: output training shape mismatch"));
            }
            params.push(tensor.f32s());
        }
        Ok(Self { hidden: h, factor: f, base: params.clone(), params, anchor: 0.0 })
    }

    pub fn set_anchor(&mut self, weight: f32) -> Result<(), String> {
        if !weight.is_finite() || weight < 0.0 {
            return Err("gliner anchor must be finite and nonnegative".into());
        }
        self.base = self.params.clone();
        self.anchor = weight;
        Ok(())
    }

    pub fn zero_grads(&self) -> Vec<Vec<f32>> {
        self.params.iter().map(|x| vec![0.0; x.len()]).collect()
    }

    pub fn params(&self) -> &[Vec<f32>] {
        &self.params
    }

    pub fn write_heads(&self, original: &[u8]) -> Result<Vec<u8>, String> {
        let file = File::parse("gliner heads", original).map_err(|e| e.message)?;
        let mut out = original.to_vec();
        for (name, values) in [
            "span_rep_layer.span_rep_layer.out_project.0.weight",
            "span_rep_layer.span_rep_layer.out_project.0.bias",
            "span_rep_layer.span_rep_layer.out_project.3.weight",
            "span_rep_layer.span_rep_layer.out_project.3.bias",
        ]
        .into_iter()
        .zip(&self.params)
        {
            let tensor =
                file.get(name, Dtype::F32, if name.ends_with("bias") { 1 } else { 2 }).map_err(|e| e.message)?;
            if values.len() * 4 != tensor.data.len() {
                return Err(format!("gliner {name}: tuned tensor length mismatch"));
            }
            let offset = tensor.data.as_ptr() as usize - original.as_ptr() as usize;
            for (at, value) in values.iter().enumerate() {
                out[offset + at * 4..offset + at * 4 + 4].copy_from_slice(&value.to_le_bytes());
            }
        }
        Ok(out)
    }

    /// Adds one sentence's balanced positive and hard-negative gradients.
    pub fn gradients(
        &self,
        input: &TrainingInput,
        candidates: &[Candidate],
        gold: &[(usize, usize, usize)],
        grads: &mut [Vec<f32>],
    ) -> Result<f64, String> {
        let (h, f, labels) = (self.hidden, self.factor, input.prompts.len() / self.hidden);
        if grads.len() != 4
            || grads.iter().zip(&self.params).any(|(a, b)| a.len() != b.len())
            || input.spans.len() != input.pairs.len() * 2 * h
            || candidates.len() != input.pairs.len() * labels
            || labels == 0
        {
            return Err("gliner output training input or gradient shape mismatch".into());
        }
        let positive: BTreeSet<(usize, usize, usize)> = gold.iter().copied().collect();
        let mut selected: BTreeSet<usize> = input
            .pairs
            .iter()
            .enumerate()
            .filter(|&(_, &(start, end))| (0..labels).any(|label| positive.contains(&(start, end, label))))
            .map(|(i, _)| i)
            .collect();
        if selected.is_empty() {
            return Ok(0.0);
        }
        let want = (selected.len() * 2).max(4);
        let mut ranked: Vec<(f32, usize)> = (0..input.pairs.len())
            .filter(|i| !selected.contains(i))
            .map(|i| {
                let score = candidates[i * labels..(i + 1) * labels].iter().map(|x| x.score).fold(0.0, f32::max);
                (score, i)
            })
            .collect();
        ranked.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        selected.extend(ranked.into_iter().take(want).map(|x| x.1));
        let scale = 1.0 / selected.len() as f64;
        let mut loss = 0.0;
        for index in selected {
            let (start, end) = input.pairs[index];
            let mut active: Vec<usize> = (0..labels).filter(|&label| positive.contains(&(start, end, label))).collect();
            let mut hard: Vec<(f32, usize)> = candidates[index * labels..(index + 1) * labels]
                .iter()
                .map(|x| (x.logit, x.label))
                .filter(|x| !active.contains(&x.1))
                .collect();
            hard.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            active.extend(hard.into_iter().map(|x| x.1).take(4usize.saturating_sub(active.len())));
            let x = &input.spans[index * 2 * h..(index + 1) * 2 * h];
            let mut z = self.params[1].clone();
            for (o, row) in z.iter_mut().enumerate() {
                for (i, &v) in x.iter().enumerate() {
                    *row += self.params[0][o * 2 * h + i] * v;
                }
            }
            let mid: Vec<f32> = z.iter().map(|&x| x.max(0.0)).collect();
            let mut dz = vec![0.0; f * h];
            for label in active {
                let prompt = &input.prompts[label * h..(label + 1) * h];
                let mut logit = 0.0;
                for (o, &p) in prompt.iter().enumerate() {
                    let value =
                        self.params[3][o] + (0..f * h).map(|i| self.params[2][o * f * h + i] * mid[i]).sum::<f32>();
                    logit += value * p;
                }
                let (start, end) = input.pairs[index];
                let positive = positive.contains(&(start, end, label));
                let weight = if positive { 4.0 } else { 1.0 };
                let probability = turbo::head::sigmoid(logit);
                let target = if positive { 1.0 } else { 0.0 };
                loss += f64::from(weight)
                    * f64::from(
                        target * probability.max(1e-7).ln() + (1.0 - target) * (1.0 - probability).max(1e-7).ln(),
                    );
                let d = (probability - target) * weight * scale as f32;
                for (o, &p) in prompt.iter().enumerate() {
                    for i in 0..f * h {
                        grads[2][o * f * h + i] += d * p * mid[i];
                        dz[i] += d * p * self.params[2][o * f * h + i];
                    }
                    grads[3][o] += d * prompt[o];
                }
            }
            for i in 0..f * h {
                let d = if z[i] > 0.0 { dz[i] } else { 0.0 };
                for (j, &v) in x.iter().enumerate() {
                    grads[0][i * 2 * h + j] += d * v;
                }
                grads[1][i] += d;
            }
        }
        let mut total = -loss * scale;
        if self.anchor > 0.0 {
            let mut penalty = 0.0;
            for ((params, base), grads) in self.params.iter().zip(&self.base).zip(grads.iter_mut()) {
                for ((param, base), grad) in params.iter().zip(base).zip(grads) {
                    let delta = param - base;
                    penalty += f64::from(delta * delta);
                    *grad += 2.0 * self.anchor * delta;
                }
            }
            total += f64::from(self.anchor) * penalty;
        }
        Ok(total)
    }

    pub fn step(&mut self, optimizer: &mut AdamW<f32>, grads: &mut [Vec<f32>]) -> Result<f64, String> {
        optimizer.step(&mut self.params, grads, &[true, false, true, false]).map_err(|e| e.message)
    }
}

pub fn output_optimizer(settings: Settings, trainer: &OutputTrainer) -> Result<AdamW<f32>, String> {
    AdamW::new(settings, &trainer.params.iter().map(Vec::len).collect::<Vec<_>>()).map_err(|e| e.message)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (OutputTrainer, TrainingInput, Vec<Candidate>) {
        let params = vec![vec![0.2, -0.1, 0.3, 0.4], vec![0.1, -0.2], vec![0.3, -0.4], vec![0.2]];
        let trainer = OutputTrainer { hidden: 1, factor: 2, base: params.clone(), params, anchor: 0.0 };
        let input = TrainingInput { spans: vec![0.5, -0.3, 0.2, 0.8], prompts: vec![0.7], pairs: vec![(0, 1), (1, 2)] };
        let candidates = vec![
            Candidate { start: 0, end: 1, label: 0, logit: 0.1, score: 0.6 },
            Candidate { start: 1, end: 2, label: 0, logit: 0.2, score: 0.7 },
        ];
        (trainer, input, candidates)
    }

    #[test]
    fn output_gradients_match_finite_differences() {
        let (mut trainer, input, candidates) = fixture();
        trainer.set_anchor(0.25).unwrap();
        for tensor in 0..trainer.params.len() {
            for at in 0..trainer.params[tensor].len() {
                let original = trainer.params[tensor][at];
                let mut grads = trainer.zero_grads();
                let loss = trainer.gradients(&input, &candidates, &[(0, 1, 0)], &mut grads).unwrap();
                assert!(loss.is_finite());
                trainer.params[tensor][at] = original + 0.001;
                let plus = trainer.gradients(&input, &candidates, &[(0, 1, 0)], &mut trainer.zero_grads()).unwrap();
                trainer.params[tensor][at] = original - 0.001;
                let minus = trainer.gradients(&input, &candidates, &[(0, 1, 0)], &mut trainer.zero_grads()).unwrap();
                trainer.params[tensor][at] = original;
                let expected = (plus - minus) / 0.002;
                assert!(
                    (grads[tensor][at] as f64 - expected).abs() < 0.002,
                    "tensor {tensor} at {at}: {} != {expected}",
                    grads[tensor][at]
                );
            }
        }
    }
}
