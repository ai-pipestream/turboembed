//! Native fine-tuning of markerV0 output and boundary heads on frozen GLiNER features.

use std::collections::BTreeSet;

use turbo::safetensors::{Dtype, File};
use turbo_trainer::adamw::{AdamW, Settings};

use crate::features::{Boundary, Prompt};
use crate::{Candidate, Config, ExtractionSchema, Model};

#[derive(Clone, Debug)]
pub struct TrainingInput {
    pub(crate) words: Vec<f32>,
    pub(crate) spans: Vec<f32>,
    pub(crate) prompts: Vec<f32>,
    pub(crate) prompt_inputs: Vec<f32>,
    pub(crate) pairs: Vec<(usize, usize)>,
}

impl Model {
    /// Runs the production forward pass once and retains the output and
    /// boundary inputs. Gold spans are `(start, end, label)`, with an exclusive end.
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
        Self::load_with_heads(config, bytes, false, false)
    }

    pub fn load_with_boundaries(config: &Config, bytes: &[u8], boundaries: bool) -> Result<Self, String> {
        Self::load_with_heads(config, bytes, boundaries, false)
    }

    pub fn load_with_heads(config: &Config, bytes: &[u8], boundaries: bool, prompt: bool) -> Result<Self, String> {
        let file = File::parse("gliner heads", bytes).map_err(|e| e.message)?;
        let (h, f) = (config.hidden, config.projection_factor);
        let mut specs: Vec<(String, Vec<usize>)> = if config.out_project_mlp {
            vec![
                ("span_rep_layer.span_rep_layer.out_project.0.weight".into(), vec![f * h, 2 * h]),
                ("span_rep_layer.span_rep_layer.out_project.0.bias".into(), vec![f * h]),
                ("span_rep_layer.span_rep_layer.out_project.3.weight".into(), vec![h, f * h]),
                ("span_rep_layer.span_rep_layer.out_project.3.bias".into(), vec![h]),
            ]
        } else {
            vec![
                ("span_rep_layer.span_rep_layer.out_project.weight".into(), vec![h, 2 * h]),
                ("span_rep_layer.span_rep_layer.out_project.bias".into(), vec![h]),
            ]
        };
        if (boundaries || prompt) && config.token_level {
            return Err("gliner token-level output training is not supported".into());
        }
        if boundaries {
            for stem in ["project_start", "project_end"] {
                let stem = format!("span_rep_layer.span_rep_layer.{stem}");
                specs.extend([
                    (format!("{stem}.0.weight"), vec![f * h, h]),
                    (format!("{stem}.0.bias"), vec![f * h]),
                    (format!("{stem}.3.weight"), vec![h, f * h]),
                    (format!("{stem}.3.bias"), vec![h]),
                ]);
            }
        }
        if prompt {
            if !config.prompt_projection {
                return Err("gliner prompt training requires a prompt projection".into());
            }
            for stem in ["0", "3"] {
                specs.extend([
                    (
                        format!("prompt_rep_layer.{stem}.weight"),
                        if stem == "0" { vec![f * h, h] } else { vec![h, f * h] },
                    ),
                    (format!("prompt_rep_layer.{stem}.bias"), if stem == "0" { vec![f * h] } else { vec![h] }),
                ]);
            }
        }
        let mut params = Vec::new();
        for (name, shape) in specs {
            let tensor = file.get(&name, Dtype::F32, shape.len()).map_err(|e| e.message)?;
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

    fn boundary_offset(&self) -> Option<usize> {
        match self.params.len() {
            10 | 14 => Some(2),
            12 | 16 => Some(4),
            _ => None,
        }
    }

    fn prompt_offset(&self) -> Option<usize> {
        match self.params.len() {
            8 | 14 | 16 => Some(self.params.len() - 4),
            _ => None,
        }
    }

    pub fn write_heads(&self, original: &[u8]) -> Result<Vec<u8>, String> {
        let file = File::parse("gliner heads", original).map_err(|e| e.message)?;
        let mut names: Vec<String> = if matches!(self.params.len(), 2 | 10 | 14) {
            vec![
                "span_rep_layer.span_rep_layer.out_project.weight".into(),
                "span_rep_layer.span_rep_layer.out_project.bias".into(),
            ]
        } else {
            vec![
                "span_rep_layer.span_rep_layer.out_project.0.weight".into(),
                "span_rep_layer.span_rep_layer.out_project.0.bias".into(),
                "span_rep_layer.span_rep_layer.out_project.3.weight".into(),
                "span_rep_layer.span_rep_layer.out_project.3.bias".into(),
            ]
        };
        if self.boundary_offset().is_some() {
            for stem in ["project_start", "project_end"] {
                for suffix in ["0.weight", "0.bias", "3.weight", "3.bias"] {
                    names.push(format!("span_rep_layer.span_rep_layer.{stem}.{suffix}"));
                }
            }
        }
        if self.prompt_offset().is_some() {
            for suffix in ["0.weight", "0.bias", "3.weight", "3.bias"] {
                names.push(format!("prompt_rep_layer.{suffix}"));
            }
        }
        let mut out = original.to_vec();
        for (name, values) in names.into_iter().zip(&self.params) {
            let tensor =
                file.get(&name, Dtype::F32, if name.ends_with("bias") { 1 } else { 2 }).map_err(|e| e.message)?;
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
        let boundary = self.boundary_offset();
        let prompt_offset = self.prompt_offset();
        let valid_words = boundary.is_none()
            || (input.words.len().is_multiple_of(h)
                && input.pairs.iter().all(|&(start, end)| start < end && end <= input.words.len() / h));
        if !matches!(grads.len(), 2 | 4 | 8 | 10 | 12 | 14 | 16)
            || grads.iter().zip(&self.params).any(|(a, b)| a.len() != b.len())
            || input.spans.len() != input.pairs.len() * 2 * h
            || candidates.len() != input.pairs.len() * labels
            || labels == 0
            || !valid_words
            || (prompt_offset.is_some() && input.prompt_inputs.len() != input.prompts.len())
        {
            return Err("gliner output training input or gradient shape mismatch".into());
        }
        let n = if boundary.is_some() { input.words.len() / h } else { 0 };
        let boundaries = boundary.map(|offset| {
            (
                Boundary::forward(&self.params[offset..], &input.words, n, h, f),
                Boundary::forward(&self.params[offset + 4..], &input.words, n, h, f),
            )
        });
        let prompt_features =
            prompt_offset.map(|offset| Prompt::forward(&self.params[offset..], &input.prompt_inputs, h, f));
        let prompt_values = prompt_features.as_ref().map_or(&input.prompts, |(_, values)| values);
        let mut dprompt = if prompt_offset.is_some() { vec![0.0; labels * h] } else { Vec::new() };
        let linear = matches!(self.params.len(), 2 | 10 | 14);
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
        let (mut loss, mut dstarts, mut dends) = (0.0, vec![0.0; n * h], vec![0.0; n * h]);
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
            let joined;
            let x = if let Some((starts, ends)) = &boundaries {
                let mut values = Vec::with_capacity(2 * h);
                values.extend_from_slice(&starts.out[start * h..(start + 1) * h]);
                values.extend_from_slice(&ends.out[(end - 1) * h..end * h]);
                joined = values;
                &joined
            } else {
                &input.spans[index * 2 * h..(index + 1) * 2 * h]
            };
            let (z, mid) = if linear {
                (Vec::new(), x.to_vec())
            } else {
                let mut z = self.params[1].clone();
                for (o, row) in z.iter_mut().enumerate() {
                    for (i, &v) in x.iter().enumerate() {
                        *row += self.params[0][o * 2 * h + i] * v;
                    }
                }
                let mid: Vec<f32> = z.iter().map(|&x| x.max(0.0)).collect();
                (z, mid)
            };
            let mut dz = if linear { Vec::new() } else { vec![0.0; f * h] };
            let mut dx = if boundary.is_some() { vec![0.0; 2 * h] } else { Vec::new() };
            let (wout, bout, weight_index, bias_index) = if linear {
                (&self.params[0], &self.params[1], 0, 1)
            } else {
                (&self.params[2], &self.params[3], 2, 3)
            };
            for label in active {
                let prompt = &prompt_values[label * h..(label + 1) * h];
                let mut logit = 0.0;
                for (o, &p) in prompt.iter().enumerate() {
                    let value = bout[o] + (0..mid.len()).map(|i| wout[o * mid.len() + i] * mid[i]).sum::<f32>();
                    logit += value * p;
                }
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
                    for i in 0..mid.len() {
                        grads[weight_index][o * mid.len() + i] += d * p * mid[i];
                        if !linear {
                            dz[i] += d * p * wout[o * mid.len() + i];
                        } else if boundary.is_some() {
                            dx[i] += d * p * wout[o * mid.len() + i];
                        }
                    }
                    grads[bias_index][o] += d * p;
                    if prompt_offset.is_some() {
                        let value = bout[o] + (0..mid.len()).map(|i| wout[o * mid.len() + i] * mid[i]).sum::<f32>();
                        dprompt[label * h + o] += d * value;
                    }
                }
            }
            if !linear {
                for i in 0..f * h {
                    let d = if z[i] > 0.0 { dz[i] } else { 0.0 };
                    for (j, &v) in x.iter().enumerate() {
                        grads[0][i * 2 * h + j] += d * v;
                        if boundary.is_some() {
                            dx[j] += self.params[0][i * 2 * h + j] * d;
                        }
                    }
                    grads[1][i] += d;
                }
            }
            if boundary.is_some() {
                dstarts[start * h..(start + 1) * h].iter_mut().zip(&dx[..h]).for_each(|(g, &d)| *g += d);
                dends[(end - 1) * h..end * h].iter_mut().zip(&dx[h..]).for_each(|(g, &d)| *g += d);
            }
        }
        if let (Some(offset), Some((starts, ends))) = (boundary, &boundaries) {
            starts.backward(&self.params[offset..], &input.words, &dstarts, &mut grads[offset..], h, f);
            ends.backward(&self.params[offset + 4..], &input.words, &dends, &mut grads[offset + 4..], h, f);
        }
        if let (Some(offset), Some((features, _))) = (prompt_offset, &prompt_features) {
            features.backward(&self.params[offset..], &input.prompt_inputs, &dprompt, &mut grads[offset..], h, f);
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
        let decay: Vec<bool> = (0..self.params.len()).map(|i| i % 2 == 0).collect();
        optimizer.step(&mut self.params, grads, &decay).map_err(|e| e.message)
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
        let input = TrainingInput {
            words: vec![0.6, -0.2],
            spans: vec![0.5, -0.3, 0.2, 0.8],
            prompts: vec![0.7],
            prompt_inputs: vec![0.4],
            pairs: vec![(0, 1), (1, 2)],
        };
        let candidates = vec![
            Candidate { start: 0, end: 1, label: 0, logit: 0.1, score: 0.6 },
            Candidate { start: 1, end: 2, label: 0, logit: 0.2, score: 0.7 },
        ];
        (trainer, input, candidates)
    }

    fn check_gradients(trainer: &mut OutputTrainer, input: &TrainingInput, candidates: &[Candidate]) {
        trainer.set_anchor(0.25).unwrap();
        for tensor in 0..trainer.params.len() {
            for at in 0..trainer.params[tensor].len() {
                let original = trainer.params[tensor][at];
                let mut grads = trainer.zero_grads();
                let loss = trainer.gradients(input, candidates, &[(0, 1, 0)], &mut grads).unwrap();
                assert!(loss.is_finite());
                trainer.params[tensor][at] = original + 0.001;
                let plus = trainer.gradients(input, candidates, &[(0, 1, 0)], &mut trainer.zero_grads()).unwrap();
                trainer.params[tensor][at] = original - 0.001;
                let minus = trainer.gradients(input, candidates, &[(0, 1, 0)], &mut trainer.zero_grads()).unwrap();
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

    #[test]
    fn output_gradients_match_finite_differences() {
        let (mut trainer, input, candidates) = fixture();
        check_gradients(&mut trainer, &input, &candidates);
    }

    #[test]
    fn linear_output_gradients_match_finite_differences() {
        let (_, input, candidates) = fixture();
        let params = vec![vec![0.3, -0.4], vec![0.2]];
        let mut trainer = OutputTrainer { hidden: 1, factor: 2, base: params.clone(), params, anchor: 0.0 };
        check_gradients(&mut trainer, &input, &candidates);
    }

    #[test]
    fn boundary_gradients_match_finite_differences() {
        let (output, input, candidates) = fixture();
        let mut params = output.params;
        for _ in 0..2 {
            params.extend([vec![0.2, -0.1], vec![0.1, -0.2], vec![0.3, -0.4], vec![0.2]]);
        }
        let mut trainer = OutputTrainer { hidden: 1, factor: 2, base: params.clone(), params, anchor: 0.0 };
        check_gradients(&mut trainer, &input, &candidates);
    }

    #[test]
    fn linear_boundary_gradients_match_finite_differences() {
        let (_, input, candidates) = fixture();
        let mut params = vec![vec![0.3, -0.4], vec![0.2]];
        for _ in 0..2 {
            params.extend([vec![0.2, -0.1], vec![0.1, -0.2], vec![0.3, -0.4], vec![0.2]]);
        }
        let mut trainer = OutputTrainer { hidden: 1, factor: 2, base: params.clone(), params, anchor: 0.0 };
        check_gradients(&mut trainer, &input, &candidates);
    }

    #[test]
    fn prompt_gradients_match_finite_differences() {
        let (output, input, candidates) = fixture();
        let mut params = output.params;
        params.extend([vec![0.2, -0.1], vec![0.1, -0.2], vec![0.3, -0.4], vec![0.2]]);
        let mut trainer = OutputTrainer { hidden: 1, factor: 2, base: params.clone(), params, anchor: 0.0 };
        check_gradients(&mut trainer, &input, &candidates);
    }

    #[test]
    fn all_head_gradients_match_finite_differences() {
        let (output, input, candidates) = fixture();
        let mut params = output.params;
        for _ in 0..3 {
            params.extend([vec![0.2, -0.1], vec![0.1, -0.2], vec![0.3, -0.4], vec![0.2]]);
        }
        let mut trainer = OutputTrainer { hidden: 1, factor: 2, base: params.clone(), params, anchor: 0.0 };
        check_gradients(&mut trainer, &input, &candidates);
    }
}
