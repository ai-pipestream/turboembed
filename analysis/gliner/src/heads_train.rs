//! Training feature extraction and weight updates for [`super::Heads`].

use std::sync::PoisonError;

use super::{
    Candidate, Dense, Lstm, Matrix, Mlp, Mode, OutputHead, PromptHead, SPANS, SpanHead, Weights, Work, finite, memory,
    sigmoid, size,
};

impl super::Heads {
    /// Frozen word and prompt features for output and boundary training,
    /// plus the ordinary inference candidates used to mine hard negatives.
    pub(crate) fn training(
        &self,
        states: &[f32],
        width: usize,
        prompts: &[usize],
        words: &[usize],
        max_width: usize,
    ) -> Result<(crate::train::TrainingInput, Vec<Candidate>), String> {
        if width != self.projection.k || prompts.iter().chain(words).any(|&i| (i + 1) * width > states.len()) {
            return Err("gliner hidden state layout mismatch".into());
        }
        let (d, labels, n) = (self.hidden, prompts.len(), words.len());
        let mut w = Work::default();
        for &i in prompts.iter().chain(words) {
            w.x.extend_from_slice(&states[i * width..(i + 1) * width]);
        }
        self.projection.run(&w.x, labels + n, &mut w.projected)?;
        self.prompt.run(&w.projected[..labels * d], labels, &mut w.mid, &mut w.prompts)?;
        let projected = std::mem::take(&mut w.projected);
        self.words(&projected[labels * d..], n, &mut w)?;
        let SpanHead::Marker { start, end, span } = &self.head else {
            return Err("gliner token-level output training is not supported".into());
        };
        start.run(&w.words, n, &mut w.mid, &mut w.starts)?;
        end.run(&w.words, n, &mut w.mid, &mut w.ends)?;
        w.starts.iter_mut().chain(w.ends.iter_mut()).for_each(|v| *v = v.max(0.0));
        let pairs: Vec<(usize, usize)> = (0..n).flat_map(|i| (i..n.min(i + max_width)).map(move |j| (i, j))).collect();
        let mut input = crate::train::TrainingInput {
            words: w.words.clone(),
            projected_words: projected[labels * d..].to_vec(),
            encoder_inputs: w.x.clone(),
            spans: Vec::new(),
            prompts: w.prompts.clone(),
            prompt_inputs: projected[..labels * d].to_vec(),
            pairs: pairs.iter().map(|&(start, end)| (start, end + 1)).collect(),
        };
        let scorer = Matrix::from_rows(&[&w.prompts], d).map_err(memory)?;
        let mut candidates = Vec::new();
        for block in pairs.chunks(SPANS) {
            w.x.clear();
            for &(i, j) in block {
                w.x.extend_from_slice(&w.starts[i * d..(i + 1) * d]);
                w.x.extend_from_slice(&w.ends[j * d..(j + 1) * d]);
            }
            input.spans.extend_from_slice(&w.x);
            span.run(&w.x, block.len(), &mut w.mid, &mut w.spans)?;
            let logits = size(&mut w.logits, block.len(), labels);
            scorer.apply(&w.spans, d, block.len(), None, logits, labels, Mode::Plain);
            finite(logits, "span score")?;
            for (&(i, j), row) in block.iter().zip(logits.chunks_exact(labels)) {
                for (label, &logit) in row.iter().enumerate() {
                    candidates.push(Candidate { start: i, end: j + 1, label, logit, score: sigmoid(logit) });
                }
            }
        }
        Ok((input, candidates))
    }

    /// Installs trained tensors in the trainer's order: output, boundary MLPs,
    /// prompt MLP at its own width, BiLSTM, then token projection.
    pub(crate) fn set_output(
        &mut self,
        params: &[Vec<f32>],
        boundaries: bool,
        prompt: bool,
        recurrent: bool,
        projection: bool,
    ) -> Result<(), String> {
        let (h, f, linear) = match &self.head {
            SpanHead::Marker { start, span, .. } => (self.hidden, start.first.n, matches!(span, OutputHead::Linear(_))),
            SpanHead::Token { .. } => return Err("gliner token-level output training is not supported".into()),
        };
        let prompt_width = match (&self.prompt, prompt) {
            (PromptHead::Mlp(head), true) => head.first.n,
            (PromptHead::Identity, true) => return Err("gliner prompt training requires a prompt projection".into()),
            (_, false) => 0,
        };
        let output_len = if linear { 2 } else { 4 };
        let prompt_offset = output_len + if boundaries { 8 } else { 0 };
        let recurrent_offset = prompt_offset + if prompt { 4 } else { 0 };
        let projection_offset = recurrent_offset + if recurrent { 8 } else { 0 };
        if params.len() != projection_offset + if projection { 2 } else { 0 } {
            return Err("gliner output training tensors do not match the model's heads".into());
        }
        let span = match &params[..output_len] {
            [w0, b0, w1, b1] => OutputHead::Mlp(Mlp {
                first: Dense::new(w0, b0.clone(), f, 2 * h)?,
                last: Dense::new(w1, b1.clone(), h, f)?,
            }),
            [weight, bias] => OutputHead::Linear(Dense::new(weight, bias.clone(), h, 2 * h)?),
            _ => unreachable!(),
        };
        let mlp = |values: &[Vec<f32>], width: usize| -> Result<Box<Mlp>, String> {
            Ok(Box::new(Mlp {
                first: Dense::new(&values[0], values[1].clone(), width, h)?,
                last: Dense::new(&values[2], values[3].clone(), h, width)?,
            }))
        };
        let boundaries = if boundaries {
            Some((mlp(&params[output_len..output_len + 4], f)?, mlp(&params[output_len + 4..output_len + 8], f)?))
        } else {
            None
        };
        let prompt = if prompt {
            Some(PromptHead::Mlp(mlp(&params[prompt_offset..prompt_offset + 4], prompt_width)?))
        } else {
            None
        };
        let recurrent = if recurrent {
            let values = &params[recurrent_offset..projection_offset];
            let forward: Vec<f32> = values[2].iter().zip(&values[3]).map(|(a, b)| a + b).collect();
            let reverse: Vec<f32> = values[6].iter().zip(&values[7]).map(|(a, b)| a + b).collect();
            let dirs = [
                Weights { w: &values[0], u: &values[1], b: &forward, reverse: false },
                Weights { w: &values[4], u: &values[5], b: &reverse, reverse: true },
            ];
            let (dev, prec) = (self.lstm.device(), self.lstm.precision());
            Some(Lstm::new_at(dev, prec, h, h / 2, &dirs).map_err(|e| format!("gliner recurrent: {}", e.message))?)
        } else {
            None
        };
        let projection = if projection {
            let values = &params[params.len() - 2..];
            Some(Dense::new(&values[0], values[1].clone(), h, self.projection.k)?)
        } else {
            None
        };
        let SpanHead::Marker { start, end, span: current } = &mut self.head else { unreachable!() };
        *current = span;
        if let Some((new_start, new_end)) = boundaries {
            *start = new_start;
            *end = new_end;
        }
        if let Some(prompt) = prompt {
            self.prompt = prompt;
        }
        if let Some(recurrent) = recurrent {
            self.lstm = recurrent;
            self.spare.lock().unwrap_or_else(PoisonError::into_inner).clear();
        }
        if let Some(projection) = projection {
            self.projection = projection;
        }
        Ok(())
    }
}
