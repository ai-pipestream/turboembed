//! Inference: word representations and raw per-token tag scores.

use std::collections::HashMap;

use crate::neural::NeuralError;

use super::BilstmPosModel;

const REPRESENTATION_CACHE_LIMIT: usize = 100_000;
impl BilstmPosModel {
    /// Computes the concatenated `[word; char_fwd_last; char_bwd_last; pretrained]` representation for `token`.
    pub fn word_representation(&self, token: &str) -> Result<Vec<f64>, NeuralError> {
        if let Ok(guard) = self.representation_cache.read()
            && let Some(ref map) = *guard
            && let Some(cached) = map.get(token)
        {
            return Ok(cached.clone());
        }
        let rep = self.compute_word_representation(token)?;
        if let Ok(mut guard) = self.representation_cache.write()
            && let Some(ref mut map) = *guard
            && map.len() < REPRESENTATION_CACHE_LIMIT
        {
            map.entry(token.to_owned()).or_insert_with(|| rep.clone());
        }
        Ok(rep)
    }

    /// Enables the thread-safe per-token word representation cache (bounded at 100,000 entries).
    pub fn enable_representation_cache(&self) {
        if let Ok(mut guard) = self.representation_cache.write()
            && guard.is_none()
        {
            *guard = Some(HashMap::new());
        }
    }

    fn compute_word_representation(&self, token: &str) -> Result<Vec<f64>, NeuralError> {
        let word_dim = self.word_embeddings[0].len();
        let char_hidden = self.char_forward.hidden_size();
        let pretrained_dim = self.pretrained_dim();
        let mut rep = vec![0.0f64; word_dim + 2 * char_hidden + pretrained_dim];
        let w_row = &self.word_embeddings[self.word_id(token)];
        rep[..word_dim].copy_from_slice(w_row);

        // As in Java, a token without characters keeps zero character states.
        let c_ids = self.char_ids(token);
        let len = c_ids.len();
        if len > 0 {
            let mut char_fwd_in = Vec::with_capacity(len);
            let mut char_bwd_in = Vec::with_capacity(len);
            for i in 0..len {
                char_fwd_in.push(self.char_embeddings[c_ids[i]].clone());
                char_bwd_in.push(self.char_embeddings[c_ids[len - 1 - i]].clone());
            }
            let fwd_h = self.char_forward.run(&char_fwd_in, None)?;
            let bwd_h = self.char_backward.run(&char_bwd_in, None)?;
            rep[word_dim..word_dim + char_hidden].copy_from_slice(&fwd_h[len - 1]);
            rep[word_dim + char_hidden..word_dim + 2 * char_hidden].copy_from_slice(&bwd_h[len - 1]);
        }

        if pretrained_dim > 0 {
            let norm = Self::normalize(token);
            if let Some(&p_row_idx) = self.pretrained_ids.get(&norm)
                && let Some(ref p_vecs) = self.pretrained_vectors
            {
                let p_row = &p_vecs[p_row_idx];
                let offset = word_dim + 2 * char_hidden;
                if let (Some(aw), Some(ab)) = (&self.adapter_weights, &self.adapter_bias) {
                    for i in 0..pretrained_dim {
                        let mut sum = ab[i];
                        for r in 0..pretrained_dim {
                            sum += aw[r][i] * f64::from(p_row[r]);
                        }
                        rep[offset + i] = sum;
                    }
                } else {
                    for i in 0..pretrained_dim {
                        rep[offset + i] = f64::from(p_row[i]);
                    }
                }
            }
        }
        for &val in &rep {
            if !val.is_finite() {
                return Err(NeuralError::NonFiniteScore(format!("Non-finite word representation: {val}")));
            }
        }
        Ok(rep)
    }

    /// Runs the full BiLSTM stack over `tokens` and returns raw per-token tag scores `[tokens.len()][tags.len()]`.
    ///
    /// The scores follow Java's `BilstmPOSModel.score` operation by
    /// operation, except that the gate sigmoids use the platform `exp` where
    /// Java uses HotSpot's `Math.exp` intrinsic, so a score can differ from
    /// the JVM's in its last bits (see the sigmoid in the LSTM module).
    pub fn score(&self, tokens: &[impl AsRef<str>]) -> Result<Vec<Vec<f64>>, NeuralError> {
        if tokens.is_empty() {
            return Err(NeuralError::InvalidArgument("tokens must not be null or empty".into()));
        }
        let n = tokens.len();
        let mut fwd_in = Vec::with_capacity(n);
        let mut bwd_in = vec![Vec::new(); n];
        for (t, tok) in tokens.iter().enumerate() {
            let rep = self.word_representation(tok.as_ref())?;
            bwd_in[n - 1 - t] = rep.clone();
            fwd_in.push(rep);
        }
        let mut fwd_h = self.word_forward.run(&fwd_in, None)?;
        let mut bwd_h = self.word_backward.run(&bwd_in, None)?;
        Self::validate_sequence_states(&fwd_h, "forward encoder")?;
        Self::validate_sequence_states(&bwd_h, "backward encoder")?;

        if let (Some(wf2), Some(wb2)) = (&self.word_forward2, &self.word_backward2) {
            let h1 = self.word_forward.hidden_size();
            let mut l2_fwd_in = vec![vec![0.0; 2 * h1]; n];
            let mut l2_bwd_in = vec![vec![0.0; 2 * h1]; n];
            for t in 0..n {
                l2_fwd_in[t][..h1].copy_from_slice(&fwd_h[t]);
                l2_fwd_in[t][h1..].copy_from_slice(&bwd_h[n - 1 - t]);
                l2_bwd_in[n - 1 - t] = l2_fwd_in[t].clone();
            }
            fwd_h = wf2.run(&l2_fwd_in, None)?;
            bwd_h = wb2.run(&l2_bwd_in, None)?;
            Self::validate_sequence_states(&fwd_h, "second forward encoder")?;
            Self::validate_sequence_states(&bwd_h, "second backward encoder")?;
        }

        let num_tags = self.tags.len();
        let h_size = match &self.word_forward2 {
            Some(wf2) => wf2.hidden_size(),
            None => self.word_forward.hidden_size(),
        };
        let mut scores = vec![vec![0.0f64; num_tags]; n];
        for t in 0..n {
            let fh = &fwd_h[t];
            let bh = &bwd_h[n - 1 - t];
            let out_t = &mut scores[t];
            for o in 0..num_tags {
                let w = &self.output_weights[o];
                // Java adds the products over the concatenated state one at a
                // time, the forward half first, so the sum rounds the same way.
                let mut sum = self.output_bias[o];
                for j in 0..h_size {
                    sum += w[j] * fh[j];
                }
                for j in 0..h_size {
                    sum += w[h_size + j] * bh[j];
                }
                if !sum.is_finite() {
                    return Err(NeuralError::NonFiniteScore(format!(
                        "Non-finite tag score at token {t}, tag {o}: {sum}"
                    )));
                }
                out_t[o] = sum;
            }
        }
        Ok(scores)
    }

    fn validate_sequence_states(states: &[Vec<f64>], name: &str) -> Result<(), NeuralError> {
        for (t, step) in states.iter().enumerate() {
            for &v in step {
                if !v.is_finite() {
                    return Err(NeuralError::NonFiniteScore(format!("Non-finite {name} state at token {t}: {v}")));
                }
            }
        }
        Ok(())
    }

    pub(super) fn pretrained_dim(&self) -> usize {
        match &self.pretrained_vectors {
            Some(vecs) => {
                if let Some(first) = vecs.first() {
                    first.len()
                } else {
                    self.word_forward
                        .input_size()
                        .saturating_sub(self.word_embeddings[0].len() + 2 * self.char_forward.hidden_size())
                }
            }
            None => 0,
        }
    }
}
