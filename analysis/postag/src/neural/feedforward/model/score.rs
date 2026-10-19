//! Scoring a feature vector with a [`FeedforwardPosModel`], and the per-slot hidden-layer
//! contribution cache the scoring path fills.

use std::sync::atomic::Ordering;

use crate::neural::NeuralError;
use crate::neural::feedforward::context::{PRETRAINED_SLOTS, SLOTS};

use super::*;

impl FeedforwardPosModel {
    /// Computes raw output scores for a 12-slot discrete feature vector on a model without pretrained vectors.
    pub fn score(&self, features: &[usize]) -> Result<Vec<f64>, NeuralError> {
        if self.pretrained_size != 0 {
            return Err(NeuralError::InvalidArgument(
                "model uses pretrained vectors; call score_with_pretrained(features, pretrained_rows)".into(),
            ));
        }
        self.score_with_pretrained(features, None)
    }

    /// Computes raw output scores for `features` and optional `pretrained_rows`.
    pub fn score_with_pretrained(
        &self,
        features: &[usize],
        pretrained_rows: Option<&[isize]>,
    ) -> Result<Vec<f64>, NeuralError> {
        if features.len() != SLOTS {
            return Err(NeuralError::InvalidArgument(format!(
                "expected {SLOTS} feature slots, got {}",
                features.len()
            )));
        }
        let num_embeddings = self.embeddings.len();
        for (slot, &row) in features.iter().enumerate() {
            if row >= num_embeddings {
                return Err(NeuralError::InvalidArgument(format!(
                    "feature id at slot {slot} out of bounds: {row} >= {num_embeddings}"
                )));
            }
        }
        if self.pretrained_size == 0 {
            if pretrained_rows.is_some() {
                return Err(NeuralError::InvalidArgument(
                    "pretrainedRows must be None when pretrainedSize == 0".into(),
                ));
            }
        } else {
            let Some(rows) = pretrained_rows else {
                return Err(NeuralError::InvalidArgument(format!("expected {PRETRAINED_SLOTS} pretrained rows")));
            };
            if rows.len() != PRETRAINED_SLOTS {
                return Err(NeuralError::InvalidArgument(format!(
                    "expected {PRETRAINED_SLOTS} pretrained rows, got {}",
                    rows.len()
                )));
            }
            let num_pretrained = self.pretrained_vectors.len();
            for (w_slot, &row) in rows.iter().enumerate() {
                if row < -1 || (row >= 0 && (row as usize) >= num_pretrained) {
                    return Err(NeuralError::InvalidArgument(format!(
                        "pretrained row at slot {w_slot} out of bounds: {row} (expected -1 or 0..{num_pretrained})"
                    )));
                }
            }
        }

        let hidden_size = self.hidden_bias.len();
        let mut hidden: Vec<f64> = self.hidden_bias.iter().map(|&b| f64::from(b)).collect();

        let active_cache = self.cache.read().ok().and_then(|g| g.clone());
        if let Some(ref c) = active_cache {
            for (slot, &row) in features.iter().enumerate() {
                let slot_cells = &c.by_slot[slot];
                if let Some(contrib) = slot_cells[row].get() {
                    for j in 0..hidden_size {
                        hidden[j] += contrib[j];
                    }
                } else if self.try_claim_cache_entry(c) {
                    let contrib = self.compute_contribution(slot, row);
                    let _ = slot_cells[row].set(contrib);
                    let stored = slot_cells[row].get().unwrap();
                    for j in 0..hidden_size {
                        hidden[j] += stored[j];
                    }
                } else {
                    self.add_slot_contribution(&mut hidden, slot, row);
                }
            }
        } else {
            for (slot, &row) in features.iter().enumerate() {
                self.add_slot_contribution(&mut hidden, slot, row);
            }
        }

        if self.pretrained_size > 0
            && let Some(rows) = pretrained_rows
        {
            let base = SLOTS * self.embedding_size;
            for (w_slot, &row) in rows.iter().enumerate() {
                if row < 0 {
                    continue;
                }
                let vec = &self.pretrained_vectors[row as usize];
                let offset = base + w_slot * self.pretrained_size;
                for j in 0..hidden_size {
                    let hw = &self.hidden_weights[j];
                    let mut sum = 0.0f64;
                    for i in 0..self.pretrained_size {
                        sum += f64::from(hw[offset + i]) * f64::from(vec[i]);
                    }
                    hidden[j] += sum;
                }
            }
        }

        for h in &mut hidden {
            let v = *h;
            *h = v * v * v;
        }

        let num_tags = self.tags.len();
        let mut out = vec![0.0f64; num_tags];
        for o in 0..num_tags {
            let w = &self.output_weights[o];
            let mut sum = f64::from(self.output_bias[o]);
            for j in 0..hidden_size {
                sum += f64::from(w[j]) * hidden[j];
            }
            out[o] = sum;
        }
        Ok(out)
    }

    fn try_claim_cache_entry(&self, cache: &ContributionCache) -> bool {
        loop {
            let rem = cache.remaining.load(Ordering::SeqCst);
            if rem <= 0 {
                return false;
            }
            if cache.remaining.compare_exchange(rem, rem - 1, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                return true;
            }
        }
    }

    fn compute_contribution(&self, slot: usize, row: usize) -> Vec<f64> {
        let hidden_size = self.hidden_bias.len();
        let e = &self.embeddings[row];
        let offset = slot * self.embedding_size;
        let mut contrib = vec![0.0f64; hidden_size];
        for j in 0..hidden_size {
            let w = &self.hidden_weights[j];
            let mut sum = 0.0f64;
            for i in 0..self.embedding_size {
                sum += f64::from(w[offset + i]) * f64::from(e[i]);
            }
            contrib[j] = sum;
        }
        contrib
    }

    /// Adds one slot's contribution to `hidden` without caching it. The slot's
    /// products are summed from zero before being added, the same order as a
    /// cached contribution, so a score does not depend on whether the cache
    /// was enabled or had room for this slot and row.
    fn add_slot_contribution(&self, hidden: &mut [f64], slot: usize, row: usize) {
        let e = &self.embeddings[row];
        let offset = slot * self.embedding_size;
        for (j, h) in hidden.iter_mut().enumerate() {
            let w = &self.hidden_weights[j];
            let mut sum = 0.0f64;
            for i in 0..self.embedding_size {
                sum += f64::from(w[offset + i]) * f64::from(e[i]);
            }
            *h += sum;
        }
    }
}
