//! Sentence-level forward and backward pass for [`super::TrainingContext`].

use crate::neural::bilstm_model::{BilstmPosModel, char_key_from_unit};
use crate::neural::lstm::{ForwardCache, LstmGradients};
use crate::neural::{JavaRng, NeuralError};

use super::context::{TrainingContext, Worker};
use super::settings::MultiTaskSample;

impl TrainingContext {
    /// Runs forward and backward passes for one sentence sample, accumulating parameter gradients into `worker`.
    pub fn sentence_gradients(
        &self,
        sample: &MultiTaskSample,
        rng: &mut JavaRng,
        worker: &mut Worker,
    ) -> Result<f64, NeuralError> {
        sample.validate()?;
        let tokens = &sample.sentence;
        let sample_tags = &sample.tags;
        let n = tokens.len();
        if n == 0 {
            return Ok(0.0);
        }
        let word_dim = self.settings.word_dim;
        let char_dim = self.settings.char_dim;
        let char_hidden = self.settings.char_hidden;
        let word_hidden = self.settings.word_hidden;
        let num_tags = self.tags.len();

        let word_embeddings = self.optimizer.params(self.idx_word_emb);
        let char_embeddings = self.optimizer.params(self.idx_char_emb);
        let output_weights = self.optimizer.params(self.idx_out_w);
        let output_bias = &self.optimizer.params(self.idx_out_b)[0];

        let mut token_word_ids = vec![0usize; n];
        let mut token_pretrained_rows = vec![None; n];
        let mut token_char_ids: Vec<Vec<usize>> = Vec::with_capacity(n);
        let mut char_fwd_in: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n);
        let mut char_bwd_in: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n);
        if worker.char_fwd_caches.len() < n {
            worker.char_fwd_caches.resize_with(n, || ForwardCache::of(1, char_hidden));
            worker.char_bwd_caches.resize_with(n, || ForwardCache::of(1, char_hidden));
        }

        let in_dim = word_dim + 2 * char_hidden + self.pretrained_dim;
        let keep_prob = 1.0 - self.settings.dropout;
        let mut word_fwd_in = vec![vec![0.0; in_dim]; n];
        let mut word_bwd_in = vec![vec![0.0; in_dim]; n];
        let mut input_masks = vec![vec![0.0; in_dim]; n];

        for t in 0..n {
            let raw = &tokens[t];
            let norm = BilstmPosModel::normalize(raw);
            let mut wid = self.word_ids.get(&norm).copied().unwrap_or(0);
            if self.settings.word_dropout > 0.0 && rng.next_double() < self.settings.word_dropout {
                wid = 0;
            }
            token_word_ids[t] = wid;

            let c_ids: Vec<usize> = raw
                .encode_utf16()
                .take(self.settings.max_word_length)
                .map(|u| {
                    let key = char_key_from_unit(u);
                    self.char_ids.get(&key).copied().unwrap_or(0)
                })
                .collect();
            let c_len = c_ids.len();
            let mut cf_in = Vec::with_capacity(c_len);
            let mut cb_in = Vec::with_capacity(c_len);
            for i in 0..c_len {
                cf_in.push(char_embeddings[c_ids[i]].clone());
                cb_in.push(char_embeddings[c_ids[c_len - 1 - i]].clone());
            }
            if worker.char_fwd_caches[t].h.len() < c_len {
                worker.char_fwd_caches[t] = ForwardCache::of(c_len, char_hidden);
                worker.char_bwd_caches[t] = ForwardCache::of(c_len, char_hidden);
            }
            let rep = &mut word_fwd_in[t];
            rep[..word_dim].copy_from_slice(&word_embeddings[wid]);
            // As in Java, a token without characters keeps zero character states.
            if c_len > 0 {
                let cf_h = self.char_forward.run(&cf_in, Some(&mut worker.char_fwd_caches[t]))?;
                let cb_h = self.char_backward.run(&cb_in, Some(&mut worker.char_bwd_caches[t]))?;
                rep[word_dim..word_dim + char_hidden].copy_from_slice(&cf_h[c_len - 1]);
                rep[word_dim + char_hidden..word_dim + 2 * char_hidden].copy_from_slice(&cb_h[c_len - 1]);
            }
            token_char_ids.push(c_ids);
            char_fwd_in.push(cf_in);
            char_bwd_in.push(cb_in);

            if self.pretrained_dim > 0 {
                let mut p_row = self.pretrained_ids.get(&norm).copied();
                if p_row.is_some()
                    && self.settings.pretrained_dropout > 0.0
                    && rng.next_double() < self.settings.pretrained_dropout
                {
                    p_row = None;
                }
                token_pretrained_rows[t] = p_row;
                if let Some(pr) = p_row {
                    let offset = word_dim + 2 * char_hidden;
                    if let (Some(idx_aw), Some(idx_ab)) = (self.idx_adapter_w, self.idx_adapter_b) {
                        let aw = self.optimizer.params(idx_aw);
                        let ab = &self.optimizer.params(idx_ab)[0];
                        let p_vec = &self.pretrained_vectors.as_ref().unwrap()[pr];
                        for i in 0..self.pretrained_dim {
                            let mut sum = ab[i];
                            for r in 0..self.pretrained_dim {
                                sum += aw[r][i] * f64::from(p_vec[r]);
                            }
                            rep[offset + i] = sum;
                        }
                    } else if let Some(ref trainable) = self.pretrained_trainable {
                        rep[offset..offset + self.pretrained_dim].copy_from_slice(&trainable[pr]);
                    } else {
                        let p_vec = &self.pretrained_vectors.as_ref().unwrap()[pr];
                        for i in 0..self.pretrained_dim {
                            rep[offset + i] = f64::from(p_vec[i]);
                        }
                    }
                }
            }
            for i in 0..in_dim {
                let m = if rng.next_double() < keep_prob { 1.0 / keep_prob } else { 0.0 };
                input_masks[t][i] = m;
                rep[i] *= m;
            }
            word_bwd_in[n - 1 - t] = rep.clone();
        }

        if worker.word_fwd_cache.as_ref().is_none_or(|c| c.h.len() < n) {
            worker.word_fwd_cache = Some(ForwardCache::of(n, word_hidden));
            worker.word_bwd_cache = Some(ForwardCache::of(n, word_hidden));
            if self.word_forward2.is_some() {
                worker.word_fwd2_cache = Some(ForwardCache::of(n, word_hidden));
                worker.word_bwd2_cache = Some(ForwardCache::of(n, word_hidden));
            }
        }
        let mut word_fwd_h = self.word_forward.run(&word_fwd_in, worker.word_fwd_cache.as_mut())?;
        let mut word_bwd_h = self.word_backward.run(&word_bwd_in, worker.word_bwd_cache.as_mut())?;

        let encoder_keep = 1.0 - self.settings.encoder_dropout;
        let mut l1_mask: Option<Vec<Vec<f64>>> = None;
        let mut l2_fwd_in: Option<Vec<Vec<f64>>> = None;
        let mut l2_bwd_in: Option<Vec<Vec<f64>>> = None;

        if let (Some(wf2), Some(wb2)) = (&self.word_forward2, &self.word_backward2) {
            let mut f_in = vec![vec![0.0; 2 * word_hidden]; n];
            let mut b_in = vec![vec![0.0; 2 * word_hidden]; n];
            let mut mask =
                if self.settings.encoder_dropout > 0.0 { Some(vec![vec![0.0; 2 * word_hidden]; n]) } else { None };
            for t in 0..n {
                f_in[t][..word_hidden].copy_from_slice(&word_fwd_h[t]);
                f_in[t][word_hidden..].copy_from_slice(&word_bwd_h[n - 1 - t]);
                if let Some(ref mut m) = mask {
                    for j in 0..(2 * word_hidden) {
                        let scale = if rng.next_double() < encoder_keep { 1.0 / encoder_keep } else { 0.0 };
                        m[t][j] = scale;
                        f_in[t][j] *= scale;
                    }
                }
                b_in[n - 1 - t] = f_in[t].clone();
            }
            word_fwd_h = wf2.run(&f_in, worker.word_fwd2_cache.as_mut())?;
            word_bwd_h = wb2.run(&b_in, worker.word_bwd2_cache.as_mut())?;
            l1_mask = mask;
            l2_fwd_in = Some(f_in);
            l2_bwd_in = Some(b_in);
        }

        let top_mask = if self.settings.encoder_dropout > 0.0 {
            let mut mask = vec![vec![0.0; 2 * word_hidden]; n];
            for t in 0..n {
                for j in 0..word_hidden {
                    let sf = if rng.next_double() < encoder_keep { 1.0 / encoder_keep } else { 0.0 };
                    mask[t][j] = sf;
                    word_fwd_h[t][j] *= sf;
                }
                for j in 0..word_hidden {
                    let sb = if rng.next_double() < encoder_keep { 1.0 / encoder_keep } else { 0.0 };
                    mask[t][word_hidden + j] = sb;
                    word_bwd_h[n - 1 - t][j] *= sb;
                }
            }
            Some(mask)
        } else {
            None
        };

        let mut d_word_fwd_h = vec![vec![0.0; word_hidden]; n];
        let mut d_word_bwd_h = vec![vec![0.0; word_hidden]; n];

        let mut emissions = vec![vec![0.0; num_tags]; n];
        let mut gold = vec![0usize; n];
        for t in 0..n {
            let fh = &word_fwd_h[t];
            let bh = &word_bwd_h[n - 1 - t];
            for o in 0..num_tags {
                let w = &output_weights[o];
                let mut sum = output_bias[o];
                for j in 0..word_hidden {
                    sum += w[j] * fh[j];
                }
                for j in 0..word_hidden {
                    sum += w[word_hidden + j] * bh[j];
                }
                if !sum.is_finite() {
                    return Err(NeuralError::NonFiniteScore("non-finite primary emission score".into()));
                }
                emissions[t][o] = sum;
            }
            gold[t] = self
                .tag_ids
                .get(&sample_tags[t])
                .copied()
                .ok_or_else(|| NeuralError::InvalidArgument(format!("unknown tag '{}'", sample_tags[t])))?;
        }

        let mut d_scores = vec![vec![0.0; num_tags]; n];
        let mut sentence_loss = self.primary_loss_and_gradients(&emissions, &gold, &mut d_scores, worker)?;

        for t in 0..n {
            let fh = &word_fwd_h[t];
            let bh = &word_bwd_h[n - 1 - t];
            let dfh = &mut d_word_fwd_h[t];
            let dbh = &mut d_word_bwd_h[n - 1 - t];
            for o in 0..num_tags {
                let ds = d_scores[t][o];
                worker.buffers[self.idx_out_b][0][o] += ds;
                let w = &output_weights[o];
                let dw = &mut worker.buffers[self.idx_out_w][o];
                for j in 0..word_hidden {
                    dw[j] += ds * fh[j];
                    dw[word_hidden + j] += ds * bh[j];
                    dfh[j] += ds * w[j];
                    dbh[j] += ds * w[word_hidden + j];
                }
            }
        }

        if let (Some(idx_xw), Some(idx_xb), Some(sample_xpos)) =
            (self.idx_xpos_w, self.idx_xpos_b, sample.xpos.as_ref())
        {
            sentence_loss += self.auxiliary_head_pass(
                &word_fwd_h,
                &word_bwd_h,
                sample_xpos,
                &self.xpos_ids,
                idx_xw,
                idx_xb,
                &mut d_word_fwd_h,
                &mut d_word_bwd_h,
                worker,
            )?;
        }

        if let (Some(idx_fw), Some(idx_fb), Some(sample_feats)) =
            (self.idx_feats_w, self.idx_feats_b, sample.feats.as_ref())
        {
            sentence_loss += self.auxiliary_head_pass(
                &word_fwd_h,
                &word_bwd_h,
                sample_feats,
                &self.feats_ids,
                idx_fw,
                idx_fb,
                &mut d_word_fwd_h,
                &mut d_word_bwd_h,
                worker,
            )?;
        }

        if let Some(ref mask) = top_mask {
            for t in 0..n {
                for j in 0..word_hidden {
                    d_word_fwd_h[t][j] *= mask[t][j];
                    d_word_bwd_h[n - 1 - t][j] *= mask[t][word_hidden + j];
                }
            }
        }

        if let (Some(wf2), Some(wb2), Some(f_in), Some(b_in)) =
            (&self.word_forward2, &self.word_backward2, l2_fwd_in.as_ref(), l2_bwd_in.as_ref())
        {
            let mut d_l2_fwd_in = vec![vec![0.0; 2 * word_hidden]; n];
            let mut d_l2_bwd_in = vec![vec![0.0; 2 * word_hidden]; n];
            let wf2_cache = worker.word_fwd2_cache.as_mut().unwrap();
            with_buffer_grads(
                &mut worker.buffers,
                [self.idx_wf2_w.unwrap(), self.idx_wf2_u.unwrap(), self.idx_wf2_b.unwrap()],
                |grads| wf2.backward(f_in, wf2_cache, &d_word_fwd_h, &mut d_l2_fwd_in, grads),
            );
            let wb2_cache = worker.word_bwd2_cache.as_mut().unwrap();
            with_buffer_grads(
                &mut worker.buffers,
                [self.idx_wb2_w.unwrap(), self.idx_wb2_u.unwrap(), self.idx_wb2_b.unwrap()],
                |grads| wb2.backward(b_in, wb2_cache, &d_word_bwd_h, &mut d_l2_bwd_in, grads),
            );
            for t in 0..n {
                let df = &d_l2_fwd_in[t];
                let db = &d_l2_bwd_in[n - 1 - t];
                let m = l1_mask.as_ref().map(|m| &m[t]);
                for j in 0..word_hidden {
                    let scale_f = m.map_or(1.0, |row| row[j]);
                    let scale_b = m.map_or(1.0, |row| row[word_hidden + j]);
                    d_word_fwd_h[t][j] = (df[j] + db[j]) * scale_f;
                    d_word_bwd_h[n - 1 - t][j] = (df[word_hidden + j] + db[word_hidden + j]) * scale_b;
                }
            }
        }

        let mut d_word_fwd_in = vec![vec![0.0; in_dim]; n];
        let mut d_word_bwd_in = vec![vec![0.0; in_dim]; n];
        let wf_cache = worker.word_fwd_cache.as_mut().unwrap();
        with_buffer_grads(&mut worker.buffers, [self.idx_wf_w, self.idx_wf_u, self.idx_wf_b], |grads| {
            self.word_forward.backward(&word_fwd_in, wf_cache, &d_word_fwd_h, &mut d_word_fwd_in, grads)
        });
        let wb_cache = worker.word_bwd_cache.as_mut().unwrap();
        with_buffer_grads(&mut worker.buffers, [self.idx_wb_w, self.idx_wb_u, self.idx_wb_b], |grads| {
            self.word_backward.backward(&word_bwd_in, wb_cache, &d_word_bwd_h, &mut d_word_bwd_in, grads)
        });

        let mut d_char_fwd_h: Vec<Vec<f64>> = Vec::new();
        let mut d_char_bwd_h: Vec<Vec<f64>> = Vec::new();
        let mut d_char_fwd_in: Vec<Vec<f64>> = Vec::new();
        let mut d_char_bwd_in: Vec<Vec<f64>> = Vec::new();
        let mut dx = vec![0.0; in_dim];

        for t in 0..n {
            let df_in = &d_word_fwd_in[t];
            let db_in = &d_word_bwd_in[n - 1 - t];
            let mask = &input_masks[t];
            for i in 0..in_dim {
                dx[i] = (df_in[i] + db_in[i]) * mask[i];
            }
            let d_w_emb = &mut worker.buffers[self.idx_word_emb][token_word_ids[t]];
            for i in 0..word_dim {
                d_w_emb[i] += dx[i];
            }
            if let Some(p_row_idx) = token_pretrained_rows[t] {
                let offset = word_dim + 2 * char_hidden;
                if let (Some(idx_aw), Some(idx_ab)) = (self.idx_adapter_w, self.idx_adapter_b) {
                    let p_vec = &self.pretrained_vectors.as_ref().unwrap()[p_row_idx];
                    for r in 0..self.pretrained_dim {
                        let dw_row = &mut worker.buffers[idx_aw][r];
                        let val = f64::from(p_vec[r]);
                        for i in 0..self.pretrained_dim {
                            dw_row[i] += dx[offset + i] * val;
                        }
                    }
                    for i in 0..self.pretrained_dim {
                        worker.buffers[idx_ab][0][i] += dx[offset + i];
                    }
                } else if self.pretrained_trainable.is_some() {
                    let d_p_emb =
                        worker.tuning_row_grads.entry(p_row_idx).or_insert_with(|| vec![0.0; self.pretrained_dim]);
                    for i in 0..self.pretrained_dim {
                        d_p_emb[i] += dx[offset + i];
                    }
                }
            }

            let c_ids = &token_char_ids[t];
            let c_len = c_ids.len();
            if c_len == 0 {
                continue;
            }
            if d_char_fwd_h.len() != c_len {
                d_char_fwd_h = vec![vec![0.0; char_hidden]; c_len];
                d_char_bwd_h = vec![vec![0.0; char_hidden]; c_len];
                d_char_fwd_in = vec![vec![0.0; char_dim]; c_len];
                d_char_bwd_in = vec![vec![0.0; char_dim]; c_len];
            } else {
                for s in 0..c_len {
                    d_char_fwd_h[s].fill(0.0);
                    d_char_bwd_h[s].fill(0.0);
                }
            }
            for j in 0..char_hidden {
                d_char_fwd_h[c_len - 1][j] = dx[word_dim + j];
                d_char_bwd_h[c_len - 1][j] = dx[word_dim + char_hidden + j];
            }

            // Java runs the forward direction's backward pass and adds its
            // embedding gradients before it does the same for the backward
            // direction, accumulating straight into the optimizer buffers.
            let cf_cache = &mut worker.char_fwd_caches[t];
            with_buffer_grads(&mut worker.buffers, [self.idx_cf_w, self.idx_cf_u, self.idx_cf_b], |grads| {
                self.char_forward.backward(&char_fwd_in[t], cf_cache, &d_char_fwd_h, &mut d_char_fwd_in, grads)
            });
            for i in 0..c_len {
                let d_f = &d_char_fwd_in[i];
                let d_c_emb = &mut worker.buffers[self.idx_char_emb][c_ids[i]];
                for k in 0..char_dim {
                    d_c_emb[k] += d_f[k];
                }
            }
            let cb_cache = &mut worker.char_bwd_caches[t];
            with_buffer_grads(&mut worker.buffers, [self.idx_cb_w, self.idx_cb_u, self.idx_cb_b], |grads| {
                self.char_backward.backward(&char_bwd_in[t], cb_cache, &d_char_bwd_h, &mut d_char_bwd_in, grads)
            });
            for i in 0..c_len {
                let d_b = &d_char_bwd_in[c_len - 1 - i];
                let d_c_emb = &mut worker.buffers[self.idx_char_emb][c_ids[i]];
                for k in 0..char_dim {
                    d_c_emb[k] += d_b[k];
                }
            }
        }
        Ok(sentence_loss)
    }
}

/// Runs `backward` with the optimizer's gradient buffers for one LSTM
/// layer (`[w, u, b]` buffer indices) as its accumulator, as Java's
/// `LstmLayer.Gradients.wrap` does, so each gradient element receives every
/// contribution in Java's order instead of a per-call partial sum. The
/// buffers are moved out and back, not copied.
fn with_buffer_grads(
    buffers: &mut [Vec<Vec<f64>>],
    [idx_w, idx_u, idx_b]: [usize; 3],
    backward: impl FnOnce(&mut LstmGradients),
) {
    let mut grads = LstmGradients {
        dw: std::mem::take(&mut buffers[idx_w]),
        du: std::mem::take(&mut buffers[idx_u]),
        db: std::mem::take(&mut buffers[idx_b][0]),
    };
    backward(&mut grads);
    buffers[idx_w] = grads.dw;
    buffers[idx_u] = grads.du;
    buffers[idx_b][0] = grads.db;
}
