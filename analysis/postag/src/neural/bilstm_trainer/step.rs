//! Loss heads, optimizer synchronization, batch updates, and model snapshotting for [`super::TrainingContext`].

use std::collections::HashMap;

use crate::neural::NeuralError;
use crate::neural::bilstm_model::BilstmPosModel;
use crate::neural::crf::CrfScorer;
use crate::neural::lstm::LstmLayer;

use super::context::{TrainingContext, Worker};

impl TrainingContext {
    pub(super) fn primary_loss_and_gradients(
        &self,
        emissions: &[Vec<f64>],
        gold: &[usize],
        d_scores: &mut [Vec<f64>],
        worker: &mut Worker,
    ) -> Result<f64, NeuralError> {
        let n = emissions.len();
        let num_tags = self.tags.len();
        if self.settings.crf {
            let idx_tr = self.idx_crf_tr.unwrap();
            let idx_st = self.idx_crf_st.unwrap();
            let idx_en = self.idx_crf_en.unwrap();
            let tr = self.optimizer.params(idx_tr);
            let st = &self.optimizer.params(idx_st)[0];
            let en = &self.optimizer.params(idx_en)[0];
            let mut d_tr = worker.buffers[idx_tr].clone();
            let mut d_st = worker.buffers[idx_st][0].clone();
            let mut d_en = worker.buffers[idx_en][0].clone();
            let loss =
                CrfScorer::loss_and_gradients(emissions, tr, st, en, gold, d_scores, &mut d_tr, &mut d_st, &mut d_en)?;
            worker.buffers[idx_tr] = d_tr;
            worker.buffers[idx_st][0] = d_st;
            worker.buffers[idx_en][0] = d_en;
            Ok(loss)
        } else {
            let mut loss = 0.0f64;
            for t in 0..n {
                let scores = &emissions[t];
                let mut max = f64::NEG_INFINITY;
                for &s in scores {
                    if !s.is_finite() {
                        return Err(NeuralError::NonFiniteScore("non-finite primary emission score".into()));
                    }
                    if s > max {
                        max = s;
                    }
                }
                let mut sum_exp = 0.0f64;
                for o in 0..num_tags {
                    let e = (scores[o] - max).exp();
                    d_scores[t][o] = e;
                    sum_exp += e;
                }
                if !sum_exp.is_finite() || sum_exp <= 0.0 {
                    return Err(NeuralError::NonFiniteScore("non-finite softmax partition".into()));
                }
                for o in 0..num_tags {
                    d_scores[t][o] /= sum_exp;
                }
                let step_loss = -(scores[gold[t]] - max - sum_exp.ln());
                if !step_loss.is_finite() {
                    return Err(NeuralError::NonFiniteScore("non-finite primary softmax loss".into()));
                }
                loss += step_loss;
                d_scores[t][gold[t]] -= 1.0;
            }
            if !loss.is_finite() {
                return Err(NeuralError::NonFiniteScore("non-finite primary softmax loss".into()));
            }
            Ok(loss)
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn auxiliary_head_pass(
        &self,
        word_fwd_h: &[Vec<f64>],
        word_bwd_h: &[Vec<f64>],
        labels: &[String],
        label_ids: &HashMap<String, usize>,
        idx_w: usize,
        idx_b: usize,
        d_word_fwd_h: &mut [Vec<f64>],
        d_word_bwd_h: &mut [Vec<f64>],
        worker: &mut Worker,
    ) -> Result<f64, NeuralError> {
        let n = labels.len();
        let num_outcomes = label_ids.len();
        let word_hidden = self.settings.word_hidden;
        let weight = self.settings.aux_loss_weight;
        let head_weights = self.optimizer.params(idx_w);
        let head_bias = &self.optimizer.params(idx_b)[0];

        let mut scores = vec![0.0; num_outcomes];
        let mut probs = vec![0.0; num_outcomes];
        let mut loss = 0.0f64;

        for t in 0..n {
            let Some(&gold) = label_ids.get(&labels[t]) else {
                continue;
            };
            let fh = &word_fwd_h[t];
            let bh = &word_bwd_h[n - 1 - t];
            let mut max = f64::NEG_INFINITY;
            for o in 0..num_outcomes {
                let w = &head_weights[o];
                let mut sum = head_bias[o];
                for j in 0..word_hidden {
                    sum += w[j] * fh[j];
                }
                for j in 0..word_hidden {
                    sum += w[word_hidden + j] * bh[j];
                }
                if !sum.is_finite() {
                    return Err(NeuralError::NonFiniteScore("non-finite auxiliary head score".into()));
                }
                scores[o] = sum;
                if sum > max {
                    max = sum;
                }
            }
            let mut sum_exp = 0.0f64;
            for o in 0..num_outcomes {
                let e = (scores[o] - max).exp();
                probs[o] = e;
                sum_exp += e;
            }
            if !sum_exp.is_finite() || sum_exp <= 0.0 {
                return Err(NeuralError::NonFiniteScore("non-finite auxiliary softmax partition".into()));
            }
            for o in 0..num_outcomes {
                probs[o] /= sum_exp;
            }
            let step_loss = weight * -(scores[gold] - max - sum_exp.ln());
            if !step_loss.is_finite() {
                return Err(NeuralError::NonFiniteScore("non-finite auxiliary softmax loss".into()));
            }
            loss += step_loss;

            let dfh = &mut d_word_fwd_h[t];
            let dbh = &mut d_word_bwd_h[n - 1 - t];
            for o in 0..num_outcomes {
                let ds = weight * (probs[o] - if o == gold { 1.0 } else { 0.0 });
                worker.buffers[idx_b][0][o] += ds;
                let w = &head_weights[o];
                let dw = &mut worker.buffers[idx_w][o];
                for j in 0..word_hidden {
                    dw[j] += ds * fh[j];
                    dw[word_hidden + j] += ds * bh[j];
                    dfh[j] += ds * w[j];
                    dbh[j] += ds * w[word_hidden + j];
                }
            }
        }
        if !loss.is_finite() {
            return Err(NeuralError::NonFiniteScore("non-finite auxiliary softmax loss".into()));
        }
        Ok(loss)
    }

    /// Syncs layer weights from `self.char_forward`..`self.word_backward2` into the optimizer before an optimizer step.
    pub(super) fn sync_layers_to_optimizer(&mut self) {
        self.optimizer.params_mut(self.idx_cf_w).clone_from_slice(self.char_forward.w());
        self.optimizer.params_mut(self.idx_cf_u).clone_from_slice(self.char_forward.u());
        self.optimizer.params_mut(self.idx_cf_b)[0].copy_from_slice(self.char_forward.b());

        self.optimizer.params_mut(self.idx_cb_w).clone_from_slice(self.char_backward.w());
        self.optimizer.params_mut(self.idx_cb_u).clone_from_slice(self.char_backward.u());
        self.optimizer.params_mut(self.idx_cb_b)[0].copy_from_slice(self.char_backward.b());

        self.optimizer.params_mut(self.idx_wf_w).clone_from_slice(self.word_forward.w());
        self.optimizer.params_mut(self.idx_wf_u).clone_from_slice(self.word_forward.u());
        self.optimizer.params_mut(self.idx_wf_b)[0].copy_from_slice(self.word_forward.b());

        self.optimizer.params_mut(self.idx_wb_w).clone_from_slice(self.word_backward.w());
        self.optimizer.params_mut(self.idx_wb_u).clone_from_slice(self.word_backward.u());
        self.optimizer.params_mut(self.idx_wb_b)[0].copy_from_slice(self.word_backward.b());

        if let (Some(wf2), Some(wb2)) = (&self.word_forward2, &self.word_backward2) {
            self.optimizer.params_mut(self.idx_wf2_w.unwrap()).clone_from_slice(wf2.w());
            self.optimizer.params_mut(self.idx_wf2_u.unwrap()).clone_from_slice(wf2.u());
            self.optimizer.params_mut(self.idx_wf2_b.unwrap())[0].copy_from_slice(wf2.b());

            self.optimizer.params_mut(self.idx_wb2_w.unwrap()).clone_from_slice(wb2.w());
            self.optimizer.params_mut(self.idx_wb2_u.unwrap()).clone_from_slice(wb2.u());
            self.optimizer.params_mut(self.idx_wb2_b.unwrap())[0].copy_from_slice(wb2.b());
        }
    }

    pub(super) fn sync_optimizer_to_layers(&mut self) {
        self.char_forward.w_mut().clone_from_slice(self.optimizer.params(self.idx_cf_w));
        self.char_forward.u_mut().clone_from_slice(self.optimizer.params(self.idx_cf_u));
        self.char_forward.b_mut().copy_from_slice(&self.optimizer.params(self.idx_cf_b)[0]);

        self.char_backward.w_mut().clone_from_slice(self.optimizer.params(self.idx_cb_w));
        self.char_backward.u_mut().clone_from_slice(self.optimizer.params(self.idx_cb_u));
        self.char_backward.b_mut().copy_from_slice(&self.optimizer.params(self.idx_cb_b)[0]);

        self.word_forward.w_mut().clone_from_slice(self.optimizer.params(self.idx_wf_w));
        self.word_forward.u_mut().clone_from_slice(self.optimizer.params(self.idx_wf_u));
        self.word_forward.b_mut().copy_from_slice(&self.optimizer.params(self.idx_wf_b)[0]);

        self.word_backward.w_mut().clone_from_slice(self.optimizer.params(self.idx_wb_w));
        self.word_backward.u_mut().clone_from_slice(self.optimizer.params(self.idx_wb_u));
        self.word_backward.b_mut().copy_from_slice(&self.optimizer.params(self.idx_wb_b)[0]);

        if let (Some(wf2), Some(wb2)) = (&mut self.word_forward2, &mut self.word_backward2) {
            wf2.w_mut().clone_from_slice(self.optimizer.params(self.idx_wf2_w.unwrap()));
            wf2.u_mut().clone_from_slice(self.optimizer.params(self.idx_wf2_u.unwrap()));
            wf2.b_mut().copy_from_slice(&self.optimizer.params(self.idx_wf2_b.unwrap())[0]);

            wb2.w_mut().clone_from_slice(self.optimizer.params(self.idx_wb2_w.unwrap()));
            wb2.u_mut().clone_from_slice(self.optimizer.params(self.idx_wb2_u.unwrap()));
            wb2.b_mut().copy_from_slice(&self.optimizer.params(self.idx_wb2_b.unwrap())[0]);
        }
    }

    pub(super) fn absorb_worker(&mut self, worker: &Worker, touched_tuning_rows: &mut HashMap<usize, Vec<f64>>) {
        self.optimizer.absorb(&worker.buffers);
        if self.pretrained_trainable.is_some() {
            for (&row, src) in &worker.tuning_row_grads {
                let dst = touched_tuning_rows.entry(row).or_insert_with(|| vec![0.0; self.pretrained_dim]);
                for i in 0..self.pretrained_dim {
                    dst[i] += src[i];
                }
            }
        }
    }

    pub(super) fn step_batch(
        &mut self,
        scale: f64,
        lr: f64,
        step_num: i32,
        touched_tuning_rows: &mut HashMap<usize, Vec<f64>>,
    ) -> Result<(), NeuralError> {
        self.sync_layers_to_optimizer();
        self.optimizer.scale_gradients(scale);
        if self.pretrained_trainable.is_some() && !touched_tuning_rows.is_empty() {
            for g in touched_tuning_rows.values_mut() {
                for v in g {
                    *v *= scale;
                }
            }
        }
        if self.settings.max_grad_norm > 0.0 {
            if self.pretrained_trainable.is_none() || touched_tuning_rows.is_empty() {
                self.optimizer.clip_gradients(self.settings.max_grad_norm)?;
            } else {
                let mut total_sq = 0.0f64;
                let mut max_scale = 0.0f64;
                for g in self.optimizer.gradients() {
                    for row in g {
                        for &v in row {
                            if !v.is_finite() {
                                return Err(NeuralError::NonFiniteScore("gradient must be finite".into()));
                            }
                            total_sq += v * v;
                            max_scale = max_scale.max(v.abs());
                        }
                    }
                }
                for g in touched_tuning_rows.values() {
                    for &v in g {
                        if !v.is_finite() {
                            return Err(NeuralError::NonFiniteScore("gradient must be finite".into()));
                        }
                        total_sq += v * v;
                        max_scale = max_scale.max(v.abs());
                    }
                }
                if max_scale > 0.0 && (!total_sq.is_finite() || total_sq < f64::MIN_POSITIVE) {
                    let mut norm_sq = 0.0f64;
                    for g in self.optimizer.gradients() {
                        for row in g {
                            for &v in row {
                                let n = v / max_scale;
                                norm_sq += n * n;
                            }
                        }
                    }
                    for g in touched_tuning_rows.values() {
                        for &v in g {
                            let n = v / max_scale;
                            norm_sq += n * n;
                        }
                    }
                    let unit_norm = norm_sq.sqrt();
                    if unit_norm > self.settings.max_grad_norm / max_scale {
                        let magnitude = self.settings.max_grad_norm / unit_norm;
                        let clip = magnitude / max_scale;
                        if clip >= f64::MIN_POSITIVE {
                            self.optimizer.scale_gradients(clip);
                            for g in touched_tuning_rows.values_mut() {
                                for v in g {
                                    *v *= clip;
                                }
                            }
                        } else {
                            for g in self.optimizer.gradients_mut() {
                                for row in g {
                                    for v in row {
                                        *v = (*v / max_scale) * magnitude;
                                    }
                                }
                            }
                            for g in touched_tuning_rows.values_mut() {
                                for v in g {
                                    *v = (*v / max_scale) * magnitude;
                                }
                            }
                        }
                    }
                } else {
                    let norm = total_sq.sqrt();
                    if norm > self.settings.max_grad_norm {
                        let clip = self.settings.max_grad_norm / norm;
                        self.optimizer.scale_gradients(clip);
                        for g in touched_tuning_rows.values_mut() {
                            for v in g {
                                *v *= clip;
                            }
                        }
                    }
                }
            }
        }
        self.optimizer.step(lr, step_num)?;
        self.sync_optimizer_to_layers();

        if let (Some(trainable), Some(tm), Some(tv)) =
            (&mut self.pretrained_trainable, &mut self.tuning_m, &mut self.tuning_v)
            && !touched_tuning_rows.is_empty()
        {
            let tuning_lr = lr * self.settings.pretrained_tuning;
            if !tuning_lr.is_finite() {
                return Err(NeuralError::NonFiniteScore("scaled learning rate must be finite".into()));
            }
            let bias_correction = (1.0 - 0.999_f64.powi(step_num)).sqrt() / (1.0 - 0.9_f64.powi(step_num));
            let step_size = tuning_lr * bias_correction;
            if !step_size.is_finite() {
                return Err(NeuralError::NonFiniteScore("pretrained tuning step size must be finite".into()));
            }
            for (&row, g) in touched_tuning_rows.iter() {
                let p = &mut trainable[row];
                let m = &mut tm[row];
                let v = &mut tv[row];
                for i in 0..self.pretrained_dim {
                    let gi = g[i];
                    let first_moment = 0.9 * m[i] + 0.1 * gi;
                    let second_moment = 0.999 * v[i] + 0.001 * gi * gi;
                    let updated = p[i] - step_size * first_moment / (second_moment.sqrt() + 1e-8);
                    if !gi.is_finite()
                        || !first_moment.is_finite()
                        || !second_moment.is_finite()
                        || !updated.is_finite()
                    {
                        return Err(NeuralError::NonFiniteScore(format!(
                            "pretrained tuning Adam update must be finite at row {row}, component {i}"
                        )));
                    }
                    m[i] = first_moment;
                    v[i] = second_moment;
                    p[i] = updated;
                }
            }
        }
        Ok(())
    }

    /// Snapshots the current parameters into a frozen [`BilstmPosModel`].
    pub fn to_model(&self) -> Result<BilstmPosModel, NeuralError> {
        let word_embeddings = copy_matrix(self.optimizer.params(self.idx_word_emb));
        let char_embeddings = copy_matrix(self.optimizer.params(self.idx_char_emb));
        let output_weights = copy_matrix(self.optimizer.params(self.idx_out_w));
        let output_bias = self.optimizer.params(self.idx_out_b)[0].clone();

        let char_forward = copy_lstm(&self.char_forward)?;
        let char_backward = copy_lstm(&self.char_backward)?;
        let word_forward = copy_lstm(&self.word_forward)?;
        let word_backward = copy_lstm(&self.word_backward)?;
        let word_forward2 = self.word_forward2.as_ref().map(copy_lstm).transpose()?;
        let word_backward2 = self.word_backward2.as_ref().map(copy_lstm).transpose()?;

        let pretrained_out = if let Some(ref trainable) = self.pretrained_trainable {
            let mut out = vec![vec![0.0f32; self.pretrained_dim]; trainable.len()];
            for (r, row) in trainable.iter().enumerate() {
                for (i, &val) in row.iter().enumerate() {
                    let f = val as f32;
                    if !val.is_finite() || !f.is_finite() {
                        return Err(NeuralError::NonFiniteScore(format!(
                            "non-finite tuned pretrained vector component at row {r}, index {i}"
                        )));
                    }
                    out[r][i] = f;
                }
            }
            Some(out)
        } else {
            self.pretrained_vectors.clone()
        };

        let transition_weights = self.idx_crf_tr.map(|idx| copy_matrix(self.optimizer.params(idx)));
        let start_weights = self.idx_crf_st.map(|idx| self.optimizer.params(idx)[0].clone());
        let end_weights = self.idx_crf_en.map(|idx| self.optimizer.params(idx)[0].clone());
        let adapter_weights = self.idx_adapter_w.map(|idx| copy_matrix(self.optimizer.params(idx)));
        let adapter_bias = self.idx_adapter_b.map(|idx| self.optimizer.params(idx)[0].clone());

        BilstmPosModel::new(
            self.word_entries.clone(),
            self.char_entries.clone(),
            self.tags.clone(),
            word_embeddings,
            char_embeddings,
            char_forward,
            char_backward,
            word_forward,
            word_backward,
            word_forward2,
            word_backward2,
            output_weights,
            output_bias,
            self.settings.max_word_length,
            if pretrained_out.is_some() { Some(self.pretrained_entries.clone()) } else { None },
            pretrained_out,
            transition_weights,
            start_weights,
            end_weights,
            adapter_weights,
            adapter_bias,
        )
    }
}

fn copy_matrix(src: &[Vec<f64>]) -> Vec<Vec<f64>> {
    src.to_vec()
}

fn copy_lstm(src: &LstmLayer) -> Result<LstmLayer, NeuralError> {
    LstmLayer::of_weights(
        src.input_size(),
        src.hidden_size(),
        copy_matrix(src.w()),
        copy_matrix(src.u()),
        src.b().to_vec(),
    )
}
