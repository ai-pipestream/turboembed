//! Stateful training context and worker scratch buffers for [`super::BilstmPosTrainer`].

use std::collections::{HashMap, HashSet};

use crate::neural::adam::AdamOptimizer;
use crate::neural::bilstm_model::{BilstmPosModel, UNKNOWN, char_key_from_unit};
use crate::neural::lstm::{ForwardCache, LstmLayer};
use crate::neural::{JavaRng, NeuralError, encode_str_modified_utf8};

use super::settings::{BilstmSettings, MultiTaskSample};

/// Per-worker reusable gradient and forward-cache scratch buffers.
#[derive(Clone, Debug)]
pub struct Worker {
    pub(super) buffers: Vec<Vec<Vec<f64>>>,
    pub(super) tuning_row_grads: HashMap<usize, Vec<f64>>,
    pub(super) word_fwd_cache: Option<ForwardCache>,
    pub(super) word_bwd_cache: Option<ForwardCache>,
    pub(super) word_fwd2_cache: Option<ForwardCache>,
    pub(super) word_bwd2_cache: Option<ForwardCache>,
    pub(super) char_fwd_caches: Vec<ForwardCache>,
    pub(super) char_bwd_caches: Vec<ForwardCache>,
}

impl Worker {
    /// Returns the registered parameter gradient buffers.
    pub fn buffers(&self) -> &[Vec<Vec<f64>>] {
        &self.buffers
    }

    /// Returns a mutable slice of the registered parameter gradient buffers.
    pub fn buffers_mut(&mut self) -> &mut [Vec<Vec<f64>>] {
        &mut self.buffers
    }
}

/// Stateful training context for [`super::BilstmPosTrainer`], exposed for finite-difference gradient verification.
#[derive(Clone, Debug)]
pub struct TrainingContext {
    pub(super) settings: BilstmSettings,
    pub(super) word_entries: Vec<(String, usize)>,
    pub(super) word_ids: HashMap<String, usize>,
    pub(super) char_entries: Vec<(Vec<u8>, usize)>,
    pub(super) char_ids: HashMap<Vec<u8>, usize>,
    pub(super) tags: Vec<String>,
    pub(super) tag_ids: HashMap<String, usize>,
    pub(super) xpos_ids: HashMap<String, usize>,
    pub(super) feats_ids: HashMap<String, usize>,
    pub(super) pretrained_entries: Vec<(String, usize)>,
    pub(super) pretrained_ids: HashMap<String, usize>,
    pub(super) pretrained_vectors: Option<Vec<Vec<f32>>>,
    pub(super) pretrained_trainable: Option<Vec<Vec<f64>>>,
    pub(super) tuning_m: Option<Vec<Vec<f64>>>,
    pub(super) tuning_v: Option<Vec<Vec<f64>>>,
    pub(super) pretrained_dim: usize,
    pub(super) char_forward: LstmLayer,
    pub(super) char_backward: LstmLayer,
    pub(super) word_forward: LstmLayer,
    pub(super) word_backward: LstmLayer,
    pub(super) word_forward2: Option<LstmLayer>,
    pub(super) word_backward2: Option<LstmLayer>,
    pub(super) optimizer: AdamOptimizer,
    pub(super) idx_word_emb: usize,
    pub(super) idx_char_emb: usize,
    pub(super) idx_cf_w: usize,
    pub(super) idx_cf_u: usize,
    pub(super) idx_cf_b: usize,
    pub(super) idx_cb_w: usize,
    pub(super) idx_cb_u: usize,
    pub(super) idx_cb_b: usize,
    pub(super) idx_wf_w: usize,
    pub(super) idx_wf_u: usize,
    pub(super) idx_wf_b: usize,
    pub(super) idx_wb_w: usize,
    pub(super) idx_wb_u: usize,
    pub(super) idx_wb_b: usize,
    pub(super) idx_out_w: usize,
    pub(super) idx_out_b: usize,
    pub(super) idx_wf2_w: Option<usize>,
    pub(super) idx_wf2_u: Option<usize>,
    pub(super) idx_wf2_b: Option<usize>,
    pub(super) idx_wb2_w: Option<usize>,
    pub(super) idx_wb2_u: Option<usize>,
    pub(super) idx_wb2_b: Option<usize>,
    pub(super) idx_crf_tr: Option<usize>,
    pub(super) idx_crf_st: Option<usize>,
    pub(super) idx_crf_en: Option<usize>,
    pub(super) idx_xpos_w: Option<usize>,
    pub(super) idx_xpos_b: Option<usize>,
    pub(super) idx_feats_w: Option<usize>,
    pub(super) idx_feats_b: Option<usize>,
    pub(super) idx_adapter_w: Option<usize>,
    pub(super) idx_adapter_b: Option<usize>,
}

impl TrainingContext {
    /// Builds and initializes a `TrainingContext` over `corpus`.
    pub fn build<F>(
        corpus: &[MultiTaskSample],
        settings: BilstmSettings,
        pretrained_fn: Option<F>,
        additional_vocab: Option<&[String]>,
    ) -> Result<Self, NeuralError>
    where
        F: Fn(&str) -> Option<Vec<f32>>,
    {
        let mut word_order: Vec<String> = Vec::new();
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        let mut char_order: Vec<Vec<u8>> = Vec::new();
        let mut char_counts: HashMap<Vec<u8>, usize> = HashMap::new();
        let mut tag_list: Vec<String> = Vec::new();
        let mut tag_ids: HashMap<String, usize> = HashMap::new();
        let mut xpos_list: Vec<String> = Vec::new();
        let mut xpos_ids: HashMap<String, usize> = HashMap::new();
        let mut feats_list: Vec<String> = Vec::new();
        let mut feats_ids: HashMap<String, usize> = HashMap::new();

        for sample in corpus {
            sample.validate()?;
            let tokens = &sample.sentence;
            let sample_tags = &sample.tags;
            let sample_xpos = sample.xpos.as_ref();
            let sample_feats = sample.feats.as_ref();
            for raw in tokens {
                let norm = BilstmPosModel::normalize(raw);
                if let Some(c) = word_counts.get_mut(&norm) {
                    *c += 1;
                } else {
                    word_order.push(norm.clone());
                    word_counts.insert(norm, 1);
                }
                for unit in raw.encode_utf16() {
                    let key = char_key_from_unit(unit);
                    if let Some(c) = char_counts.get_mut(&key) {
                        *c += 1;
                    } else {
                        char_order.push(key.clone());
                        char_counts.insert(key, 1);
                    }
                }
            }
            for tag in sample_tags {
                if !tag_ids.contains_key(tag) {
                    tag_ids.insert(tag.clone(), 0);
                    tag_list.push(tag.clone());
                }
            }
            if settings.aux_loss_weight > 0.0 {
                if let Some(xp) = sample_xpos {
                    for tag in xp {
                        if !xpos_ids.contains_key(tag) {
                            xpos_ids.insert(tag.clone(), 0);
                            xpos_list.push(tag.clone());
                        }
                    }
                }
                if let Some(ft) = sample_feats {
                    for tag in ft {
                        if !feats_ids.contains_key(tag) {
                            feats_ids.insert(tag.clone(), 0);
                            feats_list.push(tag.clone());
                        }
                    }
                }
            }
        }

        if tag_list.is_empty() {
            return Err(NeuralError::InvalidArgument("Training corpus must contain at least one tag".into()));
        }

        tag_list.sort_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
        for (i, t) in tag_list.iter().enumerate() {
            tag_ids.insert(t.clone(), i);
        }
        xpos_list.sort_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
        for (i, t) in xpos_list.iter().enumerate() {
            xpos_ids.insert(t.clone(), i);
        }
        feats_list.sort_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
        for (i, t) in feats_list.iter().enumerate() {
            feats_ids.insert(t.clone(), i);
        }

        let mut word_entries = Vec::new();
        let mut word_ids = HashMap::new();
        word_entries.push((UNKNOWN.to_owned(), 0));
        word_ids.insert(UNKNOWN.to_owned(), 0);
        for w in &word_order {
            if word_counts[w] >= settings.word_cutoff && !word_ids.contains_key(w) {
                let id = word_entries.len();
                word_entries.push((w.clone(), id));
                word_ids.insert(w.clone(), id);
            }
        }

        let mut char_entries = Vec::new();
        let mut char_ids = HashMap::new();
        let unk_bytes = encode_str_modified_utf8(UNKNOWN);
        char_entries.push((unk_bytes.clone(), 0));
        char_ids.insert(unk_bytes, 0);
        for ch in &char_order {
            if !char_ids.contains_key(ch) {
                let id = char_entries.len();
                char_entries.push((ch.clone(), id));
                char_ids.insert(ch.clone(), id);
            }
        }

        let mut pretrained_entries = Vec::new();
        let mut pretrained_ids = HashMap::new();
        let mut pretrained_dim = 0usize;
        let pretrained_vectors = if let Some(ref f) = pretrained_fn {
            let mut candidates = word_order.clone();
            let mut seen: HashSet<String> = candidates.iter().cloned().collect();
            if let Some(extra) = additional_vocab {
                for raw in extra {
                    let norm = BilstmPosModel::normalize(raw);
                    if seen.insert(norm.clone()) {
                        candidates.push(norm);
                    }
                }
            }
            let mut rows: Vec<Vec<f32>> = Vec::new();
            for word in candidates {
                if let Some(vec) = f(&word) {
                    if vec.is_empty() {
                        return Err(NeuralError::InvalidArgument(format!("empty pretrained vector for '{word}'")));
                    }
                    if pretrained_dim == 0 {
                        pretrained_dim = vec.len();
                    } else if vec.len() != pretrained_dim {
                        return Err(NeuralError::InvalidArgument(format!(
                            "pretrained vector length mismatch at '{word}': expected {pretrained_dim}, got {}",
                            vec.len()
                        )));
                    }
                    for &v in &vec {
                        if !v.is_finite() {
                            return Err(NeuralError::InvalidArgument(format!(
                                "non-finite pretrained vector component at '{word}'"
                            )));
                        }
                    }
                    let id = rows.len();
                    pretrained_entries.push((word.clone(), id));
                    pretrained_ids.insert(word, id);
                    rows.push(vec);
                }
            }
            if rows.is_empty() {
                return Err(NeuralError::InvalidArgument(
                    "pretrainedVectors did not return a vector for any vocabulary word".into(),
                ));
            }
            Some(rows)
        } else {
            None
        };

        let mut rng = JavaRng::new(settings.seed);
        let word_embeddings = random_matrix(word_entries.len(), settings.word_dim, 0.1, &mut rng);
        let char_embeddings = random_matrix(char_entries.len(), settings.char_dim, 0.1, &mut rng);
        let char_forward = LstmLayer::new(settings.char_dim, settings.char_hidden, &mut rng)?;
        let char_backward = LstmLayer::new(settings.char_dim, settings.char_hidden, &mut rng)?;
        let word_input_dim = settings.word_dim + 2 * settings.char_hidden + pretrained_dim;
        let word_forward = LstmLayer::new(word_input_dim, settings.word_hidden, &mut rng)?;
        let word_backward = LstmLayer::new(word_input_dim, settings.word_hidden, &mut rng)?;

        let num_tags = tag_list.len();
        let output_weights = xavier_output(num_tags, 2 * settings.word_hidden, &mut rng);
        let output_bias = vec![0.0; num_tags];

        let crf_tr = if settings.crf { Some(random_matrix(num_tags, num_tags, 0.1, &mut rng)) } else { None };
        let xpos_init = if !xpos_list.is_empty() {
            Some(xavier_output(xpos_list.len(), 2 * settings.word_hidden, &mut rng))
        } else {
            None
        };
        let feats_init = if !feats_list.is_empty() {
            Some(xavier_output(feats_list.len(), 2 * settings.word_hidden, &mut rng))
        } else {
            None
        };
        let (word_forward2, word_backward2) = if settings.encoder_layers == 2 {
            let wf2 = LstmLayer::new(2 * settings.word_hidden, settings.word_hidden, &mut rng)?;
            let wb2 = LstmLayer::new(2 * settings.word_hidden, settings.word_hidden, &mut rng)?;
            (Some(wf2), Some(wb2))
        } else {
            (None, None)
        };

        let mut optimizer = AdamOptimizer::new();
        let idx_word_emb = optimizer.register_matrix(word_embeddings);
        let idx_char_emb = optimizer.register_matrix(char_embeddings);
        let idx_cf_w = optimizer.register_matrix(char_forward.w().to_vec());
        let idx_cf_u = optimizer.register_matrix(char_forward.u().to_vec());
        let idx_cf_b = optimizer.register_vector(char_forward.b().to_vec());
        let idx_cb_w = optimizer.register_matrix(char_backward.w().to_vec());
        let idx_cb_u = optimizer.register_matrix(char_backward.u().to_vec());
        let idx_cb_b = optimizer.register_vector(char_backward.b().to_vec());
        let idx_wf_w = optimizer.register_matrix(word_forward.w().to_vec());
        let idx_wf_u = optimizer.register_matrix(word_forward.u().to_vec());
        let idx_wf_b = optimizer.register_vector(word_forward.b().to_vec());
        let idx_wb_w = optimizer.register_matrix(word_backward.w().to_vec());
        let idx_wb_u = optimizer.register_matrix(word_backward.u().to_vec());
        let idx_wb_b = optimizer.register_vector(word_backward.b().to_vec());
        let idx_out_w = optimizer.register_matrix(output_weights);
        let idx_out_b = optimizer.register_vector(output_bias);

        let (idx_crf_tr, idx_crf_st, idx_crf_en) = if let Some(tr) = crf_tr {
            let st = vec![0.0; num_tags];
            let en = vec![0.0; num_tags];
            (
                Some(optimizer.register_matrix(tr)),
                Some(optimizer.register_vector(st)),
                Some(optimizer.register_vector(en)),
            )
        } else {
            (None, None, None)
        };

        let (idx_wf2_w, idx_wf2_u, idx_wf2_b, idx_wb2_w, idx_wb2_u, idx_wb2_b) =
            if let (Some(wf2), Some(wb2)) = (&word_forward2, &word_backward2) {
                (
                    Some(optimizer.register_matrix(wf2.w().to_vec())),
                    Some(optimizer.register_matrix(wf2.u().to_vec())),
                    Some(optimizer.register_vector(wf2.b().to_vec())),
                    Some(optimizer.register_matrix(wb2.w().to_vec())),
                    Some(optimizer.register_matrix(wb2.u().to_vec())),
                    Some(optimizer.register_vector(wb2.b().to_vec())),
                )
            } else {
                (None, None, None, None, None, None)
            };

        let (idx_xpos_w, idx_xpos_b) = if let Some(xw) = xpos_init {
            let xb = vec![0.0; xpos_list.len()];
            (Some(optimizer.register_matrix(xw)), Some(optimizer.register_vector(xb)))
        } else {
            (None, None)
        };

        let (idx_feats_w, idx_feats_b) = if let Some(fw) = feats_init {
            let fb = vec![0.0; feats_list.len()];
            (Some(optimizer.register_matrix(fw)), Some(optimizer.register_vector(fb)))
        } else {
            (None, None)
        };

        let (pretrained_trainable, tuning_m, tuning_v) = if pretrained_dim > 0 && settings.pretrained_tuning > 0.0 {
            let p_vecs = pretrained_vectors.as_ref().unwrap();
            let rows = p_vecs.len();
            let mut trainable = vec![vec![0.0f64; pretrained_dim]; rows];
            for r in 0..rows {
                for i in 0..pretrained_dim {
                    trainable[r][i] = f64::from(p_vecs[r][i]);
                }
            }
            (
                Some(trainable),
                Some(vec![vec![0.0f64; pretrained_dim]; rows]),
                Some(vec![vec![0.0f64; pretrained_dim]; rows]),
            )
        } else {
            (None, None, None)
        };

        let (idx_adapter_w, idx_adapter_b) = if pretrained_dim > 0 && settings.pretrained_adapter {
            let mut aw = vec![vec![0.0f64; pretrained_dim]; pretrained_dim];
            for (i, row) in aw.iter_mut().enumerate() {
                row[i] = 1.0;
            }
            let ab = vec![0.0f64; pretrained_dim];
            (Some(optimizer.register_matrix(aw)), Some(optimizer.register_vector(ab)))
        } else {
            (None, None)
        };

        Ok(Self {
            settings,
            word_entries,
            word_ids,
            char_entries,
            char_ids,
            tags: tag_list,
            tag_ids,
            xpos_ids,
            feats_ids,
            pretrained_entries,
            pretrained_ids,
            pretrained_vectors,
            pretrained_trainable,
            tuning_m,
            tuning_v,
            pretrained_dim,
            char_forward,
            char_backward,
            word_forward,
            word_backward,
            word_forward2,
            word_backward2,
            optimizer,
            idx_word_emb,
            idx_char_emb,
            idx_cf_w,
            idx_cf_u,
            idx_cf_b,
            idx_cb_w,
            idx_cb_u,
            idx_cb_b,
            idx_wf_w,
            idx_wf_u,
            idx_wf_b,
            idx_wb_w,
            idx_wb_u,
            idx_wb_b,
            idx_out_w,
            idx_out_b,
            idx_wf2_w,
            idx_wf2_u,
            idx_wf2_b,
            idx_wb2_w,
            idx_wb2_u,
            idx_wb2_b,
            idx_crf_tr,
            idx_crf_st,
            idx_crf_en,
            idx_xpos_w,
            idx_xpos_b,
            idx_feats_w,
            idx_feats_b,
            idx_adapter_w,
            idx_adapter_b,
        })
    }

    /// Allocates a new [`Worker`] with scratch gradient buffers for this context.
    pub fn new_worker(&self) -> Worker {
        Worker {
            buffers: self.optimizer.new_gradient_buffers(),
            tuning_row_grads: HashMap::new(),
            word_fwd_cache: None,
            word_bwd_cache: None,
            word_fwd2_cache: None,
            word_bwd2_cache: None,
            char_fwd_caches: Vec::new(),
            char_bwd_caches: Vec::new(),
        }
    }

    /// Zeros all gradient accumulators on `worker`.
    pub fn reset_worker(&self, worker: &mut Worker) {
        AdamOptimizer::zero_buffers(&mut worker.buffers);
        worker.tuning_row_grads.clear();
    }

    /// Materializes a dense copy of the sparse pretrained-tuning row gradients from `worker`.
    pub fn dense_tuning_gradient(&self, worker: &Worker) -> Vec<Vec<f64>> {
        let Some(ref trainable) = self.pretrained_trainable else {
            return Vec::new();
        };
        let mut out = vec![vec![0.0; self.pretrained_dim]; trainable.len()];
        for (&r, row) in &worker.tuning_row_grads {
            out[r].copy_from_slice(row);
        }
        out
    }
}

fn xavier_output(rows: usize, cols: usize, rng: &mut JavaRng) -> Vec<Vec<f64>> {
    let limit = (6.0 / (rows as f64 + cols as f64)).sqrt();
    random_matrix(rows, cols, limit, rng)
}

fn random_matrix(rows: usize, cols: usize, limit: f64, rng: &mut JavaRng) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; cols]; rows];
    for row in &mut m {
        for val in row {
            *val = (rng.next_double() * 2.0 - 1.0) * limit;
        }
    }
    m
}
