//! BiLSTM(-CRF) trainer (`BilstmPosTrainer`, `BilstmSettings`, `MultiTaskSample`).

use std::collections::{HashMap, HashSet};

use crate::neural::adam::AdamOptimizer;
use crate::neural::bilstm_model::{BilstmPosModel, UNKNOWN, char_key_from_unit};
use crate::neural::bilstm_tagger::BilstmPosTagger;
use crate::neural::crf::CrfScorer;
use crate::neural::lstm::{ForwardCache, LstmGradients, LstmLayer};
use crate::neural::{JavaRng, NeuralError, PosSample, encode_str_modified_utf8};

/// A training sentence annotated with primary POS tags and optional auxiliary XPOS and morphological feature tags.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiTaskSample {
    /// Sentence tokens.
    pub sentence: Vec<String>,
    /// Primary POS tags (one per token).
    pub tags: Vec<String>,
    /// Optional fine-grained XPOS tags (one per token).
    pub xpos: Option<Vec<String>>,
    /// Optional morphological feature bundles (one per token).
    pub feats: Option<Vec<String>>,
}

impl MultiTaskSample {
    /// Creates a new `MultiTaskSample`, validating that all present tag arrays match `sentence.len()`.
    pub fn new(
        sentence: impl IntoIterator<Item = impl Into<String>>,
        tags: impl IntoIterator<Item = impl Into<String>>,
        xpos: Option<Vec<String>>,
        feats: Option<Vec<String>>,
    ) -> Result<Self, NeuralError> {
        let sentence: Vec<String> = sentence.into_iter().map(Into::into).collect();
        let tags: Vec<String> = tags.into_iter().map(Into::into).collect();
        if sentence.len() != tags.len() {
            return Err(NeuralError::InvalidArgument("tags must have the same length as sentence".into()));
        }
        if let Some(ref x) = xpos
            && x.len() != sentence.len()
        {
            return Err(NeuralError::InvalidArgument("xpos must have the same length as sentence".into()));
        }
        if let Some(ref f) = feats
            && f.len() != sentence.len()
        {
            return Err(NeuralError::InvalidArgument("feats must have the same length as sentence".into()));
        }
        Ok(Self { sentence, tags, xpos, feats })
    }
}

/// Hyperparameters for [`BilstmPosTrainer`], matching OpenNLP's `BilstmPOSTrainer.Settings`.
#[derive(Clone, Debug, PartialEq)]
pub struct BilstmSettings {
    pub word_dim: usize,
    pub char_dim: usize,
    pub char_hidden: usize,
    pub word_hidden: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub max_grad_norm: f64,
    pub dropout: f64,
    pub word_cutoff: usize,
    pub max_word_length: usize,
    pub seed: i64,
    pub threads: usize,
    pub word_dropout: f64,
    pub learning_rate_half_life: usize,
    pub crf: bool,
    pub encoder_layers: usize,
    pub pretrained_dropout: f64,
    pub encoder_dropout: f64,
    pub aux_loss_weight: f64,
    pub pretrained_tuning: f64,
    pub pretrained_adapter: bool,
}

impl BilstmSettings {
    /// Full 22-parameter constructor with validation matching OpenNLP's `BilstmPOSTrainer.Settings`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        word_dim: i32,
        char_dim: i32,
        char_hidden: i32,
        word_hidden: i32,
        epochs: i32,
        batch_size: i32,
        learning_rate: f64,
        max_grad_norm: f64,
        dropout: f64,
        word_cutoff: i32,
        max_word_length: i32,
        seed: i64,
        threads: i32,
        word_dropout: f64,
        learning_rate_half_life: i32,
        crf: bool,
        encoder_layers: i32,
        pretrained_dropout: f64,
        encoder_dropout: f64,
        aux_loss_weight: f64,
        pretrained_tuning: f64,
        pretrained_adapter: bool,
    ) -> Result<Self, NeuralError> {
        if word_dim <= 0 || char_dim <= 0 || char_hidden <= 0 || word_hidden <= 0 {
            return Err(NeuralError::InvalidArgument("all dimensions must be > 0".into()));
        }
        if epochs <= 0 || batch_size <= 0 {
            return Err(NeuralError::InvalidArgument("epochs and batchSize must be > 0".into()));
        }
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(NeuralError::InvalidArgument("learningRate must be finite and > 0".into()));
        }
        if !max_grad_norm.is_finite() || max_grad_norm <= 0.0 {
            return Err(NeuralError::InvalidArgument("clipNorm must be finite and > 0".into()));
        }
        if !dropout.is_finite() || !(0.0..1.0).contains(&dropout) {
            return Err(NeuralError::InvalidArgument("dropout must be in [0, 1)".into()));
        }
        if !word_dropout.is_finite() || !(0.0..1.0).contains(&word_dropout) {
            return Err(NeuralError::InvalidArgument("wordDropout must be in [0, 1)".into()));
        }
        if !pretrained_dropout.is_finite() || !(0.0..1.0).contains(&pretrained_dropout) {
            return Err(NeuralError::InvalidArgument("pretrainedDropout must be in [0, 1)".into()));
        }
        if !encoder_dropout.is_finite() || !(0.0..1.0).contains(&encoder_dropout) {
            return Err(NeuralError::InvalidArgument("encoderDropout must be in [0, 1)".into()));
        }
        if !aux_loss_weight.is_finite() || aux_loss_weight < 0.0 {
            return Err(NeuralError::InvalidArgument("auxLossWeight must be finite and >= 0".into()));
        }
        if !pretrained_tuning.is_finite() || pretrained_tuning < 0.0 {
            return Err(NeuralError::InvalidArgument("pretrainedTuning must be finite and >= 0".into()));
        }
        if pretrained_adapter && pretrained_tuning > 0.0 {
            return Err(NeuralError::InvalidArgument(
                "pretrainedAdapter and pretrainedTuning must not be combined".into(),
            ));
        }
        if word_cutoff <= 0 || max_word_length <= 0 {
            return Err(NeuralError::InvalidArgument("wordCutoff and maxWordLength must be > 0".into()));
        }
        if threads <= 0 {
            return Err(NeuralError::InvalidArgument("threads must be >= 1".into()));
        }
        if learning_rate_half_life < 0 {
            return Err(NeuralError::InvalidArgument("learningRateHalfLife must be >= 0".into()));
        }
        if !(1..=2).contains(&encoder_layers) {
            return Err(NeuralError::InvalidArgument("encoderLayers must be 1 or 2".into()));
        }
        Ok(Self {
            word_dim: word_dim as usize,
            char_dim: char_dim as usize,
            char_hidden: char_hidden as usize,
            word_hidden: word_hidden as usize,
            epochs: epochs as usize,
            batch_size: batch_size as usize,
            learning_rate,
            max_grad_norm,
            dropout,
            word_cutoff: word_cutoff as usize,
            max_word_length: max_word_length as usize,
            seed,
            threads: threads as usize,
            word_dropout,
            learning_rate_half_life: learning_rate_half_life as usize,
            crf,
            encoder_layers: encoder_layers as usize,
            pretrained_dropout,
            encoder_dropout,
            aux_loss_weight,
            pretrained_tuning,
            pretrained_adapter,
        })
    }

    /// Default hyperparameters matching `BilstmPOSTrainer.Settings.defaults()`.
    pub fn defaults() -> Self {
        Self::new(100, 25, 50, 128, 20, 16, 1e-3, 5.0, 0.33, 2, 40, 17, 1, 0.0, 0, false, 1, 0.0, 0.0, 1.0, 0.0, false)
            .unwrap()
    }
}

/// Per-worker reusable gradient and forward-cache scratch buffers.
#[derive(Clone, Debug)]
pub struct Worker {
    buffers: Vec<Vec<Vec<f64>>>,
    tuning_row_grads: HashMap<usize, Vec<f64>>,
    word_fwd_cache: Option<ForwardCache>,
    word_bwd_cache: Option<ForwardCache>,
    word_fwd2_cache: Option<ForwardCache>,
    word_bwd2_cache: Option<ForwardCache>,
    char_fwd_caches: Vec<ForwardCache>,
    char_bwd_caches: Vec<ForwardCache>,
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

/// Stateful training context for [`BilstmPosTrainer`], exposed for finite-difference gradient verification.
#[derive(Clone, Debug)]
pub struct TrainingContext {
    settings: BilstmSettings,
    word_entries: Vec<(String, usize)>,
    word_ids: HashMap<String, usize>,
    char_entries: Vec<(Vec<u8>, usize)>,
    char_ids: HashMap<Vec<u8>, usize>,
    tags: Vec<String>,
    tag_ids: HashMap<String, usize>,
    xpos_ids: HashMap<String, usize>,
    feats_ids: HashMap<String, usize>,
    pretrained_entries: Vec<(String, usize)>,
    pretrained_ids: HashMap<String, usize>,
    pretrained_vectors: Option<Vec<Vec<f32>>>,
    pretrained_trainable: Option<Vec<Vec<f64>>>,
    tuning_m: Option<Vec<Vec<f64>>>,
    tuning_v: Option<Vec<Vec<f64>>>,
    pretrained_dim: usize,
    char_forward: LstmLayer,
    char_backward: LstmLayer,
    word_forward: LstmLayer,
    word_backward: LstmLayer,
    word_forward2: Option<LstmLayer>,
    word_backward2: Option<LstmLayer>,
    optimizer: AdamOptimizer,
    idx_word_emb: usize,
    idx_char_emb: usize,
    idx_cf_w: usize,
    idx_cf_u: usize,
    idx_cf_b: usize,
    idx_cb_w: usize,
    idx_cb_u: usize,
    idx_cb_b: usize,
    idx_wf_w: usize,
    idx_wf_u: usize,
    idx_wf_b: usize,
    idx_wb_w: usize,
    idx_wb_u: usize,
    idx_wb_b: usize,
    idx_out_w: usize,
    idx_out_b: usize,
    idx_wf2_w: Option<usize>,
    idx_wf2_u: Option<usize>,
    idx_wf2_b: Option<usize>,
    idx_wb2_w: Option<usize>,
    idx_wb2_u: Option<usize>,
    idx_wb2_b: Option<usize>,
    idx_crf_tr: Option<usize>,
    idx_crf_st: Option<usize>,
    idx_crf_en: Option<usize>,
    idx_xpos_w: Option<usize>,
    idx_xpos_b: Option<usize>,
    idx_feats_w: Option<usize>,
    idx_feats_b: Option<usize>,
    idx_adapter_w: Option<usize>,
    idx_adapter_b: Option<usize>,
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
        let mut buffers = self.optimizer.new_gradient_buffers();
        if self.pretrained_trainable.is_some() {
            buffers.push(vec![vec![0.0; self.pretrained_dim]; 1]);
        }
        Worker {
            buffers,
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

    /// Syncs layer weights from `self.char_forward`..`self.word_backward2` into the optimizer before an optimizer step, or vice versa.
    fn sync_layers_to_optimizer(&mut self) {
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

    fn sync_optimizer_to_layers(&mut self) {
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

    /// Runs forward and backward passes for one sentence sample, accumulating parameter gradients into `worker`.
    pub fn sentence_gradients(
        &self,
        sample: &MultiTaskSample,
        rng: &mut JavaRng,
        worker: &mut Worker,
    ) -> Result<f64, NeuralError> {
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
                emissions[t][o] = sum;
            }
            gold[t] = self.tag_ids[&sample_tags[t]];
        }

        let mut d_scores = vec![vec![0.0; num_tags]; n];
        let mut sentence_loss = if self.settings.crf {
            let idx_tr = self.idx_crf_tr.unwrap();
            let idx_st = self.idx_crf_st.unwrap();
            let idx_en = self.idx_crf_en.unwrap();
            let tr = self.optimizer.params(idx_tr);
            let st = &self.optimizer.params(idx_st)[0];
            let en = &self.optimizer.params(idx_en)[0];
            let mut d_tr = worker.buffers[idx_tr].clone();
            let mut d_st = worker.buffers[idx_st][0].clone();
            let mut d_en = worker.buffers[idx_en][0].clone();
            let loss = CrfScorer::loss_and_gradients(
                &emissions,
                tr,
                st,
                en,
                &gold,
                &mut d_scores,
                &mut d_tr,
                &mut d_st,
                &mut d_en,
            )?;
            worker.buffers[idx_tr] = d_tr;
            worker.buffers[idx_st][0] = d_st;
            worker.buffers[idx_en][0] = d_en;
            loss
        } else {
            let mut loss = 0.0f64;
            for t in 0..n {
                let scores = &emissions[t];
                let mut max = f64::NEG_INFINITY;
                for &s in scores {
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
                for o in 0..num_tags {
                    d_scores[t][o] /= sum_exp;
                }
                loss += -(scores[gold[t]] - max - sum_exp.ln());
                d_scores[t][gold[t]] -= 1.0;
            }
            loss
        };

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
            );
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
            );
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

    #[allow(clippy::too_many_arguments)]
    fn auxiliary_head_pass(
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
    ) -> f64 {
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
            for o in 0..num_outcomes {
                probs[o] /= sum_exp;
            }
            loss += weight * -(scores[gold] - max - sum_exp.ln());

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
        loss
    }

    fn absorb_worker(&mut self, worker: &Worker, touched_tuning_rows: &mut HashMap<usize, Vec<f64>>) {
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

    fn step_batch(
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
                let dense_norm = self.optimizer.global_norm()?;
                let mut total_sq = dense_norm * dense_norm;
                for g in touched_tuning_rows.values() {
                    for &v in g {
                        total_sq += v * v;
                    }
                }
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
        self.optimizer.step(lr, step_num)?;
        self.sync_optimizer_to_layers();

        if let (Some(trainable), Some(tm), Some(tv)) =
            (&mut self.pretrained_trainable, &mut self.tuning_m, &mut self.tuning_v)
            && !touched_tuning_rows.is_empty()
        {
            let tuning_lr = lr * self.settings.pretrained_tuning;
            let bias_correction = (1.0 - 0.999_f64.powi(step_num)).sqrt() / (1.0 - 0.9_f64.powi(step_num));
            let step_size = tuning_lr * bias_correction;
            for (&row, g) in touched_tuning_rows.iter() {
                let p = &mut trainable[row];
                let m = &mut tm[row];
                let v = &mut tv[row];
                for i in 0..self.pretrained_dim {
                    let gi = g[i];
                    m[i] = 0.9 * m[i] + 0.1 * gi;
                    v[i] = 0.999 * v[i] + 0.001 * gi * gi;
                    p[i] -= step_size * m[i] / (v[i].sqrt() + 1e-8);
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
                    out[r][i] = val as f32;
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

    // Accessors for gradient tests:

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
        let corpus: Vec<MultiTaskSample> = samples.iter().filter(|s| !s.sentence.is_empty()).cloned().collect();
        if corpus.is_empty() {
            return Err(NeuralError::InvalidArgument("Training stream produced no samples".into()));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural::bilstm_model::char_vocabulary_from_strs;

    fn tiny_corpus() -> Vec<PosSample> {
        vec![
            PosSample::new(["The", "cat", "sat"], ["D", "N", "V"]).unwrap(),
            PosSample::new(["A", "dog", "ran"], ["D", "N", "V"]).unwrap(),
        ]
    }

    fn usage_corpus() -> Vec<PosSample> {
        let distinct = vec![
            PosSample::new(["the", "dog", "barks"], ["DT", "NN", "VBZ"]).unwrap(),
            PosSample::new(["the", "cat", "sleeps"], ["DT", "NN", "VBZ"]).unwrap(),
            PosSample::new(["a", "bird", "sings"], ["DT", "NN", "VBZ"]).unwrap(),
            PosSample::new(["dogs", "bark"], ["NNS", "VBP"]).unwrap(),
            PosSample::new(["cats", "sleep"], ["NNS", "VBP"]).unwrap(),
        ];
        let mut out = Vec::with_capacity(40 * distinct.len());
        for _ in 0..40 {
            out.extend_from_slice(&distinct);
        }
        out
    }

    #[test]
    fn serialization_round_trip_preserves_scores_pretrained_and_adapter() {
        let settings = BilstmSettings::new(
            8, 4, 4, 8, 3, 2, 5e-3, 5.0, 0.1, 1, 12, 7, 2, 0.0, 0, false, 1, 0.0, 0.0, 1.0, 0.0, false,
        )
        .unwrap();
        let model = BilstmPosTrainer::train(&tiny_corpus(), settings.clone()).unwrap();
        let mut buf = Vec::new();
        model.serialize(&mut buf).unwrap();
        let loaded = BilstmPosModel::load(&buf[..]).unwrap();
        assert_eq!(model.tags(), loaded.tags());
        let sent = ["The", "dog", "sat"];
        assert_eq!(model.score(&sent).unwrap(), loaded.score(&sent).unwrap());

        // Pretrained table + unseen lexicon word
        let vec_fn = |w: &str| Some(vec![w.len() as f32, 1.0]);
        let extra = vec!["unseen".to_owned()];
        let model_p =
            BilstmPosTrainer::train_with_vectors(&tiny_corpus(), settings, Some(vec_fn), Some(&extra), None).unwrap();
        buf.clear();
        model_p.serialize(&mut buf).unwrap();
        let loaded_p = BilstmPosModel::load(&buf[..]).unwrap();
        assert_eq!(model_p.word_representation("unseen").unwrap(), loaded_p.word_representation("unseen").unwrap());

        // Adapter model (ONLP-BLPT-5)
        let adapter_settings = BilstmSettings::new(
            8, 4, 4, 8, 3, 2, 5e-3, 5.0, 0.1, 1, 12, 7, 2, 0.0, 0, false, 1, 0.0, 0.0, 1.0, 0.0, true,
        )
        .unwrap();
        let model_a =
            BilstmPosTrainer::train_with_vectors(&tiny_corpus(), adapter_settings, Some(vec_fn), Some(&extra), None)
                .unwrap();
        buf.clear();
        model_a.serialize(&mut buf).unwrap();
        assert_eq!(&buf[2..2 + buf[1] as usize], b"ONLP-BLPT-5");
        let loaded_a = BilstmPosModel::load(&buf[..]).unwrap();
        let sent4 = ["The", "dog", "sat", "unseen"];
        assert_eq!(model_a.score(&sent4).unwrap(), loaded_a.score(&sent4).unwrap());
    }

    #[test]
    fn serialization_round_trip_preserves_one_and_two_layer_crf() {
        for layers in [1, 2] {
            let settings = BilstmSettings::new(
                8, 4, 4, 8, 3, 2, 5e-3, 5.0, 0.1, 1, 12, 7, 2, 0.0, 0, true, layers, 0.0, 0.0, 1.0, 0.0, false,
            )
            .unwrap();
            let model = BilstmPosTrainer::train(&tiny_corpus(), settings).unwrap();
            let mut buf = Vec::new();
            model.serialize(&mut buf).unwrap();
            let expected_magic = if layers == 1 { b"ONLP-BLPT-2" } else { b"ONLP-BLPT-4" };
            assert_eq!(&buf[2..2 + buf[1] as usize], expected_magic);
            let loaded = BilstmPosModel::load(&buf[..]).unwrap();
            assert!(loaded.is_crf());
            let sent = ["The", "dog", "sat"];
            assert_eq!(
                BilstmPosTagger::new(model).tag(&sent).unwrap(),
                BilstmPosTagger::new(loaded).tag(&sent).unwrap()
            );
        }
    }

    #[test]
    fn usage_and_edge_cases_match_java_deterministic_outputs() {
        let settings = BilstmSettings::new(
            16, 8, 8, 16, 60, 8, 0.05, 5.0, 0.0, 1, 16, 17, 1, 0.0, 0, false, 1, 0.0, 0.0, 1.0, 0.0, false,
        )
        .unwrap();
        let model = BilstmPosTrainer::train(&usage_corpus(), settings).unwrap();
        let tagger = BilstmPosTagger::new(model.clone());

        assert_eq!(tagger.tag(&["the", "cat", "barks"]).unwrap(), vec!["DT", "NN", "VBZ"]);
        assert_eq!(tagger.tag(&["dogs", "bark"]).unwrap(), vec!["NNS", "VBP"]);
        assert_eq!(tagger.tag(&["dog"]).unwrap(), vec!["VBZ"]);
        assert!(tagger.tag(&Vec::<&str>::new()).unwrap().is_empty());

        let overlong = "z".repeat(40);
        let prefix = &overlong[..model.max_word_length()];
        assert_eq!(tagger.tag(&[prefix]).unwrap(), tagger.tag(&[overlong.as_str()]).unwrap());

        // An empty token has no characters, so as in Java its character
        // states stay zero instead of running the character LSTMs.
        assert!(model.char_ids("").is_empty());
        let empty = model.word_representation("").unwrap();
        let word_dim = model.word_embeddings[0].len();
        let char_hidden = model.char_forward.hidden_size();
        assert!(empty[word_dim..word_dim + 2 * char_hidden].iter().all(|&v| v.to_bits() == 0));
        assert_eq!(tagger.tag(&[""]).unwrap().len(), 1);

        let unicode_tags = tagger.tag(&["İstanbul", "🐕"]).unwrap();
        assert_eq!(unicode_tags.len(), 2);
        assert_eq!(unicode_tags, tagger.tag(&["İstanbul", "🐕"]).unwrap());

        // 2-layer CRF usage test
        let accurate = BilstmSettings::new(
            16, 8, 8, 16, 60, 8, 0.05, 5.0, 0.0, 1, 16, 17, 1, 0.0, 0, true, 2, 0.0, 0.0, 1.0, 0.0, false,
        )
        .unwrap();
        let crf_tagger = BilstmPosTagger::new(BilstmPosTrainer::train(&usage_corpus(), accurate).unwrap());
        assert_eq!(crf_tagger.tag(&["the", "cat", "barks"]).unwrap(), vec!["DT", "NN", "VBZ"]);
        assert_eq!(crf_tagger.tag(&["dogs", "bark"]).unwrap(), vec!["NNS", "VBP"]);
    }

    #[test]
    fn full_model_gradients_match_finite_differences() {
        const EPSILON: f64 = 1e-6;
        const TOLERANCE: f64 = 1e-5;

        let corpus = vec![
            MultiTaskSample::new(
                ["The", "cat", "sat"],
                ["DET", "NOUN", "VERB"],
                Some(vec!["DT".into(), "NN".into(), "VBD".into()]),
                Some(vec!["_".into(), "Number=Sing".into(), "Tense=Past".into()]),
            )
            .unwrap(),
            MultiTaskSample::new(
                ["A", "dog", "ran"],
                ["DET", "NOUN", "VERB"],
                Some(vec!["DT".into(), "NN".into(), "VBD".into()]),
                Some(vec!["_".into(), "Number=Sing".into(), "Tense=Past".into()]),
            )
            .unwrap(),
            MultiTaskSample::new(
                ["Cats", "sleep"],
                ["NOUN", "VERB"],
                Some(vec!["NNS".into(), "VBP".into()]),
                Some(vec!["Number=Plur".into(), "Tense=Pres".into()]),
            )
            .unwrap(),
        ];
        let sample = &corpus[0];

        // Check CRF + 2-layer + Aux + Adapter gradients
        let crf_settings = BilstmSettings::new(
            4, 3, 3, 4, 1, 2, 1e-3, 5.0, 0.0, 1, 10, 7, 1, 0.0, 0, true, 2, 0.0, 0.0, 0.7, 0.0, true,
        )
        .unwrap();
        let vec_fn = |_: &str| Some(vec![0.5f32, -0.25f32]);
        let mut ctx = TrainingContext::build(&corpus, crf_settings, Some(vec_fn), None).unwrap();
        let mut worker = ctx.new_worker();
        ctx.reset_worker(&mut worker);
        ctx.sentence_gradients(sample, &mut JavaRng::new(0), &mut worker).unwrap();
        let analytic_out_w = worker.buffers()[ctx.idx_out_w].clone();
        let analytic_tr = worker.buffers()[ctx.idx_crf_tr.unwrap()].clone();
        let analytic_aw = worker.buffers()[ctx.idx_adapter_w.unwrap()].clone();

        for o in 0..analytic_out_w.len() {
            for j in 0..analytic_out_w[o].len() {
                let orig = ctx.testing_output_weights_mut()[o][j];
                ctx.testing_output_weights_mut()[o][j] = orig + EPSILON;
                let plus = ctx.sentence_gradients(sample, &mut JavaRng::new(0), &mut worker).unwrap();
                ctx.testing_output_weights_mut()[o][j] = orig - EPSILON;
                let minus = ctx.sentence_gradients(sample, &mut JavaRng::new(0), &mut worker).unwrap();
                ctx.testing_output_weights_mut()[o][j] = orig;
                let num = (plus - minus) / (2.0 * EPSILON);
                assert!((analytic_out_w[o][j] - num).abs() <= TOLERANCE);
            }
        }
        for j in 0..analytic_tr.len() {
            for k in 0..analytic_tr[j].len() {
                let orig = ctx.testing_transition_weights_mut().unwrap()[j][k];
                ctx.testing_transition_weights_mut().unwrap()[j][k] = orig + EPSILON;
                let plus = ctx.sentence_gradients(sample, &mut JavaRng::new(0), &mut worker).unwrap();
                ctx.testing_transition_weights_mut().unwrap()[j][k] = orig - EPSILON;
                let minus = ctx.sentence_gradients(sample, &mut JavaRng::new(0), &mut worker).unwrap();
                ctx.testing_transition_weights_mut().unwrap()[j][k] = orig;
                let num = (plus - minus) / (2.0 * EPSILON);
                assert!((analytic_tr[j][k] - num).abs() <= TOLERANCE);
            }
        }
        for r in 0..analytic_aw.len() {
            for c in 0..analytic_aw[r].len() {
                let orig = ctx.testing_adapter_weights_mut().unwrap()[r][c];
                ctx.testing_adapter_weights_mut().unwrap()[r][c] = orig + EPSILON;
                let plus = ctx.sentence_gradients(sample, &mut JavaRng::new(0), &mut worker).unwrap();
                ctx.testing_adapter_weights_mut().unwrap()[r][c] = orig - EPSILON;
                let minus = ctx.sentence_gradients(sample, &mut JavaRng::new(0), &mut worker).unwrap();
                ctx.testing_adapter_weights_mut().unwrap()[r][c] = orig;
                let num = (plus - minus) / (2.0 * EPSILON);
                assert!((analytic_aw[r][c] - num).abs() <= TOLERANCE);
            }
        }

        let _ = char_vocabulary_from_strs([("*UNK*", 0)]);
    }
}
