//! BiLSTM-CRF part-of-speech model (`BilstmPosModel`) and binary format reader/writer
//! (`ONLP-BLPT-1` through `ONLP-BLPT-5`).

mod infer;
mod load;
mod packed;
mod serialize;
mod validate;

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use turbo_analysis::ModelRef;
use turbo_unicode::java::lowercase_code_points;

use crate::neural::NeuralError;
use crate::neural::lstm::LstmLayer;

pub use load::{char_key_from_unit, char_vocabulary_from_strs};
pub(crate) use packed::Packed;
use validate::validate_tensors;

/// Unknown word and character symbol, always stored at row `0`.
pub const UNKNOWN: &str = "*UNK*";

pub(super) const MAGIC_V1: &str = "ONLP-BLPT-1";
pub(super) const MAGIC_V2: &str = "ONLP-BLPT-2";
pub(super) const MAGIC_V3: &str = "ONLP-BLPT-3";
pub(super) const MAGIC_V4: &str = "ONLP-BLPT-4";
pub(super) const MAGIC_V5: &str = "ONLP-BLPT-5";

/// Immutable parameters, vocabularies, and inference logic of a BiLSTM(-CRF) POS tagger.
#[derive(Clone, Debug)]
pub struct BilstmPosModel {
    word_entries: Vec<(String, usize)>,
    word_ids: HashMap<String, usize>,
    char_entries: Vec<(Vec<u8>, usize)>,
    char_ids: HashMap<Vec<u8>, usize>,
    tags: Vec<String>,
    pub(crate) word_embeddings: Vec<Vec<f64>>,
    char_embeddings: Vec<Vec<f64>>,
    pub(crate) char_forward: LstmLayer,
    char_backward: LstmLayer,
    word_forward: LstmLayer,
    word_backward: LstmLayer,
    word_forward2: Option<LstmLayer>,
    word_backward2: Option<LstmLayer>,
    output_weights: Vec<Vec<f64>>,
    output_bias: Vec<f64>,
    max_word_length: usize,
    pretrained_entries: Vec<(String, usize)>,
    pretrained_ids: HashMap<String, usize>,
    pretrained_vectors: Option<Vec<Vec<f32>>>,
    transition_weights: Option<Vec<Vec<f64>>>,
    start_weights: Option<Vec<f64>>,
    end_weights: Option<Vec<f64>>,
    adapter_weights: Option<Vec<Vec<f64>>>,
    adapter_bias: Option<Vec<f64>>,
    representation_cache: RepresentationCache,
    model_ref: ModelRef,
}

type RepresentationCache = Arc<RwLock<Option<HashMap<String, Vec<f64>>>>>;

fn layer_pair_mismatch(a: &Option<LstmLayer>, b: &Option<LstmLayer>) -> bool {
    a.is_some() != b.is_some()
}

fn crf_mismatch(tr: &Option<Vec<Vec<f64>>>, st: &Option<Vec<f64>>, en: &Option<Vec<f64>>) -> bool {
    !(tr.is_some() == st.is_some() && st.is_some() == en.is_some())
}

impl BilstmPosModel {
    /// Constructs a `BilstmPosModel` from its component vocabularies and weight tensors.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        word_entries: Vec<(String, usize)>,
        char_entries: Vec<(Vec<u8>, usize)>,
        tags: Vec<String>,
        word_embeddings: Vec<Vec<f64>>,
        char_embeddings: Vec<Vec<f64>>,
        char_forward: LstmLayer,
        char_backward: LstmLayer,
        word_forward: LstmLayer,
        word_backward: LstmLayer,
        word_forward2: Option<LstmLayer>,
        word_backward2: Option<LstmLayer>,
        output_weights: Vec<Vec<f64>>,
        output_bias: Vec<f64>,
        max_word_length: usize,
        pretrained_entries: Option<Vec<(String, usize)>>,
        pretrained_vectors: Option<Vec<Vec<f32>>>,
        transition_weights: Option<Vec<Vec<f64>>>,
        start_weights: Option<Vec<f64>>,
        end_weights: Option<Vec<f64>>,
        adapter_weights: Option<Vec<Vec<f64>>>,
        adapter_bias: Option<Vec<f64>>,
    ) -> Result<Self, NeuralError> {
        if tags.is_empty() || max_word_length == 0 {
            return Err(NeuralError::InvalidArgument("tags must not be empty and maxWordLength must be >= 1".into()));
        }
        if layer_pair_mismatch(&word_forward2, &word_backward2) {
            return Err(NeuralError::InvalidArgument(
                "second-layer forward and backward LSTMs must be provided together".into(),
            ));
        }
        if crf_mismatch(&transition_weights, &start_weights, &end_weights) {
            return Err(NeuralError::InvalidArgument(
                "CRF transition, start, and end weights must be provided together".into(),
            ));
        }
        if adapter_weights.is_some() != adapter_bias.is_some() {
            return Err(NeuralError::InvalidArgument("adapter weights and bias must be provided together".into()));
        }
        let pretrained_entries = pretrained_entries.unwrap_or_default();
        validate_tensors(
            &word_entries,
            &char_entries,
            &tags,
            &word_embeddings,
            &char_embeddings,
            &char_forward,
            &char_backward,
            &word_forward,
            &word_backward,
            &word_forward2,
            &word_backward2,
            &output_weights,
            &output_bias,
            &pretrained_entries,
            &pretrained_vectors,
            &transition_weights,
            &start_weights,
            &end_weights,
            &adapter_weights,
            &adapter_bias,
        )?;
        let mut word_ids = HashMap::with_capacity(word_entries.len());
        for (k, v) in &word_entries {
            word_ids.insert(k.clone(), *v);
        }
        let mut char_ids = HashMap::with_capacity(char_entries.len());
        for (k, v) in &char_entries {
            char_ids.insert(k.clone(), *v);
        }
        let mut pretrained_ids = HashMap::with_capacity(pretrained_entries.len());
        for (k, v) in &pretrained_entries {
            pretrained_ids.insert(k.clone(), *v);
        }
        let mut model = Self {
            word_entries,
            word_ids,
            char_entries,
            char_ids,
            tags,
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
            max_word_length,
            pretrained_entries,
            pretrained_ids,
            pretrained_vectors,
            transition_weights,
            start_weights,
            end_weights,
            adapter_weights,
            adapter_bias,
            representation_cache: Arc::new(RwLock::new(None)),
            model_ref: ModelRef { id: String::new(), revision: String::new(), sha256: String::new() },
        };
        model.refresh_model_ref()?;
        Ok(model)
    }

    /// Returns the provenance [`ModelRef`] for this model (including the SHA-256 of its serialized bytes).
    pub fn model_ref(&self) -> &ModelRef {
        &self.model_ref
    }

    fn refresh_model_ref(&mut self) -> Result<(), NeuralError> {
        let mut bytes = Vec::new();
        self.serialize(&mut bytes)?;
        let revision = if self.has_adapter() {
            MAGIC_V5
        } else {
            match (self.is_stacked(), self.is_crf()) {
                (false, false) => MAGIC_V1,
                (false, true) => MAGIC_V2,
                (true, false) => MAGIC_V3,
                (true, true) => MAGIC_V4,
            }
        };
        self.model_ref = ModelRef {
            id: "opennlp:BilstmPOSModel".into(),
            revision: revision.into(),
            sha256: turbo_maxent::sha256(&bytes).to_hex(),
        };
        Ok(())
    }

    /// Normalizes a token for vocabulary lookup: tokens starting with `'*'` are returned unchanged;
    /// all other tokens are lowercased character-by-character via `Character.toLowerCase(int)`.
    pub fn normalize(token: &str) -> String {
        if token.starts_with('*') { token.to_owned() } else { lowercase_code_points(token) }
    }

    /// Returns the word embedding row index for `token` (`0` for unseen words).
    pub fn word_id(&self, token: &str) -> usize {
        let norm = Self::normalize(token);
        self.word_ids.get(&norm).copied().unwrap_or(0)
    }

    /// Returns the character embedding row indices for `token` (up to
    /// `max_word_length` UTF-16 code units), empty for an empty token.
    pub fn char_ids(&self, token: &str) -> Vec<usize> {
        let mut ids = Vec::with_capacity(token.len().min(self.max_word_length));
        let mut unit_buf = Vec::with_capacity(3);
        for unit in token.encode_utf16().take(self.max_word_length) {
            unit_buf.clear();
            turbo_unicode::modified_utf8::push_modified_utf8(&mut unit_buf, u32::from(unit));
            let id = self.char_ids.get(&unit_buf).copied().unwrap_or(0);
            ids.push(id);
        }
        ids
    }

    /// Tag outcome inventory in index order.
    pub fn tags(&self) -> &[String] {
        &self.tags
    }

    /// Maximum number of leading UTF-16 code units read by the character BiLSTM.
    pub fn max_word_length(&self) -> usize {
        self.max_word_length
    }

    /// Whether the model includes a linear-chain CRF transition layer.
    pub fn is_crf(&self) -> bool {
        self.transition_weights.is_some()
    }

    /// Whether the model has a second stacked sentence BiLSTM layer.
    pub fn is_stacked(&self) -> bool {
        self.word_forward2.is_some()
    }

    /// Whether the model includes a linear adapter over pretrained vectors.
    pub fn has_adapter(&self) -> bool {
        self.adapter_weights.is_some()
    }

    /// CRF transition weights `[num_tags][num_tags]`, if [`is_crf`](Self::is_crf) is true.
    pub fn transition_weights(&self) -> Option<&[Vec<f64>]> {
        self.transition_weights.as_deref()
    }

    /// CRF start weights `[num_tags]`, if [`is_crf`](Self::is_crf) is true.
    pub fn start_weights(&self) -> Option<&[f64]> {
        self.start_weights.as_deref()
    }

    /// CRF end weights `[num_tags]`, if [`is_crf`](Self::is_crf) is true.
    pub fn end_weights(&self) -> Option<&[f64]> {
        self.end_weights.as_deref()
    }
}
