//! BiLSTM-CRF part-of-speech model (`BilstmPosModel`) and binary format reader/writer
//! (`ONLP-BLPT-1` through `ONLP-BLPT-5`).

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::{Arc, RwLock};

use turbo_unicode::java::to_lower_char;

use crate::neural::lstm::{LstmLayer, MAX_HIDDEN_SIZE};
use crate::neural::{DataReader, DataWriter, NeuralError, encode_str_modified_utf8, encode_unit_modified_utf8};

/// Unknown word and character symbol, always stored at row `0`.
pub const UNKNOWN: &str = "*UNK*";

const UNKNOWN_BYTES: &[u8] = b"*UNK*";
const REPRESENTATION_CACHE_LIMIT: usize = 100_000;
const INITIAL_CAPACITY: usize = 4096;
const MAX_DIMENSION: i32 = i32::MAX / 8;

const MAGIC_V1: &str = "ONLP-BLPT-1";
const MAGIC_V2: &str = "ONLP-BLPT-2";
const MAGIC_V3: &str = "ONLP-BLPT-3";
const MAGIC_V4: &str = "ONLP-BLPT-4";
const MAGIC_V5: &str = "ONLP-BLPT-5";

/// Immutable parameters, vocabularies, and inference logic of a BiLSTM(-CRF) POS tagger.
#[derive(Clone, Debug)]
pub struct BilstmPosModel {
    word_entries: Vec<(String, usize)>,
    word_ids: HashMap<String, usize>,
    char_entries: Vec<(Vec<u8>, usize)>,
    char_ids: HashMap<Vec<u8>, usize>,
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
    pretrained_entries: Vec<(String, usize)>,
    pretrained_ids: HashMap<String, usize>,
    pretrained_vectors: Option<Vec<Vec<f32>>>,
    transition_weights: Option<Vec<Vec<f64>>>,
    start_weights: Option<Vec<f64>>,
    end_weights: Option<Vec<f64>>,
    adapter_weights: Option<Vec<Vec<f64>>>,
    adapter_bias: Option<Vec<f64>>,
    representation_cache: RepresentationCache,
}

type RepresentationCache = Arc<RwLock<Option<HashMap<String, Vec<f64>>>>>;

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
        if word_Forward2_mismatch(&word_forward2, &word_backward2) {
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
        let mut word_ids = HashMap::with_capacity(word_entries.len());
        for (k, v) in &word_entries {
            word_ids.insert(k.clone(), *v);
        }
        let mut char_ids = HashMap::with_capacity(char_entries.len());
        for (k, v) in &char_entries {
            char_ids.insert(k.clone(), *v);
        }
        let pretrained_entries = pretrained_entries.unwrap_or_default();
        let mut pretrained_ids = HashMap::with_capacity(pretrained_entries.len());
        for (k, v) in &pretrained_entries {
            pretrained_ids.insert(k.clone(), *v);
        }
        Ok(Self {
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
        })
    }

    /// Normalizes a token for vocabulary lookup: tokens starting with `'*'` are returned unchanged;
    /// all other tokens are lowercased character-by-character via `Character.toLowerCase(int)`.
    pub fn normalize(token: &str) -> String {
        if token.starts_with('*') { token.to_owned() } else { token.chars().map(to_lower_char).collect() }
    }

    /// Returns the word embedding row index for `token` (`0` for unseen words).
    pub fn word_id(&self, token: &str) -> usize {
        let norm = Self::normalize(token);
        self.word_ids.get(&norm).copied().unwrap_or(0)
    }

    /// Returns the character embedding row indices for `token` (up to `max_word_length` UTF-16 code units).
    pub fn char_ids(&self, token: &str) -> Vec<usize> {
        if token.is_empty() {
            return vec![0];
        }
        let mut ids = Vec::with_capacity(token.len().min(self.max_word_length));
        let mut unit_buf = Vec::with_capacity(3);
        for unit in token.encode_utf16().take(self.max_word_length) {
            unit_buf.clear();
            turbo_unicode::modified_utf8::push_modified_utf8(&mut unit_buf, u32::from(unit));
            let id = self.char_ids.get(&unit_buf).copied().unwrap_or(0);
            ids.push(id);
        }
        if ids.is_empty() {
            ids.push(0);
        }
        ids
    }

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

        let c_ids = self.char_ids(token);
        let len = c_ids.len();
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
                let mut sum = self.output_bias[o];
                for j in 0..h_size {
                    sum += w[j] * fh[j] + w[h_size + j] * bh[j];
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

    fn pretrained_dim(&self) -> usize {
        match &self.pretrained_vectors {
            Some(vecs) => {
                if let Some(first) = vecs.first() {
                    first.len()
                } else {
                    self.word_forward.input_size() - self.word_embeddings[0].len() - 2 * self.char_forward.hidden_size()
                }
            }
            None => 0,
        }
    }

    /// Serializes this model in the `ONLP-BLPT-1`..`5` binary format to `writer`.
    pub fn serialize<W: Write>(&self, writer: W) -> Result<(), NeuralError> {
        let mut data = DataWriter::new(writer);
        let stacked = self.is_stacked();
        let crf = self.is_crf();
        if self.has_adapter() {
            data.write_utf(MAGIC_V5)?;
            data.write_boolean(stacked)?;
            data.write_boolean(crf)?;
        } else {
            let magic = match (stacked, crf) {
                (false, false) => MAGIC_V1,
                (false, true) => MAGIC_V2,
                (true, false) => MAGIC_V3,
                (true, true) => MAGIC_V4,
            };
            data.write_utf(magic)?;
        }
        write_str_vocabulary(&mut data, &self.word_entries)?;
        write_bytes_vocabulary(&mut data, &self.char_entries)?;
        data.write_i32(self.tags.len() as i32)?;
        for tag in &self.tags {
            data.write_utf(tag)?;
        }
        write_f64_matrix(&mut data, &self.word_embeddings)?;
        write_f64_matrix(&mut data, &self.char_embeddings)?;
        write_lstm(&mut data, &self.char_forward)?;
        write_lstm(&mut data, &self.char_backward)?;
        write_lstm(&mut data, &self.word_forward)?;
        write_lstm(&mut data, &self.word_backward)?;
        if let (Some(wf2), Some(wb2)) = (&self.word_forward2, &self.word_backward2) {
            write_lstm(&mut data, wf2)?;
            write_lstm(&mut data, wb2)?;
        }
        write_f64_matrix(&mut data, &self.output_weights)?;
        write_f64_vector(&mut data, &self.output_bias)?;
        if let (Some(tr), Some(st), Some(en)) = (&self.transition_weights, &self.start_weights, &self.end_weights) {
            write_f64_matrix(&mut data, tr)?;
            write_f64_vector(&mut data, st)?;
            write_f64_vector(&mut data, en)?;
        }
        data.write_i32(self.max_word_length as i32)?;
        if let Some(ref p_vecs) = self.pretrained_vectors {
            data.write_boolean(true)?;
            write_str_vocabulary(&mut data, &self.pretrained_entries)?;
            let vec_dim = self.pretrained_dim();
            data.write_i32(vec_dim as i32)?;
            data.write_i32(p_vecs.len() as i32)?;
            for row in p_vecs {
                for &v in row {
                    data.write_f32(v)?;
                }
            }
        } else {
            data.write_boolean(false)?;
        }
        if let (Some(aw), Some(ab)) = (&self.adapter_weights, &self.adapter_bias) {
            write_f64_matrix(&mut data, aw)?;
            write_f64_vector(&mut data, ab)?;
        }
        data.flush()?;
        Ok(())
    }

    /// Serializes this model to a file path.
    pub fn serialize_to_path(&self, path: impl AsRef<Path>) -> Result<(), NeuralError> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        self.serialize(&mut writer)?;
        writer.flush()?;
        Ok(())
    }

    /// Reads and validates a `BilstmPosModel` from a stream in `ONLP-BLPT-1`..`5` format.
    pub fn load<R: Read>(reader: R) -> Result<Self, NeuralError> {
        let mut data = DataReader::new(reader);
        let magic = data.read_utf()?;
        let (stacked, crf, adapter) = match magic.as_str() {
            MAGIC_V1 => (false, false, false),
            MAGIC_V2 => (false, true, false),
            MAGIC_V3 => (true, false, false),
            MAGIC_V4 => (true, true, false),
            MAGIC_V5 => {
                let s = data.read_boolean()?;
                let c = data.read_boolean()?;
                (s, c, true)
            }
            _ => {
                return Err(NeuralError::Format(format!("Unrecognized BiLSTM POS model header: {magic}")));
            }
        };

        let word_entries = read_str_vocabulary(&mut data, "words", false)?;
        let char_entries = read_bytes_vocabulary(&mut data, "characters", false)?;
        let tags = read_tags(&mut data)?;

        let word_embeddings = read_f64_matrix(&mut data, "wordEmbeddings")?;
        let char_embeddings = read_f64_matrix(&mut data, "charEmbeddings")?;
        validate_vocabulary_rows(&word_entries, word_embeddings.len(), "words")?;
        validate_bytes_vocabulary_rows(&char_entries, char_embeddings.len(), "characters")?;

        let char_dim = char_embeddings[0].len();
        let word_dim = word_embeddings[0].len();
        let char_forward = read_lstm(&mut data, "charForward", Some(char_dim), None)?;
        let char_backward = read_lstm(&mut data, "charBackward", Some(char_dim), Some(char_forward.hidden_size()))?;
        let word_forward = read_lstm(&mut data, "wordForward", None, None)?;
        let word_backward =
            read_lstm(&mut data, "wordBackward", Some(word_forward.input_size()), Some(word_forward.hidden_size()))?;

        let mut encoder_width = 2 * word_forward.hidden_size();
        let (word_forward2, word_backward2) = if stacked {
            let wf2 = read_lstm(&mut data, "wordForward2", Some(encoder_width), None)?;
            let wb2 = read_lstm(&mut data, "wordBackward2", Some(encoder_width), Some(wf2.hidden_size()))?;
            encoder_width = 2 * wf2.hidden_size();
            (Some(wf2), Some(wb2))
        } else {
            (None, None)
        };

        let output_weights = read_f64_matrix_exact(&mut data, "outputWeights", tags.len(), encoder_width)?;
        let output_bias = read_f64_vector_exact(&mut data, "outputBias", tags.len())?;

        let (transition_weights, start_weights, end_weights) = if crf {
            let tr = read_f64_matrix_exact(&mut data, "transitionWeights", tags.len(), tags.len())?;
            let st = read_f64_vector_exact(&mut data, "startWeights", tags.len())?;
            let en = read_f64_vector_exact(&mut data, "endWeights", tags.len())?;
            (Some(tr), Some(st), Some(en))
        } else {
            (None, None, None)
        };

        let max_word_length = read_positive_dim(&mut data, "maxWordLength")?;
        let mut expected_word_input = word_dim + 2 * char_forward.hidden_size();

        let (pretrained_entries, pretrained_vectors) = if data.read_boolean()? {
            let p_entries = read_str_vocabulary(&mut data, "pretrained", true)?;
            let vec_dim = read_positive_dim(&mut data, "pretrained vector width")?;
            let row_count = read_positive_dim(&mut data, "pretrained vector rows")?;
            validate_pretrained_rows(&p_entries, row_count)?;
            let p_vecs = read_f32_rows(&mut data, row_count, vec_dim)?;
            expected_word_input += vec_dim;
            (Some(p_entries), Some(p_vecs))
        } else {
            if adapter {
                return Err(NeuralError::Format(
                    "Invalid BiLSTM POS model: adapter requires pretrained vectors".into(),
                ));
            }
            (None, None)
        };

        if word_forward.input_size() != expected_word_input {
            return Err(NeuralError::Format(format!(
                "Invalid BiLSTM POS model: wordForward inputSize {} != {}",
                word_forward.input_size(),
                expected_word_input
            )));
        }

        let (adapter_weights, adapter_bias) = if adapter {
            let p_vecs = pretrained_vectors.as_ref().unwrap();
            let vec_dim = p_vecs[0].len();
            let aw = read_f64_matrix_exact(&mut data, "adapterWeights", vec_dim, vec_dim)?;
            let ab = read_f64_vector_exact(&mut data, "adapterBias", vec_dim)?;
            (Some(aw), Some(ab))
        } else {
            (None, None)
        };

        Self::new(
            word_entries,
            char_entries,
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
            pretrained_vectors,
            transition_weights,
            start_weights,
            end_weights,
            adapter_weights,
            adapter_bias,
        )
    }

    /// Reads and validates a `BilstmPosModel` from a file path.
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self, NeuralError> {
        let file = File::open(path)?;
        Self::load(BufReader::new(file))
    }
}

#[allow(non_snake_case)]
fn word_Forward2_mismatch(a: &Option<LstmLayer>, b: &Option<LstmLayer>) -> bool {
    a.is_some() != b.is_some()
}

fn crf_mismatch(tr: &Option<Vec<Vec<f64>>>, st: &Option<Vec<f64>>, en: &Option<Vec<f64>>) -> bool {
    !(tr.is_some() == st.is_some() && st.is_some() == en.is_some())
}

fn write_str_vocabulary<W: Write>(data: &mut DataWriter<W>, entries: &[(String, usize)]) -> Result<(), NeuralError> {
    data.write_i32(entries.len() as i32)?;
    for (k, v) in entries {
        data.write_utf(k)?;
        data.write_i32(*v as i32)?;
    }
    Ok(())
}

fn write_bytes_vocabulary<W: Write>(data: &mut DataWriter<W>, entries: &[(Vec<u8>, usize)]) -> Result<(), NeuralError> {
    data.write_i32(entries.len() as i32)?;
    for (k, v) in entries {
        data.write_utf_bytes(k)?;
        data.write_i32(*v as i32)?;
    }
    Ok(())
}

fn write_f64_matrix<W: Write>(data: &mut DataWriter<W>, m: &[Vec<f64>]) -> Result<(), NeuralError> {
    let cols = m.first().map_or(0, Vec::len);
    data.write_i32(m.len() as i32)?;
    data.write_i32(cols as i32)?;
    for row in m {
        for &v in row {
            data.write_f64(v)?;
        }
    }
    Ok(())
}

fn write_f64_vector<W: Write>(data: &mut DataWriter<W>, v: &[f64]) -> Result<(), NeuralError> {
    data.write_i32(v.len() as i32)?;
    for &x in v {
        data.write_f64(x)?;
    }
    Ok(())
}

fn write_lstm<W: Write>(data: &mut DataWriter<W>, layer: &LstmLayer) -> Result<(), NeuralError> {
    data.write_i32(layer.input_size() as i32)?;
    data.write_i32(layer.hidden_size() as i32)?;
    write_f64_matrix(data, layer.w())?;
    write_f64_matrix(data, layer.u())?;
    write_f64_vector(data, layer.b())
}

fn read_positive_dim<R: Read>(data: &mut DataReader<R>, name: &str) -> Result<usize, NeuralError> {
    let value = data.read_i32()?;
    if !(1..=MAX_DIMENSION).contains(&value) {
        return Err(NeuralError::Format(format!("Invalid BiLSTM POS model: {name} = {value}")));
    }
    Ok(value as usize)
}

fn read_non_negative_count<R: Read>(data: &mut DataReader<R>, name: &str) -> Result<usize, NeuralError> {
    let value = data.read_i32()?;
    if !(0..=MAX_DIMENSION).contains(&value) {
        return Err(NeuralError::Format(format!("Invalid BiLSTM POS model: {name} = {value}")));
    }
    Ok(value as usize)
}

fn read_str_vocabulary<R: Read>(
    data: &mut DataReader<R>,
    name: &str,
    allow_empty: bool,
) -> Result<Vec<(String, usize)>, NeuralError> {
    let size = if allow_empty {
        read_non_negative_count(data, &format!("{name} vocabulary size"))?
    } else {
        read_positive_dim(data, &format!("{name} vocabulary size"))?
    };
    let mut entries = Vec::with_capacity(size.min(INITIAL_CAPACITY));
    let mut seen = HashSet::with_capacity(size.min(INITIAL_CAPACITY));
    for _ in 0..size {
        let key = data.read_utf()?;
        let row = data.read_i32()?;
        if row < 0 {
            return Err(NeuralError::Format(format!("Invalid BiLSTM POS model: negative {name} row {row}")));
        }
        if !seen.insert(key.clone()) {
            return Err(NeuralError::Format(format!("Invalid BiLSTM POS model: duplicate {name} entry {key}")));
        }
        entries.push((key, row as usize));
    }
    Ok(entries)
}

fn read_bytes_vocabulary<R: Read>(
    data: &mut DataReader<R>,
    name: &str,
    allow_empty: bool,
) -> Result<Vec<(Vec<u8>, usize)>, NeuralError> {
    let size = if allow_empty {
        read_non_negative_count(data, &format!("{name} vocabulary size"))?
    } else {
        read_positive_dim(data, &format!("{name} vocabulary size"))?
    };
    let mut entries = Vec::with_capacity(size.min(INITIAL_CAPACITY));
    let mut seen = HashSet::with_capacity(size.min(INITIAL_CAPACITY));
    for _ in 0..size {
        let key = data.read_utf_bytes()?;
        let row = data.read_i32()?;
        if row < 0 {
            return Err(NeuralError::Format(format!("Invalid BiLSTM POS model: negative {name} row {row}")));
        }
        if !seen.insert(key.clone()) {
            return Err(NeuralError::Format(format!("Invalid BiLSTM POS model: duplicate {name} entry")));
        }
        entries.push((key, row as usize));
    }
    Ok(entries)
}

fn validate_vocabulary_rows(entries: &[(String, usize)], row_count: usize, name: &str) -> Result<(), NeuralError> {
    let mut unk_row = None;
    for (k, r) in entries {
        if *r >= row_count {
            return Err(NeuralError::Format(format!("Invalid BiLSTM POS model: {name} row {r} >= {row_count}")));
        }
        if k == UNKNOWN {
            unk_row = Some(*r);
        }
    }
    if unk_row != Some(0) {
        return Err(NeuralError::Format(format!("Invalid BiLSTM POS model: {name} must map {UNKNOWN} to row 0")));
    }
    Ok(())
}

fn validate_bytes_vocabulary_rows(
    entries: &[(Vec<u8>, usize)],
    row_count: usize,
    name: &str,
) -> Result<(), NeuralError> {
    let mut unk_row = None;
    for (k, r) in entries {
        if *r >= row_count {
            return Err(NeuralError::Format(format!("Invalid BiLSTM POS model: {name} row {r} >= {row_count}")));
        }
        if k.as_slice() == UNKNOWN_BYTES {
            unk_row = Some(*r);
        }
    }
    if unk_row != Some(0) {
        return Err(NeuralError::Format(format!("Invalid BiLSTM POS model: {name} must map {UNKNOWN} to row 0")));
    }
    Ok(())
}

fn validate_pretrained_rows(entries: &[(String, usize)], row_count: usize) -> Result<(), NeuralError> {
    for (_, r) in entries {
        if *r >= row_count {
            return Err(NeuralError::Format(format!("Invalid BiLSTM POS model: pretrained row {r} >= {row_count}")));
        }
    }
    Ok(())
}

fn read_tags<R: Read>(data: &mut DataReader<R>) -> Result<Vec<String>, NeuralError> {
    let count = read_positive_dim(data, "tag count")?;
    let mut tags = Vec::with_capacity(count.min(INITIAL_CAPACITY));
    let mut seen = HashSet::with_capacity(count.min(INITIAL_CAPACITY));
    for _ in 0..count {
        let tag = data.read_utf()?;
        if !seen.insert(tag.clone()) {
            return Err(NeuralError::Format(format!("Invalid BiLSTM POS model: duplicate tag {tag}")));
        }
        tags.push(tag);
    }
    Ok(tags)
}

fn read_lstm<R: Read>(
    data: &mut DataReader<R>,
    name: &str,
    expected_input: Option<usize>,
    expected_hidden: Option<usize>,
) -> Result<LstmLayer, NeuralError> {
    let input_size = read_positive_dim(data, &format!("{name} inputSize"))?;
    let hidden_size = read_positive_dim(data, &format!("{name} hiddenSize"))?;
    if hidden_size > MAX_HIDDEN_SIZE {
        return Err(NeuralError::Format(format!("Invalid BiLSTM POS model: {name} hiddenSize = {hidden_size}")));
    }
    if let Some(exp_in) = expected_input
        && input_size != exp_in
    {
        return Err(NeuralError::Format(format!(
            "Invalid BiLSTM POS model: {name} inputSize {input_size} != {exp_in}"
        )));
    }
    if let Some(exp_hid) = expected_hidden
        && hidden_size != exp_hid
    {
        return Err(NeuralError::Format(format!(
            "Invalid BiLSTM POS model: {name} hiddenSize {hidden_size} != {exp_hid}"
        )));
    }
    let gates = 4 * hidden_size;
    let w = read_f64_matrix_exact(data, &format!("{name}.w"), gates, input_size)?;
    let u = read_f64_matrix_exact(data, &format!("{name}.u"), gates, hidden_size)?;
    let b = read_f64_vector_exact(data, &format!("{name}.b"), gates)?;
    LstmLayer::of_weights(input_size, hidden_size, w, u, b)
}

fn read_f64_matrix<R: Read>(data: &mut DataReader<R>, name: &str) -> Result<Vec<Vec<f64>>, NeuralError> {
    let rows = read_positive_dim(data, &format!("{name} rows"))?;
    let cols = read_positive_dim(data, &format!("{name} cols"))?;
    read_f64_rows(data, name, rows, cols)
}

fn read_f64_matrix_exact<R: Read>(
    data: &mut DataReader<R>,
    name: &str,
    expected_rows: usize,
    expected_cols: usize,
) -> Result<Vec<Vec<f64>>, NeuralError> {
    let rows = read_positive_dim(data, &format!("{name} rows"))?;
    let cols = read_positive_dim(data, &format!("{name} cols"))?;
    if rows != expected_rows || cols != expected_cols {
        return Err(NeuralError::Format(format!(
            "Invalid BiLSTM POS model: {name} shape {rows}x{cols} != {expected_rows}x{expected_cols}"
        )));
    }
    read_f64_rows(data, name, rows, cols)
}

fn read_f64_rows<R: Read>(
    data: &mut DataReader<R>,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Vec<Vec<f64>>, NeuralError> {
    let mut matrix = Vec::with_capacity(rows.min(INITIAL_CAPACITY));
    for _ in 0..rows {
        matrix.push(read_f64_values(data, name, cols)?);
    }
    Ok(matrix)
}

fn read_f64_vector_exact<R: Read>(
    data: &mut DataReader<R>,
    name: &str,
    expected_len: usize,
) -> Result<Vec<f64>, NeuralError> {
    let len = read_positive_dim(data, &format!("{name} length"))?;
    if len != expected_len {
        return Err(NeuralError::Format(format!("Invalid BiLSTM POS model: {name} length {len} != {expected_len}")));
    }
    read_f64_values(data, name, len)
}

fn read_f64_values<R: Read>(data: &mut DataReader<R>, name: &str, len: usize) -> Result<Vec<f64>, NeuralError> {
    let mut out = Vec::with_capacity(len.min(INITIAL_CAPACITY));
    for _ in 0..len {
        let v = data.read_f64()?;
        if !v.is_finite() {
            return Err(NeuralError::Format(format!("Invalid BiLSTM POS model: non-finite {name} value {v}")));
        }
        out.push(v);
    }
    Ok(out)
}

fn read_f32_rows<R: Read>(data: &mut DataReader<R>, rows: usize, cols: usize) -> Result<Vec<Vec<f32>>, NeuralError> {
    let mut matrix = Vec::with_capacity(rows.min(INITIAL_CAPACITY));
    for _ in 0..rows {
        let mut row = Vec::with_capacity(cols.min(INITIAL_CAPACITY));
        for _ in 0..cols {
            let v = data.read_f32()?;
            if !v.is_finite() {
                return Err(NeuralError::Format(format!(
                    "Invalid BiLSTM POS model: non-finite pretrained vector value {v}"
                )));
            }
            row.push(v);
        }
        matrix.push(row);
    }
    Ok(matrix)
}

/// Helper to build a character vocabulary entry list from `&str` keys.
pub fn char_vocabulary_from_strs<I, S>(entries: I) -> Vec<(Vec<u8>, usize)>
where
    I: IntoIterator<Item = (S, usize)>,
    S: AsRef<str>,
{
    entries.into_iter().map(|(k, v)| (encode_str_modified_utf8(k.as_ref()), v)).collect()
}

/// Helper to build a character vocabulary entry from a single UTF-16 code unit.
pub fn char_key_from_unit(unit: u16) -> Vec<u8> {
    encode_unit_modified_utf8(unit)
}
