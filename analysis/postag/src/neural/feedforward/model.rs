//! Feedforward neural POS model parameters, scoring cache, and binary
//! `ONLP-FFPT-1` / `ONLP-FFPT-2` serialization (`FeedforwardPOSModel`).

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicIsize, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

use turbo_analysis::ModelRef;

use super::context::{ABSENT, FeedforwardPosContext, PRETRAINED_SLOTS, SLOTS, SentenceFeatures, UNKNOWN};
use crate::neural::{DataReader, DataWriter, NeuralError};

const MAGIC_V1: &str = "ONLP-FFPT-1";
const MAGIC_V2: &str = "ONLP-FFPT-2";
pub(super) const CACHE_ENTRY_LIMIT: isize = 32768;
const INITIAL_CAPACITY: usize = 4096;
const MAX_LENGTH: i32 = 1 << 30;

#[derive(Debug)]
pub(super) struct ContributionCache {
    by_slot: Vec<Vec<OnceLock<Vec<f64>>>>,
    remaining: AtomicIsize,
}

impl ContributionCache {
    fn new(row_count: usize) -> Self {
        let mut by_slot = Vec::with_capacity(SLOTS);
        for _ in 0..SLOTS {
            let mut slot_vec = Vec::with_capacity(row_count);
            slot_vec.resize_with(row_count, OnceLock::new);
            by_slot.push(slot_vec);
        }
        Self { by_slot, remaining: AtomicIsize::new(CACHE_ENTRY_LIMIT) }
    }

    #[cfg(test)]
    pub(super) fn set_budget(&self, budget: isize) {
        self.remaining.store(budget, Ordering::SeqCst);
    }

    #[cfg(test)]
    pub(super) fn budget(&self) -> isize {
        self.remaining.load(Ordering::SeqCst)
    }

    #[cfg(test)]
    pub(super) fn cached_entries(&self) -> usize {
        self.by_slot.iter().flat_map(|slot| slot.iter()).filter(|cell| cell.get().is_some()).count()
    }
}

/// Immutable parameters and vocabularies of the feedforward neural POS tagger (`ONLP-FFPT-1` / `ONLP-FFPT-2`).
#[derive(Clone, Debug)]
pub struct FeedforwardPosModel {
    word_entries: Vec<(String, usize)>,
    word_ids: HashMap<String, usize>,
    suffix_entries: Vec<(String, usize)>,
    suffix_ids: HashMap<String, usize>,
    shape_entries: Vec<(String, usize)>,
    shape_ids: HashMap<String, usize>,
    tag_feature_entries: Vec<(String, usize)>,
    tag_feature_ids: HashMap<String, usize>,
    tags: Vec<String>,
    embedding_size: usize,
    embeddings: Vec<Vec<f32>>,
    hidden_weights: Vec<Vec<f32>>,
    hidden_bias: Vec<f32>,
    output_weights: Vec<Vec<f32>>,
    output_bias: Vec<f32>,
    pretrained_size: usize,
    pretrained_entries: Vec<(String, usize)>,
    pretrained_ids: HashMap<String, usize>,
    pretrained_vectors: Vec<Vec<f32>>,
    pub(super) cache: Arc<RwLock<Option<Arc<ContributionCache>>>>,
}

impl FeedforwardPosModel {
    /// Creates a `FeedforwardPosModel` without pretrained word vectors (`ONLP-FFPT-1`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        word_entries: Vec<(String, usize)>,
        suffix_entries: Vec<(String, usize)>,
        shape_entries: Vec<(String, usize)>,
        tag_feature_entries: Vec<(String, usize)>,
        tags: Vec<String>,
        embedding_size: usize,
        embeddings: Vec<Vec<f32>>,
        hidden_weights: Vec<Vec<f32>>,
        hidden_bias: Vec<f32>,
        output_weights: Vec<Vec<f32>>,
        output_bias: Vec<f32>,
    ) -> Self {
        Self::with_pretrained(
            word_entries,
            suffix_entries,
            shape_entries,
            tag_feature_entries,
            tags,
            embedding_size,
            embeddings,
            hidden_weights,
            hidden_bias,
            output_weights,
            output_bias,
            0,
            Vec::new(),
            Vec::new(),
        )
    }

    /// Creates a `FeedforwardPosModel` with optional pretrained word vectors (`ONLP-FFPT-2` when `pretrained_size > 0`).
    #[allow(clippy::too_many_arguments)]
    pub fn with_pretrained(
        word_entries: Vec<(String, usize)>,
        suffix_entries: Vec<(String, usize)>,
        shape_entries: Vec<(String, usize)>,
        tag_feature_entries: Vec<(String, usize)>,
        tags: Vec<String>,
        embedding_size: usize,
        embeddings: Vec<Vec<f32>>,
        hidden_weights: Vec<Vec<f32>>,
        hidden_bias: Vec<f32>,
        output_weights: Vec<Vec<f32>>,
        output_bias: Vec<f32>,
        pretrained_size: usize,
        pretrained_entries: Vec<(String, usize)>,
        pretrained_vectors: Vec<Vec<f32>>,
    ) -> Self {
        let word_ids = entries_to_map(&word_entries);
        let suffix_ids = entries_to_map(&suffix_entries);
        let shape_ids = entries_to_map(&shape_entries);
        let tag_feature_ids = entries_to_map(&tag_feature_entries);
        let pretrained_ids = entries_to_map(&pretrained_entries);
        Self {
            word_entries,
            word_ids,
            suffix_entries,
            suffix_ids,
            shape_entries,
            shape_ids,
            tag_feature_entries,
            tag_feature_ids,
            tags,
            embedding_size,
            embeddings,
            hidden_weights,
            hidden_bias,
            output_weights,
            output_bias,
            pretrained_size,
            pretrained_entries,
            pretrained_ids,
            pretrained_vectors,
            cache: Arc::new(RwLock::new(None)),
        }
    }

    /// Maps the 12 context feature strings to their embedding row indices.
    pub fn feature_ids(&self, context: &[impl AsRef<str>; SLOTS]) -> [usize; SLOTS] {
        let mut ids = [0usize; SLOTS];
        // Words are matched case-insensitively, as Java's `normalize` does.
        for s in 0..5 {
            ids[s] = self.lookup(&self.word_ids, &FeedforwardPosContext::lowercase(context[s].as_ref()));
        }
        ids[5] = self.lookup(&self.suffix_ids, context[5].as_ref());
        ids[6] = self.lookup(&self.suffix_ids, context[6].as_ref());
        for s in 7..10 {
            ids[s] = self.lookup(&self.shape_ids, context[s].as_ref());
        }
        ids[10] = self.lookup(&self.tag_feature_ids, context[10].as_ref());
        ids[11] = self.lookup(&self.tag_feature_ids, context[11].as_ref());
        ids
    }

    /// Maps the 12 context features at `index` of a pre-lowercased sentence to
    /// their embedding row indices without allocating per-slot `String`s.
    pub(super) fn feature_ids_at(
        &self,
        sentence: &SentenceFeatures<'_>,
        index: usize,
        prev_tag: Option<&str>,
        prev_prev_tag: Option<&str>,
    ) -> [usize; SLOTS] {
        let idx = index as isize;
        [
            self.lookup(&self.word_ids, sentence.word_key(idx - 2)),
            self.lookup(&self.word_ids, sentence.word_key(idx - 1)),
            self.lookup(&self.word_ids, sentence.word_key(idx)),
            self.lookup(&self.word_ids, sentence.word_key(idx + 1)),
            self.lookup(&self.word_ids, sentence.word_key(idx + 2)),
            self.lookup(&self.suffix_ids, sentence.suffix_key(index, 2)),
            self.lookup(&self.suffix_ids, sentence.suffix_key(index, 3)),
            self.lookup(&self.shape_ids, sentence.shape_key(idx - 1)),
            self.lookup(&self.shape_ids, sentence.shape_key(idx)),
            self.lookup(&self.shape_ids, sentence.shape_key(idx + 1)),
            self.lookup(&self.tag_feature_ids, prev_tag.unwrap_or(ABSENT)),
            self.lookup(&self.tag_feature_ids, prev_prev_tag.unwrap_or(ABSENT)),
        ]
    }

    fn lookup(&self, table: &HashMap<String, usize>, symbol: &str) -> usize {
        table.get(symbol).copied().unwrap_or_else(|| table[UNKNOWN])
    }

    /// Resolves the 5 pretrained word-vector row indices (`-1` when absent) around `tokens[index]`.
    pub fn pretrained_rows(&self, tokens: &[impl AsRef<str>], index: usize) -> Option<[isize; PRETRAINED_SLOTS]> {
        if self.pretrained_size == 0 {
            return None;
        }
        let mut rows = [-1isize; PRETRAINED_SLOTS];
        let idx = index as isize;
        for w in 0..PRETRAINED_SLOTS {
            let pos = idx + (w as isize) - 2;
            if pos >= 0 && (pos as usize) < tokens.len() {
                let norm = FeedforwardPosContext::lowercase(tokens[pos as usize].as_ref());
                if let Some(&r) = self.pretrained_ids.get(&norm) {
                    rows[w] = r as isize;
                }
            }
        }
        Some(rows)
    }

    /// Resolves the 5 pretrained word-vector row indices (`-1` when absent)
    /// around `sentence[index]` without allocating per-window `String`s.
    pub(super) fn pretrained_rows_at(
        &self,
        sentence: &SentenceFeatures<'_>,
        index: usize,
    ) -> Option<[isize; PRETRAINED_SLOTS]> {
        if self.pretrained_size == 0 {
            return None;
        }
        let mut rows = [-1isize; PRETRAINED_SLOTS];
        let idx = index as isize;
        let len = sentence.len() as isize;
        for w in 0..PRETRAINED_SLOTS {
            let pos = idx + (w as isize) - 2;
            if pos >= 0
                && pos < len
                && let Some(&r) = self.pretrained_ids.get(sentence.word_key(pos))
            {
                rows[w] = r as isize;
            }
        }
        Some(rows)
    }

    /// Enables the thread-safe contribution cache (bounded at 32,768 `(slot, row)` entries).
    pub fn enable_scoring_cache(&self) {
        if let Ok(mut guard) = self.cache.write()
            && guard.is_none()
        {
            *guard = Some(Arc::new(ContributionCache::new(self.embeddings.len())));
        }
    }

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

    /// Returns the provenance [`ModelRef`] for this model (including the SHA-256 of its serialized bytes).
    pub fn model_ref(&self) -> ModelRef {
        let mut bytes = Vec::new();
        self.serialize(&mut bytes).expect("serializing an in-memory FeedforwardPosModel to Vec<u8> cannot fail");
        let revision = if self.pretrained_size > 0 { MAGIC_V2 } else { MAGIC_V1 };
        ModelRef {
            id: "opennlp:FeedforwardPOSModel".into(),
            revision: revision.into(),
            sha256: turbo_maxent::sha256(&bytes).to_hex(),
        }
    }

    /// Tag outcome inventory in index order.
    pub fn tags(&self) -> &[String] {
        &self.tags
    }

    /// Width of each discrete embedding row.
    pub fn embedding_size(&self) -> usize {
        self.embedding_size
    }

    /// Learned discrete embedding table.
    pub fn embeddings(&self) -> &[Vec<f32>] {
        &self.embeddings
    }

    /// Hidden-layer weight matrix.
    pub fn hidden_weights(&self) -> &[Vec<f32>] {
        &self.hidden_weights
    }

    /// Hidden-layer bias vector.
    pub fn hidden_bias(&self) -> &[f32] {
        &self.hidden_bias
    }

    /// Output-layer weight matrix.
    pub fn output_weights(&self) -> &[Vec<f32>] {
        &self.output_weights
    }

    /// Output-layer bias vector.
    pub fn output_bias(&self) -> &[f32] {
        &self.output_bias
    }

    /// Mutable output-layer bias vector (for test fault injection).
    pub fn output_bias_mut(&mut self) -> &mut [f32] {
        &mut self.output_bias
    }

    /// Width of each pretrained word vector (`0` when unused).
    pub fn pretrained_size(&self) -> usize {
        self.pretrained_size
    }

    /// Whether the model includes a pretrained word-vector table.
    pub fn uses_pretrained_vectors(&self) -> bool {
        self.pretrained_size > 0
    }

    /// Frozen pretrained word-vector rows.
    pub fn pretrained_vectors(&self) -> &[Vec<f32>] {
        &self.pretrained_vectors
    }

    /// Serializes this model in `ONLP-FFPT-1` or `ONLP-FFPT-2` format to `writer`.
    pub fn serialize<W: Write>(&self, writer: W) -> Result<(), NeuralError> {
        let mut out = DataWriter::new(writer);
        out.write_utf(if self.pretrained_size > 0 { MAGIC_V2 } else { MAGIC_V1 })?;
        write_vocab(&mut out, &self.word_entries)?;
        write_vocab(&mut out, &self.suffix_entries)?;
        write_vocab(&mut out, &self.shape_entries)?;
        write_vocab(&mut out, &self.tag_feature_entries)?;
        out.write_i32(self.tags.len() as i32)?;
        for tag in &self.tags {
            out.write_utf(tag)?;
        }
        out.write_i32(self.embedding_size as i32)?;
        write_f32_matrix(&mut out, &self.embeddings)?;
        write_f32_matrix(&mut out, &self.hidden_weights)?;
        write_f32_vector(&mut out, &self.hidden_bias)?;
        write_f32_matrix(&mut out, &self.output_weights)?;
        write_f32_vector(&mut out, &self.output_bias)?;
        if self.pretrained_size > 0 {
            out.write_i32(self.pretrained_size as i32)?;
            write_vocab(&mut out, &self.pretrained_entries)?;
            write_f32_matrix(&mut out, &self.pretrained_vectors)?;
        }
        out.flush()?;
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

    /// Reads and validates a `FeedforwardPosModel` from a stream in `ONLP-FFPT-1` or `ONLP-FFPT-2` format.
    pub fn load<R: Read>(reader: R) -> Result<Self, NeuralError> {
        let mut inp = DataReader::new(reader);
        let magic = inp.read_utf()?;
        if magic != MAGIC_V1 && magic != MAGIC_V2 {
            return Err(NeuralError::Format(format!("not a FeedforwardPOSModel stream (magic '{magic}')")));
        }
        let word_entries = read_vocab(&mut inp, true)?;
        let suffix_entries = read_vocab(&mut inp, true)?;
        let shape_entries = read_vocab(&mut inp, true)?;
        let tag_feature_entries = read_vocab(&mut inp, true)?;

        let num_tags = read_count(&mut inp, "tag count", 1)?;
        let mut tags = Vec::with_capacity(num_tags.min(INITIAL_CAPACITY));
        let mut seen_tags = HashSet::with_capacity(num_tags.min(INITIAL_CAPACITY));
        for _ in 0..num_tags {
            let tag = inp.read_utf()?;
            if !seen_tags.insert(tag.clone()) {
                return Err(NeuralError::Format(format!("duplicate tag: {tag}")));
            }
            tags.push(tag);
        }

        let embedding_size = read_count(&mut inp, "embedding size", 1)?;
        let embeddings = read_f32_matrix(&mut inp, "embeddings", false)?;
        let hidden_weights = read_f32_matrix(&mut inp, "hidden weights", false)?;
        let hidden_bias = read_f32_vector(&mut inp, "hidden bias", 1)?;
        let output_weights = read_f32_matrix(&mut inp, "output weights", false)?;
        let output_bias = read_f32_vector(&mut inp, "output bias", 1)?;

        let (pretrained_size, pretrained_entries, pretrained_vectors) = if magic == MAGIC_V2 {
            let p_size = read_count(&mut inp, "pretrained vector size", 1)?;
            let p_entries = read_vocab(&mut inp, false)?;
            let p_vecs = read_f32_matrix(&mut inp, "pretrained vectors", true)?;
            validate_rows(&p_entries, p_vecs.len(), "pretrained")?;
            if !p_vecs.is_empty() && p_vecs[0].len() != p_size {
                return Err(NeuralError::Format(format!("pretrained vector width {} != {p_size}", p_vecs[0].len())));
            }
            (p_size, p_entries, p_vecs)
        } else {
            (0, Vec::new(), Vec::new())
        };

        validate_rows(&word_entries, embeddings.len(), "words")?;
        validate_rows(&suffix_entries, embeddings.len(), "suffixes")?;
        validate_rows(&shape_entries, embeddings.len(), "shapes")?;
        validate_rows(&tag_feature_entries, embeddings.len(), "tags")?;

        if embeddings[0].len() != embedding_size {
            return Err(NeuralError::Format(format!("embedding width {} != {embedding_size}", embeddings[0].len())));
        }
        let expected_input = SLOTS
            .checked_mul(embedding_size)
            .and_then(|a| PRETRAINED_SLOTS.checked_mul(pretrained_size).and_then(|b| a.checked_add(b)))
            .ok_or_else(|| NeuralError::Format("input width overflow".into()))?;
        if hidden_weights[0].len() != expected_input {
            return Err(NeuralError::Format(format!(
                "hidden input width {} != {expected_input}",
                hidden_weights[0].len()
            )));
        }
        if hidden_bias.len() != hidden_weights.len() {
            return Err(NeuralError::Format(format!(
                "hidden bias length {} != {}",
                hidden_bias.len(),
                hidden_weights.len()
            )));
        }
        if output_weights.len() != tags.len() {
            return Err(NeuralError::Format(format!("output row count {} != {}", output_weights.len(), tags.len())));
        }
        if output_weights[0].len() != hidden_weights.len() {
            return Err(NeuralError::Format(format!(
                "output column count {} != {}",
                output_weights[0].len(),
                hidden_weights.len()
            )));
        }
        if output_bias.len() != tags.len() {
            return Err(NeuralError::Format(format!("output bias length {} != {}", output_bias.len(), tags.len())));
        }

        Ok(Self::with_pretrained(
            word_entries,
            suffix_entries,
            shape_entries,
            tag_feature_entries,
            tags,
            embedding_size,
            embeddings,
            hidden_weights,
            hidden_bias,
            output_weights,
            output_bias,
            pretrained_size,
            pretrained_entries,
            pretrained_vectors,
        ))
    }

    /// Reads and validates a `FeedforwardPosModel` from a file path.
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self, NeuralError> {
        let file = File::open(path)?;
        Self::load(BufReader::new(file))
    }
}

fn entries_to_map(entries: &[(String, usize)]) -> HashMap<String, usize> {
    let mut map = HashMap::with_capacity(entries.len());
    for (k, v) in entries {
        map.insert(k.clone(), *v);
    }
    map
}

fn write_vocab<W: Write>(out: &mut DataWriter<W>, entries: &[(String, usize)]) -> Result<(), NeuralError> {
    out.write_i32(entries.len() as i32)?;
    for (k, v) in entries {
        out.write_utf(k)?;
        out.write_i32(*v as i32)?;
    }
    Ok(())
}

fn read_vocab<R: Read>(inp: &mut DataReader<R>, require_unknown: bool) -> Result<Vec<(String, usize)>, NeuralError> {
    let size = read_count(inp, "vocabulary size", usize::from(require_unknown))?;
    let mut entries = Vec::with_capacity(size.min(INITIAL_CAPACITY));
    let mut seen = HashSet::with_capacity(size.min(INITIAL_CAPACITY));
    let mut has_unk = false;
    for _ in 0..size {
        let key = inp.read_utf()?;
        let row = inp.read_i32()?;
        if row < 0 {
            return Err(NeuralError::Format(format!("negative row index: {row}")));
        }
        if !seen.insert(key.clone()) {
            return Err(NeuralError::Format(format!("duplicate vocabulary entry: {key}")));
        }
        if key == UNKNOWN {
            has_unk = true;
        }
        entries.push((key, row as usize));
    }
    if require_unknown && !has_unk {
        return Err(NeuralError::Format(format!("missing required vocabulary entry: {UNKNOWN}")));
    }
    Ok(entries)
}

fn validate_rows(entries: &[(String, usize)], row_count: usize, name: &str) -> Result<(), NeuralError> {
    for (k, r) in entries {
        if *r >= row_count {
            return Err(NeuralError::Format(format!("{name} row index for '{k}' out of bounds: {r} >= {row_count}")));
        }
    }
    Ok(())
}

fn write_f32_matrix<W: Write>(out: &mut DataWriter<W>, m: &[Vec<f32>]) -> Result<(), NeuralError> {
    let cols = m.first().map_or(0, Vec::len);
    out.write_i32(m.len() as i32)?;
    out.write_i32(cols as i32)?;
    for row in m {
        for &v in row {
            out.write_f32(v)?;
        }
    }
    Ok(())
}

fn read_f32_matrix<R: Read>(
    inp: &mut DataReader<R>,
    name: &str,
    allow_empty: bool,
) -> Result<Vec<Vec<f32>>, NeuralError> {
    let rows = read_count(inp, &format!("{name} rows"), usize::from(!allow_empty))?;
    let cols = read_count(inp, &format!("{name} columns"), 0)?;
    if rows == 0 {
        if cols != 0 {
            return Err(NeuralError::Format(format!("{name} columns must be 0 when rows is 0: {cols}")));
        }
        return Ok(Vec::new());
    }
    if cols < 1 {
        return Err(NeuralError::Format(format!("invalid {name} columns: {cols}")));
    }
    let mut m = Vec::with_capacity(rows.min(INITIAL_CAPACITY));
    for _ in 0..rows {
        m.push(read_f32_values(inp, name, cols)?);
    }
    Ok(m)
}

fn write_f32_vector<W: Write>(out: &mut DataWriter<W>, v: &[f32]) -> Result<(), NeuralError> {
    out.write_i32(v.len() as i32)?;
    for &x in v {
        out.write_f32(x)?;
    }
    Ok(())
}

fn read_f32_vector<R: Read>(inp: &mut DataReader<R>, name: &str, min: usize) -> Result<Vec<f32>, NeuralError> {
    let len = read_count(inp, &format!("{name} length"), min)?;
    read_f32_values(inp, name, len)
}

fn read_f32_values<R: Read>(inp: &mut DataReader<R>, name: &str, len: usize) -> Result<Vec<f32>, NeuralError> {
    let mut out = Vec::with_capacity(len.min(INITIAL_CAPACITY));
    for _ in 0..len {
        let v = inp.read_f32()?;
        if !v.is_finite() {
            return Err(NeuralError::Format(format!("non-finite value in {name}: {v}")));
        }
        out.push(v);
    }
    Ok(out)
}

fn read_count<R: Read>(inp: &mut DataReader<R>, name: &str, min: usize) -> Result<usize, NeuralError> {
    let val = inp.read_i32()?;
    if val < (min as i32) || val > MAX_LENGTH {
        return Err(NeuralError::Format(format!("invalid {name}: {val}")));
    }
    Ok(val as usize)
}
