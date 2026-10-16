//! Feedforward neural part-of-speech tagger (`#205`), ported from OpenNLP's
//! `FeedforwardPOSContext`, `FeedforwardPOSModel`, `FeedforwardPOSTagger`, and
//! `FeedforwardPOSTrainer`.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicIsize, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

use turbo_analysis::{Annotator, Error, ModelRef, Refs, View};
use turbo_analysis_layers::{PosTags, TagSet, Tokens};
use turbo_unicode::java::code_point::{is_digit, is_letter, is_upper_case};
use turbo_unicode::java::to_lower_char;

use crate::neural::{DataReader, DataWriter, JavaRng, NeuralError, NeuralSequence, PosSample};

/// Number of discrete feature slots extracted for each token position.
pub const SLOTS: usize = 12;

/// Number of pretrained word-vector window slots (`[-2, -1, 0, +1, +2]`).
pub const PRETRAINED_SLOTS: usize = 5;

/// Padding symbol for positions outside the sentence bounds or before the first two tags.
pub const ABSENT: &str = "*NULL*";

/// Fallback symbol for words, suffixes, shapes, or tags absent from the vocabulary.
pub const UNKNOWN: &str = "*UNK*";

const SHAPE_LOWER: &str = "*lower*";
const SHAPE_CAP: &str = "*cap*";
const SHAPE_ALLCAPS: &str = "*allcaps*";
const SHAPE_DIGIT: &str = "*digit*";
const SHAPE_ALNUM: &str = "*alnum*";
const SHAPE_OTHER: &str = "*other*";

const MAGIC_V1: &str = "ONLP-FFPT-1";
const MAGIC_V2: &str = "ONLP-FFPT-2";
const CACHE_ENTRY_LIMIT: isize = 32768;
const INITIAL_CAPACITY: usize = 4096;
const MAX_LENGTH: i32 = 1 << 30;
const ADAGRAD_EPS: f64 = 1e-6;

/// Extracts the 12 discrete feature strings for the feedforward neural POS tagger.
pub struct FeedforwardPosContext;

impl FeedforwardPosContext {
    /// Extracts the 12 feature strings for `tokens[index]` given the two preceding assigned tags.
    pub fn extract(
        tokens: &[impl AsRef<str>],
        index: usize,
        prev_tag: Option<&str>,
        prev_prev_tag: Option<&str>,
    ) -> [String; SLOTS] {
        let idx = index as isize;
        let w0 = Self::word(tokens, idx - 2);
        let w1 = Self::word(tokens, idx - 1);
        let w2 = Self::word(tokens, idx);
        let w3 = Self::word(tokens, idx + 1);
        let w4 = Self::word(tokens, idx + 2);
        let lowered = Self::token(tokens, idx).map(Self::lowercase_all).unwrap_or_default();
        let suf2 = Self::suffix(&lowered, 2);
        let suf3 = Self::suffix(&lowered, 3);
        let sh0 = Self::shape_opt(Self::token(tokens, idx - 1)).to_owned();
        let sh1 = Self::shape_opt(Self::token(tokens, idx)).to_owned();
        let sh2 = Self::shape_opt(Self::token(tokens, idx + 1)).to_owned();
        let pt1 = prev_tag.unwrap_or(ABSENT).to_owned();
        let pt2 = prev_prev_tag.unwrap_or(ABSENT).to_owned();
        [w0, w1, w2, w3, w4, suf2, suf3, sh0, sh1, sh2, pt1, pt2]
    }

    fn token<S: AsRef<str>>(tokens: &[S], index: isize) -> Option<&str> {
        if index < 0 || (index as usize) >= tokens.len() { None } else { Some(tokens[index as usize].as_ref()) }
    }

    fn word<S: AsRef<str>>(tokens: &[S], index: isize) -> String {
        match Self::token(tokens, index) {
            None => ABSENT.to_owned(),
            Some(w) => Self::lowercase(w),
        }
    }

    pub(crate) fn lowercase(word: &str) -> String {
        if word.starts_with('*') { word.to_owned() } else { word.chars().map(to_lower_char).collect() }
    }

    /// Lowercases a whole token for the suffix features. Unlike
    /// [`Self::lowercase`] this does not pass symbols starting with `*`
    /// through, matching Java's `StringUtil.toLowerCase` on the token.
    fn lowercase_all(word: &str) -> String {
        word.chars().map(to_lower_char).collect()
    }

    /// Returns the last `length` UTF-16 code units of `word`, or `word`
    /// itself when it is no longer, as Java's `substring` does.
    ///
    /// When the cut falls between the two halves of a surrogate pair, Java
    /// returns a string starting with a lone low surrogate, which a Rust
    /// string cannot hold. That half is replaced by U+FFFD. A model file
    /// whose vocabulary holds a lone surrogate is rejected on load, so on
    /// a Java model such a suffix maps to the unknown symbol either way.
    pub fn suffix(word: &str, length: usize) -> String {
        let mut units = 0;
        for (start, c) in word.char_indices().rev() {
            units += c.len_utf16();
            if units == length {
                return word[start..].to_owned();
            }
            if units > length {
                let rest = &word[start + c.len_utf8()..];
                let mut out = String::with_capacity(rest.len() + 3);
                out.push('\u{FFFD}');
                out.push_str(rest);
                return out;
            }
        }
        word.to_owned()
    }

    fn shape_opt(word: Option<&str>) -> &'static str {
        match word {
            None => ABSENT,
            Some(w) => Self::shape(w),
        }
    }

    /// Classifies `word` into one of the six coarse orthographic shape
    /// symbols, as Java's `FeedforwardPOSContext.shape` does: it walks the
    /// UTF-16 code units, so a supplementary character (a surrogate pair)
    /// is neither a letter nor a digit, and a letter that is not uppercase
    /// counts as lowercase. A word with letters and no digit is all caps
    /// when no letter is lowercase, capitalized when its first code unit
    /// is uppercase, and lowercase otherwise.
    pub fn shape(word: &str) -> &'static str {
        let mut letter = false;
        let mut digit = false;
        let mut upper = false;
        let mut lower = false;
        for unit in word.encode_utf16() {
            let cp = u32::from(unit);
            if is_digit(cp) {
                digit = true;
            } else if is_letter(cp) {
                letter = true;
                if is_upper_case(cp) {
                    upper = true;
                } else {
                    lower = true;
                }
            }
        }
        if letter && digit {
            SHAPE_ALNUM
        } else if digit {
            SHAPE_DIGIT
        } else if !letter {
            SHAPE_OTHER
        } else if upper && !lower {
            SHAPE_ALLCAPS
        } else if word.encode_utf16().next().is_some_and(|unit| is_upper_case(u32::from(unit))) {
            SHAPE_CAP
        } else {
            SHAPE_LOWER
        }
    }

    /// All six coarse orthographic shape symbols in registration order.
    pub fn all_shapes() -> [&'static str; 6] {
        [SHAPE_LOWER, SHAPE_CAP, SHAPE_ALLCAPS, SHAPE_DIGIT, SHAPE_ALNUM, SHAPE_OTHER]
    }
}

#[derive(Debug)]
struct ContributionCache {
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
    fn set_budget(&self, budget: isize) {
        self.remaining.store(budget, Ordering::SeqCst);
    }

    #[cfg(test)]
    fn budget(&self) -> isize {
        self.remaining.load(Ordering::SeqCst)
    }

    #[cfg(test)]
    fn cached_entries(&self) -> usize {
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
    cache: Arc<RwLock<Option<Arc<ContributionCache>>>>,
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
        for s in 0..5 {
            ids[s] = self.lookup(&self.word_ids, context[s].as_ref());
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

/// Thread-safe greedy left-to-right neural POS tagger backed by a [`FeedforwardPosModel`].
#[derive(Clone, Debug)]
pub struct FeedforwardPosTagger {
    model: Arc<FeedforwardPosModel>,
    tagset: Arc<TagSet>,
    model_ref: ModelRef,
}

impl FeedforwardPosTagger {
    /// Creates a new `FeedforwardPosTagger` over `model` and enables its contribution cache.
    pub fn new(model: impl Into<Arc<FeedforwardPosModel>>) -> Self {
        let model = model.into();
        model.enable_scoring_cache();
        let labels: Vec<&str> = model.tags().iter().map(String::as_str).collect();
        let tagset = TagSet::new("feedforward", &labels);
        let model_ref = model.model_ref();
        Self { model, tagset, model_ref }
    }

    /// Returns a reference to the underlying [`FeedforwardPosModel`].
    pub fn model(&self) -> &FeedforwardPosModel {
        &self.model
    }

    /// Returns the [`TagSet`] of the model's outcomes.
    pub fn tagset(&self) -> &Arc<TagSet> {
        &self.tagset
    }

    /// Tags a tokenized sentence greedily from left to right.
    pub fn tag(&self, sentence: &[impl AsRef<str>]) -> Result<Vec<String>, NeuralError> {
        Ok(self.decode(sentence)?.into_outcomes())
    }

    /// Returns a 1-element array containing the greedy tag sequence with per-token softmax probabilities.
    pub fn top_k_sequences(&self, sentence: &[impl AsRef<str>]) -> Result<Vec<NeuralSequence>, NeuralError> {
        Ok(vec![self.decode(sentence)?])
    }

    fn decode(&self, sentence: &[impl AsRef<str>]) -> Result<NeuralSequence, NeuralError> {
        let n = sentence.len();
        let mut assigned: Vec<String> = Vec::with_capacity(n);
        let mut probs: Vec<f64> = Vec::with_capacity(n);
        let tags = self.model.tags();

        for i in 0..n {
            let prev = if i >= 1 { Some(assigned[i - 1].as_str()) } else { None };
            let prev_prev = if i >= 2 { Some(assigned[i - 2].as_str()) } else { None };
            let ctx = FeedforwardPosContext::extract(sentence, i, prev, prev_prev);
            let f_ids = self.model.feature_ids(&ctx);
            let p_rows = self.model.pretrained_rows(sentence, i);
            let scores = self.model.score_with_pretrained(&f_ids, p_rows.as_ref().map(|r| &r[..]))?;

            let mut best = 0usize;
            for (o, &s) in scores.iter().enumerate() {
                if !s.is_finite() {
                    return Err(NeuralError::NonFiniteScore("model produced a non-finite tag score".into()));
                }
                if o > 0 && s > scores[best] {
                    best = o;
                }
            }
            assigned.push(tags[best].clone());

            let max = scores[best];
            let mut sum_exp = 0.0f64;
            for &s in &scores {
                sum_exp += turbo_jvm::strict_math::exp(s - max);
            }
            probs.push(1.0 / sum_exp);
        }
        Ok(NeuralSequence::from_outcomes_and_probs(assigned, probs))
    }
}

impl NeuralSequence {
    fn into_outcomes(self) -> Vec<String> {
        self.outcomes
    }
}

impl Annotator for FeedforwardPosTagger {
    const ID: &'static str = "turbo-postag:feedforward";
    type Reads = (View, Tokens);
    type Writes = (PosTags,);

    fn annotate(&self, (view, tokens): Refs<'_, Self>) -> Result<(PosTags,), Error> {
        if tokens.spans.view() != view.id() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!(
                    "the tokens are in view {}, not in view {} that is tagged",
                    tokens.spans.view().0,
                    view.id().0
                ),
            });
        }
        let text = view.text();
        let (starts, ends) = tokens.spans.columns();
        let n = starts.len();
        let mut tags = Vec::with_capacity(n);
        let mut probs = Vec::with_capacity(n);
        let mut words: Vec<&str> = Vec::new();
        let mut from = 0;
        while from < n {
            let to = match &tokens.sentence {
                Some(col) => from + col[from..].partition_point(|&s| s == col[from]),
                None => n,
            };
            words.clear();
            words.extend((from..to).map(|i| &text[starts[i] as usize..ends[i] as usize]));
            if !words.is_empty() {
                let seq =
                    self.decode(&words).map_err(|e| Error::Annotator { annotator: Self::ID, detail: e.to_string() })?;
                for outcome in seq.outcomes() {
                    let tag = self.tagset.tag(outcome).expect("outcome is in tagset");
                    tags.push(tag);
                }
                probs.extend_from_slice(seq.probs());
            }
            from = to;
        }
        Ok((PosTags { tagset: self.tagset.clone(), tags, prob: Some(probs) },))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.model_ref.clone())
    }
}

/// Hyperparameters for [`FeedforwardPosTrainer`], matching `FeedforwardPOSTrainer.Settings`.
#[derive(Clone, Debug, PartialEq)]
pub struct FeedforwardSettings {
    pub embedding_size: usize,
    pub hidden_size: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub l2: f64,
    pub dropout: f64,
    pub word_cutoff: usize,
    pub suffix_cutoff: usize,
    pub seed: i64,
}

impl FeedforwardSettings {
    /// Creates and validates `FeedforwardSettings`, matching `FeedforwardPOSTrainer.Settings`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        embedding_size: i32,
        hidden_size: i32,
        epochs: i32,
        batch_size: i32,
        learning_rate: f64,
        l2: f64,
        dropout: f64,
        word_cutoff: i32,
        suffix_cutoff: i32,
        seed: i64,
    ) -> Result<Self, NeuralError> {
        if embedding_size <= 0 {
            return Err(NeuralError::InvalidArgument(format!("embeddingSize must be positive: {embedding_size}")));
        }
        if hidden_size <= 0 {
            return Err(NeuralError::InvalidArgument(format!("hiddenSize must be positive: {hidden_size}")));
        }
        if epochs <= 0 {
            return Err(NeuralError::InvalidArgument(format!("epochs must be positive: {epochs}")));
        }
        if batch_size <= 0 {
            return Err(NeuralError::InvalidArgument(format!("batchSize must be positive: {batch_size}")));
        }
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(NeuralError::InvalidArgument(format!(
                "learningRate must be finite and positive: {learning_rate}"
            )));
        }
        if !l2.is_finite() || l2 < 0.0 {
            return Err(NeuralError::InvalidArgument(format!("l2 must be finite and non-negative: {l2}")));
        }
        if !dropout.is_finite() || !(0.0..1.0).contains(&dropout) {
            return Err(NeuralError::InvalidArgument(format!("dropout must be finite and in [0, 1): {dropout}")));
        }
        if word_cutoff < 0 {
            return Err(NeuralError::InvalidArgument(format!("wordCutoff must be non-negative: {word_cutoff}")));
        }
        if suffix_cutoff < 0 {
            return Err(NeuralError::InvalidArgument(format!("suffixCutoff must be non-negative: {suffix_cutoff}")));
        }
        Ok(Self {
            embedding_size: embedding_size as usize,
            hidden_size: hidden_size as usize,
            epochs: epochs as usize,
            batch_size: batch_size as usize,
            learning_rate,
            l2,
            dropout,
            word_cutoff: word_cutoff as usize,
            suffix_cutoff: suffix_cutoff as usize,
            seed,
        })
    }

    /// Default hyperparameters matching `FeedforwardPOSTrainer.Settings.defaults()`.
    pub fn defaults() -> Self {
        Self::new(32, 128, 10, 64, 0.05, 1e-6, 0.2, 2, 2, 42).unwrap()
    }
}

struct Example {
    features: [usize; SLOTS],
    pretrained_rows: Option<[isize; PRETRAINED_SLOTS]>,
    gold: usize,
}

/// Mini-batch Adagrad trainer for [`FeedforwardPosModel`], ported from `FeedforwardPOSTrainer`.
pub struct FeedforwardPosTrainer;

impl FeedforwardPosTrainer {
    /// Trains a [`FeedforwardPosModel`] without pretrained word vectors (`ONLP-FFPT-1`).
    pub fn train(samples: &[PosSample], settings: FeedforwardSettings) -> Result<FeedforwardPosModel, NeuralError> {
        Self::train_internal(samples, settings, None::<fn(&str) -> Option<Vec<f32>>>, None)
    }

    /// Trains a [`FeedforwardPosModel`] with pretrained word vectors (`ONLP-FFPT-2`).
    pub fn train_with_vectors<F>(
        samples: &[PosSample],
        settings: FeedforwardSettings,
        vectors: F,
    ) -> Result<FeedforwardPosModel, NeuralError>
    where
        F: Fn(&str) -> Option<Vec<f32>>,
    {
        Self::train_internal(samples, settings, Some(vectors), None)
    }

    /// Trains a [`FeedforwardPosModel`] with pretrained word vectors and an additional lexicon (`ONLP-FFPT-2`).
    pub fn train_with_lexicon<F>(
        samples: &[PosSample],
        settings: FeedforwardSettings,
        vectors: F,
        additional_words: &[String],
    ) -> Result<FeedforwardPosModel, NeuralError>
    where
        F: Fn(&str) -> Option<Vec<f32>>,
    {
        Self::train_internal(samples, settings, Some(vectors), Some(additional_words))
    }

    fn train_internal<F>(
        samples: &[PosSample],
        settings: FeedforwardSettings,
        vectors: Option<F>,
        additional_words: Option<&[String]>,
    ) -> Result<FeedforwardPosModel, NeuralError>
    where
        F: Fn(&str) -> Option<Vec<f32>>,
    {
        let mut corpus: Vec<&PosSample> = Vec::new();
        let mut word_order: Vec<String> = Vec::new();
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        let mut suffix_order: Vec<String> = Vec::new();
        let mut suffix_counts: HashMap<String, usize> = HashMap::new();
        let mut tag_order: Vec<String> = Vec::new();
        let mut tag_set: HashSet<String> = HashSet::new();

        for sample in samples {
            if sample.sentence.is_empty() {
                continue;
            }
            corpus.push(sample);
            for i in 0..sample.sentence.len() {
                let lower = FeedforwardPosContext::lowercase(&sample.sentence[i]);
                if let Some(c) = word_counts.get_mut(&lower) {
                    *c += 1;
                } else {
                    word_order.push(lower.clone());
                    word_counts.insert(lower.clone(), 1);
                }
                let lowered = FeedforwardPosContext::lowercase_all(&sample.sentence[i]);
                let s2 = FeedforwardPosContext::suffix(&lowered, 2);
                if let Some(c) = suffix_counts.get_mut(&s2) {
                    *c += 1;
                } else {
                    suffix_order.push(s2.clone());
                    suffix_counts.insert(s2, 1);
                }
                let s3 = FeedforwardPosContext::suffix(&lowered, 3);
                if let Some(c) = suffix_counts.get_mut(&s3) {
                    *c += 1;
                } else {
                    suffix_order.push(s3.clone());
                    suffix_counts.insert(s3, 1);
                }
                let tg = &sample.tags[i];
                if tag_set.insert(tg.clone()) {
                    tag_order.push(tg.clone());
                }
            }
        }
        if corpus.is_empty() {
            return Err(NeuralError::InvalidArgument("Training stream produced no non-empty POS samples.".into()));
        }

        let mut next_row = 0usize;
        let mut word_entries = Vec::new();
        let mut suffix_entries = Vec::new();
        let mut shape_entries = Vec::new();
        let mut tag_feature_entries = Vec::new();

        next_row = register_vocab(&mut word_entries, &word_order, &word_counts, settings.word_cutoff, next_row);
        next_row = register_vocab(&mut suffix_entries, &suffix_order, &suffix_counts, settings.suffix_cutoff, next_row);
        next_row = register_all(&mut shape_entries, FeedforwardPosContext::all_shapes().iter().copied(), next_row);
        next_row = register_all(&mut tag_feature_entries, tag_order.iter().map(String::as_str), next_row);
        let total_rows = next_row;

        let mut pretrained_entries = Vec::new();
        let mut pretrained_rows_vec: Vec<Vec<f32>> = Vec::new();
        let mut pretrained_size = 0usize;
        if let Some(ref vec_fn) = vectors {
            let mut seen_p = HashSet::new();
            for word in &word_order {
                if let Some(found) = vec_fn(word) {
                    let row = check_vector(word, &found, &mut pretrained_size)?;
                    pretrained_entries.push((word.clone(), pretrained_rows_vec.len()));
                    seen_p.insert(word.clone());
                    pretrained_rows_vec.push(row);
                }
            }
            if let Some(extra) = additional_words {
                for raw in extra {
                    let word = FeedforwardPosContext::lowercase(raw);
                    if seen_p.contains(&word) || word_counts.contains_key(&word) {
                        continue;
                    }
                    if let Some(found) = vec_fn(&word) {
                        let row = check_vector(&word, &found, &mut pretrained_size)?;
                        pretrained_entries.push((word.clone(), pretrained_rows_vec.len()));
                        seen_p.insert(word);
                        pretrained_rows_vec.push(row);
                    }
                }
            }
            if pretrained_rows_vec.is_empty() {
                return Err(NeuralError::InvalidArgument("vectors returned null for every vocabulary word".into()));
            }
        }

        let tags = tag_order;
        let mut tag_index = HashMap::with_capacity(tags.len());
        for (i, t) in tags.iter().enumerate() {
            tag_index.insert(t.clone(), i);
        }

        let emb_size = settings.embedding_size;
        let hid_size = settings.hidden_size;
        let input_size = SLOTS * emb_size + PRETRAINED_SLOTS * pretrained_size;
        let num_tags = tags.len();

        let shell = FeedforwardPosModel::with_pretrained(
            word_entries.clone(),
            suffix_entries.clone(),
            shape_entries.clone(),
            tag_feature_entries.clone(),
            tags.clone(),
            emb_size,
            vec![vec![0.0; emb_size]; total_rows],
            vec![vec![0.0; input_size]; hid_size],
            vec![0.0; hid_size],
            vec![vec![0.0; hid_size]; num_tags],
            vec![0.0; num_tags],
            pretrained_size,
            pretrained_entries.clone(),
            pretrained_rows_vec.clone(),
        );

        let mut examples: Vec<Example> = Vec::new();
        for sample in &corpus {
            let sent = &sample.sentence;
            let gold_tags = &sample.tags;
            for i in 0..sent.len() {
                let prev = if i >= 1 { Some(gold_tags[i - 1].as_str()) } else { None };
                let prev_prev = if i >= 2 { Some(gold_tags[i - 2].as_str()) } else { None };
                let ctx = FeedforwardPosContext::extract(sent, i, prev, prev_prev);
                examples.push(Example {
                    features: shell.feature_ids(&ctx),
                    pretrained_rows: shell.pretrained_rows(sent, i),
                    gold: tag_index[&gold_tags[i]],
                });
            }
        }

        let mut init_rng = JavaRng::new(settings.seed);
        let mut embeddings = uniform_f32_matrix(&mut init_rng, total_rows, emb_size, 0.01);
        let mut hidden_weights =
            uniform_f32_matrix(&mut init_rng, hid_size, input_size, (6.0 / (input_size + hid_size) as f64).sqrt());
        let mut hidden_bias = vec![0.0f32; hid_size];
        let mut output_weights =
            uniform_f32_matrix(&mut init_rng, num_tags, hid_size, (6.0 / (hid_size + num_tags) as f64).sqrt());
        let mut output_bias = vec![0.0f32; num_tags];

        let mut g2_emb = vec![vec![0.0f64; emb_size]; total_rows];
        let mut g2_hw = vec![vec![0.0f64; input_size]; hid_size];
        let mut g2_hb = vec![0.0f64; hid_size];
        let mut g2_ow = vec![vec![0.0f64; hid_size]; num_tags];
        let mut g2_ob = vec![0.0f64; num_tags];

        let mut d_emb = vec![vec![0.0f64; emb_size]; total_rows];
        let mut d_hw = vec![vec![0.0f64; input_size]; hid_size];
        let mut d_hb = vec![0.0f64; hid_size];
        let mut d_ow = vec![vec![0.0f64; hid_size]; num_tags];
        let mut d_ob = vec![0.0f64; num_tags];
        let mut touched_rows: Vec<usize> = Vec::new();
        let mut row_touched = vec![false; total_rows];

        let mut x = vec![0.0f64; input_size];
        let mut dx = vec![0.0f64; input_size];
        let mut hidden_delta = vec![0.0f64; hid_size];
        let mut pre = vec![0.0f64; hid_size];
        let mut h = vec![0.0f64; hid_size];
        let mut mask = vec![false; hid_size];
        let mut probs = vec![0.0f64; num_tags];

        let mut rng = JavaRng::new(settings.seed);
        let mut order: Vec<usize> = (0..examples.len()).collect();
        let keep_prob = 1.0 - settings.dropout;
        let pretrained_offset = SLOTS * emb_size;

        for _epoch in 0..settings.epochs {
            rng.shuffle(&mut order);
            let mut batch_start = 0;
            while batch_start < order.len() {
                let batch_end = (batch_start + settings.batch_size).min(order.len());
                let batch = (batch_end - batch_start) as f64;

                for j in 0..hid_size {
                    d_hw[j].fill(0.0);
                }
                d_hb.fill(0.0);
                for o in 0..num_tags {
                    d_ow[o].fill(0.0);
                }
                d_ob.fill(0.0);
                for &r in &touched_rows {
                    d_emb[r].fill(0.0);
                    row_touched[r] = false;
                }
                touched_rows.clear();

                for idx in &order[batch_start..batch_end] {
                    let ex = &examples[*idx];
                    for s in 0..SLOTS {
                        let e_row = &embeddings[ex.features[s]];
                        let offset = s * emb_size;
                        for d in 0..emb_size {
                            x[offset + d] = f64::from(e_row[d]);
                        }
                    }
                    if pretrained_size > 0
                        && let Some(ref p_rows) = ex.pretrained_rows
                    {
                        for w in 0..PRETRAINED_SLOTS {
                            let offset = pretrained_offset + w * pretrained_size;
                            let row = p_rows[w];
                            if row < 0 {
                                x[offset..offset + pretrained_size].fill(0.0);
                            } else {
                                let vec = &pretrained_rows_vec[row as usize];
                                for i in 0..pretrained_size {
                                    x[offset + i] = f64::from(vec[i]);
                                }
                            }
                        }
                    }

                    for j in 0..hid_size {
                        mask[j] = rng.next_double() < keep_prob;
                        if !mask[j] {
                            pre[j] = 0.0;
                            h[j] = 0.0;
                            continue;
                        }
                        let w = &hidden_weights[j];
                        let mut sum = f64::from(hidden_bias[j]);
                        for i in 0..input_size {
                            sum += f64::from(w[i]) * x[i];
                        }
                        pre[j] = sum;
                        h[j] = sum * sum * sum / keep_prob;
                    }

                    let mut max_score = f64::NEG_INFINITY;
                    for o in 0..num_tags {
                        let w = &output_weights[o];
                        let mut sum = f64::from(output_bias[o]);
                        for j in 0..hid_size {
                            sum += f64::from(w[j]) * h[j];
                        }
                        probs[o] = sum;
                        if sum > max_score {
                            max_score = sum;
                        }
                    }
                    let mut z = 0.0f64;
                    for o in 0..num_tags {
                        let p = (probs[o] - max_score).exp();
                        probs[o] = p;
                        z += p;
                    }
                    for o in 0..num_tags {
                        probs[o] /= z;
                    }

                    hidden_delta.fill(0.0);
                    dx.fill(0.0);
                    for o in 0..num_tags {
                        let delta = probs[o] - if o == ex.gold { 1.0 } else { 0.0 };
                        d_ob[o] += delta;
                        let dow = &mut d_ow[o];
                        let w_row = &output_weights[o];
                        for j in 0..hid_size {
                            dow[j] += delta * h[j];
                            hidden_delta[j] += delta * f64::from(w_row[j]);
                        }
                    }
                    for j in 0..hid_size {
                        if !mask[j] {
                            continue;
                        }
                        let d_pre = hidden_delta[j] * 3.0 * pre[j] * pre[j] / keep_prob;
                        d_hb[j] += d_pre;
                        let w = &hidden_weights[j];
                        let dw = &mut d_hw[j];
                        for i in 0..input_size {
                            dw[i] += d_pre * x[i];
                            dx[i] += d_pre * f64::from(w[i]);
                        }
                    }
                    for s in 0..SLOTS {
                        let row = ex.features[s];
                        if !row_touched[row] {
                            row_touched[row] = true;
                            touched_rows.push(row);
                        }
                        let de = &mut d_emb[row];
                        let offset = s * emb_size;
                        for i in 0..emb_size {
                            de[i] += dx[offset + i];
                        }
                    }
                }

                let lr = settings.learning_rate;
                let l2 = settings.l2;
                adagrad_matrix(&mut hidden_weights, &d_hw, &mut g2_hw, batch, lr, l2)?;
                adagrad_vector(&mut hidden_bias, &d_hb, &mut g2_hb, batch, lr)?;
                adagrad_matrix(&mut output_weights, &d_ow, &mut g2_ow, batch, lr, l2)?;
                adagrad_vector(&mut output_bias, &d_ob, &mut g2_ob, batch, lr)?;
                for &row in &touched_rows {
                    let e_row = &mut embeddings[row];
                    let g_row = &d_emb[row];
                    let a_row = &mut g2_emb[row];
                    for d in 0..emb_size {
                        let g = g_row[d] / batch;
                        a_row[d] += g * g;
                        e_row[d] = update_weight(e_row[d], g, a_row[d], lr)?;
                    }
                }

                batch_start = batch_end;
            }
        }

        Ok(FeedforwardPosModel::with_pretrained(
            word_entries,
            suffix_entries,
            shape_entries,
            tag_feature_entries,
            tags,
            emb_size,
            embeddings,
            hidden_weights,
            hidden_bias,
            output_weights,
            output_bias,
            pretrained_size,
            pretrained_entries,
            pretrained_rows_vec,
        ))
    }
}

fn check_vector(word: &str, found: &[f32], pretrained_size: &mut usize) -> Result<Vec<f32>, NeuralError> {
    if found.is_empty() {
        return Err(NeuralError::InvalidArgument(format!("empty vector returned for '{word}'")));
    }
    if *pretrained_size == 0 {
        *pretrained_size = found.len();
    } else if found.len() != *pretrained_size {
        return Err(NeuralError::InvalidArgument(format!(
            "vector length mismatch for '{word}': expected {}, got {}",
            *pretrained_size,
            found.len()
        )));
    }
    for (i, &v) in found.iter().enumerate() {
        if !v.is_finite() {
            return Err(NeuralError::InvalidArgument(format!(
                "non-finite vector component for '{word}' at index {i}: {v}"
            )));
        }
    }
    Ok(found.to_vec())
}

fn register_vocab(
    entries: &mut Vec<(String, usize)>,
    order: &[String],
    counts: &HashMap<String, usize>,
    cutoff: usize,
    mut next_row: usize,
) -> usize {
    let mut seen = HashSet::new();
    entries.push((UNKNOWN.to_owned(), next_row));
    seen.insert(UNKNOWN.to_owned());
    next_row += 1;
    entries.push((ABSENT.to_owned(), next_row));
    seen.insert(ABSENT.to_owned());
    next_row += 1;
    for s in order {
        if counts[s] >= cutoff && seen.insert(s.clone()) {
            entries.push((s.clone(), next_row));
            next_row += 1;
        }
    }
    next_row
}

fn register_all<'a>(
    entries: &mut Vec<(String, usize)>,
    symbols: impl IntoIterator<Item = &'a str>,
    mut next_row: usize,
) -> usize {
    let mut seen = HashSet::new();
    entries.push((UNKNOWN.to_owned(), next_row));
    seen.insert(UNKNOWN.to_owned());
    next_row += 1;
    entries.push((ABSENT.to_owned(), next_row));
    seen.insert(ABSENT.to_owned());
    next_row += 1;
    for s in symbols {
        if seen.insert(s.to_owned()) {
            entries.push((s.to_owned(), next_row));
            next_row += 1;
        }
    }
    next_row
}

fn uniform_f32_matrix(rng: &mut JavaRng, rows: usize, cols: usize, scale: f64) -> Vec<Vec<f32>> {
    let mut m = vec![vec![0.0f32; cols]; rows];
    for row in &mut m {
        for val in row {
            *val = ((rng.next_double() * 2.0 - 1.0) * scale) as f32;
        }
    }
    m
}

fn adagrad_matrix(
    w: &mut [Vec<f32>],
    dw: &[Vec<f64>],
    g2: &mut [Vec<f64>],
    batch: f64,
    lr: f64,
    l2: f64,
) -> Result<(), NeuralError> {
    for r in 0..w.len() {
        let w_row = &mut w[r];
        let g_row = &dw[r];
        let a_row = &mut g2[r];
        for c in 0..w_row.len() {
            let g = g_row[c] / batch + l2 * f64::from(w_row[c]);
            a_row[c] += g * g;
            w_row[c] = update_weight(w_row[c], g, a_row[c], lr)?;
        }
    }
    Ok(())
}

fn adagrad_vector(b: &mut [f32], db: &[f64], g2: &mut [f64], batch: f64, lr: f64) -> Result<(), NeuralError> {
    for i in 0..b.len() {
        let g = db[i] / batch;
        g2[i] += g * g;
        b[i] = update_weight(b[i], g, g2[i], lr)?;
    }
    Ok(())
}

#[inline]
fn update_weight(weight: f32, gradient: f64, accumulator: f64, learning_rate: f64) -> Result<f32, NeuralError> {
    let updated = (f64::from(weight) - learning_rate * gradient / (accumulator.sqrt() + ADAGRAD_EPS)) as f32;
    if !gradient.is_finite() || !accumulator.is_finite() || !updated.is_finite() {
        return Err(NeuralError::NonFiniteScore(
            "training produced non-finite values; check learningRate, l2 and word vectors".into(),
        ));
    }
    Ok(updated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural::crf::CrfScoreTable;

    fn corpus() -> Vec<PosSample> {
        let distinct = vec![
            PosSample::new(["the", "dog", "barks"], ["DT", "NN", "VBZ"]).unwrap(),
            PosSample::new(["the", "cat", "sleeps"], ["DT", "NN", "VBZ"]).unwrap(),
            PosSample::new(["dogs", "bark"], ["NNS", "VBP"]).unwrap(),
            PosSample::new(["she", "eats", "fish"], ["PRP", "VBZ", "NN"]).unwrap(),
        ];
        let mut out = Vec::with_capacity(40 * distinct.len());
        for _ in 0..40 {
            out.extend_from_slice(&distinct);
        }
        out
    }

    #[test]
    fn shapes_and_suffixes_match_java_conformance() {
        // Each row is what Java's FeedforwardPOSContext.shape returns.
        let shape_cases = [
            ("Paris", "*cap*"),
            ("Mr.", "*cap*"),
            ("U.S.", "*allcaps*"),
            ("3.5", "*digit*"),
            ("iPhone", "*lower*"),
            ("ALL-CAPS", "*allcaps*"),
            ("x", "*lower*"),
            ("X", "*allcaps*"),
            ("hello-world", "*lower*"),
            ("Hello-World", "*cap*"),
            ("McDonald's", "*cap*"),
            ("USA", "*allcaps*"),
            ("2020", "*digit*"),
            ("1,000", "*digit*"),
            ("B2B", "*alnum*"),
            ("3rd", "*alnum*"),
            ("--", "*other*"),
            (".", "*other*"),
            (",", "*other*"),
            ("$", "*other*"),
            ("(", "*other*"),
            ("dog", "*lower*"),
            ("", "*other*"),
            // Titlecase and other letters are not uppercase, so they count as lowercase.
            ("\u{1C5}", "*lower*"),
            ("\u{1C4}\u{1C5}", "*cap*"),
            ("\u{4E2D}\u{6587}", "*lower*"),
            // Non-ASCII decimal digits are digits.
            ("\u{0663}", "*digit*"),
            // Java walks UTF-16 code units, so a surrogate pair is neither a letter nor a digit.
            ("\u{10400}\u{10428}", "*other*"),
            ("\u{10428}1", "*digit*"),
            ("\u{10400}a", "*lower*"),
            ("\u{10400}A", "*allcaps*"),
            // A circled letter is uppercase but not a letter.
            ("\u{24B6}b", "*cap*"),
        ];
        for (word, expected) in shape_cases {
            assert_eq!(FeedforwardPosContext::shape(word), expected, "{word}");
        }

        // Java takes the last UTF-16 code units; a cut inside a surrogate pair leaves U+FFFD.
        let suffix_cases = [
            ("dog", 2, "og"),
            ("dog", 3, "dog"),
            ("a", 2, "a"),
            ("a", 3, "a"),
            ("", 2, ""),
            ("running", 3, "ing"),
            ("a\u{10428}", 2, "\u{10428}"),
            ("a\u{10428}", 3, "a\u{10428}"),
            ("\u{10428}\u{10429}\u{1042A}", 2, "\u{1042A}"),
            ("\u{10428}\u{10429}\u{1042A}", 3, "\u{FFFD}\u{1042A}"),
            ("ab\u{10428}", 2, "\u{10428}"),
            ("ab\u{10428}", 3, "b\u{10428}"),
        ];
        for (word, len, expected) in suffix_cases {
            assert_eq!(FeedforwardPosContext::suffix(word, len), expected, "{word} {len}");
        }

        // Java lowercases the whole token for the suffixes; only the word
        // lookup passes symbols starting with `*` through unchanged.
        let symbols = FeedforwardPosContext::extract(&["*ABCD", "Mr."], 0, None, None);
        assert_eq!(symbols[2], "*ABCD");
        assert_eq!(&symbols[5..10], ["cd", "bcd", ABSENT, "*allcaps*", "*cap*"]);
    }

    #[test]
    fn training_inference_and_round_trip_match_java_exact_outputs() {
        let settings = FeedforwardSettings::new(16, 32, 80, 32, 0.05, 0.0, 0.0, 1, 1, 17).unwrap();
        let model = FeedforwardPosTrainer::train(&corpus(), settings).unwrap();
        let tagger = FeedforwardPosTagger::new(model.clone());

        assert_eq!(tagger.tag(&["the", "dog", "barks"]).unwrap(), vec!["DT", "NN", "VBZ"]);
        assert_eq!(tagger.tag(&["she", "eats", "fish"]).unwrap(), vec!["PRP", "VBZ", "NN"]);
        assert_eq!(tagger.tag(&["dog"]).unwrap(), vec!["NN"]);
        assert_eq!(tagger.tag(&["the", "wug", "barks"]).unwrap(), vec!["DT", "NN", "VBZ"]);
        assert_eq!(tagger.tag(&["blorp", "quexination"]).unwrap(), vec!["NNS", "VBZ"]);
        assert!(tagger.tag(&Vec::<&str>::new()).unwrap().is_empty());

        let seqs = tagger.top_k_sequences(&["the", "dog", "barks"]).unwrap();
        assert_eq!(seqs.len(), 1);
        assert_eq!(seqs[0].outcomes(), &["DT", "NN", "VBZ"]);
        for &p in seqs[0].probs() {
            assert!(p > 0.9 && p <= 1.0);
        }

        let mut buf = Vec::new();
        model.serialize(&mut buf).unwrap();
        let reloaded = FeedforwardPosModel::load(&buf[..]).unwrap();
        let reloaded_tagger = FeedforwardPosTagger::new(reloaded);
        assert_eq!(
            tagger.tag(&["the", "wug", "barks"]).unwrap(),
            reloaded_tagger.tag(&["the", "wug", "barks"]).unwrap()
        );
    }

    #[test]
    fn pretrained_vectors_and_lexicon_separate_indistinguishable_words() {
        let class_a = ["aazqx", "bazqx", "cazqx", "dazqx"];
        let class_b = ["eezqx", "fezqx", "gezqx", "hezqx"];
        let mut samples = Vec::new();
        for w in class_a {
            samples.push(PosSample::new([w], ["A"]).unwrap());
        }
        for w in class_b {
            samples.push(PosSample::new([w], ["B"]).unwrap());
        }
        let settings = FeedforwardSettings::new(8, 16, 60, 4, 0.1, 1e-8, 0.0, 2, 2, 17).unwrap();
        let vec_fn = |w: &str| Some(if w.as_bytes()[0] <= b'd' { vec![1.0f32, 0.0f32] } else { vec![0.0f32, 1.0f32] });
        let lexicon = vec!["aazqy".to_owned(), "eezqy".to_owned()];
        let model = FeedforwardPosTrainer::train_with_lexicon(&samples, settings, vec_fn, &lexicon).unwrap();
        assert!(model.uses_pretrained_vectors());
        let tagger = FeedforwardPosTagger::new(model.clone());
        for w in class_a {
            assert_eq!(tagger.tag(&[w]).unwrap(), vec!["A"]);
        }
        for w in class_b {
            assert_eq!(tagger.tag(&[w]).unwrap(), vec!["B"]);
        }
        assert_eq!(tagger.tag(&["aazqy"]).unwrap(), vec!["A"]);
        assert_eq!(tagger.tag(&["eezqy"]).unwrap(), vec!["B"]);

        let mut buf = Vec::new();
        model.serialize(&mut buf).unwrap();
        assert_eq!(&buf[2..2 + buf[1] as usize], b"ONLP-FFPT-2");
        let loaded = FeedforwardPosModel::load(&buf[..]).unwrap();
        assert_eq!(
            FeedforwardPosTagger::new(loaded).tag(&["aazqx", "eezqx", "zzzqx"]).unwrap(),
            tagger.tag(&["aazqx", "eezqx", "zzzqx"]).unwrap()
        );
    }

    #[test]
    fn scoring_cache_budget_and_cancellation() {
        let vocab = vec![(UNKNOWN.to_owned(), 0), (ABSENT.to_owned(), 0), ("cat".to_owned(), 0)];
        let emb = vec![vec![1.0f32, CrfScoreTable::scalb(1.0, -24) as f32]];
        let mut hw = vec![vec![0.0f32; 24]];
        hw[0][0] = 1.0;
        hw[0][1] = 1.0;
        let hb = vec![-1.0f32];
        let scale = CrfScoreTable::scalb(1.0, 80) as f32;
        let ow = vec![vec![-scale], vec![scale]];
        let ob = vec![0.0f32, 0.0f32];

        let model = FeedforwardPosModel::new(
            vocab.clone(),
            vocab.clone(),
            vocab.clone(),
            vocab,
            vec!["A".into(), "B".into()],
            2,
            emb,
            hw,
            hb,
            ow,
            ob,
        );
        let ctx = FeedforwardPosContext::extract(&["cat"], 0, None, None);
        let f_ids = model.feature_ids(&ctx);
        let direct = model.score(&f_ids).unwrap();
        model.enable_scoring_cache();
        let cached1 = model.score(&f_ids).unwrap();
        let cached2 = model.score(&f_ids).unwrap();
        assert_eq!(direct, cached1);
        assert_eq!(direct, cached2);

        if let Some(ref c) = *model.cache.read().unwrap() {
            assert!(c.budget() < CACHE_ENTRY_LIMIT);
            assert!(c.cached_entries() > 0);
            c.set_budget(0);
            assert_eq!(c.budget(), 0);
        }
    }

    /// Builds a one-row model whose single slot adds two products of 2^-53 to
    /// a hidden bias of 1.0: summed one by one into the bias they round away,
    /// summed from zero first they do not.
    fn order_sensitive_model() -> FeedforwardPosModel {
        let vocab = vec![(UNKNOWN.to_owned(), 0), (ABSENT.to_owned(), 0)];
        let tiny = CrfScoreTable::scalb(1.0, -53) as f32;
        let emb = vec![vec![tiny, tiny]];
        let mut hw = vec![vec![0.0f32; 24]];
        hw[0][0] = 1.0;
        hw[0][1] = 1.0;
        let scale = CrfScoreTable::scalb(1.0, 60) as f32;
        FeedforwardPosModel::new(
            vocab.clone(),
            vocab.clone(),
            vocab.clone(),
            vocab,
            vec!["A".into(), "B".into()],
            2,
            emb,
            hw,
            vec![1.0f32],
            vec![vec![-scale], vec![scale]],
            vec![0.0f32, 0.0f32],
        )
    }

    #[test]
    fn scores_do_not_depend_on_the_cache_state() {
        let ctx = FeedforwardPosContext::extract(&["cat"], 0, None, None);

        let uncached = order_sensitive_model();
        let f_ids = uncached.feature_ids(&ctx);
        let direct = uncached.score(&f_ids).unwrap();

        let cached = order_sensitive_model();
        cached.enable_scoring_cache();
        let with_room = cached.score(&f_ids).unwrap();

        let full = order_sensitive_model();
        full.enable_scoring_cache();
        if let Some(ref c) = *full.cache.read().unwrap() {
            c.set_budget(0);
        }
        let without_room = full.score(&f_ids).unwrap();

        assert_eq!(direct, with_room);
        assert_eq!(without_room, with_room);
    }
}
