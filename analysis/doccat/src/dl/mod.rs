//! Transformer document categorizer (`DocumentCategorizerDL`), ported from
//! OpenNLP 3.x (`opennlp.dl.doccat.DocumentCategorizerDL`,
//! `DocumentCategorizerConfig`, `ClassificationScoringStrategy`,
//! `AverageClassificationScoringStrategy`, and `InferenceOptions`) onto the
//! analysis API (`#208`).
//!
//! # Architecture
//!
//! - [`DocumentCategorizerConfig`] parses `"id2label"` from a Hugging Face
//!   `config.json` string (`from_json`, `read_categories_from_json`).
//! - [`DlInferenceOptions`] holds the windowing and text-preprocessing
//!   options (`lower_case` default `true`, `document_split_size` default `250`,
//!   `split_overlap_size` default `50`, `normalize_whitespace`,
//!   `normalize_dashes`, `include_token_type_ids`, and [`DlChunkingMode`]).
//! - [`whitespace_chunks`] splits long text on Unicode `White_Space` into
//!   overlapping word chunks (`document_split_size` words, `split_overlap_size`
//!   overlap) joined by single spaces, matching `AbstractDL.whitespaceChunks`.
//! - [`softmax`] validates finiteness of every `f32` logit and computes
//!   numerically stable softmax probabilities in `f64`, matching
//!   `AbstractDL.softmax`.
//! - [`ClassificationScoringStrategy`] (with default
//!   [`AverageClassificationScoringStrategy`]) combines per-chunk probability
//!   distributions into a single document or span distribution.
//! - [`DlSequenceScorer`] abstracts the sequence-classification transformer
//!   session (implemented on turboembed's `CPU`, `CUDA`, `Metal`, `Level Zero`,
//!   and `NPU` backends by `turbo_analysis_embed::BundleSequenceScorer`).
//! - [`DocumentCategorizerDL`] and [`DlSpanCategorizer`] categorize tokenized
//!   or raw text and implement [`turbo_analysis::Annotator`] reading
//!   `(View, Option<Subwords>)` (and `L: SpanCategorizable`: `Chunks`,
//!   `Structure`, or `Sentences`) to write `Categories` (`turbo:categories`)
//!   and `Categories<L>` (`turbo:categories<L>`).

mod annotate;
mod config;
mod options;
mod scoring;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Write as _};
use std::sync::Arc;

use turbo_analysis::{Execution, ModelRef, TagSet};
use turbo_analysis_layers::Subwords;

use crate::layer::SpanCategorizable;

pub use annotate::DlSpanCategorizer;
pub use config::{DocumentCategorizerConfig, ID2LABEL_KEY};
pub use options::{
    DEFAULT_DOCUMENT_SPLIT_SIZE, DEFAULT_LOWER_CASE, DEFAULT_SPLIT_OVERLAP_SIZE, DlChunkingMode, DlInferenceOptions,
    preprocess_text, whitespace_chunks, window_ranges,
};
pub use scoring::{AverageClassificationScoringStrategy, ClassificationScoringStrategy, softmax};

/// Errors returned when constructing or running [`DocumentCategorizerDL`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DlDoccatError {
    /// Invalid or missing `config.json` / `id2label` category map.
    InvalidConfig(String),
    /// Invalid [`DlInferenceOptions`] (e.g., `document_split_size == 0` or
    /// `split_overlap_size >= document_split_size`).
    InvalidOptions(String),
    /// The input document array is empty or contains no non-whitespace tokens.
    EmptyDocument(String),
    /// Model logits are empty, non-finite (`NaN`/`Infinity`), or do not match
    /// the configured category count.
    InvalidLogits(String),
    /// Per-chunk or final classification scores are empty, ragged, non-finite,
    /// or do not match the configured category count.
    InvalidScores(String),
    /// The underlying sequence-classification backend failed.
    Backend(String),
}

impl fmt::Display for DlDoccatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DlDoccatError::InvalidConfig(msg) => write!(f, "invalid DocumentCategorizerDL config: {msg}"),
            DlDoccatError::InvalidOptions(msg) => write!(f, "invalid DocumentCategorizerDL options: {msg}"),
            DlDoccatError::EmptyDocument(msg) => write!(f, "empty document for DocumentCategorizerDL: {msg}"),
            DlDoccatError::InvalidLogits(msg) => write!(f, "invalid DocumentCategorizerDL logits: {msg}"),
            DlDoccatError::InvalidScores(msg) => write!(f, "invalid DocumentCategorizerDL scores: {msg}"),
            DlDoccatError::Backend(msg) => write!(f, "DocumentCategorizerDL backend error: {msg}"),
        }
    }
}

impl std::error::Error for DlDoccatError {}

/// A sequence-classification transformer model that scores text chunks or
/// pre-tokenized `opennlp:subwords` windows into raw per-category `f32` logits.
pub trait DlSequenceScorer: Send + Sync {
    /// Ordered category labels (`0..num_categories`).
    fn categories(&self) -> &[String];

    /// Provenance model reference recorded on written `Categories` layers.
    fn model_ref(&self) -> ModelRef;

    /// Execution backend, device, and precision recorded on written layers.
    fn execution(&self) -> Option<Execution> {
        None
    }

    /// Maximum number of content subword pieces per window (excluding the
    /// template's special tokens such as `[CLS]` and `[SEP]`).
    fn max_content_tokens(&self) -> usize {
        510
    }

    /// Scorer-specific configuration fingerprint folded into
    /// [`DocumentCategorizerDL`]'s provenance `config_hash`.
    fn scorer_config_hash(&self) -> u64 {
        0
    }

    /// Scores each text chunk in `chunks` using `options` and returns one row
    /// of raw `f32` logits (`[categories().len()]`) per chunk.
    fn score_chunks(&self, chunks: &[String], options: &DlInferenceOptions) -> Result<Vec<Vec<f32>>, DlDoccatError>;

    /// Scores pre-tokenized subword piece windows from `opennlp:subwords` and
    /// returns one row of raw `f32` logits (`[categories().len()]`) per window.
    fn score_subword_windows(&self, windows: &[&[i32]]) -> Result<Vec<Vec<f32>>, DlDoccatError>;
}

/// Transformer document categorizer ported from
/// `opennlp.dl.doccat.DocumentCategorizerDL`.
pub struct DocumentCategorizerDL<
    S: DlSequenceScorer,
    C: ClassificationScoringStrategy = AverageClassificationScoringStrategy,
> {
    scorer: S,
    categories: Vec<String>,
    label_to_index: BTreeMap<String, usize>,
    tag_set: Arc<TagSet>,
    options: DlInferenceOptions,
    strategy: C,
}

impl<S: DlSequenceScorer> DocumentCategorizerDL<S, AverageClassificationScoringStrategy> {
    /// Creates a `DocumentCategorizerDL` with default [`DlInferenceOptions`]
    /// (`lower_case = true`, `document_split_size = 250`,
    /// `split_overlap_size = 50`) and [`AverageClassificationScoringStrategy`].
    pub fn new(scorer: S) -> Result<Self, DlDoccatError> {
        Self::with_options(scorer, DlInferenceOptions::default())
    }

    /// Creates a `DocumentCategorizerDL` with `options` and
    /// [`AverageClassificationScoringStrategy`].
    pub fn with_options(scorer: S, options: DlInferenceOptions) -> Result<Self, DlDoccatError> {
        Self::with_options_and_strategy(scorer, options, AverageClassificationScoringStrategy)
    }
}

impl<S: DlSequenceScorer, C: ClassificationScoringStrategy> DocumentCategorizerDL<S, C> {
    /// Creates a `DocumentCategorizerDL` with custom [`DlInferenceOptions`] and
    /// [`ClassificationScoringStrategy`].
    pub fn with_options_and_strategy(
        scorer: S,
        options: DlInferenceOptions,
        strategy: C,
    ) -> Result<Self, DlDoccatError> {
        options.validate()?;
        let raw_labels = scorer.categories();
        if raw_labels.is_empty() {
            return Err(DlDoccatError::InvalidConfig("No categories found in sequence classifier".into()));
        }
        if raw_labels.len() > u16::MAX as usize {
            return Err(DlDoccatError::InvalidConfig(format!(
                "{} categories exceed u16 tag capacity",
                raw_labels.len()
            )));
        }
        let mut categories = Vec::with_capacity(raw_labels.len());
        let mut label_to_index = BTreeMap::new();
        for (i, label) in raw_labels.iter().enumerate() {
            if label.trim().is_empty() {
                return Err(DlDoccatError::InvalidConfig(format!("Category {i} is blank")));
            }
            if label_to_index.insert(label.clone(), i).is_some() {
                return Err(DlDoccatError::InvalidConfig(format!("Duplicate category label {label:?}")));
            }
            categories.push(label.clone());
        }
        let label_refs: Vec<&str> = categories.iter().map(String::as_str).collect();
        let tag_set = TagSet::new("category", &label_refs);
        Ok(Self { scorer, categories, label_to_index, tag_set, options, strategy })
    }

    /// Wraps this categorizer into a [`DlSpanCategorizer`] for span layer `L`
    /// (`Chunks`, `Structure`, or `Sentences`).
    pub fn into_span_categorizer<L: SpanCategorizable>(self) -> DlSpanCategorizer<S, L, C> {
        DlSpanCategorizer::from_categorizer(self)
    }

    /// Returns a reference to the underlying [`DlSequenceScorer`].
    pub fn scorer(&self) -> &S {
        &self.scorer
    }

    /// Returns the configured [`DlInferenceOptions`].
    pub fn options(&self) -> &DlInferenceOptions {
        &self.options
    }

    /// Returns the configured [`ClassificationScoringStrategy`].
    pub fn scoring_strategy(&self) -> &C {
        &self.strategy
    }

    /// Returns the category [`TagSet`].
    pub fn categories(&self) -> &Arc<TagSet> {
        &self.tag_set
    }

    /// Returns the number of categories (`DocumentCategorizerDL.getNumberOfCategories`).
    pub fn number_of_categories(&self) -> usize {
        self.categories.len()
    }

    /// Returns the category name at `index` (`DocumentCategorizerDL.getCategory`).
    pub fn get_category(&self, index: usize) -> Option<&str> {
        self.categories.get(index).map(String::as_str)
    }

    /// Returns the index of `category` (`DocumentCategorizerDL.getIndex`).
    pub fn get_index(&self, category: &str) -> Option<usize> {
        self.label_to_index.get(category).copied()
    }

    /// Categorizes `text` (`text[0]`) using OpenNLP's `DocumentCategorizerDL`
    /// pipeline: splits `text[0]` into overlapping whitespace chunks, runs the
    /// sequence-classification model on each chunk, applies `softmax` in `f64`
    /// per chunk, and aggregates chunk probabilities with
    /// [`ClassificationScoringStrategy`].
    pub fn categorize(&self, text: &[&str]) -> Result<Vec<f64>, DlDoccatError> {
        if text.is_empty() {
            return Err(DlDoccatError::EmptyDocument("Text array must not be null or empty".into()));
        }
        let chunks = whitespace_chunks(text[0], self.options.document_split_size, self.options.split_overlap_size)?;
        if chunks.is_empty() {
            return Err(DlDoccatError::EmptyDocument(
                "The document to categorize must contain at least one non-whitespace token".into(),
            ));
        }
        let logit_rows = self.scorer.score_chunks(&chunks, &self.options)?;
        if logit_rows.len() != chunks.len() {
            return Err(DlDoccatError::InvalidLogits(format!(
                "Scorer returned {} logit rows for {} chunks",
                logit_rows.len(),
                chunks.len()
            )));
        }
        self.combine_logit_rows(&logit_rows)
    }

    /// Convenience helper for `self.categorize(&[text])`.
    pub fn categorize_text(&self, text: &str) -> Result<Vec<f64>, DlDoccatError> {
        self.categorize(&[text])
    }

    /// Categorizes a document directly from its pre-tokenized `opennlp:subwords`
    /// (`Subwords`) layer.
    pub fn categorize_subwords(&self, subwords: &Subwords) -> Result<Vec<f64>, DlDoccatError> {
        let expected = self.scorer.model_ref();
        if subwords.model != expected {
            return Err(DlDoccatError::InvalidConfig(format!(
                "Subwords model {} (revision {}) does not match scorer model {} (revision {})",
                subwords.model.id, subwords.model.revision, expected.id, expected.revision
            )));
        }
        self.categorize_subword_slice(&subwords.ids)
    }

    /// Categorizes a slice of pre-tokenized subword piece IDs by windowing over
    /// `ids` with `min(document_split_size, max_content_tokens)` and
    /// `split_overlap_size`, running each window through the sequence
    /// classifier, applying `softmax` in `f64`, and combining window scores.
    pub fn categorize_subword_slice(&self, ids: &[i32]) -> Result<Vec<f64>, DlDoccatError> {
        if ids.is_empty() {
            return Err(DlDoccatError::EmptyDocument(
                "The document to categorize must contain at least one subword token".into(),
            ));
        }
        let max_content = self.scorer.max_content_tokens().max(1);
        let win_size = self.options.document_split_size.min(max_content).max(1);
        let overlap = self.options.split_overlap_size.min(win_size.saturating_sub(1));
        let ranges = window_ranges(ids.len(), win_size, overlap)?;
        let slices: Vec<&[i32]> = ranges.iter().map(|r| &ids[r.clone()]).collect();
        let logit_rows = self.scorer.score_subword_windows(&slices)?;
        if logit_rows.len() != slices.len() {
            return Err(DlDoccatError::InvalidLogits(format!(
                "Scorer returned {} logit rows for {} subword windows",
                logit_rows.len(),
                slices.len()
            )));
        }
        self.combine_logit_rows(&logit_rows)
    }

    fn combine_logit_rows(&self, logit_rows: &[Vec<f32>]) -> Result<Vec<f64>, DlDoccatError> {
        let num_categories = self.categories.len();
        let mut scores = Vec::with_capacity(logit_rows.len());
        for logits in logit_rows {
            if logits.len() != num_categories {
                return Err(DlDoccatError::InvalidLogits(format!(
                    "Model output size ({}) does not match configured category count ({num_categories})",
                    logits.len()
                )));
            }
            scores.push(softmax(logits)?);
        }
        let distribution = self.strategy.score(&scores)?;
        self.validate_outcome_vector(&distribution)?;
        Ok(distribution)
    }

    fn validate_outcome_vector(&self, outcome: &[f64]) -> Result<(), DlDoccatError> {
        let num_categories = self.categories.len();
        if outcome.is_empty() {
            return Err(DlDoccatError::InvalidScores("Outcome array must not be null or empty".into()));
        }
        if outcome.len() != num_categories {
            return Err(DlDoccatError::InvalidScores(format!(
                "Score vector size ({}) does not match configured category count ({num_categories})",
                outcome.len()
            )));
        }
        for &value in outcome {
            if !value.is_finite() {
                return Err(DlDoccatError::InvalidScores(format!("Outcome score is not finite: {value}")));
            }
        }
        Ok(())
    }

    /// Returns the winning category index from `outcome`, breaking ties toward
    /// the lowest index (`DocumentCategorizerDL.maxIndex`).
    pub fn best_index(&self, outcome: &[f64]) -> Result<usize, DlDoccatError> {
        self.validate_outcome_vector(outcome)?;
        let mut best = 0usize;
        for (i, &val) in outcome.iter().enumerate().skip(1) {
            if val > outcome[best] {
                best = i;
            }
        }
        Ok(best)
    }

    /// Returns the winning category name from `outcome`
    /// (`DocumentCategorizerDL.getBestCategory`).
    pub fn best_category(&self, outcome: &[f64]) -> Result<&str, DlDoccatError> {
        let idx = self.best_index(outcome)?;
        self.get_category(idx)
            .ok_or_else(|| DlDoccatError::InvalidConfig(format!("Category mapping does not contain index: {idx}")))
    }

    /// Formats all category outcomes and their probabilities as
    /// `cat0[0.1234]  cat1[0.8766]`.
    pub fn all_results(&self, results: &[f64]) -> String {
        if results.len() != self.number_of_categories() || results.is_empty() {
            return String::new();
        }
        let mut sb = String::new();
        for (i, &p) in results.iter().enumerate() {
            if i > 0 {
                sb.push_str("  ");
            }
            let name = self.get_category(i).unwrap_or("");
            let _ = write!(sb, "{name}[{p:.4}]");
        }
        sb
    }

    /// Categorizes `text` and returns a map from category name to score
    /// (`DocumentCategorizerDL.scoreMap`).
    pub fn score_map(&self, text: &[&str]) -> Result<BTreeMap<String, f64>, DlDoccatError> {
        let scores = self.categorize(text)?;
        let mut map = BTreeMap::new();
        for (i, &score) in scores.iter().enumerate() {
            if let Some(cat) = self.get_category(i) {
                map.insert(cat.to_owned(), score);
            }
        }
        Ok(map)
    }

    /// Categorizes `text` and returns scores sorted in ascending order, each
    /// paired with the set of categories sharing that exact score
    /// (`DocumentCategorizerDL.sortedScoreMap`).
    pub fn sorted_score_map(&self, text: &[&str]) -> Result<Vec<(f64, BTreeSet<String>)>, DlDoccatError> {
        let scores = self.categorize(text)?;
        let mut by_bits: BTreeMap<u64, (f64, BTreeSet<String>)> = BTreeMap::new();
        for (i, &score) in scores.iter().enumerate() {
            if let Some(cat) = self.get_category(i) {
                let normalized = if score == 0.0 { 0.0 } else { score };
                by_bits
                    .entry(normalized.to_bits())
                    .or_insert_with(|| (normalized, BTreeSet::new()))
                    .1
                    .insert(cat.to_owned());
            }
        }
        let mut out: Vec<(f64, BTreeSet<String>)> = by_bits.into_values().collect();
        out.sort_by(|a, b| a.0.total_cmp(&b.0));
        Ok(out)
    }

    pub(crate) fn config_hash_value(&self) -> u64 {
        fn feed(hash: &mut u64, bytes: &[u8]) {
            for &b in bytes {
                *hash = (*hash ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
            }
        }
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        feed(&mut hash, b"opennlp:doccat-dl");
        feed(
            &mut hash,
            &[
                u8::from(self.options.lower_case),
                u8::from(self.options.normalize_whitespace),
                u8::from(self.options.normalize_dashes),
                u8::from(self.options.include_token_type_ids),
                self.options.chunking_mode as u8,
            ],
        );
        feed(&mut hash, &(self.options.document_split_size as u64).to_le_bytes());
        feed(&mut hash, &(self.options.split_overlap_size as u64).to_le_bytes());
        feed(&mut hash, &(self.scorer.max_content_tokens() as u64).to_le_bytes());
        feed(&mut hash, &self.scorer.scorer_config_hash().to_le_bytes());
        feed(&mut hash, self.strategy.strategy_id().as_bytes());
        feed(&mut hash, &[0]);
        feed(&mut hash, &self.strategy.strategy_hash().to_le_bytes());
        for cat in &self.categories {
            feed(&mut hash, cat.as_bytes());
            feed(&mut hash, &[0]);
        }
        hash
    }
}
