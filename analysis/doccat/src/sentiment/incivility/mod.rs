//! Multi-label incivility and toxicity annotator (`#323`).
//!
//! Scores sentences, retrieval chunks, or structure blocks across the six
//! multi-label dimensions of [`Incivility`] (`turbo:incivility` from `#319`),
//! following the Oasis (ICDMW 2019) and HKBU (ICA 2026) label set:
//! - [`IncivilityLabel::Insult`] (`"insult"`)
//! - [`IncivilityLabel::Profanity`] (`"profanity"`)
//! - [`IncivilityLabel::Threat`] (`"threat"`)
//! - [`IncivilityLabel::IdentityAttack`] (`"identity_attack"`)
//! - [`IncivilityLabel::Intolerance`] (`"intolerance"`)
//! - [`IncivilityLabel::SevereToxicity`] (`"severe_toxicity"`)
//!
//! [`IncivilityAnnotator`] (`ID = "turbo-doccat:incivility"`) combines:
//! 1. **Lexicon and phrase matching (`IncivilityLexicon`)**: deterministic,
//!    zero-regex token and multi-token phrase matching over `opennlp:tokens`
//!    that assigns per-label weights and records the exact [`IxRange<Tokens>`]
//!    evidence span for every matched expression.
//! 2. **Multi-label embedding head (`IncivilityVectorScorer`)**: scores
//!    `Vectors<Sentences>` or `Vectors<Chunks>` with a multi-label sigmoid
//!    classifier head or prototype centroid scorer from `#321`. When the
//!    embedding head predicts a positive label (`score >= threshold`) that had
//!    no direct lexicon match in that unit, the annotator attributes evidence
//!    to the unit's highest-salience content token span so every positive
//!    label carries at least one non-empty [`IxRange<Tokens>`] evidence span.
//! 3. **Optional reply-context conflict scoring (`with_reply_context`)**:
//!    following Opinion Conflicts (Zhang et al., 2018, arXiv:1809.00289),
//!    when `use_reply_context` is enabled and `turbo:structure` ([`Structure`])
//!    is present with a parent block or a preceding [`Block::Quote`] block,
//!    scores antagonistic second-person reply patterns against the quoted or
//!    parent block. [`ModelRef::id`] ends in `:reply-context` or
//!    `:no-reply-context` after the setting; it does not say whether a
//!    [`Structure`] layer was present on the document.

use std::collections::BTreeMap;
use std::sync::Arc;

use turbo_analysis::{Annotator, Error, Ix, IxRange, Layer, ModelRef, Refs, View};
use turbo_analysis_layers::{
    Block, Chunks, Incivility, IncivilityItem, IncivilityLabel, IncivilityScores, IncivilityTarget, Sentences,
    Structure, Tokens, Vectors,
};
use turbo_maxent::sha256;

/// Default score threshold (`0.5`) at or above which an [`IncivilityLabel`] is
/// considered positive and must carry at least one [`IxRange<Tokens>`]
/// evidence span.
pub const DEFAULT_INCIVILITY_THRESHOLD: f64 = 0.5;

/// One `(label, token_range)` evidence span attributing an [`IncivilityLabel`]
/// to a contiguous token slice.
pub type IncivilityEvidenceSpan = (IncivilityLabel, IxRange<Tokens>);

/// A multi-label embedding head that scores a sentence or chunk vector
/// (`&[f32]`) into [`IncivilityScores`] (`0.0..=1.0` per label). Neither
/// `ClassifierHead` nor `LabelCentroids` in `turbo-analysis-embed` implements
/// it; a wrapper calls their `score_incivility_vector` (see the adapters in
/// `analysis/embed/tests/heads.rs`). When [`IncivilityVectorScorer::model`]
/// names an encoder, the annotator refuses vectors from any other encoder.
pub trait IncivilityVectorScorer: Send + Sync {
    /// Scores a single embedding vector `vec` (`&[f32]`) and returns its six
    /// [`IncivilityScores`] values in `0.0..=1.0`.
    fn score_incivility(&self, vec: &[f32]) -> Result<IncivilityScores, String>;

    /// Scores `vec` with `context`, the vector of the unit it follows or
    /// of the post it answers, when the annotator gives one. A scorer
    /// that reads no context ignores it.
    fn score_incivility_in_context(&self, vec: &[f32], context: Option<&[f32]>) -> Result<IncivilityScores, String> {
        let _ = context;
        self.score_incivility(vec)
    }

    /// Optional [`ModelRef`] describing the underlying embedding head.
    fn model(&self) -> Option<ModelRef> {
        None
    }

    /// Deterministic configuration hash for provenance tracking.
    fn config_hash(&self) -> u64 {
        0
    }
}

/// Which unit span layer [`IncivilityAnnotator`] scores.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum IncivilityUnit {
    /// Score each sentence in `opennlp:sentences` (`IncivilityTarget::Sentence`).
    #[default]
    Sentences = 0,
    /// Score each retrieval chunk in `turbo:chunks` (`IncivilityTarget::Chunk`).
    Chunks = 1,
    /// Score each structure block in `turbo:structure` (`IncivilityTarget::Block`).
    Blocks = 2,
}

/// Which signals [`IncivilityAnnotator`] scores with. The stage is named
/// in its model reference and its configuration hash.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum IncivilityStage {
    /// The lexicon (and reply-context rules) alone; a vector scorer, if
    /// set, is not read.
    Lexicon = 0,
    /// The vector scorer alone; it must be set, and every unit needs a
    /// vector.
    Learned = 1,
    /// Both, the head's scores fused with the lexicon's.
    #[default]
    Fused = 2,
}

/// The context vector the vector scorer reads beside each unit's own.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum IncivilityVectorContext {
    /// None.
    #[default]
    None,
    /// The vector of the unit before it (none for the first unit, or after
    /// a unit with no vector).
    Preceding,
    /// One vector for every unit: the post the document replies to.
    Parent(Arc<[f32]>),
}

/// Deterministic phrase lexicon for [`IncivilityAnnotator`], with a canonical
/// SHA-256 content digest for [`ModelRef::sha256`].
#[derive(Clone, Debug)]
pub struct IncivilityLexicon {
    /// The `(tokens, label, weight)` entries in canonical `(tokens, label)`
    /// order; the digest reads this order, so it must never change.
    entries: Vec<(Vec<String>, IncivilityLabel, f64)>,
    /// First token of at least one entry to the half-open run of its entries
    /// in `entries`, sorted by token, so a scan visits only the patterns that
    /// can start at a unit's token.
    first_token: Vec<(String, u32, u32)>,
    sha256: String,
}

impl Default for IncivilityLexicon {
    fn default() -> Self {
        Self::bundled()
    }
}

mod annotate;
mod head;
mod metrics;
mod rules;
mod scoring;
mod tables;
#[cfg(test)]
mod tests;
mod vocabulary;
pub use head::{IncivilityFitOptions, IncivilityHeadError, IncivilityVectorHead};
pub use metrics::{GoldIncivilityUnit, IncivilityMetrics, LabelMetrics, evaluate_incivility_corpus};
use rules::*;
use tables::*;

/// Multi-label incivility and toxicity annotator (`turbo-doccat:incivility`).
///
/// Reads `(View, Tokens, Option<Sentences>, Option<Chunks>, Option<Structure>, Option<Vectors<Sentences>>, Option<Vectors<Chunks>>)`
/// and writes [`Incivility`] (`turbo:incivility`).
#[derive(Clone)]
pub struct IncivilityAnnotator {
    /// Phrase lexicon providing per-label scores and exact token-range
    /// evidence spans.
    pub lexicon: Arc<IncivilityLexicon>,
    /// Which unit layer to score (`Sentences`, `Chunks`, or `Blocks`).
    pub unit: IncivilityUnit,
    /// Optional multi-label embedding head over `Vectors<Sentences>` or
    /// `Vectors<Chunks>`.
    pub vector_scorer: Option<Arc<dyn IncivilityVectorScorer>>,
    /// Score threshold (`0.5` by default) at or above which a label is
    /// considered positive and guaranteed to carry at least one evidence span.
    pub threshold: f64,
    /// Whether to score reply-context conflicts against parent or preceding
    /// quote blocks when `turbo:structure` is present on the document.
    pub use_reply_context: bool,
    /// The signals scored with: lexicon, learned head, or both.
    pub stage: IncivilityStage,
    /// The context vector the head reads beside each unit's own.
    pub vector_context: IncivilityVectorContext,
}

impl Default for IncivilityAnnotator {
    fn default() -> Self {
        Self::new()
    }
}

impl IncivilityAnnotator {
    /// Creates an [`IncivilityAnnotator`] scoring `Sentences` with the bundled
    /// lexicon, `threshold = 0.5`, and `use_reply_context = false`.
    pub fn new() -> Self {
        IncivilityAnnotator {
            lexicon: Arc::new(IncivilityLexicon::bundled()),
            unit: IncivilityUnit::Sentences,
            vector_scorer: None,
            threshold: DEFAULT_INCIVILITY_THRESHOLD,
            use_reply_context: false,
            stage: IncivilityStage::Fused,
            vector_context: IncivilityVectorContext::None,
        }
    }

    /// Sets the [`IncivilityLexicon`].
    pub fn with_lexicon(mut self, lexicon: Arc<IncivilityLexicon>) -> Self {
        self.lexicon = lexicon;
        self
    }

    /// Sets the target unit layer (`Sentences`, `Chunks`, or `Blocks`).
    pub fn with_unit(mut self, unit: IncivilityUnit) -> Self {
        self.unit = unit;
        self
    }

    /// Attaches a multi-label [`IncivilityVectorScorer`] (such as a
    /// `ClassifierHead` or `LabelCentroids` head from `turbo-analysis-embed`).
    pub fn with_vector_scorer(mut self, scorer: Arc<dyn IncivilityVectorScorer>) -> Self {
        self.vector_scorer = Some(scorer);
        self
    }

    /// Sets the positive label threshold (finite values clamped to `0.05..=0.99`).
    /// Non-finite thresholds cause scoring to return an error.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = if threshold.is_finite() { threshold.clamp(0.05, 0.99) } else { threshold };
        self
    }

    /// Enables or disables reply-context conflict scoring over `turbo:structure`
    /// (`use_reply_context = false` by default, recorded in provenance).
    pub fn with_reply_context(mut self, enabled: bool) -> Self {
        self.use_reply_context = enabled;
        self
    }

    /// Sets the stage: lexicon, learned head or both.
    pub fn with_stage(mut self, stage: IncivilityStage) -> Self {
        self.stage = stage;
        self
    }

    /// Sets the context vector the head reads.
    pub fn with_vector_context(mut self, context: IncivilityVectorContext) -> Self {
        self.vector_context = context;
        self
    }

    /// Reads `parent` as the vector of the post this document replies to,
    /// beside every unit's own.
    pub fn with_parent_vector(self, parent: &[f32]) -> Self {
        self.with_vector_context(IncivilityVectorContext::Parent(parent.into()))
    }
}
