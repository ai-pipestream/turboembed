//! Native document-level and sentence-level polarity classification (#1526).
//!
//! Combines [`LexiconSentiment`] (VADER) with a learned three-class
//! (`positive`, `negative`, `neutral`) [`PolarityVectorHead`] trained end to
//! end with `turbo-trainer` (`AdamW`, `linear`, `xent`, `Rng`):
//!
//! - [`PolarityAnnotator`] (`turbo-doccat:polarity`) scores document-level
//!   polarity into `turbo:categories` (bound to [`SENTIMENT`] via
//!   [`PolarityAnnotator::bound`]) and implements [`CalibratableCategorizer`](crate::calibrated::CalibratableCategorizer).
//! - [`SentencePolarityAnnotator`] (`turbo-doccat:sentence-polarity`) scores
//!   sentence-level polarity into `turbo:opinions` and
//!   `turbo:categories<turbo:sentences>` (bound to [`SENTENCE_SENTIMENT`]
//!   via [`SentencePolarityAnnotator::bound`]).

use std::fmt;
use std::sync::Arc;

use turbo_analysis::{Annotator, Bound, Key, ModelRef, TagSet};
use turbo_analysis_layers::Categories;
use turbo_sha256::sha256;

use crate::scorers::{PolarityStage, PolarityVectorScorer};
use crate::sentiment::lexicon::{LexiconSentiment, SENTENCE_SENTIMENT};

pub mod annotate;
pub mod fit;
pub mod head;
pub mod metrics;
#[cfg(test)]
mod tests;

pub use annotate::{blend_polarity_scores, vader_to_polarity_scores};
pub use fit::{PolarityExample, PolarityFitConfig, PolarityFitReport, parse_polarity_label, parse_polarity_tsv};
pub use head::{
    POLARITY_COUNT, POLARITY_NAMES, PolarityVectorHead, index_polarity, polarity_index, softmax_polarity,
    validate_polarity_scores,
};
pub use metrics::{PolarityClassMetrics, PolarityMetrics, evaluate_polarity, evaluate_predictions, majority_baseline};

/// Named instance of `Categories` for document-level sentiment polarity.
pub const SENTIMENT: Key<Categories> = Key::named("sentiment");

/// Default convex combination weight (`0.65`) given to the learned
/// [`PolarityVectorScorer`] in [`PolarityStage::Hybrid`].
pub const DEFAULT_HEAD_WEIGHT: f64 = 0.65;

/// Document-level three-class polarity annotator (`positive`, `negative`,
/// `neutral`).
#[derive(Clone)]
pub struct PolarityAnnotator {
    pub(crate) stage: PolarityStage,
    pub(crate) lexicon: LexiconSentiment,
    pub(crate) vector_scorer: Option<Arc<dyn PolarityVectorScorer>>,
    pub(crate) head_weight: f64,
    pub(crate) categories: Arc<TagSet>,
}

impl fmt::Debug for PolarityAnnotator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PolarityAnnotator")
            .field("stage", &self.stage)
            .field("head_weight", &self.head_weight)
            .field("has_vector_scorer", &self.vector_scorer.is_some())
            .finish()
    }
}

impl Default for PolarityAnnotator {
    fn default() -> Self {
        Self::new()
    }
}

impl PolarityAnnotator {
    /// Creates a [`PolarityAnnotator`] in [`PolarityStage::Lexicon`] mode
    /// backed by [`LexiconSentiment::new`].
    pub fn new() -> Self {
        Self {
            stage: PolarityStage::Lexicon,
            lexicon: LexiconSentiment::new(),
            vector_scorer: None,
            head_weight: DEFAULT_HEAD_WEIGHT,
            categories: TagSet::new("sentiment:polarity", &POLARITY_NAMES),
        }
    }

    /// Creates a [`PolarityAnnotator`] in [`PolarityStage::Learned`] mode
    /// backed by `scorer`.
    pub fn from_head(scorer: Arc<dyn PolarityVectorScorer>) -> Self {
        Self::new().with_vector_scorer(scorer).with_stage(PolarityStage::Learned)
    }

    /// Creates a [`PolarityAnnotator`] in [`PolarityStage::Hybrid`] mode
    /// combining `lexicon` and `scorer` with `head_weight` in `0.0..=1.0`.
    pub fn hybrid(lexicon: LexiconSentiment, scorer: Arc<dyn PolarityVectorScorer>, head_weight: f64) -> Self {
        Self::new()
            .with_lexicon(lexicon)
            .with_vector_scorer(scorer)
            .with_head_weight(head_weight)
            .with_stage(PolarityStage::Hybrid)
    }

    /// Sets the [`PolarityStage`].
    pub fn with_stage(mut self, stage: PolarityStage) -> Self {
        self.stage = stage;
        self
    }

    /// Sets the [`LexiconSentiment`] rule scorer.
    pub fn with_lexicon(mut self, lexicon: LexiconSentiment) -> Self {
        self.lexicon = lexicon;
        self
    }

    /// Attaches a learned [`PolarityVectorScorer`] head.
    pub fn with_vector_scorer(mut self, scorer: Arc<dyn PolarityVectorScorer>) -> Self {
        self.vector_scorer = Some(scorer);
        self
    }

    /// Sets the convex combination weight (`0.0..=1.0`) for [`PolarityStage::Hybrid`].
    pub fn with_head_weight(mut self, head_weight: f64) -> Self {
        self.head_weight = head_weight;
        self
    }

    /// Returns the configured [`PolarityStage`].
    pub fn stage(&self) -> PolarityStage {
        self.stage
    }

    /// Returns the `TagSet` (`["positive", "negative", "neutral"]`).
    pub fn categories_tagset(&self) -> &Arc<TagSet> {
        &self.categories
    }

    /// Binds output `0` (`Categories`) to the `"sentiment"` instance ([`SENTIMENT`]).
    pub fn bound(self) -> Bound<Self> {
        Bound::new(self).write(0, SENTIMENT.name)
    }

    pub(crate) fn effective_model_ref(&self) -> ModelRef {
        combined_model_ref(self.stage, &self.lexicon, self.vector_scorer.as_deref(), self.head_weight)
    }
}

/// Sentence-level three-class polarity annotator writing `Opinions` and
/// `Categories<Sentences>`.
#[derive(Clone)]
pub struct SentencePolarityAnnotator {
    pub(crate) stage: PolarityStage,
    pub(crate) lexicon: LexiconSentiment,
    pub(crate) vector_scorer: Option<Arc<dyn PolarityVectorScorer>>,
    pub(crate) head_weight: f64,
    pub(crate) categories: Arc<TagSet>,
}

impl fmt::Debug for SentencePolarityAnnotator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SentencePolarityAnnotator")
            .field("stage", &self.stage)
            .field("head_weight", &self.head_weight)
            .field("has_vector_scorer", &self.vector_scorer.is_some())
            .finish()
    }
}

impl Default for SentencePolarityAnnotator {
    fn default() -> Self {
        Self::new()
    }
}

impl SentencePolarityAnnotator {
    /// Creates a [`SentencePolarityAnnotator`] in [`PolarityStage::Lexicon`] mode.
    pub fn new() -> Self {
        Self {
            stage: PolarityStage::Lexicon,
            lexicon: LexiconSentiment::new(),
            vector_scorer: None,
            head_weight: DEFAULT_HEAD_WEIGHT,
            categories: TagSet::new("sentiment:polarity", &POLARITY_NAMES),
        }
    }

    /// Creates a [`SentencePolarityAnnotator`] in [`PolarityStage::Learned`] mode.
    pub fn from_head(scorer: Arc<dyn PolarityVectorScorer>) -> Self {
        Self::new().with_vector_scorer(scorer).with_stage(PolarityStage::Learned)
    }

    /// Creates a [`SentencePolarityAnnotator`] in [`PolarityStage::Hybrid`] mode.
    pub fn hybrid(lexicon: LexiconSentiment, scorer: Arc<dyn PolarityVectorScorer>, head_weight: f64) -> Self {
        Self::new()
            .with_lexicon(lexicon)
            .with_vector_scorer(scorer)
            .with_head_weight(head_weight)
            .with_stage(PolarityStage::Hybrid)
    }

    /// Sets the [`PolarityStage`].
    pub fn with_stage(mut self, stage: PolarityStage) -> Self {
        self.stage = stage;
        self
    }

    /// Sets the [`LexiconSentiment`] rule scorer.
    pub fn with_lexicon(mut self, lexicon: LexiconSentiment) -> Self {
        self.lexicon = lexicon;
        self
    }

    /// Attaches a learned [`PolarityVectorScorer`] head.
    pub fn with_vector_scorer(mut self, scorer: Arc<dyn PolarityVectorScorer>) -> Self {
        self.vector_scorer = Some(scorer);
        self
    }

    /// Sets the convex combination weight (`0.0..=1.0`) for [`PolarityStage::Hybrid`].
    pub fn with_head_weight(mut self, head_weight: f64) -> Self {
        self.head_weight = head_weight;
        self
    }

    /// Returns the configured [`PolarityStage`].
    pub fn stage(&self) -> PolarityStage {
        self.stage
    }

    /// Binds output `1` (`Categories<Sentences>`) to `"sentiment"` ([`SENTENCE_SENTIMENT`]).
    pub fn bound(self) -> Bound<Self> {
        Bound::new(self).write(1, SENTENCE_SENTIMENT.name)
    }

    pub(crate) fn effective_model_ref(&self) -> ModelRef {
        combined_model_ref(self.stage, &self.lexicon, self.vector_scorer.as_deref(), self.head_weight)
    }
}

fn combined_model_ref(
    stage: PolarityStage,
    lexicon: &LexiconSentiment,
    scorer: Option<&dyn PolarityVectorScorer>,
    head_weight: f64,
) -> ModelRef {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(stage.model_id().as_bytes());
    bytes.push(0);
    if matches!(stage, PolarityStage::Lexicon | PolarityStage::Hybrid)
        && let Some(lex_ref) = Annotator::model(lexicon)
    {
        bytes.extend_from_slice(lex_ref.id.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(lex_ref.sha256.as_bytes());
        bytes.push(0);
    }
    if matches!(stage, PolarityStage::Learned | PolarityStage::Hybrid) {
        if let Some(s) = scorer {
            bytes.extend_from_slice(&s.config_hash().to_le_bytes());
            if let Some(m) = s.model() {
                bytes.extend_from_slice(m.id.as_bytes());
                bytes.push(0);
                bytes.extend_from_slice(m.sha256.as_bytes());
                bytes.push(0);
            }
        }
        if stage == PolarityStage::Hybrid {
            bytes.extend_from_slice(&head_weight.to_bits().to_le_bytes());
        }
    }
    let digest = sha256(&bytes).to_hex();
    ModelRef { id: stage.model_id().into(), revision: "1".into(), sha256: digest }
}
