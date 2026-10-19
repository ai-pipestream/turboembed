//! Sentence-level sarcasm and irony detection (`turbo:pragmatics`, #324).
//!
//! Scores each sentence in `0.0..=1.0` and writes [`Pragmatics`]
//! (`turbo:pragmatics`), aligned to [`Sentences`] and never folded into
//! polarity (`turbo:categories`), so downstream consumers decide whether to
//! flip or discount a positive polarity score.
//!
//! Three signals are combined in priority order:
//!
//! 1. **Incongruity between a positive sentiment expression and a negative
//!    situation** in the same sentence, following Riloff et al. (EMNLP 2013,
//!    *Sarcasm as Contrast between a Positive Sentiment and Negative
//!    Situation*). Positive spans come from [`Opinions`] (`turbo:opinions`)
//!    or sarcastic positive framing cues; negative spans come from negative
//!    situation phrases (`waiting in line`, `stuck in traffic`, `server
//!    crashed`, `flight cancelled`, etc.) and negative [`Opinions`]. Clauses
//!    separated by an adversative conjunction (`but`, `however`, `although`,
//!    `though`, `yet`, `whereas`, `while`, `except`) without a sarcastic
//!    framing cue are treated as literal mixed sentiment rather than sarcasm.
//! 2. **Text against emoji polarity mismatch**, contrasting a positive text
//!    opinion against a negative emoji from [`Emoji`] (`turbo:emoji`, #183,
//!    scored via [`turbo_normalize::emoji::opennlp_annotation`]), or a
//!    negative text opinion/situation against a positive emoji.
//! 3. **Sentence embedding head** ([`PragmaticsVectorScorer`] over
//!    `Vectors<Sentences>`, or `sarcasm`/`irony` probabilities in
//!    `Categories<Sentences>`) for implicit sarcasm and irony with no
//!    explicit surface contrast.
//!
//! Every positive prediction carries its contrasted [`Incongruity`] token
//! spans (`Some(Incongruity { positive, negative })`) when explicit contrast
//! was found, or `None` when scored by the embedding head alone.

use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use turbo_analysis::{Annotator, Error, Ix, IxRange, Layer, ModelRef, Refs, Span, View};
use turbo_analysis_layers::{
    Categories, Incongruity, Opinions, Pragmatics, PragmaticsItem, Sentences, Tokens, Vectors,
};
use turbo_normalize::emoji::opennlp_annotation;
use turbo_normalize_layers::Emoji;

use super::lexicon::VaderLexicon;

/// Default minimum effective polarity for a positive opinion span.
pub const DEFAULT_MIN_POSITIVE_POLARITY: f64 = 0.4;

/// Default minimum magnitude of effective polarity for a negative opinion span.
pub const DEFAULT_MIN_NEGATIVE_POLARITY: f64 = 0.4;

/// Default classification threshold (`0.5`) above which a sentence is
/// considered sarcastic or ironic.
pub const DEFAULT_PRAGMATICS_THRESHOLD: f64 = 0.5;

/// Sentence-vector scorer used by [`PragmaticsAnnotator`] to score
/// `Vectors<Sentences>` rows in `0.0..=1.0` when a sentence has no explicit
/// surface contrast (or to reinforce explicit contrast).
pub trait PragmaticsVectorScorer: Send + Sync + fmt::Debug {
    /// Scores one sentence embedding row `vec`, returning a sarcasm/irony
    /// probability in `0.0..=1.0`.
    fn score_sentence_vector(&self, vec: &[f32]) -> Result<f64, String>;

    /// Deterministic configuration and weight digest for provenance.
    fn config_hash(&self) -> u64;

    /// Optional model reference recorded in provenance.
    fn model_ref(&self) -> Option<ModelRef> {
        None
    }
}

mod candidates;
mod contrast;
mod scoring;
mod tables;
#[cfg(test)]
mod tests;
mod tokens;
use tables::*;
use tokens::*;

#[derive(Copy, Clone, Debug)]
struct SpanCandidate {
    tokens: IxRange<Tokens>,
    strength: f64,
    cued: bool,
}

/// Sentence-level sarcasm and irony annotator (`turbo-doccat:pragmatics`).
///
/// Reads [`View`], [`Sentences`], [`Tokens`], and optional [`Opinions`],
/// [`Categories<Sentences>`], [`Emoji`], and [`Vectors<Sentences>`], and
/// writes [`Pragmatics`] (`turbo:pragmatics`).
#[derive(Clone, Debug)]
pub struct PragmaticsAnnotator {
    lexicon: Arc<VaderLexicon>,
    min_positive_polarity: f64,
    min_negative_polarity: f64,
    positive_threshold: f64,
    vector_scorer: Option<Arc<dyn PragmaticsVectorScorer>>,
}

impl Default for PragmaticsAnnotator {
    fn default() -> Self {
        Self::new()
    }
}

impl PragmaticsAnnotator {
    /// Creates a `PragmaticsAnnotator` with the bundled MIT VADER lexicon and
    /// default thresholds.
    pub fn new() -> Self {
        Self::with_lexicon(VaderLexicon::bundled())
    }

    /// Creates a `PragmaticsAnnotator` with a custom [`VaderLexicon`].
    pub fn with_lexicon(lexicon: Arc<VaderLexicon>) -> Self {
        PragmaticsAnnotator {
            lexicon,
            min_positive_polarity: DEFAULT_MIN_POSITIVE_POLARITY,
            min_negative_polarity: DEFAULT_MIN_NEGATIVE_POLARITY,
            positive_threshold: DEFAULT_PRAGMATICS_THRESHOLD,
            vector_scorer: None,
        }
    }

    /// Attaches a sentence-embedding scorer (`Vectors<Sentences>` -> `[0.0, 1.0]`)
    /// used when a sentence has no explicit contrast pair (or to reinforce
    /// explicit contrast).
    pub fn with_vector_scorer(mut self, scorer: Arc<dyn PragmaticsVectorScorer>) -> Self {
        self.vector_scorer = Some(scorer);
        self
    }

    /// Sets the minimum effective polarity magnitudes required for positive
    /// and negative opinion spans.
    pub fn with_polarity_thresholds(mut self, min_positive: f64, min_negative: f64) -> Self {
        self.min_positive_polarity = if min_positive.is_finite() { min_positive.max(0.0) } else { min_positive };
        self.min_negative_polarity = if min_negative.is_finite() { min_negative.max(0.0) } else { min_negative };
        self
    }

    /// Sets the classification threshold (`0.0..=1.0`) above which a sentence
    /// is treated as positive for sarcasm/irony.
    pub fn with_positive_threshold(mut self, threshold: f64) -> Self {
        self.positive_threshold = if threshold.is_finite() { threshold.clamp(0.0, 1.0) } else { threshold };
        self
    }

    /// The configured classification threshold (`0.0..=1.0`).
    pub fn positive_threshold(&self) -> f64 {
        self.positive_threshold
    }
}

impl Annotator for PragmaticsAnnotator {
    const ID: &'static str = "turbo-doccat:pragmatics";
    const VERSION: &'static str = "1";
    type Reads = (
        View,
        Sentences,
        Tokens,
        Option<Opinions>,
        Option<Categories<Sentences>>,
        Option<Emoji>,
        Option<Vectors<Sentences>>,
    );
    type Writes = (Pragmatics,);

    fn annotate(
        &self,
        (view, sentences, tokens, opinions, categories, emoji, vectors): Refs<'_, Self>,
    ) -> Result<(Pragmatics,), Error> {
        let pragmatics = self.score_document(view, sentences, tokens, opinions, categories, emoji, vectors)?;
        Ok((pragmatics,))
    }

    fn model(&self) -> Option<ModelRef> {
        if let Some(scorer) = &self.vector_scorer
            && let Some(m) = scorer.model_ref()
        {
            return Some(m);
        }
        Some(ModelRef {
            id: "vader-lexicon+riloff-pragmatics".into(),
            revision: "vader-1.0".into(),
            sha256: self.lexicon.sha256_hex().into(),
        })
    }

    fn config_hash(&self) -> u64 {
        let mut h = crate::hash::FoldHasher::default();
        self.lexicon.sha256_hex().hash(&mut h);
        self.min_positive_polarity.to_bits().hash(&mut h);
        self.min_negative_polarity.to_bits().hash(&mut h);
        self.positive_threshold.to_bits().hash(&mut h);
        if let Some(s) = &self.vector_scorer {
            1u8.hash(&mut h);
            s.config_hash().hash(&mut h);
        } else {
            0u8.hash(&mut h);
        }
        h.finish()
    }
}
