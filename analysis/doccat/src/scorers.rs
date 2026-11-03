//! Vector-head and category scorer traits and layer re-exports for sentiment,
//! polarity, emotion, incivility, aspect, and pragmatics annotators.
//!
//! This module imports only `turbo-analysis` and `turbo-analysis-layers`, so
//! encoder-backed heads in `turbo-analysis-embed` and rule/lexicon annotators
//! in `turbo-doccat` share these traits directly.

use std::fmt;
use std::sync::Arc;

use turbo_analysis::{IxRange, ModelRef, Tag, TagSet};
pub use turbo_analysis_layers::{
    Aspect, AspectTarget, Aspects, EMOTION_DIMENSIONS, EmotionDimension, EmotionLayer, EmotionScores, Incivility,
    IncivilityItem, IncivilityLabel, IncivilityScores, IncivilityTarget, Incongruity, Opinion, Opinions, Polarity,
    PolarityScores, Pragmatics, PragmaticsItem, Tokens,
};

/// Sentence-vector scorer used by [`crate::sentiment::pragmatics::PragmaticsAnnotator`]
/// to score `Vectors<Sentences>` rows in `0.0..=1.0` when a sentence has no
/// explicit surface contrast (or to reinforce explicit contrast).
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

/// Which signals [`crate::sentiment::emotion::EmotionAnnotator`] scores with.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum EmotionStage {
    #[default]
    Lexicon = 0,
    Learned = 1,
    Hybrid = 2,
}

/// Which unit layer [`crate::sentiment::emotion::EmotionAnnotator`] scores.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum EmotionUnit {
    #[default]
    Sentences = 0,
    Chunks = 1,
}

/// One `(label, token range)` evidence pair for [`EmotionLayer`].
pub type EmotionEvidence = (EmotionDimension, IxRange<Tokens>);

/// A multi-label emotion head over one vector per unit.
pub trait EmotionVectorScorer: Send + Sync {
    /// The 28 scores of `vec`, each in `0.0..=1.0`.
    fn score_emotions(&self, vec: &[f32]) -> Result<EmotionScores, String>;

    /// The encoder the vectors must come from, when the head names one.
    fn model(&self) -> Option<ModelRef> {
        None
    }

    /// A hash of the head's configuration, for provenance.
    fn config_hash(&self) -> u64 {
        0
    }
}

/// Number of [`IncivilityLabel`] dimensions (`6`).
pub const INCIVILITY_DIMENSIONS: usize = IncivilityLabel::ALL.len();

/// One `(label, token_range)` evidence span attributing an [`IncivilityLabel`]
/// to a contiguous token slice.
pub type IncivilityEvidenceSpan = (IncivilityLabel, IxRange<Tokens>);

/// Which unit span layer [`crate::sentiment::incivility::IncivilityAnnotator`] scores.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum IncivilityUnit {
    /// Score each sentence in `turbo:sentences` (`IncivilityTarget::Sentence`).
    #[default]
    Sentences = 0,
    /// Score each retrieval chunk in `turbo:chunks` (`IncivilityTarget::Chunk`).
    Chunks = 1,
    /// Score each structure block in `turbo:structure` (`IncivilityTarget::Block`).
    Blocks = 2,
}

/// Which signals [`crate::sentiment::incivility::IncivilityAnnotator`] scores
/// with. The stage is named in its model reference and its configuration hash.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum IncivilityStage {
    /// The lexicon (and reply-context rules) alone; a vector scorer, if
    /// set, is not read.
    Lexicon = 0,
    /// The vector scorer alone; it must be set, and a unit with no vector
    /// fails the document.
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

/// A multi-label embedding head that scores a sentence or chunk vector
/// (`&[f32]`) into [`IncivilityScores`] (`0.0..=1.0` per label).
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

/// Optional scorer that assigns an aspect category [`Tag`] from a [`TagSet`]
/// to an aspect (explicit or implicit) given its surface text and enclosing
/// sentence text.
///
/// Implemented in `turbo-analysis-embed` by `LabelCentroids`-backed category
/// scorers as well as keyword/centroid hybrid scorers.
pub trait AspectCategoryScorer: Send + Sync + fmt::Debug {
    /// The aspect category [`TagSet`] produced by this scorer.
    fn categories(&self) -> &Arc<TagSet>;

    /// Scores the aspect described by `aspect_text` (empty for implicit
    /// aspects), `opinion_text`, and `sentence_text`, returning the best
    /// category [`Tag`] if confidence exceeds the scorer's threshold.
    fn score_category(&self, aspect_text: &str, opinion_text: &str, sentence_text: &str) -> Option<Tag>;

    /// Deterministic configuration digest for provenance.
    fn config_hash(&self) -> u64;
}

/// Which signals [`crate::sentiment::polarity::PolarityAnnotator`] and
/// [`crate::sentiment::polarity::SentencePolarityAnnotator`] score with.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PolarityStage {
    /// Deterministic VADER lexicon and negation rules alone.
    #[default]
    Lexicon = 0,
    /// Learned [`PolarityVectorScorer`] head over `Vectors` / `Vectors<Sentences>` alone.
    Learned = 1,
    /// Convex combination of VADER lexicon polarity probabilities and the
    /// learned [`PolarityVectorScorer`] head probabilities.
    Hybrid = 2,
}

impl PolarityStage {
    pub const fn model_id(self) -> &'static str {
        match self {
            PolarityStage::Lexicon => "polarity:lexicon:v1",
            PolarityStage::Learned => "polarity:learned:v1",
            PolarityStage::Hybrid => "polarity:hybrid:v1",
        }
    }
}

/// Three-class (`positive`, `negative`, `neutral`) dense-vector polarity scorer
/// used by [`crate::sentiment::polarity::PolarityAnnotator`] and
/// [`crate::sentiment::polarity::SentencePolarityAnnotator`].
pub trait PolarityVectorScorer: Send + Sync {
    /// Scores one dense embedding vector `vec` (`&[f32]`), returning normalized
    /// [`PolarityScores`] in `0.0..=1.0` summing to `1.0`.
    fn score_polarity(&self, vec: &[f32]) -> Result<PolarityScores, String>;

    /// Optional [`ModelRef`] identifying the encoder that must have produced
    /// the input vector layer.
    fn model(&self) -> Option<ModelRef> {
        None
    }

    /// Deterministic configuration and weight digest mixed into provenance.
    fn config_hash(&self) -> u64 {
        0
    }
}
