//! Sentiment, polarity, aspect, emotion, incivility, and pragmatics analysis.
//!
//! - [`polarity`]: document-level ([`PolarityAnnotator`]) and sentence-level
//!   ([`SentencePolarityAnnotator`]) polarity classification combining
//!   [`LexiconSentiment`] with a learned [`PolarityVectorHead`] trained with
//!   `turbo-trainer`.
//! - [`lexicon`]: VADER lexicon and dependency-negation sentiment scorer
//!   ([`LexiconSentiment`]).
//! - [`aspect`]: rule-based aspect and opinion extractor
//!   ([`RuleAspectSentiment`]) and [`AspectCategoryScorer`].
//! - [`emotion`]: 28-label GoEmotions lexicon and vector-head annotator
//!   ([`EmotionAnnotator`], [`EmotionVectorHead`], [`EmotionVectorScorer`]).
//! - [`incivility`]: 6-label incivility and toxicity lexicon and vector-head
//!   annotator ([`IncivilityAnnotator`], [`IncivilityVectorHead`],
//!   [`IncivilityVectorScorer`]).
//! - [`pragmatics`]: sentence-level sarcasm and irony annotator
//!   ([`PragmaticsAnnotator`], [`PragmaticsVectorScorer`]).

pub mod aspect;
#[cfg(feature = "classical")]
mod detector;
pub mod emotion;
pub mod incivility;
pub mod lexicon;
#[cfg(feature = "classical")]
mod model;
pub mod polarity;
pub mod pragmatics;
mod token_key;
pub use token_key::{embed_token_key, token_key};
mod validation;

pub use crate::scorers::{
    AspectCategoryScorer, EmotionEvidence, EmotionStage, EmotionUnit, EmotionVectorScorer, INCIVILITY_DIMENSIONS,
    IncivilityEvidenceSpan, IncivilityStage, IncivilityUnit, IncivilityVectorContext, IncivilityVectorScorer,
    PolarityStage, PolarityVectorScorer, PragmaticsVectorScorer,
};
pub use aspect::{
    AbsaMetrics, AbsaTriplet, DEFAULT_ASPECT_CATEGORIES, RuleAspectSentiment, evaluate_absa_corpus, extract_triplets,
};
#[cfg(feature = "classical")]
pub use detector::{Scratch, SentimentMe};
pub use emotion::{
    DEFAULT_EMOTION_THRESHOLD, EmotionAnnotator, EmotionFitOptions, EmotionHeadError, EmotionMetrics,
    EmotionVectorHead, EmotionZooError, GOEMOTIONS_BGE_SMALL_HEAD, GOEMOTIONS_MINILM_HEAD, GOEMOTIONS_MODERNBERT_HEAD,
    GoldEmotionUnit, evaluate_emotion_corpus, evaluate_emotion_corpus_with_thresholds,
};
pub use incivility::{
    CIVIL_COMMENTS_BGE_SMALL_CONTEXT_HEAD, CIVIL_COMMENTS_BGE_SMALL_HEAD, CIVIL_COMMENTS_MINILM_CONTEXT_HEAD,
    CIVIL_COMMENTS_MINILM_HEAD, CIVIL_COMMENTS_MODERNBERT_CONTEXT_HEAD, CIVIL_COMMENTS_MODERNBERT_HEAD,
    DEFAULT_INCIVILITY_FUSION_GATE, DEFAULT_INCIVILITY_THRESHOLD, GoldBiasUnit, GoldIncivilityUnit, GoldToxicSpanUnit,
    IdentityTerms, IncivilityAnnotator, IncivilityBiasMetrics, IncivilityFitOptions, IncivilityHeadError,
    IncivilityLexicon, IncivilityMetrics, IncivilityVectorHead, IncivilityZooError, LabelBias, LabelMetrics,
    SubgroupBias, ToxicSpanMetrics, char_span_to_token_range, char_span_to_utf8_span, evaluate_incivility_bias,
    evaluate_incivility_corpus, evaluate_incivility_corpus_with_thresholds, evaluate_toxic_spans, incivility_token_key,
    roc_auc,
};
pub use lexicon::{
    B_DECR, B_INCR, C_INCR, COMPOUND_THRESHOLD, DEFAULT_ALPHA, LexiconSentiment, N_SCALAR, NegationMode,
    SENTENCE_SENTIMENT, SentenceSentiment, VaderLexicon, VaderScore,
};
#[cfg(feature = "classical")]
pub use model::{
    COMPONENT, EMOJI_FACTORY, MODEL_ENTRY, SentimentFactory, SentimentModel, SentimentModelError, TOKEN_FACTORY,
};
pub use polarity::{
    DEFAULT_HEAD_WEIGHT, MASC_MODERNBERT_POLARITY_HEAD, POLARITY_COUNT, POLARITY_NAMES, PolarityAnnotator,
    PolarityClassMetrics, PolarityExample, PolarityFitConfig, PolarityFitReport, PolarityMetrics, PolarityVectorHead,
    PolarityZooError, SENTIMENT, SentencePolarityAnnotator, blend_polarity_scores, evaluate_polarity,
    evaluate_predictions, index_polarity, majority_baseline, parse_polarity_label, parse_polarity_tsv, polarity_index,
    softmax_polarity, vader_to_polarity_scores, validate_polarity_scores,
};
pub use pragmatics::{
    DEFAULT_MIN_NEGATIVE_POLARITY, DEFAULT_MIN_POSITIVE_POLARITY, DEFAULT_PRAGMATICS_THRESHOLD, PragmaticsAnnotator,
};
