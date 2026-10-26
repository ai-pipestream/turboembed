//! OpenNLP's sentiment detector, `opennlp.tools.sentiment`: `SentimentME`
//! at prediction, the `SentimentModel` file it runs, and the context
//! generators of `SentimentFactory` and `EmojiSentimentFactory`.
//!
//! `SentimentME` is a maxent document classifier. Its context is the
//! document's tokens as they are, with no prefix; the emoji factory's
//! context generator appends, for each token in turn, the features
//! OpenNLP's `EmojiAnnotator.collectFeatures` gives it
//! (`turbo_normalize::emoji::features`): `emojiSentiment=`,
//! `emojiType=`, `emojiCategory=` and `emojiRegion=`. The sentiment
//! scores behind `emojiSentiment=` are OpenNLP's own, from its
//! project-authored `emoji-annotations.txt`.
//!
//! [`SentimentMe`] is an annotator that reads `opennlp:tokens` and writes
//! `turbo:categories`: every outcome of the model, best first, with its
//! probability. The layer is a document categorizer's, so a pipeline that
//! runs both a topic categorizer and this one binds this one's output to
//! the [`SENTIMENT`] instance ([`SentimentMe::bound`]).
//!
//! Training is in `turbo-train-components` (`sentiment`), which builds
//! its events with [`SentimentFactory::context`].

pub mod aspect;
mod detector;
pub mod emotion;
pub mod incivility;
pub mod lexicon;
mod model;
pub mod pragmatics;
mod token_key;
pub use token_key::{embed_token_key, token_key};
mod validation;

pub use aspect::{
    AbsaMetrics, AbsaTriplet, AspectCategoryScorer, DEFAULT_ASPECT_CATEGORIES, RuleAspectSentiment,
    evaluate_absa_corpus, extract_triplets,
};
pub use detector::{SENTIMENT, Scratch, SentimentMe};
pub use emotion::{
    DEFAULT_EMOTION_THRESHOLD, EmotionAnnotator, EmotionFitOptions, EmotionHeadError, EmotionMetrics, EmotionStage,
    EmotionUnit, EmotionVectorHead, EmotionVectorScorer, EmotionZooError, GOEMOTIONS_BGE_SMALL_HEAD,
    GOEMOTIONS_MINILM_HEAD, GoldEmotionUnit, evaluate_emotion_corpus, evaluate_emotion_corpus_with_thresholds,
};
pub use incivility::{
    CIVIL_COMMENTS_BGE_SMALL_CONTEXT_HEAD, CIVIL_COMMENTS_BGE_SMALL_HEAD, CIVIL_COMMENTS_MINILM_CONTEXT_HEAD,
    CIVIL_COMMENTS_MINILM_HEAD, DEFAULT_INCIVILITY_FUSION_GATE, DEFAULT_INCIVILITY_THRESHOLD, GoldBiasUnit,
    GoldIncivilityUnit, GoldToxicSpanUnit, INCIVILITY_DIMENSIONS, IdentityTerms, IncivilityAnnotator,
    IncivilityBiasMetrics, IncivilityFitOptions, IncivilityHeadError, IncivilityLexicon, IncivilityMetrics,
    IncivilityStage, IncivilityUnit, IncivilityVectorContext, IncivilityVectorHead, IncivilityVectorScorer,
    IncivilityZooError, LabelBias, LabelMetrics, SubgroupBias, ToxicSpanMetrics, char_span_to_token_range,
    char_span_to_utf8_span, evaluate_incivility_bias, evaluate_incivility_corpus,
    evaluate_incivility_corpus_with_thresholds, evaluate_toxic_spans, roc_auc,
};
pub use lexicon::{
    B_DECR, B_INCR, C_INCR, COMPOUND_THRESHOLD, DEFAULT_ALPHA, LexiconSentiment, N_SCALAR, NegationMode,
    SENTENCE_SENTIMENT, SentenceSentiment, VaderLexicon, VaderScore,
};
pub use model::{
    COMPONENT, EMOJI_FACTORY, MODEL_ENTRY, SentimentFactory, SentimentModel, SentimentModelError, TOKEN_FACTORY,
};
pub use pragmatics::{
    DEFAULT_MIN_NEGATIVE_POLARITY, DEFAULT_MIN_POSITIVE_POLARITY, DEFAULT_PRAGMATICS_THRESHOLD, PragmaticsAnnotator,
    PragmaticsVectorScorer,
};
