//! Document categorization, zero-shot and NLI claim verification, and
//! sentiment, polarity, aspect, emotion, incivility, and pragmatics analysis.
//!
//! # Overview
//!
//! - The category layer is the shared [`turbo_analysis_layers::Categories`]:
//!   `Categories` (`turbo:categories`) for the whole document, and
//!   `Categories<Sentences>`, `Categories<Structure>`, and `Categories<Chunks>`
//!   for each sentence, block, or chunk. [`SpanCategorizable`] marks span
//!   layers that span categorizers can score item by item.
//! - [`scorers`] defines the vector-head and category scorer traits
//!   ([`PolarityVectorScorer`], [`PragmaticsVectorScorer`],
//!   [`EmotionVectorScorer`], [`IncivilityVectorScorer`], and
//!   [`AspectCategoryScorer`]) and re-exports the sentiment and pragmatics
//!   layer types.
//! - [`dl`] implements sequence-classification document and span categorization
//!   ([`DocumentCategorizerDL`], [`DlSpanCategorizer`]) over an encoder-backed
//!   [`DlSequenceScorer`].
//! - [`multilabel`] implements multi-label categorization
//!   ([`MultiLabelDoccatModel`], [`MultiLabelCategorizer`],
//!   [`MultiLabelSpanCategorizer`]), one-vs-rest training ([`train_multilabel`]),
//!   and per-label threshold calibration ([`ThresholdCalibrator`]).
//! - [`zeroshot`] and [`claim_verify`] implement NLI-backed zero-shot
//!   categorization ([`ZeroShotCategorizer`], [`ZeroShotAnnotator`]) and claim
//!   verification ([`ClaimVerificationAnnotator`]).
//! - [`sentiment`] implements document-level and sentence-level polarity
//!   classification ([`PolarityAnnotator`], [`SentencePolarityAnnotator`],
//!   [`PolarityVectorHead`]), VADER lexicon and dependency-negation sentiment
//!   ([`LexiconSentiment`]), rule-based aspect and opinion extraction
//!   ([`RuleAspectSentiment`]), 28-label emotion scoring ([`EmotionAnnotator`],
//!   [`EmotionVectorHead`]), 6-label incivility and toxicity scoring
//!   ([`IncivilityAnnotator`], [`IncivilityVectorHead`]), and sentence-level
//!   sarcasm and irony detection ([`PragmaticsAnnotator`]).
//!
//! See `docs/doccat.md` and `docs/sentiment.md` for architecture and testing details.

pub mod calibrated;
#[cfg(feature = "classical")]
pub mod categorizer;
pub mod claim_verify;
pub mod dl;
pub mod error;
pub mod features;
mod hash;
pub mod layer;
pub mod model;
pub mod multilabel;
pub mod scorers;
pub mod selection;
pub mod sentiment;
#[cfg(feature = "classical")]
pub mod svm;
pub mod train;
pub mod zeroshot;

#[cfg(feature = "classical")]
pub use categorizer::{DocumentCategorizerME, SpanCategorizer};
pub use claim_verify::{ClaimVerificationAnnotator, ClaimVerifyConfig};
pub use dl::{
    AverageClassificationScoringStrategy, ClassificationScoringStrategy, DlChunkingMode, DlDoccatError,
    DlInferenceOptions, DlSequenceScorer, DlSpanCategorizer, DocumentCategorizerConfig, DocumentCategorizerDL,
    preprocess_text as dl_preprocess_text, softmax as dl_softmax, whitespace_chunks,
};
pub use error::DoccatError;
pub use features::{
    BAG_OF_WORDS_CLASS, BagOfWordsFeatureGenerator, BuiltinFeatureGenerator, DoccatContextGenerator,
    DoccatFeatureGenerator, EMOJI_CLASS, EmojiFeatureGenerator, FeatureGenerator, NGRAM_CLASS, NGramFeatureGenerator,
    PatternMode,
};
pub use layer::SpanCategorizable;
#[cfg(feature = "classical")]
pub use model::DoccatModel;
pub use model::{COMPONENT_NAME, FACTORY_NAME, FEATURE_GENERATORS_KEY, MODEL_ENTRY, parse_feature_generators};
pub use multilabel::{
    COMPONENT_NAME as MULTILABEL_COMPONENT_NAME, LinearHead, MultiLabelCategorizer, MultiLabelDoccatModel,
    MultiLabelError, MultiLabelReport, MultiLabelSample, MultiLabelSpanCategorizer, MultiLabelTrainingParameters, Prf,
    ThresholdCalibrator, evaluate, train_multilabel,
};
#[cfg(feature = "classical")]
pub use multilabel::{open_package as open_multilabel_package, package_multilabel_model};
pub use scorers::{
    AspectCategoryScorer, EmotionEvidence, EmotionStage, EmotionUnit, EmotionVectorScorer, INCIVILITY_DIMENSIONS,
    IncivilityEvidenceSpan, IncivilityStage, IncivilityUnit, IncivilityVectorContext, IncivilityVectorScorer,
    PolarityStage, PolarityVectorScorer, PragmaticsVectorScorer,
};
pub use selection::{DoccatTrainer, FeatureSelection, SelectedFeatures, SelectionMetric};
pub use sentiment::{
    AbsaMetrics, AbsaTriplet, CIVIL_COMMENTS_BGE_SMALL_CONTEXT_HEAD, CIVIL_COMMENTS_BGE_SMALL_HEAD,
    CIVIL_COMMENTS_MINILM_CONTEXT_HEAD, CIVIL_COMMENTS_MINILM_HEAD, DEFAULT_ASPECT_CATEGORIES,
    DEFAULT_EMOTION_THRESHOLD, DEFAULT_HEAD_WEIGHT, DEFAULT_INCIVILITY_FUSION_GATE, DEFAULT_INCIVILITY_THRESHOLD,
    EmotionAnnotator, EmotionFitOptions, EmotionHeadError, EmotionMetrics, EmotionVectorHead, EmotionZooError,
    GOEMOTIONS_BGE_SMALL_HEAD, GOEMOTIONS_MINILM_HEAD, GoldBiasUnit, GoldEmotionUnit, GoldIncivilityUnit,
    GoldToxicSpanUnit, IdentityTerms, IncivilityAnnotator, IncivilityBiasMetrics, IncivilityFitOptions,
    IncivilityHeadError, IncivilityLexicon, IncivilityMetrics, IncivilityVectorHead, IncivilityZooError, LabelBias,
    LabelMetrics, LexiconSentiment, NegationMode, POLARITY_COUNT, POLARITY_NAMES, PolarityAnnotator,
    PolarityClassMetrics, PolarityExample, PolarityFitConfig, PolarityFitReport, PolarityMetrics, PolarityVectorHead,
    PragmaticsAnnotator, RuleAspectSentiment, SENTENCE_SENTIMENT, SENTIMENT, SentencePolarityAnnotator,
    SentenceSentiment, SubgroupBias, ToxicSpanMetrics, VaderLexicon, VaderScore, blend_polarity_scores,
    char_span_to_token_range, char_span_to_utf8_span, evaluate_absa_corpus, evaluate_emotion_corpus,
    evaluate_emotion_corpus_with_thresholds, evaluate_incivility_bias, evaluate_incivility_corpus,
    evaluate_incivility_corpus_with_thresholds, evaluate_polarity, evaluate_predictions, evaluate_toxic_spans,
    extract_triplets, incivility_token_key, index_polarity, majority_baseline, parse_polarity_label,
    parse_polarity_tsv, polarity_index, roc_auc, softmax_polarity, vader_to_polarity_scores, validate_polarity_scores,
};
#[cfg(feature = "classical")]
pub use sentiment::{
    COMPONENT as SENTIMENT_COMPONENT, EMOJI_FACTORY, MODEL_ENTRY as SENTIMENT_MODEL_ENTRY, Scratch as SentimentScratch,
    SentimentFactory, SentimentMe, SentimentModel, SentimentModelError, TOKEN_FACTORY,
};
#[cfg(feature = "classical")]
pub use svm::{SvmCategorizer, SvmDoccatModel};
#[cfg(feature = "classical")]
pub use train::doccat_index;
pub use train::{DoccatEventStream, DoccatFactory, doccat_events};
pub use zeroshot::{
    NliScorer, WindowAggregation, ZeroShotAnnotator, ZeroShotCategorizer, ZeroShotConfig, ZeroShotError, ZeroShotMode,
    ZeroShotSpanAnnotator,
};

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    #[test]
    fn bag_of_words_all_tokens_and_only_letters() {
        let extra = BTreeMap::new();
        let all = BagOfWordsFeatureGenerator::new();
        assert!(all.collect_features(&[], &extra).is_empty());
        assert_eq!(
            all.collect_features(&["it", "is", "12.345", "feet", "long"], &extra),
            vec!["bow=it", "bow=is", "bow=12.345", "bow=feet", "bow=long"]
        );

        let only_letters = BagOfWordsFeatureGenerator::only_all_letter_tokens();
        assert_eq!(
            only_letters.collect_features(&["it", "is", "12.345", "feet", "long"], &extra),
            vec!["bow=it", "bow=is", "bow=feet", "bow=long"]
        );
    }

    #[test]
    fn bag_of_words_only_letters_reads_utf16_units() {
        let extra = BTreeMap::new();
        let only_letters = BagOfWordsFeatureGenerator::only_all_letter_tokens();
        assert_eq!(only_letters.pattern_mode(), turbo_sequence::PatternMode::Utf16Units);
        assert_eq!(only_letters.collect_features(&["ab", "\u{20000}", "a\u{20000}"], &extra), vec!["bow=ab"]);
        let code_points = only_letters.with_pattern_mode(turbo_sequence::PatternMode::CodePoints);
        assert_eq!(code_points.collect_features(&["ab", "\u{20000}"], &extra), vec!["bow=ab", "bow=\u{20000}"]);
    }

    #[test]
    fn ngram_feature_generator_extracts_windows() {
        let extra = BTreeMap::new();
        let tokens = ["a", "b", "c", "d", "e", "f", "g"];

        assert!(NGramFeatureGenerator::new(0, 1).is_err());
        assert!(NGramFeatureGenerator::new(2, 1).is_err());

        let default_gen = NGramFeatureGenerator::default();
        assert!(default_gen.collect_features(&[], &extra).is_empty());
        assert!(default_gen.collect_features(&["only_one"], &extra).is_empty());

        let unigram = NGramFeatureGenerator::new(1, 1).unwrap();
        assert_eq!(
            unigram.collect_features(&tokens, &extra),
            vec!["ng=:a", "ng=:b", "ng=:c", "ng=:d", "ng=:e", "ng=:f", "ng=:g"]
        );

        let bigram = NGramFeatureGenerator::new(2, 2).unwrap();
        assert_eq!(
            bigram.collect_features(&tokens, &extra),
            vec!["ng=:a:b", "ng=:b:c", "ng=:c:d", "ng=:d:e", "ng=:e:f", "ng=:f:g"]
        );

        let trigram = NGramFeatureGenerator::new(3, 3).unwrap();
        assert_eq!(
            trigram.collect_features(&tokens, &extra),
            vec!["ng=:a:b:c", "ng=:b:c:d", "ng=:c:d:e", "ng=:d:e:f", "ng=:e:f:g"]
        );

        let gram12 = NGramFeatureGenerator::new(1, 2).unwrap();
        assert_eq!(
            gram12.collect_features(&tokens, &extra),
            vec![
                "ng=:a", "ng=:a:b", "ng=:b", "ng=:b:c", "ng=:c", "ng=:c:d", "ng=:d", "ng=:d:e", "ng=:e", "ng=:e:f",
                "ng=:f", "ng=:f:g", "ng=:g"
            ]
        );
    }

    #[test]
    fn emoji_feature_generator_extracts_metadata() {
        let extra = BTreeMap::new();
        let generator = EmojiFeatureGenerator::new();

        let features = generator.collect_features(&["great", "pizza", "\u{1F355}", "\u{1F600}"], &extra);
        assert_eq!(
            features,
            vec![
                "emojiSentiment=0",
                "emojiType=FOOD",
                "emojiCategory=FOOD_AND_DRINK",
                "emojiSentiment=2",
                "emojiType=FACE",
                "emojiCategory=SMILEYS_AND_EMOTION"
            ]
        );

        let flag_features = generator.collect_features(&["match", "\u{1F1EB}\u{1F1F7}"], &extra);
        assert!(flag_features.contains(&"emojiRegion=FR".to_owned()));
        assert!(flag_features.contains(&"emojiType=FLAG".to_owned()));
        assert!(flag_features.contains(&"emojiCategory=FLAGS".to_owned()));

        let plain = generator.collect_features(&["no", "emoji", "here", "", "\u{1F1E9}"], &extra);
        assert!(plain.is_empty());
    }
}
