//! Document categorization (`DocumentCategorizerME`, `DoccatModel`,
//! `DoccatFactory`, `SvmCategorizer`, `SvmDoccatModel`, feature generators
//! and `SpanCategorizer`), ported from OpenNLP 3.x (`opennlp.tools.doccat.*`
//! and `opennlp.tools.ml.libsvm.doccat.*`) onto the analysis API.
//!
//! # Overview
//!
//! - The layer every categorizer here writes is the shared
//!   [`turbo_analysis_layers::Categories`]: `Categories` (`turbo:categories`)
//!   for the whole document, `Categories<Sentences>`, `Categories<Structure>`
//!   and `Categories<Chunks>` for each sentence, block or chunk.
//!   [`SpanCategorizable`] is a layer of spans [`SpanCategorizer`] can
//!   categorize.
//! - [`DoccatFeatureGenerator`] and [`FeatureGenerator`]: feature extraction
//!   from tokenized text, with built-in implementations
//!   [`BagOfWordsFeatureGenerator`], [`NGramFeatureGenerator`], and
//!   [`EmojiFeatureGenerator`], chained by [`DoccatContextGenerator`].
//! - [`DoccatModel`]: loads and validates OpenNLP `DocumentCategorizerME`
//!   `.bin` model archives (`doccat.model` and `manifest.properties`).
//! - [`DocumentCategorizerME`] and [`SpanCategorizer`]: categorize tokenized
//!   text or run in an [`turbo_analysis::Analyzer`] pipeline to write
//!   `turbo:categories`, `Categories<Sentences>`, `Categories<Structure>`,
//!   or `Categories<Chunks>`.
//! - [`DoccatFactory`] and [`DoccatEventStream`]: the feature generators a
//!   model is trained with and the training events of `DocumentSample`s.
//!   `turbo_train_components::doccat` trains and packages the model file.
//! - [`svm`]: the SVM categorizer of `opennlp-ml-libsvm`
//!   (`DocumentCategorizerSVM`): [`SvmDoccatModel`] opens a model exported
//!   as a LIBSVM text model and a sidecar, and [`SvmCategorizer`] runs
//!   `categorize` and, as an annotator, reads `opennlp:tokens` and writes
//!   `turbo:categories`. Its LIBSVM predictor ([`svm::libsvm`]) is a port
//!   of LIBSVM's Java prediction code.
//! - [`sentiment`] is `SentimentME` (`opennlp.tools.sentiment`), a maxent
//!   classifier over the tokens and, with `EmojiSentimentFactory`, their
//!   emoji features: [`SentimentModel`] opens its model file and
//!   [`SentimentMe`] reads `opennlp:tokens` and writes the `sentiment`
//!   instance of `turbo:categories`.
//!
//! docs/doccat.md describes the model files, the parity tests and their
//! tolerances, and why the predictor is a port rather than linfa-svm;
//! docs/sentiment.md describes the sentiment detector.

pub mod calibrated;
pub mod categorizer;
pub mod error;
pub mod features;
mod hash;
pub mod java;
pub mod layer;
pub mod model;
pub mod sentiment;
pub mod svm;
pub mod train;

pub use categorizer::{DocumentCategorizerME, SpanCategorizer};
pub use error::DoccatError;
pub use features::{
    BAG_OF_WORDS_CLASS, BagOfWordsFeatureGenerator, BuiltinFeatureGenerator, DoccatContextGenerator,
    DoccatFeatureGenerator, EMOJI_CLASS, EmojiFeatureGenerator, FeatureGenerator, NGRAM_CLASS, NGramFeatureGenerator,
};
pub use layer::SpanCategorizable;
pub use model::{
    COMPONENT_NAME, DoccatModel, FACTORY_NAME, FEATURE_GENERATORS_KEY, MODEL_ENTRY, parse_feature_generators,
};
pub use sentiment::{
    AbsaMetrics, AbsaTriplet, AspectCategoryScorer, DEFAULT_ASPECT_CATEGORIES, DEFAULT_INCIVILITY_THRESHOLD,
    GoldIncivilityUnit, IncivilityAnnotator, IncivilityLexicon, IncivilityMetrics, IncivilityUnit,
    IncivilityVectorScorer, LabelMetrics, LexiconSentiment, NegationMode, PragmaticsAnnotator, PragmaticsVectorScorer,
    RuleAspectSentiment, SENTENCE_SENTIMENT, SentenceSentiment, SentimentFactory, SentimentMe, SentimentModel,
    VaderLexicon, VaderScore, evaluate_absa_corpus, evaluate_incivility_corpus, extract_triplets,
};
pub use svm::{ModelError, Scratch, SvmCategorizer, SvmDoccatModel};
pub use train::{DoccatEventStream, DoccatFactory, doccat_events, doccat_index};

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
    fn bag_of_words_only_letters_reads_utf16_units_like_the_jvm() {
        // StringPattern.recognize looks at one UTF-16 unit at a time, and a
        // surrogate is not a letter, so a token holding a supplementary
        // letter (U+20000, a CJK ideograph) is not all letters on the JVM.
        let extra = BTreeMap::new();
        let only_letters = BagOfWordsFeatureGenerator::only_all_letter_tokens();
        assert_eq!(only_letters.pattern_mode(), turbo_sequence::PatternMode::Utf16Units);
        assert_eq!(only_letters.collect_features(&["ab", "\u{20000}", "a\u{20000}"], &extra), vec!["bow=ab"]);
        let code_points = only_letters.with_pattern_mode(turbo_sequence::PatternMode::CodePoints);
        assert_eq!(code_points.collect_features(&["ab", "\u{20000}"], &extra), vec!["bow=ab", "bow=\u{20000}"]);
    }

    #[test]
    fn ngram_feature_generator_matches_opennlp_tests() {
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

        let gram13 = NGramFeatureGenerator::new(1, 3).unwrap();
        assert_eq!(
            gram13.collect_features(&tokens, &extra),
            vec![
                "ng=:a",
                "ng=:a:b",
                "ng=:a:b:c",
                "ng=:b",
                "ng=:b:c",
                "ng=:b:c:d",
                "ng=:c",
                "ng=:c:d",
                "ng=:c:d:e",
                "ng=:d",
                "ng=:d:e",
                "ng=:d:e:f",
                "ng=:e",
                "ng=:e:f",
                "ng=:e:f:g",
                "ng=:f",
                "ng=:f:g",
                "ng=:g"
            ]
        );
    }

    #[test]
    fn emoji_feature_generator_matches_opennlp_tests() {
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
