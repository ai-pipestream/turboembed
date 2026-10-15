//! Document categorization (`DocumentCategorizerME`, `DoccatModel`,
//! `DoccatFactory`, feature generators, and `Categories<L>`), ported from
//! OpenNLP 3.x (`opennlp.tools.doccat.*`) onto the analysis API.
//!
//! # Overview
//!
//! - [`Categories`]: layer `turbo:categories<L>` aligned to `View` (whole
//!   document), `Sentences` (sentence-level), `Structure` (paragraph/section),
//!   or `Chunks` (chunk-level).
//! - [`FeatureGenerator`]: trait for extracting document features from
//!   tokenized text, with built-in implementations
//!   [`BagOfWordsFeatureGenerator`], [`NGramFeatureGenerator`], and
//!   [`EmojiFeatureGenerator`], chained by [`DoccatContextGenerator`].
//! - [`DoccatModel`]: loads and validates OpenNLP `DocumentCategorizerME`
//!   `.bin` model archives (`doccat.model` and `manifest.properties`).
//! - [`DocumentCategorizerME`] and [`SpanCategorizer`]: categorize tokenized
//!   text or run in an [`turbo_analysis::Analyzer`] pipeline to write
//!   `Categories<View>`, `Categories<Sentences>`, `Categories<Structure>`, or
//!   `Categories<Chunks>`.
//! - [`train_doccat`], [`DoccatFactory`], and [`DoccatEventStream`]: train and
//!   package `DoccatModel` archives from `DocumentSample`s.

pub mod categorizer;
pub mod error;
pub mod features;
pub mod layer;
pub mod model;
pub mod train;

pub use categorizer::{DocumentCategorizerME, SpanCategorizer};
pub use error::DoccatError;
pub use features::{
    BAG_OF_WORDS_CLASS, BagOfWordsFeatureGenerator, BuiltinFeatureGenerator, DoccatContextGenerator, EMOJI_CLASS,
    EmojiFeatureGenerator, FeatureGenerator, NGRAM_CLASS, NGramFeatureGenerator,
};
pub use layer::{Categories, CategoryRow, SpanCategorizable};
pub use model::{
    COMPONENT_NAME, DoccatModel, FACTORY_NAME, FEATURE_GENERATORS_KEY, MODEL_ENTRY, parse_feature_generators,
};
pub use train::{
    DoccatEventStream, DoccatFactory, TrainedDoccatModel, doccat_events, doccat_index, package_doccat_model,
    train_doccat, train_doccat_from_events,
};

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use turbo_analysis::{Analyzer, Document, IxRange, Key, Layer, Provenance, Span, Spans, View, ViewId};
    use turbo_analysis_layers::{Block, Chunks, Cut, Sentences, Structure};
    use turbo_formats::DocumentSample;
    use turbo_formats::native::DocumentReader;
    use turbo_maxent::ModelFile;
    use turbo_tokenize::WhitespaceTokenizer;
    use turbo_train::{Key as ParamKey, TrainingConfiguration, TrainingParameters};
    use turbo_train_components::package::Stamp;

    use super::*;

    fn parse_samples(corpus: &str) -> Vec<DocumentSample<'_>> {
        DocumentReader::new(corpus, Default::default()).unwrap().map(|r| r.expect("valid sample")).collect()
    }

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

    #[test]
    fn simple_training_and_categorization_matches_opennlp_test() {
        let corpus = "\
1\ta b c
1\ta b c 1 2
1\ta b c 3 4
0\tx y z
0\tx y z 5 6
0\tx y z 7 8
";
        let samples = parse_samples(corpus);
        let mut params = TrainingParameters::new();
        params.put(ParamKey::ITERATIONS, "100");
        params.put(ParamKey::CUTOFF, "0");

        let factory = DoccatFactory::new();
        let mut config = TrainingConfiguration::default();
        let trained = train_doccat("x-unspecified", &samples, &params, &factory, &mut config, Stamp::at(0))
            .expect("train succeeds");

        let doccat = DocumentCategorizerME::new(trained.model);
        assert_eq!(doccat.number_of_categories(), 2);
        assert_eq!(doccat.get_category(0), Some("1"));
        assert_eq!(doccat.get_category(1), Some("0"));
        assert_eq!(doccat.get_index("1"), Some(0));
        assert_eq!(doccat.get_index("0"), Some(1));

        let a_probs = doccat.categorize(&["a"]);
        assert_eq!(doccat.best_category(&a_probs), "1");
        assert!((a_probs[0] - 0.759_384_690_516_128_4).abs() < 1e-15);
        assert!((a_probs[1] - 0.240_615_309_483_871_4).abs() < 1e-15);

        let b_probs = doccat.categorize(&["x"]);
        assert_eq!(doccat.best_category(&b_probs), "0");
        assert!((b_probs[0] - 0.240_615_309_483_871_4).abs() < 1e-15);
        assert!((b_probs[1] - 0.759_384_690_516_128_4).abs() < 1e-15);

        let abx_probs = doccat.categorize(&["a", "b", "x"]);
        assert_eq!(doccat.best_category(&abx_probs), "1");
        assert!((abx_probs[0] - 0.759_384_690_516_128_4).abs() < 1e-15);
        assert!((abx_probs[1] - 0.240_615_309_483_871_47).abs() < 1e-15);

        let tie_probs = doccat.categorize(&["1", "5"]);
        assert_eq!(doccat.best_category(&tie_probs), "1");
        assert!((tie_probs[0] - 0.5).abs() < 1e-15);
        assert!((tie_probs[1] - 0.5).abs() < 1e-15);

        let sorted = doccat.sorted_score_map(&["a"]);
        let (_, highest_cats) = sorted.last().expect("non-empty sorted score map");
        assert_eq!(highest_cats.len(), 1);
        assert!(highest_cats.contains("1"));

        let score_map = doccat.score_map(&["a"]);
        assert!(score_map["1"] > score_map["0"]);

        let formatted = doccat.all_results(&a_probs);
        assert_eq!(formatted, "1[0.7594]  0[0.2406]");
    }

    #[test]
    fn multi_generator_jvm_parity() {
        let corpus = "\
pos\tgreat pizza \u{1F355} \u{1F600}
pos\tgreat service \u{1F600}
pos\ttasty pizza \u{1F355}
neg\tbad pizza \u{1F621}
neg\tbad service \u{1F621}
neg\tcold pizza \u{1F621}
";
        let samples = parse_samples(corpus);
        let mut params = TrainingParameters::new();
        params.put(ParamKey::ITERATIONS, "100");
        params.put(ParamKey::CUTOFF, "0");

        let factory = DoccatFactory::with_feature_generators(vec![
            BuiltinFeatureGenerator::BagOfWords(BagOfWordsFeatureGenerator::new()),
            BuiltinFeatureGenerator::NGram(NGramFeatureGenerator::default()),
            BuiltinFeatureGenerator::Emoji(EmojiFeatureGenerator::new()),
        ]);
        let mut config = TrainingConfiguration::default();
        let trained =
            train_doccat("en", &samples, &params, &factory, &mut config, Stamp::at(0)).expect("train succeeds");

        let doccat = DocumentCategorizerME::new(trained.model);
        let p_pos = doccat.categorize(&["great", "pizza", "\u{1F600}"]);
        assert_eq!(doccat.best_category(&p_pos), "pos");
        assert!((p_pos[0] - 0.910_972_647_566_720_5).abs() < 1e-15);
        assert!((p_pos[1] - 0.089_027_352_433_279_53).abs() < 1e-15);

        let p_neg = doccat.categorize(&["bad", "service", "\u{1F621}"]);
        assert_eq!(doccat.best_category(&p_neg), "neg");
        assert!((p_neg[0] - 0.030_228_971_739_969_714).abs() < 1e-15);
        assert!((p_neg[1] - 0.969_771_028_260_030_3).abs() < 1e-15);

        let p_pizza = doccat.categorize(&["pizza"]);
        assert_eq!(doccat.best_category(&p_pizza), "neg");
        assert!((p_pizza[0] - 0.387_259_882_781_278_04).abs() < 1e-15);
        assert!((p_pizza[1] - 0.612_740_117_218_721_9).abs() < 1e-15);
    }

    #[test]
    fn doccat_factory_manifest_roundtrip() {
        let corpus = "\
1\ta b c
1\ta b c 1 2
1\ta b c 3 4
0\tx y z
0\tx y z 5 6
0\tx y z 7 8
";
        let samples = parse_samples(corpus);
        let mut params = TrainingParameters::new();
        params.put(ParamKey::ITERATIONS, "50");
        params.put(ParamKey::CUTOFF, "0");

        let factory = DoccatFactory::with_feature_generators(vec![
            BuiltinFeatureGenerator::BagOfWords(BagOfWordsFeatureGenerator::new()),
            BuiltinFeatureGenerator::NGram(NGramFeatureGenerator::default()),
            BuiltinFeatureGenerator::Emoji(EmojiFeatureGenerator::new()),
        ]);
        let mut config = TrainingConfiguration::default();
        let trained =
            train_doccat("en", &samples, &params, &factory, &mut config, Stamp::at(0)).expect("train succeeds");

        let archive_bytes = trained.package.to_bytes();
        let reopened = DoccatModel::open(ModelFile::open(archive_bytes).unwrap()).unwrap();
        assert_eq!(reopened.language(), "en");
        assert_eq!(reopened.feature_generators().len(), 3);
        assert_eq!(reopened.feature_generators()[0].class_name(), BAG_OF_WORDS_CLASS);
        assert_eq!(reopened.feature_generators()[1].class_name(), NGRAM_CLASS);
        assert_eq!(reopened.feature_generators()[2].class_name(), EMOJI_CLASS);
    }

    #[test]
    fn insufficient_training_data_is_rejected() {
        let corpus = "1\ta b c\n";
        let samples = parse_samples(corpus);
        let mut params = TrainingParameters::new();
        params.put(ParamKey::ITERATIONS, "100");
        params.put(ParamKey::CUTOFF, "5");

        let factory = DoccatFactory::new();
        let mut config = TrainingConfiguration::default();
        assert!(train_doccat("x-unspecified", &samples, &params, &factory, &mut config, Stamp::at(0)).is_err());
    }

    #[test]
    fn analyzer_writes_categories_on_view_sentences_structure_and_chunks() {
        let corpus = "\
sports\tgoal match team win stadium score
sports\tgoal match player coach league
sports\tteam win tournament match score
tech\tcompiler rust kernel memory thread
tech\tcompiler cpu memory binary cache
tech\trust kernel thread allocator latency
";
        let samples = parse_samples(corpus);
        let mut params = TrainingParameters::new();
        params.put(ParamKey::ITERATIONS, "100");
        params.put(ParamKey::CUTOFF, "0");

        let trained = train_doccat(
            "en",
            &samples,
            &params,
            &DoccatFactory::new(),
            &mut TrainingConfiguration::default(),
            Stamp::at(0),
        )
        .unwrap();

        let doccat = DocumentCategorizerME::new(trained.model);
        let sent_cat: SpanCategorizer<Sentences> = doccat.for_spans();
        let struct_cat: SpanCategorizer<Structure> = doccat.for_spans();
        let chunk_cat: SpanCategorizer<Chunks> = doccat.for_spans();

        let text = "goal match team win. compiler rust kernel memory.";
        let mut doc = Document::new(text);

        // Sentence 0: "goal match team win." (0..20)
        // Sentence 1: "compiler rust kernel memory." (21..49)
        let mut sent_spans = Spans::new(ViewId::SOURCE);
        sent_spans.push(Span::new(0, 20));
        sent_spans.push(Span::new(21, 49));
        doc.add(&Key::DEFAULT, Sentences { spans: sent_spans }, Provenance::by("test:sent")).unwrap();

        // Structure blocks matching the two sentences
        let mut struct_spans = Spans::new(ViewId::SOURCE);
        struct_spans.push(Span::new(0, 20));
        struct_spans.push(Span::new(21, 49));
        doc.add(
            &Key::DEFAULT,
            Structure {
                spans: struct_spans,
                kind: vec![Block::Paragraph, Block::Paragraph],
                parent: vec![None, None],
                depth: vec![0, 0],
            },
            Provenance::by("test:struct"),
        )
        .unwrap();

        // Chunks matching the two sentences
        let mut chunk_spans = Spans::new(ViewId::SOURCE);
        chunk_spans.push(Span::new(0, 20));
        chunk_spans.push(Span::new(21, 49));
        doc.add(
            &Key::DEFAULT,
            Chunks {
                spans: chunk_spans,
                level: vec![0, 0],
                parent: vec![None, None],
                sentences: vec![IxRange::new(0, 1), IxRange::new(1, 2)],
                cut: vec![Cut::DocumentStart, Cut::Sentence],
                cost: vec![0, 1000],
                model_tokens: vec![4, 4],
                over_budget: vec![false, false],
            },
            Provenance::by("test:chunks"),
        )
        .unwrap();

        let analyzer = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Structure>::DEFAULT)
            .given(Key::<Chunks>::DEFAULT)
            .add(WhitespaceTokenizer::default())
            .add(doccat)
            .add(sent_cat)
            .add(struct_cat)
            .add(chunk_cat)
            .build()
            .unwrap();
        let mut docs = [doc];
        for res in analyzer.run(&mut docs) {
            res.unwrap();
        }
        let doc = &docs[0];

        // 1. Whole-document Categories<View>
        let view_cats = doc.get::<Categories<View>>().unwrap();
        assert_eq!(view_cats.len(), 1);
        assert_eq!(Categories::<View>::NAME.to_string(), "turbo:categories<turbo:view>");
        let row0 = view_cats.get(0).unwrap();
        assert!((row0.scores.iter().sum::<f64>() - 1.0).abs() < 1e-9);

        // 2. Sentence-level Categories<Sentences>
        let sent_cats = doc.get::<Categories<Sentences>>().unwrap();
        assert_eq!(sent_cats.len(), 2);
        assert_eq!(Categories::<Sentences>::NAME.to_string(), "turbo:categories<opennlp:sentences>");
        assert_eq!(sent_cats.best_category(0), "sports");
        assert_eq!(sent_cats.best_category(1), "tech");

        // 3. Structure-level Categories<Structure>
        let struct_cats = doc.get::<Categories<Structure>>().unwrap();
        assert_eq!(struct_cats.len(), 2);
        assert_eq!(Categories::<Structure>::NAME.to_string(), "turbo:categories<turbo:structure>");
        assert_eq!(struct_cats.best_category(0), "sports");
        assert_eq!(struct_cats.best_category(1), "tech");

        // 4. Chunk-level Categories<Chunks>
        let chunk_cats = doc.get::<Categories<Chunks>>().unwrap();
        assert_eq!(chunk_cats.len(), 2);
        assert_eq!(Categories::<Chunks>::NAME.to_string(), "turbo:categories<turbo:chunks>");
        assert_eq!(chunk_cats.best_category(0), "sports");
        assert_eq!(chunk_cats.best_category(1), "tech");
        assert!(chunk_cats.score_of(0, "sports").unwrap() > 0.7);
        assert!(chunk_cats.score_of(1, "tech").unwrap() > 0.7);
    }
}
