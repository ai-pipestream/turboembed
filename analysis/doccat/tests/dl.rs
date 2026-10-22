//! Unit and pipeline integration tests for `DocumentCategorizerDL` (`#208`),
//! porting OpenNLP's `DocumentCategorizerConfigTest`,
//! `AverageClassificationScoringStrategyTest`, and `DocumentCategorizerDLTest`
//! plus `Annotator` pipeline tests over `opennlp:subwords`, `turbo:chunks`,
//! `turbo:structure`, and `opennlp:sentences`.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use turbo_analysis::{
    Analyzer, Annotator, Document, Execution, IxRange, Key, Layer, ModelRef, Provenance, Span, Spans, Tag, View, ViewId,
};
use turbo_analysis_layers::{Block, Categories, Chunks, Cut, Sentences, Structure, Subwords};
use turbo_doccat::{
    AverageClassificationScoringStrategy, ClassificationScoringStrategy, DlChunkingMode, DlDoccatError,
    DlInferenceOptions, DlSequenceScorer, DlSpanCategorizer, DocumentCategorizerConfig, DocumentCategorizerDL,
    dl_preprocess_text, dl_softmax, whitespace_chunks,
};

struct MockSequenceScorer {
    labels: Vec<String>,
    model: ModelRef,
    chunk_logits: BTreeMap<String, Vec<f32>>,
    default_logits: Vec<f32>,
    max_content_tokens: usize,
}

impl MockSequenceScorer {
    fn new(labels: &[&str], default_logits: Vec<f32>) -> Self {
        Self {
            labels: labels.iter().map(|&s| s.to_owned()).collect(),
            model: ModelRef {
                id: "test/bert-seq-doccat".into(),
                revision: "rev1".into(),
                sha256: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".into(),
            },
            chunk_logits: BTreeMap::new(),
            default_logits,
            max_content_tokens: 8,
        }
    }

    fn with_chunk(mut self, chunk: &str, logits: Vec<f32>) -> Self {
        self.chunk_logits.insert(chunk.to_owned(), logits);
        self
    }

    fn with_max_content_tokens(mut self, max_content_tokens: usize) -> Self {
        self.max_content_tokens = max_content_tokens;
        self
    }
}

impl DlSequenceScorer for MockSequenceScorer {
    fn categories(&self) -> &[String] {
        &self.labels
    }

    fn model_ref(&self) -> ModelRef {
        self.model.clone()
    }

    fn execution(&self) -> Option<Execution> {
        Some(Execution { backend: "cpu".into(), device: "mock-cpu".into(), precision: "f32 (model)".into() })
    }

    fn max_content_tokens(&self) -> usize {
        self.max_content_tokens
    }

    fn score_chunks(&self, chunks: &[String], options: &DlInferenceOptions) -> Result<Vec<Vec<f32>>, DlDoccatError> {
        let mut out = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            let processed = dl_preprocess_text(chunk, options);
            let row = self
                .chunk_logits
                .get(&processed)
                .or_else(|| self.chunk_logits.get(chunk))
                .cloned()
                .unwrap_or_else(|| self.default_logits.clone());
            out.push(row);
        }
        Ok(out)
    }

    fn score_subword_windows(&self, windows: &[&[i32]]) -> Result<Vec<Vec<f32>>, DlDoccatError> {
        let mut out = Vec::with_capacity(windows.len());
        for win in windows {
            let key = win.iter().map(i32::to_string).collect::<Vec<_>>().join(",");
            let row = self.chunk_logits.get(&key).cloned().unwrap_or_else(|| self.default_logits.clone());
            out.push(row);
        }
        Ok(out)
    }
}

#[test]
fn document_categorizer_config_matches_opennlp_config_test() {
    let json = r#"{ "_name_or_path": "nlptown/bert-base-multilingual-uncased-sentiment", "architectures": [ "BertForSequenceClassification" ], "id2label": { "0": "1 star", "1": "2 stars", "2": "3 stars", "3": "4 stars", "4": "5 stars" } }"#;
    let config = DocumentCategorizerConfig::from_json(json).expect("valid id2label json");
    assert_eq!(config.id2label().len(), 5);
    assert_eq!(config.id2label().get("0").map(String::as_str), Some("1 star"));
    assert_eq!(config.id2label().get("1").map(String::as_str), Some("2 stars"));
    assert_eq!(config.id2label().get("2").map(String::as_str), Some("3 stars"));
    assert_eq!(config.id2label().get("3").map(String::as_str), Some("4 stars"));
    assert_eq!(config.id2label().get("4").map(String::as_str), Some("5 stars"));

    let ordered = config.ordered_categories().unwrap();
    assert_eq!(ordered, vec!["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]);

    let indexed = DocumentCategorizerConfig::read_categories_from_json(json).unwrap();
    assert_eq!(indexed.get(&0).map(String::as_str), Some("1 star"));
    assert_eq!(indexed.get(&4).map(String::as_str), Some("5 stars"));

    // Invalid JSON without id2label or with malformed/nested/trailing payloads returns None.
    assert!(DocumentCategorizerConfig::from_json("bad json").is_none());
    assert!(DocumentCategorizerConfig::from_json(r#"{"other": 1}"#).is_none());
    assert!(DocumentCategorizerConfig::from_json(r#"garbage "id2label": {"0": "a", "1": "b"}"#).is_none());
    assert!(DocumentCategorizerConfig::from_json(r#"{"nested": {"id2label": {"0": "a", "1": "b"}}}"#).is_none());
    assert!(DocumentCategorizerConfig::from_json(r#"{"id2label": {"0": "a", "1": "b"}} trailing"#).is_none());
    assert!(
        DocumentCategorizerConfig::from_json(r#"{"id2label": {"0": "a", "1": "b"}, "id2label": {"0": "a", "1": "b"}}"#)
            .is_none()
    );
    assert!(DocumentCategorizerConfig::from_json(r#"{"id2label": {"0": "a"}, "bad": [1, 2}"#).is_none());

    // Valid top-level id2label alongside booleans, nulls, numbers, escapes, and nested objects/arrays succeeds.
    let rich_json = r#"{"num": -12.5e+2, "ok": true, "no": false, "nil": null, "nested": {"id2label": "ignored", "arr": [1, {"k": "v\u0020"}]}, "id2label": {"0": "neg", "1": "pos"}}"#;
    let rich_cfg = DocumentCategorizerConfig::from_json(rich_json).expect("rich top-level config.json");
    assert_eq!(rich_cfg.ordered_categories().unwrap(), vec!["neg", "pos"]);

    // Non-contiguous or non-numeric keys in read_categories_from_json fail cleanly.
    let gap_json = r#"{"id2label": {"0": "neg", "2": "pos"}}"#;
    assert!(DocumentCategorizerConfig::read_categories_from_json(gap_json).is_err());
    let non_int_json = r#"{"id2label": {"zero": "neg", "one": "pos"}}"#;
    assert!(DocumentCategorizerConfig::read_categories_from_json(non_int_json).is_err());
}

#[test]
fn average_classification_scoring_strategy_matches_opennlp_test_and_validates_inputs() {
    let strategy = AverageClassificationScoringStrategy::new();
    let scores = vec![vec![0.1, 0.2, 0.3, 0.5, 0.4], vec![0.3, 0.3, 0.2, 0.5, 0.5]];
    let results = strategy.score(&scores).unwrap();
    let expected = [0.2, 0.25, 0.25, 0.5, 0.45];
    assert_eq!(results.len(), expected.len());
    for (got, want) in results.iter().zip(&expected) {
        assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
    }

    assert!(strategy.score(&[]).is_err());
    assert!(strategy.score(&[vec![]]).is_err());
    assert!(strategy.score(&[vec![0.5, 0.5], vec![0.5]]).is_err());
    assert!(strategy.score(&[vec![0.5, f64::NAN]]).is_err());
    assert!(strategy.score(&[vec![0.5, f64::INFINITY]]).is_err());
}

#[test]
fn softmax_and_whitespace_chunks_match_opennlp_abstract_dl() {
    assert!(dl_softmax(&[]).is_err());
    assert!(dl_softmax(&[0.0, f32::NAN]).is_err());
    assert!(dl_softmax(&[0.0, f32::INFINITY]).is_err());
    assert!(dl_softmax(&[0.0, f32::NEG_INFINITY]).is_err());

    let probs = dl_softmax(&[0.0, 0.0]).unwrap();
    assert!((probs[0] - 0.5).abs() < 1e-15);
    assert!((probs[1] - 0.5).abs() < 1e-15);

    // Shift invariance with large logits.
    let p_small = dl_softmax(&[1.0, 2.0, 3.0]).unwrap();
    let p_shifted = dl_softmax(&[51.0, 52.0, 53.0]).unwrap();
    for (a, b) in p_small.iter().zip(&p_shifted) {
        assert!((a - b).abs() < 1e-12);
    }

    // Invalid split options are rejected.
    assert!(whitespace_chunks("a b c", 0, 0).is_err());
    assert!(whitespace_chunks("a b c", 3, 3).is_err());
    assert!(whitespace_chunks("a b c", 3, 4).is_err());

    // Empty and Unicode-whitespace-only strings produce zero chunks.
    assert!(whitespace_chunks("", 250, 50).unwrap().is_empty());
    assert!(whitespace_chunks(" \t\r\n", 250, 50).unwrap().is_empty());
    assert!(whitespace_chunks("\u{00A0}\u{2003}", 250, 50).unwrap().is_empty());

    // Overlapping window geometry matches AbstractDL.whitespaceChunks:
    // 7 words with documentSplitSize=3, splitOverlapSize=1 -> [0..3], [2..5], [4..7].
    let chunks = whitespace_chunks("w0  w1\tw2\nw3\u{00A0}w4 w5 w6", 3, 1).unwrap();
    assert_eq!(chunks, vec!["w0 w1 w2", "w2 w3 w4", "w4 w5 w6"]);

    // Preprocessing normalizes whitespace, dashes, and case in order.
    let opts =
        DlInferenceOptions::default().with_normalize_whitespace(true).with_normalize_dashes(true).with_lower_case(true);
    let processed = dl_preprocess_text("Hello\u{00A0}World\u{2013}Test", &opts);
    assert_eq!(processed, "hello world-test");
}

#[test]
fn document_categorizer_dl_matches_opennlp_test_suite_and_edge_cases() {
    let labels = ["very negative", "negative", "neutral", "positive", "very positive"];
    // Logits chosen so chunk "i am happy" strongly favors "very positive" (index 4)
    // and chunk "i am angry" strongly favors "very negative" (index 0).
    let scorer = MockSequenceScorer::new(&labels, vec![0.0, 0.0, 0.0, 0.0, 0.0])
        .with_chunk("i am happy", vec![-3.0, -2.0, -1.0, 1.0, 3.0])
        .with_chunk("i am angry", vec![3.0, 1.0, -1.0, -2.0, -3.0]);

    let cat = DocumentCategorizerDL::new(scorer).unwrap();
    assert_eq!(cat.number_of_categories(), 5);
    assert_eq!(cat.get_index("very negative"), Some(0));
    assert_eq!(cat.get_index("very positive"), Some(4));
    assert_eq!(cat.get_index("unknown"), None);
    assert_eq!(cat.get_category(0), Some("very negative"));
    assert_eq!(cat.get_category(4), Some("very positive"));
    assert_eq!(cat.get_category(5), None);

    let happy_scores = cat.categorize(&["I am happy"]).unwrap();
    assert_eq!(cat.best_category(&happy_scores).unwrap(), "very positive");
    assert_eq!(cat.best_index(&happy_scores).unwrap(), 4);

    let angry_scores = cat.categorize(&["I am angry"]).unwrap();
    assert_eq!(cat.best_category(&angry_scores).unwrap(), "very negative");
    assert_eq!(cat.best_index(&angry_scores).unwrap(), 0);

    let map = cat.score_map(&["I am happy"]).unwrap();
    assert_eq!(map.len(), 5);
    assert!(map["very positive"] > map["very negative"]);

    let sorted = cat.sorted_score_map(&["I am happy"]).unwrap();
    assert_eq!(sorted.len(), 5);
    assert_eq!(sorted.last().unwrap().1, BTreeSet::from(["very positive".to_owned()]));

    // Ties in best_category break toward the lowest index (matching maxIndex in Java).
    let tied = vec![0.2, 0.4, 0.4, 0.0, 0.0];
    assert_eq!(cat.best_index(&tied).unwrap(), 1);
    assert_eq!(cat.best_category(&tied).unwrap(), "negative");

    // Empty / blank / Unicode-whitespace-only inputs are rejected with EmptyDocument.
    assert!(matches!(cat.categorize(&[]), Err(DlDoccatError::EmptyDocument(_))));
    assert!(matches!(cat.categorize(&[""]), Err(DlDoccatError::EmptyDocument(_))));
    assert!(matches!(cat.categorize(&["   \t\r\n  "]), Err(DlDoccatError::EmptyDocument(_))));
    assert!(matches!(cat.categorize(&["\u{00A0}\u{2003}"]), Err(DlDoccatError::EmptyDocument(_))));

    // Outcome vector validation in best_category / best_index.
    assert!(cat.best_category(&[]).is_err());
    assert!(cat.best_category(&[0.5, 0.5]).is_err());
    assert!(cat.best_category(&[0.2, 0.2, f64::NAN, 0.2, 0.2]).is_err());
}

#[test]
fn annotator_pipeline_writes_document_and_span_categories_in_both_chunking_modes() {
    let labels = ["neg", "pos"];
    let scorer = MockSequenceScorer::new(&labels, vec![0.0, 0.0])
        .with_chunk("good news", vec![-2.0, 2.0])
        .with_chunk("bad news", vec![2.0, -2.0])
        .with_chunk("good news bad news", vec![-0.5, 1.5])
        .with_chunk("10,11", vec![-2.0, 2.0])
        .with_chunk("20,21", vec![2.0, -2.0])
        .with_chunk("10,11,20,21", vec![-0.5, 1.5]);
    let model_ref = scorer.model_ref();

    let text = "good news bad news";
    let mut sub_spans = Spans::new(ViewId::SOURCE);
    sub_spans.push(Span::new(0, 4));
    sub_spans.push(Span::new(5, 9));
    sub_spans.push(Span::new(10, 13));
    sub_spans.push(Span::new(14, 18));
    let subwords = Subwords { spans: sub_spans, ids: vec![10, 11, 20, 21], model: model_ref.clone(), token: None };

    let mut sent_spans = Spans::new(ViewId::SOURCE);
    sent_spans.push(Span::new(0, 9));
    sent_spans.push(Span::new(10, 18));
    let sentences = Sentences { spans: sent_spans };

    let mut chunk_spans = Spans::new(ViewId::SOURCE);
    chunk_spans.push(Span::new(0, 9));
    chunk_spans.push(Span::new(10, 18));
    let chunks = Chunks {
        spans: chunk_spans,
        level: vec![0, 0],
        parent: vec![None, None],
        sentences: vec![IxRange::new(0, 1), IxRange::new(1, 2)],
        cut: vec![Cut::DocumentStart, Cut::Sentence],
        cost: vec![0, 1000],
        model_tokens: vec![4, 4],
        over_budget: vec![false, false],
    };

    let mut struct_spans = Spans::new(ViewId::SOURCE);
    struct_spans.push(Span::new(0, 9));
    struct_spans.push(Span::new(10, 18));
    let structure = Structure {
        spans: struct_spans,
        kind: vec![Block::Paragraph, Block::Paragraph],
        parent: vec![None, None],
        depth: vec![0, 0],
    };

    let mut base_doc = Document::new(text);
    base_doc.add(&Key::DEFAULT, subwords, Provenance::by("test")).unwrap();
    base_doc.add(&Key::DEFAULT, sentences, Provenance::by("test")).unwrap();
    base_doc.add(&Key::DEFAULT, chunks, Provenance::by("test")).unwrap();
    base_doc.add(&Key::DEFAULT, structure, Provenance::by("test")).unwrap();

    for mode in [DlChunkingMode::WhitespaceWords, DlChunkingMode::Subwords] {
        let opts = DlInferenceOptions::default().with_chunking_mode(mode);
        let doc_cat = Arc::new(
            DocumentCategorizerDL::with_options(
                MockSequenceScorer::new(&labels, vec![0.0, 0.0])
                    .with_chunk("good news", vec![-2.0, 2.0])
                    .with_chunk("bad news", vec![2.0, -2.0])
                    .with_chunk("good news bad news", vec![-0.5, 1.5])
                    .with_chunk("10,11", vec![-2.0, 2.0])
                    .with_chunk("20,21", vec![2.0, -2.0])
                    .with_chunk("10,11,20,21", vec![-0.5, 1.5]),
                opts,
            )
            .unwrap(),
        );
        let chunk_cat = DlSpanCategorizer::<_, Chunks>::new(Arc::clone(&doc_cat));
        let struct_cat = DlSpanCategorizer::<_, Structure>::new(Arc::clone(&doc_cat));
        let sent_cat = DlSpanCategorizer::<_, Sentences>::new(Arc::clone(&doc_cat));

        let doc = base_doc.clone();

        let view = doc.get::<View>().unwrap();
        let sw = doc.get::<Subwords>();
        let (doc_cats,) = doc_cat.annotate((view, sw)).unwrap();
        assert_eq!(doc_cats.best_category(0), "pos");

        let (chunk_cats,) = chunk_cat.annotate((view, sw, doc.get::<Chunks>().unwrap())).unwrap();
        assert_eq!(chunk_cats.len(), 2);
        assert_eq!(chunk_cats.best_category(0), "pos");
        assert_eq!(chunk_cats.best_category(1), "neg");

        let (struct_cats,) = struct_cat.annotate((view, sw, doc.get::<Structure>().unwrap())).unwrap();
        assert_eq!(struct_cats.len(), 2);
        assert_eq!(struct_cats.best_category(0), "pos");
        assert_eq!(struct_cats.best_category(1), "neg");

        let (sent_cats,) = sent_cat.annotate((view, sw, doc.get::<Sentences>().unwrap())).unwrap();
        assert_eq!(sent_cats.len(), 2);
        assert_eq!(sent_cats.best_category(0), "pos");
        assert_eq!(sent_cats.best_category(1), "neg");

        // Verify Analyzer pipeline wiring with Subwords, Chunks, and Structure.
        let analyzer = Analyzer::builder()
            .given(Key::<Subwords>::DEFAULT)
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Chunks>::DEFAULT)
            .given(Key::<Structure>::DEFAULT)
            .add(chunk_cat)
            .add(struct_cat)
            .build()
            .unwrap();
        let mut docs = [doc];
        for r in analyzer.run(&mut docs) {
            r.unwrap();
        }
        assert_eq!(docs[0].get::<Categories<Chunks>>().unwrap().best_category(0), "pos");
        assert_eq!(docs[0].get::<Categories<Chunks>>().unwrap().best_category(1), "neg");
        assert_eq!(docs[0].get::<Categories<Structure>>().unwrap().best_category(0), "pos");
        assert_eq!(docs[0].get::<Categories<Structure>>().unwrap().best_category(1), "neg");
        let prov = docs[0].provenance(Key::<Categories<Chunks>>::DEFAULT.id()).unwrap();
        assert_eq!(prov.annotator, "opennlp:doccat-dl-spans");
        assert_eq!(prov.model.as_ref(), Some(&model_ref));
    }

    // Mismatched subword model reference is rejected.
    let bad_sw = Subwords {
        spans: Spans::new(ViewId::SOURCE),
        ids: vec![],
        model: ModelRef { id: "wrong/model".into(), revision: "1".into(), sha256: "00".repeat(32) },
        token: None,
    };
    let doc_cat = DocumentCategorizerDL::new(scorer).unwrap();
    let doc = Document::new(text);
    assert!(doc_cat.annotate((doc.get::<View>().unwrap(), Some(&bad_sw))).is_err());

    // Blank document in Annotator produces Tag::NONE abstention rather than failing the pipeline.
    let blank_doc = Document::new("   ");
    let (blank_cats,) = doc_cat.annotate((blank_doc.get::<View>().unwrap(), None)).unwrap();
    assert_eq!(blank_cats.category[0], Tag::NONE);
    assert_eq!(blank_cats.best_category(0), "");
}

struct WeightedFirstChunkStrategy {
    first_weight: f64,
}

impl ClassificationScoringStrategy for WeightedFirstChunkStrategy {
    fn score(&self, scores: &[Vec<f64>]) -> Result<Vec<f64>, DlDoccatError> {
        let mut out = scores[0].clone();
        for v in &mut out {
            *v *= self.first_weight;
        }
        Ok(out)
    }

    fn strategy_hash(&self) -> u64 {
        self.first_weight.to_bits()
    }
}

#[test]
fn overlapping_subwords_empty_unit_abstention_and_config_hash_fingerprints() {
    let labels = ["neg", "pos"];
    let scorer = MockSequenceScorer::new(&labels, vec![1.0, -1.0]).with_chunk("10,11,12", vec![-3.0, 3.0]);
    let model_ref = scorer.model_ref();

    // Subwords with overlapping spans [0..4, 1..4, 4..5] and IDs [10, 11, 12] in "abcde".
    // Sentence 0 is 2..5 (must select all three overlapping pieces [10, 11, 12], not just [11, 12]);
    // Sentence 1 is empty 2..2 inside piece 0..4 (must abstain with Tag::NONE, not score piece 11).
    let text = "abcde";
    let mut sub_spans = Spans::new(ViewId::SOURCE);
    sub_spans.push(Span::new(0, 4));
    sub_spans.push(Span::new(1, 4));
    sub_spans.push(Span::new(4, 5));
    let subwords = Subwords { spans: sub_spans, ids: vec![10, 11, 12], model: model_ref, token: None };

    let mut sent_spans = Spans::new(ViewId::SOURCE);
    sent_spans.push(Span::new(2, 5));
    let mut empty_sent_spans = Spans::new(ViewId::SOURCE);
    empty_sent_spans.push(Span::new(2, 2));

    let opts = DlInferenceOptions::default().with_chunking_mode(DlChunkingMode::Subwords);
    let doc_cat = Arc::new(DocumentCategorizerDL::with_options(scorer, opts.clone()).unwrap());
    let sent_cat = DlSpanCategorizer::<_, Sentences>::new(Arc::clone(&doc_cat));

    let doc = Document::new(text);
    let view = doc.get::<View>().unwrap();
    let (cats,) = sent_cat.annotate((view, Some(&subwords), &Sentences { spans: sent_spans })).unwrap();
    assert_eq!(cats.best_category(0), "pos");

    let (empty_cats,) = sent_cat.annotate((view, Some(&subwords), &Sentences { spans: empty_sent_spans })).unwrap();
    assert_eq!(empty_cats.category[0], Tag::NONE);
    assert_eq!(empty_cats.best_category(0), "");

    // Different max_content_tokens or custom strategy parameters must produce distinct config_hash values.
    let split_opts = DlInferenceOptions::default().with_document_split_size(4).with_split_overlap_size(0);
    let cat_cap2 = DocumentCategorizerDL::with_options(
        MockSequenceScorer::new(&labels, vec![0.0, 0.0]).with_max_content_tokens(2),
        split_opts.clone(),
    )
    .unwrap();
    let cat_cap4 = DocumentCategorizerDL::with_options(
        MockSequenceScorer::new(&labels, vec![0.0, 0.0]).with_max_content_tokens(4),
        split_opts.clone(),
    )
    .unwrap();
    assert_ne!(cat_cap2.config_hash(), cat_cap4.config_hash());

    let cat_strat_a = DocumentCategorizerDL::with_options_and_strategy(
        MockSequenceScorer::new(&labels, vec![0.0, 0.0]),
        split_opts.clone(),
        WeightedFirstChunkStrategy { first_weight: 0.25 },
    )
    .unwrap();
    let cat_strat_b = DocumentCategorizerDL::with_options_and_strategy(
        MockSequenceScorer::new(&labels, vec![0.0, 0.0]),
        split_opts,
        WeightedFirstChunkStrategy { first_weight: 0.75 },
    )
    .unwrap();
    assert_ne!(cat_strat_a.config_hash(), cat_strat_b.config_hash());
}
