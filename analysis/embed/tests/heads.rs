//! Integration tests for `centroids` and `heads` (#321):
//!
//! 1. Exact parity with `org.apache.opennlp.grpc.chunk.CentroidsTest` and
//!    `ChunkEmbedProcessor.buildCategoryGroup` (`OPENNLP-1833`).
//! 2. `LabelCentroids` built from example texts (`from_text_examples`) and
//!    vector slices (`from_vector_examples`), binary `TCENT001` round-trip,
//!    and `VectorHeadAnnotator` categorization over `tiny-bert-bundle`.
//! 3. `ClassifierHead` 1-layer linear and 2-layer MLP (`Relu`, `Gelu`, `Tanh`)
//!    construction and fitting (`fit_single_label`, `fit_multi_label`),
//!    binary `THEAD001` round-trip, and deterministic `f64` forward evaluation.
//! 4. `SentenceSpanHead`, `ChunkSpanHead`, `OpinionSpanHead`, `AspectSpanHead`
//!    (with `score_aspects_polarity`), `IncivilityHeadAnnotator`, and
//!    `PragmaticsHeadAnnotator`, verifying that multiple opinion and aspect
//!    spans inside a single sentence receive distinct span-pooled embeddings
//!    and distinct polarity predictions.
//! 5. Sentiment classification comparison between `LexiconSentiment` (#320),
//!    `LabelCentroids` (#321), and `ClassifierHead` (#321) on a self-contained
//!    benchmark and on MASC 3.0.0 MPQA opinion annotations (`CC BY 3.0 US`)
//!    when `TURBO_MASC_ROOT` names the corpus' `data` directory and
//!    `TURBO_MINILM_BUNDLE` a sentence-transformers/all-MiniLM-L6-v2 bundle.
//! 6. CPU and GPU (`TURBO_TEST_DEVICE` / `TURBO_TEST_REQUIRE_CUDA=1`) parity
//!    for both `LabelCentroids` and `ClassifierHead`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use turbo_analysis::*;
use turbo_analysis_embed::*;
use turbo_analysis_layers::{
    Aspect, AspectTarget, Aspects, Categories, Chunks, Cut, Incivility, IncivilityLabel, Opinion, Opinions, Polarity,
    PolarityScores, Pragmatics, Sentences, Subwords, Tokens, Vectors,
};
use turbo_doccat::sentiment::LexiconSentiment;
use turbo_tokenize::SimpleTokenizer;

fn testdata() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata")
}

fn bert() -> PathBuf {
    testdata().join("tiny-bert-bundle")
}

fn open_subwords() -> SubwordAnnotator {
    SubwordAnnotator::open(&bert()).unwrap_or_else(|e| panic!("{e}"))
}

fn open_cpu(max_seq: u32) -> (SubwordAnnotator, Arc<VectorAnnotator>) {
    let sub = open_subwords();
    let o =
        VectorOptions { encoder: EncoderOptions { max_seq, ..EncoderOptions::default() }, ..VectorOptions::default() };
    let vec = Arc::new(VectorAnnotator::open(&bert(), &o).unwrap_or_else(|e| panic!("{e}")));
    (sub, vec)
}

fn make_doc_with_sentences(text: &str, sentences: &[(u32, u32)]) -> Document {
    let mut doc = Document::new(text.to_string());
    let mut spans = Spans::new(ViewId::SOURCE);
    for &(start, end) in sentences {
        spans.push(Span::new(start, end));
    }
    doc.add(&Key::DEFAULT, Sentences { spans }, Provenance::by("test")).unwrap();
    doc
}

fn embed_single_sentence(vec_ann: &Arc<VectorAnnotator>, text: &str) -> Vec<f32> {
    let a = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .add(open_subwords())
        .add(SentenceVectors(Arc::clone(vec_ann)))
        .build()
        .unwrap();
    let mut docs = [make_doc_with_sentences(text, &[(0, text.len() as u32)])];
    a.run(&mut docs).remove(0).unwrap();
    docs[0].get::<Vectors<Sentences>>().unwrap().row(Ix::new(0)).to_vec()
}

#[test]
fn centroids_match_opennlp_grpc_centroids_test() {
    // 1. meanWithoutNorm: [[1, 3, 5], [3, 5, 7]] -> [2, 4, 6]
    let v = [1.0f32, 3.0, 5.0];
    let w = [3.0f32, 5.0, 7.0];
    let c = centroid(&[&v[..], &w[..]], CentroidNorm::None).unwrap().unwrap();
    assert_eq!(c, vec![2.0f32, 4.0, 6.0]);

    // 2. meanWithL2Norm: [[3, 0], [3, 4]] -> mean [3, 2], norm sqrt(13)
    let a = [3.0f32, 0.0];
    let b = [3.0f32, 4.0];
    let c = centroid(&[&a[..], &b[..]], CentroidNorm::L2).unwrap().unwrap();
    let n = 13.0f64.sqrt();
    assert!((c[0] as f64 - 3.0 / n).abs() < 1e-6);
    assert!((c[1] as f64 - 2.0 / n).abs() < 1e-6);
    let norm_sq: f64 = c.iter().map(|&x| (x as f64) * (x as f64)).sum();
    assert!((norm_sq - 1.0).abs() < 1e-6);

    // 3. zeroVectorWithL2Norm is rejected with a clear error (no NaN)
    let z = [0.0f32, 0.0, 0.0];
    assert!(centroid(&[&z[..], &z[..]], CentroidNorm::L2).is_err());
    assert_eq!(centroid(&[&z[..], &z[..]], CentroidNorm::None).unwrap().unwrap(), vec![0.0f32, 0.0, 0.0]);

    // 4. empty input returns Ok(None)
    let empty: [&[f32]; 0] = [];
    assert_eq!(centroid(&empty, CentroidNorm::None).unwrap(), None);

    // 5. rejectsMismatchedDimensions
    let d2 = [1.0f32, 2.0];
    let d3 = [1.0f32, 2.0, 3.0];
    assert!(centroid(&[&d2[..], &d3[..]], CentroidNorm::None).is_err());
}

#[test]
fn category_sentence_centroids_matches_chunk_embed_processor_group() {
    let text = "First positive line. Second negative line. Third positive line.";
    let s0 = (0, 20);
    let s1 = (21, 42);
    let s2 = (43, 63);
    let doc = make_doc_with_sentences(text, &[s0, s1, s2]);
    let sentences = doc.get::<Sentences>().unwrap();

    let model = ModelRef { id: "m".into(), revision: "1".into(), sha256: "0".into() };
    let vecs = Vectors::<Sentences>::new(
        4,
        vec![
            1.0, 0.0, 0.0, 0.0, // s0
            0.0, 1.0, 0.0, 0.0, // s1
            0.0, 0.0, 1.0, 0.0, // s2
        ],
        Vec::new(),
        model,
    );

    let tags = TagSet::new("sentiment", &["positive", "negative"]);
    let mut cats = Categories::<Sentences>::with_capacity(Arc::clone(&tags), 3);
    cats.push(Tag(0), &[0.9, 0.1]);
    cats.push(Tag(1), &[0.2, 0.8]);
    cats.push(Tag(0), &[0.85, 0.15]);

    let grouped = category_sentence_centroids(sentences, &cats, &vecs, None, CentroidNorm::L2).unwrap();
    assert_eq!(grouped.len(), 2);
    assert_eq!(grouped[0].0, Tag(0));
    assert_eq!(grouped[0].1, vec![Ix::new(0), Ix::new(2)]);
    assert_eq!(grouped[0].2, Span::new(0, 63));
    assert_eq!(grouped[1].0, Tag(1));
    assert_eq!(grouped[1].1, vec![Ix::new(1)]);
    assert_eq!(grouped[1].2, Span::new(21, 42));

    let inv_sqrt2 = (0.5f64).sqrt() as f32;
    assert!((grouped[0].3[0] - inv_sqrt2).abs() < 1e-6);
    assert!((grouped[0].3[2] - inv_sqrt2).abs() < 1e-6);
    assert_eq!(grouped[1].3, vec![0.0, 1.0, 0.0, 0.0]);
}

#[test]
fn label_centroids_from_text_examples_and_binary_roundtrip() {
    let (sub, vec_ann) = open_cpu(64);
    let pos_examples = [
        "We loved the friendly staff and the delicious breakfast.",
        "This camera takes crisp photos and works wonderfully.",
        "A delightful book with warm characters and a joyful ending.",
    ];
    let neg_examples = [
        "We hated the rude service and the cold soggy meal.",
        "This camera crashes constantly and wastes battery.",
        "A boring story with clumsy dialogue and a dreadful ending.",
    ];
    let centroids = LabelCentroids::from_text_examples(
        &vec_ann,
        &sub,
        "sentiment",
        &[("positive", &pos_examples[..]), ("negative", &neg_examples[..])],
        CentroidNorm::L2,
        0.1,
    )
    .unwrap();
    assert_eq!(centroids.dim(), 32);
    assert_eq!(centroids.tags().len(), 2);
    assert_eq!(centroids.tags().label(Tag(0)), "positive");
    assert_eq!(centroids.tags().label(Tag(1)), "negative");

    let bytes = centroids.to_bytes();
    let decoded = LabelCentroids::from_bytes(&bytes).unwrap();
    assert_eq!(centroids, decoded);

    // Run VectorHeadAnnotator<Sentences> with LabelCentroids through an Analyzer.
    let head_step = Bound::new(VectorHeadAnnotator::<Sentences>::new(decoded)).write(0, "sentiment");
    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .add(sub)
        .add(SentenceVectors(Arc::clone(&vec_ann)))
        .step(head_step)
        .build()
        .unwrap();

    let mut docs = [make_doc_with_sentences(pos_examples[0], &[(0, pos_examples[0].len() as u32)])];
    analyzer.run(&mut docs).remove(0).unwrap();
    let cats = docs[0].get_key::<Categories<Sentences>>(&Key::named("sentiment")).unwrap();
    assert_eq!(cats.len(), 1);
    assert_eq!(cats.best_category(0), "positive");
    let scores = cats.row_scores(0);
    assert!((scores[0] + scores[1] - 1.0).abs() < 1e-6);
}

#[test]
fn classifier_head_linear_and_mlp_fit_and_binary_roundtrip() {
    let (_sub, vec_ann) = open_cpu(64);
    let pos_texts = [
        "I love this wonderful film and its brilliant cast.",
        "The meal was fresh, flavorful, and thoroughly enjoyable.",
        "Helpful support resolved our question quickly and politely.",
        "Smooth performance and reliable battery life all week.",
    ];
    let neg_texts = [
        "I hate this terrible film and its awful script.",
        "The meal was stale, tasteless, and completely disappointing.",
        "Rude support ignored our question and wasted our afternoon.",
        "Sluggish performance and broken battery after one day.",
    ];

    let mut examples: Vec<(Vec<f32>, Tag)> = Vec::new();
    for (label_idx, group) in [&pos_texts[..], &neg_texts[..]].iter().enumerate() {
        for text in *group {
            let v = embed_single_sentence(&vec_ann, text);
            examples.push((v, Tag(label_idx as u16)));
        }
    }

    let tags = TagSet::new("sentiment", &["positive", "negative"]);
    let model = vec_ann.model().clone();

    // 1. Fit a 1-layer linear ClassifierHead.
    let linear =
        ClassifierHead::fit_single_label(model.clone(), 32, Arc::clone(&tags), &examples, 1e-4, 150, 1.5).unwrap();
    let bytes = linear.to_bytes();
    let linear_rt = ClassifierHead::from_bytes(&bytes).unwrap();
    assert_eq!(linear, linear_rt);

    let mut probs = [0.0f64; 2];
    for (v, want_tag) in &examples {
        let (got_tag, _) = linear_rt.score(v, &mut probs).unwrap();
        assert_eq!(got_tag, *want_tag);
    }

    // 2. Construct 2-layer MLP ClassifierHeads with Relu, Gelu, and Tanh from
    // the fitted linear weights by routing positive and negative channels through
    // the hidden units, and verify binary round-trip and predictions.
    let l1_linear = &linear.layers()[0];
    for act in [HiddenActivation::Relu, HiddenActivation::Gelu, HiddenActivation::Tanh] {
        let w1 = l1_linear.weights.clone();
        let b1 = l1_linear.bias.clone();
        let w2 = vec![
            2.0, -2.0, // class 0
            -2.0, 2.0, // class 1
        ];
        let b2 = vec![0.0, 0.0];
        let mlp =
            ClassifierHead::mlp(model.clone(), 32, 2, Arc::clone(&tags), HeadMode::SingleLabel, act, w1, b1, w2, b2)
                .unwrap();
        let mlp_rt = ClassifierHead::from_bytes(&mlp.to_bytes()).unwrap();
        assert_eq!(mlp, mlp_rt);
        for (i, (v, want_tag)) in examples.iter().enumerate() {
            let (got_tag, _) = mlp_rt.score(v, &mut probs).unwrap();
            assert_eq!(got_tag, *want_tag, "{act:?} sample {i}");
        }
    }

    // 3. Fit a multi-label head and verify independent sigmoid probabilities.
    let multi_examples: Vec<(&[f32], [bool; 2])> = examples
        .iter()
        .map(|(v, t)| (v.as_slice(), if *t == Tag(0) { [true, false] } else { [false, true] }))
        .collect();
    let multi = ClassifierHead::fit_multi_label(model, 32, Arc::clone(&tags), &multi_examples, 1e-5, 400, 2.5).unwrap();
    assert_eq!(multi.mode(), HeadMode::MultiLabel);
    for (v, want_tag) in &examples {
        multi.score(v, &mut probs).unwrap();
        if *want_tag == Tag(0) {
            assert!(probs[0] > 0.7 && probs[1] < 0.3, "{probs:?}");
        } else {
            assert!(probs[1] > 0.7 && probs[0] < 0.3, "{probs:?}");
        }
    }
}

#[test]
fn vector_head_and_span_heads_write_expected_layers() {
    let (_sub, vec_ann) = open_cpu(64);

    // Mixed-polarity sentence containing two distinct opinion spans and two aspects.
    // Tokens from SimpleTokenizer on "I love the camera, though I hate the battery.":
    // 0: "I" (0..1)
    // 1: "love" (2..6)
    // 2: "the" (7..10)
    // 3: "camera" (11..17)
    // 4: "," (17..18)
    // 5: "though" (19..25)
    // 6: "I" (26..27)
    // 7: "hate" (28..32)
    // 8: "the" (33..36)
    // 9: "battery" (37..44)
    // 10: "." (44..45)
    let text = "I love the camera, though I hate the battery.";
    let mut doc = make_doc_with_sentences(text, &[(0, text.len() as u32)]);
    let mut c = Chunks {
        spans: Spans::new(ViewId::SOURCE),
        level: vec![0],
        parent: vec![None],
        sentences: vec![IxRange::new(0, 1)],
        cut: vec![Cut::Sentence],
        cost: vec![0],
        model_tokens: vec![0],
        over_budget: vec![false],
    };
    c.spans.push(Span::new(0, text.len() as u32));
    doc.add(&Key::DEFAULT, c, Provenance::by("test")).unwrap();

    let prep = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Chunks>::DEFAULT)
        .add(SimpleTokenizer::default())
        .add(open_subwords())
        .add(SentenceChunkVectors(Arc::clone(&vec_ann)))
        .build()
        .unwrap();
    let mut docs = [doc];
    prep.run(&mut docs).remove(0).unwrap();
    let mut doc = docs.into_iter().next().unwrap();

    let (opinions, aspects) = {
        let tokens = doc.get::<Tokens>().unwrap();
        assert_eq!(tokens.len(), 11);
        let sentences = doc.get::<Sentences>().unwrap();

        let opinions = Opinions::from_items(
            tokens,
            [
                Opinion {
                    tokens: IxRange::new(0, 4), // "I love the camera"
                    polarity: 0.0,
                    intensity: 1.0,
                    negated: false,
                },
                Opinion {
                    tokens: IxRange::new(6, 10), // "I hate the battery"
                    polarity: 0.0,
                    intensity: 1.0,
                    negated: false,
                },
            ],
        )
        .unwrap();
        let aspect_cats = TagSet::new("aspect-cat", &["hardware"]);
        let op0 = [Ix::<Opinions>::new(0)];
        let op1 = [Ix::<Opinions>::new(1)];
        let aspects = Aspects::from_items(
            tokens,
            Some(sentences),
            Some(&opinions),
            Arc::clone(&aspect_cats),
            [
                Aspect {
                    target: AspectTarget::Term {
                        tokens: IxRange::new(3, 4), // "camera"
                        sentence: Some(Ix::new(0)),
                    },
                    category: Some(Tag(0)),
                    polarity: PolarityScores::new(0.0, 0.0, 1.0),
                    opinions: &op0,
                },
                Aspect {
                    target: AspectTarget::Term {
                        tokens: IxRange::new(9, 10), // "battery"
                        sentence: Some(Ix::new(0)),
                    },
                    category: Some(Tag(0)),
                    polarity: PolarityScores::new(0.0, 0.0, 1.0),
                    opinions: &op1,
                },
            ],
        )
        .unwrap();
        (opinions, aspects)
    };
    doc.add(&Key::DEFAULT, opinions.clone(), Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, aspects.clone(), Provenance::by("test")).unwrap();

    // Embed the two opinion spans and the two aspect spans via `embed_spans`
    // to verify that two spans inside the same sentence receive distinct
    // span-pooled embeddings.
    let (opinion_vecs, aspect_vecs) = {
        let tokens = doc.get::<Tokens>().unwrap();
        let sentences = doc.get::<Sentences>().unwrap();
        let subwords = doc.get::<Subwords>().unwrap();
        let op_spans = [opinions.span(Ix::new(0), tokens), opinions.span(Ix::new(1), tokens)];
        let opinion_vecs: Vectors<Opinions> = vec_ann
            .embed_spans("test:opinions", subwords, Some(&sentences.spans), tokens.spans.view(), &op_spans)
            .unwrap();
        assert_eq!(opinion_vecs.len(), 2);
        assert_ne!(opinion_vecs.row(Ix::new(0)), opinion_vecs.row(Ix::new(1)));

        let asp_spans = [
            aspects.span(Ix::new(0), tokens, Some(sentences)).unwrap(),
            aspects.span(Ix::new(1), tokens, Some(sentences)).unwrap(),
        ];
        let aspect_vecs: Vectors<Aspects> = vec_ann
            .embed_spans("test:aspects", subwords, Some(&sentences.spans), tokens.spans.view(), &asp_spans)
            .unwrap();
        assert_eq!(aspect_vecs.len(), 2);
        assert_ne!(aspect_vecs.row(Ix::new(0)), aspect_vecs.row(Ix::new(1)));
        (opinion_vecs, aspect_vecs)
    };

    // Fit an opinion polarity head on the positive and negative span vectors
    // and an aspect polarity head on the camera and battery span vectors.
    let pol_tags = TagSet::new("sentiment", &["positive", "negative", "neutral"]);
    let op_head = ClassifierHead::fit_single_label(
        vec_ann.model().clone(),
        32,
        Arc::clone(&pol_tags),
        &[(opinion_vecs.row(Ix::new(0)), Tag(0)), (opinion_vecs.row(Ix::new(1)), Tag(1))],
        1e-4,
        100,
        2.0,
    )
    .unwrap();
    let asp_head = ClassifierHead::fit_single_label(
        vec_ann.model().clone(),
        32,
        Arc::clone(&pol_tags),
        &[(aspect_vecs.row(Ix::new(0)), Tag(0)), (aspect_vecs.row(Ix::new(1)), Tag(1))],
        1e-4,
        100,
        2.0,
    )
    .unwrap();

    // Build multi-label Incivility and Pragmatics heads.
    let inciv_names = IncivilityLabel::ALL.map(|l| l.name());
    let inciv_tags = TagSet::new("incivility", &inciv_names);
    let inciv_head = ClassifierHead::linear(
        vec_ann.model().clone(),
        32,
        inciv_tags,
        HeadMode::MultiLabel,
        vec![0.0; 6 * 32],
        vec![-1.0, -2.0, -3.0, -2.5, -2.0, -3.5],
    )
    .unwrap();
    let prag_tags = TagSet::new("pragmatics", &["sarcasm", "irony"]);
    let prag_head = ClassifierHead::linear(
        vec_ann.model().clone(),
        32,
        prag_tags,
        HeadMode::MultiLabel,
        vec![0.0; 2 * 32],
        vec![0.8, -0.8],
    )
    .unwrap();

    let sent_head = SentenceSpanHead::new(Arc::clone(&vec_ann), op_head.clone());
    let chunk_head = ChunkSpanHead::new(Arc::clone(&vec_ann), op_head.clone());
    let opinion_span_head = OpinionSpanHead::new(Arc::clone(&vec_ann), op_head);
    let aspect_span_head = AspectSpanHead::new(Arc::clone(&vec_ann), asp_head);
    let inciv_ann = IncivilityHeadAnnotator(Arc::new(inciv_head));
    let prag_ann = PragmaticsHeadAnnotator(Arc::new(prag_head));

    let heads_pipeline = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Chunks>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Subwords>::DEFAULT)
        .given(Key::<Vectors<Sentences>>::DEFAULT)
        .given(Key::<Vectors<Chunks>>::DEFAULT)
        .given(Key::<Opinions>::DEFAULT)
        .given(Key::<Aspects>::DEFAULT)
        .add(sent_head)
        .add(chunk_head)
        .add(opinion_span_head)
        .add(aspect_span_head.clone())
        .add(inciv_ann)
        .add(prag_ann)
        .build()
        .unwrap();

    let mut docs = [doc];
    heads_pipeline.run(&mut docs).remove(0).unwrap();
    let doc = &docs[0];

    // Verify opinion spans received opposite polarity predictions inside the same sentence.
    let op_cats = doc.get::<Categories<Opinions>>().unwrap();
    assert_eq!(op_cats.len(), 2);
    assert_eq!(op_cats.best_category(0), "positive");
    assert_eq!(op_cats.best_category(1), "negative");

    // Verify aspect spans received opposite polarity predictions and populate `Aspects`.
    let asp_cats = doc.get::<Categories<Aspects>>().unwrap();
    assert_eq!(asp_cats.len(), 2);
    assert_eq!(asp_cats.best_category(0), "positive");
    assert_eq!(asp_cats.best_category(1), "negative");

    let scored_aspects = aspect_span_head
        .score_aspects_polarity(
            doc.get::<Subwords>().unwrap(),
            doc.get::<Tokens>().unwrap(),
            Some(doc.get::<Sentences>().unwrap()),
            Some(&opinions),
            &aspects,
        )
        .unwrap();
    assert_eq!(scored_aspects.get(Ix::new(0)).polarity.label(), Polarity::Positive);
    assert_eq!(scored_aspects.get(Ix::new(1)).polarity.label(), Polarity::Negative);

    // Verify sentence and chunk categories and Incivility/Pragmatics layers were populated.
    assert_eq!(doc.get::<Categories<Sentences>>().unwrap().len(), 1);
    assert_eq!(doc.get::<Categories<Chunks>>().unwrap().len(), 1);
    let inciv = doc.get::<Incivility>().unwrap();
    assert_eq!(inciv.len(), 1);
    assert!(inciv.scores(Ix::new(0)).insult > 0.2);
    let prag = doc.get::<Pragmatics>().unwrap();
    assert_eq!(prag.len(), 1);
    assert!(prag.get(Ix::new(0)).score > 0.6);
}

/// Self-contained sentiment evaluation benchmark comparing `LexiconSentiment` (#320),
/// `LabelCentroids` (#321), and `ClassifierHead` (#321), plus MASC 3.0.0 MPQA
/// opinion evaluation when `TURBO_MASC_ROOT` and `TURBO_MINILM_BUNDLE` are set.
#[test]
fn sentiment_comparison_against_lexicon_and_masc_mpqa() {
    let (_sub, vec_ann) = open_cpu(64);

    let train_data: &[(&str, usize)] = &[
        ("The service was friendly, attentive, and genuinely welcoming.", 0),
        ("We loved the crisp screen and the fast camera.", 0),
        ("A brilliant performance with warm humor and heartfelt moments.", 0),
        ("Fresh ingredients and rich flavors made dinner wonderful.", 0),
        ("Reliable software that saves time and works smoothly.", 0),
        ("The hotel room was spotless, quiet, and comfortable.", 0),
        ("Excellent build quality and responsive customer support.", 0),
        ("An inspiring book that kept us smiling from start to finish.", 0),
        ("The service was rude, careless, and completely unhelpful.", 1),
        ("We hated the blurry screen and the sluggish camera.", 1),
        ("A tedious performance with awkward dialogue and dull pacing.", 1),
        ("Stale ingredients and bland flavors ruined dinner completely.", 1),
        ("Buggy software that crashes often and loses data.", 1),
        ("The hotel room was dirty, noisy, and uncomfortable.", 1),
        ("Flimsy build quality and terrible customer support.", 1),
        ("A dreary book that left us frustrated from start to finish.", 1),
    ];

    // Held-out sentences that recombine the sentiment vocabulary in new combinations
    // so both the synthetic 32-dim `tiny-bert-bundle` test weights and pretrained
    // models can generalize across sentences.
    let eval_data: &[(&str, usize)] = &[
        ("Friendly customer support and reliable software made work wonderful.", 0),
        ("We loved the spotless hotel room and the comfortable dinner.", 0),
        ("A crisp camera with excellent build quality and smooth performance.", 0),
        ("Warm humor and an inspiring performance kept us smiling.", 0),
        ("Fresh flavors and attentive service made the evening welcoming.", 0),
        ("Quiet rooms and responsive support made our stay comfortable.", 0),
        ("Rude customer support and buggy software ruined work completely.", 1),
        ("We hated the dirty hotel room and the stale dinner.", 1),
        ("A blurry camera with flimsy build quality and sluggish performance.", 1),
        ("Awkward dialogue and a dreary performance left us frustrated.", 1),
        ("Bland flavors and careless service made the evening terrible.", 1),
        ("Noisy rooms and unhelpful support made our stay uncomfortable.", 1),
    ];

    let embed_items = |va: &Arc<VectorAnnotator>, sub_path: &Path, items: &[(&str, usize)]| -> Vec<(Vec<f32>, Tag)> {
        let sub = SubwordAnnotator::open(sub_path).unwrap();
        let a = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .add(sub)
            .add(SentenceVectors(Arc::clone(va)))
            .build()
            .unwrap();
        items
            .iter()
            .map(|&(text, label)| {
                let mut docs = [make_doc_with_sentences(text, &[(0, text.len() as u32)])];
                a.run(&mut docs).remove(0).unwrap();
                let v = docs[0].get::<Vectors<Sentences>>().unwrap().row(Ix::new(0)).to_vec();
                (v, Tag(label as u16))
            })
            .collect()
    };

    let train_vecs = embed_items(&vec_ann, &bert(), train_data);
    let eval_vecs = embed_items(&vec_ann, &bert(), eval_data);
    let dim = vec_ann.dim();

    let pos_vecs: Vec<&[f32]> = train_vecs.iter().filter(|(_, t)| *t == Tag(0)).map(|(v, _)| v.as_slice()).collect();
    let neg_vecs: Vec<&[f32]> = train_vecs.iter().filter(|(_, t)| *t == Tag(1)).map(|(v, _)| v.as_slice()).collect();

    let tags = TagSet::new("sentiment", &["positive", "negative"]);
    let centroids = LabelCentroids::from_vector_examples(
        vec_ann.model().clone(),
        dim,
        "sentiment",
        &[("positive", &pos_vecs[..]), ("negative", &neg_vecs[..])],
        CentroidNorm::L2,
        0.1,
    )
    .unwrap();

    let linear_head =
        ClassifierHead::fit_single_label(vec_ann.model().clone(), dim, Arc::clone(&tags), &train_vecs, 1e-4, 250, 1.5)
            .unwrap();

    let lexicon = LexiconSentiment::new();
    let mut probs = [0.0f64; 2];

    let mut lex_correct = 0usize;
    let mut cent_correct = 0usize;
    let mut lin_correct = 0usize;

    for (i, &(text, want)) in eval_data.iter().enumerate() {
        let want_tag = Tag(want as u16);
        let (v, _) = &eval_vecs[i];
        if centroids.score_single_label(v, &mut probs).unwrap().0 == want_tag {
            cent_correct += 1;
        }
        if linear_head.score(v, &mut probs).unwrap().0 == want_tag {
            lin_correct += 1;
        }
        let vscore = lexicon.score_text(text);
        let lex_tag = if vscore.compound >= 0.0 { Tag(0) } else { Tag(1) };
        if lex_tag == want_tag {
            lex_correct += 1;
        }
    }

    println!(
        "Self-contained eval ({} sentences): Lexicon={}/{}, Centroids={}/{}, LinearHead={}/{}",
        eval_data.len(),
        lex_correct,
        eval_data.len(),
        cent_correct,
        eval_data.len(),
        lin_correct,
        eval_data.len(),
    );
    assert!(cent_correct >= 10, "centroids accuracy {cent_correct}/{}", eval_data.len());
    assert!(lin_correct >= 10, "linear head accuracy {lin_correct}/{}", eval_data.len());

    // Optional MASC 3.0.0 MPQA corpus evaluation: TURBO_MASC_ROOT is the corpus'
    // `data` directory and TURBO_MINILM_BUNDLE an all-MiniLM-L6-v2 bundle.
    let masc_root = std::env::var_os("TURBO_MASC_ROOT").map(PathBuf::from);
    let minilm_path = std::env::var_os("TURBO_MINILM_BUNDLE").map(PathBuf::from);
    if let (Some(masc_root), Some(minilm_path)) = (masc_root, minilm_path)
        && masc_root.is_dir()
    {
        let samples = load_masc_mpqa_samples(&masc_root);
        assert!(samples.len() >= 200, "expected >= 200 MASC MPQA samples, got {}", samples.len());
        let split = (samples.len() * 4) / 5;
        let (m_train, m_test) = samples.split_at(split);

        let use_cuda = std::env::var("TURBO_TEST_REQUIRE_CUDA").as_deref() == Ok("1");
        let o = VectorOptions {
            encoder: EncoderOptions {
                device: if use_cuda { Device::Backend("cuda".into()) } else { Device::Cpu },
                max_seq: 64,
                ..EncoderOptions::default()
            },
            ..VectorOptions::default()
        };
        let (eval_bundle_path, m_vec_ann) = if minilm_path.is_dir()
            && let Ok(va) = VectorAnnotator::open(&minilm_path, &o)
        {
            (minilm_path.clone(), Arc::new(va))
        } else {
            (bert(), Arc::clone(&vec_ann))
        };
        let m_dim = m_vec_ann.dim();
        let m_sub = SubwordAnnotator::open(&eval_bundle_path).unwrap();
        let m_pipe = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .add(m_sub)
            .add(SentenceVectors(Arc::clone(&m_vec_ann)))
            .build()
            .unwrap();
        let embed_masc = |slice: &[(String, usize)]| -> Vec<(Vec<f32>, Tag)> {
            let mut docs: Vec<Document> =
                slice.iter().map(|(t, _)| make_doc_with_sentences(t, &[(0, t.len() as u32)])).collect();
            for r in m_pipe.run(&mut docs) {
                r.unwrap();
            }
            docs.iter()
                .zip(slice)
                .map(|(d, (_, l))| {
                    let v = d.get::<Vectors<Sentences>>().unwrap().row(Ix::new(0)).to_vec();
                    (v, Tag(*l as u16))
                })
                .collect()
        };

        let m_train_vecs = embed_masc(m_train);
        let m_test_vecs = embed_masc(m_test);

        let m_pos: Vec<&[f32]> = m_train_vecs.iter().filter(|(_, t)| *t == Tag(0)).map(|(v, _)| v.as_slice()).collect();
        let m_neg: Vec<&[f32]> = m_train_vecs.iter().filter(|(_, t)| *t == Tag(1)).map(|(v, _)| v.as_slice()).collect();

        let m_centroids = LabelCentroids::from_vector_examples(
            m_vec_ann.model().clone(),
            m_dim,
            "sentiment",
            &[("positive", &m_pos[..]), ("negative", &m_neg[..])],
            CentroidNorm::L2,
            0.1,
        )
        .unwrap();
        let m_linear = ClassifierHead::fit_single_label(
            m_vec_ann.model().clone(),
            m_dim,
            Arc::clone(&tags),
            &m_train_vecs,
            1e-4,
            300,
            1.5,
        )
        .unwrap();

        let mut m_lex_ok = 0usize;
        let mut m_cent_ok = 0usize;
        let mut m_lin_ok = 0usize;
        for (i, (text, want)) in m_test.iter().enumerate() {
            let want_tag = Tag(*want as u16);
            let (v, _) = &m_test_vecs[i];
            if m_centroids.score_single_label(v, &mut probs).unwrap().0 == want_tag {
                m_cent_ok += 1;
            }
            if m_linear.score(v, &mut probs).unwrap().0 == want_tag {
                m_lin_ok += 1;
            }
            let vscore = lexicon.score_text(text);
            let lex_tag = if vscore.compound >= 0.0 { Tag(0) } else { Tag(1) };
            if lex_tag == want_tag {
                m_lex_ok += 1;
            }
        }
        let n = m_test.len();
        println!(
            "MASC 3.0.0 MPQA polarity spans on {} (dim={}, train={}, test={}): Lexicon={}/{} ({:.1}%), Centroids={}/{} ({:.1}%), LinearHead={}/{} ({:.1}%)",
            m_vec_ann.model().id,
            m_dim,
            m_train.len(),
            n,
            m_lex_ok,
            n,
            100.0 * m_lex_ok as f64 / n as f64,
            m_cent_ok,
            n,
            100.0 * m_cent_ok as f64 / n as f64,
            m_lin_ok,
            n,
            100.0 * m_lin_ok as f64 / n as f64,
        );
    }
}

/// Extracts `(span_text, label)` pairs (`0 = positive`, `1 = negative`) from
/// MASC 3.0.0 `*-mpqa.xml` GrAF standoff files paired with their `.txt` files.
fn load_masc_mpqa_samples(root: &Path) -> Vec<(String, usize)> {
    let mut mpqa_files = Vec::new();
    collect_mpqa_files(root, &mut mpqa_files);
    mpqa_files.sort();

    let mut samples = Vec::new();
    for mpqa_path in mpqa_files {
        let stem = mpqa_path.file_name().unwrap().to_str().unwrap().strip_suffix("-mpqa.xml").unwrap();
        let txt_path = mpqa_path.with_file_name(format!("{stem}.txt"));
        let Ok(raw_txt) = std::fs::read_to_string(&txt_path) else { continue };
        let Ok(xml) = std::fs::read_to_string(&mpqa_path) else { continue };
        let char_to_byte: Vec<usize> =
            raw_txt.char_indices().map(|(b, _)| b).chain(std::iter::once(raw_txt.len())).collect();

        let mut regions: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        let mut nodes: BTreeMap<String, Vec<String>> = BTreeMap::new();

        for line in xml.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("<region ")
                && let (Some(id), Some(anchors)) = (attr_val(trimmed, "xml:id"), attr_val(trimmed, "anchors"))
            {
                let mut parts = anchors.split_whitespace();
                if let (Some(s), Some(e)) = (parts.next(), parts.next())
                    && let (Ok(cs), Ok(ce)) = (s.parse::<usize>(), e.parse::<usize>())
                    && cs < ce
                    && ce < char_to_byte.len()
                {
                    regions.insert(id.to_string(), (char_to_byte[cs], char_to_byte[ce]));
                }
            }
        }

        let mut cur_node: Option<String> = None;
        let mut cur_ref: Option<String> = None;
        let mut cur_polarity: Option<usize> = None;
        for line in xml.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("<node ") {
                cur_node = attr_val(trimmed, "xml:id").map(str::to_string);
            } else if trimmed.starts_with("<link ") {
                if let (Some(nid), Some(targets)) = (&cur_node, attr_val(trimmed, "targets")) {
                    nodes.insert(nid.clone(), targets.split_whitespace().map(str::to_string).collect());
                }
            } else if trimmed == "</node>" {
                cur_node = None;
            } else if trimmed.starts_with("<a ") {
                let label = attr_val(trimmed, "label").unwrap_or("");
                if matches!(label, "direct-subjective" | "expressive-subjectivity" | "attitude") {
                    cur_ref = attr_val(trimmed, "ref").map(str::to_string);
                    cur_polarity = None;
                } else {
                    cur_ref = None;
                }
            } else if trimmed.starts_with("<f ") && cur_ref.is_some() {
                if attr_val(trimmed, "name") == Some("polarity") {
                    cur_polarity = match attr_val(trimmed, "value") {
                        Some("positive") => Some(0),
                        Some("negative") => Some(1),
                        _ => None,
                    };
                }
            } else if trimmed == "</a>"
                && let (Some(nref), Some(pol)) = (cur_ref.take(), cur_polarity.take())
                && let Some(rids) = nodes.get(&nref)
            {
                for rid in rids {
                    if let Some(&(bs, be)) = regions.get(rid) {
                        let snippet = raw_txt[bs..be].trim();
                        if snippet.len() >= 12 && snippet.len() <= 240 && snippet.contains(' ') {
                            samples.push((snippet.to_string(), pol));
                        }
                    }
                }
            }
        }
    }
    samples
}

fn collect_mpqa_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect_mpqa_files(&p, out);
        } else if p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.ends_with("-mpqa.xml")) {
            out.push(p);
        }
    }
}

fn attr_val<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let pat = format!("{key}=\"");
    let start = line.find(&pat)? + pat.len();
    let rest = &line[start..];
    let end = rest.find('"')?;
    Some(&rest[..end])
}

#[test]
fn heads_and_centroids_cpu_and_device_parity() {
    let want = match std::env::var("TURBO_TEST_DEVICE") {
        Ok(w) => w,
        Err(_) if std::env::var("TURBO_TEST_REQUIRE_CUDA").as_deref() == Ok("1") => "cuda".to_string(),
        Err(_) => {
            println!("skipped: neither TURBO_TEST_DEVICE nor TURBO_TEST_REQUIRE_CUDA=1 is set");
            return;
        }
    };
    let device = match want.parse() {
        Ok(i) => Device::Index(i),
        Err(_) => Device::Backend(want.clone()),
    };
    let (sub, cpu_vec) = open_cpu(64);
    let dev_opts = VectorOptions {
        encoder: EncoderOptions { device, max_seq: 64, ..EncoderOptions::default() },
        ..VectorOptions::default()
    };
    let dev_vec =
        Arc::new(VectorAnnotator::open(&bert(), &dev_opts).unwrap_or_else(|e| panic!("TURBO_TEST_DEVICE={want}: {e}")));
    let exec = &dev_vec.info().execution;
    assert_ne!(exec.backend, "cpu", "{exec:?}");

    let pos_examples = [
        "We loved the friendly staff and the delicious breakfast.",
        "This camera takes crisp photos and works wonderfully.",
    ];
    let neg_examples =
        ["We hated the rude service and the cold soggy meal.", "This camera crashes constantly and wastes battery."];
    let centroids = LabelCentroids::from_text_examples(
        &cpu_vec,
        &sub,
        "sentiment",
        &[("positive", &pos_examples[..]), ("negative", &neg_examples[..])],
        CentroidNorm::L2,
        0.1,
    )
    .unwrap();

    let tags = TagSet::new("sentiment", &["positive", "negative"]);
    let head = ClassifierHead::fit_single_label(
        cpu_vec.model().clone(),
        32,
        tags,
        &[(centroids.centroid_of(Tag(0)), Tag(0)), (centroids.centroid_of(Tag(1)), Tag(1))],
        1e-4,
        150,
        1.5,
    )
    .unwrap();

    let text =
        "We loved the friendly staff and the delicious breakfast. We hated the rude service and the cold soggy meal.";
    let s0 = (0, 56);
    let s1 = (57, text.len() as u32);

    let run_pipeline = |vec_ann: &Arc<VectorAnnotator>| -> (Vec<f64>, Vec<f64>) {
        let doc = make_doc_with_sentences(text, &[s0, s1]);
        let tok_pipe =
            Analyzer::builder().given(Key::<Sentences>::DEFAULT).add(SimpleTokenizer::default()).build().unwrap();
        let mut docs = [doc];
        tok_pipe.run(&mut docs).remove(0).unwrap();
        let mut doc = docs.into_iter().next().unwrap();
        let opinions = {
            let tokens = doc.get::<Tokens>().unwrap();
            // Sentence 0 tokens: 0..10, Sentence 1 tokens: 10..21
            Opinions::from_items(
                tokens,
                [
                    Opinion { tokens: IxRange::new(0, 10), polarity: 0.0, intensity: 1.0, negated: false },
                    Opinion {
                        tokens: IxRange::new(10, tokens.len() as u32),
                        polarity: 0.0,
                        intensity: 1.0,
                        negated: false,
                    },
                ],
            )
            .unwrap()
        };
        doc.add(&Key::DEFAULT, opinions, Provenance::by("test")).unwrap();

        let pipe = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Tokens>::DEFAULT)
            .given(Key::<Opinions>::DEFAULT)
            .add(open_subwords())
            .add(SentenceVectors(Arc::clone(vec_ann)))
            .step(Bound::new(VectorHeadAnnotator::<Sentences>::new(centroids.clone())).write(0, "cent-sentiment"))
            .step(Bound::new(OpinionSpanHead::new(Arc::clone(vec_ann), head.clone())).write(0, "head-sentiment"))
            .build()
            .unwrap();

        let mut docs = [doc];
        pipe.run(&mut docs).remove(0).unwrap();

        let cent_cats = docs[0].get_key::<Categories<Sentences>>(&Key::named("cent-sentiment")).unwrap();
        let head_cats = docs[0].get_key::<Categories<Opinions>>(&Key::named("head-sentiment")).unwrap();
        assert_eq!(cent_cats.best_category(0), "positive");
        assert_eq!(cent_cats.best_category(1), "negative");
        assert_eq!(head_cats.best_category(0), "positive");
        assert_eq!(head_cats.best_category(1), "negative");
        (
            cent_cats.row_scores(0).iter().chain(cent_cats.row_scores(1)).copied().collect(),
            head_cats.row_scores(0).iter().chain(head_cats.row_scores(1)).copied().collect(),
        )
    };

    let (cpu_cent, cpu_head) = run_pipeline(&cpu_vec);
    let (dev_cent_1, dev_head_1) = run_pipeline(&dev_vec);
    let (dev_cent_2, dev_head_2) = run_pipeline(&dev_vec);

    // Repeated runs on the GPU must be bit-for-bit identical.
    assert_eq!(dev_cent_1, dev_cent_2);
    assert_eq!(dev_head_1, dev_head_2);

    // CPU and GPU scores must agree within the f32 precision tolerance.
    for (a, b) in cpu_cent.iter().zip(&dev_cent_1) {
        assert!((a - b).abs() <= 1e-5, "centroid score diff: cpu={a} dev={b}");
    }
    for (a, b) in cpu_head.iter().zip(&dev_head_1) {
        assert!((a - b).abs() <= 1e-5, "head score diff: cpu={a} dev={b}");
    }
    println!("{exec:?}: CPU/device head and centroid scores match within 1e-5");
}

#[derive(Debug)]
struct HeadPragmaticsScorer(Arc<ClassifierHead>);

impl turbo_doccat::PragmaticsVectorScorer for HeadPragmaticsScorer {
    fn score_sentence_vector(&self, vec: &[f32]) -> Result<f64, String> {
        let mut probs = vec![0.0f64; self.0.tags().len()];
        self.0.score(vec, &mut probs)?;
        let pos_idx = self
            .0
            .tags()
            .tag("sarcastic")
            .or_else(|| self.0.tags().tag("ironic"))
            .or_else(|| self.0.tags().tag("sarcasm"))
            .unwrap_or(Tag(0))
            .0 as usize;
        Ok(probs[pos_idx].clamp(0.0, 1.0))
    }

    fn config_hash(&self) -> u64 {
        self.0.config_hash()
    }

    fn model_ref(&self) -> Option<ModelRef> {
        Some(self.0.model().clone())
    }
}

#[test]
fn pragmatics_annotator_combines_incongruity_emoji_and_embedding_head() {
    use turbo_doccat::{PragmaticsAnnotator, PragmaticsVectorScorer};
    use turbo_normalize::emoji::EmojiAnnotate;

    let (sub, vec_ann) = open_cpu(64);

    // Train a sentence-level ClassifierHead on implicit irony examples vs literal sentences.
    let sarcastic_train = [
        "Another 7am status meeting on Saturday morning, my absolute dream.",
        "Nothing says competence like pushing straight to production on Friday night.",
        "Clearly the best use of four engineering weeks was renaming a button.",
    ];
    let literal_train = [
        "We deployed the patch on Tuesday morning after running the test suite.",
        "The team fixed the database indexing bug and reduced query latency.",
        "Our documentation now covers all configuration flags and error codes.",
    ];

    let centroids = LabelCentroids::from_text_examples(
        &vec_ann,
        &sub,
        "pragmatics",
        &[("sarcastic", &sarcastic_train[..]), ("literal", &literal_train[..])],
        CentroidNorm::L2,
        0.1,
    )
    .unwrap();

    let tags = TagSet::new("pragmatics", &["sarcastic", "literal"]);
    let mut train_vecs: Vec<(Vec<f32>, Tag)> = Vec::new();
    for s in &sarcastic_train {
        let doc = make_doc_with_sentences(s, &[(0, s.len() as u32)]);
        let pipe = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .add(open_subwords())
            .add(SentenceVectors(Arc::clone(&vec_ann)))
            .build()
            .unwrap();
        let mut docs = [doc];
        pipe.run(&mut docs).remove(0).unwrap();
        let v = docs[0].get::<Vectors<Sentences>>().unwrap().row(Ix::new(0)).to_vec();
        train_vecs.push((v, Tag(0)));
    }
    for s in &literal_train {
        let doc = make_doc_with_sentences(s, &[(0, s.len() as u32)]);
        let pipe = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .add(open_subwords())
            .add(SentenceVectors(Arc::clone(&vec_ann)))
            .build()
            .unwrap();
        let mut docs = [doc];
        pipe.run(&mut docs).remove(0).unwrap();
        let v = docs[0].get::<Vectors<Sentences>>().unwrap().row(Ix::new(0)).to_vec();
        train_vecs.push((v, Tag(1)));
    }
    let train_refs: Vec<(&[f32], Tag)> = train_vecs.iter().map(|(v, t)| (v.as_slice(), *t)).collect();
    let head = Arc::new(
        ClassifierHead::fit_single_label(vec_ann.model().clone(), 32, tags, &train_refs, 1e-4, 200, 2.0).unwrap(),
    );

    // Build a 4-sentence document exercising all three #324 signals + literal control:
    //   0. Riloff incongruity: "I just love waiting two hours on hold for customer service."
    //   1. Emoji mismatch: "What a wonderful way to start the morning 😭"
    //   2. Implicit irony (embedding head only, no explicit contrast):
    //      "Another 7am status meeting on Saturday morning, my absolute dream."
    //   3. Literal positive control (polarity untouched, pragmatics near 0):
    //      "We deployed the patch on Tuesday morning after running the test suite."
    let s0 = "I just love waiting two hours on hold for customer service.";
    let s1 = "What a wonderful way to start the morning 😭";
    let s2 = "Another 7am status meeting on Saturday morning, my absolute dream.";
    let s3 = "We deployed the patch on Tuesday morning after running the test suite.";
    let full = format!("{s0} {s1} {s2} {s3}");
    let o0 = (0, s0.len() as u32);
    let o1 = (o0.1 + 1, o0.1 + 1 + s1.len() as u32);
    let o2 = (o1.1 + 1, o1.1 + 1 + s2.len() as u32);
    let o3 = (o2.1 + 1, full.len() as u32);

    let scorer: Arc<dyn PragmaticsVectorScorer> = Arc::new(HeadPragmaticsScorer(Arc::clone(&head)));
    let prag_ann = PragmaticsAnnotator::new().with_vector_scorer(scorer);

    let pipe = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .add(SimpleTokenizer::default())
        .add(EmojiAnnotate::default())
        .add(LexiconSentiment::new())
        .add(open_subwords())
        .add(SentenceVectors(Arc::clone(&vec_ann)))
        .step(Bound::new(VectorHeadAnnotator::<Sentences>::new(centroids)).write(0, "prag-centroids"))
        .add(prag_ann)
        .build()
        .unwrap();

    let mut docs = [make_doc_with_sentences(&full, &[o0, o1, o2, o3])];
    pipe.run(&mut docs).remove(0).unwrap();
    let doc = &docs[0];

    let prag = doc.get::<Pragmatics>().unwrap();
    assert_eq!(prag.len(), 4);

    // The centroid head wrote its own Categories<Sentences> instance; it is not
    // what PragmaticsAnnotator read (that was the lexicon polarity at the
    // default instance), so check it on its own: the implicit-irony training
    // sentence lands on "sarcastic" and the literal one on "literal".
    let cent = doc.get_key::<Categories<Sentences>>(&Key::named("prag-centroids")).unwrap();
    assert_eq!(cent.len(), 4);
    assert_eq!(cent.best_category(2), "sarcastic");
    assert_eq!(cent.best_category(3), "literal");

    // Sentence 0: Riloff incongruity -> high score and Some(Incongruity).
    let p0 = prag.get(Ix::new(0));
    assert!(p0.score >= 0.65, "expected high Riloff incongruity score, got {}", p0.score);
    assert!(p0.evidence.is_some(), "expected Some(Incongruity) on sentence 0");

    // Sentence 1: Emoji mismatch -> high score and Some(Incongruity) with negative emoji span.
    let p1 = prag.get(Ix::new(1));
    assert!(p1.score >= 0.65, "expected high emoji mismatch score, got {}", p1.score);
    assert!(p1.evidence.is_some(), "expected Some(Incongruity) on sentence 1");

    // Sentence 2: Implicit irony via embedding head -> high score and `evidence: None`.
    let p2 = prag.get(Ix::new(2));
    assert!(p2.score >= 0.60, "expected high embedding-head score, got {}", p2.score);
    assert!(p2.evidence.is_none(), "expected evidence: None for head-only prediction on sentence 2");

    // Sentence 3: Literal sentence -> low score and `evidence: None`.
    let p3 = prag.get(Ix::new(3));
    assert!(p3.score < 0.35, "expected low score on literal sentence 3, got {}", p3.score);
    assert!(p3.evidence.is_none());

    // When TURBO_TEST_DEVICE or TURBO_TEST_REQUIRE_CUDA=1 is set, verify CPU vs GPU parity.
    let want = match std::env::var("TURBO_TEST_DEVICE") {
        Ok(w) => Some(w),
        Err(_) if std::env::var("TURBO_TEST_REQUIRE_CUDA").as_deref() == Ok("1") => Some("cuda".to_string()),
        Err(_) => None,
    };
    if let Some(want) = want {
        let device = match want.parse() {
            Ok(i) => Device::Index(i),
            Err(_) => Device::Backend(want.clone()),
        };
        let dev_opts = VectorOptions {
            encoder: EncoderOptions { device, max_seq: 64, ..EncoderOptions::default() },
            ..VectorOptions::default()
        };
        let dev_vec = Arc::new(
            VectorAnnotator::open(&bert(), &dev_opts).unwrap_or_else(|e| panic!("TURBO_TEST_DEVICE={want}: {e}")),
        );
        let dev_scorer: Arc<dyn PragmaticsVectorScorer> = Arc::new(HeadPragmaticsScorer(Arc::clone(&head)));
        let dev_prag_ann = PragmaticsAnnotator::new().with_vector_scorer(dev_scorer);
        let dev_pipe = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .add(SimpleTokenizer::default())
            .add(EmojiAnnotate::default())
            .add(LexiconSentiment::new())
            .add(open_subwords())
            .add(SentenceVectors(Arc::clone(&dev_vec)))
            .add(dev_prag_ann)
            .build()
            .unwrap();
        let mut dev_docs = [make_doc_with_sentences(&full, &[o0, o1, o2, o3])];
        dev_pipe.run(&mut dev_docs).remove(0).unwrap();
        let dev_prag = dev_docs[0].get::<Pragmatics>().unwrap();
        assert_eq!(dev_prag.len(), prag.len());
        for i in 0..prag.len() {
            let ix = Ix::new(i as u32);
            let cpu_item = prag.get(ix);
            let dev_item = dev_prag.get(ix);
            assert_eq!(cpu_item.evidence, dev_item.evidence, "sentence {i} evidence mismatch");
            assert!(
                (cpu_item.score - dev_item.score).abs() <= 1e-5,
                "sentence {i} pragmatics score diff: cpu={} dev={}",
                cpu_item.score,
                dev_item.score
            );
        }
    }
}

#[test]
fn learned_and_rule_aspect_sentiment_extract_triplets_and_match_cpu_device() {
    use turbo_analysis_embed::{LabeledAbsaDoc, LearnedAspectSentiment, SubwordVectors};
    use turbo_doccat::{
        AbsaTriplet, DEFAULT_ASPECT_CATEGORIES, RuleAspectSentiment, evaluate_absa_corpus, extract_triplets,
    };

    let (_sub, vec_ann) = open_cpu(64);

    // 10-sentence benchmark covering multi-aspect contrast ("battery life is great
    // but the screen scratches"), laptop/hardware, restaurant/hospitality,
    // negation, and implicit aspects.
    struct BenchRow<'a> {
        text: &'a str,
        aspects: &'a [(IxRange<Tokens>, &'a str, &'a str)],
        opinions: &'a [(IxRange<Tokens>, &'a str)],
        gold: &'a [AbsaTriplet],
    }

    let benchmark: &[BenchRow<'_>] = &[
        BenchRow {
            text: "battery life is great but the screen scratches",
            aspects: &[(IxRange::new(0, 2), "battery", "positive"), (IxRange::new(6, 7), "display", "negative")],
            opinions: &[(IxRange::new(3, 4), "positive"), (IxRange::new(7, 8), "negative")],
            gold: &[
                AbsaTriplet {
                    aspect_tokens: Some((0, 2)),
                    sentence: 0,
                    opinion_tokens: (3, 4),
                    polarity: Polarity::Positive,
                },
                AbsaTriplet {
                    aspect_tokens: Some((6, 7)),
                    sentence: 0,
                    opinion_tokens: (7, 8),
                    polarity: Polarity::Negative,
                },
            ],
        },
        BenchRow {
            text: "The pizza was delicious and the staff was friendly.",
            aspects: &[(IxRange::new(1, 2), "food", "positive"), (IxRange::new(6, 7), "service", "positive")],
            opinions: &[(IxRange::new(3, 4), "positive"), (IxRange::new(8, 9), "positive")],
            gold: &[
                AbsaTriplet {
                    aspect_tokens: Some((1, 2)),
                    sentence: 0,
                    opinion_tokens: (3, 4),
                    polarity: Polarity::Positive,
                },
                AbsaTriplet {
                    aspect_tokens: Some((6, 7)),
                    sentence: 0,
                    opinion_tokens: (8, 9),
                    polarity: Polarity::Positive,
                },
            ],
        },
        BenchRow {
            text: "The keyboard is responsive, however the fan noise is loud.",
            aspects: &[(IxRange::new(1, 2), "build", "positive"), (IxRange::new(7, 9), "performance", "negative")],
            opinions: &[(IxRange::new(3, 4), "positive"), (IxRange::new(10, 11), "negative")],
            gold: &[
                AbsaTriplet {
                    aspect_tokens: Some((1, 2)),
                    sentence: 0,
                    opinion_tokens: (3, 4),
                    polarity: Polarity::Positive,
                },
                AbsaTriplet {
                    aspect_tokens: Some((7, 9)),
                    sentence: 0,
                    opinion_tokens: (10, 11),
                    polarity: Polarity::Negative,
                },
            ],
        },
        BenchRow {
            text: "The customer service was not helpful.",
            aspects: &[(IxRange::new(1, 3), "service", "negative")],
            opinions: &[(IxRange::new(5, 6), "positive")], // negated in text -> effective negative
            gold: &[AbsaTriplet {
                aspect_tokens: Some((1, 3)),
                sentence: 0,
                opinion_tokens: (5, 6),
                polarity: Polarity::Negative,
            }],
        },
        BenchRow {
            text: "The build quality is sturdy and the trackpad is smooth.",
            aspects: &[(IxRange::new(1, 3), "build", "positive"), (IxRange::new(7, 8), "build", "positive")],
            opinions: &[(IxRange::new(4, 5), "positive"), (IxRange::new(9, 10), "positive")],
            gold: &[
                AbsaTriplet {
                    aspect_tokens: Some((1, 3)),
                    sentence: 0,
                    opinion_tokens: (4, 5),
                    polarity: Polarity::Positive,
                },
                AbsaTriplet {
                    aspect_tokens: Some((7, 8)),
                    sentence: 0,
                    opinion_tokens: (9, 10),
                    polarity: Polarity::Positive,
                },
            ],
        },
        BenchRow {
            text: "The pasta was soggy and the waiter was rude.",
            aspects: &[(IxRange::new(1, 2), "food", "negative"), (IxRange::new(6, 7), "service", "negative")],
            opinions: &[(IxRange::new(3, 4), "negative"), (IxRange::new(8, 9), "negative")],
            gold: &[
                AbsaTriplet {
                    aspect_tokens: Some((1, 2)),
                    sentence: 0,
                    opinion_tokens: (3, 4),
                    polarity: Polarity::Negative,
                },
                AbsaTriplet {
                    aspect_tokens: Some((6, 7)),
                    sentence: 0,
                    opinion_tokens: (8, 9),
                    polarity: Polarity::Negative,
                },
            ],
        },
        BenchRow {
            text: "The ambience was cozy and the wine list was generous.",
            aspects: &[(IxRange::new(1, 2), "ambience", "positive"), (IxRange::new(6, 8), "food", "positive")],
            opinions: &[(IxRange::new(3, 4), "positive"), (IxRange::new(9, 10), "positive")],
            gold: &[
                AbsaTriplet {
                    aspect_tokens: Some((1, 2)),
                    sentence: 0,
                    opinion_tokens: (3, 4),
                    polarity: Polarity::Positive,
                },
                AbsaTriplet {
                    aspect_tokens: Some((6, 8)),
                    sentence: 0,
                    opinion_tokens: (9, 10),
                    polarity: Polarity::Positive,
                },
            ],
        },
        BenchRow {
            text: "The software is buggy and the interface is clunky.",
            aspects: &[(IxRange::new(1, 2), "software", "negative"), (IxRange::new(6, 7), "software", "negative")],
            opinions: &[(IxRange::new(3, 4), "negative"), (IxRange::new(8, 9), "negative")],
            gold: &[
                AbsaTriplet {
                    aspect_tokens: Some((1, 2)),
                    sentence: 0,
                    opinion_tokens: (3, 4),
                    polarity: Polarity::Negative,
                },
                AbsaTriplet {
                    aspect_tokens: Some((6, 7)),
                    sentence: 0,
                    opinion_tokens: (8, 9),
                    polarity: Polarity::Negative,
                },
            ],
        },
        BenchRow {
            text: "Way too expensive and completely overpriced.",
            aspects: &[],
            opinions: &[(IxRange::new(2, 3), "negative"), (IxRange::new(5, 6), "negative")],
            gold: &[
                AbsaTriplet { aspect_tokens: None, sentence: 0, opinion_tokens: (2, 3), polarity: Polarity::Negative },
                AbsaTriplet { aspect_tokens: None, sentence: 0, opinion_tokens: (5, 6), polarity: Polarity::Negative },
            ],
        },
    ];

    // Prepare tokenized + subword-segmented documents for fitting the learned heads.
    let prep_pipe = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .add(SimpleTokenizer::default())
        .add(open_subwords())
        .build()
        .unwrap();

    let mut prep_docs: Vec<Document> =
        benchmark.iter().map(|row| make_doc_with_sentences(row.text, &[(0, row.text.len() as u32)])).collect();
    for res in prep_pipe.run(&mut prep_docs) {
        res.unwrap();
    }

    let labeled_docs: Vec<LabeledAbsaDoc<'_>> = benchmark
        .iter()
        .zip(&prep_docs)
        .map(|(row, doc)| LabeledAbsaDoc {
            subwords: doc.get::<Subwords>().unwrap(),
            sentences: doc.get::<Sentences>().unwrap(),
            tokens: doc.get::<Tokens>().unwrap(),
            aspects: row.aspects,
            opinions: row.opinions,
        })
        .collect();

    let learned =
        LearnedAspectSentiment::fit(Arc::clone(&vec_ann), DEFAULT_ASPECT_CATEGORIES, &labeled_docs, 1e-5, 600, 5.0)
            .unwrap();

    let rule_pipe = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .add(SimpleTokenizer::default())
        .add(RuleAspectSentiment::new())
        .build()
        .unwrap();

    let learned_pipe = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .add(SimpleTokenizer::default())
        .add(open_subwords())
        .add(SubwordVectors(Arc::clone(&vec_ann)))
        .add(learned.clone())
        .build()
        .unwrap();

    let mut rule_docs: Vec<Document> =
        benchmark.iter().map(|row| make_doc_with_sentences(row.text, &[(0, row.text.len() as u32)])).collect();
    for res in rule_pipe.run(&mut rule_docs) {
        res.unwrap();
    }

    let mut learned_docs: Vec<Document> =
        benchmark.iter().map(|row| make_doc_with_sentences(row.text, &[(0, row.text.len() as u32)])).collect();
    for res in learned_pipe.run(&mut learned_docs) {
        res.unwrap();
    }

    // Verify provenance distinguishes the rule path from the learned path while
    // writing the exact same `(Opinions, Aspects)` layer types.
    let rule_prov = rule_docs[0].provenance(Key::<Aspects>::DEFAULT.id()).unwrap();
    let learned_prov = learned_docs[0].provenance(Key::<Aspects>::DEFAULT.id()).unwrap();
    assert_eq!(rule_prov.annotator, "turbo-doccat:aspect-sentiment-rule");
    assert_eq!(rule_prov.model.as_ref().unwrap().id, "double-propagation:vader");
    assert_eq!(learned_prov.annotator, "turbo:aspect-sentiment-learned");
    assert_eq!(learned_prov.model.as_ref().unwrap().id, vec_ann.model().id);
    assert_eq!(learned_prov.execution.as_ref().unwrap().backend, "cpu");

    // Verify `"battery life is great but the screen scratches"` on both paths!
    for (label, doc) in [("rule", &rule_docs[0]), ("learned", &learned_docs[0])] {
        let asps = doc.get::<Aspects>().unwrap();
        let ops = doc.get::<Opinions>().unwrap();
        assert_eq!(asps.len(), 2, "{label}: expected 2 aspects");
        assert_eq!(ops.len(), 2, "{label}: expected 2 opinions");
        let a0 = asps.get(Ix::new(0));
        let a1 = asps.get(Ix::new(1));
        assert_eq!(
            a0.target,
            AspectTarget::Term { tokens: IxRange::new(0, 2), sentence: Some(Ix::new(0)) },
            "{label}: aspect 0 target"
        );
        assert_eq!(a0.polarity.label(), Polarity::Positive, "{label}: aspect 0 polarity");
        assert_eq!(a0.opinions, &[Ix::<Opinions>::new(0)], "{label}: aspect 0 opinions");

        assert_eq!(
            a1.target,
            AspectTarget::Term { tokens: IxRange::new(6, 7), sentence: Some(Ix::new(0)) },
            "{label}: aspect 1 target"
        );
        assert_eq!(a1.polarity.label(), Polarity::Negative, "{label}: aspect 1 polarity");
        assert_eq!(a1.opinions, &[Ix::<Opinions>::new(1)], "{label}: aspect 1 opinions");
    }

    // Verify implicit aspect sentence (`"Way too expensive and completely overpriced."`) on both paths.
    let last_idx = benchmark.len() - 1;
    for (label, doc) in [("rule", &rule_docs[last_idx]), ("learned", &learned_docs[last_idx])] {
        let asps = doc.get::<Aspects>().unwrap();
        assert_eq!(asps.len(), 1, "{label}: expected 1 implicit aspect");
        let a0 = asps.get(Ix::new(0));
        assert_eq!(a0.target, AspectTarget::Implicit { sentence: Ix::new(0) }, "{label}: implicit target");
        assert_eq!(a0.polarity.label(), Polarity::Negative, "{label}: implicit polarity");
    }

    let gold_corpus: Vec<Vec<AbsaTriplet>> = benchmark.iter().map(|r| r.gold.to_vec()).collect();
    let rule_pred: Vec<Vec<AbsaTriplet>> =
        rule_docs.iter().map(|d| extract_triplets(d.get::<Opinions>().unwrap(), d.get::<Aspects>().unwrap())).collect();
    let learned_pred: Vec<Vec<AbsaTriplet>> = learned_docs
        .iter()
        .map(|d| extract_triplets(d.get::<Opinions>().unwrap(), d.get::<Aspects>().unwrap()))
        .collect();

    let rule_metrics = evaluate_absa_corpus(&gold_corpus, &rule_pred);
    let learned_metrics = evaluate_absa_corpus(&gold_corpus, &learned_pred);
    println!(
        "ABSA training fixture (9 docs): Rule [aspect F1={:.3}, opinion F1={:.3}, triplet F1={:.3}], Learned [aspect F1={:.3}, opinion F1={:.3}, triplet F1={:.3}]",
        rule_metrics.aspect_f1,
        rule_metrics.opinion_f1,
        rule_metrics.triplet_f1,
        learned_metrics.aspect_f1,
        learned_metrics.opinion_f1,
        learned_metrics.triplet_f1,
    );
    assert!(rule_metrics.aspect_f1 >= 0.95 && rule_metrics.triplet_f1 >= 0.95);
    assert!(learned_metrics.aspect_f1 >= 0.90 && learned_metrics.triplet_f1 >= 0.90);

    // Evaluate both paths on a disjoint 4-sentence held-out set not seen during `LearnedAspectSentiment::fit`.
    // Note: `tiny-bert-bundle` is a 32-dim synthetic fixture with uncalibrated positional weights,
    // whereas `RuleAspectSentiment` uses the lexicon + propagation rules.
    let held_out: &[(&str, &[AbsaTriplet])] = &[
        (
            "The keyboard is sturdy and the trackpad is smooth.",
            &[
                AbsaTriplet {
                    aspect_tokens: Some((1, 2)),
                    sentence: 0,
                    opinion_tokens: (3, 4),
                    polarity: Polarity::Positive,
                },
                AbsaTriplet {
                    aspect_tokens: Some((6, 7)),
                    sentence: 0,
                    opinion_tokens: (8, 9),
                    polarity: Polarity::Positive,
                },
            ],
        ),
        (
            "The pizza was soggy and the waiter was rude.",
            &[
                AbsaTriplet {
                    aspect_tokens: Some((1, 2)),
                    sentence: 0,
                    opinion_tokens: (3, 4),
                    polarity: Polarity::Negative,
                },
                AbsaTriplet {
                    aspect_tokens: Some((6, 7)),
                    sentence: 0,
                    opinion_tokens: (8, 9),
                    polarity: Polarity::Negative,
                },
            ],
        ),
        (
            "The customer service was not friendly.",
            &[AbsaTriplet {
                aspect_tokens: Some((1, 3)),
                sentence: 0,
                opinion_tokens: (5, 6),
                polarity: Polarity::Negative,
            }],
        ),
        (
            "Way too expensive and completely overpriced!",
            &[
                AbsaTriplet { aspect_tokens: None, sentence: 0, opinion_tokens: (2, 3), polarity: Polarity::Negative },
                AbsaTriplet { aspect_tokens: None, sentence: 0, opinion_tokens: (5, 6), polarity: Polarity::Negative },
            ],
        ),
    ];
    let mut held_rule_docs: Vec<Document> =
        held_out.iter().map(|(text, _)| make_doc_with_sentences(text, &[(0, text.len() as u32)])).collect();
    for res in rule_pipe.run(&mut held_rule_docs) {
        res.unwrap();
    }
    let mut held_learned_docs: Vec<Document> =
        held_out.iter().map(|(text, _)| make_doc_with_sentences(text, &[(0, text.len() as u32)])).collect();
    for res in learned_pipe.run(&mut held_learned_docs) {
        res.unwrap();
    }
    let held_gold: Vec<Vec<AbsaTriplet>> = held_out.iter().map(|(_, g)| g.to_vec()).collect();
    let held_rule_pred: Vec<Vec<AbsaTriplet>> = held_rule_docs
        .iter()
        .map(|d| extract_triplets(d.get::<Opinions>().unwrap(), d.get::<Aspects>().unwrap()))
        .collect();
    let held_learned_pred: Vec<Vec<AbsaTriplet>> = held_learned_docs
        .iter()
        .map(|d| extract_triplets(d.get::<Opinions>().unwrap(), d.get::<Aspects>().unwrap()))
        .collect();
    let held_rule_m = evaluate_absa_corpus(&held_gold, &held_rule_pred);
    let held_learned_m = evaluate_absa_corpus(&held_gold, &held_learned_pred);
    println!(
        "ABSA held-out (4 docs): Rule [aspect F1={:.3}, opinion F1={:.3}, triplet F1={:.3}], Learned [aspect F1={:.3}, opinion F1={:.3}, triplet F1={:.3}]",
        held_rule_m.aspect_f1,
        held_rule_m.opinion_f1,
        held_rule_m.triplet_f1,
        held_learned_m.aspect_f1,
        held_learned_m.opinion_f1,
        held_learned_m.triplet_f1,
    );
    assert!(held_rule_m.aspect_f1 >= 0.95 && held_rule_m.triplet_f1 >= 0.95);
    assert!(held_learned_m.aspect_f1 >= 0.90 && held_learned_m.triplet_f1 >= 0.85);

    // When TURBO_TEST_DEVICE or TURBO_TEST_REQUIRE_CUDA=1 is set, verify CPU vs GPU parity for LearnedAspectSentiment.
    let want = match std::env::var("TURBO_TEST_DEVICE") {
        Ok(w) => Some(w),
        Err(_) if std::env::var("TURBO_TEST_REQUIRE_CUDA").as_deref() == Ok("1") => Some("cuda".to_string()),
        Err(_) => None,
    };
    if let Some(want) = want {
        let device = match want.parse() {
            Ok(i) => Device::Index(i),
            Err(_) => Device::Backend(want.clone()),
        };
        let dev_opts = VectorOptions {
            encoder: EncoderOptions { device, max_seq: 64, ..EncoderOptions::default() },
            ..VectorOptions::default()
        };
        let dev_vec = Arc::new(
            VectorAnnotator::open(&bert(), &dev_opts).unwrap_or_else(|e| panic!("TURBO_TEST_DEVICE={want}: {e}")),
        );
        let mut dev_learned = learned.clone();
        dev_learned.vectors = Arc::clone(&dev_vec);

        let dev_pipe = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .add(SimpleTokenizer::default())
            .add(open_subwords())
            .add(SubwordVectors(Arc::clone(&dev_vec)))
            .add(dev_learned)
            .build()
            .unwrap();

        let mut dev_docs: Vec<Document> =
            benchmark.iter().map(|row| make_doc_with_sentences(row.text, &[(0, row.text.len() as u32)])).collect();
        for res in dev_pipe.run(&mut dev_docs) {
            res.unwrap();
        }
        let dev_prov = dev_docs[0].provenance(Key::<Aspects>::DEFAULT.id()).unwrap();
        assert_ne!(dev_prov.execution.as_ref().unwrap().backend, "cpu");

        for (cpu_doc, dev_doc) in learned_docs.iter().zip(&dev_docs) {
            let cpu_asps = cpu_doc.get::<Aspects>().unwrap();
            let dev_asps = dev_doc.get::<Aspects>().unwrap();
            assert_eq!(cpu_asps.len(), dev_asps.len());
            for i in 0..cpu_asps.len() {
                let ix = Ix::new(i as u32);
                let ca = cpu_asps.get(ix);
                let da = dev_asps.get(ix);
                assert_eq!(ca.target, da.target);
                assert_eq!(ca.category, da.category);
                assert_eq!(ca.opinions, da.opinions);
                assert!((ca.polarity.positive - da.polarity.positive).abs() <= 1e-5);
                assert!((ca.polarity.negative - da.polarity.negative).abs() <= 1e-5);
                assert!((ca.polarity.neutral - da.polarity.neutral).abs() <= 1e-5);
            }
        }
    }
}
