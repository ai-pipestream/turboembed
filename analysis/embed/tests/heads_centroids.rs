//! Centroids and heads (#321): exact parity with the opennlp-grpc
//! `CentroidsTest` and `ChunkEmbedProcessor.buildCategoryGroup`
//! (`OPENNLP-1833`); `LabelCentroids` from example texts and vectors with
//! the binary `TCENT001` round-trip; `ClassifierHead` linear and MLP
//! fitting with the binary `THEAD001` round-trip.

mod heads_support;

use std::sync::Arc;

use turbo_analysis::*;
use turbo_analysis_embed::*;
use turbo_analysis_layers::{Categories, Sentences, Vectors};

use heads_support::*;

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
