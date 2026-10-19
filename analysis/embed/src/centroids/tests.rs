use super::*;

fn test_model() -> ModelRef {
    ModelRef { id: "tiny-bert".into(), revision: "rev1".into(), sha256: "0123456789abcdef".into() }
}

// Parity tests matching CentroidsTest.java on OPENNLP-1833-grpc-uber-server.

#[test]
fn averages_vectors_element_wise() {
    let v1 = [0.0f32, 2.0, 4.0];
    let v2 = [2.0f32, 4.0, 8.0];
    let c = centroid(&[&v1[..], &v2[..]], CentroidNorm::None).unwrap().unwrap();
    assert_eq!(c, vec![1.0f32, 3.0, 6.0]);
}

#[test]
fn l2_normalizes_the_arithmetic_mean() {
    let v1 = [3.0f32, 4.0];
    let v2 = [3.0f32, 4.0];
    let c = centroid(&[&v1[..], &v2[..]], CentroidNorm::L2).unwrap().unwrap();
    assert_eq!(c, vec![0.6f32, 0.8]);
}

#[test]
fn refuses_to_normalize_a_zero_centroid() {
    let v1 = [1.0f32, 0.0];
    let v2 = [-1.0f32, 0.0];
    let err = centroid(&[&v1[..], &v2[..]], CentroidNorm::L2).unwrap_err();
    assert!(err.contains("zero norm"), "{err}");
}

#[test]
fn single_vector_is_its_own_centroid() {
    let v1 = [1.5f32, -2.5];
    let c = centroid(&[&v1[..]], CentroidNorm::None).unwrap().unwrap();
    assert_eq!(c, vec![1.5f32, -2.5]);
}

#[test]
fn empty_input_yields_none() {
    let empty: [&[f32]; 0] = [];
    assert_eq!(centroid(&empty, CentroidNorm::None).unwrap(), None);
}

#[test]
fn label_centroids_round_trip_and_deterministic_scoring() {
    let pos_examples: &[&[f32]] = &[&[0.9, 0.1, 0.0], &[0.8, 0.2, 0.0]];
    let neg_examples: &[&[f32]] = &[&[-0.9, -0.1, 0.0], &[-0.8, -0.2, 0.0]];
    let neu_examples: &[&[f32]] = &[&[0.0, 0.0, 1.0], &[0.0, 0.1, 0.9]];
    let lc = LabelCentroids::from_vector_examples(
        test_model(),
        3,
        "sentiment",
        &[("positive", pos_examples), ("negative", neg_examples), ("neutral", neu_examples)],
        CentroidNorm::L2,
        0.25,
    )
    .unwrap();

    let bytes = lc.to_bytes();
    let loaded = LabelCentroids::from_bytes(&bytes).unwrap();
    assert_eq!(lc, loaded);

    let mut probs1 = [0.0f64; 3];
    let mut probs2 = [0.0f64; 3];
    let (tag1, p1) = lc.score_single_label(&[0.95, 0.05, 0.0], &mut probs1).unwrap();
    let (tag2, p2) = loaded.score_single_label(&[0.95, 0.05, 0.0], &mut probs2).unwrap();
    assert_eq!(tag1, Tag(0));
    assert_eq!(tag1, tag2);
    assert_eq!(p1.to_bits(), p2.to_bits());
    assert_eq!(probs1, probs2);
    assert!((probs1.iter().sum::<f64>() - 1.0).abs() < 1e-12);
}
