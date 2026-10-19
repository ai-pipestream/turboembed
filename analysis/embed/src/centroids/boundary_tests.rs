use super::*;

fn model() -> ModelRef {
    ModelRef { id: "test".into(), revision: "1".into(), sha256: "test".into() }
}

fn head(temperature: f64) -> LabelCentroids {
    LabelCentroids::from_centroids(
        model(),
        1,
        TagSet::new("test", &["a", "b"]),
        vec![1.0, 2.0],
        temperature,
        CentroidNorm::None,
    )
    .unwrap()
}

#[test]
fn nonfinite_query_vectors_are_refused() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(head(1.0).dot_scores(&[value], &mut [0.0; 2]).is_err());
    }
}

#[test]
fn tiny_temperature_preserves_the_winner() {
    let mut p = [0.0; 2];
    let (tag, _) = head(f64::from_bits(1)).score_single_label(&[1.0], &mut p).unwrap();
    assert_eq!(tag, Tag(1));
    assert_eq!(p, [0.0, 1.0]);
}

#[test]
fn nonfinite_multilabel_configuration_is_refused() {
    for (bias, scale) in [(f64::NAN, 1.0), (0.0, f64::INFINITY), (f64::INFINITY, 0.0)] {
        assert!(head(1.0).score_multi_label(&[1.0], bias, scale, &mut [0.0; 2]).is_err());
    }
}

#[test]
fn unindexable_label_count_is_refused() {
    let labels: Vec<String> = (0..65537).map(|i| format!("l{i}")).collect();
    let refs: Vec<&str> = labels.iter().map(String::as_str).collect();
    assert!(
        LabelCentroids::from_centroids(
            model(),
            1,
            TagSet::new("test", &refs),
            vec![0.0; refs.len()],
            1.0,
            CentroidNorm::None
        )
        .is_err()
    );
}

#[test]
fn malformed_vector_columns_are_refused_before_categorizing() {
    let mut vectors = Vectors::<Sentences>::new(1, vec![1.0], vec![], model());
    vectors.dim = 0;
    assert!(head(1.0).categorize_vectors("test", &vectors).is_err());
    let vectors = Vectors::<Sentences>::new(1, vec![1.0], vec![Ix::new(3)], model());
    assert!(head(1.0).categorize_vectors("test", &vectors).is_err());
}

#[test]
fn truncated_centroid_payload_and_trailing_data_are_refused() {
    let bytes = head(1.0).to_bytes();
    for n in 0..bytes.len() {
        assert!(LabelCentroids::from_bytes(&bytes[..n]).is_err());
    }
    let mut extra = bytes;
    extra.push(0);
    assert!(LabelCentroids::from_bytes(&extra).is_err());
}

#[test]
fn ambiguous_labels_are_refused() {
    for names in [["a", "a"], ["a", " "]] {
        assert!(
            LabelCentroids::from_centroids(
                model(),
                1,
                TagSet::new("test", &names),
                vec![1.0, 2.0],
                1.0,
                CentroidNorm::None
            )
            .is_err()
        );
    }
}

#[test]
fn grouping_rejects_invalid_category_columns() {
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 1));
    let sentences = Sentences { spans };
    let mut categories = Categories::<Sentences>::with_capacity(TagSet::new("test", &["a"]), 1);
    categories.push(Tag(0), &[1.0]);
    categories.scores.clear();
    let vectors = Vectors::new(1, vec![1.0], vec![], model());
    assert!(category_sentence_centroids(&sentences, &categories, &vectors, None, CentroidNorm::None).is_err());
}
