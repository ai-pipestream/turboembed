/// The least defined cosine between corresponding nonempty vectors.
/// Invalid shapes, zero norms and nonfinite values return negative infinity,
/// so an undefined comparison never clears a positive agreement floor.
pub fn least_cosine(a: &[Vec<f32>], b: &[Vec<f32>]) -> f64 {
    if a.is_empty() || a.len() != b.len() {
        return f64::NEG_INFINITY;
    }
    a.iter()
        .zip(b)
        .try_fold(1.0_f64, |least, (x, y)| {
            if x.is_empty() || x.len() != y.len() {
                return None;
            }
            let dot: f64 = x.iter().zip(y).map(|(p, q)| f64::from(*p) * f64::from(*q)).sum();
            let norm = |v: &[f32]| v.iter().map(|p| f64::from(*p).powi(2)).sum::<f64>().sqrt();
            let (nx, ny) = (norm(x), norm(y));
            if !nx.is_finite() || !ny.is_finite() || nx == 0.0 || ny == 0.0 {
                return None;
            }
            let cosine = dot / (nx * ny);
            cosine.is_finite().then(|| least.min(cosine))
        })
        .unwrap_or(f64::NEG_INFINITY)
}

#[test]
fn undefined_or_mismatched_vectors_cannot_pass_a_cosine_floor() {
    for (a, b) in [
        (vec![vec![0.0, 0.0]], vec![vec![1.0, 0.0]]),
        (vec![vec![0.0]], vec![vec![0.0]]),
        (vec![vec![f32::NAN]], vec![vec![1.0]]),
        (vec![vec![f32::INFINITY]], vec![vec![1.0]]),
        (vec![], vec![]),
        (vec![vec![]], vec![vec![]]),
        (vec![vec![1.0]], vec![vec![1.0, 2.0]]),
        (vec![vec![1.0], vec![1.0]], vec![vec![1.0]]),
    ] {
        assert_eq!(least_cosine(&a, &b), f64::NEG_INFINITY, "{a:?} against {b:?}");
    }
}
#[test]
fn defined_cosines_keep_the_worst_row() {
    assert_eq!(least_cosine(&[vec![1.0, 0.0]], &[vec![2.0, 0.0]]), 1.0);
    assert_eq!(least_cosine(&[vec![1.0, 0.0]], &[vec![0.0, 1.0]]), 0.0);
    assert_eq!(least_cosine(&[vec![1.0], vec![1.0]], &[vec![1.0], vec![-1.0]]), -1.0);
}
