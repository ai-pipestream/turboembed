//! Real legacy-input behavior and compatibility controls for checked callers.

use turbo_analysis::{Ix, Layer, ModelRef};
use turbo_analysis_layers::{Sentences, Vectors};
use turbo_chunker::{BoundaryDriftCurve, Drift, Rows, Score, drift};

fn vectors(dim: u32, data: Vec<f32>, empty: &[u32]) -> Vectors<Sentences> {
    Vectors::new(
        dim,
        data,
        empty.iter().copied().map(Ix::new).collect(),
        ModelRef { id: "drift-input".into(), revision: "1".into(), sha256: "0".repeat(64) },
    )
}

#[test]
fn legacy_curve_silently_truncates_a_ragged_vector_layer() {
    let vectors = vectors(2, vec![1.0, 0.0, 9.0], &[]);
    assert!(vectors.check().is_err());
    let curve = BoundaryDriftCurve::compute(&vectors, &Drift::default());
    assert_eq!(vectors.data.len(), 3);
    assert_eq!(curve.raw_cosine_micro.len(), 1, "the trailing coordinate was silently ignored");
    assert_eq!(curve.cost.len(), 1);
    assert_eq!(drift(&Rows::of(&vectors), &Drift::default()), (curve.cost, curve.reason));
}

#[test]
fn legacy_nonfinite_rows_are_deliberately_not_scored() {
    let vectors = vectors(2, vec![1.0, 0.0, f32::NAN, 1.0, 0.0, 1.0], &[]);
    vectors.check().unwrap();
    let rows = Rows::of(&vectors);
    assert_eq!(rows.present(), [0, 2]);
    for score in [Score::Depth, Score::Distance] {
        let d = Drift { window: 1.try_into().unwrap(), smoothing: 0, score, ..Drift::default() };
        let curve = BoundaryDriftCurve::compute(&vectors, &d);
        assert_eq!(drift(&rows, &d), (curve.cost, curve.reason));
    }
}
