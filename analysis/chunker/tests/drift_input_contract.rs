//! Real legacy-input behavior and compatibility controls for checked callers.

use turbo_analysis::{Ix, Layer, ModelRef};
use turbo_analysis_layers::{Sentences, Vectors};
use turbo_chunker::{BoundaryDriftCurve, Drift, Rows, Score, drift, try_drift};

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

fn refused(v: &Vectors<Sentences>, expected: &str) {
    let data: Vec<_> = v.data.iter().map(|x| x.to_bits()).collect();
    let empty = v.empty.clone();
    let d = Drift::default();
    assert_eq!(BoundaryDriftCurve::try_compute(v, &d).unwrap_err(), expected);
    assert_eq!(try_drift(v, &d).unwrap_err(), expected);
    assert_eq!(v.data.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), data);
    assert_eq!(v.empty, empty);
}

#[test]
fn checked_drift_refuses_ragged_and_zero_dimension_layers() {
    for v in [
        vectors(0, vec![], &[]),
        vectors(0, vec![1.0], &[]),
        vectors(2, vec![1.0], &[]),
        vectors(2, vec![1.0, 0.0, 9.0], &[]),
        vectors(3, vec![1.0, 0.0, 0.0, 9.0, 8.0], &[]),
    ] {
        refused(&v, &v.check().unwrap_err());
    }
}

#[test]
fn checked_drift_refuses_invalid_empty_row_metadata() {
    for empty in [&[1, 1][..], &[2, 1], &[3], &[0]] {
        let v = vectors(2, vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0], empty);
        refused(&v, &v.check().unwrap_err());
    }
    let v = vectors(2, vec![f32::NAN, 0.0], &[0]);
    refused(&v, &v.check().unwrap_err());
}

#[test]
fn checked_drift_names_nonfinite_row_and_coordinate() {
    for row in 0..3 {
        for coordinate in 0..2 {
            for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut v = vectors(2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[]);
                v.data[row * 2 + coordinate] = value;
                v.check().unwrap();
                refused(&v, &format!("row {row} coordinate {coordinate} is not finite"));
            }
        }
    }
}

#[test]
fn checked_and_legacy_drift_are_bit_identical_for_valid_layers() {
    let mut topic = Vec::new();
    for angle in [0.0f64, 4.0, 8.0, 0.0, 4.0, 8.0, 90.0, 85.0, 90.0, 85.0, 90.0, 85.0] {
        topic.extend([angle.to_radians().cos() as f32, angle.to_radians().sin() as f32]);
    }
    for v in [
        vectors(2, topic, &[]),
        vectors(2, vec![], &[]),
        vectors(2, vec![1.0, -0.0], &[]),
        vectors(2, vec![0.0; 6], &[0, 1, 2]),
        vectors(2, vec![0.0; 6], &[]),
        vectors(2, vec![1.0, 0.0, 0.0, -0.0, 0.0, 1.0], &[1]),
        vectors(2, vec![f32::MAX, f32::MAX, -f32::MAX, f32::MAX], &[]),
        vectors(2, vec![f32::from_bits(1), 0.0, 0.0, f32::from_bits(1)], &[]),
    ] {
        v.check().unwrap();
        for score in [Score::Depth, Score::Distance] {
            for smoothing in [0, 1, u32::MAX] {
                for window in [1u32, 3] {
                    let d = Drift { score, smoothing, window: window.try_into().unwrap(), ..Drift::default() };
                    let old = BoundaryDriftCurve::compute(&v, &d);
                    assert_eq!(BoundaryDriftCurve::try_compute(&v, &d).unwrap(), old);
                    assert_eq!(try_drift(&v, &d).unwrap(), drift(&Rows::of(&v), &d));
                    assert_eq!(try_drift(&v, &d).unwrap(), (old.cost, old.reason));
                }
            }
        }
    }
}
