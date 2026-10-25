use turbo_analysis::{Ix, ModelRef};
use turbo_analysis_layers::{Chunks, Sentences, Vectors};
use turbo_chunker::projection::{ProjectionDims, VectorProjection};

fn model() -> ModelRef {
    ModelRef { id: "test".into(), revision: "0".into(), sha256: "0".into() }
}

fn vectors<L: turbo_analysis::Layer>(dim: u32, rows: &[&[f32]], empty: &[u32]) -> Vectors<L> {
    let mut data = Vec::new();
    for r in rows {
        data.extend_from_slice(r);
    }
    Vectors::new(dim, data, empty.iter().map(|&i| Ix::new(i)).collect(), model())
}

#[test]
fn collinear_points_project_onto_one_axis() {
    // Points on the line t * (1, 2, 0): one component carries all the
    // variance, the second coordinate is zero, ordering follows t.
    let v = vectors::<Sentences>(3, &[&[0.0, 0.0, 5.0], &[1.0, 2.0, 5.0], &[3.0, 6.0, 5.0], &[-1.0, -2.0, 5.0]], &[]);
    let p = VectorProjection::compute(&v, ProjectionDims::D2).unwrap();
    assert_eq!(p.coords.len(), 8);
    assert!((p.explained_variance[0] - 1.0).abs() < 1e-6, "{:?}", p.explained_variance);
    assert_eq!(p.explained_variance[1], 0.0);
    for i in 0..4u32 {
        assert!(p.row(Ix::new(i))[1].abs() < 1e-6, "row {i} off the line");
    }
    let xs: Vec<f32> = (0..4u32).map(|i| p.row(Ix::new(i))[0]).collect();
    assert!(xs[3] < xs[0] && xs[0] < xs[1] && xs[1] < xs[2], "sign canonicalized along (1, 2, 0): {xs:?}");
    // Deterministic: recomputing gives identical bits.
    let again = VectorProjection::compute(&v, ProjectionDims::D2).unwrap();
    assert_eq!(p.coords, again.coords);
    assert_eq!(p.model.id, "test");
}

#[test]
fn empty_rows_are_excluded_and_zeroed() {
    let v = vectors::<Chunks>(2, &[&[1.0, 0.0], &[0.0, 0.0], &[2.0, 0.0], &[3.0, 0.0]], &[1]);
    let p = VectorProjection::compute(&v, ProjectionDims::D3).unwrap();
    assert_eq!(p.row(Ix::new(1)), &[0.0, 0.0, 0.0]);
    assert_eq!(p.empty, vec![Ix::new(1)]);
    // Adjacent cosines: pair (0,1) and (1,2) touch the empty row; (2,3)
    // are parallel; the last row has no successor.
    assert_eq!(p.adjacent_cosine.len(), 4);
    assert_eq!(p.adjacent_cosine[0], None);
    assert_eq!(p.adjacent_cosine[1], None);
    assert_eq!(p.adjacent_cosine[2], Some(1.0));
    assert_eq!(p.adjacent_cosine[3], None);
    // Mean is over rows 0,2,3 only (mean x = 2): centred coords follow.
    assert!(p.row(Ix::new(0))[0] < 0.0 && p.row(Ix::new(3))[0] > 0.0);
    assert!(p.row(Ix::new(2))[0].abs() < 1e-6);
}

#[test]
fn degenerate_inputs_project_to_zero() {
    let one = vectors::<Chunks>(4, &[&[1.0, 2.0, 3.0, 4.0]], &[]);
    let p = VectorProjection::compute(&one, ProjectionDims::D3).unwrap();
    assert_eq!(p.coords, vec![0.0, 0.0, 0.0]);
    assert_eq!(p.explained_variance, [0.0, 0.0, 0.0]);
    assert_eq!(p.adjacent_cosine, vec![None]);

    let all_empty = vectors::<Chunks>(2, &[&[0.0, 0.0], &[0.0, 0.0]], &[0, 1]);
    let p = VectorProjection::compute(&all_empty, ProjectionDims::D2).unwrap();
    assert!(p.coords.iter().all(|&x| x == 0.0));
    // Identical rows: no variance in any direction.
    let same = vectors::<Chunks>(2, &[&[1.0, 1.0], &[1.0, 1.0], &[1.0, 1.0]], &[]);
    let p = VectorProjection::compute(&same, ProjectionDims::D2).unwrap();
    assert!(p.coords.iter().all(|&x| x == 0.0));
    assert_eq!(p.explained_variance, [0.0, 0.0, 0.0]);
    assert_eq!(p.adjacent_cosine[0], Some(1.0));
    // A zero-norm (but not listed empty) row has no cosine.
    let zero = vectors::<Chunks>(2, &[&[0.0, 0.0], &[1.0, 0.0]], &[]);
    let p = VectorProjection::compute(&zero, ProjectionDims::D2).unwrap();
    assert_eq!(p.adjacent_cosine[0], None);
}

#[test]
fn projection_refusals() {
    let bad = vectors::<Chunks>(2, &[&[1.0, f32::INFINITY], &[2.0, 0.0]], &[]);
    match VectorProjection::compute(&bad, ProjectionDims::D2) {
        Err(e) => assert!(e.contains("row 0 coordinate 1"), "{e}"),
        Ok(_) => panic!("non-finite accepted"),
    }
    // A ragged data buffer fails the layer's own check.
    let ragged = Vectors::<Chunks>::new(3, vec![1.0, 2.0], vec![], model());
    assert!(VectorProjection::compute(&ragged, ProjectionDims::D2).is_err());
    // More dims than the data has: coordinates past the spectrum stay 0.
    let v = vectors::<Chunks>(1, &[&[1.0], &[2.0], &[4.0]], &[]);
    let p = VectorProjection::compute(&v, ProjectionDims::D3).unwrap();
    for i in 0..3u32 {
        assert_eq!(&p.row(Ix::new(i))[1..], &[0.0, 0.0]);
    }
    assert!((p.explained_variance[0] - 1.0).abs() < 1e-6);
}

#[test]
fn drift_curve_matches_drift_bit_for_bit() {
    use turbo_analysis_layers::Cut;
    use turbo_chunker::Rows;
    use turbo_chunker::projection::BoundaryDriftCurve;
    use turbo_chunker::{Drift, Score, drift};

    // Six sentences near 0 degrees, six near 90 (the drift tests' own
    // fixture shape), as unit vectors in the plane.
    let mut deg: Vec<f64> = (0..6).map(|i| f64::from(i % 3) * 4.0).collect();
    deg.extend((0..6).map(|i| 90.0 - f64::from(i % 2) * 5.0));
    let data: Vec<f32> = deg.iter().flat_map(|d| [d.to_radians().cos() as f32, d.to_radians().sin() as f32]).collect();
    let v = Vectors::<Sentences>::new(2, data, vec![], model());
    for d in [Drift::default(), Drift { score: Score::Distance, floor: 0, ..Drift::default() }] {
        let curve = BoundaryDriftCurve::compute(&v, &d);
        let (cost, reason) = drift(&Rows::of(&v), &d);
        assert_eq!(curve.cost, cost, "costs differ under {:?}", d.score);
        assert_eq!(curve.reason, reason);
        assert_eq!(curve.raw_cosine_micro.len(), 12);
        // Unscored ends: no cosine, no smoothed value, zero score/cost.
        for g in [0, 1, 10, 11] {
            assert_eq!(curve.raw_cosine_micro[g], None, "gap {g}");
            assert_eq!(curve.smoothed_cosine_micro[g], None);
            assert_eq!((curve.depth_score_micro[g], curve.cost[g]), (0, 0));
        }
        // Scored gaps have both series, and the smoothed value at the
        // valley is the minimum of the smoothed series.
        assert!(curve.raw_cosine_micro[5].is_some() && curve.smoothed_cosine_micro[5].is_some());
        if d.score == Score::Depth {
            assert_eq!(curve.cheapest_gap(), Some(5));
            assert_eq!(curve.topic_shift_gaps().collect::<Vec<_>>(), vec![5]);
            assert_eq!(curve.reason[5], Cut::TopicShift);
            assert!(curve.depth_score_micro[5] > 0);
            let min_smoothed = curve.smoothed_cosine_micro.iter().flatten().min().unwrap();
            assert_eq!(curve.smoothed_cosine_micro[5], Some(*min_smoothed));
        }
    }
    // No sentences at all: every series is empty.
    let empty = Vectors::<Sentences>::new(2, vec![], vec![], model());
    let curve = BoundaryDriftCurve::compute(&empty, &Drift::default());
    assert!(curve.raw_cosine_micro.is_empty() && curve.cost.is_empty() && curve.cheapest_gap().is_none());
}
