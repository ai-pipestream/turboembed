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
fn finite_inputs_with_unrepresentable_projection_are_refused() {
    // Two rows in two dimensions select covariance; three select Gram.
    for dim in [2, 3] {
        let positive = vec![f32::MAX; dim];
        let negative = vec![-f32::MAX; dim];
        let input = vectors::<Chunks>(dim as u32, &[&positive, &negative], &[]);
        for dims in [ProjectionDims::D2, ProjectionDims::D3] {
            match VectorProjection::compute(&input, dims) {
                Err(error) => assert!(error.contains("row 0 projection coordinate 0"), "{error}"),
                Ok(_) => panic!("unrepresentable projected coordinates accepted"),
            }
        }
    }
}

#[test]
fn large_finite_representable_projections_are_preserved() {
    for dim in [2, 3] {
        let positive = vec![f32::MAX / 4.0; dim];
        let negative = vec![-f32::MAX / 4.0; dim];
        let input = vectors::<Chunks>(dim as u32, &[&positive, &negative], &[]);
        for dims in [ProjectionDims::D2, ProjectionDims::D3] {
            let projection = VectorProjection::compute(&input, dims).unwrap();
            assert!(projection.coords.iter().all(|coordinate| coordinate.is_finite()));
            assert!(projection.coords[0] > 0.0);
            assert!((projection.explained_variance[0] - 1.0).abs() < 1e-6);
            assert!((projection.adjacent_cosine[0].unwrap() + 1.0).abs() < 1e-6);
        }
    }
}
