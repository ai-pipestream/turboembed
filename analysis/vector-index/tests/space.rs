//! Vectors from another model, dimension or normalization, and vectors
//! that cannot be scored, are refused with the named error.

mod common;

use common::{index_of, model, space, values};
use turbo_analysis::Ix;
use turbo_analysis_layers::{Sentences, Vectors};
use turbo_vector_index::{IndexBuilder, IndexError, MAX_LENGTH, Metric, Normalization, Space, Which};

fn unit(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>().sqrt();
    v.iter().map(|&x| (f64::from(x) / n) as f32).collect()
}

#[test]
fn a_query_from_another_model_dimension_or_normalization_is_refused() {
    let sp = space(16);
    let index = index_of(&sp, Metric::Cosine, &[values(1, 16), values(2, 16)]);
    let q = values(3, 16);
    let other = Space::new(model("bge-small"), 16, Normalization::None);
    assert_eq!(
        index.top_k(&other, &q, 1),
        Err(IndexError::Model { index: Box::new(sp.model.clone()), other: Box::new(other.model.clone()) })
    );
    let mut revised = sp.clone();
    revised.model.revision = "2".into();
    assert!(matches!(index.top_k(&revised, &q, 1), Err(IndexError::Model { .. })));
    let mut resha = sp.clone();
    resha.model.sha256 = "other".into();
    assert!(matches!(index.top_k_batch(&resha, &q, 1), Err(IndexError::Model { .. })));
    let wide = Space::new(sp.model.clone(), 32, Normalization::None);
    assert_eq!(index.top_k(&wide, &values(3, 32), 1), Err(IndexError::Dimension { index: 16, other: 32 }));
    assert_eq!(index.top_k(&sp, &values(3, 15), 1), Err(IndexError::Dimension { index: 16, other: 15 }));
    let normed = Space::new(sp.model.clone(), 16, Normalization::Unit);
    assert_eq!(
        index.top_k(&normed, &unit(&q), 1),
        Err(IndexError::Normalization { index: Normalization::None, other: Normalization::Unit })
    );
    assert!(matches!(index.score(&other, &q, 0), Err(IndexError::Model { .. })));
    // A batch whose length is not a whole number of queries.
    assert_eq!(index.top_k_batch(&sp, &values(3, 40), 1), Err(IndexError::Dimension { index: 16, other: 40 }));
    assert_eq!(index.top_k_batch(&sp, &[], 1), Err(IndexError::Dimension { index: 16, other: 0 }));
}

#[test]
fn zero_non_finite_and_overlong_vectors_are_refused_on_insert_and_query() {
    let sp = space(4);
    let mut b = IndexBuilder::new(sp.clone(), Metric::Dot).unwrap();
    assert_eq!(b.push(&[0.0; 4], ()), Err(IndexError::Zero(Which::Row(0))));
    assert_eq!(b.push(&[0.0, f32::NAN, 1.0, 0.0], ()), Err(IndexError::NonFinite(Which::Row(0), 1)));
    assert_eq!(b.push(&[1.0, 0.0, 0.0, f32::INFINITY], ()), Err(IndexError::NonFinite(Which::Row(0), 3)));
    let huge = [f32::MAX, 0.0, 0.0, 0.0];
    assert!(matches!(b.push(&huge, ()), Err(IndexError::TooLong(Which::Row(0), l)) if l > MAX_LENGTH));
    assert_eq!(b.push(&[1.0; 3], ()), Err(IndexError::Dimension { index: 4, other: 3 }));
    assert!(b.is_empty());
    assert_eq!(b.push(&[1.0, 2.0, 3.0, 4.0], ()), Ok(0));
    let index = b.build();
    assert_eq!(index.len(), 1);
    assert_eq!(index.top_k(&sp, &[0.0; 4], 1), Err(IndexError::Zero(Which::Query(0))));
    assert_eq!(index.top_k(&sp, &[f32::NAN, 0.0, 0.0, 1.0], 1), Err(IndexError::NonFinite(Which::Query(0), 0)));
    assert!(matches!(index.top_k(&sp, &huge, 1), Err(IndexError::TooLong(Which::Query(0), _))));
    let batch = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    assert_eq!(index.top_k_batch(&sp, &batch, 1), Err(IndexError::Zero(Which::Query(1))));
    assert_eq!(IndexBuilder::<()>::new(space(0), Metric::Dot).err(), Some(IndexError::ZeroDimension));
}

#[test]
fn a_unit_space_refuses_vectors_that_are_not_unit_length() {
    let sp = Space::new(model("e5-small"), 8, Normalization::Unit);
    let mut b = IndexBuilder::new(sp.clone(), Metric::Dot).unwrap();
    let raw = values(9, 8);
    assert!(matches!(b.push(&raw, ()), Err(IndexError::NotUnit(Which::Row(0), _))));
    b.push(&unit(&raw), ()).unwrap();
    let index = b.build();
    assert!(matches!(index.top_k(&sp, &values(4, 8), 1), Err(IndexError::NotUnit(Which::Query(0), _))));
    assert_eq!(index.top_k(&sp, &unit(&values(4, 8)), 1).unwrap().len(), 1);
}

fn layer(dim: u32, rows: &[Vec<f32>], empty: &[u32]) -> Vectors<Sentences> {
    let data = rows.iter().flatten().copied().collect();
    Vectors::new(dim, data, empty.iter().map(|&i| Ix::new(i)).collect(), model("e5-small"))
}

#[test]
fn a_layer_adds_the_items_that_have_vectors() {
    let sp = space(4);
    let rows = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0; 4], vec![0.0, 1.0, 0.0, 0.0], vec![0.0; 4]];
    let vectors = layer(4, &rows, &[1, 3]);
    let mut b = IndexBuilder::new(Space::of(&vectors, Normalization::None), Metric::Cosine).unwrap();
    assert_eq!(b.push_layer(&vectors, |ix| ix.get()), Ok(2));
    let index = b.build();
    assert_eq!((index.len(), *index.meta(0), *index.meta(1)), (2, 0, 2));
    let hit = index.top_k(&sp, &[0.0, 3.0, 0.0, 0.0], 1).unwrap()[0];
    assert_eq!(*index.meta(hit.row), 2);
    // An empty item of the layer is no query either.
    let empty = vectors.row(Ix::new(1));
    assert_eq!(
        index.top_k(&Space::of(&vectors, Normalization::None), empty, 1),
        Err(IndexError::Zero(Which::Query(0)))
    );
}

#[test]
fn a_layer_from_another_model_or_dimension_or_with_a_bad_row_adds_nothing() {
    let sp = space(4);
    let mut b = IndexBuilder::new(sp.clone(), Metric::Cosine).unwrap();
    b.push(&[1.0, 1.0, 0.0, 0.0], usize::MAX).unwrap();
    let mut other = layer(4, &[vec![1.0, 0.0, 0.0, 0.0]], &[]);
    other.model = model("bge-small");
    assert!(matches!(b.push_layer(&other, |ix| ix.get()), Err(IndexError::Model { .. })));
    let wide = layer(2, &[vec![1.0, 0.0]], &[]);
    assert_eq!(b.push_layer(&wide, |ix| ix.get()), Err(IndexError::Dimension { index: 4, other: 2 }));
    // A zero row the layer does not list as empty is refused, and the rows
    // of the layer before it are taken back out.
    let unlisted = layer(4, &[vec![1.0, 0.0, 0.0, 0.0], vec![0.0; 4]], &[]);
    assert_eq!(b.push_layer(&unlisted, |ix| ix.get()), Err(IndexError::Zero(Which::Row(2))));
    // A layer that fails its own check: an empty item whose row is not zeros.
    let bad = layer(4, &[vec![1.0, 0.0, 0.0, 0.0]], &[0]);
    assert!(matches!(b.push_layer(&bad, |ix| ix.get()), Err(IndexError::Layer(_))));
    assert_eq!(b.len(), 1);
    // A row added after the refusals is the next row, and scores as itself.
    assert_eq!(b.push(&[0.0, 0.0, 2.0, 0.0], 7), Ok(1));
    let index = b.build();
    assert_eq!(index.top_k(&sp, &[1.0, 0.0, 0.0, 0.0], 5).unwrap().len(), 2);
    assert_eq!(index.row(1), [0.0, 0.0, 1.0, 0.0]);
    let hit = index.top_k(&sp, &[0.0, 0.0, 1.0, 0.0], 1).unwrap()[0];
    assert_eq!((hit.row, hit.score, *index.meta(hit.row)), (1, 1.0, 7));
}

#[test]
fn rows_are_copied_and_stored_as_scored() {
    let sp = space(3);
    let mut v = vec![3.0, 0.0, 4.0];
    let mut b = IndexBuilder::new(sp.clone(), Metric::Cosine).unwrap();
    b.push(&v, ()).unwrap();
    v[0] = -100.0;
    let index = b.build();
    assert_eq!(index.row(0), [0.6, 0.0, 0.8]);
    let q = [3.0, 0.0, 4.0];
    let hits = index.top_k(&sp, &q, 1).unwrap();
    assert_eq!((hits[0].row, hits[0].score), (0, 1.0));
    assert_eq!(q, [3.0, 0.0, 4.0]);
}
