//! A single query is one vector: a slice of several is refused rather than
//! answered for its last vector.

mod common;

use common::{index_of, space};
use turbo_vector_index::{IndexError, Metric};

#[test]
fn a_single_query_of_several_vectors_is_refused() {
    for dim in [1u32, 2, 5] {
        let sp = space(dim);
        let d = dim as usize;
        let rows: Vec<Vec<f32>> = (0..d).map(|i| (0..d).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect();
        for metric in [Metric::Dot, Metric::Cosine] {
            for index in [index_of(&sp, metric, &rows), index_of(&sp, metric, &[])] {
                for n in [2usize, 3] {
                    let query: Vec<f32> = (0..n * d).map(|i| (i % d + 1) as f32).collect();
                    for k in [0usize, 1, 2, usize::MAX] {
                        assert_eq!(
                            index.top_k(&sp, &query, k),
                            Err(IndexError::Dimension { index: dim, other: n * d }),
                            "dim {dim}, {metric:?}, {n} vectors, k {k}"
                        );
                    }
                    assert_eq!(index.top_k_batch(&sp, &query, 1).unwrap().len(), n);
                }
            }
        }
    }
}
