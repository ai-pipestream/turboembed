//! Queries from several threads at once answer what one thread does.

mod common;

use common::{index_of, space, values};
use turbo_vector_index::Metric;

#[test]
fn concurrent_queries_return_the_single_thread_result() {
    let dim = 384;
    let sp = space(dim as u32);
    let rows: Vec<Vec<f32>> = (0..3000).map(|i| values(i, dim)).collect();
    let index = index_of(&sp, Metric::Cosine, &rows);
    let queries: Vec<Vec<f32>> = (0..64).map(|i| values(10_000 + i, dim)).collect();
    let want: Vec<_> = queries.iter().map(|q| index.top_k(&sp, q, 25).unwrap()).collect();
    let batch: Vec<f32> = queries.iter().flatten().copied().collect();
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..8)
            .map(|t| {
                let (index, sp, queries, batch) = (&index, &sp, &queries, &batch);
                s.spawn(move || {
                    let mut got = Vec::new();
                    for round in 0..4 {
                        if (t + round) % 2 == 0 {
                            got.push(index.top_k_batch(sp, batch, 25).unwrap());
                        } else {
                            got.push(queries.iter().map(|q| index.top_k(sp, q, 25).unwrap()).collect());
                        }
                    }
                    got
                })
            })
            .collect();
        for h in handles {
            for got in h.join().unwrap() {
                assert_eq!(got, want);
            }
        }
    });
}
