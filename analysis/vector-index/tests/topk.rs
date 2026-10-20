//! Top-k against a full sort of every row's score, on random and
//! adversarial rows.

mod common;

use common::{brute_force, index_of, reference, space, values};
use turbo_vector_index::{Hit, Metric};

fn rows(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    (0..n).map(|i| values(seed + i as u64, dim)).collect()
}

fn same(a: &[Hit], b: &[Hit]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.row == y.row && x.score.to_bits() == y.score.to_bits())
}

#[test]
fn top_k_is_the_head_of_a_full_sort_on_random_rows() {
    for (dim, n) in [(384usize, 1500usize), (1024, 600), (7, 300), (33, 2000)] {
        for metric in [Metric::Dot, Metric::Cosine] {
            let sp = space(dim as u32);
            let rs = rows(n, dim, 7 * dim as u64);
            let index = index_of(&sp, metric, &rs);
            for (qi, k) in [1usize, 5, 10, 100, n - 1, n, n + 5].into_iter().enumerate() {
                let q = values(90_000 + qi as u64, dim);
                let got = index.top_k(&sp, &q, k).unwrap();
                let want = brute_force(&index, &sp, &q, k);
                assert_eq!(got.len(), k.min(n));
                assert!(same(&got, &want), "dim {dim}, {metric:?}, k {k}");
            }
        }
    }
}

#[test]
fn a_score_is_the_metric_to_within_f32_rounding() {
    let dim = 1024;
    let sp = space(dim as u32);
    let rs = rows(200, dim, 3);
    for metric in [Metric::Dot, Metric::Cosine] {
        let index = index_of(&sp, metric, &rs);
        let q = values(77, dim);
        for (row, r) in rs.iter().enumerate() {
            let got = f64::from(index.score(&sp, &q, row as u32).unwrap());
            let want = reference(metric, &q, r);
            let scale = match metric {
                Metric::Dot => 1.0 + want.abs(),
                Metric::Cosine => 1.0,
            };
            assert!((got - want).abs() <= 2e-5 * scale, "{metric:?} row {row}: {got} against {want}");
        }
    }
}

#[test]
fn equal_scores_keep_insertion_order() {
    // Duplicates, a row of the query's direction scaled (equal cosines),
    // and rows orthogonal to the query whose scores are 0.0 and -0.0.
    let sp = space(4);
    let rs = vec![
        vec![0.0, 1.0, 0.0, 0.0],
        vec![1.0, 2.0, 3.0, 4.0],
        vec![0.0, -1.0, 0.0, 0.0],
        vec![2.0, 4.0, 6.0, 8.0],
        vec![1.0, 2.0, 3.0, 4.0],
        vec![0.0, 0.0, 0.0, 5.0],
        vec![-1.0, 0.0, 0.0, 0.0],
        vec![1.0, 2.0, 3.0, 4.0],
    ];
    let q = [1.0, 0.0, 0.0, 0.0];
    let q2 = [1.0, 2.0, 3.0, 4.0];
    let cos = index_of(&sp, Metric::Cosine, &rs);
    let order = |hits: Vec<Hit>| hits.iter().map(|h| h.row).collect::<Vec<_>>();
    assert_eq!(order(cos.top_k(&sp, &q2, 4).unwrap()), [1, 3, 4, 7]);
    // Rows 0, 2 and 5 score 0 against q (2 as -0.0), so they keep their order.
    assert_eq!(order(cos.top_k(&sp, &q, 8).unwrap()), [1, 3, 4, 7, 0, 2, 5, 6]);
    let dot = index_of(&sp, Metric::Dot, &rs);
    assert_eq!(order(dot.top_k(&sp, &q2, 3).unwrap()), [3, 1, 4]);
    for k in 1..=9 {
        for query in [&q, &q2] {
            assert!(same(&cos.top_k(&sp, query, k).unwrap(), &brute_force(&cos, &sp, query, k)), "k {k}");
            assert!(same(&dot.top_k(&sp, query, k).unwrap(), &brute_force(&dot, &sp, query, k)), "k {k}");
        }
    }
}

#[test]
fn many_equal_rows_tie_in_insertion_order_across_tiles() {
    // More rows than one scan tile, all the same, so every score ties.
    let dim = 64;
    let sp = space(dim as u32);
    let r = values(5, dim);
    let rs = vec![r.clone(); 5000];
    let index = index_of(&sp, Metric::Dot, &rs);
    let hits = index.top_k(&sp, &values(6, dim), 300).unwrap();
    assert!(hits.iter().enumerate().all(|(i, h)| h.row == i as u32));
}

#[test]
fn k_zero_and_an_empty_index_answer_no_hits() {
    let sp = space(8);
    let index = index_of(&sp, Metric::Cosine, &[]);
    assert!(index.is_empty());
    assert_eq!(index.top_k(&sp, &values(1, 8), 10).unwrap(), []);
    let index = index_of(&sp, Metric::Cosine, &rows(10, 8, 1));
    assert_eq!(index.top_k(&sp, &values(1, 8), 0).unwrap(), []);
}

#[test]
fn a_batch_answers_what_each_query_alone_does() {
    for dim in [384usize, 1024, 20] {
        let sp = space(dim as u32);
        let index = index_of(&sp, Metric::Cosine, &rows(700, dim, 11));
        for nq in [1usize, 3, 4, 5, 16, 17, 40] {
            let qs = values(500 + nq as u64, nq * dim);
            let batch = index.top_k_batch(&sp, &qs, 10).unwrap();
            assert_eq!(batch.len(), nq);
            for (q, hits) in qs.chunks(dim).zip(&batch) {
                assert!(same(hits, &index.top_k(&sp, q, 10).unwrap()), "dim {dim}, {nq} queries");
            }
        }
    }
}

#[test]
fn self_retrieval_ranks_the_row_first_and_scaling_changes_no_cosine() {
    let dim = 384;
    let sp = space(dim as u32);
    let rs = rows(400, dim, 21);
    let index = index_of(&sp, Metric::Cosine, &rs);
    for row in [0usize, 17, 399] {
        let hits = index.top_k(&sp, &rs[row], 3).unwrap();
        assert_eq!(hits[0].row, row as u32);
        let scaled: Vec<f32> = rs[row].iter().map(|x| x * 1024.0).collect();
        assert!(same(&index.top_k(&sp, &scaled, 3).unwrap(), &hits));
        let tiny: Vec<f32> = rs[row].iter().map(|x| x * 1e-30).collect();
        assert_eq!(index.top_k(&sp, &tiny, 1).unwrap()[0].row, row as u32);
    }
}
