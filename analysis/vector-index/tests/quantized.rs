//! TurboQuant storage: the same queries and refusals as the exact index,
//! estimates within the paper's distortion bounds, and codes pinned to
//! their bytes.

mod common;

use common::{index_of, model, reference, space, values};
use turbo_vector_index::{
    Estimator, Hit, IndexBuilder, IndexError, Metric, Normalization, QuantizedIndex, Space, TurboQuant,
};

const ESTIMATORS: [Estimator; 2] = [Estimator::Mse, Estimator::InnerProduct];

fn unit(v: Vec<f32>) -> Vec<f32> {
    let n = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>().sqrt();
    v.iter().map(|&x| (f64::from(x) / n) as f32).collect()
}

fn quantized(space: &Space, metric: Metric, rows: &[Vec<f32>], config: TurboQuant) -> QuantizedIndex<usize> {
    let mut b = IndexBuilder::new(space.clone(), metric).unwrap();
    for (i, r) in rows.iter().enumerate() {
        b.push(r, i).unwrap();
    }
    b.build_quantized(config)
}

#[test]
fn bit_widths_outside_1_to_4_are_refused() {
    for bits in [0, 5, 8] {
        assert_eq!(TurboQuant::new(bits, Estimator::Mse, 7), Err(IndexError::Bits(bits)));
    }
    for bits in 1..=4 {
        assert!(TurboQuant::new(bits, Estimator::InnerProduct, 7).is_ok());
    }
}

#[test]
fn a_query_from_another_space_is_refused_as_the_exact_index_refuses_it() {
    let sp = space(24);
    let index =
        quantized(&sp, Metric::Dot, &[values(1, 24), values(2, 24)], TurboQuant::new(3, Estimator::Mse, 1).unwrap());
    let q = values(3, 24);
    let other = Space::new(model("bge-small"), 24, Normalization::None);
    assert!(matches!(index.top_k(&other, &q, 1), Err(IndexError::Model { .. })));
    let wide = Space::new(sp.model.clone(), 32, Normalization::None);
    assert_eq!(index.top_k(&wide, &values(3, 32), 1), Err(IndexError::Dimension { index: 24, other: 32 }));
    let normed = Space::new(sp.model.clone(), 24, Normalization::Unit);
    assert!(matches!(index.top_k(&normed, &q, 1), Err(IndexError::Normalization { .. })));
    assert!(matches!(index.score(&sp, &[0.0; 24], 0), Err(IndexError::Zero(_))));
    assert!(matches!(index.top_k_batch(&sp, &values(4, 25), 1), Err(IndexError::Dimension { .. })));
}

#[test]
fn hits_carry_the_score_of_their_row_and_rank_like_a_full_sort() {
    let sp = space(37);
    let rows: Vec<Vec<f32>> = (0..300).map(|i| values(i + 1, 37)).collect();
    let queries: Vec<f32> = (0..3).flat_map(|i| values(1000 + i, 37)).collect();
    for metric in [Metric::Dot, Metric::Cosine] {
        for estimator in ESTIMATORS {
            let index = quantized(&sp, metric, &rows, TurboQuant::new(2, estimator, 9).unwrap());
            let batch = index.top_k_batch(&sp, &queries, 12).unwrap();
            for (q, hits) in queries.chunks(37).zip(&batch) {
                assert_eq!(&index.top_k(&sp, q, 12).unwrap(), hits);
                let mut all: Vec<Hit> =
                    (0..300).map(|row| Hit { row, score: index.score(&sp, q, row).unwrap() }).collect();
                all.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
                all.truncate(12);
                assert_eq!(&all, hits);
            }
            assert!(index.top_k(&sp, &queries[..37], 0).unwrap().is_empty());
            assert_eq!(index.top_k(&sp, &queries[..37], 1000).unwrap().len(), 300);
        }
    }
}

/// The mean and mean square of `estimate - exact` over every query and
/// row, for unit rows and queries of dimension `dim`.
fn errors(dim: u32, config: TurboQuant, metric: Metric) -> (f64, f64) {
    let sp = Space::new(model("e5-small"), dim, Normalization::Unit);
    let rows: Vec<Vec<f32>> = (0..400).map(|i| unit(values(i + 1, dim as usize))).collect();
    let index = quantized(&sp, metric, &rows, config);
    let (mut sum, mut squares, mut n) = (0.0, 0.0, 0.0);
    for i in 0..20 {
        let q = unit(values(5000 + i, dim as usize));
        for hit in index.top_k(&sp, &q, rows.len()).unwrap() {
            let e = f64::from(hit.score) - reference(metric, &q, &rows[hit.row as usize]);
            sum += e;
            squares += e * e;
            n += 1.0;
        }
    }
    (sum / n, squares / n)
}

#[test]
fn inner_product_estimates_stay_within_the_paper_bound_and_are_unbiased() {
    // Theorem 2: E[(<y, x~> - <y, x>)^2] <= sqrt(3) pi^2 |y|^2 / d * 4^-b for
    // unit x, with d the rotated (padded) dimension.
    for dim in [100u32, 128] {
        let d = 128.0;
        let mut last = f64::INFINITY;
        for bits in 1..=4u8 {
            let (mean, mse) = errors(dim, TurboQuant::new(bits, Estimator::InnerProduct, 11).unwrap(), Metric::Dot);
            let bound = 3f64.sqrt() * std::f64::consts::PI.powi(2) / d * 4f64.powi(-i32::from(bits));
            assert!(mse <= bound, "dim {dim} bits {bits}: mse {mse} over the bound {bound}");
            assert!(mse < last, "dim {dim} bits {bits}: mse {mse} not below {last}");
            // 8000 estimates: the mean of an unbiased estimate is within a
            // few standard errors of 0.
            assert!(mean.abs() < 4.0 * (mse / 8000.0).sqrt(), "dim {dim} bits {bits}: mean error {mean}");
            last = mse;
        }
    }
}

#[test]
fn grid_estimates_stay_within_the_paper_bound() {
    // Theorem 1: |x - x~|^2 <= sqrt(3) pi / 2 * 4^-b for unit x; a random
    // unit query sees 1/d of it in its dot product.
    let d = 128.0;
    let mut last = f64::INFINITY;
    for bits in 1..=4u8 {
        let (_, mse) = errors(100, TurboQuant::new(bits, Estimator::Mse, 11).unwrap(), Metric::Dot);
        let bound = 3f64.sqrt() * std::f64::consts::PI / 2.0 / d * 4f64.powi(-i32::from(bits));
        assert!(mse <= bound, "bits {bits}: mse {mse} over the bound {bound}");
        assert!(mse < last);
        last = mse;
    }
}

#[test]
fn a_dimension_that_is_not_a_power_of_two_scores_like_its_padding() {
    for dim in [1u32, 5, 9, 100, 300] {
        let sp = space(dim);
        let rows: Vec<Vec<f32>> = (0..50).map(|i| values(i + 1, dim as usize)).collect();
        let exact = index_of(&sp, Metric::Cosine, &rows);
        let index = quantized(&sp, Metric::Cosine, &rows, TurboQuant::new(4, Estimator::InnerProduct, 3).unwrap());
        let q = values(77, dim as usize);
        let top = index.top_k(&sp, &q, 50).unwrap();
        assert_eq!(top.len(), 50);
        assert!(top.iter().all(|h| h.score.is_finite() && (-1.0..=1.0).contains(&h.score)));
        let best = exact.top_k(&sp, &q, 1).unwrap()[0].row;
        assert!(top[..5].iter().any(|h| h.row == best), "dim {dim}: exact best {best} not in the top 5");
    }
}

/// FNV-1a over every row's codes, then over the index's scores of one
/// query, bit for bit.
fn fingerprint(index: &QuantizedIndex<usize>, sp: &Space, q: &[f32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut eat = |b: u8| {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    };
    for row in 0..index.len() as u32 {
        index.codes(row).iter().for_each(|&b| eat(b));
        index.score(sp, q, row).unwrap().to_bits().to_le_bytes().into_iter().for_each(&mut eat);
    }
    h
}

#[test]
fn codes_and_scores_are_pinned_to_their_bytes() {
    let sp = space(100);
    let rows: Vec<Vec<f32>> = (0..64).map(|i| values(i + 1, 100)).collect();
    let q = values(9999, 100);
    let mut got = Vec::new();
    for estimator in ESTIMATORS {
        for bits in 1..=4 {
            let config = TurboQuant::new(bits, estimator, 0x5eed).unwrap();
            let a = quantized(&sp, Metric::Dot, &rows, config);
            let b = quantized(&sp, Metric::Dot, &rows, config);
            assert_eq!(fingerprint(&a, &sp, &q), fingerprint(&b, &sp, &q));
            assert_eq!(a.bytes_per_row(), 128 * bits as usize / 8 + 8);
            got.push(fingerprint(&a, &sp, &q));
        }
    }
    let pinned: [u64; 8] = [
        0x8384_1dea_6473_6584,
        0xa9ab_6d8c_ac02_09d0,
        0xe244_4086_3297_062e,
        0xa8d8_c561_c8cb_c886,
        0x38e5_8d2f_d4a5_fcf9,
        0x2b82_0cd7_4e4c_5c18,
        0x943d_f151_f8c9_3b93,
        0x5685_6cfa_09fa_5e28,
    ];
    assert_eq!(got, pinned, "codes or scores changed: {got:#018x?}");
}
