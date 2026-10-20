//! TurboQuant storage: the same queries and refusals as the exact index,
//! estimates within the paper's distortion bound, and codes pinned to
//! their bytes.

mod common;

use common::{index_of, model, reference, space, values};
use turbo_vector_index::{Hit, IndexBuilder, IndexError, Metric, Normalization, QuantizedIndex, Space, TurboQuant};

fn unit(v: Vec<f32>) -> Vec<f32> {
    let n = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>().sqrt();
    v.iter().map(|&x| (f64::from(x) / n) as f32).collect()
}

fn quantized(
    space: &Space,
    metric: Metric,
    rows: &[Vec<f32>],
    config: TurboQuant,
) -> Result<QuantizedIndex<usize>, IndexError> {
    let mut b = IndexBuilder::new(space.clone(), metric).unwrap();
    for (i, r) in rows.iter().enumerate() {
        b.push(r, i).unwrap();
    }
    b.build_quantized(config)
}

#[test]
fn bit_widths_outside_2_to_4_are_refused() {
    for bits in [0, 1, 5, 8] {
        assert_eq!(TurboQuant::new(bits, false), Err(IndexError::Bits(bits)));
    }
    for bits in 2..=4 {
        assert!(TurboQuant::new(bits, true).is_ok());
    }
    let rows = [values(1, 16)];
    for bits in [0, 1, 5, 8] {
        let got = quantized(&space(16), Metric::Dot, &rows, TurboQuant { bits, calibrate: false });
        assert_eq!(got.err(), Some(IndexError::Bits(bits)), "built with {bits} bits");
    }
}

#[test]
fn a_dimension_that_is_not_a_multiple_of_8_is_refused_at_build() {
    let config = TurboQuant::new(4, false).unwrap();
    for dim in [1u32, 12, 100] {
        let rows = vec![values(1, dim as usize)];
        let got = quantized(&space(dim), Metric::Dot, &rows, config);
        assert!(matches!(got, Err(IndexError::Quantized(_))), "dim {dim}");
    }
    assert!(quantized(&space(16), Metric::Dot, &[values(1, 16)], config).is_ok());
}

#[test]
fn a_query_from_another_space_is_refused_as_the_exact_index_refuses_it() {
    let sp = space(24);
    let index =
        quantized(&sp, Metric::Dot, &[values(1, 24), values(2, 24)], TurboQuant::new(3, false).unwrap()).unwrap();
    let q = values(3, 24);
    let other = Space::new(model("bge-small"), 24, Normalization::None);
    assert!(matches!(index.top_k(&other, &q, 1), Err(IndexError::Model { .. })));
    let wide = Space::new(sp.model.clone(), 32, Normalization::None);
    assert_eq!(index.top_k(&wide, &values(3, 32), 1), Err(IndexError::Dimension { index: 24, other: 32 }));
    let normed = Space::new(sp.model.clone(), 24, Normalization::Unit);
    assert!(matches!(index.top_k(&normed, &q, 1), Err(IndexError::Normalization { .. })));
    assert!(matches!(index.score(&sp, &[0.0; 24], 0), Err(IndexError::Zero(_))));
    assert!(matches!(index.top_k(&sp, &[f32::NAN; 24], 1), Err(IndexError::NonFinite(..))));
    assert!(matches!(index.top_k_batch(&sp, &values(4, 25), 1), Err(IndexError::Dimension { .. })));
}

#[test]
fn hits_carry_the_score_of_their_row_and_rank_like_a_full_sort() {
    let sp = space(40);
    let rows: Vec<Vec<f32>> = (0..300).map(|i| values(i + 1, 40)).collect();
    let queries: Vec<f32> = (0..3).flat_map(|i| values(1000 + i, 40)).collect();
    for metric in [Metric::Dot, Metric::Cosine] {
        for calibrate in [false, true] {
            let index = quantized(&sp, metric, &rows, TurboQuant::new(2, calibrate).unwrap()).unwrap();
            assert_eq!(index.len(), 300);
            assert_eq!(*index.meta(7), 7);
            let batch = index.top_k_batch(&sp, &queries, 12).unwrap();
            for (q, hits) in queries.chunks(40).zip(&batch) {
                assert_eq!(&index.top_k(&sp, q, 12).unwrap(), hits);
                let mut all: Vec<Hit> =
                    (0..300).map(|row| Hit { row, score: index.score(&sp, q, row).unwrap() }).collect();
                all.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
                let scores = |h: &[Hit]| h.iter().map(|h| h.score).collect::<Vec<_>>();
                assert_eq!(scores(&all[..12]), scores(hits));
                for h in hits {
                    assert_eq!(index.score(&sp, q, h.row).unwrap(), h.score);
                }
            }
            assert!(index.top_k(&sp, &queries[..40], 0).unwrap().is_empty());
            assert_eq!(index.top_k(&sp, &queries[..40], 1000).unwrap().len(), 300);
        }
    }
}

#[test]
fn an_empty_index_answers_with_no_hits() {
    let index = quantized(&space(8), Metric::Dot, &[], TurboQuant::new(4, true).unwrap()).unwrap();
    assert!(index.is_empty());
    assert_eq!(index.top_k_batch(&space(8), &values(1, 16), 5).unwrap(), vec![Vec::<Hit>::new(); 2]);
}

/// The mean and mean square of `estimate - exact` over every query and
/// row, for unit rows and queries of dimension `dim`.
fn errors(dim: u32, config: TurboQuant) -> (f64, f64) {
    let sp = Space::new(model("e5-small"), dim, Normalization::Unit);
    let rows: Vec<Vec<f32>> = (0..400).map(|i| unit(values(i + 1, dim as usize))).collect();
    let index = quantized(&sp, Metric::Dot, &rows, config).unwrap();
    let (mut sum, mut squares, mut n) = (0.0, 0.0, 0.0);
    for i in 0..20 {
        let q = unit(values(5000 + i, dim as usize));
        for hit in index.top_k(&sp, &q, rows.len()).unwrap() {
            let e = f64::from(hit.score) - reference(Metric::Dot, &q, &rows[hit.row as usize]);
            sum += e;
            squares += e * e;
            n += 1.0;
        }
    }
    (sum / n, squares / n)
}

#[test]
fn estimates_stay_within_the_paper_bound_and_are_unbiased() {
    // Theorem 2: E[(<y, x~> - <y, x>)^2] <= sqrt(3) pi^2 |y|^2 / d * 4^-b.
    let d = 128.0;
    let mut last = f64::INFINITY;
    for bits in 2..=4u8 {
        let (mean, mse) = errors(128, TurboQuant::new(bits, false).unwrap());
        let bound = 3f64.sqrt() * std::f64::consts::PI.powi(2) / d * 4f64.powi(-i32::from(bits));
        assert!(mse <= bound, "bits {bits}: mse {mse} over the bound {bound}");
        assert!(mse < last, "bits {bits}: mse {mse} not below {last}");
        assert!(mean.abs() < 4.0 * (mse / 8000.0).sqrt(), "bits {bits}: mean error {mean}");
        last = mse;
    }
}

#[test]
fn the_exact_best_row_is_near_the_top() {
    for dim in [8u32, 96, 384] {
        let sp = space(dim);
        let rows: Vec<Vec<f32>> = (0..64).map(|i| values(i + 1, dim as usize)).collect();
        let exact = index_of(&sp, Metric::Cosine, &rows);
        let index = quantized(&sp, Metric::Cosine, &rows, TurboQuant::new(4, true).unwrap()).unwrap();
        let q = values(77, dim as usize);
        let top = index.top_k(&sp, &q, 64).unwrap();
        assert_eq!(top.len(), 64);
        assert!(top.iter().all(|h| h.score.is_finite() && (-1.0..=1.0).contains(&h.score)));
        let best = exact.top_k(&sp, &q, 1).unwrap()[0].row;
        assert!(top[..5].iter().any(|h| h.row == best), "dim {dim}: exact best {best} not in the top 5");
    }
}

fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3))
}

#[test]
fn codes_are_pinned_to_their_bytes() {
    let sp = space(128);
    let rows: Vec<Vec<f32>> = (0..64).map(|i| values(i + 1, 128)).collect();
    let mut got = Vec::new();
    for calibrate in [false, true] {
        for bits in 2..=4 {
            let config = TurboQuant::new(bits, calibrate).unwrap();
            let a = quantized(&sp, Metric::Dot, &rows, config).unwrap().to_bytes();
            let b = quantized(&sp, Metric::Dot, &rows, config).unwrap().to_bytes();
            assert_eq!(a, b);
            got.push(fnv(&a));
        }
    }
    let pinned: [u64; 6] = [
        0x5ae0_5882_3fc7_3a9e,
        0xbe8b_3bd8_6ff5_c967,
        0xa7b3_ede4_9440_aebe,
        0x9898_dfc1_f3f0_e6cc,
        0xc6c0_2c18_68ab_13cb,
        0xb6f2_1de4_b633_b37e,
    ];
    assert_eq!(got, pinned, "codes changed: {got:#018x?}");
}
