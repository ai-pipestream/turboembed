//! The kernel paths against the scalar one, bit for bit.

use turbo_analysis::ModelRef;

use crate::kernel::{self, Path};
use crate::{IndexBuilder, Metric, Normalization, Space};

/// A fixed sequence of values in `[-1, 1)`, from a 64-bit LCG.
fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            ((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0
        })
        .collect()
}

fn space(dim: u32) -> Space {
    let model = ModelRef { id: "m".into(), revision: "1".into(), sha256: "00".into() };
    Space::new(model, dim, Normalization::None)
}

#[test]
fn every_path_scores_the_same_bits_as_the_scalar_one() {
    let paths = kernel::available();
    for dim in (1..=40).chain([47, 63, 64, 65, 384, 1024]) {
        for metric in [Metric::Dot, Metric::Cosine] {
            let n = 1 + (dim as usize * 7) % 23;
            let mut b = IndexBuilder::new(space(dim), metric).unwrap();
            for r in 0..n {
                b.push(&values(r as u64 + 1, dim as usize), r).unwrap();
            }
            let index = b.build();
            let nq = 1 + dim as usize % 9;
            let queries = values(1000 + dim as u64, nq * dim as usize);
            let sp = space(dim);
            let index = index.with_path(Path::Scalar);
            let want = index.top_k_batch(&sp, &queries, n).unwrap();
            let mut index = index;
            for &p in &paths {
                index = index.with_path(p);
                let got = index.top_k_batch(&sp, &queries, n).unwrap();
                for (w, g) in want.iter().zip(&got) {
                    let wb: Vec<(u32, u32)> = w.iter().map(|h| (h.row, h.score.to_bits())).collect();
                    let gb: Vec<(u32, u32)> = g.iter().map(|h| (h.row, h.score.to_bits())).collect();
                    assert_eq!(wb, gb, "{p:?}, dim {dim}, {metric:?}, {nq} queries, {n} rows");
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn the_avx512_path_needs_avx2_and_fma_as_well() {
    use crate::kernel::paths_for;
    assert_eq!(paths_for(true, false, false), [Path::Scalar]);
    assert_eq!(paths_for(true, true, false), [Path::Scalar]);
    assert_eq!(paths_for(true, false, true), [Path::Scalar]);
    assert_eq!(paths_for(false, true, true), [Path::Scalar, Path::Avx2]);
    assert_eq!(paths_for(true, true, true), [Path::Scalar, Path::Avx2, Path::Avx512]);
}
