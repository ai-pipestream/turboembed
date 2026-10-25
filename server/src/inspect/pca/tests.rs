use turbo_analysis::{ModelRef, Span};
use turbo_analysis_layers::Chunks;
use turbo_analysis_layers::Vectors;

use super::compute_pca;
use crate::offsets::{OffsetMode, Offsetter};

fn model() -> ModelRef {
    ModelRef { id: "m".into(), revision: "r".into(), sha256: "s".into() }
}

fn project(rows: &[[f32; 4]], k: u32) -> Vec<(f32, f32)> {
    let text = "abcdefghij";
    let data: Vec<f32> = rows.iter().flatten().copied().collect();
    let vectors = Vectors::<Chunks>::new(4, data, Vec::new(), model());
    let spans: Vec<Span> = (0..rows.len() as u32).map(|i| Span::new(i, i + 1)).collect();
    let levels = vec![0u8; rows.len()];
    let off = Offsetter::new(text, OffsetMode::Utf8);
    let p = compute_pca(&vectors, &spans, &levels, text, &off, k).unwrap();
    p.points.iter().map(|q| (q.x, q.y)).collect()
}

#[test]
fn pca_is_scale_equivariant() {
    let base = [[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0], [1.0, 1.0, 1.0, 0.5]];
    let s = 1e-7_f32;
    let small: Vec<[f32; 4]> = base.iter().map(|r| r.map(|x| x * s)).collect();
    let big = project(&base, 2);
    let tiny = project(&small, 2);
    for ((bx, by), (tx, ty)) in big.iter().zip(&tiny) {
        assert!((bx * s - tx).abs() <= 1e-3 * s * bx.abs().max(1.0), "x {bx} vs {tx}");
        assert!((by * s - ty).abs() <= 1e-3 * s * by.abs().max(1.0), "y {by} vs {ty}");
    }
}

#[test]
fn gram_and_covariance_three_axis_projection_preserve_scale_and_variance() {
    let rows = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 0.0],
        [1.0, 1.0, 1.0, 0.5],
        [-1.0, 0.0, 0.5, 0.0],
        [2.0, -1.0, 0.0, 0.25],
    ];
    let text = "abcdef";
    let offsetter = Offsetter::new(text, OffsetMode::Utf8);
    // Four rows take the Gram path; six rows take the covariance path.
    for n in [4, 6] {
        let spans: Vec<_> = (0..n as u32).map(|i| Span::new(i, i + 1)).collect();
        let levels = vec![0; n];
        let run = |scale: f32| {
            let data = rows[..n].iter().flatten().map(|x| x * scale).collect();
            let vectors = Vectors::<Chunks>::new(4, data, Vec::new(), model());
            compute_pca(&vectors, &spans, &levels, text, &offsetter, 3).unwrap()
        };
        let base = run(1.0);
        assert_eq!(base.points.len(), n);
        assert!(base.explained_variance.iter().all(|&v| v > 0.0));
        for scale in [1e-20, 1e-7, 1e20] {
            let scaled = run(scale);
            for (a, b) in base.points.iter().zip(&scaled.points) {
                for (original, actual) in [a.x, a.y, a.z].into_iter().zip([b.x, b.y, b.z]) {
                    let rescaled = actual / scale;
                    assert!((original - rescaled).abs() < 1e-4, "n={n}, scale={scale}: {original} vs {rescaled}");
                }
                if let (Some(a), Some(b)) = (a.adjacent_cosine, b.adjacent_cosine) {
                    assert!((a - b).abs() < 1e-6);
                } else {
                    assert_eq!(a.adjacent_cosine, b.adjacent_cosine);
                }
            }
            for (&a, &b) in base.explained_variance.iter().zip(&scaled.explained_variance) {
                assert!((a - b).abs() < 1e-6, "n={n}, scale={scale}: variance {a} vs {b}");
            }
        }
    }
}
