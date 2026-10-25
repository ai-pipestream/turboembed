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
