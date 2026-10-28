//! The cos/sin table of rotary position embeddings (#1461), built once
//! per model by the core and handed to every backend in
//! turbo_backend_model.rope_table, so that every device rotates Q and K
//! with the same bits and no backend has trigonometry of its own.
//!
//! Entry (p, i) of a base's table is cos and sin of `p * inv(i)`, with
//! `inv(i) = 1 / theta^(2i / d)` for a head of width d, as transformers'
//! default RoPE computes it (modeling_rope_utils.py, ModernBERT's and
//! nomic-bert's rotary embedding): the exponent, the inverse frequency
//! and the angle are each rounded to F32 as upstream rounds them, since
//! at position 8192 the angle's F32 rounding moves cos and sin by far
//! more than any later step does; cos and sin of that angle are then
//! taken in F64 and rounded once. Column i of a head rotates with column
//! `i + d / 2` (rotate_half).

/// The table: `[2 (global, local)][2 (cos, sin)][positions][pairs]`,
/// the local half computed from `local` (the global base again for a
/// model without sliding-window layers). Err is the bytes it could not
/// allocate.
pub fn table(global: f64, local: f64, positions: usize, pairs: usize) -> Result<Vec<f32>, usize> {
    let half = positions.checked_mul(pairs).ok_or(usize::MAX)?;
    let len = half.checked_mul(4).ok_or(usize::MAX)?;
    let mut out = Vec::new();
    out.try_reserve_exact(len).map_err(|_| len.saturating_mul(size_of::<f32>()))?;
    let width = (2 * pairs) as f32;
    for theta in [global, local] {
        let inv: Vec<f32> =
            (0..pairs).map(|i| 1.0 / ((theta as f32) as f64).powf(((2 * i) as f32 / width) as f64) as f32).collect();
        let angle = |p: usize, i: usize| (p as f32 * inv[i]) as f64;
        for p in 0..positions {
            out.extend((0..pairs).map(|i| angle(p, i).cos() as f32));
        }
        for p in 0..positions {
            out.extend((0..pairs).map(|i| angle(p, i).sin() as f32));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_is_laid_out_by_base_then_cos_and_sin() {
        let (positions, pairs) = (5, 3);
        let t = table(160000.0, 10000.0, positions, pairs).unwrap();
        assert_eq!(t.len(), 4 * positions * pairs);
        let at = |base: usize, sin: usize, p: usize, i: usize| t[((base * 2 + sin) * positions + p) * pairs + i];
        for base in 0..2 {
            for i in 0..pairs {
                // Position 0 is no rotation.
                assert_eq!((at(base, 0, 0, i), at(base, 1, 0, i)), (1.0, 0.0));
            }
            // Pair 0's frequency is 1 for every base.
            assert_eq!(at(base, 0, 3, 0), (3.0f64).cos() as f32);
            assert_eq!(at(base, 1, 3, 0), (3.0f64).sin() as f32);
        }
        // The pairs past 0 turn slower under the larger base.
        assert!(at(0, 1, 4, 2) < at(1, 1, 4, 2));
    }

    /// Within F32 rounding of the angle of the F64 formula, at the
    /// positions a long row reaches.
    #[test]
    fn the_table_follows_the_rope_formula() {
        let (theta, positions, pairs) = (160000.0f64, 8192, 32);
        let t = table(theta, theta, positions, pairs).unwrap();
        for p in [1, 63, 64, 2047, 4096, 8191] {
            for i in [0, 1, 15, 31] {
                let angle = p as f64 * theta.powf(-((2 * i) as f64) / (2 * pairs) as f64);
                // The F32 angle is within an ulp or two of the F64 one.
                let tol = 4.0 * (p as f64) * f32::EPSILON as f64 + 1e-7;
                let cos = t[p * pairs + i] as f64;
                let sin = t[(positions + p) * pairs + i] as f64;
                assert!((cos - angle.cos()).abs() <= tol, "cos p {p} i {i}: {cos} vs {}", angle.cos());
                assert!((sin - angle.sin()).abs() <= tol, "sin p {p} i {i}: {sin} vs {}", angle.sin());
            }
        }
    }

    #[test]
    fn a_table_too_large_to_hold_says_so() {
        assert_eq!(table(1000.0, 1000.0, usize::MAX / 2, 32), Err(usize::MAX));
    }
}
