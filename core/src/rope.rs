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

    /// cos and sin as transformers computes them (its default RoPE in
    /// F32: `inv_freq = 1 / theta ** (arange(0, d, 2) / d)`, then
    /// `position * inv_freq`, then cos and sin), from the reference
    /// container's torch 2.14.0+cpu, as F32 bits, at head width 64.
    const UPSTREAM: [(f64, usize, usize, u32, u32); 36] = [
        (160000.0, 1, 1, 0x3f45d205, 0x3f227d83),
        (160000.0, 1, 15, 0x3f7fff91, 0x3b6e4215),
        (160000.0, 1, 31, 0x3f800000, 0x37187c4d),
        (160000.0, 2047, 1, 0x3f7af9c2, 0x3e49e17b),
        (160000.0, 2047, 15, 0x3ecd0af1, 0x3f6a932b),
        (160000.0, 2047, 31, 0x3f7ff4a8, 0x3c9866fd),
        (160000.0, 4096, 1, 0xbe4cc9cb, 0x3f7ad40f),
        (160000.0, 4096, 15, 0xbf2f3f45, 0x3f3a9cf2),
        (160000.0, 4096, 31, 0x3f7fd297, 0x3d187349),
        (160000.0, 8191, 1, 0xbf75afee, 0x3e8fdbaa),
        (160000.0, 8191, 15, 0xbd87f503, 0xbf7f6f6e),
        (160000.0, 8191, 31, 0x3f7f4a79, 0x3d98537e),
        (10000.0, 1, 1, 0x3f3b54b0, 0x3f2e7ace),
        (10000.0, 1, 15, 0x3f7ffa2c, 0x3c5a7a4a),
        (10000.0, 1, 31, 0x3f800000, 0x390bd472),
        (10000.0, 2047, 1, 0xbeb6f76f, 0x3f6f1888),
        (10000.0, 2047, 15, 0xbf0f341c, 0x3f543324),
        (10000.0, 2047, 31, 0x3f768576, 0x3e8a0848),
        (10000.0, 4096, 1, 0x3f1cec81, 0xbf4a43b3),
        (10000.0, 4096, 15, 0xbeb2d854, 0xbf6fdfc9),
        (10000.0, 4096, 31, 0x3f5ac075, 0x3f04fadb),
        (10000.0, 8191, 1, 0xbf5792bd, 0xbf0a12bb),
        (10000.0, 8191, 15, 0xbf3f46b6, 0x3f2a252f),
        (10000.0, 8191, 31, 0x3eebc0e6, 0x3f633f29),
        (1000.0, 1, 1, 0x3f3147ef, 0x3f38ae99),
        (1000.0, 1, 15, 0x3f7fcd8c, 0x3d20b18e),
        (1000.0, 1, 31, 0x3f7ffff3, 0x3aa2a6f4),
        (1000.0, 2047, 1, 0xbf79aa1c, 0xbe62672f),
        (1000.0, 2047, 15, 0x3e5d04c4, 0xbf79f742),
        (1000.0, 2047, 31, 0xbf531599, 0x3f10d7b7),
        (1000.0, 4096, 1, 0xbeef728a, 0x3f624773),
        (1000.0, 4096, 15, 0xbf5ef8e4, 0xbefb8cdf),
        (1000.0, 4096, 31, 0x3eb961d5, 0xbf6ea154),
        (1000.0, 8191, 1, 0xbf7c693c, 0xbe2add27),
        (1000.0, 8191, 15, 0x3f0ce7cc, 0x3f55bb90),
        (1000.0, 8191, 31, 0xbf3d1794, 0xbf2c91c0),
    ];

    /// Within 2 ULP of upstream at every sampled entry, positions to 8191
    /// included: the angle is rounded as upstream rounds it, so only the
    /// last step's cos or sin may differ.
    #[test]
    fn the_table_matches_transformers() {
        let (positions, pairs) = (8192, 32);
        for theta in [160000.0, 10000.0, 1000.0] {
            let t = table(theta, theta, positions, pairs).unwrap();
            for &(_, p, i, cos, sin) in UPSTREAM.iter().filter(|u| u.0 == theta) {
                for (got, want) in [(t[p * pairs + i], cos), (t[(positions + p) * pairs + i], sin)] {
                    let ulps = (got.to_bits() as i64 - want as i64).abs();
                    assert!(ulps <= 2, "theta {theta} p {p} i {i}: {got} is {ulps} ULP from {}", f32::from_bits(want));
                }
            }
        }
    }

    #[test]
    fn a_table_too_large_to_hold_says_so() {
        assert_eq!(table(1000.0, 1000.0, usize::MAX / 2, 32), Err(usize::MAX));
    }
}
