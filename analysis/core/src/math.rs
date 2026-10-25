//! Shared sequential `f64` vector arithmetic for similarity
//! scoring across the analysis crates.
//!
//! Each product widens to `f64` first and the sums fold in index
//! order, so results are platform-independent and stable across
//! runs. Callers keep their own cosine contracts (denominator
//! shape and zero-vector sentinel) on top of these primitives;
//! the pinned evaluation kernels in `turbo-eval` keep their own
//! accumulation orders on purpose and do not use this module.

/// The dot product of two vectors of one length, summed in `f64`
/// in index order. Zips to the shorter slice, as the callers'
/// hand-written loops did.
pub fn dot_f64(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum()
}

/// The sum of squares of `a`, summed in `f64` in index order:
/// the squared L2 norm, before any square root.
pub fn sum_sq_f64(a: &[f32]) -> f64 {
    a.iter().map(|x| f64::from(*x) * f64::from(*x)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values_and_zip_truncation() {
        assert_eq!(dot_f64(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
        assert_eq!(dot_f64(&[1.0, 2.0, 3.0], &[4.0, 5.0]), 14.0);
        assert_eq!(dot_f64(&[], &[]), 0.0);
        assert_eq!(sum_sq_f64(&[3.0, 4.0]), 25.0);
        assert_eq!(sum_sq_f64(&[]), 0.0);
    }

    #[test]
    fn accumulation_is_f64_not_f32() {
        // 2^24 + 1 is not representable in f32; an f32 fold would
        // lose the unit terms, the widened fold keeps them.
        let a = [16_777_216.0f32, 1.0, 1.0];
        assert_eq!(dot_f64(&a, &[1.0; 3]), 16_777_218.0f64);
    }
}
