//! The error function, and GELU with it, to f64 rounding: the gated MLP's
//! activation and its derivative. A fit good to 1e-7, as an inference
//! kernel may use, would leave a finite-difference check at 1e-6 nothing
//! to agree with, so both are computed from series that converge.

use std::f64::consts::{FRAC_1_SQRT_2, FRAC_2_SQRT_PI, PI};

use crate::real::Real;

/// sqrt(2 pi), the Gaussian density's normaliser.
const SQRT_2PI: f64 = 2.506_628_274_631_000_7;

/// erf(x) within 2e-14: the Taylor series under 2.5, where its terms stay
/// small enough to cancel to that, and one minus the continued fraction of
/// erfc beyond it.
pub fn erf(x: f64) -> f64 {
    let a = x.abs();
    let r = if a < 2.5 {
        let z = a * a;
        let (mut term, mut sum, mut n) = (a, a, 0.0);
        loop {
            n += 1.0;
            term *= -z / n;
            let t = term / (2.0 * n + 1.0);
            sum += t;
            if t.abs() <= 1e-17 * sum.abs() {
                break;
            }
        }
        sum * FRAC_2_SQRT_PI
    } else if a >= 6.5 {
        // erfc(6.5) is 3.8e-20: under half an ulp of 1.
        1.0
    } else {
        1.0 - erfc_tail(a)
    };
    if x < 0.0 { -r } else { r }
}

/// erfc(x) for x of 2.5 or more, by its continued fraction evaluated
/// from the tail: exp(-x^2) / sqrt(pi) / (x + (1/2) / (x + 1 / (x + (3/2)
/// / (x + ...)))), whose k-th partial numerator is k / 2. Sixty terms
/// leave it within rounding at 2.5, and it converges faster beyond.
fn erfc_tail(x: f64) -> f64 {
    let mut f = x;
    for k in (1..=60).rev() {
        f = x + (k as f64 * 0.5) / f;
    }
    (-x * x).exp() / (PI.sqrt() * f)
}

/// GELU by the error function, 0.5 x (1 + erf(x / sqrt 2)).
pub fn gelu<R: Real>(x: R) -> R {
    let v = x.to_f64();
    R::from_f64(0.5 * v * (1.0 + erf(v * FRAC_1_SQRT_2)))
}

/// d GELU / dx: the Gaussian's distribution function plus x times its
/// density.
pub fn gelu_grad<R: Real>(x: R) -> R {
    let v = x.to_f64();
    R::from_f64(0.5 * (1.0 + erf(v * FRAC_1_SQRT_2)) + v * (-0.5 * v * v).exp() / SQRT_2PI)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_matches_tabulated_values() {
        // Abramowitz and Stegun, table 7.1, and erfc at the far points.
        for (x, want) in [
            (0.0, 0.0),
            (0.5, 0.520_499_877_813_046_5),
            (1.0, 0.842_700_792_949_714_9),
            (2.0, 0.995_322_265_018_952_7),
            (2.5, 0.999_593_047_982_555),
            (3.0, 0.999_977_909_503_001_4),
            (4.0, 0.999_999_984_582_742_1),
        ] {
            assert!((erf(x) - want).abs() < 2e-14, "erf({x}) is {}, not {want}", erf(x));
            assert_eq!(erf(-x), -erf(x));
        }
        assert!((erfc_tail(5.0) - 1.537_459_794_428_035e-12).abs() < 1e-26);
        assert_eq!(erf(7.0), 1.0);
    }

    /// The series and the fraction agree where they meet.
    #[test]
    fn erf_is_continuous_at_the_switch() {
        let below = erf(2.5 - 1e-12);
        let above = erf(2.5);
        assert!((above - below).abs() < 1e-11, "{below} vs {above}");
    }

    #[test]
    fn gelu_derivative_matches_a_central_difference() {
        for i in -40..=40 {
            let x = i as f64 * 0.15;
            let h = 1e-6;
            let numeric = (gelu(x + h) - gelu(x - h)) / (2.0 * h);
            let analytic = gelu_grad(x);
            assert!((numeric - analytic).abs() < 1e-8, "gelu'({x}): {numeric} vs {analytic}");
        }
        assert_eq!(gelu(0.0f32), 0.0);
        assert!((gelu(1.0f32) - 0.841_344_7).abs() < 1e-6);
    }
}
