//! Row-wise functions: LayerNorm, e^x and GELU.

/// The sum of eight lanes, in one fixed order.
#[inline(always)]
pub(super) fn lanes8(s: [f64; 8]) -> f64 {
    ((s[0] + s[4]) + (s[1] + s[5])) + ((s[2] + s[6]) + (s[3] + s[7]))
}

/// y = (y - mean) / sqrt(var + eps) * w + b, with the mean and the biased
/// variance summed in F64, in eight lanes: element i into lane i % 8.
#[inline(always)]
pub(in crate::cpu) fn layer_norm(y: &mut [f32], w: &[f32], b: &[f32], eps: f32) {
    let n = y.len() as f64;
    let mut s = [0.0f64; 8];
    for (i, &v) in y.iter().enumerate() {
        s[i % 8] += v as f64;
    }
    let mean = lanes8(s) / n;
    let mut s = [0.0f64; 8];
    for (i, &v) in y.iter().enumerate() {
        let c = v as f64 - mean;
        s[i % 8] += c * c;
    }
    let var = lanes8(s) / n;
    let inv = 1.0 / (var + eps as f64).sqrt();
    for ((y, &w), &b) in y.iter_mut().zip(w).zip(b) {
        *y = ((*y as f64 - mean) * inv) as f32 * w + b;
    }
}

/// e^x in F32 to about an ulp, with no branch, so a loop of it vectorizes:
/// Cephes' expf. Under -80 it is 0. e^-80 is 1.8e-35: its callers only
/// weigh it against terms near 1 (a softmax's largest, 2 - erfc), and a
/// value that reached the subnormal range would send every later multiply
/// of it down the processor's slow path.
#[inline(always)]
pub(super) fn exp(x: f32) -> f32 {
    const LO: f32 = -80.0;
    let c = x.clamp(LO, 88.0);
    let n = (c * std::f32::consts::LOG2_E + 0.5).floor();
    let r = c - n * 0.693_359_4;
    let r = r - n * -2.121_944_4e-4;
    let r2 = r * r;
    let p =
        ((((1.987_569_1e-4 * r + 1.398_2e-3) * r + 8.333_452e-3) * r + 4.166_579_6e-2) * r + 0.166_666_65) * r + 0.5;
    let e = p * r2 + r + 1.0;
    // 2^n from n's bits: adding 1.5 * 2^23 puts the integer n in the low
    // bits of the sum exactly (|n| < 2^22), with no conversion the
    // compiler would take out of the vector registers.
    let n_int = (n + 12_582_912.0).to_bits().wrapping_sub(0x4b40_0000);
    let scale = f32::from_bits(n_int.wrapping_add(127) << 23);
    if x < LO { 0.0 } else { e * scale }
}

/// GELU with the error function, as upstream's "gelu": 0.5 x (1 + erf(x /
/// sqrt 2)) = 0.5 x (2 - erfc(x / sqrt 2)), with erfc(z) = t exp(-z^2 +
/// P(t)) by Numerical Recipes' erfcc Chebyshev fit, whose fractional error
/// is under 1.2e-7 everywhere. In F32, but for its one sensitive part: the
/// exponent's -z^2 = -x^2 / 2, which moves erfc by 2z^2 times its own
/// relative error, is carried exactly as a sum of two floats into the
/// exponential. For x < 0 it is 0.5 x erfc(|z|), with no cancellation.
/// Its relative error, fit and rounding together, is under 5e-7 (tested).
#[inline(always)]
pub(crate) fn gelu(x: f32) -> f32 {
    // Past 16, erfc(|z|) is far under what exp keeps: the clamp keeps x^2
    // finite and changes nothing.
    let xc = x.clamp(-16.0, 16.0);
    let z = xc.abs() * std::f32::consts::FRAC_1_SQRT_2;
    let t = 1.0 / (1.0 + 0.5 * z);
    let p = -1.265_512_2
        + t * (1.000_023_7
            + t * (0.374_091_96
                + t * (0.096_784_18
                    + t * (-0.186_288_06
                        + t * (0.278_868_07
                            + t * (-1.135_204 + t * (1.488_515_9 + t * (-0.822_152_2 + t * 0.170_872_77))))))));
    // -x^2 / 2 = hi + lo exactly, and hi + p = a + e exactly.
    let (sq, sq_lo) = two_product(xc, xc);
    let (hi, lo) = (-0.5 * sq, -0.5 * sq_lo);
    let (a, e) = two_sum(hi, p);
    let r = t * (exp(a) * (1.0 + (e + lo)));
    0.5 * x * if x >= 0.0 { 2.0 - r } else { r }
}

/// a * b = p + e exactly (Dekker, with Veltkamp's split), in plain
/// multiplies and adds: the same bits with or without FMA.
#[inline(always)]
pub(super) fn two_product(a: f32, b: f32) -> (f32, f32) {
    let split = |v: f32| {
        let c = 4097.0 * v;
        let hi = c - (c - v);
        (hi, v - hi)
    };
    let p = a * b;
    let ((ah, al), (bh, bl)) = (split(a), split(b));
    (p, ((ah * bh - p) + ah * bl + al * bh) + al * bl)
}

/// a + b = s + e exactly (Knuth).
#[inline(always)]
pub(super) fn two_sum(a: f32, b: f32) -> (f32, f32) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// erf by its Taylor series in F64, which converges everywhere and is
    /// exact to rounding for |x| under 3.
    fn erf_series(x: f64) -> f64 {
        let z = x * x;
        let (mut term, mut sum, mut n) = (x, x, 0.0);
        loop {
            n += 1.0;
            term *= -z / n;
            let t = term / (2.0 * n + 1.0);
            sum += t;
            if t.abs() < 1e-17 * sum.abs().max(1e-300) {
                break;
            }
        }
        sum * std::f64::consts::FRAC_2_SQRT_PI
    }

    #[test]
    fn gelu_matches_its_definition() {
        for (x, want) in
            [(0.5, 0.520_499_877_813_046_5), (1.0, 0.842_700_792_949_714_9), (2.0, 0.995_322_265_018_952_7)]
        {
            assert!((erf_series(x) - want).abs() < 1e-15, "the series at {x}");
        }
        // Under -5 the series itself loses 1e-9 to cancellation.
        let mut worst = 0.0f64;
        for i in -5000..=6000 {
            let x = (i as f32 / 1000.0) as f64;
            let want = 0.5 * x * (1.0 + erf_series(x / std::f64::consts::SQRT_2));
            let got = gelu(x as f32) as f64;
            assert!((got - want).abs() <= 1e-6 * want.abs().max(1e-3), "gelu({x}) is {got}, not {want}");
            if want.abs() > 1e-3 {
                worst = worst.max((got - want).abs() / want.abs());
            }
        }
        println!("gelu's worst relative error on [-5, 6] where it is over 1e-3: {worst:.2e}");
        assert!(worst < 5e-7, "gelu's relative error reaches {worst:.2e}");
        assert_eq!(gelu(1e30), 1e30);
        assert_eq!(gelu(-1e30), 0.0);
        assert_eq!(gelu(0.0), 0.0);
    }

    #[test]
    fn exp_is_within_two_ulps() {
        let mut worst = 0.0f64;
        // Over the range it is not 0, and under the clamp at 88.
        for i in -79_999..=87_900 {
            let x = i as f32 / 1000.0 - 0.000_37;
            let (got, want) = (exp(x) as f64, (x as f64).exp());
            worst = worst.max(((got - want) / want).abs());
        }
        assert!(worst < 2.0 * f32::EPSILON as f64, "exp's relative error reaches {worst}");
        assert_eq!(exp(-80.01), 0.0);
        assert!(exp(-79.99) >= f32::MIN_POSITIVE);
        assert_eq!(exp(f32::NEG_INFINITY), 0.0);
    }
}
