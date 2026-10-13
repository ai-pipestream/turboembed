//! A correctly rounded natural logarithm, for code that ports OpenNLP's
//! calls to `Math.log`.
//!
//! `Math.log` is allowed to differ from `StrictMath.log` (fdlibm) by one
//! unit in the last place, and what it returns depends on the JVM build,
//! the platform and the flags: HotSpot on x86-64 computes it with its own
//! stub routine, not fdlibm; other ports have other routines, and with
//! the stub turned off (`-XX:-UseLibmIntrinsic`) it is fdlibm again.
//! Parity with the JVM is therefore parity with one build. The build
//! compared is the one the goldens are generated with, Temurin 21.0.12
//! on x86-64 (the recipe container of `analysis/goldens-tool`): its
//! `Math.log` was compared with this function for every integer from 1
//! to 2^24 and for 4 million random integers up to 2^63, and they differ
//! on one argument, 14237211, where the true logarithm lies within 1e-22
//! of the midpoint between two doubles and the JVM returns the upper one,
//! which is not the nearest. Only integers were compared, which are the
//! arguments OpenNLP's unigram segmenter passes (counts and their total)
//! besides 0.5; how often that JVM is not correctly rounded on other
//! doubles was not measured. On the same integers fdlibm differs from the
//! JVM on 112599 of the 2^24, the platform's `f64::ln` (glibc) on 414.
//!
//! The logarithm is computed in double-double arithmetic (about 104 bits)
//! and rounded once. No error bound is proven; it agreed with a 70 digit
//! decimal logarithm on 250,000 positive normal doubles (uniform over the
//! bit patterns, near 1, and integers up to 2^63). It may round the wrong
//! way where the true value lies within about 2^-100 of a midpoint
//! between two doubles. It takes no platform function (no `ln`, no
//! `mul_add`), so the result is the same on every platform. It costs
//! about 0.8 microseconds a call, some 60 times `strict_math::log`, so it
//! suits code that takes a logarithm while loading a model, not per
//! token.

/// A double-double: the unevaluated sum `hi + lo` with `|lo| <= ulp(hi) / 2`.
#[derive(Copy, Clone, Debug)]
struct Dd {
    hi: f64,
    lo: f64,
}

/// `a + b` exactly, as a rounded sum and its error (Knuth).
#[inline]
const fn two_sum(a: f64, b: f64) -> Dd {
    let s = a + b;
    let bb = s - a;
    Dd { hi: s, lo: (a - (s - bb)) + (b - bb) }
}

/// `a + b` exactly, for `|a| >= |b|` (Dekker).
#[inline]
const fn quick_two_sum(a: f64, b: f64) -> Dd {
    let s = a + b;
    Dd { hi: s, lo: b - (s - a) }
}

/// `a` as two halves of 26 bits each, whose sum is `a` (Dekker).
#[inline]
const fn split(a: f64) -> (f64, f64) {
    let t = 134_217_729.0 * a;
    let hi = t - (t - a);
    (hi, a - hi)
}

/// `a * b` exactly, as a rounded product and its error (Dekker).
#[inline]
const fn two_prod(a: f64, b: f64) -> Dd {
    let p = a * b;
    let (ah, al) = split(a);
    let (bh, bl) = split(b);
    Dd { hi: p, lo: ((ah * bh - p) + ah * bl + al * bh) + al * bl }
}

#[inline]
const fn add(a: Dd, b: Dd) -> Dd {
    let s = two_sum(a.hi, b.hi);
    let t = two_sum(a.lo, b.lo);
    let s = quick_two_sum(s.hi, s.lo + t.hi);
    quick_two_sum(s.hi, s.lo + t.lo)
}

#[inline]
fn mul(a: Dd, b: Dd) -> Dd {
    let p = two_prod(a.hi, b.hi);
    quick_two_sum(p.hi, p.lo + (a.hi * b.lo + a.lo * b.hi))
}

#[inline]
const fn mul_f64(a: Dd, b: f64) -> Dd {
    let p = two_prod(a.hi, b);
    quick_two_sum(p.hi, p.lo + a.lo * b)
}

/// `a / b`, by three rounds of long division.
const fn div(a: Dd, b: Dd) -> Dd {
    let q1 = a.hi / b.hi;
    let r = add(a, neg(mul_f64(b, q1)));
    let q2 = r.hi / b.hi;
    let r = add(r, neg(mul_f64(b, q2)));
    let q3 = r.hi / b.hi;
    add(quick_two_sum(q1, q2), Dd { hi: q3, lo: 0.0 })
}

#[inline]
const fn neg(a: Dd) -> Dd {
    Dd { hi: -a.hi, lo: -a.lo }
}

/// `ln 2` to about 107 bits.
const LN2: Dd = Dd { hi: f64::from_bits(0x3FE6_2E42_FEFA_39EF), lo: f64::from_bits(0x3C7A_BC9E_3B39_803F) };

/// The terms of the series past the first. With `|s| <= 0.1716` each term
/// is at most 0.0295 times the one before, and 0.0295^24 < 2^-121.
const TERMS: usize = 24;

/// `1 / (2j + 1)` for `j` from 0 to [`TERMS`], the series' coefficients,
/// worked out once when the crate is compiled rather than on every call.
const RECIPROCALS: [Dd; TERMS + 1] = {
    let mut table = [Dd { hi: 0.0, lo: 0.0 }; TERMS + 1];
    let mut j = 0;
    while j <= TERMS {
        table[j] = div(Dd { hi: 1.0, lo: 0.0 }, Dd { hi: (2 * j + 1) as f64, lo: 0.0 });
        j += 1;
    }
    table
};

/// The natural logarithm of `x`, correctly rounded but for the limit
/// the module documentation states. A positive normal `x` takes the
/// double-double path; every other argument (NaN, a negative number,
/// zero, a subnormal, infinity) gets `StrictMath.log`'s answer, which for
/// all of those but a subnormal is the value the IEEE special cases fix,
/// and for a subnormal may differ from `Math.log` by a unit in the last
/// place.
pub fn log(x: f64) -> f64 {
    if !(f64::MIN_POSITIVE..f64::INFINITY).contains(&x) {
        return crate::strict_math::log(x);
    }
    // x = 2^k * m with m in [sqrt(2)/2, sqrt(2)].
    let bits = x.to_bits();
    let mut k = ((bits >> 52) & 0x7FF) as i32 - 1023;
    let mut m = f64::from_bits((bits & 0x000F_FFFF_FFFF_FFFF) | 0x3FF0_0000_0000_0000);
    if m > std::f64::consts::SQRT_2 {
        m *= 0.5;
        k += 1;
    }
    // ln m = 2 atanh(s) = 2 (s + s^3/3 + s^5/5 + ...), s = (m - 1) / (m + 1).
    // m - 1 is exact (Sterbenz), m + 1 is kept exactly as a double-double.
    let s = div(Dd { hi: m - 1.0, lo: 0.0 }, two_sum(m, 1.0));
    let s2 = mul(s, s);
    let mut sum = RECIPROCALS[TERMS];
    for &r in RECIPROCALS[..TERMS].iter().rev() {
        sum = add(mul(sum, s2), r);
    }
    let ln_m = mul_f64(mul(s, sum), 2.0);
    let r = add(mul_f64(LN2, f64::from(k)), ln_m);
    r.hi + r.lo
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Arguments with the bits of `Math.log` and `StrictMath.log` as
    /// JDK 21.0.12 (Temurin, amd64) returned them.
    const JVM: [(f64, u64, u64); 11] = [
        (3.0, 0x3FF1_93EA_7AAD_030B, 0x3FF1_93EA_7AAD_030A),
        (9170.0, 0x4022_3F54_A1C5_04C1, 0x4022_3F54_A1C5_04C2),
        (136837.0, 0x4027_A731_0000_E6DB, 0x4027_A731_0000_E6DC),
        (141614.0, 0x4027_B8C2_AF40_20B3, 0x4027_B8C2_AF40_20B4),
        (147674.0, 0x4027_CE36_E040_6E43, 0x4027_CE36_E040_6E44),
        (277862.0, 0x4029_11DB_C61C_3609, 0x4029_11DB_C61C_360A),
        (60101967.0, 0x4031_E95B_8BB8_4672, 0x4031_E95B_8BB8_4672),
        (0.5, 0xBFE6_2E42_FEFA_39EF, 0xBFE6_2E42_FEFA_39EF),
        (1.0, 0x0000_0000_0000_0000, 0x0000_0000_0000_0000),
        (2.0, 0x3FE6_2E42_FEFA_39EF, 0x3FE6_2E42_FEFA_39EF),
        // Long.MAX_VALUE as a double, 2^63.
        (9_223_372_036_854_775_808.0, 0x4045_D589_F2FE_5107, 0x4045_D589_F2FE_5107),
    ];

    #[test]
    fn equals_math_log_where_strict_math_does_not() {
        for (x, math, strict) in JVM {
            assert_eq!(log(x).to_bits(), math, "{x}");
            assert_eq!(crate::strict_math::log(x).to_bits(), strict, "{x}");
        }
    }

    #[test]
    fn the_one_integer_up_to_2_pow_24_where_the_jvm_is_not_nearest() {
        // The JVM gives 0x403078ABAD64E47A (16.471369588024096); the true
        // value, 16.47136958802409422..., is nearer 0x403078ABAD64E479.
        assert_eq!(log(14237211.0).to_bits(), 0x4030_78AB_AD64_E479);
    }

    #[test]
    fn special_cases_are_strict_math_s() {
        assert!(log(f64::NAN).is_nan());
        assert!(log(-1.0).is_nan());
        assert_eq!(log(0.0), f64::NEG_INFINITY);
        assert_eq!(log(f64::INFINITY), f64::INFINITY);
    }

    #[test]
    fn within_an_ulp_of_fdlibm() {
        let mut x = 1e-300;
        while x < 1e300 {
            let (a, b) = (log(x), crate::strict_math::log(x));
            assert!((a.to_bits() as i64 - b.to_bits() as i64).abs() <= 1, "{x}: {a} {b}");
            x *= 1.37;
        }
    }
}
