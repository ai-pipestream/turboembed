//! `StrictMath.exp` and `StrictMath.log`: ports of fdlibm 5.3's `e_exp.c`
//! and `e_log.c`.
//!
//! Java specifies `StrictMath` as the results of fdlibm, and JDK 21
//! computes them with a Java translation of it (`java.lang.FdLibm.Exp` and
//! `FdLibm.Log`). OpenNLP's maxent, perceptron and naive Bayes models call
//! `StrictMath.exp` and `StrictMath.log`, so a port that must give the same
//! probabilities bit for bit needs the same functions; Rust's `f64::exp`
//! and `f64::ln` come from the platform's libm and may differ in the last
//! bit.
//!
//! The code follows the C source line by line: the same constants (written
//! from the hexadecimal words the source gives, so no decimal conversion is
//! involved), the same branches and the same order of operations. Rust
//! does not reassociate or contract floating point arithmetic, so each
//! expression rounds as it does in C and Java. Do not simplify the
//! arithmetic: an algebraically equal form rounds differently.
//!
//! The functions read and write the two 32-bit halves of a double, as
//! fdlibm's `__HI` and `__LO` macros do, through [`hi`], [`lo`] and
//! [`with_hi`].

/// The high 32 bits of `x` (sign, exponent and the top 20 bits of the
/// fraction), as fdlibm's `__HI(x)`.
#[inline]
fn hi(x: f64) -> i32 {
    (x.to_bits() >> 32) as i32
}

/// The low 32 bits of `x`, as fdlibm's `__LO(x)`.
#[inline]
fn lo(x: f64) -> i32 {
    x.to_bits() as i32
}

/// `x` with its high 32 bits replaced by `high`, as an assignment to
/// fdlibm's `__HI(x)`.
#[inline]
fn with_hi(x: f64, high: i32) -> f64 {
    f64::from_bits((x.to_bits() & 0xFFFF_FFFF) | ((high as u32 as u64) << 32))
}

/// 1.0e+300, given in decimal in the source.
const HUGE: f64 = 1.0e300;
/// 2^-1000.
const TWOM1000: f64 = f64::from_bits(0x0170_0000_0000_0000);
/// 2^54.
const TWO54: f64 = f64::from_bits(0x4350_0000_0000_0000);
/// ln 2 split in two: the high part has its low 32 bits zero (and its
/// last 21 fraction bits), so `k * LN2_HI` is exact for the `k` that occur.
const LN2_HI: f64 = f64::from_bits(0x3FE6_2E42_FEE0_0000);
const LN2_LO: f64 = f64::from_bits(0x3DEA_39EF_3579_3C76);

/// `e^x`, exactly as `java.lang.StrictMath.exp`.
///
/// The argument is reduced to `x = k ln 2 + r` with `|r| <= 0.5 ln 2`,
/// `e^r` is found from a rational approximation, and `k` is added to the
/// exponent. Special cases are those of fdlibm: NaN gives NaN, +inf gives
/// +inf, -inf gives +0, `x` above 709.78 overflows to +inf and `x` below
/// -745.13 underflows to +0.
pub fn exp(x: f64) -> f64 {
    const HALF: [f64; 2] = [0.5, -0.5];
    const O_THRESHOLD: f64 = f64::from_bits(0x4086_2E42_FEFA_39EF);
    const U_THRESHOLD: f64 = f64::from_bits(0xC087_4910_D52D_3051);
    const LN2HI: [f64; 2] = [LN2_HI, f64::from_bits(0xBFE6_2E42_FEE0_0000)];
    const LN2LO: [f64; 2] = [LN2_LO, f64::from_bits(0xBDEA_39EF_3579_3C76)];
    const INVLN2: f64 = f64::from_bits(0x3FF7_1547_652B_82FE);
    const P1: f64 = f64::from_bits(0x3FC5_5555_5555_553E);
    const P2: f64 = f64::from_bits(0xBF66_C16C_16BE_BD93);
    const P3: f64 = f64::from_bits(0x3F11_566A_AF25_DE2C);
    const P4: f64 = f64::from_bits(0xBEBB_BD41_C5D2_6BF1);
    const P5: f64 = f64::from_bits(0x3E66_3769_72BE_A4D0);

    let mut x = x;
    let mut hi_part = 0.0;
    let mut lo_part = 0.0;
    let mut k: i32 = 0;

    let mut hx = hi(x);
    let xsb = ((hx >> 31) & 1) as usize; // sign bit of x
    hx &= 0x7FFF_FFFF; // high word of |x|

    // Non-finite arguments, overflow and underflow.
    if hx >= 0x4086_2E42 {
        // |x| >= 709.78...
        if hx >= 0x7FF0_0000 {
            if ((hx & 0xF_FFFF) | lo(x)) != 0 {
                return x + x; // NaN
            }
            return if xsb == 0 { x } else { 0.0 }; // exp(+-inf) = {inf, 0}
        }
        if x > O_THRESHOLD {
            return HUGE * HUGE; // overflow
        }
        if x < U_THRESHOLD {
            return TWOM1000 * TWOM1000; // underflow
        }
    }

    // Argument reduction.
    if hx > 0x3FD6_2E42 {
        // |x| > 0.5 ln 2
        if hx < 0x3FF0_A2B2 {
            // and |x| < 1.5 ln 2
            hi_part = x - LN2HI[xsb];
            lo_part = LN2LO[xsb];
            k = 1 - xsb as i32 - xsb as i32;
        } else {
            // A double to int conversion truncates toward zero in C and in
            // Java, as `as` does.
            k = (INVLN2 * x + HALF[xsb]) as i32;
            let t = k as f64;
            hi_part = x - t * LN2HI[0]; // t * ln2_hi is exact here
            lo_part = t * LN2LO[0];
        }
        x = hi_part - lo_part;
    } else if hx < 0x3E30_0000 {
        // |x| < 2^-28
        if HUGE + x > 1.0 {
            return 1.0 + x; // the comparison raises inexact in C
        }
    } else {
        k = 0;
    }

    // x is now in the primary range.
    let t = x * x;
    let c = x - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
    if k == 0 {
        return 1.0 - ((x * c) / (c - 2.0) - x);
    }
    let y = 1.0 - ((lo_part - (x * c) / (2.0 - c)) - hi_part);
    if k >= -1021 {
        with_hi(y, hi(y).wrapping_add(k << 20)) // add k to y's exponent
    } else {
        with_hi(y, hi(y).wrapping_add((k + 1000) << 20)) * TWOM1000
    }
}

/// The natural logarithm of `x`, exactly as `java.lang.StrictMath.log`.
///
/// `x` is written as `2^k (1 + f)` with `sqrt(2)/2 < 1 + f < sqrt(2)`, and
/// `log(1 + f)` is found from a polynomial in `s = f / (2 + f)`. Special
/// cases are those of fdlibm: NaN or a negative `x` (including -inf) gives
/// NaN, +0 and -0 give -inf, +inf gives +inf and 1 gives +0. Subnormal
/// arguments are scaled by 2^54 first.
pub fn log(x: f64) -> f64 {
    const LG1: f64 = f64::from_bits(0x3FE5_5555_5555_5593);
    const LG2: f64 = f64::from_bits(0x3FD9_9999_9997_FA04);
    const LG3: f64 = f64::from_bits(0x3FD2_4924_9422_9359);
    const LG4: f64 = f64::from_bits(0x3FCC_71C5_1D8E_78AF);
    const LG5: f64 = f64::from_bits(0x3FC7_4664_96CB_03DE);
    const LG6: f64 = f64::from_bits(0x3FC3_9A09_D078_C69F);
    const LG7: f64 = f64::from_bits(0x3FC2_F112_DF3E_5244);
    /// The literal 0.33333333333333333 of the source, which rounds to the
    /// double nearest 1/3.
    const THIRD: f64 = f64::from_bits(0x3FD5_5555_5555_5555);
    /// A variable in the C source (`static double zero`), so the
    /// divisions by it below happen at run time there as here.
    const ZERO: f64 = 0.0;

    let mut x = x;
    let mut hx = hi(x);
    let lx = lo(x);

    let mut k: i32 = 0;
    if hx < 0x0010_0000 {
        // x < 2^-1022
        if ((hx & 0x7FFF_FFFF) | lx) == 0 {
            return -TWO54 / ZERO; // log(+-0) = -inf
        }
        if hx < 0 {
            // x - x is +0 for finite x and NaN for -inf, so this is NaN.
            #[allow(clippy::eq_op)]
            return (x - x) / ZERO; // log(-#) = NaN
        }
        // Subnormal: scale x up.
        k -= 54;
        x *= TWO54;
        hx = hi(x);
    }
    if hx >= 0x7FF0_0000 {
        return x + x;
    }
    k += (hx >> 20) - 1023;
    hx &= 0x000F_FFFF;
    let mut i = (hx + 0x9_5F64) & 0x10_0000;
    x = with_hi(x, hx | (i ^ 0x3FF0_0000)); // normalize x or x/2
    k += i >> 20;
    let f = x - 1.0;
    if (0x000F_FFFF & (2 + hx)) < 3 {
        // |f| < 2^-20
        if f == ZERO {
            if k == 0 {
                return ZERO;
            }
            let dk = k as f64;
            return dk * LN2_HI + dk * LN2_LO;
        }
        let r = f * f * (0.5 - THIRD * f);
        if k == 0 {
            return f - r;
        }
        let dk = k as f64;
        return dk * LN2_HI - ((r - dk * LN2_LO) - f);
    }
    let s = f / (2.0 + f);
    let dk = k as f64;
    let z = s * s;
    i = hx - 0x6_147A;
    let w = z * z;
    let j = 0x6_B851 - hx;
    let t1 = w * (LG2 + w * (LG4 + w * LG6));
    let t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7)));
    i |= j;
    let r = t2 + t1;
    if i > 0 {
        let hfsq = 0.5 * f * f;
        if k == 0 { f - (hfsq - s * (hfsq + r)) } else { dk * LN2_HI - ((hfsq - (s * (hfsq + r) + dk * LN2_LO)) - f) }
    } else if k == 0 {
        f - s * (f - r)
    } else {
        dk * LN2_HI - ((s * (f - r) - dk * LN2_LO) - f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::excessive_precision)] // the literals are copied from the C source
    fn hex_constants_are_the_decimal_values_of_the_source() {
        // The C source gives each constant in decimal with its hex words;
        // the decimal literals must parse to the same bits.
        assert_eq!(LN2_HI, 6.93147180369123816490e-01);
        assert_eq!(LN2_LO, 1.90821492927058770002e-10);
        assert_eq!(TWO54, 1.80143985094819840000e+16);
        assert_eq!(TWOM1000, 9.33263618503218878990e-302);
        assert_eq!(f64::from_bits(0x3FD5_5555_5555_5555), 0.33333333333333333);
    }

    #[test]
    fn special_cases() {
        assert!(exp(f64::NAN).is_nan());
        assert_eq!(exp(f64::INFINITY), f64::INFINITY);
        assert_eq!(exp(f64::NEG_INFINITY).to_bits(), 0.0f64.to_bits());
        assert_eq!(exp(0.0), 1.0);
        assert_eq!(exp(-0.0), 1.0);
        assert_eq!(exp(709.8), f64::INFINITY);
        assert_eq!(exp(-745.2).to_bits(), 0.0f64.to_bits());
        // Just above the underflow threshold the result is the smallest
        // subnormal.
        assert_eq!(exp(-745.13), f64::from_bits(1));

        assert!(log(f64::NAN).is_nan());
        assert!(log(-1.0).is_nan());
        assert!(log(f64::NEG_INFINITY).is_nan());
        assert!(log(-f64::from_bits(1)).is_nan());
        assert_eq!(log(0.0), f64::NEG_INFINITY);
        assert_eq!(log(-0.0), f64::NEG_INFINITY);
        assert_eq!(log(f64::INFINITY), f64::INFINITY);
        assert_eq!(log(1.0).to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn values_are_within_an_ulp_of_the_platform() {
        // fdlibm is accurate to under one ulp, so it agrees with any good
        // libm to within one ulp. The bit-exact check is against the JVM's
        // vectors in tests/; this one catches a broken port without them.
        let ulps = |a: f64, b: f64| (a.to_bits() as i64 - b.to_bits() as i64).abs();
        let mut x = -700.0;
        while x < 700.0 {
            assert!(ulps(exp(x), x.exp()) <= 1, "exp({x})");
            x += 0.37;
        }
        let mut x = 1e-300;
        while x < 1e300 {
            assert!(ulps(log(x), x.ln()) <= 1, "log({x})");
            x *= 1.7;
        }
        assert!(ulps(log(f64::from_bits(1)), f64::from_bits(1).ln()) <= 1);
    }
}
