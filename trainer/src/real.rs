//! The number type every op is written over: f32 for training, f64 for
//! the gradient checks, which then compare against finite differences
//! with no f32 rounding in the way.

use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

pub trait Real:
    Copy
    + Debug
    + Default
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + Send
    + Sync
    + 'static
{
    const ZERO: Self;
    const ONE: Self;
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn tanh(self) -> Self;
    fn sqrt(self) -> Self;
    fn is_finite(self) -> bool;
    /// The nearest bfloat16 (8 significand bits, f32's exponent), ties to
    /// even, straight from this type's bits so no rounding comes between;
    /// NaN and the infinities stay.
    fn bf16(self) -> Self;

    /// The logistic sigmoid, stable at both tails.
    fn sigmoid(self) -> Self {
        if self >= Self::ZERO {
            Self::ONE / (Self::ONE + (-self).exp())
        } else {
            let e = self.exp();
            e / (Self::ONE + e)
        }
    }
}

macro_rules! real {
    ($t:ty, $bf16:expr) => {
        impl Real for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            fn from_f64(v: f64) -> Self {
                v as $t
            }
            fn to_f64(self) -> f64 {
                self as f64
            }
            fn exp(self) -> Self {
                <$t>::exp(self)
            }
            fn ln(self) -> Self {
                <$t>::ln(self)
            }
            fn tanh(self) -> Self {
                <$t>::tanh(self)
            }
            fn sqrt(self) -> Self {
                <$t>::sqrt(self)
            }
            fn is_finite(self) -> bool {
                <$t>::is_finite(self)
            }
            fn bf16(self) -> Self {
                $bf16(self)
            }
        }
    };
}

/// `v` rounded to bfloat16 by dropping the low 16 bits of its single,
/// ties to even.
fn bf16_of_f32(v: f32) -> f32 {
    let bits = v.to_bits();
    if v.is_nan() {
        return v;
    }
    let lsb = (bits >> 16) & 1;
    f32::from_bits(bits.wrapping_add(0x7FFF + lsb) & 0xFFFF_0000)
}

/// `v` rounded to bfloat16 straight from its double: the low 45 bits
/// of the significand drop, ties to even, so a value a single would have
/// rounded first comes out as bfloat16's nearest; a double past the
/// range of a single becomes the infinity, as the single would.
fn bf16_of_f64(v: f64) -> f64 {
    if v.is_nan() || v == 0.0 {
        return v;
    }
    if v.abs() < f32::MIN_POSITIVE as f64 {
        // bfloat16's subnormals sit on the grid of 2^-133.
        return (v * 2f64.powi(133)).round_ties_even() * 2f64.powi(-133);
    }
    let bits = v.to_bits();
    let lsb = (bits >> 45) & 1;
    let rounded = f64::from_bits(bits.wrapping_add((1 << 44) - 1 + lsb) & !((1 << 45) - 1));
    // The grid point past the largest bfloat16 is 2^128, past the single.
    if rounded.abs() > f32::MAX as f64 { v.signum() * f64::INFINITY } else { rounded }
}

real!(f32, bf16_of_f32);
real!(f64, bf16_of_f64);

/// The dot product of `a` and `b` over their common length, in eight
/// running sums combined at the end, which the compiler keeps in vector
/// registers where one running sum would hold it to a scalar chain.
pub fn dot<R: Real>(a: &[R], b: &[R]) -> R {
    let n = a.len().min(b.len());
    let (ac, at) = a[..n].as_chunks::<8>();
    let (bc, bt) = b[..n].as_chunks::<8>();
    let mut s = [R::ZERO; 8];
    for (x, y) in ac.iter().zip(bc) {
        for k in 0..8 {
            s[k] += x[k] * y[k];
        }
    }
    let mut t = ((s[0] + s[4]) + (s[1] + s[5])) + ((s[2] + s[6]) + (s[3] + s[7]));
    for (x, y) in at.iter().zip(bt) {
        t += *x * *y;
    }
    t
}
