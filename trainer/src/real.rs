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
    ($t:ty) => {
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
        }
    };
}

real!(f32);
real!(f64);

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
