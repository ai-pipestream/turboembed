//! `PlaceProfiles.standardize`: every column centered and divided by its
//! population standard deviation, then every profile divided by its
//! largest absolute value, as OpenNLP computes it with `BigDecimal`.
//!
//! For a column of `n` values `x`, OpenNLP works with the exact
//! `d = n * x - sum(x)` (centering without rounding the mean), takes
//! `sqrt(sum(d^2) / n)` with the quotient and the root rounded to 34
//! digits, and divides each `d` by it, rounded again. A column whose `d`
//! are all zero (a constant column, or a table of one place) becomes zero.
//! Each profile is then divided by its largest absolute value, rounded to
//! 34 digits and converted to the nearest double; a profile of zeros stays
//! zero. The last step leaves each cosine the same up to rounding, and
//! with the largest value of every profile at 1 no double overflows; a
//! value much smaller than the largest of its profile can still round to
//! zero.
//!
//! The doubles are exact binary fractions `m * 2^e`, so a column is held
//! as integers over its smallest `2^e`, and the sums and squares are exact
//! here too. The order of the places does not matter: every sum is exact.

use super::decimal::{Decimal, Integer, Natural};

/// A finite double as `negative`, `m` and `e` with the value `±m * 2^e`;
/// `m` is zero for both zeros.
fn decompose(x: f64) -> (bool, u64, i64) {
    let bits = x.to_bits();
    let negative = bits >> 63 == 1;
    let biased = ((bits >> 52) & 0x7FF) as i64;
    let fraction = bits & ((1 << 52) - 1);
    if biased == 0 { (negative, fraction, -1074) } else { (negative, fraction | (1 << 52), biased - 1075) }
}

/// Standardizes `values`, `width` metrics per place in rows, in place.
/// Every value must be finite.
pub(crate) fn standardize(values: &mut [f64], width: usize) {
    if width == 0 || values.is_empty() {
        return;
    }
    let places = values.len() / width;
    let mut standardized = vec![Decimal::ZERO; values.len()];
    let mut column = Vec::with_capacity(places);
    for m in 0..width {
        column.clear();
        column.extend(values[m..].iter().step_by(width).copied());
        for (row, z) in standardize_column(&column).into_iter().enumerate() {
            standardized[row * width + m] = z;
        }
    }
    let mut buffer = String::new();
    for (profile, out) in standardized.chunks_exact(width).zip(values.chunks_exact_mut(width)) {
        let scale = profile.iter().copied().fold(Decimal::ZERO, |s, z| if z.cmp_abs(s).is_gt() { z.abs() } else { s });
        for (z, v) in profile.iter().zip(out) {
            *v = if scale.is_zero() || z.is_zero() { 0.0 } else { z.div(scale).to_f64(&mut buffer) };
        }
    }
}

/// The standardized values of one column, rounded to 34 digits.
fn standardize_column(column: &[f64]) -> Vec<Decimal> {
    let n = column.len() as u64;
    let parts: Vec<(bool, u64, i64)> = column.iter().map(|&x| decompose(x)).collect();
    let Some(low) = parts.iter().filter(|p| p.1 != 0).map(|p| p.2).min() else {
        return vec![Decimal::ZERO; column.len()];
    };
    // Each value as an integer count of 2^low.
    let scaled: Vec<Integer> = parts
        .iter()
        .map(|&(negative, m, e)| {
            if m == 0 { Integer::default() } else { Integer::new(negative, Natural::from_u64(m).shl((e - low) as u64)) }
        })
        .collect();
    let sum = scaled.iter().fold(Integer::default(), |s, x| s.add(x));
    let count = Natural::from_u64(n);
    let neg_sum = sum.negate();
    // d = n * x - sum, still in units of 2^low.
    let centered: Vec<Integer> =
        scaled.iter().map(|x| Integer::new(x.negative, x.magnitude.mul(&count)).add(&neg_sum)).collect();
    let squares = centered.iter().fold(Natural::default(), |s, d| s.add(&d.magnitude.mul(&d.magnitude)));
    if squares.is_zero() {
        return vec![Decimal::ZERO; column.len()];
    }
    // sum(d^2) * 2^(2 low) / n, with the power of two on the side it fits.
    let (numerator, denominator) =
        if low >= 0 { (squares.shl(2 * low as u64), count) } else { (squares, count.shl(2 * low.unsigned_abs())) };
    let variance = Decimal::quotient(false, &numerator, &denominator, 0);
    let deviation = variance.sqrt();
    let deviation_coefficient = Natural::from_u128(deviation.coefficient);
    let (up, down) = if low >= 0 { (low as u64, 0) } else { (0, low.unsigned_abs()) };
    let denominator = deviation_coefficient.shl(down);
    centered
        .iter()
        .map(|d| Decimal::quotient(d.negative, &d.magnitude.shl(up), &denominator, -deviation.exponent))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(values: &[f64], width: usize) -> Vec<f64> {
        let mut v = values.to_vec();
        standardize(&mut v, width);
        v
    }

    #[test]
    fn doubles_decompose_exactly() {
        for x in [1.0, -2.5, 0.1, f64::MAX, f64::MIN_POSITIVE, 5e-324, -0.0, 0.0] {
            let (negative, m, e) = decompose(x);
            let back = m as f64 * 2f64.powi(e as i32 / 2) * 2f64.powi(e as i32 - e as i32 / 2);
            assert_eq!(if negative { -back } else { back }, x, "{x}");
        }
    }

    #[test]
    fn one_column_becomes_its_signs() {
        // Each profile has one metric, so dividing by the largest absolute
        // value leaves -1, 0 or 1.
        assert_eq!(run(&[1.0, 2.0, 3.0], 1), [-1.0, 0.0, 1.0]);
    }

    #[test]
    fn constant_columns_and_single_places_are_zero() {
        assert_eq!(run(&[4.0, 1.0, 4.0, 2.0], 2), [0.0, -1.0, 0.0, 1.0]);
        assert_eq!(run(&[7.0, -3.0], 2), [0.0, 0.0]);
    }

    #[test]
    fn extreme_columns_do_not_overflow() {
        let v = run(&[f64::MAX, 1.0, -f64::MAX, 2.0, 5e-324, 3.0], 2);
        assert!(v.iter().all(|x| x.is_finite() && x.abs() <= 1.0), "{v:?}");
        // (1.22.., -1.22..), (-1.22.., 0) and (0 underflowed, 1.22..).
        assert_eq!(v, [1.0, -1.0, -1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn two_metrics_scale_by_the_larger() {
        // Two places: every column standardizes to (-1, 1).
        assert_eq!(run(&[0.0, 0.0, 10.0, 1.0], 2), [-1.0, -1.0, 1.0, 1.0]);
        // Three places: the first column (1, 2, 3) is zero in the middle
        // row, the second (0, 0, 3) negative there, so that row scales to
        // (0, -1).
        let v = run(&[1.0, 0.0, 2.0, 0.0, 3.0, 3.0], 2);
        assert_eq!(v[2], 0.0);
        assert_eq!(v[3], -1.0);
    }
}
