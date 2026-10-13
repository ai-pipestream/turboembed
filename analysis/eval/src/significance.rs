//! Paired significance tests over per query score differences.
//!
//! Both tests draw from turbo-jvm's [`JavaRandom`] (`java.util.Random`)
//! with a fixed seed, so the same differences give the same result to the
//! bit on every machine, and a JVM program drawing the same values in the
//! same order would agree. Each test makes its own generator from the
//! seed, so one result does not depend on whether the other was computed
//! first.

use serde::{Deserialize, Serialize};
use turbo_jvm::random::JavaRandom;

use crate::Result;
use crate::metrics::mean;

/// The seed both tests use. Any fixed value would do; this one is the
/// issue number of the evaluation harness.
pub const SEED: i64 = 154;

/// The resamples each test draws.
pub const RESAMPLES: u32 = 10_000;

/// A paired bootstrap of the mean difference.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bootstrap {
    /// The observed mean of the differences.
    pub mean: f64,
    /// The 2.5th and 97.5th percentiles of the resampled means: the 95
    /// percent percentile interval.
    pub low: f64,
    pub high: f64,
}

fn check(diffs: &[f64]) -> Result<i32> {
    if diffs.is_empty() {
        return Err("no paired differences".into());
    }
    if let Some(d) = diffs.iter().find(|d| !d.is_finite()) {
        return Err(format!("a paired difference is {d}"));
    }
    i32::try_from(diffs.len()).map_err(|_| format!("{} differences, more than java.util.Random can index", diffs.len()))
}

/// Draws `resamples` samples of the differences with replacement, each
/// the size of the input, with indices from `nextInt(n)` in order, and
/// returns the observed mean with the 95 percent interval of the
/// resampled means.
///
/// Percentiles use linear interpolation between order statistics (type 7
/// of Hyndman and Fan, the default of R's `quantile` and NumPy's
/// `percentile`): with the means sorted ascending as m[0..B), the p
/// percentile is m[h] interpolated at h = (B - 1) p.
pub fn bootstrap(diffs: &[f64], resamples: u32, seed: i64) -> Result<Bootstrap> {
    let n = check(diffs)?;
    if resamples == 0 {
        return Err("no resamples".into());
    }
    let mut random = JavaRandom::new(seed);
    let mut means: Vec<f64> = (0..resamples)
        .map(|_| {
            let sum: f64 = (0..n).map(|_| diffs[random.next_int_bounded(n) as usize]).sum();
            sum / f64::from(n)
        })
        .collect();
    means.sort_by(f64::total_cmp);
    Ok(Bootstrap { mean: mean(diffs.iter().copied()), low: percentile(&means, 0.025), high: percentile(&means, 0.975) })
}

/// Type 7 percentile of ascending `sorted`, which is not empty.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    let h = (sorted.len() - 1) as f64 * p;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
}

/// The two sided p value of a paired sign flip permutation test: under
/// the null hypothesis each difference is as likely to have either sign.
/// Each resample flips the sign of each difference when `nextBoolean()`
/// is true, in order, and counts when the absolute mean is at least the
/// observed one. The estimate is (count + 1) / (resamples + 1), which
/// counts the observed assignment and is never 0.
///
/// "At least" allows a relative slack of 1e-12, so a resample whose mean
/// equals the observed one except for rounding (the same values summed
/// with other signs) counts.
pub fn permutation_p(diffs: &[f64], resamples: u32, seed: i64) -> Result<f64> {
    let n = check(diffs)?;
    if resamples == 0 {
        return Err("no resamples".into());
    }
    let observed = mean(diffs.iter().copied()).abs();
    let at_least = observed - 1e-12 * observed.max(1.0);
    let mut random = JavaRandom::new(seed);
    let mut count: u32 = 0;
    for _ in 0..resamples {
        let sum: f64 = diffs.iter().map(|&d| if random.next_boolean() { -d } else { d }).sum();
        if (sum / f64::from(n)).abs() >= at_least {
            count += 1;
        }
    }
    Ok(f64::from(count + 1) / f64::from(resamples + 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<f64> {
        // Fixed differences with mixed signs, as a set of per query nDCG
        // differences might look.
        (0..50).map(|i| ((i * 37 % 23) as f64 - 9.0) / 100.0).collect()
    }

    #[test]
    fn constant_differences_give_a_point_interval() {
        // Every resample of [0.25; 7] has mean 0.25 (the sum of seven 0.25
        // is exact), so the interval is [0.25, 0.25].
        let b = bootstrap(&[0.25; 7], RESAMPLES, SEED).unwrap();
        assert_eq!(b, Bootstrap { mean: 0.25, low: 0.25, high: 0.25 });
    }

    #[test]
    fn bootstrap_is_reproducible_to_the_bit() {
        let d = sample();
        let a = bootstrap(&d, RESAMPLES, SEED).unwrap();
        let b = bootstrap(&d, RESAMPLES, SEED).unwrap();
        assert_eq!(a.low.to_bits(), b.low.to_bits());
        assert_eq!(a.high.to_bits(), b.high.to_bits());
        assert_eq!(a.mean.to_bits(), b.mean.to_bits());
        let other = bootstrap(&d, RESAMPLES, SEED + 1).unwrap();
        assert_ne!((a.low.to_bits(), a.high.to_bits()), (other.low.to_bits(), other.high.to_bits()));
        assert!(a.low < a.mean && a.mean < a.high, "{a:?}");
    }

    #[test]
    fn bootstrap_draws_java_random_indices() {
        // One resample of two values: its indices are the first two
        // nextInt(2) values of new Random(SEED), so its mean is fixed by
        // the generator alone. With one resample the interval is that
        // single mean at both ends.
        let d = [0.0, 1.0];
        let mut r = JavaRandom::new(SEED);
        let want = (d[r.next_int_bounded(2) as usize] + d[r.next_int_bounded(2) as usize]) / 2.0;
        let b = bootstrap(&d, 1, SEED).unwrap();
        assert_eq!((b.low, b.high), (want, want));
    }

    #[test]
    fn percentile_interpolates() {
        // Sorted [0, 10, 20, 30, 40]: h = 4 p.
        // p = 0.5: h = 2, value 20. p = 0.1: h = 0.4, 0 + 0.4 * 10 = 4.
        let s = [0.0, 10.0, 20.0, 30.0, 40.0];
        assert_eq!(percentile(&s, 0.5), 20.0);
        assert!((percentile(&s, 0.1) - 4.0).abs() < 1e-12);
        assert_eq!(percentile(&s, 1.0), 40.0);
    }

    #[test]
    fn permutation_known_answers() {
        // All zero: every flip gives |mean| 0 >= 0, so p = 10001/10001 = 1.
        assert_eq!(permutation_p(&[0.0; 5], RESAMPLES, SEED).unwrap(), 1.0);
        // One difference: either sign gives |mean| 1, so p = 1.
        assert_eq!(permutation_p(&[1.0], RESAMPLES, SEED).unwrap(), 1.0);
        // Two equal differences: ++ and -- give |mean| 1, +- and -+ give
        // 0, so the exact p is 2/4 = 0.5. The estimate over 10,000 flips
        // has a standard error of sqrt(0.25 / 10000) = 0.005; allow four.
        let p = permutation_p(&[1.0, 1.0], RESAMPLES, SEED).unwrap();
        assert!((p - 0.5).abs() < 0.02, "{p}");
        // Ten equal positive differences: only all + and all - reach
        // |mean| 1, so the exact p is 2/1024 = 0.00195. Expected count
        // about 19.5 with a standard deviation of about 4.4.
        let p = permutation_p(&[0.3; 10], RESAMPLES, SEED).unwrap();
        assert!((0.0005..0.0045).contains(&p), "{p}");
    }

    #[test]
    fn permutation_is_reproducible_to_the_bit() {
        let d = sample();
        let a = permutation_p(&d, RESAMPLES, SEED).unwrap();
        let b = permutation_p(&d, RESAMPLES, SEED).unwrap();
        assert_eq!(a.to_bits(), b.to_bits());
    }

    #[test]
    fn bad_input_is_refused() {
        assert!(bootstrap(&[], RESAMPLES, SEED).is_err());
        assert!(bootstrap(&[f64::NAN], RESAMPLES, SEED).is_err());
        assert!(bootstrap(&[1.0], 0, SEED).is_err());
        assert!(permutation_p(&[], RESAMPLES, SEED).is_err());
        assert!(permutation_p(&[f64::INFINITY], RESAMPLES, SEED).is_err());
    }
}
