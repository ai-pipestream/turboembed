use std::ops::AddAssign;

use turbo_jvm::random::JavaRandom;

use super::{Counts, Measure, PairedUnit};

/// Counts summed over units, in `T`: `u64` where no sum of the set's
/// draws can pass `u64::MAX`, `u128` otherwise.
#[derive(Copy, Clone, Default)]
pub(super) struct Sums<T> {
    true_positives: T,
    selected: T,
    target: T,
}

/// Sums that never overflow: `units.len()` counts of at most `u64::MAX`.
pub(super) type Totals = Sums<u128>;

impl<T: Copy + AddAssign + From<u64> + Into<u128>> Sums<T> {
    pub(super) fn add(&mut self, other: &Counts) {
        self.true_positives += T::from(other.true_positives);
        self.selected += T::from(other.selected);
        self.target += T::from(other.target);
    }

    /// The score of the sums under `measure`, in points. A `u64` sum
    /// converts to the same `f64` as the `u128` of it, so both widths
    /// score a draw to the bit.
    pub(super) fn score(self, measure: Measure) -> f64 {
        let ratio = |n: T, d: T| {
            let (n, d): (u128, u128) = (n.into(), d.into());
            if d > 0 { n as f64 / d as f64 } else { 0.0 }
        };
        let score = match measure {
            Measure::Accuracy => ratio(self.true_positives, self.target),
            Measure::FMeasure => {
                let (p, r) = (ratio(self.true_positives, self.selected), ratio(self.true_positives, self.target));
                if p + r > 0.0 { 2.0 * (p * r) / (p + r) } else { 0.0 }
            }
        };
        100.0 * score
    }
}

pub(super) fn total<'a>(counts: impl Iterator<Item = &'a Counts>) -> Totals {
    let mut sum = Totals::default();
    counts.for_each(|c| sum.add(c));
    sum
}

/// Each model's score and their difference on `resamples` draws of `n`
/// units, summing in `u64` when `n` times the largest count fits in it.
pub(super) fn draws(
    measure: Measure,
    units: &[PairedUnit],
    n: i32,
    resamples: u32,
    random: &mut JavaRandom,
) -> [Vec<f64>; 3] {
    let largest = units
        .iter()
        .flat_map(|u| [u.candidate, u.reference])
        .flat_map(|c| [c.true_positives, c.selected, c.target])
        .max()
        .unwrap_or(0);
    if u128::from(largest) * n as u128 <= u128::from(u64::MAX) {
        draw::<u64>(measure, units, n, resamples, random)
    } else {
        draw::<u128>(measure, units, n, resamples, random)
    }
}

fn draw<T: Copy + Default + AddAssign + From<u64> + Into<u128>>(
    measure: Measure,
    units: &[PairedUnit],
    n: i32,
    resamples: u32,
    random: &mut JavaRandom,
) -> [Vec<f64>; 3] {
    let len = resamples as usize;
    let (mut cand, mut refr, mut diff) = (Vec::with_capacity(len), Vec::with_capacity(len), Vec::with_capacity(len));
    for _ in 0..resamples {
        let (mut c, mut r) = (Sums::<T>::default(), Sums::<T>::default());
        for _ in 0..n {
            let u = &units[random.next_int_bounded(n) as usize];
            c.add(&u.candidate);
            r.add(&u.reference);
        }
        let (c, r) = (c.score(measure), r.score(measure));
        cand.push(c);
        refr.push(r);
        diff.push(c - r);
    }
    [cand, refr, diff]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_gate::SEED;

    #[test]
    fn both_widths_draw_the_same_scores_and_a_wide_set_sums_in_u128() {
        let f = |true_positives, selected, target| Counts { true_positives, selected, target };
        let units: Vec<PairedUnit> =
            (0..40).map(|i| PairedUnit { candidate: f(i % 7, 8, 9), reference: f(i % 5, 7, 9) }).collect();
        for measure in [Measure::FMeasure, Measure::Accuracy] {
            let narrow = draw::<u64>(measure, &units, 40, 50, &mut JavaRandom::new(SEED));
            let wide = draw::<u128>(measure, &units, 40, 50, &mut JavaRandom::new(SEED));
            assert_eq!(narrow, wide);
            assert_eq!(draws(measure, &units, 40, 50, &mut JavaRandom::new(SEED)), narrow);
        }
        // 40 draws of u64::MAX / 39 pass u64::MAX: summed in u128, the
        // half stays a half.
        let big = u64::MAX / 39;
        let units = vec![PairedUnit { candidate: f(big / 2, big, big), reference: f(big, big, big) }; 40];
        let [cand, refr, _] = draws(Measure::Accuracy, &units, 40, 3, &mut JavaRandom::new(SEED));
        assert!(cand.iter().all(|&c| (c - 50.0).abs() < 1e-9), "{cand:?}");
        assert!(refr.iter().all(|&r| r == 100.0), "{refr:?}");
    }
}
