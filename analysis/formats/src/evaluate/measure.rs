//! `FMeasure` and `Mean`: the scores every evaluator keeps.
//!
//! True positives are a multiset intersection ([`count_true_positives`]):
//! each prediction matches at most one reference, so precision and recall
//! stay within 0 and 1.

use std::collections::HashMap;
use std::hash::Hash;

use crate::sample::Name;

/// A typed span. Offsets are token indices for names and phrases and byte
/// offsets for tokens and sentences.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct LabeledSpan<'t> {
    pub start: u32,
    pub end: u32,
    pub kind: Option<&'t str>,
}

impl<'t> LabeledSpan<'t> {
    pub fn new(start: u32, end: u32, kind: Option<&'t str>) -> LabeledSpan<'t> {
        LabeledSpan { start, end, kind }
    }
}

impl<'t> From<&Name<'t>> for LabeledSpan<'t> {
    fn from(name: &Name<'t>) -> LabeledSpan<'t> {
        LabeledSpan { start: name.start, end: name.end, kind: name.kind }
    }
}

/// Precision, recall and F-measure over all the samples evaluated.
///
/// The counts add up over every sample, so the scores are those of all
/// the items together, not a mean of per-sample scores. Each prediction
/// counts against at most one reference ([`count_true_positives`]), so
/// precision and recall stay within 0 and 1.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct FMeasure {
    selected: u64,
    target: u64,
    true_positives: u64,
}

impl FMeasure {
    /// Adds the true positives of `predictions` against `references`
    /// ([`count_true_positives`]), the number of predictions and the
    /// number of references.
    pub fn update<T: Eq + Hash>(&mut self, references: &[T], predictions: &[T]) {
        self.true_positives += count_true_positives(references, predictions);
        self.selected += predictions.len() as u64;
        self.target += references.len() as u64;
    }

    /// Adds the counts of `other` to these.
    pub fn merge(&mut self, other: &FMeasure) {
        self.selected += other.selected;
        self.target += other.target;
        self.true_positives += other.true_positives;
    }

    /// The true positives over the predictions, or 0 with no predictions.
    pub fn precision(&self) -> f64 {
        if self.selected > 0 { self.true_positives as f64 / self.selected as f64 } else { 0.0 }
    }

    /// The true positives over the references, or 0 with no references.
    /// It is at most 1.
    pub fn recall(&self) -> f64 {
        if self.target > 0 { self.true_positives as f64 / self.target as f64 } else { 0.0 }
    }

    /// `2 * (precision * recall) / (precision + recall)`, or -1 when
    /// precision and recall add up to 0 or less.
    pub fn f_measure(&self) -> f64 {
        let (p, r) = (self.precision(), self.recall());
        if p + r > 0.0 { 2.0 * (p * r) / (p + r) } else { -1.0 }
    }

    /// The number of predictions.
    pub fn selected(&self) -> u64 {
        self.selected
    }

    /// The number of references.
    pub fn target(&self) -> u64 {
        self.target
    }

    pub fn true_positives(&self) -> u64 {
        self.true_positives
    }
}

/// The true positives of `predictions` against `references`: the size of
/// their multiset intersection. Each value counts as many times as it
/// occurs in the input where it occurs fewer times, so references `[a]`
/// and predictions `[a, a]` make one true positive. Order does not
/// matter. It runs in linear time over a count of the references.
pub fn count_true_positives<T: Eq + Hash>(references: &[T], predictions: &[T]) -> u64 {
    let mut unmatched: HashMap<&T, u64> = HashMap::with_capacity(references.len());
    for reference in references {
        *unmatched.entry(reference).or_default() += 1;
    }
    let mut true_positives = 0;
    for prediction in predictions {
        if let Some(left) = unmatched.get_mut(prediction)
            && *left > 0
        {
            *left -= 1;
            true_positives += 1;
        }
    }
    true_positives
}

/// The true positives over the predictions ([`count_true_positives`]), NaN
/// with no predictions.
pub fn precision<T: Eq + Hash>(references: &[T], predictions: &[T]) -> f64 {
    if predictions.is_empty() {
        return f64::NAN;
    }
    count_true_positives(references, predictions) as f64 / predictions.len() as f64
}

/// The true positives over the references ([`count_true_positives`]), NaN
/// with no references.
pub fn recall<T: Eq + Hash>(references: &[T], predictions: &[T]) -> f64 {
    if references.is_empty() {
        return f64::NAN;
    }
    count_true_positives(references, predictions) as f64 / references.len() as f64
}

/// The arithmetic mean of values.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Mean {
    sum: f64,
    count: u64,
}

impl Mean {
    /// Adds `value` once.
    pub fn add(&mut self, value: f64) {
        self.add_times(value, 1);
    }

    /// Adds `value` `count` times: the sum grows by `value * count`.
    pub fn add_times(&mut self, value: f64, count: u64) {
        self.sum += value * count as f64;
        self.count += count;
    }

    /// The sum over the count, or 0 when nothing was added.
    pub fn mean(&self) -> f64 {
        if self.count > 0 { self.sum / self.count as f64 } else { 0.0 }
    }

    /// The number of values added.
    pub fn count(&self) -> u64 {
        self.count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_repeated_prediction_matches_one_reference() {
        assert_eq!(count_true_positives(&["a"], &["a", "a"]), 1);
        let mut f = FMeasure::default();
        f.update(&["a"], &["a", "a"]);
        assert_eq!((f.true_positives(), f.precision(), f.recall()), (1, 0.5, 1.0));
        assert!(f.f_measure() <= 1.0);
        assert_eq!(f.f_measure(), 2.0 * 0.5 / 1.5);
        assert_eq!((precision(&["a"], &["a", "a"]), recall(&["a"], &["a", "a"])), (0.5, 1.0));
    }

    #[test]
    fn true_positives_are_a_multiset_intersection() {
        assert_eq!(count_true_positives(&["a", "a"], &["a"]), 1);
        assert_eq!(count_true_positives(&["a", "a"], &["a", "a"]), 2);
        assert_eq!(count_true_positives(&["a", "a", "b"], &["a", "a", "a", "b", "b"]), 3);
        assert_eq!(count_true_positives(&["a", "b", "a"], &["a", "a", "a"]), 2);
        assert_eq!(count_true_positives(&["a", "b", "a"], &["a", "b", "a"]), 3);
        assert_eq!(count_true_positives(&["b", "a"], &["a", "b"]), 2);
        assert_eq!(count_true_positives::<&str>(&[], &["a"]), 0);
        assert_eq!(count_true_positives::<&str>(&["a"], &[]), 0);
        assert_eq!(count_true_positives::<&str>(&[], &[]), 0);
        let mut f = FMeasure::default();
        f.update(&["a", "a", "b"], &["a", "a", "a", "b", "b"]);
        assert_eq!((f.precision(), f.recall()), (0.6, 1.0));
        assert!(f.f_measure() <= 1.0);
    }

    #[test]
    fn empty_inputs() {
        let mut f = FMeasure::default();
        f.update::<&str>(&[], &[]);
        assert_eq!((f.selected(), f.target(), f.true_positives()), (0, 0, 0));
        assert_eq!((f.precision(), f.recall(), f.f_measure()), (0.0, 0.0, -1.0));
        assert!(precision::<&str>(&["a"], &[]).is_nan());
        assert!(recall::<&str>(&[], &["a"]).is_nan());
        assert!(precision::<&str>(&[], &[]).is_nan());
    }

    #[test]
    fn scores_and_their_zero_cases() {
        let mut f = FMeasure::default();
        assert_eq!((f.precision(), f.recall(), f.f_measure()), (0.0, 0.0, -1.0));
        f.update(&["a"], &["a", "b"]);
        assert_eq!((f.precision(), f.recall()), (0.5, 1.0));
        let mut g = FMeasure::default();
        g.update(&["x", "y"], &["z"]);
        f.merge(&g);
        assert_eq!((f.selected(), f.target(), f.true_positives()), (3, 3, 1));
        assert_eq!(precision(&["a"], &["a", "b"]), 0.5);
    }

    #[test]
    fn mean() {
        let mut m = Mean::default();
        assert_eq!(m.mean(), 0.0);
        m.add(1.0);
        m.add(0.0);
        m.add_times(1.0, 2);
        assert_eq!((m.mean(), m.count()), (0.75, 4));
    }
}
