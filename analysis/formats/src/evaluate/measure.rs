//! `FMeasure` and `Mean`: the scores every evaluator keeps.
//!
//! Standard evaluation counts true positives as a multiset intersection
//! ([`count_true_positives`]). OpenNLP's own counting, which can score a
//! recall above 1, is in [`opennlp_count_true_positives`] for golden comparisons only.

use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;

use crate::sample::Name;

/// A span with a type, compared as `opennlp.tools.util.Span.equals`
/// compares: by start, end and type. The offsets are token indices for
/// names and phrases and byte offsets for tokens and sentences.
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

    /// `Span.toString`: `[start..end)`, then a space and the type when
    /// there is one.
    pub fn java_string(&self) -> String {
        match self.kind {
            Some(k) => format!("[{}..{}) {k}", self.start, self.end),
            None => format!("[{}..{})", self.start, self.end),
        }
    }
}

impl<'t> From<&Name<'t>> for LabeledSpan<'t> {
    fn from(name: &Name<'t>) -> LabeledSpan<'t> {
        LabeledSpan { start: name.start, end: name.end, kind: name.kind }
    }
}

impl fmt::Display for LabeledSpan<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.java_string())
    }
}

/// How [`FMeasure`] counts the true positives of a sample.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
enum Scoring {
    /// The size of the multiset intersection of references and
    /// predictions ([`count_true_positives`]): each prediction matches at
    /// most one reference, so precision and recall stay within 0 and 1.
    #[default]
    Standard,
    /// OpenNLP's `FMeasure.countTruePositives`
    /// ([`opennlp_count_true_positives`]), which counts repeated
    /// predictions more than once. It exists only with the `jvm-goldens`
    /// feature, to compare with the scores the JVM wrote into goldens.
    #[cfg(feature = "jvm-goldens")]
    OpenNlp,
}

/// Precision, recall and F-measure over all the samples evaluated, as
/// `opennlp.tools.util.eval.FMeasure` keeps them.
///
/// The counts add up over every sample, so the scores are those of all
/// the items together, not a mean of per-sample scores (the Java
/// documentation says otherwise; the code adds counts). The default
/// counts each prediction against at most one reference, so precision
/// and recall stay within 0 and 1. With the `jvm-goldens` feature,
/// `FMeasure::opennlp` counts them as OpenNLP does, for golden
/// comparisons only.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct FMeasure {
    selected: u64,
    target: u64,
    true_positives: u64,
    scoring: Scoring,
}

impl FMeasure {
    /// Empty counts that count true positives as OpenNLP does:
    /// compatibility scoring, only for comparing with JVM goldens.
    #[cfg(feature = "jvm-goldens")]
    pub fn opennlp() -> FMeasure {
        FMeasure { scoring: Scoring::OpenNlp, ..FMeasure::default() }
    }

    /// `updateScores`: adds the true positives of `predictions` against
    /// `references`, the number of
    /// predictions and the number of references.
    pub fn update<T: Eq + Hash>(&mut self, references: &[T], predictions: &[T]) {
        self.true_positives += match self.scoring {
            Scoring::Standard => count_true_positives(references, predictions),
            #[cfg(feature = "jvm-goldens")]
            Scoring::OpenNlp => opennlp_count_true_positives(references, predictions),
        };
        self.selected += predictions.len() as u64;
        self.target += references.len() as u64;
    }

    /// `mergeInto`: adds the counts of `other` to these.
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
    /// It is at most 1, except with OpenNLP's counting under the
    /// `jvm-goldens` feature, which can make it exceed 1.
    pub fn recall(&self) -> f64 {
        if self.target > 0 { self.true_positives as f64 / self.target as f64 } else { 0.0 }
    }

    /// `2 * (precision * recall) / (precision + recall)`, or -1 when
    /// precision and recall add up to 0 or less, as Java returns.
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

/// OpenNLP-compatible scoring, kept only to compare with the scores the
/// JVM wrote into goldens: `FMeasure.countTruePositives`, which is not a
/// count of matched pairs. Use [`count_true_positives`] for evaluation.
///
/// For each reference it counts every remaining prediction equal to it,
/// then removes from the predictions the first one equal to the last
/// prediction it matched. The last match is kept from one reference to
/// the next, so a reference that matches nothing removes another copy of
/// the previous reference's match. Repeated predictions of one reference
/// therefore all count: references `[a]` and predictions `[a, a]` make
/// two true positives, a recall of 2. It is quadratic, as Java's is.
#[cfg(feature = "jvm-goldens")]
pub fn opennlp_count_true_positives<T: PartialEq>(references: &[T], predictions: &[T]) -> u64 {
    let mut remaining: Vec<&T> = predictions.iter().collect();
    let mut true_positives = 0;
    let mut matched: Option<&T> = None;
    for reference in references {
        for &prediction in &remaining {
            if reference == prediction {
                matched = Some(prediction);
                true_positives += 1;
            }
        }
        if let Some(m) = matched
            && let Some(at) = remaining.iter().position(|&p| p == m)
        {
            remaining.remove(at);
        }
    }
    true_positives
}

/// OpenNLP-compatible `FMeasure.precision`, for golden comparisons only:
/// [`opennlp_count_true_positives`] over the predictions, NaN with no
/// predictions.
#[cfg(feature = "jvm-goldens")]
pub fn opennlp_precision<T: PartialEq>(references: &[T], predictions: &[T]) -> f64 {
    if predictions.is_empty() {
        return f64::NAN;
    }
    opennlp_count_true_positives(references, predictions) as f64 / predictions.len() as f64
}

/// OpenNLP-compatible `FMeasure.recall`, for golden comparisons only:
/// [`opennlp_count_true_positives`] over the references, NaN with no
/// references. It can exceed 1.
#[cfg(feature = "jvm-goldens")]
pub fn opennlp_recall<T: PartialEq>(references: &[T], predictions: &[T]) -> f64 {
    if references.is_empty() {
        return f64::NAN;
    }
    opennlp_count_true_positives(references, predictions) as f64 / references.len() as f64
}

/// The arithmetic mean of values: `opennlp.tools.util.eval.Mean`.
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

    #[cfg(feature = "jvm-goldens")]
    #[test]
    fn opennlp_true_positives_count_repeats_and_carry_the_last_match() {
        assert_eq!(opennlp_count_true_positives(&["a"], &["a", "a"]), 2);
        assert_eq!(opennlp_count_true_positives(&["a", "a"], &["a"]), 1);
        // The first `a` counts all three and removes one; `b` matches
        // nothing and removes another; the last `a` counts the one left.
        assert_eq!(opennlp_count_true_positives(&["a", "b", "a"], &["a", "a", "a"]), 4);
        assert_eq!(opennlp_count_true_positives(&["a", "b", "a"], &["a", "b", "a"]), 4);
        assert_eq!(opennlp_count_true_positives::<&str>(&[], &["a"]), 0);
        assert_eq!((opennlp_precision(&["a"], &["a", "a"]), opennlp_recall(&["a"], &["a", "a"])), (1.0, 2.0));
        let mut f = FMeasure::opennlp();
        f.update(&["a"], &["a", "a"]);
        assert_eq!((f.precision(), f.recall()), (1.0, 2.0));
        assert_eq!(f.f_measure(), 2.0 * 2.0 / 3.0);
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
    fn opennlp_scoring_is_off_by_default_and_only_the_golden_test_turns_it_on() {
        let manifest = include_str!("../../Cargo.toml");
        let section = |name: &str| {
            let start = manifest.find(name).unwrap_or_else(|| panic!("{name} in Cargo.toml"));
            let rest = &manifest[start + name.len()..];
            &rest[..rest.find("\n[").unwrap_or(rest.len())]
        };
        let features = section("[features]");
        assert!(
            !features.lines().any(|l| l.trim_start().starts_with("default")),
            "jvm-goldens is not a default feature"
        );
        let test = section("[[test]]");
        assert!(test.contains("name = \"evaluate\"") && test.contains("required-features = [\"jvm-goldens\"]"));
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
