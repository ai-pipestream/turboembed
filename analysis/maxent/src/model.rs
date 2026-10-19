//! A maxent or perceptron model and its evaluation.
//!
//! The parameters are stored by predicate in three columns: the
//! parameters of predicate `p` are `pred_outcome[s..e]` and
//! `pred_param[s..e]` with `s = pred_start[p]` and `e = pred_start[p + 1]`,
//! each an (outcome, weight) pair. This is OpenNLP's `Context` array laid
//! out flat, so evaluating a context is one hash lookup per feature and
//! one add per (outcome, weight) pair, with no allocation.

use std::collections::HashMap;

use turbo_jvm::strict_math;
use turbo_unicode::modified_utf8::push_str_modified_utf8;

use crate::hash::BuildFoldHasher;
use crate::java_string::{JavaString, utf8_is_modified_utf8};

/// `NaiveBayesModel.getSmoothedProbability`: Lidstone smoothing delta.
const LIDSTONE_DELTA: f64 = 0.05;
/// `java.lang.Double.MIN_VALUE` (`2^-1074`), the smallest positive subnormal
/// `double`.
const JAVA_DOUBLE_MIN_VALUE: f64 = f64::from_bits(1);

/// Which trainer made a model, which decides how its sums become
/// probabilities. The names are the type strings at the start of the
/// model data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelKind {
    /// `GIS`: generalized iterative scaling maxent. Sums start from the
    /// uniform log prior `ln(1 / outcomes)`, then `exp` and normalize.
    Gis,
    /// `QN`: L-BFGS maxent. Sums start from zero, then
    /// `exp(x - logsumexp)`.
    Qn,
    /// `Perceptron`: sums start from zero, are divided by the largest
    /// magnitude when it is above one, then `exp` and normalize.
    Perceptron,
    /// `NaiveBayes`: multinomial naive Bayes with Lidstone smoothing
    /// (`delta = 0.05`) and empirical outcome priors in `LogProbabilities`.
    NaiveBayes,
}

impl ModelKind {
    /// The type string OpenNLP writes for this kind.
    pub fn type_string(self) -> &'static str {
        match self {
            ModelKind::Gis => "GIS",
            ModelKind::Qn => "QN",
            ModelKind::Perceptron => "Perceptron",
            ModelKind::NaiveBayes => "NaiveBayes",
        }
    }
}

/// A model read from OpenNLP's binary model format (see
/// [`MaxentModel::read`]).
#[derive(Debug, Clone)]
pub struct MaxentModel {
    pub(crate) kind: ModelKind,
    /// The outcome labels in index order, as the JVM holds them: a label
    /// can hold half of a surrogate pair.
    pub(crate) outcomes: Vec<JavaString>,
    /// Each outcome label as text, `None` where it holds half of a
    /// surrogate pair, so that a component whose labels are text reads
    /// them without decoding.
    pub(crate) outcome_texts: Vec<Option<Box<str>>>,
    /// Predicate name, as Java modified UTF-8, to predicate index. A name
    /// listed twice maps to its last listing, as OpenNLP's `LinkedHashMap`
    /// does.
    pub(crate) predicates: HashMap<Box<[u8]>, u32, BuildFoldHasher>,
    /// One more entry than there are listed predicates.
    pub(crate) pred_start: Vec<u32>,
    pub(crate) pred_outcome: Vec<u32>,
    pub(crate) pred_param: Vec<f64>,
    /// What every sum starts from: the uniform log prior for GIS, zero
    /// otherwise.
    pub(crate) start: f64,
    /// For `NaiveBayes`: the sum of parameters per outcome across all
    /// predicates (`NaiveBayesModel.initOutcomeTotals`). Empty for other
    /// model kinds.
    pub(crate) outcome_totals: Vec<f64>,
    /// For `NaiveBayes`: the number of listed predicates (`predLabels.length`),
    /// used in Lidstone smoothing. Zero for other model kinds.
    pub(crate) vocabulary: f64,
}

impl MaxentModel {
    /// The trainer that made the model.
    pub fn kind(&self) -> ModelKind {
        self.kind
    }

    /// The number of outcomes; `eval` writes this many probabilities.
    pub fn num_outcomes(&self) -> usize {
        self.outcomes.len()
    }

    /// The name of outcome `i`.
    pub fn outcome(&self, i: usize) -> &JavaString {
        &self.outcomes[i]
    }

    /// The outcome names in index order.
    pub fn outcomes(&self) -> impl ExactSizeIterator<Item = &JavaString> + Clone {
        self.outcomes.iter()
    }

    /// The name of outcome `i` as text, or `None` when it holds half of a
    /// surrogate pair.
    pub fn outcome_str(&self, i: usize) -> Option<&str> {
        self.outcome_texts[i].as_deref()
    }

    /// The outcome names as text, in index order, or the first outcome
    /// that is not text (one holding half of a surrogate pair), for a
    /// component whose outcomes are labels it reads as text.
    pub fn outcome_texts(&self) -> Result<Vec<&str>, &JavaString> {
        self.outcomes.iter().zip(&self.outcome_texts).map(|(o, t)| t.as_deref().ok_or(o)).collect()
    }

    /// The index of the first outcome with this name, as
    /// `AbstractModel.getIndex` finds it. The name can be text or a
    /// [`JavaString`].
    pub fn outcome_index<N: ?Sized>(&self, name: &N) -> Option<usize>
    where
        JavaString: PartialEq<N>,
    {
        self.outcomes.iter().position(|o| o == name)
    }

    /// The number of distinct predicate names.
    pub fn num_predicates(&self) -> usize {
        self.predicates.len()
    }

    /// The total number of (outcome, weight) parameters.
    pub fn num_parameters(&self) -> usize {
        self.pred_param.len()
    }

    /// The distinct predicate names, as Java modified UTF-8, in no
    /// particular order.
    pub fn predicates(&self) -> impl Iterator<Item = &[u8]> {
        self.predicates.keys().map(|k| &**k)
    }

    /// Whether the model has a predicate with this name.
    pub fn has_predicate(&self, name: &str) -> bool {
        match modified_utf8(name, &mut Vec::new()) {
            Some(encoded) => self.predicates.contains_key(encoded),
            None => self.predicates.contains_key(name.as_bytes()),
        }
    }

    /// Whether the model has a predicate with this name, given as Java
    /// modified UTF-8.
    pub fn has_predicate_modified_utf8(&self, name: &[u8]) -> bool {
        self.predicates.contains_key(name)
    }

    /// The index of the predicate with this name, given as Java modified
    /// UTF-8, for [`pattern`](Self::pattern) and
    /// [`parameters`](Self::parameters).
    pub fn predicate_index(&self, name: &[u8]) -> Option<usize> {
        self.predicates.get(name).map(|&i| i as usize)
    }

    /// The outcomes predicate `p` has a parameter for, by outcome index,
    /// in the model's order (`Context.getOutcomes`).
    pub fn pattern(&self, p: usize) -> &[u32] {
        &self.pred_outcome[self.pred_start[p] as usize..self.pred_start[p + 1] as usize]
    }

    /// The parameters of predicate `p`, one per entry of its
    /// [`pattern`](Self::pattern) (`Context.getParameters`).
    pub fn parameters(&self, p: usize) -> &[f64] {
        &self.pred_param[self.pred_start[p] as usize..self.pred_start[p + 1] as usize]
    }

    /// Writes the probability of each outcome given the context's features
    /// into `out`, computed as the Java model of this kind computes it,
    /// with the same operations in the same order and `StrictMath`'s `exp`
    /// and `log`. A feature the model does not know contributes nothing; a
    /// feature given twice counts twice.
    ///
    /// Predicates are keyed by Java modified UTF-8. A feature holding no
    /// U+0000 and no character outside the Basic Multilingual Plane is the
    /// same bytes in UTF-8 and is looked up as it is; any other is encoded
    /// into a buffer first. Callers that already build their features as
    /// modified UTF-8 use [`eval_modified_utf8`](Self::eval_modified_utf8).
    ///
    /// # Panics
    ///
    /// If `out.len()` is not [`num_outcomes`](Self::num_outcomes).
    pub fn eval(&self, context: &[&str], out: &mut [f64]) {
        self.start_sums(out);
        let mut buffer = Vec::new();
        for feature in context {
            match modified_utf8(feature, &mut buffer) {
                Some(encoded) => self.add_feature(encoded, out),
                None => self.add_feature(feature.as_bytes(), out),
            }
        }
        self.finish(out);
    }

    /// [`eval`](Self::eval) with features given as Java modified UTF-8
    /// bytes, looked up without copying. Plain UTF-8 holding U+0000 or a
    /// character outside the Basic Multilingual Plane will not match the
    /// predicate the JVM would have found.
    ///
    /// # Panics
    ///
    /// If `out.len()` is not [`num_outcomes`](Self::num_outcomes).
    pub fn eval_modified_utf8(&self, context: &[&[u8]], out: &mut [f64]) {
        self.eval_features(context.iter().copied(), out);
    }

    /// [`eval_modified_utf8`](Self::eval_modified_utf8) over features from
    /// an iterator, so a caller holding its features in one buffer need not
    /// build a slice of them.
    pub fn eval_features<'c, I>(&self, context: I, out: &mut [f64])
    where
        I: IntoIterator<Item = &'c [u8]>,
    {
        self.start_sums(out);
        for feature in context {
            self.add_feature(feature, out);
        }
        self.finish(out);
    }

    /// Sets every sum to its starting value.
    #[inline]
    fn start_sums(&self, out: &mut [f64]) {
        assert_eq!(out.len(), self.outcomes.len(), "the output must hold one probability per outcome");
        out.fill(self.start);
    }

    /// Adds the parameters of one feature's predicate, if the model has it.
    #[inline]
    fn add_feature(&self, feature: &[u8], out: &mut [f64]) {
        if let Some(&p) = self.predicates.get(feature) {
            let p = p as usize;
            let range = self.pred_start[p] as usize..self.pred_start[p + 1] as usize;
            if self.kind == ModelKind::NaiveBayes {
                let active_outcomes = &self.pred_outcome[range.clone()];
                let active_parameters = &self.pred_param[range];
                let mut ai = 0;
                for (i, (&denominator, sum)) in self.outcome_totals.iter().zip(out.iter_mut()).enumerate() {
                    if ai >= active_outcomes.len() {
                        break;
                    }
                    let numerator = if active_outcomes[ai] as usize == i {
                        let v = active_parameters[ai];
                        ai += 1;
                        v
                    } else {
                        0.0
                    };
                    let prob = (numerator + LIDSTONE_DELTA) / (denominator + LIDSTONE_DELTA * self.vocabulary);
                    *sum += strict_math::log(prob);
                }
            } else {
                for (&o, &w) in self.pred_outcome[range.clone()].iter().zip(&self.pred_param[range]) {
                    out[o as usize] += w;
                }
            }
        }
    }

    /// Turns the sums into probabilities as this kind of model does.
    fn finish(&self, out: &mut [f64]) {
        match self.kind {
            ModelKind::Gis => exp_normalize(out, 1.0),
            ModelKind::Qn => {
                let lse = log_sum_of_exps(out);
                for x in out.iter_mut() {
                    *x = strict_math::exp(*x - lse);
                }
            }
            ModelKind::Perceptron => {
                let mut max = 1.0f64;
                for &x in out.iter() {
                    if max < x.abs() {
                        max = x.abs();
                    }
                }
                exp_normalize(out, max);
            }
            ModelKind::NaiveBayes => {
                let mut total = 0.0;
                for &outcome_total in &self.outcome_totals {
                    total += outcome_total;
                }
                for (x, &numerator) in out.iter_mut().zip(&self.outcome_totals) {
                    *x += strict_math::log(numerator / total);
                }
                let mut highest_log_probability = f64::NEG_INFINITY;
                for &p in out.iter() {
                    if p > highest_log_probability {
                        highest_log_probability = p;
                    }
                }
                let mut sum = 0.0;
                for p in out.iter_mut() {
                    let temp_p = strict_math::exp(*p - highest_log_probability);
                    if !temp_p.is_nan() {
                        sum += temp_p;
                        *p = temp_p;
                    } else {
                        *p = f64::NAN;
                    }
                }
                for p in out.iter_mut() {
                    if p.is_nan() {
                        *p = 0.0;
                    } else if sum > JAVA_DOUBLE_MIN_VALUE {
                        *p /= sum;
                    }
                }
            }
        }
    }
}

/// Each label as text, `None` where it holds half of a surrogate pair.
pub(crate) fn outcome_texts(outcomes: &[JavaString]) -> Vec<Option<Box<str>>> {
    outcomes.iter().map(|o| o.to_str().map(Box::from)).collect()
}

/// `s` as Java modified UTF-8, written into `buffer`, or `None` when `s`
/// holds no U+0000 and no four-byte sequence, so its UTF-8 bytes are
/// already its modified UTF-8.
#[inline]
fn modified_utf8<'b>(s: &str, buffer: &'b mut Vec<u8>) -> Option<&'b [u8]> {
    if utf8_is_modified_utf8(s) {
        return None;
    }
    buffer.clear();
    push_str_modified_utf8(buffer, s);
    Some(buffer)
}

/// The index of the most probable outcome in what
/// [`MaxentModel::eval`] wrote: the first index holding the maximum, as
/// `ArrayMath.argmax` finds it.
///
/// # Panics
///
/// If `probabilities` is empty.
pub fn best_outcome(probabilities: &[f64]) -> usize {
    argmax(probabilities)
}

/// `x = exp(x / divisor)` for every element, then each divided by their
/// sum, summed in index order as the Java loops do.
fn exp_normalize(out: &mut [f64], divisor: f64) {
    let mut sum = 0.0;
    for x in out.iter_mut() {
        *x = if divisor == 1.0 { strict_math::exp(*x) } else { strict_math::exp(*x / divisor) };
        sum += *x;
    }
    for x in out.iter_mut() {
        *x /= sum;
    }
}

/// `ArrayMath.logSumOfExps`: the maximum plus the log of the sum of
/// `exp(x - max)` over every element that is not negative infinity.
fn log_sum_of_exps(x: &[f64]) -> f64 {
    let max = x[argmax(x)];
    let mut sum = 0.0;
    for &v in x {
        if v != f64::NEG_INFINITY {
            sum += strict_math::exp(v - max);
        }
    }
    max + strict_math::log(sum)
}

/// `ArrayMath.argmax`: an element replaces the best so far only when the
/// best is less than it, so ties and NaNs keep the earlier index.
fn argmax(x: &[f64]) -> usize {
    assert!(!x.is_empty(), "argmax of an empty array");
    let mut best = 0;
    for i in 1..x.len() {
        if x[best] < x[i] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests;
