//! A maxent or perceptron model and its evaluation.
//!
//! The parameters are stored by predicate in three columns: the
//! parameters of predicate `p` are `pred_outcome[s..e]` and
//! `pred_param[s..e]` with `s = pred_start[p]` and `e = pred_start[p + 1]`,
//! each an (outcome, weight) pair. This is OpenNLP's `Context` array laid
//! out flat, so evaluating a context is one hash lookup per feature and
//! one add per (outcome, weight) pair, with no allocation.

use std::collections::HashMap;

use crate::hash::BuildFoldHasher;

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
}

impl ModelKind {
    /// The type string OpenNLP writes for this kind.
    pub fn type_string(self) -> &'static str {
        match self {
            ModelKind::Gis => "GIS",
            ModelKind::Qn => "QN",
            ModelKind::Perceptron => "Perceptron",
        }
    }
}

/// A model read from OpenNLP's binary model format (see
/// [`MaxentModel::read`]).
#[derive(Debug, Clone)]
pub struct MaxentModel {
    pub(crate) kind: ModelKind,
    pub(crate) outcomes: Vec<Box<str>>,
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
    pub fn outcome(&self, i: usize) -> &str {
        &self.outcomes[i]
    }

    /// The outcome names in index order.
    pub fn outcomes(&self) -> impl Iterator<Item = &str> {
        self.outcomes.iter().map(|o| &**o)
    }

    /// The index of the first outcome with this name, as
    /// `AbstractModel.getIndex` finds it.
    pub fn outcome_index(&self, name: &str) -> Option<usize> {
        self.outcomes.iter().position(|o| &**o == name)
    }

    /// The number of distinct predicate names.
    pub fn num_predicates(&self) -> usize {
        self.predicates.len()
    }

    /// The total number of (outcome, weight) parameters.
    pub fn num_parameters(&self) -> usize {
        self.pred_param.len()
    }

    /// Whether the model has a predicate with this name, given as Java
    /// modified UTF-8.
    pub fn has_predicate(&self, name: &[u8]) -> bool {
        self.predicates.contains_key(name)
    }

    /// Writes the probability of each outcome given the context's features
    /// into `out`, exactly as the Java model of this kind computes it.
    /// Features are predicate names in Java modified UTF-8 (UTF-8 for text
    /// without NUL or characters outside the Basic Multilingual Plane). A
    /// feature the model does not know contributes nothing; a feature
    /// given twice counts twice.
    ///
    /// # Panics
    ///
    /// If `out.len()` is not [`num_outcomes`](Self::num_outcomes).
    pub fn eval(&self, context: &[&[u8]], out: &mut [f64]) {
        self.eval_features(context.iter().copied(), out);
    }

    /// [`eval`](Self::eval) over features from an iterator, so a caller
    /// holding its features in one buffer need not build a slice of them.
    pub fn eval_features<'c, I>(&self, context: I, out: &mut [f64])
    where
        I: IntoIterator<Item = &'c [u8]>,
    {
        assert_eq!(out.len(), self.outcomes.len(), "the output must hold one probability per outcome");
        out.fill(self.start);
        for feature in context {
            if let Some(&p) = self.predicates.get(feature) {
                let p = p as usize;
                let range = self.pred_start[p] as usize..self.pred_start[p + 1] as usize;
                for (&o, &w) in self.pred_outcome[range.clone()].iter().zip(&self.pred_param[range]) {
                    out[o as usize] += w;
                }
            }
        }
        match self.kind {
            ModelKind::Gis => exp_normalize(out, 1.0),
            ModelKind::Qn => {
                let lse = log_sum_of_exps(out);
                for x in out.iter_mut() {
                    *x = (*x - lse).exp();
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
        }
    }

    /// The index of the most probable outcome: the first index holding the
    /// maximum, as `ArrayMath.argmax` finds it.
    ///
    /// # Panics
    ///
    /// If `probabilities` is empty.
    pub fn best_outcome(&self, probabilities: &[f64]) -> usize {
        argmax(probabilities)
    }
}

/// `x = exp(x / divisor)` for every element, then each divided by their
/// sum, summed in index order as the Java loops do.
fn exp_normalize(out: &mut [f64], divisor: f64) {
    let mut sum = 0.0;
    for x in out.iter_mut() {
        *x = if divisor == 1.0 { x.exp() } else { (*x / divisor).exp() };
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
            sum += (v - max).exp();
        }
    }
    max + sum.ln()
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
mod tests {
    use super::*;

    /// A model with outcomes a, b, c and predicates p (a: 1, b: 2) and
    /// q (c: -0.5).
    fn model(kind: ModelKind) -> MaxentModel {
        let mut predicates = HashMap::default();
        predicates.insert(Box::from(&b"p"[..]), 0);
        predicates.insert(Box::from(&b"q"[..]), 1);
        MaxentModel {
            kind,
            outcomes: vec!["a".into(), "b".into(), "c".into()],
            predicates,
            pred_start: vec![0, 2, 3],
            pred_outcome: vec![0, 1, 2],
            pred_param: vec![1.0, 2.0, -0.5],
            start: if kind == ModelKind::Gis { (1.0f64 / 3.0).ln() } else { 0.0 },
        }
    }

    fn eval(m: &MaxentModel, context: &[&[u8]]) -> Vec<f64> {
        let mut out = vec![f64::NAN; m.num_outcomes()];
        m.eval(context, &mut out);
        out
    }

    #[test]
    fn gis_adds_the_uniform_prior_then_normalizes() {
        let got = eval(&model(ModelKind::Gis), &[b"p", b"q", b"unknown"]);
        let r = (1.0f64 / 3.0).ln();
        let e = [(r + 1.0).exp(), (r + 2.0).exp(), (r - 0.5).exp()];
        let s = e[0] + e[1] + e[2];
        assert_eq!(got, vec![e[0] / s, e[1] / s, e[2] / s]);
    }

    #[test]
    fn qn_subtracts_the_log_sum_of_exps() {
        let got = eval(&model(ModelKind::Qn), &[b"p", b"p"]);
        let x = [2.0f64, 4.0, 0.0];
        let lse = 4.0 + ((2.0f64 - 4.0).exp() + 1.0 + (0.0f64 - 4.0).exp()).ln();
        assert_eq!(got, x.iter().map(|v| (v - lse).exp()).collect::<Vec<_>>());
    }

    #[test]
    fn perceptron_scales_by_the_largest_magnitude_above_one() {
        let got = eval(&model(ModelKind::Perceptron), &[b"p", b"q"]);
        let e = [(1.0f64 / 2.0).exp(), (2.0f64 / 2.0).exp(), (-0.5f64 / 2.0).exp()];
        let s = e[0] + e[1] + e[2];
        assert_eq!(got, vec![e[0] / s, e[1] / s, e[2] / s]);
        // Sums all within [-1, 1] are not scaled.
        let got = eval(&model(ModelKind::Perceptron), &[b"q"]);
        let e = [1.0f64, 1.0, (-0.5f64).exp()];
        let s = e[0] + e[1] + e[2];
        assert_eq!(got, vec![e[0] / s, e[1] / s, e[2] / s]);
    }

    #[test]
    fn an_empty_context_gives_uniform_probabilities() {
        for kind in [ModelKind::Gis, ModelKind::Qn, ModelKind::Perceptron] {
            let got = eval(&model(kind), &[]);
            assert!(got.iter().all(|&p| (p - 1.0 / 3.0).abs() < 1e-15), "{kind:?}: {got:?}");
        }
    }

    #[test]
    fn argmax_keeps_the_first_maximum() {
        let m = model(ModelKind::Gis);
        assert_eq!(m.best_outcome(&[0.2, 0.4, 0.4]), 1);
        assert_eq!(m.best_outcome(&[0.5, 0.1, 0.4]), 0);
        assert_eq!(m.best_outcome(&[f64::NAN, 0.1, 0.4]), 0);
    }

    #[test]
    fn outcome_lookups() {
        let m = model(ModelKind::Qn);
        assert_eq!(m.outcome_index("b"), Some(1));
        assert_eq!(m.outcome_index("z"), None);
        assert_eq!(m.outcome(2), "c");
        assert_eq!(m.num_predicates(), 2);
        assert_eq!(m.num_parameters(), 3);
        assert!(m.has_predicate(b"q"));
    }

    #[test]
    #[should_panic(expected = "one probability per outcome")]
    fn a_wrong_output_length_panics() {
        model(ModelKind::Gis).eval(&[], &mut [0.0; 2]);
    }
}
