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
            for (&o, &w) in self.pred_outcome[range.clone()].iter().zip(&self.pred_param[range]) {
                out[o as usize] += w;
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
        }
    }
}

/// `s` as Java modified UTF-8, written into `buffer`, or `None` when `s`
/// holds no U+0000 and no four-byte sequence, so its UTF-8 bytes are
/// already its modified UTF-8.
#[inline]
fn modified_utf8<'b>(s: &str, buffer: &'b mut Vec<u8>) -> Option<&'b [u8]> {
    if !s.bytes().any(|b| b == 0 || b >= 0xF0) {
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
mod tests {
    use super::*;
    use turbo_jvm::strict_math::{exp, log};

    /// A model with outcomes a, b, c and predicates p (a: 1, b: 2),
    /// q (c: -0.5), and two whose modified UTF-8 differs from UTF-8:
    /// "n\0" (a: 3) and "e\u{1F600}" (b: 3).
    fn model(kind: ModelKind) -> MaxentModel {
        let mut predicates = HashMap::default();
        predicates.insert(Box::from(&b"p"[..]), 0);
        predicates.insert(Box::from(&b"q"[..]), 1);
        predicates.insert(Box::from(&b"n\xC0\x80"[..]), 2);
        predicates.insert(Box::from(&b"e\xED\xA0\xBD\xED\xB8\x80"[..]), 3);
        MaxentModel {
            kind,
            outcomes: vec!["a".into(), "b".into(), "c".into()],
            predicates,
            pred_start: vec![0, 2, 3, 4, 5],
            pred_outcome: vec![0, 1, 2, 0, 1],
            pred_param: vec![1.0, 2.0, -0.5, 3.0, 3.0],
            start: if kind == ModelKind::Gis { log(1.0 / 3.0) } else { 0.0 },
        }
    }

    fn eval(m: &MaxentModel, context: &[&str]) -> Vec<f64> {
        let mut out = vec![f64::NAN; m.num_outcomes()];
        m.eval(context, &mut out);
        let mut bytes = vec![f64::NAN; m.num_outcomes()];
        let encoded: Vec<Vec<u8>> = context
            .iter()
            .map(|f| {
                let mut v = Vec::new();
                push_str_modified_utf8(&mut v, f);
                v
            })
            .collect();
        let encoded: Vec<&[u8]> = encoded.iter().map(Vec::as_slice).collect();
        m.eval_modified_utf8(&encoded, &mut bytes);
        assert_eq!(out, bytes, "eval and eval_modified_utf8 disagree on {context:?}");
        out
    }

    #[test]
    fn predicates_are_listed_as_modified_utf8() {
        let m = model(ModelKind::Gis);
        let mut names: Vec<&[u8]> = m.predicates().collect();
        names.sort();
        assert_eq!(names, [&b"e\xED\xA0\xBD\xED\xB8\x80"[..], b"n\xC0\x80", b"p", b"q"]);
        assert!(names.iter().all(|n| m.has_predicate_modified_utf8(n)));
    }

    #[test]
    fn gis_adds_the_uniform_prior_then_normalizes() {
        let got = eval(&model(ModelKind::Gis), &["p", "q", "unknown"]);
        let r = log(1.0 / 3.0);
        let e = [exp(r + 1.0), exp(r + 2.0), exp(r - 0.5)];
        let s = e[0] + e[1] + e[2];
        assert_eq!(got, vec![e[0] / s, e[1] / s, e[2] / s]);
    }

    #[test]
    fn qn_subtracts_the_log_sum_of_exps() {
        let got = eval(&model(ModelKind::Qn), &["p", "p"]);
        let x = [2.0f64, 4.0, 0.0];
        let lse = 4.0 + log(exp(2.0 - 4.0) + exp(0.0) + exp(0.0 - 4.0));
        assert_eq!(got, x.iter().map(|v| exp(v - lse)).collect::<Vec<_>>());
    }

    #[test]
    fn perceptron_scales_by_the_largest_magnitude_above_one() {
        let got = eval(&model(ModelKind::Perceptron), &["p", "q"]);
        let e = [exp(1.0 / 2.0), exp(2.0 / 2.0), exp(-0.5 / 2.0)];
        let s = e[0] + e[1] + e[2];
        assert_eq!(got, vec![e[0] / s, e[1] / s, e[2] / s]);
        // Sums all within [-1, 1] are not scaled.
        let got = eval(&model(ModelKind::Perceptron), &["q"]);
        let e = [exp(0.0), exp(0.0), exp(-0.5)];
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
    fn str_features_are_looked_up_as_modified_utf8() {
        let m = model(ModelKind::Qn);
        let x = [3.0f64, 3.0, 0.0];
        let lse = 3.0 + log(exp(0.0) + exp(0.0) + exp(0.0 - 3.0));
        let want: Vec<f64> = x.iter().map(|v| exp(v - lse)).collect();
        assert_eq!(eval(&m, &["n\0", "e\u{1F600}"]), want);

        // The same features given as plain UTF-8 bytes match nothing.
        let mut out = [0.0; 3];
        m.eval_modified_utf8(&[b"n\0", "e\u{1F600}".as_bytes()], &mut out);
        let mut empty = [0.0; 3];
        m.eval_modified_utf8(&[], &mut empty);
        assert_eq!(out, empty);

        assert!(m.has_predicate("n\0"));
        assert!(m.has_predicate("e\u{1F600}"));
        assert!(m.has_predicate_modified_utf8(b"n\xC0\x80"));
        assert!(!m.has_predicate_modified_utf8(b"n\0"));
    }

    #[test]
    fn argmax_keeps_the_first_maximum() {
        assert_eq!(best_outcome(&[0.2, 0.4, 0.4]), 1);
        assert_eq!(best_outcome(&[0.5, 0.1, 0.4]), 0);
        assert_eq!(best_outcome(&[f64::NAN, 0.1, 0.4]), 0);
    }

    #[test]
    fn outcome_lookups() {
        let m = model(ModelKind::Qn);
        assert_eq!(m.outcome_index("b"), Some(1));
        assert_eq!(m.outcome_index("z"), None);
        assert_eq!(m.outcome(2), "c");
        assert_eq!(m.num_predicates(), 4);
        assert_eq!(m.num_parameters(), 5);
        assert!(m.has_predicate("q"));
        assert!(!m.has_predicate("r"));
    }

    #[test]
    #[should_panic(expected = "one probability per outcome")]
    fn a_wrong_output_length_panics() {
        model(ModelKind::Gis).eval(&[], &mut [0.0; 2]);
    }

    #[test]
    #[should_panic(expected = "one probability per outcome")]
    fn a_wrong_output_length_panics_for_bytes() {
        model(ModelKind::Gis).eval_modified_utf8(&[], &mut [0.0; 2]);
    }
}
