//! The add-on's `LiblinearModel`: the liblinear model with the outcome
//! labels and the predicate map, and `eval`.

use std::collections::HashMap;

use super::error::ModelError;
use super::exp;
use super::linear::LinearModel;
use crate::java_string::JavaString;

/// A model the liblinear add-on trained or read: the liblinear model, the
/// indexer's outcome labels, and its predicate map (`predMap`), predicate
/// to id.
#[derive(Debug, Clone)]
pub struct LiblinearModel {
    pub(crate) linear: LinearModel,
    pub(crate) outcomes: Vec<JavaString>,
    /// The predicate map's entries in the order they were put, which, with
    /// the keys, decides the order a `HashMap` writes them in.
    pub(crate) predicates: Vec<(JavaString, i32)>,
    /// The predicate map by modified UTF-8, for lookups.
    lookup: HashMap<Box<[u8]>, i32>,
}

/// Equal when the liblinear models, the outcome labels and the predicate
/// ids are, whatever order the predicates were put in.
impl PartialEq for LiblinearModel {
    fn eq(&self, other: &Self) -> bool {
        self.linear == other.linear && self.outcomes == other.outcomes && self.lookup == other.lookup
    }
}

impl LiblinearModel {
    /// The model from its parts. `predicates` are the map's entries in the
    /// order they were put, each predicate once.
    pub(crate) fn new(linear: LinearModel, outcomes: Vec<JavaString>, predicates: Vec<(JavaString, i32)>) -> Self {
        debug_assert_eq!(
            predicates.iter().map(|(p, _)| p).collect::<std::collections::HashSet<_>>().len(),
            predicates.len(),
            "each predicate once"
        );
        let mut lookup = HashMap::with_capacity(predicates.len());
        for (p, id) in &predicates {
            lookup.insert(Box::<[u8]>::from(p.as_bytes()), *id);
        }
        LiblinearModel { linear, outcomes, predicates, lookup }
    }

    /// A model from a liblinear model the JVM wrote (read with
    /// [`format::load_model`](super::format::load_model)), the outcome
    /// labels and the predicate map's entries in the order they were put.
    /// Fails when a predicate comes twice, an id is negative, or the model
    /// has more classes than there are outcome labels: [`format::read`]
    /// (super::format::read) refuses the same.
    pub fn from_parts(
        linear: LinearModel,
        outcomes: Vec<JavaString>,
        predicates: Vec<(JavaString, i32)>,
    ) -> Result<Self, ModelError> {
        if linear.nr_class() > outcomes.len() {
            return Err(ModelError::Invalid(format!(
                "{} classes but {} outcome labels, which eval cannot hold",
                linear.nr_class(),
                outcomes.len()
            )));
        }
        let mut seen = std::collections::HashSet::with_capacity(predicates.len());
        for (p, id) in &predicates {
            if !(0..i32::MAX).contains(id) {
                return Err(ModelError::PredicateId { predicate: p.to_string_lossy(), id: *id });
            }
            if !seen.insert(p) {
                return Err(ModelError::DuplicatePredicate(p.to_string_lossy()));
            }
        }
        Ok(LiblinearModel::new(linear, outcomes, predicates))
    }

    /// The liblinear model.
    pub fn linear(&self) -> &LinearModel {
        &self.linear
    }

    /// `getNumOutcomes`: the indexer's outcome labels, which may be more
    /// than the model's classes.
    pub fn num_outcomes(&self) -> usize {
        self.outcomes.len()
    }

    /// `getOutcome(i)`.
    pub fn outcome(&self, i: usize) -> &JavaString {
        &self.outcomes[i]
    }

    pub fn outcomes(&self) -> &[JavaString] {
        &self.outcomes
    }

    /// `getIndex(outcome)`: the first outcome label equal to `outcome`.
    pub fn outcome_index(&self, outcome: &str) -> Option<usize> {
        self.outcomes.iter().position(|o| o.eq_str(outcome))
    }

    /// The predicate map's entries, in the order they were put.
    pub fn predicates(&self) -> &[(JavaString, i32)] {
        &self.predicates
    }

    /// The id the predicate map gives a predicate, by its modified UTF-8.
    pub fn predicate_id(&self, predicate: &[u8]) -> Option<i32> {
        self.lookup.get(predicate).copied()
    }

    /// `eval(context)`: the probabilities `Linear.predictProbability`
    /// writes into an array of [`num_outcomes`](Self::num_outcomes)
    /// zeros, for the features the predicate map knows (each with value
    /// one; a feature given twice counts twice; an unknown one is skipped).
    ///
    /// The probabilities are in liblinear's class order
    /// ([`LinearModel::labels`]), not by outcome id, and the add-on reads
    /// them by outcome id. The two agree when every outcome had an event
    /// left after the cutoff and the index was sorted, which puts the
    /// classes in outcome order; otherwise position `i` holds the
    /// probability of the class with label `labels()[i]`, and the
    /// positions past the classes stay zero. A two class model writes the
    /// first class's sigmoid and one minus it; a model with more classes
    /// writes each class's sigmoid divided by their sum.
    ///
    /// # Panics
    ///
    /// If `out.len()` is not [`num_outcomes`](Self::num_outcomes).
    pub fn eval(&self, context: &[&str], out: &mut [f64]) {
        assert_eq!(out.len(), self.outcomes.len(), "the output must hold one probability per outcome");
        let mut buffer = Vec::new();
        self.eval_ids(
            context.iter().filter_map(|f| {
                if utf8_is_modified_utf8(f) {
                    self.lookup.get(f.as_bytes()).copied()
                } else {
                    buffer.clear();
                    buffer.extend_from_slice(JavaString::new(f).as_bytes());
                    self.lookup.get(buffer.as_slice()).copied()
                }
            }),
            out,
        );
    }

    /// [`eval`](Self::eval) with features given as Java modified UTF-8.
    pub fn eval_modified_utf8(&self, context: &[&[u8]], out: &mut [f64]) {
        assert_eq!(out.len(), self.outcomes.len(), "the output must hold one probability per outcome");
        self.eval_ids(context.iter().filter_map(|f| self.lookup.get(*f).copied()), out);
    }

    /// `predictValues` then `predictProbability` over predicate ids.
    fn eval_ids(&self, ids: impl Iterator<Item = i32>, out: &mut [f64]) {
        let m = &self.linear;
        out.fill(0.0);
        let nr_class = m.nr_class();
        let nr_w = m.nr_w();
        // A predicate id names feature id + 1, which counts while it is at
        // most n: the bias feature's index is n when there is one, and the
        // add-on does not add the bias feature to what it scores.
        let n = m.w_size();
        for id in ids {
            let idx = id as usize + 1;
            if idx <= n {
                let w = &m.w[(idx - 1) * nr_w..idx * nr_w];
                for (o, &wi) in out[..nr_w].iter_mut().zip(w) {
                    *o += wi;
                }
            }
        }
        for o in &mut out[..nr_w] {
            *o = 1.0 / (1.0 + exp(-*o));
        }
        if nr_class == 2 {
            out[1] = 1.0 - out[0];
        } else {
            let mut sum = 0.0;
            for &o in &out[..nr_class] {
                sum += o;
            }
            for o in &mut out[..nr_class] {
                *o /= sum;
            }
        }
    }

    /// `getBestOutcome`: the first position holding the largest
    /// probability.
    pub fn best_outcome(probabilities: &[f64]) -> usize {
        let mut best = 0;
        for i in 1..probabilities.len() {
            if probabilities[i] > probabilities[best] {
                best = i;
            }
        }
        best
    }
}

/// Whether `s` is its own modified UTF-8: no U+0000 (two bytes there) and
/// no character outside the Basic Multilingual Plane (two surrogates
/// there).
#[inline]
fn utf8_is_modified_utf8(s: &str) -> bool {
    !s.bytes().any(|b| b == 0 || b >= 0xF0)
}
