//! `opennlp.tools.ml.naivebayes`: `NaiveBayesTrainer` behind
//! `Algorithm=NAIVEBAYES` and `NaiveBayesModel`, computed as the JVM
//! computes them so that every parameter and probability agrees bit for
//! bit.
//!
//! What the port keeps from the Java code:
//!
//! - `NaiveBayesTrainer.isSortAndMerge()` is `false`, so `train_events`
//!   sets `Sort=false` before running the data indexer.
//! - Every predicate gets the full outcome pattern `0..num_outcomes` and
//!   starts with `0.0` for every outcome.
//! - For each indexed event `ei`, repeated `numTimesEventsSeen[ei]` times,
//!   each context predicate `pi` adds `1.0` (when `values == null`) or
//!   `1.0 * (double) values[ei][ci]` to `params[pi][targetOutcome]`.
//! - `NaiveBayesModel` computes `outcomeTotals[o]` by summing the
//!   parameters for outcome `o` across all predicates in predicate order,
//!   and evaluates a context with Lidstone smoothing (`delta = 0.05` and
//!   `vocabulary = predLabels.length`) and empirical outcome priors in
//!   `LogProbabilities` (`StrictMath.log` and `StrictMath.exp`).

use std::collections::HashMap;

use turbo_jvm::strict_math;

use crate::error::{Error, ONE_OUTCOME, Result};
use crate::event::{Event, EventChecksum};
use crate::gis::TrainingReport;
use crate::index::{DataIndex, IndexParams};
use crate::java_string::JavaString;
use crate::params::{DEFAULT_CUTOFF, Key, TrainerType, TrainingParameters};

/// `NaiveBayesModel.getSmoothedProbability`: Lidstone smoothing delta.
const LIDSTONE_DELTA: f64 = 0.05;
/// `java.lang.Double.MIN_VALUE` (`2^-1074`), the smallest positive subnormal
/// `double`.
const JAVA_DOUBLE_MIN_VALUE: f64 = f64::from_bits(1);

/// A trained naive Bayes model: `NaiveBayesModel`'s data structures. Every
/// predicate has a parameter for every outcome, in outcome order.
#[derive(Debug, Clone, PartialEq)]
pub struct NaiveBayesModel {
    outcome_labels: Vec<JavaString>,
    pred_labels: Vec<JavaString>,
    /// `pmap`: predicate label to index.
    pmap: HashMap<JavaString, u32>,
    /// `0..num_outcomes`, the pattern of every predicate.
    pattern: Vec<u32>,
    /// Predicate `p`'s parameters are `parameters[p * n..(p + 1) * n]` for
    /// `n` outcomes.
    parameters: Vec<f64>,
    /// `NaiveBayesModel.outcomeTotals`: sum of parameters per outcome across
    /// all predicates.
    outcome_totals: Vec<f64>,
}

impl NaiveBayesModel {
    pub fn outcome_labels(&self) -> &[JavaString] {
        &self.outcome_labels
    }

    /// The predicates in the index's order, which is `pmap`'s order.
    pub fn pred_labels(&self) -> &[JavaString] {
        &self.pred_labels
    }

    pub fn num_predicates(&self) -> usize {
        self.pred_labels.len()
    }

    pub fn num_outcomes(&self) -> usize {
        self.outcome_labels.len()
    }

    /// The outcome ids predicate `p` has parameters for: all of them.
    pub fn pattern(&self, _p: usize) -> &[u32] {
        &self.pattern
    }

    /// Predicate `p`'s parameters, one per outcome.
    pub fn parameters(&self, p: usize) -> &[f64] {
        let n = self.outcome_labels.len();
        &self.parameters[p * n..(p + 1) * n]
    }

    /// `NaiveBayesModel.outcomeTotals`: sum of parameters per outcome.
    pub fn outcome_totals(&self) -> &[f64] {
        &self.outcome_totals
    }

    /// `NaiveBayesModel.eval(String[] context, float[] values, double[] outsums)`.
    ///
    /// # Panics
    ///
    /// If `out.len()` is not [`num_outcomes`](Self::num_outcomes), or if
    /// `values` is `Some(v)` and `v.len() < context.len()`.
    pub fn eval(&self, context: &[&str], values: Option<&[f32]>, out: &mut [f64]) {
        assert_eq!(out.len(), self.outcome_labels.len(), "the output must hold one probability per outcome");
        if let Some(v) = values {
            assert!(v.len() >= context.len(), "one value per context element");
        }
        out.fill(0.0);
        let vocabulary = self.pred_labels.len() as f64;
        let mut value = 1.0;
        for (ci, &feature) in context.iter().enumerate() {
            if let Some(&p) = self.pmap.get(&JavaString::new(feature)) {
                let active_params = self.parameters(p as usize);
                if let Some(v) = values {
                    value = f64::from(v[ci]);
                }
                for (i, (&denominator, sum)) in self.outcome_totals.iter().zip(out.iter_mut()).enumerate() {
                    let numerator = active_params[i] * value;
                    let prob = (numerator + LIDSTONE_DELTA) / (denominator + LIDSTONE_DELTA * vocabulary);
                    *sum += strict_math::log(prob);
                }
            }
        }
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

/// `TrainerFactory.getEventTrainer(...).train(events)` for `NAIVEBAYES`:
/// validates `Cutoff` and `Iterations`, computes the event checksum, runs
/// the indexer named by the parameters with `sort` off and the default
/// cutoff when none is given, then calls [`train`].
pub fn train_events(events: &[Event], params: &TrainingParameters) -> Result<(NaiveBayesModel, TrainingReport)> {
    // AbstractTrainer.validate, before the indexer runs.
    params.get_int(Key::CUTOFF)?;
    params.get_int(Key::ITERATIONS)?;
    let mut checksum = EventChecksum::new();
    for e in events {
        checksum.update(e);
    }
    let mut indexing = params.clone();
    indexing.put(Key::SORT, false);
    if indexing.get_int(Key::CUTOFF)?.unwrap_or(-1) == -1 {
        indexing.put(Key::CUTOFF, DEFAULT_CUTOFF);
    }
    let index = DataIndex::index(events, &IndexParams::from_training_parameters(&indexing)?)?;
    let model = train(&index, params)?;
    Ok((model, TrainingReport { event_hash: Some(checksum.value()), trainer_type: TrainerType::Event }))
}

/// `AbstractEventTrainer.train(DataIndexer)` for `NAIVEBAYES`: validates
/// `Cutoff` and `Iterations`, refuses an index with one outcome, then calls
/// [`train_model`].
pub fn train(index: &DataIndex, params: &TrainingParameters) -> Result<NaiveBayesModel> {
    params.get_int(Key::CUTOFF)?;
    params.get_int(Key::ITERATIONS)?;
    if index.outcome_labels().len() <= 1 {
        return Err(Error::InsufficientTrainingData(ONE_OUTCOME));
    }
    train_model(index)
}

/// `NaiveBayesTrainer.trainModel(DataIndexer)` without the outcome count
/// check. An index that keeps values with an event without any is
/// [`Error::MissingValues`].
pub fn train_model(index: &DataIndex) -> Result<NaiveBayesModel> {
    index.check_values()?;
    let outcome_labels = index.outcome_labels().to_vec();
    let pred_labels = index.pred_labels().to_vec();
    let num_outcomes = outcome_labels.len();
    let num_preds = pred_labels.len();
    let pmap = pred_labels.iter().enumerate().map(|(i, l)| (l.clone(), i as u32)).collect();
    let pattern = (0..num_outcomes as u32).collect();
    let mut parameters = vec![0.0; num_outcomes * num_preds];

    for ei in 0..index.num_unique_events() {
        let target = index.outcome(ei) as usize;
        let context = index.context(ei);
        let values = index.values(ei);
        for _ in 0..index.seen(ei) {
            for (ci, &pi) in context.iter().enumerate() {
                let step = values.map_or(1.0, |v| f64::from(v[ci]));
                parameters[pi as usize * num_outcomes + target] += step;
            }
        }
    }

    let mut outcome_totals = vec![0.0; num_outcomes];
    for p in 0..num_preds {
        for (o, total) in outcome_totals.iter_mut().enumerate() {
            *total += parameters[p * num_outcomes + o];
        }
    }

    Ok(NaiveBayesModel { outcome_labels, pred_labels, pmap, pattern, parameters, outcome_totals })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_valued_events_accumulate_weighted_counts() {
        let events = [
            Event::with_values(
                JavaString::new("pos"),
                vec![JavaString::new("good"), JavaString::new("film")],
                Some(vec![2.5, 1.0]),
            )
            .unwrap(),
            Event::with_values(
                JavaString::new("neg"),
                vec![JavaString::new("bad"), JavaString::new("film")],
                Some(vec![3.0, 1.0]),
            )
            .unwrap(),
        ];
        let mut params = TrainingParameters::new();
        params.put(Key::ALGORITHM, "NAIVEBAYES");
        params.put(Key::DATA_INDEXER, "OnePassRealValue");
        params.put(Key::CUTOFF, 0);
        let (model, report) = train_events(&events, &params).unwrap();
        assert_eq!(report.trainer_type, TrainerType::Event);
        assert_eq!(model.outcome_totals(), &[3.5, 4.0]);
        let mut probs = [0.0; 2];
        model.eval(&["good"], Some(&[2.0]), &mut probs);
        assert!(probs[0] > probs[1], "{probs:?}");
    }
}
