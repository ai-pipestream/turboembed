//! `opennlp.tools.ml.perceptron`: the perceptron trainer behind
//! `Algorithm=PERCEPTRON`, the sequence perceptron trainer behind
//! `PERCEPTRON_SEQUENCE`, and the model as `PerceptronModel` holds it,
//! computed as the JVM computes them so that every parameter agrees bit
//! for bit after every iteration.
//!
//! `PerceptronTrainer` runs over the indexed events, each as often as it
//! was seen, scores it with the current parameters (`PerceptronModel.eval`
//! without normalization: the sums of the parameters), and when the best
//! outcome is wrong adds the step to the target's parameters and takes it
//! from the predicted outcome's, for every predicate of the context. With
//! `UseAverage` (the default) the parameters are summed after every
//! iteration and the model is their mean. Training stops when the accuracy
//! differs from each of the previous three by less than `Tolerance`.
//!
//! `SimplePerceptronSequenceTrainer` runs over sequences of events. For
//! each sequence the stream tags the events with the current model
//! ([`SequenceSource::update_context`]); when any tag is wrong, the counts
//! of the correct events' features are added to and the counts of the
//! tagged events' features taken from the parameters, per outcome. With
//! `UseAverage` the mean over every iteration and sequence is kept lazily:
//! the time each parameter held its value, counted in sequences, weights
//! it.
//!
//! What the port keeps from the Java code, because the numbers depend on
//! it:
//!
//! - `float` where OpenNLP has `float`: a sequence's feature counts are
//!   `Float`s, and an event's values stay `f32` until they meet a `double`.
//! - The order of every sum: the events in index order, each repeated as
//!   often as seen with the parameters updated between the repetitions,
//!   and the parameters' sums per iteration.
//! - `UseSkippedAveraging` turns averaging on but changes nothing else:
//!   OpenNLP computes `useAverage && useSkippedAveraging && (i < 20 ||
//!   isPerfectSquare(i)) || useAverage`, which is `useAverage`. Every
//!   iteration is summed.
//! - The progress monitor hears of iterations 1 to 9 and every tenth.
//! - The sequence trainer keeps the time of each parameter's last change
//!   with the value truncated to an `int`, and weights it by `int`
//!   arithmetic.
//!
//! `StepSizeDecrease` above zero shrinks the step by that fraction before
//! every iteration; OpenNLP takes the value as a fraction, though its
//! check allows up to 100.

use std::collections::HashMap;

use crate::error::{Error, ONE_OUTCOME, Result};
use crate::event::{Event, EventChecksum};
use crate::gis::TrainingReport;
use crate::index::{DataIndex, IndexParams};
use crate::java_string::JavaString;
use crate::monitor::{
    DefaultProgressMonitor, IterDeltaAccuracyUnderTolerance, ProgressMonitor, StopCriteria, TrainingConfiguration,
    TrainingMeasure,
};
use crate::params::{DEFAULT_CUTOFF, DEFAULT_ITERATIONS, DEFAULT_TOLERANCE, Key, TrainerType, TrainingParameters};
use turbo_jvm::strict_math;

#[path = "perceptron_sequence.rs"]
mod sequence;
pub use sequence::{Sequence, SequenceSource, train_sequence_model, train_sequences};

/// What `PerceptronTrainer.doTrain` reads from the training parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerceptronParams {
    /// `Iterations`, default 100.
    pub iterations: i32,
    /// `UseAverage`, default true; forced on by `UseSkippedAveraging`.
    pub use_average: bool,
    /// `UseSkippedAveraging`, default false. Kept for the record; it
    /// changes nothing beyond `use_average` (see the module documentation).
    pub use_skipped_averaging: bool,
    /// `StepSizeDecrease` when above zero: the fraction the step shrinks
    /// by before each iteration.
    pub step_size_decrease: Option<f64>,
    /// `Tolerance`, default 0.00001.
    pub tolerance: f64,
}

impl Default for PerceptronParams {
    fn default() -> PerceptronParams {
        PerceptronParams {
            iterations: DEFAULT_ITERATIONS,
            use_average: true,
            use_skipped_averaging: false,
            step_size_decrease: None,
            tolerance: DEFAULT_TOLERANCE,
        }
    }
}

impl PerceptronParams {
    /// `PerceptronTrainer.doTrain`'s reading of the parameters, with
    /// `setStepSizeDecrease` and `setTolerance` refusing what they refuse
    /// (`IllegalArgumentException`).
    pub fn from_training_parameters(params: &TrainingParameters) -> Result<PerceptronParams> {
        let mut use_average = params.get_bool(Key::USE_AVERAGE)?.unwrap_or(true);
        let use_skipped_averaging = params.get_bool(Key::USE_SKIPPED_AVERAGING)?.unwrap_or(false);
        if use_skipped_averaging {
            use_average = true;
        }
        let decrease = params.get_double(Key::STEP_SIZE_DECREASE)?.unwrap_or(0.0);
        let tolerance = params.get_double(Key::TOLERANCE)?.unwrap_or(DEFAULT_TOLERANCE);
        let step_size_decrease = if decrease > 0.0 {
            if decrease > 100.0 {
                return Err(Error::StepSizeDecrease(decrease));
            }
            Some(decrease)
        } else {
            None
        };
        if tolerance < 0.0 {
            return Err(Error::Tolerance(tolerance));
        }
        Ok(PerceptronParams {
            iterations: params.get_int(Key::ITERATIONS)?.unwrap_or(DEFAULT_ITERATIONS),
            use_average,
            use_skipped_averaging,
            step_size_decrease,
            tolerance,
        })
    }
}

/// A trained perceptron model: `PerceptronModel`'s data structures. Every
/// predicate has a parameter for every outcome, in outcome order; the
/// writer drops the zero ones.
#[derive(Debug, Clone, PartialEq)]
pub struct PerceptronModel {
    outcome_labels: Vec<JavaString>,
    pred_labels: Vec<JavaString>,
    /// `pmap`: predicate label to index.
    pmap: HashMap<JavaString, u32>,
    /// `0..num_outcomes`, the pattern of every predicate.
    pattern: Vec<u32>,
    /// Predicate `p`'s parameters are `parameters[p * n..(p + 1) * n]` for
    /// `n` outcomes.
    pub(super) parameters: Vec<f64>,
}

impl PerceptronModel {
    pub(super) fn zeros(outcome_labels: Vec<JavaString>, pred_labels: Vec<JavaString>) -> PerceptronModel {
        let pmap = pred_labels.iter().enumerate().map(|(i, l)| (l.clone(), i as u32)).collect();
        let parameters = vec![0.0; outcome_labels.len() * pred_labels.len()];
        let pattern = (0..outcome_labels.len() as u32).collect();
        PerceptronModel { outcome_labels, pred_labels, pmap, pattern, parameters }
    }

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

    /// Predicate `p`'s parameters, one per outcome (`Context.getParameters`).
    pub fn parameters(&self, p: usize) -> &[f64] {
        let n = self.num_outcomes();
        &self.parameters[p * n..(p + 1) * n]
    }

    /// Every parameter, predicates in order, each predicate's per outcome.
    pub fn all_parameters(&self) -> &[f64] {
        &self.parameters
    }

    /// `pmap.get(predicate)`.
    pub fn predicate_index(&self, label: &JavaString) -> Option<u32> {
        self.pmap.get(label).copied()
    }

    /// `PerceptronModel.eval(String[] context, float[] values)`: the
    /// probabilities of the outcomes, into `out`. Unknown predicates are
    /// skipped; a value applies to the predicate at its position.
    ///
    /// # Panics
    ///
    /// If `out.len()` is not [`num_outcomes`](Self::num_outcomes), or if
    /// `values` is `Some(v)` and `v.len() < context.len()`.
    pub fn eval(&self, context: &[JavaString], values: Option<&[f32]>, out: &mut [f64]) {
        assert_eq!(out.len(), self.outcome_labels.len(), "the output must hold one probability per outcome");
        if let Some(v) = values {
            assert!(v.len() >= context.len(), "one value per context element");
        }
        out.fill(0.0);
        for (ci, label) in context.iter().enumerate() {
            if let Some(p) = self.predicate_index(label) {
                let value = values.map_or(1.0, |v| f64::from(v[ci]));
                self.sum_features(p, value, out);
            }
        }
        // The normalization of eval(..., normalize = true).
        let mut max_prior = 1.0;
        for &x in out.iter() {
            if max_prior < x.abs() {
                max_prior = x.abs();
            }
        }
        let mut normal = 0.0;
        for x in out.iter_mut() {
            *x = strict_math::exp(*x / max_prior);
            normal += *x;
        }
        for x in out.iter_mut() {
            *x /= normal;
        }
    }

    /// `ArrayMath.sumFeatures` for one predicate.
    #[inline]
    fn sum_features(&self, p: u32, value: f64, out: &mut [f64]) {
        for (o, &param) in out.iter_mut().zip(self.parameters(p as usize)) {
            *o += param * value;
        }
    }

    /// The raw sums of an indexed event's context (`eval` with
    /// `normalize = false`), into `out`.
    fn sums(&self, context: &[u32], values: Option<&[f32]>, out: &mut [f64]) {
        out.fill(0.0);
        for (ci, &p) in context.iter().enumerate() {
            let value = values.map_or(1.0, |v| f64::from(v[ci]));
            self.sum_features(p, value, out);
        }
    }

    #[inline]
    pub(super) fn cell(&mut self, p: u32, o: usize) -> &mut f64 {
        let n = self.outcome_labels.len();
        &mut self.parameters[p as usize * n + o]
    }
}

/// `ArrayMath.argmax`: the first index of the largest value.
fn argmax(x: &[f64]) -> usize {
    let mut max = 0;
    for (i, &v) in x.iter().enumerate().skip(1) {
        if x[max] < v {
            max = i;
        }
    }
    max
}

/// What a trainer tells after every iteration: the iteration and the
/// parameters a run stopping there would return.
pub type Observer<'a> = &'a mut dyn FnMut(&PerceptronIteration, &[f64]);

/// One iteration of `PerceptronTrainer` as its progress monitor would hear
/// of it, had it heard of every one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerceptronIteration {
    /// 1-based.
    pub iteration: i32,
    /// Events classified right, each counted as often as seen.
    pub correct: i32,
    /// `numEvents`.
    pub total: i32,
    /// `correct / total`.
    pub accuracy: f64,
}

/// `TrainerFactory.getEventTrainer(...).train(events)` for `PERCEPTRON`:
/// the checksum over the events, the indexer named by the parameters with
/// `sort` off and the default cutoff when none is given, then [`train`].
pub fn train_events(
    events: &[Event],
    params: &TrainingParameters,
    config: &mut TrainingConfiguration,
) -> Result<(PerceptronModel, TrainingReport)> {
    // AbstractTrainer.validate, before the indexer runs: Cutoff and
    // Iterations must parse (the indexer reads Cutoff itself).
    params.get_int(Key::ITERATIONS)?;
    params.get_int(Key::CUTOFF)?;
    let mut checksum = EventChecksum::new();
    for e in events {
        checksum.update(e);
    }
    let mut indexing = params.clone();
    indexing.put(Key::SORT, false);
    indexing.put_if_absent(Key::CUTOFF, DEFAULT_CUTOFF);
    let index = DataIndex::index(events, &IndexParams::from_training_parameters(&indexing)?)?;
    let model = train(&index, params, config)?;
    Ok((model, TrainingReport { event_hash: Some(checksum.value()), trainer_type: TrainerType::Event }))
}

/// `AbstractEventTrainer.train(DataIndexer)` for `PERCEPTRON`: refuses an
/// index with one outcome, then trains with [`PerceptronParams`] read from
/// the parameters and `IterDeltaAccuracyUnderTolerance` from them too.
pub fn train(
    index: &DataIndex,
    params: &TrainingParameters,
    config: &mut TrainingConfiguration,
) -> Result<PerceptronModel> {
    if index.outcome_labels().len() <= 1 {
        return Err(Error::InsufficientTrainingData(ONE_OUTCOME));
    }
    let perceptron = PerceptronParams::from_training_parameters(params)?;
    let default_stop = IterDeltaAccuracyUnderTolerance::from_parameters(params)?;
    train_model(index, &perceptron, config, &default_stop, None)
}

/// `PerceptronTrainer.trainModel(iterations, indexer, cutoff, useAverage)`
/// without the outcome check, with `default_stop` where `config` has no
/// criterion and the default monitor where it has none. After every
/// iteration an observer, when given, hears the iteration and the
/// parameters a run stopping there would return: the mean so far with
/// `UseAverage`, the parameters themselves without. An index that keeps
/// values with an event without any is [`Error::MissingValues`].
pub fn train_model(
    index: &DataIndex,
    params: &PerceptronParams,
    config: &mut TrainingConfiguration,
    default_stop: &dyn StopCriteria,
    mut observe: Option<Observer<'_>>,
) -> Result<PerceptronModel> {
    // The first iteration reads values[ei][ci] of every event.
    index.check_values()?;
    let mut default_monitor = DefaultProgressMonitor::new();
    let monitor: &mut dyn ProgressMonitor = config.monitor.as_deref_mut().unwrap_or(&mut default_monitor);
    let stop: &dyn StopCriteria = config.stop.as_deref().unwrap_or(default_stop);
    let mut model = PerceptronModel::zeros(index.outcome_labels().to_vec(), index.pred_labels().to_vec());
    let num_outcomes = model.num_outcomes();
    let mut summed = vec![0.0; model.parameters.len()];
    let mut times_summed = 0;
    // The mean so far, for the observer.
    let mut mean = vec![0.0; if observe.is_some() && params.use_average { model.parameters.len() } else { 0 }];
    let total = index.num_events() as i32;
    let mut dist = vec![0.0; num_outcomes];
    let (mut prev1, mut prev2, mut prev3) = (0.0, 0.0, 0.0);
    let mut step = 1.0;
    for i in 1..=params.iterations {
        if let Some(decrease) = params.step_size_decrease {
            step *= 1.0 - decrease;
        }
        let mut correct = 0;
        for ei in 0..index.num_unique_events() {
            let target = index.outcome(ei) as usize;
            let context = index.context(ei);
            let values = index.values(ei);
            for _ in 0..index.seen(ei) {
                model.sums(context, values, &mut dist);
                let max = argmax(&dist);
                if max != target {
                    for (ci, &p) in context.iter().enumerate() {
                        match values {
                            None => {
                                *model.cell(p, target) += step;
                                *model.cell(p, max) += -step;
                            }
                            Some(v) => {
                                *model.cell(p, target) += step * f64::from(v[ci]);
                                *model.cell(p, max) += -step * f64::from(v[ci]);
                            }
                        }
                    }
                } else {
                    correct += 1;
                }
            }
        }
        let accuracy = f64::from(correct) / f64::from(total);
        if i < 10 || i % 10 == 0 {
            monitor.finished_iteration(i, correct, total, TrainingMeasure::Accuracy, accuracy);
        }
        if params.use_average {
            times_summed += 1;
            for (s, &p) in summed.iter_mut().zip(&model.parameters) {
                *s += p;
            }
        }
        if let Some(observe) = observe.as_deref_mut() {
            let output: &[f64] = if params.use_average {
                for (m, &s) in mean.iter_mut().zip(&summed) {
                    *m = s / f64::from(times_summed);
                }
                &mean
            } else {
                &model.parameters
            };
            observe(&PerceptronIteration { iteration: i, correct, total, accuracy }, output);
        }
        if stop.test(prev1 - accuracy) && stop.test(prev2 - accuracy) && stop.test(prev3 - accuracy) {
            monitor.finished_training(params.iterations, Some(stop));
            break;
        }
        prev1 = prev2;
        prev2 = prev3;
        prev3 = accuracy;
    }
    if !monitor.is_training_finished() {
        monitor.finished_training(params.iterations, None);
    }
    monitor.display(true);
    if params.use_average {
        // With no iteration run this is 0.0 / 0, NaN, as on the JVM.
        for (p, s) in model.parameters.iter_mut().zip(&summed) {
            *p = s / f64::from(times_summed);
        }
    }
    Ok(model)
}
