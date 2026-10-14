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
use crate::index::{DataIndex, IndexParams, IndexerKind};
use crate::java_string::JavaString;
use crate::monitor::{
    DefaultProgressMonitor, IterDeltaAccuracyUnderTolerance, ProgressMonitor, StopCriteria, TrainingConfiguration,
    TrainingMeasure,
};
use crate::params::{DEFAULT_CUTOFF, DEFAULT_ITERATIONS, DEFAULT_TOLERANCE, Key, TrainerType, TrainingParameters};
use turbo_jvm::strict_math;

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
    outcome_labels: Vec<String>,
    pred_labels: Vec<JavaString>,
    /// `pmap`: predicate label to index.
    pmap: HashMap<JavaString, u32>,
    /// `0..num_outcomes`, the pattern of every predicate.
    pattern: Vec<u32>,
    /// Predicate `p`'s parameters are `parameters[p * n..(p + 1) * n]` for
    /// `n` outcomes.
    parameters: Vec<f64>,
}

impl PerceptronModel {
    fn zeros(outcome_labels: Vec<String>, pred_labels: Vec<JavaString>) -> PerceptronModel {
        let pmap = pred_labels.iter().enumerate().map(|(i, l)| (l.clone(), i as u32)).collect();
        let parameters = vec![0.0; outcome_labels.len() * pred_labels.len()];
        let pattern = (0..outcome_labels.len() as u32).collect();
        PerceptronModel { outcome_labels, pred_labels, pmap, pattern, parameters }
    }

    pub fn outcome_labels(&self) -> &[String] {
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
    pub fn eval(&self, context: &[JavaString], values: Option<&[f32]>, out: &mut [f64]) {
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
    fn cell(&mut self, p: u32, o: usize) -> &mut f64 {
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
    Ok(train_model(index, &perceptron, config, &default_stop, &mut |_, _| {}))
}

/// `PerceptronTrainer.trainModel(iterations, indexer, cutoff, useAverage)`
/// without the outcome check, with `default_stop` where `config` has no
/// criterion and the default monitor where it has none. After every
/// iteration `observe` is given the iteration and the parameters a run
/// stopping there would return: the mean so far with `UseAverage`, the
/// parameters themselves without.
pub fn train_model(
    index: &DataIndex,
    params: &PerceptronParams,
    config: &mut TrainingConfiguration,
    default_stop: &dyn StopCriteria,
    observe: &mut dyn FnMut(&PerceptronIteration, &[f64]),
) -> PerceptronModel {
    let mut default_monitor = DefaultProgressMonitor::new();
    let monitor: &mut dyn ProgressMonitor = match &mut config.monitor {
        Some(m) => m.as_mut(),
        None => &mut default_monitor,
    };
    let stop: &dyn StopCriteria = match &config.stop {
        Some(s) => s.as_ref(),
        None => default_stop,
    };
    let mut model = PerceptronModel::zeros(index.outcome_labels().to_vec(), index.pred_labels().to_vec());
    let num_outcomes = model.num_outcomes();
    let mut summed = vec![0.0; model.parameters.len()];
    let mut times_summed = 0;
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
        let result;
        let output: &[f64] = if params.use_average {
            result = summed.iter().map(|s| s / f64::from(times_summed)).collect::<Vec<f64>>();
            &result
        } else {
            &model.parameters
        };
        observe(&PerceptronIteration { iteration: i, correct, total, accuracy }, output);
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
        for (p, s) in model.parameters.iter_mut().zip(&summed) {
            *p = s / f64::from(times_summed);
        }
    }
    model
}

/// `Sequence<T>` without its source: the events of one sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct Sequence {
    pub events: Vec<Event>,
}

/// `SequenceStream<Event>`: the sequences, read again every iteration, and
/// `updateContext`, which tags a sequence's events with a model and
/// returns them with the model's outcomes, one per event in order.
pub trait SequenceSource {
    fn sequences(&self) -> &[Sequence];

    /// `updateContext(sequence, model)`: the events as the model tags
    /// them. Each must carry an outcome of the model's.
    fn update_context(&self, sequence: &Sequence, model: &PerceptronModel) -> Vec<Event>;
}

/// `SimplePerceptronSequenceTrainer`'s record of a parameter's last change,
/// for the lazy average: `updates[pi][oi]`.
#[derive(Debug, Clone, Copy, Default)]
struct LastChange {
    /// The parameter after the change, truncated to an `int`.
    value: i32,
    /// The 0-based iteration of the change.
    iteration: i32,
    /// The sequence of the change.
    sequence: i32,
}

/// `TrainerFactory.getEventModelSequenceTrainer(...).train(stream)` for
/// `PERCEPTRON_SEQUENCE`: `Iterations`, `Cutoff` and `UseAverage` from the
/// parameters, the events indexed by `OnePassDataIndexer` unsorted, then
/// the sequence perceptron. There is no outcome check here: a stream with
/// one outcome trains.
pub fn train_sequences(
    source: &dyn SequenceSource,
    params: &TrainingParameters,
) -> Result<(PerceptronModel, TrainingReport)> {
    let iterations = params.get_int(Key::ITERATIONS)?.unwrap_or(DEFAULT_ITERATIONS);
    let cutoff = params.get_int(Key::CUTOFF)?.unwrap_or(DEFAULT_CUTOFF);
    let use_average = params.get_bool(Key::USE_AVERAGE)?.unwrap_or(true);
    let model = train_sequence_model(source, iterations, cutoff, use_average, &mut |_, _| {})?;
    Ok((model, TrainingReport { event_hash: None, trainer_type: TrainerType::EventModelSequence }))
}

/// `SimplePerceptronSequenceTrainer.trainModel(iterations, stream, cutoff,
/// useAverage)`. After every iteration `observe` is given the iteration
/// (with the events tagged right over the events, as the trainer logs
/// them) and the parameters a run stopping there would return.
pub fn train_sequence_model(
    source: &dyn SequenceSource,
    iterations: i32,
    cutoff: i32,
    use_average: bool,
    observe: &mut dyn FnMut(&PerceptronIteration, &[f64]),
) -> Result<PerceptronModel> {
    let sequences = source.sequences();
    let events: Vec<Event> = sequences.iter().flat_map(|s| s.events.iter().cloned()).collect();
    let index = DataIndex::index(&events, &IndexParams::new(IndexerKind::OnePass, cutoff, false))?;
    // trainingStats, after the last iteration, reads outcomeList[oei] for
    // every event of the stream; the list has one entry per event kept.
    if (index.num_events() as usize) < events.len() {
        return Err(Error::SequenceEventsDropped { indexed: index.num_events(), streamed: events.len() });
    }
    let num_sequences = sequences.len() as i32;
    let total = index.num_events() as i32;
    let mut model = PerceptronModel::zeros(index.outcome_labels().to_vec(), index.pred_labels().to_vec());
    let num_outcomes = model.num_outcomes();
    let omap: HashMap<String, usize> = index.outcome_labels().iter().enumerate().map(|(i, o)| (o.clone(), i)).collect();
    let mut average = vec![0.0; model.parameters.len()];
    let mut changes = vec![LastChange::default(); model.parameters.len()];

    for iteration in 0..iterations {
        let mut correct = 0;
        for (si, sequence) in (0i32..).zip(sequences) {
            let tagged = source.update_context(sequence, &model);
            assert_eq!(tagged.len(), sequence.events.len(), "updateContext must return one event per event");
            let mut update = false;
            for (t, e) in tagged.iter().zip(&sequence.events) {
                if t.outcome() != e.outcome() {
                    update = true;
                } else {
                    correct += 1;
                }
            }
            if !update {
                continue;
            }
            // `featureCounts`: one map per outcome, filled for this sequence.
            let mut counts: Vec<HashMap<&JavaString, f32>> = (0..num_outcomes).map(|_| HashMap::new()).collect();
            for e in &sequence.events {
                let oi = omap[e.outcome()];
                for (ci, feature) in e.context().iter().enumerate() {
                    let value = e.values().map_or(1.0f32, |v| v[ci]);
                    counts[oi].entry(feature).and_modify(|c| *c += value).or_insert(value);
                }
            }
            for t in &tagged {
                let oi = *omap
                    .get(t.outcome())
                    .unwrap_or_else(|| panic!("updateContext tagged {:?}, not an outcome", t.outcome()));
                for (ci, feature) in t.context().iter().enumerate() {
                    let value = t.values().map_or(1.0f32, |v| v[ci]);
                    let c = match counts[oi].get(feature) {
                        None => -value,
                        Some(&c) => c - value,
                    };
                    if c == 0.0 {
                        counts[oi].remove(feature);
                    } else {
                        counts[oi].insert(feature, c);
                    }
                }
            }
            for (oi, c) in counts.iter().enumerate() {
                for (feature, &count) in c {
                    let Some(pi) = model.predicate_index(feature) else { continue };
                    *model.cell(pi, oi) += f64::from(count);
                    if use_average {
                        let at = pi as usize * num_outcomes + oi;
                        let last = changes[at];
                        if last.value != 0 {
                            let held = num_sequences
                                .wrapping_mul(iteration.wrapping_sub(last.iteration))
                                .wrapping_add(si.wrapping_sub(last.sequence));
                            average[at] += f64::from(last.value.wrapping_mul(held));
                        }
                        changes[at] = LastChange { value: *model.cell(pi, oi) as i32, iteration, sequence: si };
                    }
                }
            }
        }
        let accuracy = f64::from(correct) / f64::from(total);
        let result;
        let output: &[f64] = if use_average {
            result = finish_average(&average, &changes, num_sequences, iteration + 1);
            &result
        } else {
            &model.parameters
        };
        observe(&PerceptronIteration { iteration: iteration + 1, correct, total, accuracy }, output);
    }
    if use_average {
        model.parameters = finish_average(&average, &changes, num_sequences, iterations);
    }
    Ok(model)
}

/// The end of `nextIteration` on the last iteration: each parameter's
/// pending weight added and the sum divided by the iterations times the
/// sequences. `ran` is the iterations run so far, which the trainer only
/// ever reaches with `ran == iterations`; the observer sees the mean a run
/// of `ran` iterations would end with.
fn finish_average(average: &[f64], changes: &[LastChange], num_sequences: i32, ran: i32) -> Vec<f64> {
    let total = f64::from(ran) * f64::from(num_sequences);
    let mut out = average.to_vec();
    for (at, p) in out.iter_mut().enumerate() {
        let last = changes[at];
        if last.value != 0 {
            let held = num_sequences.wrapping_mul(ran.wrapping_sub(last.iteration)).wrapping_sub(last.sequence);
            *p += f64::from(last.value.wrapping_mul(held));
        }
        if *p != 0.0 {
            *p /= total;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::IndexerKind;

    fn event(outcome: &str, context: &str) -> Event {
        Event::from_strs(outcome, &context.split(' ').collect::<Vec<_>>())
    }

    fn events() -> Vec<Event> {
        vec![
            event("yes", "w=a len=1 cap"),
            event("yes", "w=a len=2"),
            event("yes", "w=b cap"),
            event("no", "w=c len=1"),
            event("no", "w=c len=3"),
            event("no", "w=d len=3"),
            event("maybe", "w=b len=2"),
            event("maybe", "w=d cap"),
        ]
    }

    fn index() -> DataIndex {
        DataIndex::index(&events(), &IndexParams::new(IndexerKind::OnePass, 0, false)).unwrap()
    }

    #[test]
    fn params_come_from_the_training_parameters_and_are_validated() {
        let mut p = TrainingParameters::new();
        assert_eq!(PerceptronParams::from_training_parameters(&p).unwrap(), PerceptronParams::default());
        p.put(Key::USE_AVERAGE, false);
        p.put(Key::STEP_SIZE_DECREASE, 0.5);
        p.put(Key::TOLERANCE, "0.1");
        p.put(Key::ITERATIONS, 3);
        assert_eq!(
            PerceptronParams::from_training_parameters(&p).unwrap(),
            PerceptronParams {
                iterations: 3,
                use_average: false,
                use_skipped_averaging: false,
                step_size_decrease: Some(0.5),
                tolerance: 0.1
            }
        );
        // Skipped averaging forces averaging on.
        p.put(Key::USE_SKIPPED_AVERAGING, true);
        let got = PerceptronParams::from_training_parameters(&p).unwrap();
        assert!(got.use_average && got.use_skipped_averaging);
        p.put(Key::STEP_SIZE_DECREASE, 101.0);
        assert_eq!(PerceptronParams::from_training_parameters(&p), Err(Error::StepSizeDecrease(101.0)));
        p.put(Key::STEP_SIZE_DECREASE, -1.0);
        assert_eq!(PerceptronParams::from_training_parameters(&p).unwrap().step_size_decrease, None);
        p.put(Key::TOLERANCE, -0.5);
        assert_eq!(PerceptronParams::from_training_parameters(&p), Err(Error::Tolerance(-0.5)));
    }

    #[test]
    fn one_outcome_is_refused_before_training() {
        let events = [event("s", "a"), event("s", "b")];
        let index = DataIndex::index(&events, &IndexParams::new(IndexerKind::OnePass, 0, false)).unwrap();
        assert_eq!(
            train(&index, &TrainingParameters::new(), &mut TrainingConfiguration::default()).err(),
            Some(Error::InsufficientTrainingData(ONE_OUTCOME))
        );
    }

    #[test]
    fn training_fits_the_events_and_reports_some_iterations() {
        let mut p = TrainingParameters::defaults();
        p.put(Key::CUTOFF, 0);
        p.put(Key::ITERATIONS, 100);
        p.put(Key::TOLERANCE, 0.0);
        // The raw parameters: the mean over the iterations can leave an
        // event wrongly tagged that the last iteration's parameters tag right.
        p.put(Key::USE_AVERAGE, false);
        let mut config = TrainingConfiguration::new(Some(Box::new(DefaultProgressMonitor::new())), None);
        let (model, report) = train_events(&events(), &p, &mut config).unwrap();
        assert_eq!(model.outcome_labels(), ["yes", "no", "maybe"]);
        assert_eq!(model.num_predicates(), 8);
        assert_eq!(model.pattern(0), [0, 1, 2]);
        assert_eq!(report.trainer_type, TrainerType::Event);
        let mut out = vec![0.0; 3];
        for e in events() {
            model.eval(e.context(), None, &mut out);
            let best = model.outcome_labels()[argmax(&out)].as_str();
            assert_eq!(best, e.outcome(), "{e:?}: {out:?}");
        }
        assert!(config.monitor.unwrap().is_training_finished());
    }

    #[test]
    fn the_monitor_hears_of_the_first_nine_and_every_tenth_iteration() {
        use std::cell::RefCell;
        use std::rc::Rc;
        struct Heard(Rc<RefCell<Vec<i32>>>);
        impl ProgressMonitor for Heard {
            fn finished_iteration(&mut self, i: i32, _: i32, _: i32, m: TrainingMeasure, _: f64) {
                assert_eq!(m, TrainingMeasure::Accuracy);
                self.0.borrow_mut().push(i);
            }
            fn finished_training(&mut self, _: i32, _: Option<&dyn StopCriteria>) {}
            fn is_training_finished(&self) -> bool {
                false
            }
            fn display(&mut self, _: bool) {}
        }
        let heard = Rc::new(RefCell::new(Vec::new()));
        let params = PerceptronParams { iterations: 25, tolerance: 0.0, ..PerceptronParams::default() };
        let mut config = TrainingConfiguration::new(Some(Box::new(Heard(Rc::clone(&heard)))), None);
        let stop = IterDeltaAccuracyUnderTolerance { tolerance: 0.0 };
        let mut seen = Vec::new();
        train_model(&index(), &params, &mut config, &stop, &mut |it, _| seen.push(it.iteration));
        // The observer hears every iteration; the monitor only the first
        // nine and every tenth.
        assert_eq!(seen, (1..=25).collect::<Vec<_>>());
        assert_eq!(*heard.borrow(), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]);
    }

    #[test]
    fn a_shorter_run_is_a_prefix_and_averaging_is_the_mean() {
        let index = index();
        let stop = IterDeltaAccuracyUnderTolerance { tolerance: 0.0 };
        let raw = PerceptronParams { iterations: 6, use_average: false, tolerance: 0.0, ..PerceptronParams::default() };
        let mut raws = Vec::new();
        let model = train_model(&index, &raw, &mut TrainingConfiguration::default(), &stop, &mut |_, p| {
            raws.push(p.to_vec());
        });
        assert_eq!(model.all_parameters(), &raws[5][..]);
        let short = PerceptronParams { iterations: 2, ..raw };
        let two = train_model(&index, &short, &mut TrainingConfiguration::default(), &stop, &mut |_, _| {});
        assert_eq!(two.all_parameters(), &raws[1][..]);
        // The averaged model is the mean of the raw parameters.
        let averaged = PerceptronParams { use_average: true, ..raw };
        let mut means = Vec::new();
        let model = train_model(&index, &averaged, &mut TrainingConfiguration::default(), &stop, &mut |_, p| {
            means.push(p.to_vec());
        });
        for k in 0..6 {
            let mut sum = vec![0.0; raws[0].len()];
            for r in &raws[..=k] {
                for (s, x) in sum.iter_mut().zip(r) {
                    *s += x;
                }
            }
            let mean: Vec<f64> = sum.iter().map(|s| s / (k + 1) as f64).collect();
            assert_eq!(means[k], mean, "after iteration {}", k + 1);
        }
        assert_eq!(model.all_parameters(), &means[5][..]);
        // Skipped averaging changes nothing.
        let skipped = PerceptronParams { use_skipped_averaging: true, ..averaged };
        let same = train_model(&index, &skipped, &mut TrainingConfiguration::default(), &stop, &mut |_, _| {});
        assert_eq!(same, model);
    }

    #[test]
    fn the_tolerance_stops_after_three_steady_iterations() {
        let index = index();
        let params = PerceptronParams { iterations: 100, ..PerceptronParams::default() };
        let stop = IterDeltaAccuracyUnderTolerance::default();
        let mut config = TrainingConfiguration::new(Some(Box::new(DefaultProgressMonitor::new())), None);
        let mut accuracies = Vec::new();
        train_model(&index, &params, &mut config, &stop, &mut |it, _| accuracies.push(it.accuracy));
        let n = accuracies.len();
        assert!((4..100).contains(&n), "{n}");
        let last = accuracies[n - 1];
        assert!(accuracies[n - 4..n - 1].iter().all(|a| (a - last).abs() < stop.tolerance));
    }

    #[test]
    fn the_step_decreases_by_the_fraction() {
        let index = index();
        let stop = IterDeltaAccuracyUnderTolerance { tolerance: 0.0 };
        let params = PerceptronParams {
            iterations: 1,
            use_average: false,
            step_size_decrease: Some(0.5),
            tolerance: 0.0,
            ..PerceptronParams::default()
        };
        let model = train_model(&index, &params, &mut TrainingConfiguration::default(), &stop, &mut |_, _| {});
        // One iteration with step 0.5: every parameter is a multiple of 0.5.
        assert!(model.all_parameters().iter().all(|p| (p * 2.0).fract() == 0.0));
        assert!(model.all_parameters().iter().any(|&p| p == 0.5 || p == -0.5));
    }

    /// Sequences of two events, tagged by the model on each event's context.
    struct Pairs(Vec<Sequence>);

    impl SequenceSource for Pairs {
        fn sequences(&self) -> &[Sequence] {
            &self.0
        }

        fn update_context(&self, sequence: &Sequence, model: &PerceptronModel) -> Vec<Event> {
            let mut out = vec![0.0; model.num_outcomes()];
            sequence
                .events
                .iter()
                .map(|e| {
                    model.eval(e.context(), e.values(), &mut out);
                    Event::with_values(
                        &model.outcome_labels()[argmax(&out)],
                        e.context().to_vec(),
                        e.values().map(<[f32]>::to_vec),
                    )
                    .unwrap()
                })
                .collect()
        }
    }

    fn pairs() -> Pairs {
        Pairs(events().chunks(2).map(|c| Sequence { events: c.to_vec() }).collect())
    }

    #[test]
    fn the_sequence_trainer_fits_the_sequences() {
        let mut p = TrainingParameters::new();
        p.put(Key::CUTOFF, 0);
        p.put(Key::ITERATIONS, 10);
        let (model, report) = train_sequences(&pairs(), &p).unwrap();
        assert_eq!(report.trainer_type, TrainerType::EventModelSequence);
        assert_eq!(report.event_hash, None);
        assert_eq!(model.num_predicates(), 8);
        let source = pairs();
        for s in source.sequences() {
            let tagged = source.update_context(s, &model);
            for (t, e) in tagged.iter().zip(&s.events) {
                assert_eq!(t.outcome(), e.outcome());
            }
        }
        // Without averaging the parameters are whole counts.
        p.put(Key::USE_AVERAGE, false);
        let (raw, _) = train_sequences(&pairs(), &p).unwrap();
        assert!(raw.all_parameters().iter().all(|x| x.fract() == 0.0));
        assert_ne!(raw, model);
    }

    #[test]
    fn the_sequence_trainer_fails_as_the_jvm_does_when_the_cutoff_drops_an_event() {
        // Only "cap" is seen three times, so cutoff 3 drops every event
        // without it: five of the eight.
        let mut p = TrainingParameters::new();
        p.put(Key::CUTOFF, 3);
        p.put(Key::ITERATIONS, 2);
        let index = DataIndex::index(&events(), &IndexParams::new(IndexerKind::OnePass, 3, false)).unwrap();
        assert_eq!(index.num_events(), 3);
        assert_eq!(train_sequences(&pairs(), &p).err(), Some(Error::SequenceEventsDropped { indexed: 3, streamed: 8 }));
    }

    #[test]
    fn a_shorter_sequence_run_is_a_prefix() {
        let source = pairs();
        let mut seen = Vec::new();
        let model =
            train_sequence_model(&source, 5, 0, true, &mut |it, p| seen.push((it.iteration, p.to_vec()))).unwrap();
        assert_eq!(seen.len(), 5);
        assert_eq!(model.all_parameters(), &seen[4].1[..]);
        let three = train_sequence_model(&source, 3, 0, true, &mut |_, _| {}).unwrap();
        assert_eq!(three.all_parameters(), &seen[2].1[..]);
    }
}
