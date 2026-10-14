//! `opennlp.tools.ml.maxent.GISTrainer`: generalized iterative scaling,
//! the trainer behind `Algorithm=MAXENT`, computed as the JVM computes
//! it so that every parameter agrees bit for bit after every iteration.
//!
//! What the port keeps from the Java code, because the numbers depend on
//! it:
//!
//! - `float` where OpenNLP has `float`: the predicate counts are summed in
//!   `f32` (`numTimesEventsSeen * value` as a `float` product), and the
//!   correction constant of real valued events is an `f32` sum of the
//!   event's values, compared and kept as a `double`.
//! - `StrictMath.log` and `StrictMath.exp` (`turbo_jvm::strict_math`) for
//!   the prior, the probabilities, the log likelihood and the update.
//! - The order of every sum: the expectations accumulate event by event
//!   in index order, with the event's count multiplied last
//!   (`dist * value * seen`), the log likelihood and the correct count
//!   accumulate over the events, and with `Threads` above one the events
//!   are cut into the same chunks as the JVM's tasks and each chunk's sums
//!   are added in task order. The JVM adds the tasks' log likelihoods in
//!   the order the tasks finish, which is not fixed, so with more than one
//!   thread the JVM itself does not reproduce its log likelihood bit for
//!   bit; the parameters do not depend on that order.
//! - The stopping rules: training stops early when the log likelihood
//!   falls (OpenNLP logs "Model Diverging"), or when the stop criterion
//!   says so (by default, a rise under `LLThreshold`), tested from the
//!   second iteration on.
//!
//! The values an `OnePassRealValue` index keeps are the values of the
//! event's full context (see [`crate::index`]); the trainer reads them by
//! position as the JVM does, so after a cutoff drop it uses the same wrong
//! values the JVM uses.

use crate::error::{Error, ONE_OUTCOME, Result};
use crate::event::{Event, EventChecksum};
use crate::index::{DataIndex, IndexParams};
use crate::java_string::JavaString;
use crate::monitor::{
    DefaultProgressMonitor, LogLikelihoodThresholdBreached, ProgressMonitor, StopCriteria, TrainingConfiguration,
    TrainingMeasure,
};
use crate::params::{DEFAULT_CUTOFF, DEFAULT_ITERATIONS, Key, TrainerType, TrainingParameters};
use turbo_jvm::strict_math;

/// `GISTrainer`'s smoothing: none (the default), the simple kind that
/// gives every predicate every outcome with `SmoothingObservation` as the
/// observed count of the unseen ones, or the Gaussian prior with sigma.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Smoothing {
    None,
    Simple { observation: f64 },
    Gaussian { sigma: f64 },
}

/// `SmoothingObservation`'s default.
pub const DEFAULT_SMOOTHING_OBSERVATION: f64 = 0.1;
/// `GaussianSmoothingSigma`'s default.
pub const DEFAULT_GAUSSIAN_SIGMA: f64 = 2.0;

/// What `GISTrainer` reads from the training parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GisParams {
    /// `Iterations`, default 100.
    pub iterations: i32,
    /// `Threads`, default 1: how the events are chunked (see the module
    /// documentation); every chunk is computed here on one thread.
    pub threads: i32,
    pub smoothing: Smoothing,
}

impl Default for GisParams {
    fn default() -> GisParams {
        GisParams { iterations: DEFAULT_ITERATIONS, threads: 1, smoothing: Smoothing::None }
    }
}

impl GisParams {
    /// `GISTrainer.init` and `doTrain`: `Iterations`, `Threads`,
    /// `Smoothing` with `SmoothingObservation`, `GaussianSmoothing` with
    /// `GaussianSmoothingSigma`. Both smoothings at once is an error, as is
    /// a thread count below one.
    pub fn from_training_parameters(params: &TrainingParameters) -> Result<GisParams> {
        let simple = params.get_bool(Key::SMOOTHING)?.unwrap_or(false);
        let gaussian = params.get_bool(Key::GAUSSIAN_SMOOTHING)?.unwrap_or(false);
        let smoothing = match (simple, gaussian) {
            (true, true) => return Err(Error::BothSmoothings),
            (true, false) => Smoothing::Simple {
                observation: params.get_double(Key::SMOOTHING_OBSERVATION)?.unwrap_or(DEFAULT_SMOOTHING_OBSERVATION),
            },
            (false, true) => Smoothing::Gaussian {
                sigma: params.get_double(Key::GAUSSIAN_SMOOTHING_SIGMA)?.unwrap_or(DEFAULT_GAUSSIAN_SIGMA),
            },
            (false, false) => Smoothing::None,
        };
        let threads = params.get_int(Key::THREADS)?.unwrap_or(1);
        if threads <= 0 {
            return Err(Error::Threads(threads));
        }
        Ok(GisParams { iterations: params.get_int(Key::ITERATIONS)?.unwrap_or(DEFAULT_ITERATIONS), threads, smoothing })
    }
}

/// A trained GIS model: `GISModel`'s data structures. Predicate `p` has
/// the outcome pattern `outcomes[pattern_starts[p]..pattern_starts[p+1]]`
/// and one parameter per pattern entry at the same positions of
/// `parameters`.
#[derive(Debug, Clone, PartialEq)]
pub struct GisModel {
    outcome_labels: Vec<JavaString>,
    pred_labels: Vec<JavaString>,
    pattern_starts: Vec<u32>,
    outcomes: Vec<u32>,
    parameters: Vec<f64>,
}

impl GisModel {
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

    /// The outcome ids predicate `p` has parameters for (`Context.getOutcomes`).
    pub fn pattern(&self, p: usize) -> &[u32] {
        &self.outcomes[self.pattern_starts[p] as usize..self.pattern_starts[p + 1] as usize]
    }

    /// Predicate `p`'s parameters, one per entry of its pattern
    /// (`Context.getParameters`).
    pub fn parameters(&self, p: usize) -> &[f64] {
        &self.parameters[self.pattern_starts[p] as usize..self.pattern_starts[p + 1] as usize]
    }

    /// Every parameter, predicates in order, each predicate's in pattern
    /// order.
    pub fn all_parameters(&self) -> &[f64] {
        &self.parameters
    }
}

/// What `AbstractEventTrainer.train(ObjectStream<Event>)` adds to the
/// report map that becomes manifest entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrainingReport {
    /// `Training-Eventhash`: the event checksum; absent when the trainer
    /// was given an index rather than events.
    pub event_hash: Option<u32>,
    /// `TrainerType`.
    pub trainer_type: TrainerType,
}

impl TrainingReport {
    /// The manifest entries, as `(key, value)` in the order OpenNLP adds them.
    pub fn manifest_entries(&self) -> Vec<(&'static str, String)> {
        let mut entries = Vec::new();
        if let Some(hash) = self.event_hash {
            entries.push(("Training-Eventhash", hash.to_string()));
        }
        entries.push((Key::TRAINER_TYPE, self.trainer_type.value().to_owned()));
        entries
    }
}

/// `TrainerFactory.getEventTrainer(...).train(events)` for `MAXENT`: the
/// checksum over the events, the indexer named by the parameters with
/// `sort` on and the default cutoff when none is given, then
/// [`train`]. `config` supplies the monitor and the stop criterion; a
/// missing one is the trainer's default.
pub fn train_events(
    events: &[Event],
    params: &TrainingParameters,
    config: &mut TrainingConfiguration,
) -> Result<(GisModel, TrainingReport)> {
    let mut checksum = EventChecksum::new();
    for e in events {
        checksum.update(e);
    }
    // AbstractTrainer.validate, before the indexer runs: Cutoff and
    // Iterations must parse (the indexer reads Cutoff itself).
    params.get_int(Key::ITERATIONS)?;
    let mut indexing = params.clone();
    indexing.put(Key::SORT, true);
    if indexing.get_int(Key::CUTOFF)?.unwrap_or(-1) == -1 {
        indexing.put(Key::CUTOFF, DEFAULT_CUTOFF);
    }
    let index = DataIndex::index(events, &IndexParams::from_training_parameters(&indexing)?)?;
    let model = train(&index, params, config)?;
    Ok((model, TrainingReport { event_hash: Some(checksum.value()), trainer_type: TrainerType::Event }))
}

/// `AbstractEventTrainer.train(DataIndexer)` for `MAXENT`: refuses an
/// index with one outcome, then trains with [`GisParams`] read from the
/// parameters and the stop criterion's threshold from them too.
pub fn train(index: &DataIndex, params: &TrainingParameters, config: &mut TrainingConfiguration) -> Result<GisModel> {
    if index.outcome_labels().len() <= 1 {
        return Err(Error::InsufficientTrainingData(ONE_OUTCOME));
    }
    let gis = GisParams::from_training_parameters(params)?;
    let default_stop = LogLikelihoodThresholdBreached::from_parameters(params)?;
    train_model(index, &gis, config, &default_stop, &mut |_, _| {})
}

/// `GISTrainer.trainModel(iterations, indexer, threads)`: trains without
/// the outcome check, with `default_stop` where `config` has no criterion
/// and the default monitor where it has none. After every iteration,
/// `observe` is given the iteration number and every parameter (the layout
/// of [`GisModel::all_parameters`]). A thread count under one is refused
/// as `trainModel` refuses it (`IllegalArgumentException`,
/// [`Error::Threads`]).
pub fn train_model(
    index: &DataIndex,
    params: &GisParams,
    config: &mut TrainingConfiguration,
    default_stop: &dyn StopCriteria,
    observe: &mut dyn FnMut(i32, &[f64]),
) -> Result<GisModel> {
    if params.threads <= 0 {
        return Err(Error::Threads(params.threads));
    }
    // "Incorporating indexed data": the first read of values[ei][ci].
    index.check_values()?;
    let mut default_monitor = DefaultProgressMonitor::new();
    let monitor: &mut dyn ProgressMonitor = match &mut config.monitor {
        Some(m) => m.as_mut(),
        None => &mut default_monitor,
    };
    let stop: &dyn StopCriteria = match &config.stop {
        Some(s) => s.as_ref(),
        None => default_stop,
    };
    let mut trainer = Trainer::new(index, params);
    trainer.find_parameters(params.iterations, monitor, stop, observe);
    Ok(GisModel {
        outcome_labels: index.outcome_labels().to_vec(),
        pred_labels: index.pred_labels().to_vec(),
        pattern_starts: trainer.pattern_starts,
        outcomes: trainer.outcomes,
        parameters: trainer.params,
    })
}

/// The trainer's state between iterations: the per predicate arrays
/// `params`, `observedExpects` and `modelExpects` share one layout, that
/// of [`GisModel`].
struct Trainer<'i> {
    index: &'i DataIndex,
    num_outcomes: usize,
    threads: usize,
    smoothing: Smoothing,
    correction_constant: f64,
    pattern_starts: Vec<u32>,
    outcomes: Vec<u32>,
    params: Vec<f64>,
    observed: Vec<f64>,
    /// One array per thread, each as long as `params`.
    expects: Vec<Vec<f64>>,
    /// `UniformPrior`'s log prior: `StrictMath.log(1.0 / numOutcomes)`.
    log_prior: f64,
}

/// What `ModelExpectationComputeTask` returns for its chunk of events: the
/// chunk's log likelihood, the events it counted (each as often as it was
/// seen) and how many of those the model's best outcome got right.
struct Chunk {
    log_likelihood: f64,
    events: i32,
    correct: i32,
}

impl<'i> Trainer<'i> {
    fn new(index: &'i DataIndex, params: &GisParams) -> Trainer<'i> {
        let num_outcomes = index.outcome_labels().len();
        let num_preds = index.pred_labels().len();
        let n = index.num_unique_events();

        // The correction constant: the longest context, or for real valued
        // events the largest float sum of an event's values.
        let mut correction_constant = 0.0f64;
        for ei in 0..n {
            match index.values(ei) {
                None => {
                    let len = index.context(ei).len() as f64;
                    if len > correction_constant {
                        correction_constant = len;
                    }
                }
                Some(v) => {
                    let mut cl: f32 = v[0];
                    for &x in &v[1..] {
                        cl += x;
                    }
                    if f64::from(cl) > correction_constant {
                        correction_constant = f64::from(cl);
                    }
                }
            }
        }

        // predCount[pred][outcome], a float.
        let mut pred_count = vec![0.0f32; num_preds * num_outcomes];
        for ei in 0..n {
            let outcome = index.outcome(ei) as usize;
            let seen = index.seen(ei) as f32;
            let values = index.values(ei);
            for (j, &p) in index.context(ei).iter().enumerate() {
                let cell = &mut pred_count[p as usize * num_outcomes + outcome];
                match values {
                    Some(v) => *cell += seen * v[j],
                    None => *cell += seen,
                }
            }
        }

        // The outcome patterns, and the observed expectations.
        let mut pattern_starts = Vec::with_capacity(num_preds + 1);
        let mut outcomes = Vec::new();
        let mut observed = Vec::new();
        for p in 0..num_preds {
            pattern_starts.push(outcomes.len() as u32);
            let counts = &pred_count[p * num_outcomes..(p + 1) * num_outcomes];
            for (oi, &count) in counts.iter().enumerate() {
                let active = match params.smoothing {
                    Smoothing::Simple { .. } => true,
                    _ => count > 0.0,
                };
                if !active {
                    continue;
                }
                outcomes.push(oi as u32);
                observed.push(if count > 0.0 {
                    f64::from(count)
                } else if let Smoothing::Simple { observation } = params.smoothing {
                    observation
                } else {
                    0.0
                });
            }
        }
        pattern_starts.push(outcomes.len() as u32);
        // Checked by train_model, as trainModel checks it.
        let threads = params.threads as usize;
        Trainer {
            index,
            num_outcomes,
            threads,
            smoothing: params.smoothing,
            correction_constant,
            params: vec![0.0; outcomes.len()],
            expects: vec![vec![0.0; outcomes.len()]; threads],
            pattern_starts,
            outcomes,
            observed,
            log_prior: strict_math::log(1.0 / num_outcomes as f64),
        }
    }

    fn pattern_range(&self, p: u32) -> std::ops::Range<usize> {
        self.pattern_starts[p as usize] as usize..self.pattern_starts[p as usize + 1] as usize
    }

    /// `findParameters`.
    fn find_parameters(
        &mut self,
        iterations: i32,
        monitor: &mut dyn ProgressMonitor,
        stop: &dyn StopCriteria,
        observe: &mut dyn FnMut(i32, &[f64]),
    ) {
        let mut prev_ll = 0.0;
        for i in 1..=iterations {
            let curr_ll = self.next_iteration(i, monitor);
            observe(i, &self.params);
            if i > 1 {
                if prev_ll > curr_ll {
                    // OpenNLP logs "Model Diverging: loglikelihood decreased".
                    break;
                }
                if stop.test(curr_ll - prev_ll) {
                    monitor.finished_training(iterations, Some(stop));
                    break;
                }
            }
            prev_ll = curr_ll;
        }
        if !monitor.is_training_finished() {
            monitor.finished_training(iterations, None);
        }
        monitor.display(true);
    }

    /// `ModelExpectationComputeTask.call` over the events `start..start + len`
    /// into thread `t`'s expectations.
    fn chunk(&mut self, t: usize, start: usize, len: usize, dist: &mut [f64]) -> Chunk {
        let index = self.index;
        let mut out = Chunk { log_likelihood: 0.0, events: 0, correct: 0 };
        for ei in start..start + len {
            let context = index.context(ei);
            let values = index.values(ei);
            // UniformPrior.logPrior, then GISModel.eval: sumFeatures and the
            // normalized exponentials.
            dist.fill(self.log_prior);
            for (ci, &p) in context.iter().enumerate() {
                let value = values.map_or(1.0, |v| f64::from(v[ci]));
                let range = self.pattern_range(p);
                for (oid, param) in self.outcomes[range.clone()].iter().zip(&self.params[range]) {
                    dist[*oid as usize] += param * value;
                }
            }
            let mut normal = 0.0;
            for d in dist.iter_mut() {
                *d = strict_math::exp(*d);
                normal += *d;
            }
            for d in dist.iter_mut() {
                *d /= normal;
            }
            let seen = index.seen(ei);
            let seen_f = f64::from(seen);
            for (j, &p) in context.iter().enumerate() {
                let range = self.pattern_range(p);
                let expects = &mut self.expects[t][range.clone()];
                for (e, &oi) in expects.iter_mut().zip(&self.outcomes[range]) {
                    *e += match values {
                        Some(v) => dist[oi as usize] * f64::from(v[j]) * seen_f,
                        None => dist[oi as usize] * seen_f,
                    };
                }
            }
            let outcome = index.outcome(ei) as usize;
            out.log_likelihood += strict_math::log(dist[outcome]) * seen_f;
            out.events += seen as i32;
            // ArrayMath.argmax: the first of the largest.
            let mut max = 0;
            for (i, &d) in dist.iter().enumerate().skip(1) {
                if dist[max] < d {
                    max = i;
                }
            }
            if max == outcome {
                out.correct += seen as i32;
            }
        }
        out
    }

    /// `nextIteration`: the expectations under the current parameters,
    /// then the parameter update; returns the log likelihood.
    fn next_iteration(&mut self, iteration: i32, monitor: &mut dyn ProgressMonitor) -> f64 {
        let n = self.index.num_unique_events();
        let task_size = n / self.threads;
        let left_over = n % self.threads;
        let mut dist = vec![0.0; self.num_outcomes];
        let mut log_likelihood = 0.0;
        let mut events = 0;
        let mut correct = 0;
        for t in 0..self.threads {
            let (start, len) =
                if t < left_over { (t * task_size + t, task_size + 1) } else { (t * task_size + left_over, task_size) };
            let chunk = self.chunk(t, start, len, &mut dist);
            events += chunk.events;
            correct += chunk.correct;
            log_likelihood += chunk.log_likelihood;
        }
        // The other threads' expectations are added into the first's.
        for t in 1..self.threads {
            for i in 0..self.params.len() {
                self.expects[0][i] += self.expects[t][i];
            }
        }
        for i in 0..self.params.len() {
            let update = match self.smoothing {
                Smoothing::Gaussian { sigma } => self.gaussian_update(i, sigma),
                // OpenNLP logs "Model expects == 0" when the model expectation
                // is zero and divides anyway.
                _ => {
                    (strict_math::log(self.observed[i]) - strict_math::log(self.expects[0][i]))
                        / self.correction_constant
                }
            };
            self.params[i] += update;
            for expects in &mut self.expects {
                expects[i] = 0.0;
            }
        }
        monitor.finished_iteration(iteration, correct, events, TrainingMeasure::LogLikelihood, log_likelihood);
        log_likelihood
    }

    /// `gaussianUpdate`: Newton's method on the penalized update, at most
    /// fifty steps.
    fn gaussian_update(&self, i: usize, sigma: f64) -> f64 {
        let param = self.params[i];
        let model_value = self.expects[0][i];
        let observed_value = self.observed[i];
        let mut x0 = 0.0;
        for _ in 0..50 {
            let tmp = model_value * strict_math::exp(self.correction_constant * x0);
            let f = tmp + (param + x0) / sigma - observed_value;
            let fp = tmp * self.correction_constant + 1.0 / sigma;
            if fp == 0.0 {
                break;
            }
            let x = x0 - f / fp;
            if (x - x0).abs() < 0.000001 {
                x0 = x;
                break;
            }
            x0 = x;
        }
        x0
    }
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

    #[test]
    fn params_come_from_the_training_parameters() {
        let mut p = TrainingParameters::new();
        assert_eq!(GisParams::from_training_parameters(&p).unwrap(), GisParams::default());
        p.put(Key::ITERATIONS, 7);
        p.put(Key::THREADS, "3");
        p.put(Key::SMOOTHING, true);
        let g = GisParams::from_training_parameters(&p).unwrap();
        assert_eq!(g, GisParams { iterations: 7, threads: 3, smoothing: Smoothing::Simple { observation: 0.1 } });
        p.put(Key::GAUSSIAN_SMOOTHING, "true");
        assert_eq!(GisParams::from_training_parameters(&p), Err(Error::BothSmoothings));
        p.put(Key::SMOOTHING, false);
        p.put(Key::GAUSSIAN_SMOOTHING_SIGMA, 0.5);
        assert_eq!(GisParams::from_training_parameters(&p).unwrap().smoothing, Smoothing::Gaussian { sigma: 0.5 });
        p.put(Key::THREADS, 0);
        assert_eq!(GisParams::from_training_parameters(&p), Err(Error::Threads(0)));
    }

    #[test]
    fn one_outcome_is_refused_before_training() {
        let events = [event("s", "a"), event("s", "b")];
        let index = DataIndex::index(&events, &IndexParams::new(IndexerKind::OnePass, 0, true)).unwrap();
        let mut config = TrainingConfiguration::default();
        assert_eq!(
            train(&index, &TrainingParameters::new(), &mut config).err(),
            Some(Error::InsufficientTrainingData(ONE_OUTCOME))
        );
    }

    #[test]
    fn iterations_must_parse_before_the_events_are_looked_at() {
        // AbstractTrainer.validate runs before the indexer: with no events
        // and Iterations that does not parse, the JVM reports Iterations.
        let mut p = TrainingParameters::new();
        p.put(Key::ITERATIONS, "many");
        let mut config = TrainingConfiguration::default();
        assert!(matches!(train_events(&[], &p, &mut config), Err(Error::ParameterFormat { .. })));
        // Threads is read in doTrain, after the outcome check.
        let events = [event("s", "a"), event("s", "b")];
        let mut p = TrainingParameters::new();
        p.put(Key::THREADS, 0);
        p.put(Key::CUTOFF, 0);
        assert_eq!(train_events(&events, &p, &mut config).err(), Some(Error::InsufficientTrainingData(ONE_OUTCOME)));
    }

    #[test]
    fn an_event_without_values_in_a_real_value_index_is_refused() {
        // GISTrainer reads values[ei][ci] while incorporating the index and
        // throws NullPointerException on the event without values.
        let events = [
            Event::with_values("yes", vec![JavaString::new("a")], Some(vec![2.0])).unwrap(),
            Event::with_values("no", vec![JavaString::new("b")], None).unwrap(),
        ];
        let index = DataIndex::index(&events, &IndexParams::new(IndexerKind::OnePassRealValue, 0, true)).unwrap();
        let mut config = TrainingConfiguration::default();
        assert_eq!(
            train(&index, &TrainingParameters::new(), &mut config).err(),
            Some(Error::MissingValues { event: 1 })
        );
    }

    #[test]
    fn training_fits_the_events_and_reports_every_iteration() {
        let mut p = TrainingParameters::defaults();
        p.put(Key::CUTOFF, 0);
        p.put(Key::ITERATIONS, 30);
        let mut config = TrainingConfiguration::new(Some(Box::new(DefaultProgressMonitor::new())), None);
        let (model, report) = train_events(&events(), &p, &mut config).unwrap();
        assert_eq!(model.outcome_labels(), ["yes", "no", "maybe"]);
        assert_eq!(model.num_predicates(), 8);
        // "cap" was seen with "yes" and "maybe" only.
        let cap = model.pred_labels().iter().position(|l| *l == "cap").unwrap();
        assert_eq!(model.pattern(cap), [0, 2]);
        assert_eq!(model.parameters(cap).len(), 2);
        assert_eq!(report.event_hash, Some(crate::event::checksum(&events())));
        assert_eq!(report.manifest_entries()[1], ("TrainerType", "Event".to_string()));
        let monitor = config.monitor.unwrap();
        // The default monitor keeps its lines only until display(true), which
        // the trainer calls; it is a trait object here, so look at the
        // parameters instead: thirty iterations moved every one of them.
        assert!(monitor.is_training_finished());
        assert!(model.all_parameters().iter().all(|&x| x != 0.0));
    }

    #[test]
    fn the_observer_sees_the_parameters_after_each_iteration() {
        let index = DataIndex::index(&events(), &IndexParams::new(IndexerKind::OnePass, 0, true)).unwrap();
        let mut seen = Vec::new();
        let params = GisParams { iterations: 5, ..GisParams::default() };
        let stop = LogLikelihoodThresholdBreached::default();
        let model = train_model(&index, &params, &mut TrainingConfiguration::default(), &stop, &mut |i, p| {
            seen.push((i, p.to_vec()));
        })
        .unwrap();
        assert_eq!(seen.iter().map(|(i, _)| *i).collect::<Vec<_>>(), [1, 2, 3, 4, 5]);
        assert_eq!(seen[4].1, model.all_parameters());
        // A run of two iterations is the first two of a run of five.
        let params = GisParams { iterations: 2, ..GisParams::default() };
        let two = train_model(&index, &params, &mut TrainingConfiguration::default(), &stop, &mut |_, _| {}).unwrap();
        assert_eq!(two.all_parameters(), &seen[1].1[..]);
    }

    #[test]
    fn a_thread_count_under_one_is_refused() {
        let index = DataIndex::index(&events(), &IndexParams::new(IndexerKind::OnePass, 0, true)).unwrap();
        let stop = LogLikelihoodThresholdBreached::default();
        for threads in [0, -3] {
            let params = GisParams { threads, ..GisParams::default() };
            let result = train_model(&index, &params, &mut TrainingConfiguration::default(), &stop, &mut |_, _| {});
            assert_eq!(result.err(), Some(Error::Threads(threads)));
        }
    }

    #[test]
    fn chunking_by_threads_changes_only_the_order_of_sums() {
        let index = DataIndex::index(&events(), &IndexParams::new(IndexerKind::OnePass, 0, true)).unwrap();
        let stop = LogLikelihoodThresholdBreached::default();
        let one =
            train_model(&index, &GisParams::default(), &mut TrainingConfiguration::default(), &stop, &mut |_, _| {})
                .unwrap();
        let params = GisParams { threads: 3, ..GisParams::default() };
        let three = train_model(&index, &params, &mut TrainingConfiguration::default(), &stop, &mut |_, _| {}).unwrap();
        for (a, b) in one.all_parameters().iter().zip(three.all_parameters()) {
            assert!((a - b).abs() < 1e-9, "{a} vs {b}");
        }
    }

    #[test]
    fn simple_smoothing_gives_every_predicate_every_outcome() {
        let index = DataIndex::index(&events(), &IndexParams::new(IndexerKind::OnePass, 0, true)).unwrap();
        let params =
            GisParams { iterations: 3, smoothing: Smoothing::Simple { observation: 0.1 }, ..GisParams::default() };
        let stop = LogLikelihoodThresholdBreached::default();
        let model = train_model(&index, &params, &mut TrainingConfiguration::default(), &stop, &mut |_, _| {}).unwrap();
        for p in 0..model.num_predicates() {
            assert_eq!(model.pattern(p), [0, 1, 2]);
        }
    }
}
