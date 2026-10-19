//! `opennlp.tools.ml.maxent.quasinewton`: the L-BFGS maxent trainer behind
//! `Algorithm=MAXENT_QN`, computed as the JVM computes it so that every
//! parameter agrees bit for bit after every iteration.
//!
//! `QNTrainer` hands the negative log likelihood of the indexed events
//! (`NegLogLikelihood`, or `ParallelNegLogLikelihood` with `Threads` above
//! one) to `QNMinimizer`, which wraps it in the L2 penalty, keeps the last
//! `NumOfUpdates` corrections for the two loop recursion, and takes a
//! backtracking line search step per iteration (`LineSearch`; the
//! constrained one of Andrew and Gao when `L1Cost` is above zero). It stops
//! when the function's change rate or the relative gradient norm falls
//! under `1e-4`, the step under `1e-10`, the function evaluations over
//! `MaxFctEval`, or after `Iterations` iterations. When both costs are on
//! the result is rescaled by `sqrt(1 + L2Cost)` at the end.
//!
//! What the port keeps from the Java code, because the numbers depend on
//! it:
//!
//! - Every sum in the order the loops add it: an event's sums over its
//!   context in context order, the log likelihood and the gradient over
//!   the events in index order, and with `Threads` above one the events
//!   cut into the JVM's chunks (`n / threads` each, the last one taking the
//!   rest) with each chunk's sums added in chunk order. Unlike `GISTrainer`,
//!   the JVM adds the chunks in a fixed order here, so a run with several
//!   threads is reproducible there too.
//! - `StrictMath.exp` and `log` (`turbo_jvm::strict_math`) in the log sum
//!   of exponentials; `sqrt` is IEEE 754's in both.
//! - A value from an `OnePassRealValue` index is read at the context
//!   position, as the JVM reads it (see [`crate::index`]).
//! - `UpdateInfo` discards the oldest correction by shifting array
//!   references, so after the first discard the newest and the second
//!   newest slots are one array and every write to the newest overwrites
//!   the other; with each further discard one more slot joins. The two
//!   loop recursion then works on copies of the latest correction with the
//!   older `rho` values. The aliasing is reproduced with slot indices.
//! - `QNTrainer` tells its progress monitor nothing and reads no stop
//!   criterion, so a `TrainingConfiguration` has no effect on it; the
//!   observer of [`train_model`] is how a caller sees the iterations.
//!
//! The Armijo loop halves the step until the condition holds. With a NaN
//! objective it never holds and the JVM does not terminate; here the loop
//! stops with [`Error::LineSearch`] once the step has reached zero.

use crate::error::{Error, ONE_OUTCOME, Result};
use crate::event::{Event, EventChecksum};
use crate::gis::TrainingReport;
use crate::index::{DataIndex, IndexParams};
use crate::java_string::JavaString;
use crate::params::{DEFAULT_CUTOFF, DEFAULT_ITERATIONS, Key, TrainerType, TrainingParameters};
use turbo_jvm::strict_math;

mod minimizer;
mod objective;
#[cfg(test)]
mod tests;

use minimizer::Minimizer;
use objective::NegLogLikelihood;

/// `QNTrainer.L1COST_DEFAULT` and `L2COST_DEFAULT`.
pub const DEFAULT_COST: f64 = 0.1;
/// `QNTrainer.M_DEFAULT`: corrections kept for the two loop recursion.
pub const DEFAULT_UPDATES: i32 = 15;
/// `QNTrainer.MAX_FCT_EVAL_DEFAULT`.
pub const DEFAULT_MAX_FCT_EVAL: i32 = 30000;
/// `QNMinimizer.CONVERGE_TOLERANCE`: the function change rate under which
/// training stops.
pub const CONVERGE_TOLERANCE: f64 = 1e-4;
/// `QNMinimizer.REL_GRAD_NORM_TOL`: the relative gradient norm under which
/// training stops.
pub const REL_GRAD_NORM_TOL: f64 = 1e-4;
/// `QNMinimizer.MIN_STEP_SIZE`.
pub const MIN_STEP_SIZE: f64 = 1e-10;
/// `LineSearch.C`: the Armijo constant.
const ARMIJO: f64 = 0.0001;
/// `LineSearch.RHO`: how the step shrinks.
const SHRINK: f64 = 0.5;

/// What `QNTrainer` reads from the training parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QnParams {
    /// `Iterations`, default 100.
    pub iterations: i32,
    /// `Threads`, default 1: how the events are chunked (see the module
    /// documentation); every chunk is computed here on one thread.
    pub threads: i32,
    /// `L1Cost`, default 0.1.
    pub l1_cost: f64,
    /// `L2Cost`, default 0.1.
    pub l2_cost: f64,
    /// `NumOfUpdates`, default 15.
    pub updates: i32,
    /// `MaxFctEval`, default 30000.
    pub max_fct_eval: i32,
}

impl Default for QnParams {
    fn default() -> QnParams {
        QnParams {
            iterations: DEFAULT_ITERATIONS,
            threads: 1,
            l1_cost: DEFAULT_COST,
            l2_cost: DEFAULT_COST,
            updates: DEFAULT_UPDATES,
            max_fct_eval: DEFAULT_MAX_FCT_EVAL,
        }
    }
}

impl QnParams {
    /// `QNTrainer.init` and `validate`: the parameters, refused as the JVM
    /// refuses them (`IllegalArgumentException`) when `NumOfUpdates` or
    /// `MaxFctEval` is not positive, `Threads` is under one, or a cost is
    /// negative. `Iterations` is checked by [`train_model`], as
    /// `QNMinimizer`'s constructor checks it.
    pub fn from_training_parameters(params: &TrainingParameters) -> Result<QnParams> {
        let qn = QnParams {
            iterations: params.get_int(Key::ITERATIONS)?.unwrap_or(DEFAULT_ITERATIONS),
            threads: params.get_int(Key::THREADS)?.unwrap_or(1),
            l1_cost: params.get_double(Key::L1_COST)?.unwrap_or(DEFAULT_COST),
            l2_cost: params.get_double(Key::L2_COST)?.unwrap_or(DEFAULT_COST),
            updates: params.get_int(Key::NUM_OF_UPDATES)?.unwrap_or(DEFAULT_UPDATES),
            max_fct_eval: params.get_int(Key::MAX_FCT_EVAL)?.unwrap_or(DEFAULT_MAX_FCT_EVAL),
        };
        qn.validate()?;
        Ok(qn)
    }

    /// `QNTrainer.validate`, in its order.
    fn validate(&self) -> Result<()> {
        if self.updates <= 0 {
            return Err(Error::NotPositive { key: Key::NUM_OF_UPDATES, value: self.updates });
        }
        if self.max_fct_eval <= 0 {
            return Err(Error::NotPositive { key: Key::MAX_FCT_EVAL, value: self.max_fct_eval });
        }
        if self.threads < 1 {
            return Err(Error::NotPositive { key: Key::THREADS, value: self.threads });
        }
        if self.l1_cost < 0.0 {
            return Err(Error::NegativeCost { key: Key::L1_COST, value: self.l1_cost });
        }
        if self.l2_cost < 0.0 {
            return Err(Error::NegativeCost { key: Key::L2_COST, value: self.l2_cost });
        }
        Ok(())
    }

    /// Whether the minimizer runs with the L1 penalty: the pseudo gradient
    /// and the constrained line search.
    fn l1(&self) -> bool {
        self.l1_cost > 0.0
    }
}

/// Why the minimizer stopped: `QNMinimizer.isConverged`'s four tests, or
/// the iteration limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QnStop {
    /// The function change rate fell under [`CONVERGE_TOLERANCE`].
    ChangeRate,
    /// The gradient's L2 norm over `max(1, |x|)` fell under
    /// [`REL_GRAD_NORM_TOL`].
    GradientNorm,
    /// The step fell under [`MIN_STEP_SIZE`].
    StepSize,
    /// More function evaluations than `MaxFctEval`.
    FunctionEvaluations,
    /// `Iterations` iterations were run.
    Iterations,
}

impl QnStop {
    /// A short name: the `stop` the training goldens record.
    pub fn name(self) -> &'static str {
        match self {
            QnStop::ChangeRate => "change-rate",
            QnStop::GradientNorm => "gradient-norm",
            QnStop::StepSize => "step-size",
            QnStop::FunctionEvaluations => "fct-eval",
            QnStop::Iterations => "iterations",
        }
    }
}

/// One iteration of the minimizer as its debug log reports it: the state
/// of the line search after the step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QnIteration {
    /// 1-based.
    pub iteration: i32,
    /// The objective at the new point: the negative log likelihood with
    /// the L2 term, and the L1 term when L1 is on.
    pub value: f64,
    /// `(previous value - value) / previous value`.
    pub change_rate: f64,
    /// The step the line search took.
    pub step_size: f64,
    /// Function evaluations so far, over the whole run.
    pub fct_evals: i32,
    /// The test that stops training after this iteration, when one does.
    pub stop: Option<QnStop>,
}

/// A trained QN model: `QNModel`'s data structures. Every predicate has a
/// parameter for every outcome, in outcome order.
#[derive(Debug, Clone, PartialEq)]
pub struct QnModel {
    outcome_labels: Vec<JavaString>,
    pred_labels: Vec<JavaString>,
    /// `0..num_outcomes`, the pattern of every predicate.
    pattern: Vec<u32>,
    /// Predicate `p`'s parameters are `parameters[p * n..(p + 1) * n]` for
    /// `n` outcomes.
    parameters: Vec<f64>,
}

impl QnModel {
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
}

/// `TrainerFactory.getEventTrainer(...).train(events)` for `MAXENT_QN`: the
/// checksum over the events, the indexer named by the parameters with
/// `sort` on and the default cutoff when none is given, then [`train`].
pub fn train_events(events: &[Event], params: &TrainingParameters) -> Result<(QnModel, TrainingReport)> {
    // QNTrainer.init reads its parameters and validate checks them before
    // the indexer runs, so a bad parameter is refused whatever the events.
    QnParams::from_training_parameters(params)?;
    let mut checksum = EventChecksum::new();
    for e in events {
        checksum.update(e);
    }
    let mut indexing = params.clone();
    indexing.put(Key::SORT, true);
    if indexing.get_int(Key::CUTOFF)?.unwrap_or(-1) == -1 {
        indexing.put(Key::CUTOFF, DEFAULT_CUTOFF);
    }
    let index = DataIndex::index(events, &IndexParams::from_training_parameters(&indexing)?)?;
    let model = train(&index, params)?;
    Ok((model, TrainingReport { event_hash: Some(checksum.value()), trainer_type: TrainerType::Event }))
}

/// `AbstractEventTrainer.train(DataIndexer)` for `MAXENT_QN`: validates the
/// parameters, refuses an index with one outcome, then trains.
pub fn train(index: &DataIndex, params: &TrainingParameters) -> Result<QnModel> {
    let qn = QnParams::from_training_parameters(params)?;
    if index.outcome_labels().len() <= 1 {
        return Err(Error::InsufficientTrainingData(ONE_OUTCOME));
    }
    Ok(train_model(index, &qn, &mut |_, _| {})?.0)
}

/// `QNTrainer.trainModel(iterations, indexer)` without the outcome check:
/// the minimizer from the origin, then the model from the point it ends at.
/// After every iteration `observe` is given the iteration and the point
/// the minimizer is at, `x[outcome * predicates + predicate]`, before the
/// rescaling of the end; [`model_parameters`] turns it into the parameters
/// a run stopping there would produce. Returns the model and why the
/// minimizer stopped.
pub fn train_model(
    index: &DataIndex,
    params: &QnParams,
    observe: &mut dyn FnMut(&QnIteration, &[f64]),
) -> Result<(QnModel, QnStop)> {
    params.validate()?;
    if params.iterations <= 0 {
        return Err(Error::NotPositive { key: Key::ITERATIONS, value: params.iterations });
    }
    // NegLogLikelihood.valueAt reads values[ei][ci] at the first evaluation.
    index.check_values()?;
    let function = NegLogLikelihood::new(index, params);
    let mut minimizer = Minimizer::new(&function, params);
    let stop = minimizer.minimize(observe)?;
    let parameters =
        model_parameters(&minimizer.search.next_point, function.num_outcomes, function.num_features, params);
    let model = QnModel {
        outcome_labels: index.outcome_labels().to_vec(),
        pred_labels: index.pred_labels().to_vec(),
        pattern: (0..function.num_outcomes as u32).collect(),
        parameters,
    };
    Ok((model, stop))
}

/// The model parameters a run that stops at `point` produces: the end of
/// `QNMinimizer.minimize` (the point scaled by `sqrt(1 + L2Cost)` when both
/// costs are on) laid out as [`QnModel::all_parameters`], each predicate's
/// parameters per outcome.
pub fn model_parameters(point: &[f64], num_outcomes: usize, num_features: usize, params: &QnParams) -> Vec<f64> {
    assert_eq!(point.len(), num_outcomes * num_features);
    let scale = if params.l1() && params.l2_cost > 0.0 { Some((1.0 + params.l2_cost).sqrt()) } else { None };
    let mut out = Vec::with_capacity(point.len());
    for p in 0..num_features {
        for o in 0..num_outcomes {
            let x = point[o * num_features + p];
            out.push(match scale {
                Some(s) => s * x,
                None => x,
            });
        }
    }
    out
}

/// `QNTrainer.ModelEvaluator.evaluate`: the share of the index's events
/// (each counted as often as seen) whose most probable outcome under
/// `point` is their outcome. `point` is laid out as the observer of
/// [`train_model`] receives it.
pub fn training_accuracy(index: &DataIndex, point: &[f64]) -> f64 {
    let num_outcomes = index.outcome_labels().len();
    let num_features = index.pred_labels().len();
    assert_eq!(point.len(), num_outcomes * num_features);
    let mut probs = vec![0.0; num_outcomes];
    let mut correct: i32 = 0;
    let mut total: i32 = 0;
    for ei in 0..index.num_unique_events() {
        // QNModel.eval(int[], float[], double[], ...) on a fresh array.
        probs.fill(0.0);
        let values = index.values(ei);
        for (i, &p) in index.context(ei).iter().enumerate() {
            let value = values.map_or(1.0, |v| f64::from(v[i]));
            for (oi, prob) in probs.iter_mut().enumerate() {
                *prob += value * point[oi * num_features + p as usize];
            }
        }
        let lse = log_sum_of_exps(&probs);
        for prob in probs.iter_mut() {
            *prob = strict_math::exp(*prob - lse);
        }
        let seen = index.seen(ei) as i32;
        if argmax(&probs) == index.outcome(ei) as usize {
            correct += seen;
        }
        total += seen;
    }
    f64::from(correct) / f64::from(total)
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

/// `ArrayMath.logSumOfExps`: `max + log(sum(exp(x - max)))`, skipping
/// negative infinities.
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

/// `ArrayMath.innerProduct`.
fn inner_product(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut product = 0.0;
    for (x, y) in a.iter().zip(b) {
        product += x * y;
    }
    product
}

/// `ArrayMath.l1norm`.
fn l1_norm(v: &[f64]) -> f64 {
    let mut norm = 0.0;
    for x in v {
        norm += x.abs();
    }
    norm
}

/// `ArrayMath.l2norm`.
fn l2_norm(v: &[f64]) -> f64 {
    inner_product(v, v).sqrt()
}

/// `StrictMath.max(1, x)`: NaN when `x` is.
fn max_one(x: f64) -> f64 {
    if x.is_nan() {
        x
    } else if 1.0 >= x {
        1.0
    } else {
        x
    }
}
