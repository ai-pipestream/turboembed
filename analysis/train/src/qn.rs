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
    outcome_labels: Vec<String>,
    pred_labels: Vec<JavaString>,
    /// `0..num_outcomes`, the pattern of every predicate.
    pattern: Vec<u32>,
    /// Predicate `p`'s parameters are `parameters[p * n..(p + 1) * n]` for
    /// `n` outcomes.
    parameters: Vec<f64>,
}

impl QnModel {
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

/// `NegLogLikelihood` and `ParallelNegLogLikelihood`, with the L2 penalty
/// of `QNMinimizer.L2RegFunction` applied on top: the objective over the
/// indexed events at a point `x[outcome * features + feature]`.
struct NegLogLikelihood<'i> {
    index: &'i DataIndex,
    num_outcomes: usize,
    num_features: usize,
    dimension: usize,
    l2_cost: f64,
    /// The events of each thread's task: `(start, len)`.
    chunks: Vec<(usize, usize)>,
}

impl<'i> NegLogLikelihood<'i> {
    fn new(index: &'i DataIndex, params: &QnParams) -> NegLogLikelihood<'i> {
        let num_outcomes = index.outcome_labels().len();
        let num_features = index.pred_labels().len();
        let n = index.num_unique_events();
        let threads = params.threads.max(1) as usize;
        let task_size = n / threads;
        let left_over = n % threads;
        let chunks = (0..threads)
            .map(|t| (t * task_size, if t == threads - 1 { task_size + left_over } else { task_size }))
            .collect();
        NegLogLikelihood {
            index,
            num_outcomes,
            num_features,
            dimension: num_outcomes * num_features,
            l2_cost: params.l2_cost,
            chunks,
        }
    }

    #[inline]
    fn index_of(&self, outcome: usize, feature: u32) -> usize {
        outcome * self.num_features + feature as usize
    }

    /// Event `ei`'s sum per outcome under `x`, into `sums`.
    #[inline]
    fn sums(&self, ei: usize, x: &[f64], sums: &mut [f64]) {
        let context = self.index.context(ei);
        let values = self.index.values(ei);
        for (oi, sum) in sums.iter_mut().enumerate() {
            *sum = 0.0;
            for (ai, &p) in context.iter().enumerate() {
                let value = values.map_or(1.0, |v| f64::from(v[ai]));
                *sum += value * x[self.index_of(oi, p)];
            }
        }
    }

    /// `NegLLComputeTask.call`: the negative log likelihood of the events
    /// `start..start + len`.
    fn value_of_chunk(&self, x: &[f64], start: usize, len: usize, sums: &mut [f64]) -> f64 {
        let mut value = 0.0;
        for ei in start..start + len {
            self.sums(ei, x, sums);
            let lse = log_sum_of_exps(sums);
            let outcome = self.index.outcome(ei) as usize;
            value -= (sums[outcome] - lse) * f64::from(self.index.seen(ei));
        }
        value
    }

    /// `L2RegFunction.valueAt`: the chunks' values added in order, then
    /// the L2 term.
    fn value_at(&self, x: &[f64], sums: &mut [f64]) -> f64 {
        let mut value = 0.0;
        if self.chunks.len() == 1 {
            value = self.value_of_chunk(x, 0, self.chunks[0].1, sums);
        } else {
            for &(start, len) in &self.chunks {
                value += self.value_of_chunk(x, start, len, sums);
            }
        }
        if self.l2_cost > 0.0 {
            value += self.l2_cost * inner_product(x, x);
        }
        value
    }

    /// `GradientComputeTask.call`: the gradient over the events
    /// `start..start + len` added into `gradient`.
    fn gradient_of_chunk(&self, x: &[f64], start: usize, len: usize, expectation: &mut [f64], gradient: &mut [f64]) {
        for ei in start..start + len {
            self.sums(ei, x, expectation);
            let lse = log_sum_of_exps(expectation);
            for e in expectation.iter_mut() {
                *e = strict_math::exp(*e - lse);
            }
            let context = self.index.context(ei);
            let values = self.index.values(ei);
            let outcome = self.index.outcome(ei) as usize;
            let seen = f64::from(self.index.seen(ei));
            for (oi, &e) in expectation.iter().enumerate() {
                let empirical = if outcome == oi { 1.0 } else { 0.0 };
                for (ai, &p) in context.iter().enumerate() {
                    let value = values.map_or(1.0, |v| f64::from(v[ai]));
                    gradient[self.index_of(oi, p)] += value * (e - empirical) * seen;
                }
            }
        }
    }

    /// `L2RegFunction.gradientAt` into `gradient`: with one thread the
    /// events accumulate into it directly; with more, each chunk into its
    /// own zeroed array (`chunk_grad`, reused) and the arrays are added in
    /// chunk order, `((0 + g0) + g1) + ...` for every element, which is
    /// `ParallelNegLogLikelihood`'s sum over its per thread arrays.
    fn gradient_at(&self, x: &[f64], expectation: &mut [f64], gradient: &mut [f64], chunk_grad: &mut [f64]) {
        if self.chunks.len() == 1 {
            gradient.fill(0.0);
            self.gradient_of_chunk(x, 0, self.chunks[0].1, expectation, gradient);
        } else {
            gradient.fill(0.0);
            for &(start, len) in &self.chunks {
                chunk_grad.fill(0.0);
                self.gradient_of_chunk(x, start, len, expectation, chunk_grad);
                for (g, &c) in gradient.iter_mut().zip(chunk_grad.iter()) {
                    *g += c;
                }
            }
        }
        if self.l2_cost > 0.0 {
            for (g, &xi) in gradient.iter_mut().zip(x) {
                *g += 2.0 * self.l2_cost * xi;
            }
        }
    }
}

/// `LineSearchResult`: the two ends of the latest step.
struct Search {
    step_size: f64,
    value_at_curr: f64,
    value_at_next: f64,
    grad_at_curr: Vec<f64>,
    grad_at_next: Vec<f64>,
    /// With L1: the pseudo gradient at `next_point`, recomputed in place
    /// after each step.
    pseudo_grad: Vec<f64>,
    curr_point: Vec<f64>,
    next_point: Vec<f64>,
    /// With L1: the orthant of the step.
    sign: Vec<f64>,
    fct_eval_count: i32,
}

impl Search {
    /// `getFuncChangeRate`.
    fn change_rate(&self) -> f64 {
        (self.value_at_curr - self.value_at_next) / self.value_at_curr
    }
}

/// `QNMinimizer.UpdateInfo`: the last `m` corrections `S` (points) and `Y`
/// (gradients) with `rho = 1 / (s . y)`. The JVM shifts the array
/// references when it discards the oldest, which aliases slots (see the
/// module documentation); `slot[i]` is which array position `i` refers to.
struct UpdateInfo {
    m: usize,
    k: usize,
    slot: Vec<usize>,
    s: Vec<Vec<f64>>,
    y: Vec<Vec<f64>>,
    rho: Vec<f64>,
    alpha: Vec<f64>,
}

impl UpdateInfo {
    fn new(m: usize, dimension: usize) -> UpdateInfo {
        UpdateInfo {
            m,
            k: 0,
            slot: (0..m).collect(),
            s: vec![vec![0.0; dimension]; m],
            y: vec![vec![0.0; dimension]; m],
            rho: vec![0.0; m],
            alpha: vec![0.0; m],
        }
    }

    /// `update(lsr)`.
    fn update(&mut self, search: &Search) {
        let position = if self.k < self.m {
            self.k
        } else {
            for i in 0..self.m - 1 {
                self.slot[i] = self.slot[i + 1];
                self.rho[i] = self.rho[i + 1];
            }
            self.m - 1
        };
        let (s, y) = (&mut self.s[self.slot[position]], &mut self.y[self.slot[position]]);
        let mut sy = 0.0;
        for j in 0..s.len() {
            s[j] = search.next_point[j] - search.curr_point[j];
            y[j] = search.grad_at_next[j] - search.grad_at_curr[j];
            sy += s[j] * y[j];
        }
        self.rho[position] = 1.0 / sy;
        if self.k < self.m {
            self.k += 1;
        }
    }

    /// `computeDirection`: the two loop recursion over the gradient in
    /// `direction`, negated at the end.
    fn direction(&mut self, direction: &mut [f64]) {
        for i in (0..self.k).rev() {
            let s = &self.s[self.slot[i]];
            let y = &self.y[self.slot[i]];
            self.alpha[i] = self.rho[i] * inner_product(s, direction);
            for (d, &yj) in direction.iter_mut().zip(y) {
                *d -= self.alpha[i] * yj;
            }
        }
        for i in 0..self.k {
            let s = &self.s[self.slot[i]];
            let y = &self.y[self.slot[i]];
            let beta = self.rho[i] * inner_product(y, direction);
            for (d, &sj) in direction.iter_mut().zip(s) {
                *d += sj * (self.alpha[i] - beta);
            }
        }
        for d in direction.iter_mut() {
            *d = -*d;
        }
    }
}

/// `QNMinimizer.minimize` in progress.
struct Minimizer<'f, 'i> {
    function: &'f NegLogLikelihood<'i>,
    params: QnParams,
    search: Search,
    updates: UpdateInfo,
    direction: Vec<f64>,
    /// Scratch for the function: one sum per outcome, the gradient, and
    /// with several threads one chunk's gradient.
    sums: Vec<f64>,
    gradient: Vec<f64>,
    chunk_grad: Vec<f64>,
}

impl<'f, 'i> Minimizer<'f, 'i> {
    fn new(function: &'f NegLogLikelihood<'i>, params: &QnParams) -> Minimizer<'f, 'i> {
        let dimension = function.dimension;
        let threads = function.chunks.len();
        Minimizer {
            function,
            params: *params,
            search: Search {
                step_size: 0.0,
                value_at_curr: 0.0,
                value_at_next: 0.0,
                grad_at_curr: vec![0.0; dimension],
                grad_at_next: vec![0.0; dimension],
                pseudo_grad: vec![0.0; if params.l1() { dimension } else { 0 }],
                curr_point: vec![0.0; dimension],
                next_point: vec![0.0; dimension],
                sign: vec![0.0; if params.l1() { dimension } else { 0 }],
                fct_eval_count: 0,
            },
            updates: UpdateInfo::new(params.updates as usize, dimension),
            direction: vec![0.0; dimension],
            sums: vec![0.0; function.num_outcomes],
            gradient: vec![0.0; dimension],
            chunk_grad: if threads > 1 { vec![0.0; dimension] } else { Vec::new() },
        }
    }

    fn value_at(&mut self, x: &[f64]) -> f64 {
        self.function.value_at(x, &mut self.sums)
    }

    /// The gradient at `x` into `out`.
    fn gradient_at(&mut self, x: &[f64], out: &mut [f64]) {
        self.function.gradient_at(x, &mut self.sums, &mut self.gradient, &mut self.chunk_grad);
        out.copy_from_slice(&self.gradient);
    }

    /// `computePseudoGrad`: the pseudo gradient of the L1 penalized
    /// objective at `x` with gradient `g`, into `pg`.
    fn pseudo_gradient(l1_cost: f64, x: &[f64], g: &[f64], pg: &mut [f64]) {
        for ((p, &xi), &gi) in pg.iter_mut().zip(x).zip(g) {
            // At zero, the one sided derivative that points downhill.
            *p = if xi < 0.0 || (xi == 0.0 && gi > l1_cost) {
                gi - l1_cost
            } else if xi > 0.0 || gi < -l1_cost {
                gi + l1_cost
            } else {
                0.0
            };
        }
    }

    /// The loop of `minimize`: returns why it stopped.
    fn minimize(&mut self, observe: &mut dyn FnMut(&QnIteration, &[f64])) -> Result<QnStop> {
        let l1 = self.params.l1();
        // The origin (next_point starts as it), its value and gradient, in
        // the search's own buffers.
        let origin = std::mem::take(&mut self.search.next_point);
        let mut grad = std::mem::take(&mut self.search.grad_at_next);
        let mut value = self.value_at(&origin);
        self.gradient_at(&origin, &mut grad);
        if l1 {
            value += self.params.l1_cost * l1_norm(&origin);
            let mut pseudo = std::mem::take(&mut self.search.pseudo_grad);
            Minimizer::pseudo_gradient(self.params.l1_cost, &origin, &grad, &mut pseudo);
            self.search.pseudo_grad = pseudo;
        }
        self.search.value_at_next = value;
        self.search.grad_at_next = grad;
        self.search.next_point = origin;

        let mut initial_step = 1.0 / l2_norm(if l1 { &self.search.pseudo_grad } else { &self.search.grad_at_next });
        for iteration in 1..=self.params.iterations {
            self.direction.copy_from_slice(if l1 { &self.search.pseudo_grad } else { &self.search.grad_at_next });
            self.updates.direction(&mut self.direction);
            if l1 {
                for (d, &pg) in self.direction.iter_mut().zip(&self.search.pseudo_grad) {
                    if *d * pg >= 0.0 {
                        *d = 0.0;
                    }
                }
                self.constrained_line_search(initial_step)?;
                let mut pseudo = std::mem::take(&mut self.search.pseudo_grad);
                Minimizer::pseudo_gradient(
                    self.params.l1_cost,
                    &self.search.next_point,
                    &self.search.grad_at_next,
                    &mut pseudo,
                );
                self.search.pseudo_grad = pseudo;
            } else {
                self.line_search(initial_step)?;
            }
            self.updates.update(&self.search);
            let stop = self.converged();
            observe(
                &QnIteration {
                    iteration,
                    value: self.search.value_at_next,
                    change_rate: self.search.change_rate(),
                    step_size: self.search.step_size,
                    fct_evals: self.search.fct_eval_count,
                    stop,
                },
                &self.search.next_point,
            );
            if let Some(stop) = stop {
                return Ok(stop);
            }
            initial_step = 1.0;
        }
        Ok(QnStop::Iterations)
    }

    /// `LineSearch.doLineSearch`: backtracking from `initial_step` along
    /// `direction` until the Armijo condition holds, then the gradient at
    /// the new point.
    fn line_search(&mut self, initial_step: f64) -> Result<()> {
        let mut step = initial_step;
        let mut count = self.search.fct_eval_count;
        let dir_gradient = inner_product(&self.direction, &self.search.grad_at_next);
        let cached = ARMIJO * dir_gradient;
        let mut next = std::mem::take(&mut self.search.curr_point);
        let value_at_next;
        loop {
            for ((n, &x), &d) in next.iter_mut().zip(&self.search.next_point).zip(&self.direction) {
                *n = x + d * step;
            }
            let value = self.value_at(&next);
            count += 1;
            if value <= self.search.value_at_next + cached * step {
                value_at_next = value;
                break;
            }
            if step == 0.0 {
                return Err(Error::LineSearch);
            }
            step *= SHRINK;
        }
        let mut grad = std::mem::take(&mut self.search.grad_at_curr);
        self.gradient_at(&next, &mut grad);
        self.set_all(step, value_at_next, grad, next, count);
        Ok(())
    }

    /// `LineSearch.doConstrainedLineSearch`: the step projected onto the
    /// orthant of the current point (its pseudo gradient's where a
    /// coordinate is zero), with the L1 term in the value and the
    /// sufficient decrease measured against the pseudo gradient.
    fn constrained_line_search(&mut self, initial_step: f64) -> Result<()> {
        let mut step = initial_step;
        let mut count = self.search.fct_eval_count;
        for ((sign, &x), &pg) in self.search.sign.iter_mut().zip(&self.search.next_point).zip(&self.search.pseudo_grad)
        {
            *sign = if x == 0.0 { -pg } else { x };
        }
        let mut next = std::mem::take(&mut self.search.curr_point);
        let value_at_next;
        loop {
            for ((n, &x), &d) in next.iter_mut().zip(&self.search.next_point).zip(&self.direction) {
                *n = x + d * step;
            }
            for (n, &sign) in next.iter_mut().zip(&self.search.sign) {
                if *n * sign <= 0.0 {
                    *n = 0.0;
                }
            }
            let value = self.value_at(&next) + self.params.l1_cost * l1_norm(&next);
            count += 1;
            let mut dir_gradient = 0.0;
            for ((&n, &x), &pg) in next.iter().zip(&self.search.next_point).zip(&self.search.pseudo_grad) {
                dir_gradient += (n - x) * pg;
            }
            if value <= self.search.value_at_next + ARMIJO * dir_gradient {
                value_at_next = value;
                break;
            }
            if step == 0.0 {
                return Err(Error::LineSearch);
            }
            step *= SHRINK;
        }
        let mut grad = std::mem::take(&mut self.search.grad_at_curr);
        self.gradient_at(&next, &mut grad);
        self.set_all(step, value_at_next, grad, next, count);
        Ok(())
    }

    /// `LineSearchResult.setAll`: the old next becomes the current, the
    /// new point the next.
    fn set_all(&mut self, step: f64, value_at_next: f64, grad_at_next: Vec<f64>, next: Vec<f64>, count: i32) {
        let s = &mut self.search;
        s.step_size = step;
        s.value_at_curr = s.value_at_next;
        s.value_at_next = value_at_next;
        s.grad_at_curr = std::mem::replace(&mut s.grad_at_next, grad_at_next);
        s.curr_point = std::mem::replace(&mut s.next_point, next);
        s.fct_eval_count = count;
    }

    /// `isConverged`, as the test that fired.
    fn converged(&self) -> Option<QnStop> {
        let s = &self.search;
        if s.change_rate() < CONVERGE_TOLERANCE {
            return Some(QnStop::ChangeRate);
        }
        let x_norm = max_one(l2_norm(&s.next_point));
        let grad_norm = l2_norm(if self.params.l1() { &s.pseudo_grad } else { &s.grad_at_next });
        if grad_norm / x_norm < REL_GRAD_NORM_TOL {
            return Some(QnStop::GradientNorm);
        }
        if s.step_size < MIN_STEP_SIZE {
            return Some(QnStop::StepSize);
        }
        if s.fct_eval_count > self.params.max_fct_eval {
            return Some(QnStop::FunctionEvaluations);
        }
        None
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

    fn index() -> DataIndex {
        DataIndex::index(&events(), &IndexParams::new(IndexerKind::OnePass, 0, true)).unwrap()
    }

    #[test]
    fn params_come_from_the_training_parameters_and_are_validated() {
        let mut p = TrainingParameters::new();
        assert_eq!(QnParams::from_training_parameters(&p).unwrap(), QnParams::default());
        p.put(Key::ITERATIONS, 7);
        p.put(Key::THREADS, "3");
        p.put(Key::L1_COST, 0.0);
        p.put(Key::L2_COST, "0.5");
        p.put(Key::NUM_OF_UPDATES, 4);
        p.put(Key::MAX_FCT_EVAL, 50);
        assert_eq!(
            QnParams::from_training_parameters(&p).unwrap(),
            QnParams { iterations: 7, threads: 3, l1_cost: 0.0, l2_cost: 0.5, updates: 4, max_fct_eval: 50 }
        );
        p.put(Key::L1_COST, -0.1);
        assert_eq!(QnParams::from_training_parameters(&p), Err(Error::NegativeCost { key: Key::L1_COST, value: -0.1 }));
        p.put(Key::L1_COST, 0.1);
        p.put(Key::THREADS, 0);
        assert_eq!(QnParams::from_training_parameters(&p), Err(Error::NotPositive { key: Key::THREADS, value: 0 }));
        p.put(Key::THREADS, 1);
        p.put(Key::NUM_OF_UPDATES, 0);
        assert_eq!(
            QnParams::from_training_parameters(&p),
            Err(Error::NotPositive { key: Key::NUM_OF_UPDATES, value: 0 })
        );
        p.put(Key::NUM_OF_UPDATES, 1);
        p.put(Key::MAX_FCT_EVAL, -1);
        assert_eq!(
            QnParams::from_training_parameters(&p),
            Err(Error::NotPositive { key: Key::MAX_FCT_EVAL, value: -1 })
        );
        // Iterations are the minimizer's to refuse.
        p.put(Key::MAX_FCT_EVAL, 1);
        p.put(Key::ITERATIONS, 0);
        assert!(QnParams::from_training_parameters(&p).is_ok());
        assert_eq!(train(&index(), &p).err(), Some(Error::NotPositive { key: Key::ITERATIONS, value: 0 }));
    }

    #[test]
    fn one_outcome_is_refused_before_training() {
        let events = [event("s", "a"), event("s", "b")];
        let index = DataIndex::index(&events, &IndexParams::new(IndexerKind::OnePass, 0, true)).unwrap();
        assert_eq!(train(&index, &TrainingParameters::new()).err(), Some(Error::InsufficientTrainingData(ONE_OUTCOME)));
    }

    #[test]
    fn a_bad_parameter_is_refused_before_the_events_are_looked_at() {
        // QNTrainer.init and validate run before the indexer: with one
        // outcome and Threads 0 the JVM reports the threads.
        let events = [event("s", "a"), event("s", "b")];
        let mut p = TrainingParameters::new();
        p.put(Key::THREADS, 0);
        assert_eq!(train_events(&events, &p).err(), Some(Error::NotPositive { key: Key::THREADS, value: 0 }));
        // So does Iterations that does not parse, with no events at all.
        let mut p = TrainingParameters::new();
        p.put(Key::ITERATIONS, "many");
        assert!(matches!(train_events(&[], &p), Err(Error::ParameterFormat { .. })));
    }

    #[test]
    fn an_event_without_values_in_a_real_value_index_is_refused() {
        // The JVM's NegLogLikelihood reads values[ei][ci] and throws
        // NullPointerException on the event without values.
        let events = [
            Event::with_values("yes", vec![JavaString::new("a")], Some(vec![2.0])).unwrap(),
            Event::with_values("no", vec![JavaString::new("b")], None).unwrap(),
        ];
        let index = DataIndex::index(&events, &IndexParams::new(IndexerKind::OnePassRealValue, 0, true)).unwrap();
        assert_eq!(train(&index, &TrainingParameters::new()).err(), Some(Error::MissingValues { event: 1 }));
        // One outcome is reported first, as the JVM checks it first.
        let one = [
            Event::with_values("s", vec![JavaString::new("b")], None).unwrap(),
            Event::with_values("s", vec![JavaString::new("a")], Some(vec![2.0])).unwrap(),
        ];
        let index = DataIndex::index(&one, &IndexParams::new(IndexerKind::OnePassRealValue, 0, true)).unwrap();
        assert_eq!(train(&index, &TrainingParameters::new()).err(), Some(Error::InsufficientTrainingData(ONE_OUTCOME)));
    }

    #[test]
    fn training_fits_the_events() {
        let mut p = TrainingParameters::defaults();
        p.put(Key::CUTOFF, 0);
        let (model, report) = train_events(&events(), &p).unwrap();
        assert_eq!(model.outcome_labels(), ["yes", "no", "maybe"]);
        assert_eq!(model.num_predicates(), 8);
        assert_eq!(model.pattern(3), [0, 1, 2]);
        assert_eq!(model.all_parameters().len(), 24);
        assert_eq!(report.event_hash, Some(crate::event::checksum(&events())));
        let index = index();
        // The elastic net keeps the weights small, but the fit is right.
        let point = {
            // Undo the layout to evaluate the accuracy at the model's point.
            let (n, f) = (model.num_outcomes(), model.num_predicates());
            let mut x = vec![0.0; n * f];
            for pr in 0..f {
                for o in 0..n {
                    x[o * f + pr] = model.parameters(pr)[o];
                }
            }
            x
        };
        assert_eq!(training_accuracy(&index, &point), 1.0);
    }

    #[test]
    fn the_observer_sees_every_iteration_and_a_shorter_run_is_a_prefix() {
        let index = index();
        let params = QnParams { l1_cost: 0.0, ..QnParams::default() };
        let mut seen = Vec::new();
        let (model, stop) = train_model(&index, &params, &mut |it, point| {
            seen.push((*it, point.to_vec()));
        })
        .unwrap();
        let ran = seen.len();
        assert!(ran > 3 && ran < 100, "{ran}");
        assert_eq!(stop, QnStop::ChangeRate);
        assert_eq!(seen[ran - 1].0.stop, Some(stop));
        assert!(seen[..ran - 1].iter().all(|(it, _)| it.stop.is_none()));
        assert_eq!(seen.iter().map(|(it, _)| it.iteration).collect::<Vec<_>>(), (1..=ran as i32).collect::<Vec<_>>());
        assert!(seen.windows(2).all(|w| w[1].0.value <= w[0].0.value));
        assert!(seen.windows(2).all(|w| w[1].0.fct_evals > w[0].0.fct_evals));
        assert_eq!(model_parameters(&seen[ran - 1].1, 3, 8, &params), model.all_parameters());
        // A run of two iterations is the first two of the full run.
        let short = QnParams { iterations: 2, ..params };
        let (two, stop) = train_model(&index, &short, &mut |_, _| {}).unwrap();
        assert_eq!(stop, QnStop::Iterations);
        assert_eq!(two.all_parameters(), model_parameters(&seen[1].1, 3, 8, &short));
    }

    #[test]
    fn the_elastic_net_rescales_the_point_at_the_end() {
        let params = QnParams::default();
        let point: Vec<f64> = (0..6).map(|i| i as f64 * 0.5).collect();
        let scaled = model_parameters(&point, 2, 3, &params);
        // Predicate major: x[o * 3 + p].
        assert_eq!(scaled[0], 0.0);
        assert_eq!(scaled[1], 1.1f64.sqrt() * 1.5);
        assert_eq!(scaled[2], 1.1f64.sqrt() * 0.5);
        let l2 = QnParams { l1_cost: 0.0, ..params };
        assert_eq!(model_parameters(&point, 2, 3, &l2), [0.0, 1.5, 0.5, 2.0, 1.0, 2.5]);
    }

    #[test]
    fn chunking_by_threads_changes_only_the_order_of_sums() {
        let index = index();
        let one = train_model(&index, &QnParams::default(), &mut |_, _| {}).unwrap().0;
        let params = QnParams { threads: 3, ..QnParams::default() };
        let three = train_model(&index, &params, &mut |_, _| {}).unwrap().0;
        assert_eq!(one.all_parameters().len(), three.all_parameters().len());
        for (a, b) in one.all_parameters().iter().zip(three.all_parameters()) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn l1_leaves_parameters_at_zero_and_a_low_evaluation_limit_stops_early() {
        let index = index();
        let params = QnParams { l1_cost: 1.0, l2_cost: 0.0, ..QnParams::default() };
        let (model, _) = train_model(&index, &params, &mut |_, _| {}).unwrap();
        assert!(model.all_parameters().contains(&0.0));
        let params = QnParams { max_fct_eval: 2, ..QnParams::default() };
        let (_, stop) = train_model(&index, &params, &mut |_, _| {}).unwrap();
        assert_eq!(stop, QnStop::FunctionEvaluations);
    }

    #[test]
    fn discarded_corrections_alias_as_the_jvm_arrays_do() {
        let mut u = UpdateInfo::new(2, 1);
        let step = |a: f64, b: f64| Search {
            step_size: 1.0,
            value_at_curr: 0.0,
            value_at_next: 0.0,
            grad_at_curr: vec![0.0],
            grad_at_next: vec![b],
            pseudo_grad: vec![],
            curr_point: vec![0.0],
            next_point: vec![a],
            sign: vec![],
            fct_eval_count: 0,
        };
        u.update(&step(1.0, 1.0));
        u.update(&step(2.0, 1.0));
        assert_eq!((u.s[0][0], u.s[1][0]), (1.0, 2.0));
        // The third correction discards the first: slot 0 now names the
        // array of slot 1, and the write into slot 1 shows in both.
        u.update(&step(3.0, 1.0));
        assert_eq!(u.slot, [1, 1]);
        assert_eq!((u.s[u.slot[0]][0], u.s[u.slot[1]][0]), (3.0, 3.0));
        assert_eq!(u.rho, [0.5, 1.0 / 3.0]);
    }

    #[test]
    fn log_sum_of_exps_skips_negative_infinity_and_argmax_takes_the_first() {
        assert_eq!(log_sum_of_exps(&[f64::NEG_INFINITY, 0.0]), 0.0);
        assert_eq!(log_sum_of_exps(&[1.0, 1.0]), 1.0 + strict_math::log(2.0));
        assert_eq!(argmax(&[1.0, 3.0, 3.0]), 1);
        assert!(max_one(f64::NAN).is_nan());
        assert_eq!(max_one(0.5), 1.0);
        assert_eq!(max_one(2.0), 2.0);
    }
}
