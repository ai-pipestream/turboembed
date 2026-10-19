//! The liblinear trainer of the OpenNLP addons (`liblinear-addon`:
//! `LiblinearTrainer`, `LiblinearModel`, `LiblinearModelSerializer`), with
//! the logistic regression solvers of liblinear-java 2.44
//! (`de.bwaldvogel:liblinear`, a port of LIBLINEAR's `linear.cpp`)
//! ported here, so a model trained here has the weights and scores the
//! JVM's has.
//!
//! The add-on is an `AbstractEventTrainer`: the indexer numbers the events
//! (sorting and merging them, as `isSortAndMerge` is true), and `doTrain`
//! hands liblinear one instance per unique event, its features the
//! predicate ids plus one, each with the number of times the event was
//! seen as its value, and, when `bias` is zero or more, a last feature of
//! that value. The model it returns scores a context through
//! `Linear.predictProbability` with every known feature at value one.
//!
//! Parameters, read by the trainer's constructor with these defaults:
//! `solverType` (none; a `SolverType` constant name), `c` (0), `eps` (0),
//! `p` (0) and `bias` (0), so `c` and `eps` must be given. The add-on's
//! sample parameters file uses `L1R_LR` with `c = 1`, `eps = 0.001` and
//! `bias = 0`.
//!
//! Only the solvers whose models the add-on can score are ported:
//! `L2R_LR` (the primal Newton method, [`newton`]), `L2R_LR_DUAL` (dual
//! coordinate descent, `dual`, which falls back to the Newton method after
//! 300 iterations) and `L1R_LR` (newGLMNET, `l1`). liblinear-java trains
//! the others, but `LiblinearModel.eval` throws for their models, so they
//! are refused here ([`Error::NotScorable`]).
//!
//! What the port keeps from the Java code, because the numbers depend on
//! it:
//!
//! - Every sum in the order the Java loops add it, and BLAS's `ddot` five
//!   products at a time and `dnrm2` from the last element.
//! - The `java.util.Random` each `Parameter` holds, seeded with zero, for
//!   the shuffles of the dual and the L1 solver, shared by the weight
//!   vectors of a multi-class model in class order. liblinear-java's L1
//!   shuffle differs from the C code's (see `l1`).
//! - `Math.max` and `Math.min` with Java's NaN and signed zero rules.
//!
//! liblinear-java calls `Math.exp` and `Math.log`, which HotSpot computes
//! with its own routines. `log` here is fdlibm's (`turbo_jvm::strict_math`),
//! which differs from HotSpot's in the last bit on some arguments; `exp` is
//! the platform's `f64::exp`, which HotSpot's `Math.exp` (not correctly
//! rounded, and not portable here) differs from in the last bit on a few
//! arguments in a thousand. A difference in the last bit can grow through
//! the iterations, so the weights agree with the JVM's to a tolerance,
//! not always to the bit; `docs/liblinear.md` has the measured numbers.
//!
//! Quirks of the add-on, kept:
//!
//! - `TrainerFactory` cannot create the trainer: it has no constructor
//!   without arguments, and its class is in the default package. It is
//!   made with `new LiblinearTrainer(params)`, which reads the parameters
//!   but does not hand them to `AbstractTrainer`, so `init(params, ...)`
//!   must follow before `train`. [`train_events`] and [`train`] are that
//!   sequence with one set of parameters.
//! - liblinear refuses an instance whose feature indexes do not ascend,
//!   and the add-on passes each event's predicate ids in the event's
//!   order, duplicates kept: training needs every event's predicates in
//!   Java's string order with none twice ([`Error::Unsorted`]).
//! - The bias feature is trained but not scored: `eval` does not add it.
//! - `eval` returns the probabilities in liblinear's class order, which
//!   the add-on reads as outcome order ([`LiblinearModel::eval`]).
//! - The serialized predicate map is in `HashMap` order ([`format`]).

mod dual;
pub mod error;
pub mod format;
mod l1;
mod linear;
mod model;
mod newton;
mod problem;
#[cfg(test)]
mod tests;

pub use error::{Error, ModelError};
pub use linear::LinearModel;
pub use model::LiblinearModel;

use crate::event::{Event, EventChecksum};
use crate::gis::TrainingReport;
use crate::index::{DataIndex, IndexParams};
use crate::params::{DEFAULT_CUTOFF, Key, TrainerType, TrainingParameters};
use problem::Rows;

/// The add-on's parameter names.
pub struct LiblinearKey;

impl LiblinearKey {
    pub const SOLVER_TYPE: &'static str = "solverType";
    pub const C: &'static str = "c";
    pub const EPS: &'static str = "eps";
    pub const P: &'static str = "p";
    pub const BIAS: &'static str = "bias";
}

pub type Result<T> = std::result::Result<T, Error>;

/// `Math.exp` as liblinear-java calls it: the platform's, which differs
/// from HotSpot's in the last bit on some arguments.
#[inline]
pub(crate) fn exp(x: f64) -> f64 {
    x.exp()
}

/// `Math.log` as liblinear-java calls it, through fdlibm's logarithm
/// (`StrictMath.log`), which differs from HotSpot's `Math.log` in the
/// last bit on some arguments but gives the same result on every
/// platform. The correctly rounded `turbo_jvm::math::log` matches HotSpot
/// more often but is some 60 times slower, and the dual solver takes a
/// logarithm at every inner step.
#[inline]
pub(crate) fn log(x: f64) -> f64 {
    turbo_jvm::strict_math::log(x)
}

/// liblinear-java's `SolverType`, by its constant names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverType {
    L2rLr,
    L2rL2lossSvcDual,
    L2rL2lossSvc,
    L2rL1lossSvcDual,
    McsvmCs,
    L1rL2lossSvc,
    L1rLr,
    L2rLrDual,
    L2rL2lossSvr,
    L2rL2lossSvrDual,
    L2rL1lossSvrDual,
    OneclassSvm,
}

impl SolverType {
    pub const ALL: [SolverType; 12] = [
        SolverType::L2rLr,
        SolverType::L2rL2lossSvcDual,
        SolverType::L2rL2lossSvc,
        SolverType::L2rL1lossSvcDual,
        SolverType::McsvmCs,
        SolverType::L1rL2lossSvc,
        SolverType::L1rLr,
        SolverType::L2rLrDual,
        SolverType::L2rL2lossSvr,
        SolverType::L2rL2lossSvrDual,
        SolverType::L2rL1lossSvrDual,
        SolverType::OneclassSvm,
    ];

    /// The constant's name, which the parameters and the text model use.
    pub const fn name(self) -> &'static str {
        match self {
            SolverType::L2rLr => "L2R_LR",
            SolverType::L2rL2lossSvcDual => "L2R_L2LOSS_SVC_DUAL",
            SolverType::L2rL2lossSvc => "L2R_L2LOSS_SVC",
            SolverType::L2rL1lossSvcDual => "L2R_L1LOSS_SVC_DUAL",
            SolverType::McsvmCs => "MCSVM_CS",
            SolverType::L1rL2lossSvc => "L1R_L2LOSS_SVC",
            SolverType::L1rLr => "L1R_LR",
            SolverType::L2rLrDual => "L2R_LR_DUAL",
            SolverType::L2rL2lossSvr => "L2R_L2LOSS_SVR",
            SolverType::L2rL2lossSvrDual => "L2R_L2LOSS_SVR_DUAL",
            SolverType::L2rL1lossSvrDual => "L2R_L1LOSS_SVR_DUAL",
            SolverType::OneclassSvm => "ONECLASS_SVM",
        }
    }

    /// `SolverType.valueOf`: the exact constant name.
    pub fn from_name(name: &str) -> Option<SolverType> {
        SolverType::ALL.into_iter().find(|s| s.name() == name)
    }

    /// `isLogisticRegressionSolver`: the solvers whose models give
    /// probabilities, and the ones ported here.
    pub const fn is_logistic_regression(self) -> bool {
        matches!(self, SolverType::L2rLr | SolverType::L1rLr | SolverType::L2rLrDual)
    }
}

/// What `LiblinearTrainer`'s constructor reads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LiblinearParams {
    pub solver: SolverType,
    /// `c`: the cost of a misclassified instance, above zero.
    pub c: f64,
    /// `eps`: the stopping tolerance, above zero.
    pub eps: f64,
    /// `p`: the SVR loss's epsilon, which the logistic regression solvers
    /// do not read; `Parameter.setP` still refuses a negative one.
    pub p: f64,
    /// `bias`: the bias feature's value, none when negative.
    pub bias: i32,
}

impl LiblinearParams {
    /// The constructor's reads, in its order: `solverType` (absent is the
    /// empty string, which names no solver), then `c`, `eps`, `p` and
    /// `bias`. A solver whose models the add-on cannot score is refused
    /// here, before any training. `c`, `eps` and `p` are checked when the
    /// add-on checks them, at training ([`LiblinearParams::check`]).
    pub fn from_training_parameters(params: &TrainingParameters) -> Result<LiblinearParams> {
        let name = params.get_str(LiblinearKey::SOLVER_TYPE)?.unwrap_or("");
        let solver = SolverType::from_name(name).ok_or_else(|| Error::UnknownSolver(name.to_owned()))?;
        let lp = LiblinearParams {
            solver,
            c: params.get_double(LiblinearKey::C)?.unwrap_or(0.0),
            eps: params.get_double(LiblinearKey::EPS)?.unwrap_or(0.0),
            p: params.get_double(LiblinearKey::P)?.unwrap_or(0.0),
            bias: params.get_int(LiblinearKey::BIAS)?.unwrap_or(0),
        };
        if !solver.is_logistic_regression() {
            return Err(Error::NotScorable(solver));
        }
        Ok(lp)
    }

    /// `new Parameter(solverType, c, eps, p)`: `setC`, `setEps` and `setP`
    /// in that order, compared as Java compares, so a NaN passes.
    pub fn check(&self) -> Result<()> {
        if self.c <= 0.0 {
            return Err(Error::C(self.c));
        }
        if self.eps <= 0.0 {
            return Err(Error::Eps(self.eps));
        }
        if self.p < 0.0 {
            return Err(Error::P(self.p));
        }
        Ok(())
    }
}

/// `new LiblinearTrainer(params)`, `init(params, ...)` and
/// `train(events)`: the parameters read, the checksum over the events, the
/// indexer the parameters name with sorting on and the default cutoff when
/// none is given, then [`train`].
pub fn train_events(events: &[Event], params: &TrainingParameters) -> Result<(LiblinearModel, TrainingReport)> {
    LiblinearParams::from_training_parameters(params)?;
    // AbstractTrainer.validate before the indexer runs.
    params.get_int(Key::CUTOFF)?;
    params.get_int(Key::ITERATIONS)?;
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

/// `AbstractEventTrainer.train(DataIndexer)` with the add-on's `doTrain`:
/// the parameters read and validated, an index with one outcome refused,
/// then [`train_model`].
pub fn train(index: &DataIndex, params: &TrainingParameters) -> Result<LiblinearModel> {
    let lp = LiblinearParams::from_training_parameters(params)?;
    params.get_int(Key::CUTOFF)?;
    params.get_int(Key::ITERATIONS)?;
    if index.outcome_labels().len() <= 1 {
        return Err(Error::Train(crate::Error::InsufficientTrainingData(crate::error::ONE_OUTCOME)));
    }
    train_model(index, &lp)
}

/// The add-on's `doTrain`: the liblinear problem from the index, the
/// parameter checks, `Linear.train`, and the model with the index's
/// outcome labels and a predicate map of each predicate to its id.
pub fn train_model(index: &DataIndex, params: &LiblinearParams) -> Result<LiblinearModel> {
    let bias = params.bias >= 0;
    let n_events = index.num_unique_events();
    // The largest feature index is that of the last feature of some event,
    // which is the largest when the features ascend, as training requires.
    let mut max_index = 0usize;
    let mut entries = 0usize;
    for i in 0..n_events {
        let context = index.context(i);
        if let Some(&last) = context.last() {
            max_index = max_index.max(last as usize + 1);
        }
        entries += context.len() + usize::from(bias);
    }
    let mut x = Rows::with_capacity(n_events, entries);
    for i in 0..n_events {
        let seen = f64::from(index.seen(i));
        for &p in index.context(i) {
            x.push(p, seen);
        }
        if bias {
            x.push(max_index as u32, f64::from(params.bias));
        }
        x.end_row();
    }
    let n = max_index + usize::from(bias);

    // new Parameter(...), then Linear.train's checks.
    params.check()?;
    for i in 0..n_events {
        if !index.context(i).is_sorted_by(|a, b| a < b) {
            return Err(Error::Unsorted { event: i });
        }
    }

    let linear = linear::train(&x, index.outcomes(), n, f64::from(params.bias), params.solver, params.c, params.eps);
    let predicates = index.pred_labels().iter().enumerate().map(|(i, p)| (p.clone(), i as i32)).collect();
    Ok(LiblinearModel::new(linear, index.outcome_labels().to_vec(), predicates))
}
