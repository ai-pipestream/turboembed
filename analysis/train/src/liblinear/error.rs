//! Why liblinear training was refused, or a liblinear model could not be
//! read or written.

use std::fmt;

use super::SolverType;

/// Why [`train_events`](super::train_events) or [`train`](super::train)
/// refused. Each variant but [`Error::NotScorable`] stands for an exception
/// the add-on or liblinear-java throws, and displays its message;
/// [`Error::Unsorted`] adds the event.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// The indexer or a training parameter getter refused
    /// (`crate::Error`'s variants: a parameter of the wrong type, no event
    /// left, one outcome).
    Train(crate::Error),
    /// `solverType` names no `SolverType` constant, or is absent (read as
    /// the empty string): `IllegalArgumentException("solverType [...] is
    /// not available!")`.
    UnknownSolver(String),
    /// A solver liblinear-java trains but whose models the add-on cannot
    /// score: `LiblinearModel.eval` calls `Linear.predictProbability`,
    /// which throws for every solver but the three logistic regression
    /// ones. Those solvers are not ported, and are refused before
    /// training.
    NotScorable(SolverType),
    /// `Parameter.setC`: `c` absent (read as zero) or not above zero.
    C(f64),
    /// `Parameter.setEps`: `eps` absent (read as zero) or not above zero.
    Eps(f64),
    /// `Parameter.setP`: a negative `p`.
    P(f64),
    /// `Linear.train` requires the features of each instance in strictly
    /// ascending index order. The add-on passes each event's predicate ids
    /// as the indexer keeps them, in the event's order with duplicates, so
    /// an event whose predicates are not in Java's string order, or that
    /// names one twice, is refused (`IllegalArgumentException`). `event` is
    /// the unique event's position in the index.
    Unsorted { event: usize },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Train(e) => e.fmt(f),
            Error::UnknownSolver(name) => write!(f, "solverType [{name}] is not available!"),
            Error::NotScorable(solver) => write!(
                f,
                "{} models cannot be scored by the liblinear add-on (probability output is only supported for logistic regression)",
                solver.name()
            ),
            Error::C(_) => f.write_str("C must not be <= 0"),
            Error::Eps(_) => f.write_str("eps must not be <= 0"),
            Error::P(_) => f.write_str("p must not be less than 0"),
            Error::Unsorted { event } => {
                write!(f, "feature nodes must be sorted by index in ascending order (unique event {event})")
            }
        }
    }
}

impl std::error::Error for Error {}

impl From<crate::Error> for Error {
    fn from(e: crate::Error) -> Error {
        Error::Train(e)
    }
}

/// Why a serialized liblinear model was refused. The reader is stricter
/// than the add-on's, which reads what it can and fails later or not at
/// all; each variant names what was wrong.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelError {
    /// The binary framing (`DataInputStream`): a length, a count or a
    /// string that runs past the end, or malformed modified UTF-8. The
    /// message is the reader's.
    Data(String),
    /// A count below zero.
    Count { what: &'static str, value: i32 },
    /// Bytes after the predicate map.
    Trailing(usize),
    /// A line of the liblinear text model (one based) that is not what
    /// `Linear.saveModel` writes.
    Line { line: usize, detail: String },
    /// The text model is consistent but unusable: a missing record, a
    /// count that does not match, or a model the add-on cannot score.
    Invalid(String),
    /// A predicate given twice in the predicate map.
    DuplicatePredicate(String),
    /// A predicate id the add-on cannot turn into a feature index (it adds
    /// one, and `FeatureNode` refuses an index under one).
    PredicateId { predicate: String, id: i32 },
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::Data(m) => write!(f, "liblinear model: {m}"),
            ModelError::Count { what, value } => write!(f, "liblinear model: {what} is {value}"),
            ModelError::Trailing(n) => write!(f, "liblinear model: {n} bytes after the predicate map"),
            ModelError::Line { line, detail } => write!(f, "liblinear model, text line {line}: {detail}"),
            ModelError::Invalid(m) => write!(f, "liblinear model: {m}"),
            ModelError::DuplicatePredicate(p) => write!(f, "liblinear model: predicate {p:?} is given twice"),
            ModelError::PredicateId { predicate, id } => {
                write!(f, "liblinear model: predicate {predicate:?} has id {id}, which is no feature index")
            }
        }
    }
}

impl std::error::Error for ModelError {}

impl From<turbo_maxent::Error> for ModelError {
    fn from(e: turbo_maxent::Error) -> ModelError {
        ModelError::Data(e.to_string())
    }
}
