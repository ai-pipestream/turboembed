use std::fmt;

/// Why training input, a parameter or an indexer run was refused. Each
/// variant stands for one exception OpenNLP throws, with the same message
/// where the message is part of OpenNLP's behaviour.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// `FileEventStream` met a line with no tokens. OpenNLP calls
    /// `StringTokenizer.nextToken` on it and throws
    /// `NoSuchElementException`, so an event file cannot have an empty
    /// line (an OpenNLP bug this port keeps as an error).
    EmptyEventLine { line: usize },
    /// `RealValueFileEventStream.parseEvent` met a line with no fields:
    /// `InvalidFormatException("An event line must start with an outcome: ...")`.
    NoOutcome { line: usize, text: String },
    /// A real valued context has a negative value:
    /// `RuntimeException("Negative values are not allowed: " + context)`.
    NegativeValue { line: usize, context: crate::java_string::JavaString },
    /// An event's values are not one per context. OpenNLP's `Event` accepts
    /// that and fails later, out of bounds, in the indexer or the trainer;
    /// this port refuses it when the event is made.
    ValuesLength { contexts: usize, values: usize },
    /// `InsufficientTrainingDataException` with OpenNLP's message: no event
    /// survived the cutoff, or the events have a single outcome.
    InsufficientTrainingData(&'static str),
    /// `TwoPassDataIndexer` writes every outcome and context with
    /// `DataOutputStream.writeUTF`, which throws `UTFDataFormatException`
    /// for a string over 65535 bytes of modified UTF-8. `what` says which
    /// string, `bytes` its encoded length.
    StringTooLong { what: &'static str, bytes: usize },
    /// A parameter holds a value of another Java type than the getter
    /// casts to, where OpenNLP throws `ClassCastException`: an `Integer`
    /// asked for as a string, a `Double` asked for as an integer, and so on.
    /// `expected` and `found` are the Java class names.
    ParameterType { key: String, expected: &'static str, found: &'static str },
    /// A parameter holds a string that does not parse as the number the
    /// getter asked for (`NumberFormatException`).
    ParameterFormat { key: String, expected: &'static str, value: String },
    /// `Properties.store` casts every value to `String`, so
    /// `TrainingParameters.serialize` throws `ClassCastException` on a
    /// parameter put as a number or a boolean. `found` is its Java class.
    SerializeType { key: String, found: &'static str },
    /// The algorithm named by the `Algorithm` parameter is none OpenNLP has
    /// built in (`IllegalArgumentException("Unknown algorithm type: ...")`).
    UnknownAlgorithm(String),
    /// The algorithm names a trainer of another kind than the one asked
    /// for: `TrainerFactory.getEventTrainer` cannot return the sequence
    /// perceptron, nor `getEventModelSequenceTrainer` an event trainer
    /// (`ClassCastException`).
    WrongTrainerType { algorithm: crate::params::Algorithm, wanted: crate::params::TrainerType },
    /// `TrainerFactory.getEventModelSequenceTrainer` without an `Algorithm`
    /// (`IllegalArgumentException("Trainer type couldn't be determined!")`).
    NoAlgorithm,
    /// `PerceptronTrainer.setTolerance` refuses a negative tolerance
    /// (`IllegalArgumentException`).
    Tolerance(f64),
    /// `PerceptronTrainer.setStepSizeDecrease` refuses a decrease over 100
    /// (`IllegalArgumentException`).
    StepSizeDecrease(f64),
    /// `PerceptronModelWriter` drops every zero parameter, and when none is
    /// left `computeOutcomePatterns` reads the first of no predicates
    /// (`ArrayIndexOutOfBoundsException`): a perceptron model whose
    /// parameters are all zero cannot be written.
    NoParameters,
    /// `SimplePerceptronSequenceTrainer` after its iterations: it counts
    /// the events it tagged right with the index's outcome list at each
    /// event's position in the stream, which is past the list's end when
    /// the cutoff dropped an event. The JVM throws
    /// `ArrayIndexOutOfBoundsException` and returns no model.
    SequenceEventsDropped { indexed: u32, streamed: usize },
    /// A `SequenceSource::update_context` returned `got` events for a
    /// sequence of `expected`. The JVM reads the tagged events by the
    /// sequence's positions and fails out of bounds.
    SequenceLength { expected: usize, got: usize },
    /// A `SequenceSource::update_context` tagged an event with an outcome
    /// the index does not have. The JVM unboxes `omap.get(outcome)`, null,
    /// and throws `NullPointerException`.
    UnknownOutcome(String),
    /// The `DataIndexer` parameter names none of the three built in
    /// indexers. OpenNLP would load it as an extension class by that name.
    UnknownIndexer(String),
    /// A parameters file could not be read as a Java properties file: the
    /// reader's message.
    Properties(String),
    /// `GISTrainer` refuses `Smoothing` and `GaussianSmoothing` together
    /// (`RuntimeException`).
    BothSmoothings,
    /// `GISTrainer` refuses a `Threads` value below one
    /// (`IllegalArgumentException`).
    Threads(i32),
    /// `QNTrainer` and `QNMinimizer` refuse a count that is not positive:
    /// `NumOfUpdates`, `MaxFctEval`, `Threads` or `Iterations`
    /// (`IllegalArgumentException`).
    NotPositive { key: &'static str, value: i32 },
    /// `QNTrainer` refuses a negative `L1Cost` or `L2Cost`
    /// (`IllegalArgumentException`).
    NegativeCost { key: &'static str, value: f64 },
    /// The QN line search found no step that satisfies the sufficient
    /// decrease condition before the step reached zero, which happens when
    /// the objective is NaN. The JVM's loop does not terminate then.
    LineSearch,
    /// An `OnePassRealValue` index keeps the events' values, and unique
    /// event `event` has none (`Event::with_values(.., None)`). The
    /// trainers read `values[ei][ci]` for every event when the index has
    /// values, so the JVM throws `NullPointerException` in `GISTrainer`,
    /// `NegLogLikelihood.valueAt` and `PerceptronTrainer`.
    MissingValues { event: usize },
}

pub type Result<T> = std::result::Result<T, Error>;

/// The message of the `InsufficientTrainingDataException` the indexers throw.
pub const NO_EVENTS: &str = "Insufficient training data to create model.";
/// The message of the `InsufficientTrainingDataException` `AbstractEventTrainer.train` throws.
pub const ONE_OUTCOME: &str = "Training data must contain more than one outcome";

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::EmptyEventLine { line } => {
                write!(f, "line {line}: an event line has no outcome (OpenNLP throws NoSuchElementException)")
            }
            Error::NoOutcome { line, text } => {
                write!(f, "line {line}: An event line must start with an outcome: \"{text}\"")
            }
            Error::NegativeValue { line, context } => {
                write!(f, "line {line}: Negative values are not allowed: {context}")
            }
            Error::ValuesLength { contexts, values } => {
                write!(f, "an event with {contexts} contexts has {values} values")
            }
            Error::InsufficientTrainingData(message) => f.write_str(message),
            Error::StringTooLong { what, bytes } => {
                write!(f, "the {what} is {bytes} bytes of modified UTF-8, over the 65535 writeUTF can write")
            }
            Error::ParameterType { key, expected, found } => {
                write!(f, "parameter {key:?}: expected a {expected}, found a {found} (ClassCastException)")
            }
            Error::ParameterFormat { key, expected, value } => {
                write!(f, "parameter {key:?}: {value:?} is not a {expected} (NumberFormatException)")
            }
            Error::SerializeType { key, found } => {
                write!(f, "parameter {key:?} is a {found}, which Properties.store cannot write (ClassCastException)")
            }
            Error::UnknownAlgorithm(name) => write!(f, "Unknown algorithm type: {name}"),
            Error::WrongTrainerType { algorithm, wanted } => {
                write!(f, "{} is not a {} trainer (ClassCastException)", algorithm.name(), wanted.value())
            }
            Error::NoAlgorithm => f.write_str("Trainer type couldn't be determined!"),
            Error::Tolerance(t) => {
                write!(f, "tolerance must be a positive number but is {}!", turbo_jvm::text::double_to_string(*t))
            }
            Error::StepSizeDecrease(d) => {
                write!(f, "decrease must be between 0 and 100 but is {}!", turbo_jvm::text::double_to_string(*d))
            }
            Error::NoParameters => f.write_str("every parameter is zero, which PerceptronModelWriter cannot write"),
            Error::SequenceLength { expected, got } => {
                write!(f, "updateContext returned {got} events for a sequence of {expected}")
            }
            Error::UnknownOutcome(outcome) => write!(f, "updateContext tagged an event {outcome:?}, not an outcome"),
            Error::SequenceEventsDropped { indexed, streamed } => write!(
                f,
                "the cutoff dropped {} of the {streamed} events, after which SimplePerceptronSequenceTrainer fails (ArrayIndexOutOfBoundsException)",
                streamed - *indexed as usize
            ),
            Error::UnknownIndexer(name) => {
                write!(f, "DataIndexer {name:?} is not OnePass, TwoPass or OnePassRealValue")
            }
            Error::Properties(e) => write!(f, "parameters file: {e}"),
            Error::BothSmoothings => f.write_str("Cannot set both Gaussian smoothing and Simple smoothing"),
            Error::Threads(n) => write!(f, "threads must be at least one or greater but is {n}!"),
            Error::NotPositive { key, value } => {
                write!(f, "parameter {key} must be positive but is {value} (IllegalArgumentException)")
            }
            Error::NegativeCost { key, value } => {
                write!(f, "parameter {key} must not be negative but is {value} (IllegalArgumentException)")
            }
            Error::LineSearch => f.write_str("the line search found no acceptable step (the JVM would not terminate)"),
            Error::MissingValues { event } => write!(
                f,
                "event {event} of a real value index has no values, which the JVM trainers read (NullPointerException)"
            ),
        }
    }
}

impl std::error::Error for Error {}

impl From<turbo_maxent::Error> for Error {
    fn from(e: turbo_maxent::Error) -> Error {
        Error::Properties(e.to_string())
    }
}
