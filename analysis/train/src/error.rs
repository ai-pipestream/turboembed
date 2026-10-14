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
    /// The `DataIndexer` parameter names none of the three built in
    /// indexers. OpenNLP would load it as an extension class by that name.
    UnknownIndexer(String),
    /// A parameters file could not be read as a Java properties file: the
    /// reader's message.
    Properties(String),
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
            Error::UnknownIndexer(name) => {
                write!(f, "DataIndexer {name:?} is not OnePass, TwoPass or OnePassRealValue")
            }
            Error::Properties(e) => write!(f, "parameters file: {e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<turbo_maxent::Error> for Error {
    fn from(e: turbo_maxent::Error) -> Error {
        Error::Properties(e.to_string())
    }
}
