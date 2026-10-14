use std::fmt;
use std::path::PathBuf;

/// What a recipe can fail on.
#[derive(Debug)]
pub enum Error {
    /// A file or directory could not be read or written.
    Io { path: PathBuf, message: String },
    /// The corpus could not be read.
    Masc(turbo_formats::masc::MascError),
    /// A held-out prediction OpenNLP's evaluator throws on.
    Eval(turbo_formats::evaluate::EvalError),
    /// Training failed.
    Train(turbo_train_segmentation::Error),
    /// A model file could not be read.
    Model(String),
    /// The zoo refused or failed.
    Zoo(turbo_zoo::ZooError),
    /// The gate or its record failed.
    Gate(String),
    /// The release has no digest recorded yet, so it cannot be fetched
    /// or checked ([`crate::masc::RELEASE`]).
    NotRecorded,
    /// The corpus directory is not the recorded release.
    CorpusDigest { expected: String, found: String },
    /// A reference model file is not the zoo's.
    ReferenceDigest { id: String, expected: String, found: String },
    /// The split leaves one side without a sample for the recipe.
    EmptySplit { what: &'static str },
}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io { path, message } => write!(f, "{}: {message}", path.display()),
            Error::Masc(e) => write!(f, "MASC: {e}"),
            Error::Eval(e) => write!(f, "evaluation: {e}"),
            Error::Train(e) => write!(f, "training: {e}"),
            Error::Model(e) => write!(f, "model: {e}"),
            Error::Zoo(e) => write!(f, "zoo: {e}"),
            Error::Gate(e) => write!(f, "gate: {e}"),
            Error::NotRecorded => {
                f.write_str("the MASC release has no recorded digest yet; see turbo_train_recipes::masc::RELEASE")
            }
            Error::CorpusDigest { expected, found } => {
                write!(f, "the corpus directory has the digest {found}, the recorded release {expected}")
            }
            Error::ReferenceDigest { id, expected, found } => {
                write!(f, "reference model {id}: SHA-256 {found}, the zoo's {expected}")
            }
            Error::EmptySplit { what } => write!(f, "the split leaves no {what}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<turbo_formats::masc::MascError> for Error {
    fn from(e: turbo_formats::masc::MascError) -> Error {
        Error::Masc(e)
    }
}

impl From<turbo_formats::evaluate::EvalError> for Error {
    fn from(e: turbo_formats::evaluate::EvalError) -> Error {
        Error::Eval(e)
    }
}

impl From<turbo_train_segmentation::Error> for Error {
    fn from(e: turbo_train_segmentation::Error) -> Error {
        Error::Train(e)
    }
}

impl From<turbo_zoo::ZooError> for Error {
    fn from(e: turbo_zoo::ZooError) -> Error {
        Error::Zoo(e)
    }
}

pub(crate) fn io(path: impl Into<PathBuf>) -> impl FnOnce(std::io::Error) -> Error {
    let path = path.into();
    move |e| Error::Io { path, message: e.to_string() }
}
