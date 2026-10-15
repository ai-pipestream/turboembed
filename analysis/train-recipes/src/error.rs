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
    Train(turbo_train_components::Error),
    /// A model file could not be read.
    Model(String),
    /// The zoo refused or failed.
    Zoo(turbo_zoo::ZooError),
    /// The gate or its record failed.
    Gate(turbo_eval::model_gate::ModelGateError),
    /// A field of the recorded release that the zoo index refuses.
    InvalidRelease { field: &'static str, value: String, message: String },
    /// A field of a model card that the zoo index refuses.
    InvalidCard { field: &'static str, message: String },
    /// A document whose path is not under the corpus directory.
    OutsideCorpus { path: PathBuf },
    /// A card that may not become a zoo entry: its gate failed or its
    /// corpus is not the recorded release.
    NotShippable { id: String, reason: &'static str },
    /// A card file that is not a card.
    Json { path: PathBuf, message: String },
    /// A card of another schema version than the `expected` one.
    CardVersion { path: PathBuf, found: u32, expected: u32 },
    /// Held-out samples of one component given to a recipe of another.
    ComponentMismatch { recipe: &'static str, samples: &'static str },
    /// The release has no digest recorded yet, so it cannot be fetched
    /// or checked ([`crate::masc::RELEASE`]).
    NotRecorded,
    /// The corpus directory is not the recorded release.
    CorpusDigest { expected: String, found: String },
    /// A reference model file does not have its recorded digest.
    ReferenceDigest { id: String, expected: String, found: String },
    /// The split leaves one side without a sample for the recipe.
    EmptySplit { what: &'static str },
    /// A treebank whose licence check refuses it.
    Licence { treebank: &'static str, reason: &'static str },
    /// A treebank file that is not the recorded one.
    TreebankFile { treebank: &'static str, file: &'static str, expected: String, found: String },
    /// A sentence of a treebank file the CoNLL-U reader refuses.
    Conllu { treebank: &'static str, file: &'static str, error: Box<turbo_formats::FormatError> },
    /// More held-out tokens than the limit have tags the reference tagger
    /// has no outcome for ([`crate::ud::LEFT_OUT_LIMIT_PERCENT`]).
    ReferenceTags { id: String, left_out: u64, tokens: u64 },
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
            Error::InvalidRelease { field, value, message } => {
                write!(f, "the recorded release's {field} {value:?}: {message}")
            }
            Error::InvalidCard { field, message } => write!(f, "the model card's {field}: {message}"),
            Error::OutsideCorpus { path } => write!(f, "{}: not under the corpus directory", path.display()),
            Error::NotShippable { id, reason } => write!(f, "{id} is not shippable: {reason}"),
            Error::Json { path, message } => write!(f, "{}: {message}", path.display()),
            Error::CardVersion { path, found, expected } => {
                write!(f, "{}: card version {found}, this build reads {expected}", path.display())
            }
            Error::ComponentMismatch { recipe, samples } => {
                write!(f, "{samples} samples given to the {recipe} recipe")
            }
            Error::NotRecorded => {
                f.write_str("the MASC release has no recorded digest yet; see turbo_train_recipes::masc::RELEASE")
            }
            Error::CorpusDigest { expected, found } => {
                write!(f, "the corpus directory has the digest {found}, the recorded release {expected}")
            }
            Error::ReferenceDigest { id, expected, found } => {
                write!(f, "reference model {id}: SHA-256 {found}, expected {expected}")
            }
            Error::EmptySplit { what } => write!(f, "the split leaves no {what}"),
            Error::Licence { treebank, reason } => {
                write!(f, "{treebank} is refused: {reason}; the recipes train only on CC BY 4.0 data")
            }
            Error::TreebankFile { treebank, file, expected, found } => {
                write!(f, "{treebank} {file}: {found}, the recorded file {expected}")
            }
            Error::Conllu { treebank, file, error } => write!(f, "{treebank} {file}: {error}"),
            Error::ReferenceTags { id, left_out, tokens } => write!(
                f,
                "{left_out} of the {tokens} held-out tokens have tags {id} has no outcome for, more than {}%",
                crate::ud::LEFT_OUT_LIMIT_PERCENT
            ),
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

impl From<turbo_train_components::Error> for Error {
    fn from(e: turbo_train_components::Error) -> Error {
        Error::Train(e)
    }
}

impl From<turbo_eval::model_gate::ModelGateError> for Error {
    fn from(e: turbo_eval::model_gate::ModelGateError) -> Error {
        Error::Gate(e)
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
