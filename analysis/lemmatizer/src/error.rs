//! Why a lemmatizer model or dictionary could not be opened.

use std::fmt;

/// A model file that is not a usable `LemmatizerME` model. Each variant
/// names what was wrong.
#[derive(Debug)]
pub enum ModelError {
    /// The file or one of its entries could not be read.
    File(turbo_maxent::Error),
    /// The beam size could not be read as OpenNLP reads it.
    Sequence(turbo_sequence::Error),
    /// The manifest's `Component-Name` is not `StatisticalLemmatizer`.
    Component(String),
    /// The manifest names a factory other than `LemmatizerFactory`. A
    /// subclass can change the context generator and the validator, which
    /// this port does not reproduce.
    Factory(String),
    /// The classifier has no outcome, so nothing could be lemmatized.
    NoOutcomes,
    /// A beam size of zero: `BeamSearch` keeps no sequence then and
    /// `LemmatizerME.lemmatize` returns no lemmas, which no caller can use.
    ZeroBeam,
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::File(e) => write!(f, "lemmatizer model: {e}"),
            ModelError::Sequence(e) => write!(f, "lemmatizer model: {e}"),
            ModelError::Component(c) => write!(f, "lemmatizer model: component {c:?} is not StatisticalLemmatizer"),
            ModelError::Factory(c) => {
                write!(f, "lemmatizer model: factory {c:?} is not opennlp.tools.lemmatizer.LemmatizerFactory")
            }
            ModelError::NoOutcomes => write!(f, "lemmatizer model: the classifier has no outcome"),
            ModelError::ZeroBeam => write!(f, "lemmatizer model: a beam size of zero lemmatizes nothing"),
        }
    }
}

impl std::error::Error for ModelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ModelError::File(e) => Some(e),
            ModelError::Sequence(e) => Some(e),
            _ => None,
        }
    }
}

impl From<turbo_maxent::Error> for ModelError {
    fn from(e: turbo_maxent::Error) -> ModelError {
        ModelError::File(e)
    }
}

impl From<turbo_sequence::Error> for ModelError {
    fn from(e: turbo_sequence::Error) -> ModelError {
        ModelError::Sequence(e)
    }
}

/// A line of a lemmatizer dictionary `DictionaryLemmatizer` would throw
/// on: fewer than three tab separated fields after Java's `split` drops
/// the trailing empty ones (an `ArrayIndexOutOfBoundsException` there).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DictionaryError {
    /// The line, counted from 1.
    pub line: usize,
    pub fields: usize,
}

impl fmt::Display for DictionaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "lemmatizer dictionary line {}: {} fields, not word, tag and lemma", self.line, self.fields)
    }
}

impl std::error::Error for DictionaryError {}

/// A lemma class `decodeShortestEditScript` would throw on. The classes a
/// model was trained with never do: `getShortestEditScript` writes a
/// digit where one is read, and every operation has its characters. A
/// class from elsewhere can.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditScriptError {
    /// An operation's index is not a decimal digit
    /// (`NumberFormatException`).
    NotADigit { unit: u16 },
    /// An operation ends before its characters
    /// (`StringIndexOutOfBoundsException`).
    Truncated,
}

impl fmt::Display for EditScriptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EditScriptError::NotADigit { unit } => {
                write!(f, "lemma class: the index U+{unit:04X} is not a decimal digit")
            }
            EditScriptError::Truncated => write!(f, "lemma class: an operation ends before its characters"),
        }
    }
}

impl std::error::Error for EditScriptError {}
