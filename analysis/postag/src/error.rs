//! Why a part-of-speech model could not be opened.

use std::fmt;

/// A model file that is not a usable `POSTaggerME` model. Each variant
/// names what was wrong.
#[derive(Debug)]
pub enum ModelError {
    /// The file or one of its entries could not be read.
    File(turbo_maxent::Error),
    /// The descriptor, a resource, the tag dictionary or the beam size
    /// could not be read as OpenNLP reads it.
    Sequence(turbo_sequence::Error),
    /// The manifest's `Component-Name` is not `POSTaggerME`.
    Component(String),
    /// The manifest names a factory other than `POSTaggerFactory`. A
    /// subclass can change the context generator, the tag dictionary and
    /// the validator, which this port does not reproduce.
    Factory(String),
    /// A classifier outcome holds half of a surrogate pair, which no tag
    /// set label can hold.
    OutcomeNotText(turbo_maxent::JavaString),
    /// The classifier has no outcome, so nothing could be tagged.
    NoOutcomes,
    /// A beam size of zero: `BeamSearch` keeps no sequence then and
    /// `POSTaggerME.tag` returns no tags, which no caller can use.
    ZeroBeam,
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::File(e) => write!(f, "part-of-speech model: {e}"),
            ModelError::Sequence(e) => write!(f, "part-of-speech model: {e}"),
            ModelError::Component(c) => write!(f, "part-of-speech model: component {c:?} is not POSTaggerME"),
            ModelError::Factory(c) => {
                write!(f, "part-of-speech model: factory {c:?} is not opennlp.tools.postag.POSTaggerFactory")
            }
            ModelError::NoOutcomes => write!(f, "part-of-speech model: the classifier has no outcome"),
            ModelError::OutcomeNotText(o) => {
                write!(f, "part-of-speech model: the outcome {o:?} holds half of a surrogate pair")
            }
            ModelError::ZeroBeam => write!(f, "part-of-speech model: a beam size of zero tags nothing"),
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
