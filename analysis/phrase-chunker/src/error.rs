//! Why a chunker model could not be opened, or chunk tags could not be
//! read as phrases.

use std::fmt;

/// A model file that is not a usable `ChunkerME` model. Each variant names
/// what was wrong.
#[derive(Debug)]
pub enum ModelError {
    /// The file or one of its entries could not be read.
    File(turbo_maxent::Error),
    /// The beam size could not be read as OpenNLP reads it.
    Sequence(turbo_sequence::Error),
    /// The manifest's `Component-Name` is not `ChunkerME`.
    Component(String),
    /// The manifest names a factory other than `ChunkerFactory`. A
    /// subclass can change the context generator and the validator, which
    /// this port does not reproduce.
    Factory(String),
    /// The classifier has no outcome, so nothing could be chunked.
    NoOutcomes,
    /// A beam size of zero: `BeamSearch` keeps no sequence then and
    /// `ChunkerME.chunk` returns no tags, which no caller can use.
    ZeroBeam,
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::File(e) => write!(f, "chunker model: {e}"),
            ModelError::Sequence(e) => write!(f, "chunker model: {e}"),
            ModelError::Component(c) => write!(f, "chunker model: component {c:?} is not ChunkerME"),
            ModelError::Factory(c) => {
                write!(f, "chunker model: factory {c:?} is not opennlp.tools.chunker.ChunkerFactory")
            }
            ModelError::NoOutcomes => write!(f, "chunker model: the classifier has no outcome"),
            ModelError::ZeroBeam => write!(f, "chunker model: a beam size of zero chunks nothing"),
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

/// A chunk tag `ChunkSample.phrasesAsSpanList` would throw on when it
/// starts a phrase: it takes the tag's `substring(2)` as the phrase's
/// kind, which throws for a tag of fewer than two UTF-16 units
/// (`StringIndexOutOfBoundsException`). A tag whose second unit is the
/// first half of a surrogate pair gives a kind no string here can hold,
/// and is refused the same way. The tags of a trained chunker model are
/// `B-<kind>`, `I-<kind>` and `O`, which never do this.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhraseError {
    /// The tag, as the classifier wrote it.
    pub tag: String,
    /// The index of the token it was the tag of.
    pub index: usize,
}

impl fmt::Display for PhraseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "chunk tag {:?} at token {} has no kind after its first two units", self.tag, self.index)
    }
}

impl std::error::Error for PhraseError {}
