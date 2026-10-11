use std::fmt;

use crate::layer::KeyId;
use crate::text::ViewId;

/// Why a layer, a step or a pipeline was refused. Every refusal names what
/// it is about.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// A step needs a layer the document does not have.
    Missing(KeyId),
    /// A layer is added a second time; layers are added once.
    Duplicate(KeyId),
    /// Spans past the end of their view, off a character boundary, or in a
    /// view the document does not have.
    Bounds { layer: KeyId, view: ViewId, detail: String },
    /// An aligned layer whose length differs from its anchor's.
    Misaligned { layer: KeyId, anchor: KeyId, len: usize, anchor_len: usize },
    /// A layer's own invariants.
    Invalid { layer: KeyId, detail: String },
    /// A pipeline that cannot run: no producer, two producers, a cycle.
    Pipeline(String),
    /// An annotator's own failure on one document.
    Annotator { annotator: &'static str, detail: String },
    /// Not a failure: the annotator declined this document, for this
    /// reason. Its outputs are recorded as skipped, so "skipped" is never
    /// mistaken for "found nothing", and steps that need them are skipped
    /// in turn.
    Skipped(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Missing(k) => write!(f, "the document has no {k} layer"),
            Error::Duplicate(k) => write!(f, "the {k} layer is already on the document"),
            Error::Bounds { layer, view, detail } => write!(f, "{layer} in view {}: {detail}", view.0),
            Error::Misaligned { layer, anchor, len, anchor_len } => {
                write!(f, "{layer} has {len} items, aligned to {anchor} with {anchor_len}")
            }
            Error::Invalid { layer, detail } => write!(f, "{layer}: {detail}"),
            Error::Pipeline(d) => write!(f, "pipeline: {d}"),
            Error::Annotator { annotator, detail } => write!(f, "{annotator}: {detail}"),
            Error::Skipped(why) => write!(f, "skipped: {why}"),
        }
    }
}

impl std::error::Error for Error {}
