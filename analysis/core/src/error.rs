use std::fmt;

use crate::layer::KeyId;
use crate::offsets::{OffsetError, OffsetUnit};
use crate::text::ViewId;

/// Why a layer, a step or a pipeline was refused. Every refusal names what
/// it is about.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A step needs a layer the document does not have.
    Missing(KeyId),
    /// A layer is added a second time; layers are added once.
    Duplicate(KeyId),
    /// A layer whose id another layer type already has on the document:
    /// one id names one type, or the wire would carry two meanings under
    /// one name.
    Clash(KeyId),
    /// Spans past the end of their view, off a character boundary, or in a
    /// view the document does not have.
    Bounds { layer: KeyId, view: ViewId, detail: String },
    /// An aligned layer whose length differs from its anchor's.
    Misaligned { layer: KeyId, anchor: KeyId, len: usize, anchor_len: usize },
    /// A layer read beside another instance of the layer it was made on:
    /// tags made on the gold tokens, read with the predicted ones. Item i
    /// of the one is not about item i of the other. `read` is the name of
    /// the instance the step reads.
    Unaligned { layer: KeyId, on: KeyId, read: &'static str },
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
    /// A UTF-16 or code point offset from outside (Java, the wire) that
    /// names no place in the view: `at`, in `unit`, and why.
    Offset { unit: OffsetUnit, at: u32, reason: OffsetError },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Missing(k) => write!(f, "the document has no {k} layer"),
            Error::Duplicate(k) => write!(f, "the {k} layer is already on the document"),
            Error::Clash(k) => write!(f, "{k} already names another layer type"),
            Error::Bounds { layer, view, detail } => write!(f, "{layer} in view {}: {detail}", view.0),
            Error::Misaligned { layer, anchor, len, anchor_len } => {
                write!(f, "{layer} has {len} items, aligned to {anchor} with {anchor_len}")
            }
            Error::Unaligned { layer, on, read } => {
                let read = KeyId { name: read, ..*on };
                write!(f, "{layer} was made on {on}, not on the {read} the step reads")
            }
            Error::Invalid { layer, detail } => write!(f, "{layer}: {detail}"),
            Error::Pipeline(d) => write!(f, "pipeline: {d}"),
            Error::Annotator { annotator, detail } => write!(f, "{annotator}: {detail}"),
            Error::Skipped(why) => write!(f, "skipped: {why}"),
            Error::Offset { unit, at, reason: OffsetError::PastEnd { len } } => {
                write!(f, "{unit} {at} is past the end of the view, which is {len} long")
            }
            Error::Offset { unit, at, reason: OffsetError::InsideCharacter } => {
                write!(f, "{unit} {at} falls between the two halves of a surrogate pair")
            }
        }
    }
}

impl std::error::Error for Error {}
