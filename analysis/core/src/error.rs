use std::fmt;

use crate::bytes::SourceError;
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
    /// An annotator's input over one of its limits: a static limit name
    /// and two counts, never document text.
    OverLimit { annotator: &'static str, limit: &'static str, max: usize, got: usize },
    /// Not a failure: the annotator declined this document, for this
    /// reason. Its outputs are recorded as skipped, so "skipped" is never
    /// mistaken for "found nothing", and steps that need them are skipped
    /// in turn.
    Skipped(String),
    /// A UTF-16 or code point offset from outside (Java, the wire) that
    /// names no place in the view: `at`, in `unit`, and why.
    Offset { unit: OffsetUnit, at: u32, reason: OffsetError },
    /// Source bytes refused when a document is decoded from them: too long
    /// for `u32` offsets, or not what the alignment says the text came
    /// from.
    Source(SourceError),
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
            Error::OverLimit { annotator, limit, max, got } => {
                write!(f, "{annotator}: over its {limit} {max}, got {got}")
            }
            Error::Skipped(why) => write!(f, "skipped: {why}"),
            Error::Offset { unit, at, reason: OffsetError::PastEnd { len } } => {
                write!(f, "{unit} {at} is past the end of the view, which is {len} long")
            }
            Error::Offset { unit, at, reason: OffsetError::InsideCharacter } => {
                write!(f, "{unit} {at} falls between the two halves of a surrogate pair")
            }
            Error::Source(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for Error {}

impl Error {
    /// An annotator's input over one of its limits: a static limit name
    /// and two counts, never document text, so a server may pass it on.
    pub fn over_limit(annotator: &'static str, limit: &'static str, max: usize, got: usize) -> Error {
        debug_assert!(known_limit(limit).is_some(), "unlisted limit name {limit:?}; add it to error::known_limit");
        Error::OverLimit { annotator, limit, max, got }
    }
}

/// The static limit name matching `name`, if `name` is one of the known
/// annotator limit names.
///
/// An imported [`ErrorRecord::OverLimit`](crate::export::ErrorRecord::OverLimit)
/// carries a wire string; matching against this closed table avoids
/// interning or publishing caller-supplied text as a limit name.
pub fn known_limit(name: &str) -> Option<&'static str> {
    match name {
        "encoder max_sequence" => Some("encoder max_sequence"),
        "glirel max_len" => Some("glirel max_len"),
        "max_classification_tasks" => Some("max_classification_tasks"),
        "max_entity_types" => Some("max_entity_types"),
        "max_fields_per_structure" => Some("max_fields_per_structure"),
        "max_labels" => Some("max_labels"),
        "max_labels_per_task" => Some("max_labels_per_task"),
        "max_mentions" => Some("max_mentions"),
        "max_pairs" => Some("max_pairs"),
        "max_prompt_subwords" => Some("max_prompt_subwords"),
        "max_structures" => Some("max_structures"),
        "max_text_bytes" => Some("max_text_bytes"),
        "max_tokens" => Some("max_tokens"),
        "max_token_bytes" => Some("max_token_bytes"),
        "max_words" => Some("max_words"),
        "scorer max_labels" => Some("scorer max_labels"),
        "scorer max_mentions" => Some("scorer max_mentions"),
        "tag set" => Some("tag set"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_over_limit_error_carries_its_limit_and_counts() {
        let e = Error::over_limit("gliner2", "encoder max_sequence", 512, 528);
        assert_eq!(e, Error::OverLimit { annotator: "gliner2", limit: "encoder max_sequence", max: 512, got: 528 });
        assert_eq!(e.to_string(), "gliner2: over its encoder max_sequence 512, got 528");
        assert_eq!(known_limit("encoder max_sequence"), Some("encoder max_sequence"));
        assert_eq!(known_limit("patient jane doe"), None);
    }
}
