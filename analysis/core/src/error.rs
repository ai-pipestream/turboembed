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

/// How an [`Error::Annotator`] detail written by [`Error::over_limit`]
/// begins.
pub const LIMIT_PREFIX: &str = "limit: ";

impl Error {
    /// An annotator's input over one of its limits: a static limit name
    /// and two counts, never document text, so a server may pass it on.
    pub fn over_limit(annotator: &'static str, limit: &'static str, max: usize, got: usize) -> Error {
        Error::Annotator { annotator, detail: format!("{LIMIT_PREFIX}{limit} {max}, got {got}") }
    }

    /// The limit and counts of an [`Error::over_limit`] error, or `None`
    /// for any other. A detail that has the prefix but not the exact
    /// shape [`limit_detail`] reads is not one, so no other detail is
    /// mistaken for it.
    pub fn limit_detail(&self) -> Option<&str> {
        let Error::Annotator { detail, .. } = self else { return None };
        limit_detail(detail)
    }
}

/// [`Error::limit_detail`] on an annotator detail string, such as the one
/// an exported error record carries. Only the exact shape
/// [`Error::over_limit`] writes reads back: a name of lower case ASCII
/// letters, digits, spaces and `_`, then two counts.
pub fn limit_detail(detail: &str) -> Option<&str> {
    let rest = detail.strip_prefix(LIMIT_PREFIX)?;
    let count = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    let (head, got) = rest.rsplit_once(", got ")?;
    let (name, max) = head.rsplit_once(' ')?;
    let plain = |b: u8| b.is_ascii_lowercase() || b.is_ascii_digit() || b" _".contains(&b);
    let named = !name.trim().is_empty() && name.bytes().all(plain);
    (named && count(max) && count(got)).then_some(rest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_over_limit_error_reads_back_and_nothing_else_does() {
        let e = Error::over_limit("gliner2", "encoder max_sequence", 512, 528);
        assert_eq!(e.limit_detail(), Some("encoder max_sequence 512, got 528"));
        assert_eq!(e.to_string(), "gliner2: limit: encoder max_sequence 512, got 528");
        let glirel = Error::over_limit("glirel", "glirel max_len", 384, 400);
        assert_eq!(glirel.limit_detail(), Some("glirel max_len 384, got 400"));
        for detail in [
            "limit: ",
            "limit: Jane Doe 3",
            "limit: a\nb",
            "panic: x",
            "over 3",
            "limit: some lower case words here 3",
            "limit: a 1, got x",
            "limit:  1, got 2",
        ] {
            let e = Error::Annotator { annotator: "x", detail: detail.to_owned() };
            assert_eq!(e.limit_detail(), None, "{detail:?}");
        }
        assert_eq!(Error::Pipeline("limit: a 1, got 2".to_owned()).limit_detail(), None);
    }
}
