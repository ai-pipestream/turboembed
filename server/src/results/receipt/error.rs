//! Sanitized layer errors (issue #1908). A layer that is not OK says why
//! as a [`LayerError`]: a kind, the turbo status a client routes it by,
//! and the code names involved. Detail strings, skip reasons, view text
//! and debug renderings never reach the wire: an annotator's detail or a
//! bounds message can quote the document.

use turbo_analysis::error::limit_detail;
use turbo_analysis::export::{ErrorRecord, LayerRef, StatusRecord};
use turbo_analysis::{Document, Error, KeyId, Status};

use crate::api::{TURBO_E_CAPACITY, TURBO_E_INTERNAL, TURBO_E_INVALID_ARGUMENT, TURBO_E_INVALID_STATE, TURBO_E_PANIC};
use crate::proto::result::{LayerError, LayerErrorKind, LayerInstanceRef, LayerWireStatus};

fn key_ref(key: KeyId) -> LayerInstanceRef {
    LayerInstanceRef { layer: key.layer.to_string(), instance: key.name.to_owned() }
}

fn record_ref(r: &LayerRef) -> LayerInstanceRef {
    LayerInstanceRef { layer: r.layer.clone(), instance: r.instance.clone() }
}

fn make(kind: LayerErrorKind, layer: Option<LayerInstanceRef>, annotator: &str) -> LayerError {
    LayerError { kind: kind as i32, turbo_code: kind_code(kind), layer, annotator: annotator.to_owned() }
}

/// The turbo status each kind is routed by: 0 for the two skip kinds.
pub fn kind_code(kind: LayerErrorKind) -> i32 {
    use LayerErrorKind as K;
    match kind {
        K::Unspecified | K::Declined | K::Upstream => 0,
        K::Annotator => TURBO_E_INTERNAL,
        K::Limit => TURBO_E_CAPACITY,
        K::Panic => TURBO_E_PANIC,
        K::Missing | K::Duplicate | K::Clash | K::Misaligned | K::Unaligned | K::Pipeline => TURBO_E_INVALID_STATE,
        K::Bounds | K::Invalid | K::Offset | K::Source => TURBO_E_INVALID_ARGUMENT,
    }
}

/// The kind name a step failure's message carries: a code name, never
/// the error's own text.
pub fn kind_name(kind: LayerErrorKind) -> &'static str {
    use LayerErrorKind as K;
    match kind {
        K::Unspecified => "unspecified",
        K::Declined => "declined",
        K::Upstream => "upstream not complete",
        K::Missing => "missing layer",
        K::Duplicate => "duplicate layer",
        K::Clash => "layer id clash",
        K::Bounds => "span out of bounds",
        K::Misaligned => "misaligned layer",
        K::Unaligned => "unaligned layer",
        K::Invalid => "invalid layer",
        K::Pipeline => "pipeline",
        K::Annotator => "annotator failure",
        K::Panic => "annotator panic",
        K::Offset => "offset",
        K::Source => "source bytes",
        K::Limit => "over a limit",
    }
}

fn panicked(detail: &str) -> bool {
    detail.starts_with("panic:")
}

/// The sanitized form of a live analysis error.
pub fn of_error(e: &Error) -> LayerError {
    use LayerErrorKind as K;
    match e {
        Error::Missing(k) => make(K::Missing, Some(key_ref(*k)), ""),
        Error::Duplicate(k) => make(K::Duplicate, Some(key_ref(*k)), ""),
        Error::Clash(k) => make(K::Clash, Some(key_ref(*k)), ""),
        Error::Bounds { layer, .. } => make(K::Bounds, Some(key_ref(*layer)), ""),
        Error::Misaligned { layer, .. } => make(K::Misaligned, Some(key_ref(*layer)), ""),
        Error::Unaligned { layer, .. } => make(K::Unaligned, Some(key_ref(*layer)), ""),
        Error::Invalid { layer, .. } => make(K::Invalid, Some(key_ref(*layer)), ""),
        Error::Pipeline(_) => make(K::Pipeline, None, ""),
        Error::Annotator { annotator, detail } if panicked(detail) => make(K::Panic, None, annotator),
        Error::Annotator { annotator, detail } if limit_detail(detail).is_some() => make(K::Limit, None, annotator),
        Error::Annotator { annotator, .. } => make(K::Annotator, None, annotator),
        Error::Skipped(_) => make(K::Declined, None, ""),
        Error::Offset { .. } => make(K::Offset, None, ""),
        Error::Source(_) => make(K::Source, None, ""),
    }
}

/// The sanitized form of an exported error record.
pub fn of_record(e: &ErrorRecord) -> LayerError {
    use LayerErrorKind as K;
    match e {
        ErrorRecord::Missing(r) => make(K::Missing, Some(record_ref(r)), ""),
        ErrorRecord::Duplicate(r) => make(K::Duplicate, Some(record_ref(r)), ""),
        ErrorRecord::Clash(r) => make(K::Clash, Some(record_ref(r)), ""),
        ErrorRecord::Bounds { layer, .. } => make(K::Bounds, Some(record_ref(layer)), ""),
        ErrorRecord::Misaligned { layer, .. } => make(K::Misaligned, Some(record_ref(layer)), ""),
        ErrorRecord::Unaligned { layer, .. } => make(K::Unaligned, Some(record_ref(layer)), ""),
        ErrorRecord::Invalid { layer, .. } => make(K::Invalid, Some(record_ref(layer)), ""),
        ErrorRecord::Pipeline(_) => make(K::Pipeline, None, ""),
        ErrorRecord::Annotator { annotator, detail } if panicked(detail) => make(K::Panic, None, annotator),
        ErrorRecord::Annotator { annotator, detail } if limit_detail(detail).is_some() => {
            make(K::Limit, None, annotator)
        }
        ErrorRecord::Annotator { annotator, .. } => make(K::Annotator, None, annotator),
        ErrorRecord::Skipped(_) => make(K::Declined, None, ""),
        ErrorRecord::Offset { .. } => make(K::Offset, None, ""),
        ErrorRecord::Source(_) => make(K::Source, None, ""),
    }
}

/// A skip: UPSTREAM when a layer it read is not complete, else DECLINED.
pub fn skip(upstream: bool) -> LayerError {
    make(if upstream { LayerErrorKind::Upstream } else { LayerErrorKind::Declined }, None, "")
}

/// The wire status and sanitized error of `key` on `doc`. A layer with no
/// status was never written; the projection lists only written layers,
/// so that is reported as UNSPECIFIED, never as OK.
pub fn layer_status(doc: &Document, key: KeyId) -> (LayerWireStatus, Option<LayerError>) {
    match doc.status(key) {
        Some(Status::Complete) => (LayerWireStatus::Ok, None),
        Some(Status::Skipped(_)) => {
            let upstream = doc
                .provenance(key)
                .is_some_and(|p| p.reads.iter().any(|r| !matches!(doc.status(*r), Some(Status::Complete))));
            (LayerWireStatus::Skipped, Some(skip(upstream)))
        }
        Some(Status::Failed(e)) => (LayerWireStatus::Failed, Some(of_error(e))),
        None => (LayerWireStatus::Unspecified, None),
    }
}

/// The wire status and sanitized error of an exported status record;
/// `complete` says whether a read layer is complete in the same graph.
pub fn record_status(
    status: &StatusRecord,
    reads: &[LayerRef],
    complete: impl Fn(&LayerRef) -> bool,
) -> (LayerWireStatus, Option<LayerError>) {
    match status {
        StatusRecord::Complete => (LayerWireStatus::Ok, None),
        StatusRecord::Skipped(_) => (LayerWireStatus::Skipped, Some(skip(reads.iter().any(|r| !complete(r))))),
        StatusRecord::Failed(e) => (LayerWireStatus::Failed, Some(of_record(e))),
    }
}
