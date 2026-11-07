//! The receipt's own consistency (issue #1908), shared by the envelope
//! validator and the stream check: the schema version, operation key,
//! attempt id, status, config digest and input identity are present and
//! well formed; every layer entry is named once, its status, ranges,
//! truncations and error agree; the item counts sum; and the document
//! status is the one the layers and steps give, CANCELLED being its own
//! evidence. Binding to a particular envelope or stream is the caller's.

use std::collections::HashSet;

use super::error::kind_code;
use super::{Operation, RECEIPT_SCHEMA_VERSION, check_operation_key, covers_whole, document_status, normalize_ranges};
use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT};
use crate::proto::result::{
    LayerError, LayerErrorKind, LayerReceipt, ProcessingReceipt, ProcessingStatus, TruncationReason,
};

pub(crate) fn bad(message: impl Into<String>) -> Failure {
    Failure::new(TURBO_E_INVALID_ARGUMENT, 0, message)
}

/// Refuses a missing error, an UNSPECIFIED or unknown kind, and a turbo
/// code other than the kind's own.
pub fn check_error(what: &str, error: Option<&LayerError>) -> Result<(), Failure> {
    let Some(e) = error else {
        return Err(bad(format!("{what} is not OK and carries no error")));
    };
    match LayerErrorKind::try_from(e.kind) {
        Ok(LayerErrorKind::Unspecified) | Err(_) => Err(bad(format!("{what} has error kind {}", e.kind))),
        Ok(kind) if e.turbo_code != kind_code(kind) => {
            Err(bad(format!("{what} has turbo code {} for error kind {}", e.turbo_code, e.kind)))
        }
        Ok(_) => Ok(()),
    }
}

fn status_of(what: &str, status: i32) -> Result<ProcessingStatus, Failure> {
    match ProcessingStatus::try_from(status) {
        Ok(ProcessingStatus::Unspecified) | Err(_) => Err(bad(format!("{what} has status {status}"))),
        Ok(s) => Ok(s),
    }
}

fn check_layer(l: &LayerReceipt, source_len: u32) -> Result<(), Failure> {
    let what = format!("receipt layer {:?} instance {:?}", l.layer, l.instance);
    let status = status_of(&what, l.status)?;
    if l.layer.is_empty() {
        return Err(bad("a receipt layer entry names no layer"));
    }
    if l.provenance_digest.as_ref().is_some_and(|d| d.len() != 32) {
        return Err(bad(format!("{what} has a provenance_digest that is not 32 bytes")));
    }
    if normalize_ranges(&l.assessed, source_len) != l.assessed {
        return Err(bad(format!("{what} has assessed ranges that are not sorted, disjoint and in bounds")));
    }
    for t in &l.truncations {
        if matches!(TruncationReason::try_from(*t), Ok(TruncationReason::Unspecified) | Err(_)) {
            return Err(bad(format!("{what} has truncation reason {t}")));
        }
    }
    let whole = covers_whole(&l.assessed, source_len) && l.truncations.is_empty();
    match status {
        ProcessingStatus::Complete if !whole => Err(bad(format!("{what} is complete over less than its text"))),
        ProcessingStatus::Partial if whole => Err(bad(format!("{what} is partial over its whole text"))),
        ProcessingStatus::Complete | ProcessingStatus::Partial if l.error.is_some() => {
            Err(bad(format!("{what} was assessed and carries an error")))
        }
        ProcessingStatus::NotAssessed | ProcessingStatus::Failed => {
            if !l.assessed.is_empty() || !l.executed.is_empty() {
                return Err(bad(format!("{what} was not assessed and lists assessed ranges or executed detectors")));
            }
            check_error(&what, l.error.as_ref())
        }
        _ => Ok(()),
    }
}

fn check_attempt_id(id: &str) -> Result<(), Failure> {
    if id.len() != 32 || !id.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return Err(bad("receipt.attempt_id is not 32 lowercase hexadecimal digits"));
    }
    Ok(())
}

/// Refuses a receipt that is not whole and self-consistent; returns its
/// status.
pub fn check_receipt_shape(r: &ProcessingReceipt) -> Result<ProcessingStatus, Failure> {
    if r.receipt_schema_version != RECEIPT_SCHEMA_VERSION {
        return Err(bad(format!(
            "receipt_schema_version is {}, not {RECEIPT_SCHEMA_VERSION}",
            r.receipt_schema_version
        )));
    }
    check_operation_key(&r.operation_key)?;
    check_attempt_id(&r.attempt_id)?;
    let status = status_of("the receipt", r.status)?;
    if r.config_digest.len() != 32 {
        return Err(bad(format!("receipt.config_digest is {} bytes, not the 32 of a SHA-256", r.config_digest.len())));
    }
    let Some(input) = &r.input else {
        return Err(bad("the receipt names no input"));
    };
    if input.raw_bytes_digest.len() != 32 || input.source_text_digest.len() != 32 {
        return Err(bad("the receipt's input digests are not 32 bytes"));
    }
    if status == ProcessingStatus::Cancelled && r.result.is_some() {
        return Err(bad("a cancelled receipt names a result"));
    }
    let mut named = HashSet::with_capacity(r.layers.len());
    for l in &r.layers {
        if !named.insert((l.layer.as_str(), l.instance.as_str())) {
            return Err(bad(format!("the receipt lists layer {:?} instance {:?} twice", l.layer, l.instance)));
        }
        check_layer(l, input.source_text_len)?;
    }
    let items: u64 = r.layers.iter().map(|l| u64::from(l.item_count)).sum();
    if items != r.item_count {
        return Err(bad(format!("receipt.item_count is {} and its layers sum to {items}", r.item_count)));
    }
    let op = Operation {
        requested_steps: r.requested_steps.clone(),
        executed_steps: r.executed_steps.clone(),
        cancelled: status == ProcessingStatus::Cancelled,
        ..Operation::default()
    };
    if document_status(&op, &r.layers) != status {
        return Err(bad("the receipt's status is not the one its layers and steps give"));
    }
    Ok(status)
}
