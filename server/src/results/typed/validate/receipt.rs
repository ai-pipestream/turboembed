//! Validation of the processing receipt and the status presence rules
//! (issue #1908) on a `TypedDocumentResult`: every layer and view status is
//! OK, SKIPPED or FAILED, never UNSPECIFIED, and carries a known error
//! exactly when it is not OK; the receipt is present, names the envelope's
//! own input and result, and every layer entry in it is well formed and
//! agrees with its status and the document status.

use super::bad;
use crate::api::Failure;
use crate::proto::result::{
    LayerError, LayerErrorKind, LayerReceipt, LayerWireStatus, ProcessingReceipt, ProcessingStatus, TruncationReason,
    TypedDocumentResult,
};
use crate::results::receipt::{
    Operation, RECEIPT_SCHEMA_VERSION, check_operation_key, covers_whole, document_status, normalize_ranges,
};

fn check_error(what: &str, error: Option<&LayerError>) -> Result<(), Failure> {
    let Some(e) = error else {
        return Err(bad(format!("{what} is not OK and carries no error")));
    };
    match LayerErrorKind::try_from(e.kind) {
        Ok(LayerErrorKind::Unspecified) | Err(_) => Err(bad(format!("{what} has error kind {}", e.kind))),
        Ok(_) => Ok(()),
    }
}

/// Refuses a status that is UNSPECIFIED or unknown, an OK status with an
/// error, and a SKIPPED or FAILED one without a known error kind.
pub(super) fn check_status(what: &str, status: i32, error: Option<&LayerError>) -> Result<(), Failure> {
    match LayerWireStatus::try_from(status) {
        Ok(LayerWireStatus::Ok) if error.is_some() => Err(bad(format!("{what} is OK and carries an error"))),
        Ok(LayerWireStatus::Ok) => Ok(()),
        Ok(LayerWireStatus::Skipped | LayerWireStatus::Failed) => check_error(what, error),
        Ok(LayerWireStatus::Unspecified) | Err(_) => Err(bad(format!("{what} has status {status}"))),
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
    if normalize_ranges(&l.assessed, source_len) != l.assessed {
        return Err(bad(format!("{what} has assessed ranges that are not sorted, disjoint and in bounds")));
    }
    for t in &l.truncations {
        if matches!(TruncationReason::try_from(*t), Ok(TruncationReason::Unspecified) | Err(_)) {
            return Err(bad(format!("{what} has truncation reason {t}")));
        }
    }
    if l.provenance_digest.as_ref().is_some_and(|d| d.len() != 32) {
        return Err(bad(format!("{what} has a provenance_digest that is not 32 bytes")));
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

/// Refuses a missing or malformed receipt, or one that names another
/// input or result than the envelope it is on.
pub(super) fn check_receipt(msg: &TypedDocumentResult) -> Result<(), Failure> {
    let Some(r): Option<&ProcessingReceipt> = msg.receipt.as_ref() else {
        return Err(bad("the typed document carries no processing receipt"));
    };
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
    if r.input != msg.input {
        return Err(bad("receipt.input is not the envelope's input identity"));
    }
    match status {
        ProcessingStatus::Cancelled if r.result.is_some() => {
            return Err(bad("a cancelled receipt names a result"));
        }
        ProcessingStatus::Cancelled => {}
        _ if r.result != msg.result => return Err(bad("receipt.result is not the envelope's result identity")),
        _ => {}
    }
    let source_len = r.input.as_ref().map_or(0, |i| i.source_text_len);
    for l in &r.layers {
        check_layer(l, source_len)?;
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
    Ok(())
}
