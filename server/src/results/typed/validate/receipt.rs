//! Validation of the status presence rules and the processing receipt on
//! a `TypedDocumentResult` (issue #1908): every layer and view status is
//! OK, SKIPPED or FAILED, never UNSPECIFIED, and carries a known error
//! with its own turbo code exactly when it is not OK. The receipt is
//! present and whole (`receipt::check`), names the envelope's own input
//! and result, and lists exactly the message's views and layers, each
//! with the envelope's provenance digest for it.

use super::bad;
use crate::api::Failure;
use crate::proto::result::{LayerError, LayerWireStatus, ProcessingStatus, TypedDocumentResult};
use crate::results::receipt::check::{check_error, check_receipt_shape};

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

/// Refuses a missing or malformed receipt, or one that names another
/// input, result or set of layers than the envelope it is on.
pub(super) fn check_receipt(msg: &TypedDocumentResult) -> Result<(), Failure> {
    let Some(r) = msg.receipt.as_ref() else {
        return Err(bad("the typed document carries no processing receipt"));
    };
    let status = check_receipt_shape(r)?;
    if r.input != msg.input {
        return Err(bad("receipt.input is not the envelope's input identity"));
    }
    if status != ProcessingStatus::Cancelled && r.result != msg.result {
        return Err(bad("receipt.result is not the envelope's result identity"));
    }
    // Every view and layer instance on the message has one provenance entry
    // (checked with the envelope); the receipt names each once, with that
    // entry's provenance digest.
    if r.layers.len() != msg.provenance.len() {
        return Err(bad(format!(
            "the receipt lists {} layers and the message has {} views and layers",
            r.layers.len(),
            msg.provenance.len()
        )));
    }
    for l in &r.layers {
        let Some(p) = msg.provenance.iter().find(|p| p.layer == l.layer && p.instance == l.instance) else {
            return Err(bad(format!("receipt layer {:?} instance {:?} is not on this message", l.layer, l.instance)));
        };
        if l.provenance_digest.as_deref().unwrap_or_default() != p.provenance_digest.as_slice() {
            return Err(bad(format!(
                "receipt layer {:?} instance {:?} has another provenance_digest than the envelope's entry",
                l.layer, l.instance
            )));
        }
    }
    Ok(())
}
