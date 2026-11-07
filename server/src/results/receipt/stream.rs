//! The consumer's side of a terminal receipt (issue #1908): a progressive
//! stream is complete only when its last event is a whole receipt
//! (`receipt::check`) bound to the caller's operation and exact input and,
//! when the stream delivered a complete graph, to exactly those graph
//! bytes. A receipt after no graph must name no result and be CANCELLED or
//! FAILED. Anything else, a dropped connection above all, is incomplete,
//! whatever the stream delivered before.

use turbo_analysis::ViewTextDigest;

use super::check::check_receipt_shape;
use crate::proto::result::analyze_progressive_event::Event;
use crate::proto::result::{AnalyzeProgressiveEvent, ProcessingReceipt, ProcessingStatus};
use crate::results::typed::identity::verify_result;

/// Why a stream is not complete.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Incomplete {
    /// The stream ended without a terminal receipt.
    NoTerminalReceipt,
    /// An event follows the receipt.
    EventsAfterReceipt,
    /// The receipt is not whole or not self-consistent.
    Malformed,
    /// The receipt names another operation key.
    OtherOperation,
    /// The receipt names another input than the text the caller sent.
    OtherInput,
    /// The receipt's result is not the identity of the delivered graph, or
    /// it names a result the stream never delivered.
    OtherResult,
    /// The receipt claims an outcome with a result but no graph arrived.
    NoGraph,
}

/// The terminal receipt of `events`, read in order, for the operation
/// under `operation_key` over `source_text` (the source view text the
/// caller sent, or that its bytes or graph decode to), or why there is none.
pub fn terminal_receipt<'a>(
    events: &'a [AnalyzeProgressiveEvent],
    operation_key: &str,
    source_text: &str,
) -> Result<&'a ProcessingReceipt, Incomplete> {
    let at =
        events.iter().position(|e| matches!(e.event, Some(Event::Receipt(_)))).ok_or(Incomplete::NoTerminalReceipt)?;
    if at + 1 != events.len() {
        return Err(Incomplete::EventsAfterReceipt);
    }
    let Some(Event::Receipt(receipt)) = &events[at].event else { unreachable!("found above") };
    let status = check_receipt_shape(receipt).map_err(|_| Incomplete::Malformed)?;
    if receipt.operation_key != operation_key {
        return Err(Incomplete::OtherOperation);
    }
    let input = receipt.input.as_ref().ok_or(Incomplete::Malformed)?;
    if input.source_text_len as usize != source_text.len()
        || input.source_text_digest != ViewTextDigest::of_text(source_text).as_bytes()
    {
        return Err(Incomplete::OtherInput);
    }
    let complete = events[..at].iter().rev().find_map(|e| match &e.event {
        Some(Event::Complete(c)) => Some(c),
        _ => None,
    });
    match complete {
        Some(c) => verify_result(receipt.result.as_ref(), &c.graph).map_err(|_| Incomplete::OtherResult)?,
        None if receipt.result.is_some() => return Err(Incomplete::OtherResult),
        None if !matches!(status, ProcessingStatus::Cancelled | ProcessingStatus::Failed) => {
            return Err(Incomplete::NoGraph);
        }
        None => {}
    }
    Ok(receipt)
}
