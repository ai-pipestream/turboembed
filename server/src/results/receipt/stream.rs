//! The consumer's side of a terminal receipt (issue #1908): a progressive
//! stream is complete only when its last event is a receipt bound to the
//! caller's operation and, when the stream delivered a complete graph,
//! to exactly those graph bytes. Anything else, a dropped connection
//! above all, is incomplete, whatever the stream delivered before.

use crate::proto::result::analyze_progressive_event::Event;
use crate::proto::result::{AnalyzeProgressiveEvent, ProcessingReceipt};
use crate::results::typed::identity::verify_result;

/// Why a stream is not complete.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Incomplete {
    /// The stream ended without a terminal receipt.
    NoTerminalReceipt,
    /// An event follows the receipt.
    EventsAfterReceipt,
    /// The receipt names another operation key.
    OtherOperation,
    /// The receipt's result is not the identity of the delivered graph.
    OtherResult,
}

/// The terminal receipt of `events`, read in order, for the operation
/// under `operation_key`, or why there is none.
pub fn terminal_receipt<'a>(
    events: &'a [AnalyzeProgressiveEvent],
    operation_key: &str,
) -> Result<&'a ProcessingReceipt, Incomplete> {
    let at =
        events.iter().position(|e| matches!(e.event, Some(Event::Receipt(_)))).ok_or(Incomplete::NoTerminalReceipt)?;
    if at + 1 != events.len() {
        return Err(Incomplete::EventsAfterReceipt);
    }
    let Some(Event::Receipt(receipt)) = &events[at].event else { unreachable!("found above") };
    if receipt.operation_key != operation_key {
        return Err(Incomplete::OtherOperation);
    }
    let complete = events[..at].iter().rev().find_map(|e| match &e.event {
        Some(Event::Complete(c)) => Some(c),
        _ => None,
    });
    if let Some(complete) = complete
        && verify_result(receipt.result.as_ref(), &complete.graph).is_err()
    {
        return Err(Incomplete::OtherResult);
    }
    Ok(receipt)
}
