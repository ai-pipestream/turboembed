//! The terminal receipt of a run that a live deadline cancels (#2405,
//! #2408): a real `grpc-timeout` timer fires while the first step is in
//! flight, the PII step never starts, and the stream still ends with a
//! CANCELLED receipt before the deadline status. A pipeline that plans
//! PII findings names its input by length alone on that receipt, although
//! the document holds no `turbo:pii` layer yet: the text is the same
//! before the PII step runs as after. With `full_text`, or with no PII
//! step planned, the receipt names the input by digest as before.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use prost::Message;
use tokio::sync::mpsc;
use tonic::{Code, Status};
use turbo_analysis::{Analyzer, Annotator, Document, Error, Refs, View, ViewTextDigest};
use turbo_analysis_layers::{Numbers, Pii};

use super::{Answer, run_progressive_task};
use crate::deadline::Deadline;
use crate::offsets::OffsetMode;
use crate::proto::result::analyze_progressive_event::Event;
use crate::proto::result::{AnalyzeProgressiveEvent, ProcessingReceipt, ProcessingStatus};
use crate::results::disclosure::{Disclosure, plans_pii};
use crate::results::receipt::Operation;
use crate::results::receipt::stream::terminal_receipt;

/// A source short enough to guess: the whole text is the address.
const TEXT: &str = "alice.planted@example.com";
const KEY: &str = "batch-9/doc-3";

/// The step in flight when the deadline fires: it returns only once the
/// request's own deadline has cancelled the run, however long that takes.
struct InFlight(Arc<AtomicBool>);

impl Annotator for InFlight {
    const ID: &'static str = "test:in-flight";
    type Reads = (View,);
    type Writes = (Numbers,);

    fn annotate(&self, (v,): Refs<'_, Self>) -> Result<(Numbers,), Error> {
        while !self.0.load(Ordering::Acquire) {
            std::thread::sleep(Duration::from_millis(1));
        }
        Ok((Numbers::new(v.id()),))
    }
}

/// A PII step that needs the in-flight step's layer, so a cancelled run
/// never starts it.
struct PiiAfter;

impl Annotator for PiiAfter {
    const ID: &'static str = "test:pii-after";
    type Reads = (View, Numbers);
    type Writes = (Pii,);

    fn annotate(&self, (v, _): Refs<'_, Self>) -> Result<(Pii,), Error> {
        Ok((Pii::new(v.id()),))
    }
}

/// Everything the stream of a run over TEXT carried, in order, when its
/// deadline fires while the first step is in flight. `pii` adds the PII
/// step behind that one.
fn cancelled_by_its_deadline(pii: bool, disclosure: Disclosure) -> Vec<Result<AnalyzeProgressiveEvent, Status>> {
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(1).enable_time().build().unwrap();
    // The timer of a deadline is a task on the runtime it is made in.
    let deadline = {
        let _in_runtime = rt.enter();
        Deadline::with_timeout(Duration::from_millis(500))
    };
    let cancel = Arc::clone(deadline.cancel_flag());
    let builder = Analyzer::builder().add(InFlight(Arc::clone(&cancel)));
    let analyzer = if pii { builder.add(PiiAfter) } else { builder }.build().unwrap();
    assert_eq!(plans_pii(&analyzer), pii);
    let mut op = Operation::start(KEY);
    op.requested_steps = analyzer.step_ids().into_iter().map(str::to_owned).collect();
    let answer = Answer {
        doc_id: "d".into(),
        offsets: OffsetMode::Utf8.to_proto(),
        disclosure,
        binding: None,
        typed: None,
        metrics: crate::metrics::Metrics::new(),
    };
    let (tx, mut rx) = mpsc::channel(32);
    assert!(!cancel.load(Ordering::Acquire), "the deadline is live when the run starts");
    let delivered = run_progressive_task(answer, Document::new(TEXT.to_owned()), analyzer, cancel, deadline, tx, op);
    assert!(delivered.is_none(), "a cancelled run hands nothing to the sink");
    let mut stream = Vec::new();
    while let Some(item) = rx.blocking_recv() {
        stream.push(item);
    }
    stream
}

/// The events before the deadline status, and the receipt they end with.
fn events_and_receipt(
    stream: &[Result<AnalyzeProgressiveEvent, Status>],
) -> (Vec<AnalyzeProgressiveEvent>, ProcessingReceipt) {
    let (last, before) = stream.split_last().expect("the stream carried something");
    assert_eq!(last.as_ref().unwrap_err().code(), Code::DeadlineExceeded, "the stream ends with the deadline status");
    let events: Vec<AnalyzeProgressiveEvent> = before.iter().map(|e| e.clone().expect("an event")).collect();
    assert!(matches!(events[0].event, Some(Event::Started(_))), "the run started before its deadline");
    let Some(Event::Receipt(receipt)) = events.last().and_then(|e| e.event.clone()) else {
        panic!("the receipt comes before the deadline status: {events:?}")
    };
    assert_eq!(receipt.status, ProcessingStatus::Cancelled as i32);
    assert!(receipt.result.is_none(), "a cancelled run names no result");
    assert!(receipt.layers.iter().all(|l| l.layer != "turbo:pii"), "the PII step never ran: {:?}", receipt.layers);
    // The consumer's check takes it as this operation's terminal receipt.
    assert_eq!(terminal_receipt(&events, KEY, TEXT).unwrap(), &receipt);
    (events, receipt)
}

fn holds(bytes: &[u8], needle: &[u8]) -> bool {
    bytes.windows(needle.len()).any(|w| w == needle)
}

#[test]
fn a_run_cancelled_before_its_pii_step_ends_with_a_receipt_that_names_no_digest() {
    let digest = ViewTextDigest::of_text(TEXT);
    let hex: String = digest.as_bytes().iter().map(|b| format!("{b:02x}")).collect();
    for disclosure in [Disclosure::FINDINGS, Disclosure { rewritten_text: true, ..Disclosure::FINDINGS }] {
        let (events, receipt) = events_and_receipt(&cancelled_by_its_deadline(true, disclosure));
        let input = receipt.input.as_ref().unwrap();
        assert!(input.source_text_digest.is_empty() && input.raw_bytes_digest.is_empty(), "{input:?}");
        // The receipt keeps the lengths: its assessed ranges are in them.
        assert_eq!((input.raw_bytes_len, input.source_text_len), (TEXT.len() as u64, TEXT.len() as u32));
        assert!(receipt.layers.iter().all(|l| l.provenance_digest.is_none()), "{:?}", receipt.layers);
        for (i, e) in events.iter().enumerate() {
            let bytes = e.encode_to_vec();
            assert!(!holds(&bytes, digest.as_bytes()), "event {i} carries the SHA-256 of the source");
            assert!(!holds(&bytes, hex.as_bytes()), "event {i} carries the hexadecimal SHA-256 of the source");
        }
    }
}

/// The controls: the same cancelled run names its input by digest when
/// the request may have the source, and when no PII step was planned.
#[test]
fn a_cancelled_run_that_may_carry_its_source_keeps_the_input_digests() {
    let digest = ViewTextDigest::of_text(TEXT);
    for (pii, disclosure) in [(true, Disclosure::FULL), (false, Disclosure::FINDINGS)] {
        let (_, receipt) = events_and_receipt(&cancelled_by_its_deadline(pii, disclosure));
        let input = receipt.input.as_ref().unwrap();
        assert_eq!(input.source_text_digest, digest.as_bytes(), "pii step planned: {pii}");
        assert_eq!(input.raw_bytes_digest, digest.as_bytes(), "pii step planned: {pii}");
        assert_eq!(input.source_text_len as usize, TEXT.len());
    }
}
