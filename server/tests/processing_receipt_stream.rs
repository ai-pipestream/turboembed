//! Terminal receipts on the progressive stream (issue #1908): the stream
//! ends with a receipt bound to the operation key and the delivered graph,
//! a stream cut before it is incomplete, and a retry under the same key
//! gets a new attempt id and the same result.

mod common;

use tokio_stream::StreamExt as _;
use turbo_kserve::proto::result::analyze_progressive_event::Event;
use turbo_kserve::proto::result::{
    AnalyzePipelineRequest, AnalyzeProgressiveEvent, OffsetMode, ProcessingReceipt, ProcessingStatus,
};
use turbo_kserve::results::receipt::stream::{Incomplete, terminal_receipt};

use common::*;

const TEXT: &str = "# Heading\n\nThe receipt closes the stream. Nothing here is a finding.";

fn request(key: &str) -> AnalyzePipelineRequest {
    AnalyzePipelineRequest {
        doc_id: "doc-receipt".into(),
        text: TEXT.into(),
        offsets: OffsetMode::Utf8 as i32,
        profile: "minimal".into(),
        operation_key: key.into(),
        ..Default::default()
    }
}

async fn run(client: &mut Analysis, req: AnalyzePipelineRequest) -> Vec<AnalyzeProgressiveEvent> {
    let mut stream = client.analyze_document_progressive(req).await.unwrap().into_inner();
    let mut events = Vec::new();
    while let Some(item) = stream.next().await {
        events.push(item.unwrap());
    }
    events
}

#[tokio::test]
async fn the_stream_ends_with_a_receipt_and_a_cut_stream_is_incomplete() {
    let (server, mut client) = serve().await;
    let events = run(&mut client, request("batch-7/doc-1")).await;
    let receipt = terminal_receipt(&events, "batch-7/doc-1", TEXT).expect("a terminal receipt");
    assert_eq!(receipt.status, ProcessingStatus::Complete as i32);
    assert!(receipt.result.is_some());
    assert_eq!(receipt.input.as_ref().unwrap().source_text_len, TEXT.len() as u32);
    let Some(Event::Started(started)) = &events[0].event else { panic!("no started event") };
    assert_eq!(receipt.requested_steps.len(), started.step_ids.len());
    let mut executed = receipt.executed_steps.clone();
    let mut requested = receipt.requested_steps.clone();
    executed.sort();
    requested.sort();
    assert_eq!(executed, requested);
    assert_eq!(events.last().unwrap().sequence, events.len() as u64);

    // A client that lost the connection before the receipt holds an
    // incomplete stream, though it has the complete graph.
    let cut = &events[..events.len() - 1];
    assert!(matches!(cut.last().unwrap().event, Some(Event::Complete(_))));
    assert_eq!(terminal_receipt(cut, "batch-7/doc-1", TEXT).unwrap_err(), Incomplete::NoTerminalReceipt);
    assert_eq!(terminal_receipt(&events[..1], "batch-7/doc-1", TEXT).unwrap_err(), Incomplete::NoTerminalReceipt);
    assert_eq!(terminal_receipt(&events, "batch-7/doc-2", TEXT).unwrap_err(), Incomplete::OtherOperation);
    let mut extra = events.clone();
    extra.push(events[1].clone());
    assert_eq!(terminal_receipt(&extra, "batch-7/doc-1", TEXT).unwrap_err(), Incomplete::EventsAfterReceipt);
    let mut forged = events.clone();
    if let Some(Event::Receipt(r)) = &mut forged.last_mut().unwrap().event {
        r.result.as_mut().unwrap().content_digest[0] ^= 1;
    }
    assert_eq!(terminal_receipt(&forged, "batch-7/doc-1", TEXT).unwrap_err(), Incomplete::OtherResult);
    server.stop().await;
}

#[tokio::test]
async fn a_retry_under_the_same_key_gets_a_new_attempt_id() {
    let (server, mut client) = serve().await;
    let first = run(&mut client, request("retry-key")).await;
    let second = run(&mut client, request("retry-key")).await;
    let (a, b) =
        (terminal_receipt(&first, "retry-key", TEXT).unwrap(), terminal_receipt(&second, "retry-key", TEXT).unwrap());
    assert_ne!(a.attempt_id, b.attempt_id);
    assert_eq!(a.result, b.result, "the same input and pipeline give the same result identity");
    assert_eq!(a.input, b.input);

    let refused = client.analyze_document_progressive(request("not a key")).await.unwrap_err();
    assert_eq!(refused.code(), tonic::Code::InvalidArgument);
    assert!(!refused.message().contains("not a key"));
    server.stop().await;
}

fn receipt_event(r: ProcessingReceipt) -> AnalyzeProgressiveEvent {
    AnalyzeProgressiveEvent { sequence: 1, event: Some(Event::Receipt(r)) }
}

#[test]
fn a_default_receipt_under_an_empty_key_is_not_a_terminal_receipt() {
    let lone = [receipt_event(ProcessingReceipt::default())];
    assert_eq!(terminal_receipt(&lone, "", TEXT).unwrap_err(), Incomplete::Malformed);
    assert_eq!(terminal_receipt(&lone, "", "").unwrap_err(), Incomplete::Malformed);
}

#[tokio::test]
async fn the_stream_receipt_is_bound_to_input_and_graph_on_its_own_side() {
    let (server, mut client) = serve().await;
    // An empty key is a key like any other: echoed and matched exactly.
    let events = run(&mut client, request("")).await;
    let good = terminal_receipt(&events, "", TEXT).expect("an unkeyed stream still has its receipt").clone();
    assert_eq!(terminal_receipt(&events, "x", TEXT).unwrap_err(), Incomplete::OtherOperation);
    assert_eq!(terminal_receipt(&events, "", "other text").unwrap_err(), Incomplete::OtherInput);

    // The receipt side mutated, the graph side as delivered.
    let with = |f: &dyn Fn(&mut ProcessingReceipt)| {
        let mut e = events.clone();
        if let Some(Event::Receipt(r)) = &mut e.last_mut().unwrap().event {
            f(r);
        }
        terminal_receipt(&e, "", TEXT).map(|_| ()).unwrap_err()
    };
    assert_eq!(with(&|r| r.status = 0), Incomplete::Malformed);
    assert_eq!(with(&|r| r.receipt_schema_version = 0), Incomplete::Malformed);
    assert_eq!(with(&|r| r.attempt_id.clear()), Incomplete::Malformed);
    assert_eq!(with(&|r| r.input = None), Incomplete::Malformed);
    assert_eq!(with(&|r| r.result = None), Incomplete::OtherResult);
    assert_eq!(with(&|r| r.input.as_mut().unwrap().source_text_digest[0] ^= 1), Incomplete::OtherInput);
    // The graph side mutated, the receipt as sent.
    let mut e = events.clone();
    let at = e.len() - 2;
    if let Some(Event::Complete(c)) = &mut e[at].event {
        let mut g = c.graph.to_vec();
        g.push(0);
        c.graph = g.into();
    }
    assert_eq!(terminal_receipt(&e, "", TEXT).unwrap_err(), Incomplete::OtherResult);

    // No graph: only a CANCELLED or FAILED receipt naming no result ends
    // the stream, as a deadline does.
    let no_graph: Vec<_> = events.iter().filter(|e| !matches!(e.event, Some(Event::Complete(_)))).cloned().collect();
    assert_eq!(terminal_receipt(&no_graph, "", TEXT).unwrap_err(), Incomplete::OtherResult);
    let mut cancelled = good.clone();
    cancelled.result = None;
    cancelled.status = ProcessingStatus::Cancelled as i32;
    let mut stream = no_graph.clone();
    *stream.last_mut().unwrap() = receipt_event(cancelled.clone());
    assert_eq!(terminal_receipt(&stream, "", TEXT).unwrap().status, ProcessingStatus::Cancelled as i32);
    let mut complete_no_result = good;
    complete_no_result.result = None;
    *stream.last_mut().unwrap() = receipt_event(complete_no_result);
    assert_eq!(terminal_receipt(&stream, "", TEXT).unwrap_err(), Incomplete::NoGraph);
    server.stop().await;
}
