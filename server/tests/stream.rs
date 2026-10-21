//! The streaming bulk document analysis call (issue #722) over a real
//! gRPC client with the server in-process on the CPU: byte-for-byte
//! parity with the single-document call in both ordering modes,
//! micro-batching that leaves every result unchanged, backpressure that
//! holds a non-reading client to the advertised caps, cancellation that
//! drops queued work counted in model executions, and per-document
//! failures the stream continues past. Nothing is mocked.

mod common;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use tokio_stream::wrappers::ReceiverStream;
use tonic::Request;
use turbo_kserve::proto::result::analyze_stream_response::Outcome;
use turbo_kserve::proto::result::{AnalyzeOptions, AnalyzeStreamRequest, CapabilitiesRequest};
use turbo_kserve::stream;

use common::*;

/// The single-document call's vector row for one text, what the stream
/// must answer byte for byte.
async fn unary(k: &mut Kserve, text: &str) -> Vec<u8> {
    let out = k.model_infer(texts_req(NAME, &[text.as_bytes()])).await.unwrap().into_inner();
    assert_eq!(out.outputs[0].shape, [1, 32]);
    out.raw_output_contents[0].to_vec()
}

async fn unary_each(k: &mut Kserve, texts: &[String]) -> Vec<Vec<u8>> {
    let mut v = Vec::new();
    for t in texts {
        v.push(unary(k, t).await);
    }
    v
}

#[tokio::test]
async fn stream_results_match_single_document_calls_byte_for_byte_in_both_orders() {
    let (s, mut c) = serve().await;
    let mut k = kserve(&s).await;
    let texts = harness_texts();
    let want = unary_each(&mut k, &texts).await;

    for (order, name) in [(0, "input order"), (1, "completion order")] {
        let mut stream = open_stream(&mut c, order, 0, 0).await;
        for (i, text) in texts.iter().enumerate() {
            stream.send(&format!("doc-{i}"), NAME, text, None).await;
        }
        let stream = stream.half_close();
        let answers = stream.collect().await;
        let got: HashMap<_, _> = answers.iter().cloned().collect();
        for (i, want_row) in want.iter().enumerate() {
            let id = format!("doc-{i}");
            match got.get(&id).unwrap_or_else(|| panic!("{name}: {id} is answered")) {
                Outcome::Vectors(v) => {
                    assert_eq!(&v.data, want_row, "{name}: {id} differs from the single-document call");
                    assert_eq!(v.model, NAME);
                }
                Outcome::Failure(f) => panic!("{name}: {id} failed: {f:?}"),
            }
        }
        if order == 0 {
            let in_order: Vec<String> = answers.iter().map(|(id, _)| id.clone()).collect();
            assert_eq!(in_order, stream_ids(&texts), "input order answers as sent");
        }
    }
}

fn stream_ids(texts: &[String]) -> Vec<String> {
    (0..texts.len()).map(|i| format!("doc-{i}")).collect()
}

#[tokio::test]
async fn documents_waiting_on_one_model_run_as_one_batch_with_the_same_results() {
    let (s, mut c) = serve().await;
    let mut k = kserve(&s).await;
    let texts: Vec<String> = harness_texts().iter().take(8).cloned().collect();
    let want = unary_each(&mut k, &texts).await;
    let before = s.stream_stats().runs;

    let mut stream = open_stream(&mut c, 1, 0, 1000).await;
    for (i, text) in texts.iter().enumerate() {
        stream.send(&format!("b-{i}"), NAME, text, None).await;
    }
    let got = by_id(stream.half_close()).await;
    for (i, want_row) in want.iter().enumerate() {
        match got.get(&format!("b-{i}")).unwrap() {
            Outcome::Vectors(v) => assert_eq!(&v.data, want_row, "b-{i} differs run batched"),
            Outcome::Failure(f) => panic!("b-{i} failed: {f:?}"),
        }
    }
    // Eight documents of one model and one options ran as one model
    // execution, not eight: the batch gathered them.
    assert_eq!(s.stream_stats().runs - before, 1, "8 same-key documents, 1 model execution");
}

#[tokio::test]
async fn a_client_that_sends_without_reading_is_held_to_the_advertised_caps() {
    let (s, mut c) = serve().await;
    let list = c.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
    let caps = list.server.unwrap().stream.unwrap();
    let item_cap = caps.max_queued_items_per_stream as usize;
    let byte_cap = caps.max_queued_bytes_per_stream as usize;

    let stream = open_stream(&mut c, 1, 0, 0).await;
    let n = item_cap * 3;
    let sent = Arc::new(AtomicUsize::new(0));
    let sender = tokio::spawn({
        let (tx, sent) = (stream.tx.clone(), sent.clone());
        async move {
            for i in 0..n {
                tx.send(stream_doc(&format!("flood-{i}"), NAME, &"held by backpressure ".repeat(64), None))
                    .await
                    .unwrap();
                sent.fetch_add(1, Ordering::Relaxed);
            }
        }
    });
    // While the client never reads, the server's queue never passes
    // the caps the capability listing advertises, however the buffers
    // underneath absorb.
    let sample = tokio::time::timeout(Duration::from_millis(500), async {
        while !sender.is_finished() {
            let stats = s.stream_stats();
            assert!(stats.queued_items <= item_cap, "queued_items over the advertised cap");
            assert!(stats.queued_bytes <= byte_cap, "queued_bytes over the advertised cap");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await;
    let finished_on_its_own = sample.is_ok();
    sender.abort();
    let sent_n = sent.load(Ordering::Relaxed);
    assert!(finished_on_its_own || sent_n > item_cap, "the client sent {} past its first cap", sent_n);

    // The stream is healthy past the flood: every sent document
    // answers, and the queue is empty again.
    let mut stream = stream.half_close();
    let mut answered = 0;
    while stream.responses.message().await.unwrap().is_some() {
        answered += 1;
    }
    assert_eq!(answered, sent_n);
    assert_eq!(s.stream_stats().queued_items, 0);
}

#[tokio::test]
async fn cancelling_a_stream_drops_the_work_queued_for_it() {
    let (s, mut c) = serve().await;
    let before = s.stream_stats().runs;
    let mut stream = open_stream(&mut c, 1, 1, 1000).await;
    // One document runs alone (window 1, long linger); the rest arrive
    // after it executed and wait queued.
    stream.send("a", NAME, "first", None).await;
    let a_ran = tokio::time::timeout(Duration::from_secs(10), async {
        while s.stream_stats().runs == before {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await;
    assert!(a_ran.is_ok(), "the first document never ran");
    for i in 0..7 {
        stream.send(&format!("q-{i}"), NAME, "queued then dropped", None).await;
    }
    // Cancel: the request and the response both go away.
    drop(stream.tx);
    drop(stream.responses);
    // The seven queued documents never execute: the model-execution
    // count settles at the one run already made.
    let settled = tokio::time::timeout(Duration::from_secs(5), async {
        let mut last = s.stream_stats().runs;
        loop {
            tokio::time::sleep(Duration::from_millis(100)).await;
            let now = s.stream_stats().runs;
            if now == last {
                break now;
            }
            last = now;
        }
    })
    .await
    .expect("the model-execution count never settled");
    assert_eq!(settled, before + 1, "queued work ran after the cancellation");
}

#[tokio::test]
async fn a_failed_document_returns_its_error_with_its_id_and_the_stream_continues() {
    let (s, mut c) = serve().await;
    let mut k = kserve(&s).await;
    let good = "a perfectly ordinary document";
    let over = "x ".repeat(200);
    let want_good = unary(&mut k, good).await;

    let mut stream = open_stream(&mut c, 0, 0, 0).await;
    stream.send("ok-1", NAME, good, None).await;
    stream.send("no-model", "served-nowhere", good, None).await;
    stream.send("ok-2", NAME, good, None).await;
    stream.send("bad-option", NAME, good, Some(AnalyzeOptions { truncate: 9, ..Default::default() })).await;
    // One options, two documents in one batch: the over-limit document
    // fails and the sound one beside it still answers byte for byte.
    stream.send("sound", NAME, good, Some(AnalyzeOptions { truncate: 1, ..Default::default() })).await;
    stream.send("over-limit", NAME, &over, Some(AnalyzeOptions { truncate: 1, ..Default::default() })).await;
    let answers = stream.half_close().collect().await;
    let got: HashMap<_, _> = answers.iter().cloned().collect();

    // Input order: the answers come back in the order sent, failures
    // and all.
    let in_order: Vec<String> = answers.iter().map(|(id, _)| id.clone()).collect();
    assert_eq!(in_order, vec!["ok-1", "no-model", "ok-2", "bad-option", "sound", "over-limit"]);

    match got.get("ok-1").unwrap() {
        Outcome::Vectors(v) => assert_eq!(v.data, want_good),
        o => panic!("ok-1: {o:?}"),
    }
    match got.get("no-model").unwrap() {
        Outcome::Failure(f) => assert_eq!(f.turbo_code, 1280, "TURBO_E_BUNDLE_NOT_FOUND"),
        o => panic!("no-model: {o:?}"),
    }
    match got.get("bad-option").unwrap() {
        Outcome::Failure(f) => assert_eq!(f.turbo_code, 262, "TURBO_E_INVALID_ENUM"),
        o => panic!("bad-option: {o:?}"),
    }
    match got.get("sound").unwrap() {
        Outcome::Vectors(v) => assert_eq!(v.data, want_good, "the sound document beside a failing one"),
        o => panic!("sound: {o:?}"),
    }
    // The same refusal the single-document call makes, with the same code.
    match got.get("over-limit").unwrap() {
        Outcome::Failure(f) => {
            assert_eq!(f.turbo_code, unary_over(&mut k).await, "the stream mirrors the unary refusal")
        }
        o => panic!("over-limit: {o:?}"),
    }
}

/// The single-document call's refusal for the over-limit document, the
/// parity the stream's failure mirrors.
async fn unary_over(k: &mut Kserve) -> i32 {
    let mut req = texts_req(NAME, &["x ".repeat(200).as_bytes()]);
    req.parameters.insert(
        "truncate".into(),
        turbo_kserve::proto::InferParameter {
            parameter_choice: Some(turbo_kserve::proto::infer_parameter::ParameterChoice::StringParam(
                "TRUNCATE_NONE".into(),
            )),
        },
    );
    let e = k.model_infer(req).await.expect_err("the single-document call refuses too");
    assert_eq!(refusal(&e), (tonic::Code::OutOfRange, 771));
    771
}

#[tokio::test]
async fn the_stream_advertises_and_enforces_its_caps() {
    let (_s, mut c) = serve().await;
    let list = c.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
    let caps = list.server.unwrap().stream.unwrap();
    assert_eq!(caps.max_queued_items_per_stream, stream::MAX_QUEUED_ITEMS_PER_STREAM as u64);
    assert_eq!(caps.max_queued_bytes_per_stream, stream::MAX_QUEUED_BYTES_PER_STREAM as u64);
    assert_eq!(caps.max_inflight_items_per_stream, stream::MAX_INFLIGHT_ITEMS_PER_STREAM as u64);
    assert_eq!(caps.default_batch_linger_ms, stream::DEFAULT_BATCH_LINGER.as_millis() as u64);

    // A header past an advertised cap refuses, naming the limit, before
    // anything is allocated.
    let e = open_err(&mut c, stream_header(0, (caps.max_inflight_items_per_stream + 1) as u32, 0)).await;
    assert_eq!(e.code(), tonic::Code::InvalidArgument);
    assert!(e.message().contains("max_inflight"), "{}", e.message());
    let e = open_err(&mut c, stream_header(2, 0, 0)).await;
    assert_eq!(e.code(), tonic::Code::InvalidArgument);
    assert!(e.message().contains("order"), "{}", e.message());
    let e = open_err(&mut c, stream_doc("x", NAME, "hi", None)).await;
    assert!(e.message().contains("first message"), "{}", e.message());
}

/// Opening a stream that refuses before its first answer: the status
/// the call itself returns.
async fn open_err(c: &mut Analysis, first: AnalyzeStreamRequest) -> tonic::Status {
    let (tx, rx) = tokio::sync::mpsc::channel(4);
    tx.send(first).await.unwrap();
    drop(tx);
    c.analyze_stream(Request::new(ReceiverStream::new(rx))).await.expect_err("the stream refuses")
}
