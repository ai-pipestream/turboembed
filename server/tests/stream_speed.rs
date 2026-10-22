//! The speed bar of issue #722, release only (`cargo test --release -p
//! turbo-kserve --test stream_speed -- --ignored`): the same documents
//! through one AnalyzeStream must answer at least as fast as the same
//! documents as concurrent single-document calls, on the same server,
//! CPU and thread count. Rounds alternate which side runs first, so a
//! warm cache or a lazy allocator never decides the answer.

mod common;

use std::time::{Duration, Instant};

use tokio_stream::wrappers::ReceiverStream;
use tonic::Request;
use turbo_kserve::proto::result::analyze_stream_request::Kind;
use turbo_kserve::proto::result::{AnalyzeDocument, AnalyzeStreamHeader, AnalyzeStreamRequest, CapabilitiesResponse};

use common::*;

const ROUNDS: usize = 6;
const CONCURRENCY: usize = 8;

fn harness_docs(n: usize) -> Vec<String> {
    let texts = harness_texts();
    (0..n).map(|i| texts[i % texts.len()].clone()).collect()
}

/// One round of `n` documents through one stream: send them all, read
/// them all, and check the answers come back whole.
async fn stream_round(c: &mut Analysis, docs: &[String], linger_ms: u32) -> Duration {
    let (tx, rx) = tokio::sync::mpsc::channel(64);
    tx.send(AnalyzeStreamRequest {
        kind: Some(Kind::Header(AnalyzeStreamHeader { order: 1, max_inflight: 0, batch_linger_ms: linger_ms })),
    })
    .await
    .unwrap();
    let mut responses = c.analyze_stream(Request::new(ReceiverStream::new(rx))).await.unwrap().into_inner();
    let start = Instant::now();
    for (i, text) in docs.iter().enumerate() {
        tx.send(AnalyzeStreamRequest {
            kind: Some(Kind::Document(AnalyzeDocument {
                id: format!("s-{i}"),
                model: NAME.into(),
                text: text.clone().into(),
                options: None,
            })),
        })
        .await
        .unwrap();
    }
    drop(tx);
    let mut answered = 0;
    while responses.message().await.unwrap().is_some() {
        answered += 1;
    }
    let took = start.elapsed();
    assert_eq!(answered, docs.len(), "every stream document answered");
    took
}

/// One round of `n` documents as concurrent single-document calls,
/// `CONCURRENCY` at a time.
async fn unary_round(k: &mut Kserve, docs: &[String]) -> Duration {
    let start = Instant::now();
    let mut set = tokio::task::JoinSet::new();
    let permits = std::sync::Arc::new(tokio::sync::Semaphore::new(CONCURRENCY));
    for (i, text) in docs.iter().enumerate() {
        let mut k = k.clone();
        let text = text.clone();
        let permits = permits.clone();
        set.spawn(async move {
            let _held = permits.acquire().await.unwrap();
            let mut req = texts_req(NAME, &[text.as_bytes()]);
            req.id = format!("u-{i}");
            let out = k.model_infer(req).await.unwrap().into_inner();
            assert_eq!(out.outputs[0].shape, [1, 32]);
        });
    }
    while set.join_next().await.is_some() {}
    start.elapsed()
}

fn report(name: &str, n: usize, rounds: usize, took: Duration) -> f64 {
    let dps = (rounds * n) as f64 / took.as_secs_f64();
    println!("{name}: {n} docs x {rounds} rounds in {:.2?}: {dps:.0} docs/sec", took);
    dps
}

#[test]
#[ignore = "release speed bar: run for the receipt (docs/receipts/server-722)"]
fn stream_documents_per_second_match_concurrent_single_calls() {
    let n = 512;
    let docs = harness_docs(n);
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap();
    let (s, mut c) = rt.block_on(serve_many(8));
    let mut k = rt.block_on(kserve(&s));
    let list = rt.block_on(async {
        c.capabilities(turbo_kserve::proto::result::CapabilitiesRequest {}).await.unwrap().into_inner()
    });
    let linger = listed_linger(list);
    let (mut stream_total, mut unary_total) = (Duration::ZERO, Duration::ZERO);
    for round in 0..ROUNDS {
        let (t_stream, t_unary) = if round % 2 == 0 {
            let a = rt.block_on(stream_round(&mut c, &docs, linger));
            let b = rt.block_on(unary_round(&mut k, &docs));
            (a, b)
        } else {
            let b = rt.block_on(unary_round(&mut k, &docs));
            let a = rt.block_on(stream_round(&mut c, &docs, linger));
            (a, b)
        };
        stream_total += t_stream;
        unary_total += t_unary;
        println!("round {round}: stream {:.2?}, unary {:.2?}", t_stream, t_unary);
    }
    let stream = report("analyze stream", n, ROUNDS, stream_total);
    let unary = report("concurrent single calls", n, ROUNDS, unary_total);
    println!("stream/unary: {:.2}x", stream / unary);
    assert!(stream >= unary, "stream {stream:.0} docs/sec under unary {unary:.0} docs/sec");
    rt.block_on(s.stop());
}

/// A server with more sessions than the unary side's concurrency, so
/// both sides run on the same footing.
async fn serve_many(sessions: u32) -> (turbo_kserve::Server, Analysis) {
    let s = turbo_kserve::Server::start(
        "127.0.0.1:0".parse().unwrap(),
        vec![model_config(sessions)],
        turbo_kserve::config::DEFAULT_MAX_MESSAGE_BYTES,
        turbo_kserve::Serve { admin: true },
    )
    .await
    .unwrap();
    s.load().await.unwrap();
    let c = turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient::connect(format!(
        "http://{}",
        s.local_addr()
    ))
    .await
    .unwrap();
    (s, c)
}

fn model_config(sessions: u32) -> turbo_kserve::config::ModelConfig {
    turbo_kserve::config::ModelConfig {
        bundle: tiny().to_str().unwrap().to_string(),
        name: None,
        device: turbo_kserve::config::Device::Index(cpu()),
        precision: 0,
        max_batch: 0,
        max_seq: 0,
        sessions,
    }
}

fn listed_linger(list: CapabilitiesResponse) -> u32 {
    list.server.unwrap().stream.unwrap().default_batch_linger_ms as u32
}
