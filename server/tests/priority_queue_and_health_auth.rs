//! Regression tests for PR #2370 blocking fixes:
//! - `grpc.health.v1.Health/{Check,Watch}` exemption from bearer auth (`probe::check`)
//! - Bounded async priority queue (`MAX_QUEUE_DEPTH`) preventing Tokio blocking-pool starvation
//! - Plain `ModelInfer` without `x-turbo-priority` never jumping ahead of queued priority work
//! - `AnalyzeQuery` / `RerankAndHighlight` `x-turbo-priority` validation

mod common;

use std::time::Duration;

use tonic::metadata::MetadataValue;
use tonic::{Code, Request};
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::probe::{self, Probe};
use turbo_kserve::proto::ServerReadyRequest;
use turbo_kserve::proto::grpc_inference_service_client::GrpcInferenceServiceClient;
use turbo_kserve::proto::health::health_check_response::ServingStatus;
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::result::{
    AnalyzeQueryRequest, QueryDenseTargetProto, RerankAndHighlightRequest, RerankCandidateInputProto,
    RerankHighlightMode,
};
use turbo_kserve::{AuthConfig, MAX_QUEUE_DEPTH, PRIORITY_HEADER, Serve, Server};

use common::*;

fn loopback() -> std::net::SocketAddr {
    "127.0.0.1:0".parse().unwrap()
}

fn model_cfg(sessions: u32) -> ModelConfig {
    ModelConfig {
        bundle: tiny().to_str().unwrap().to_string(),
        name: None,
        device: Device::Index(cpu()),
        precision: 0,
        max_batch: 0,
        max_seq: 0,
        sessions,
    }
}

#[tokio::test]
async fn health_probe_succeeds_with_bearer_auth_enabled() {
    let auth = AuthConfig::from_tokens(["infer-only"], ["admin-only"]).unwrap();
    let serve = Serve { auth: Some(auth), ..Serve::default() };
    let server = Server::start(loopback(), vec![model_cfg(1)], MAX, serve).await.unwrap();
    server.load().await.unwrap();

    let probe_cfg =
        Probe { addr: server.local_addr(), service: String::new(), timeout: Duration::from_secs(3), tls: None };
    let status = probe::check(&probe_cfg).await.unwrap();
    assert_eq!(status, ServingStatus::Serving);

    // Other RPCs still require bearer auth.
    let mut kserve = GrpcInferenceServiceClient::connect(format!("http://{}", server.local_addr())).await.unwrap();
    let err = kserve.server_ready(ServerReadyRequest {}).await.unwrap_err();
    assert_eq!(err.code(), Code::Unauthenticated);

    server.stop().await;
}

#[tokio::test]
async fn priority_queue_burst_refuses_excess_immediately_without_starving_plain_calls() {
    let server = Server::start(loopback(), vec![model_cfg(1)], MAX, Serve::default()).await.unwrap();
    server.load().await.unwrap();
    let endpoint = format!("http://{}", server.local_addr());

    let held = server.hold(NAME).unwrap();
    let shared_ch = tonic::transport::Endpoint::from_shared(endpoint.clone()).unwrap().connect().await.unwrap();

    // Send 600 interactive calls without grpc-timeout while the single session is held.
    let mut handles = Vec::with_capacity(600);
    for _ in 0..600 {
        let ch = shared_ch.clone();
        handles.push(tokio::spawn(async move {
            let mut ks = GrpcInferenceServiceClient::new(ch);
            let mut req = Request::new(texts_req(NAME, &[b"burst"]));
            req.metadata_mut().insert(PRIORITY_HEADER, MetadataValue::from_static("interactive"));
            ks.model_infer(req).await
        }));
    }

    // Wait until the bounded queue reaches MAX_QUEUE_DEPTH.
    wait_for("the queue to fill", NO_ANSWER, || {
        let (iq, bq) = server.priority_queue_depths(NAME).unwrap();
        iq + bq == MAX_QUEUE_DEPTH
    })
    .await;

    // A plain call without x-turbo-priority is refused as busy: it is
    // answered at all only if the queued calls left blocking threads free.
    let mut ks = GrpcInferenceServiceClient::new(shared_ch.clone());
    let plain = within("a plain call behind the queue", NO_ANSWER, ks.model_infer(texts_req(NAME, &[b"plain"])));
    assert_eq!(plain.await.unwrap_err().code(), Code::Unavailable);

    drop(held);

    let mut ok_count = 0usize;
    let mut busy_count = 0usize;
    for h in handles {
        match h.await.unwrap() {
            Ok(_) => ok_count += 1,
            Err(e) => {
                assert_eq!(e.code(), Code::Unavailable, "unexpected error: {e}");
                busy_count += 1;
            }
        }
    }
    assert_eq!(ok_count, MAX_QUEUE_DEPTH);
    assert_eq!(busy_count, 600 - MAX_QUEUE_DEPTH);

    server.stop().await;
}

#[tokio::test]
async fn analyze_query_and_rerank_validate_priority_header() {
    let server = Server::start(loopback(), vec![model_cfg(1)], MAX, Serve::default()).await.unwrap();
    server.load().await.unwrap();
    let mut client = AnalysisServiceClient::connect(format!("http://{}", server.local_addr())).await.unwrap();

    let mut bad_q = Request::new(AnalyzeQueryRequest {
        query: "hello world".into(),
        dense_targets: vec![QueryDenseTargetProto { model: NAME.into(), ..Default::default() }],
        ..Default::default()
    });
    bad_q.metadata_mut().insert(PRIORITY_HEADER, MetadataValue::from_static("urgent"));
    assert_eq!(client.analyze_query(bad_q).await.unwrap_err().code(), Code::InvalidArgument);

    let mut bad_r = Request::new(RerankAndHighlightRequest {
        query: "hello".into(),
        rerank_model: NAME.into(),
        ..Default::default()
    });
    bad_r.metadata_mut().insert(PRIORITY_HEADER, MetadataValue::from_static("urgent"));
    assert_eq!(client.rerank_and_highlight(bad_r).await.unwrap_err().code(), Code::InvalidArgument);

    let mut bulk_q = Request::new(AnalyzeQueryRequest {
        query: "hello world".into(),
        dense_targets: vec![QueryDenseTargetProto { model: NAME.into(), ..Default::default() }],
        ..Default::default()
    });
    bulk_q.metadata_mut().insert(PRIORITY_HEADER, MetadataValue::from_static("bulk"));
    assert_eq!(client.analyze_query(bulk_q).await.unwrap().into_inner().dense_results.len(), 1);

    server.stop().await;
}

/// Enough running time that only a call that never answers runs it out. The
/// stall this guards against is a hang, not a slow call, so nothing here
/// asserts how long an answer takes.
const NO_ANSWER: Duration = Duration::from_secs(60);

fn interactive<T>(msg: T) -> Request<T> {
    let mut req = Request::new(msg);
    req.metadata_mut().insert(PRIORITY_HEADER, MetadataValue::from_static("interactive"));
    req
}

fn rerank_req(rerank: &str, embed: &str) -> RerankAndHighlightRequest {
    let mode = if embed.is_empty() { RerankHighlightMode::None } else { RerankHighlightMode::RangePool };
    RerankAndHighlightRequest {
        query: "hello".into(),
        rerank_model: rerank.into(),
        embed_model: embed.into(),
        highlight_mode: mode as i32,
        candidates: vec![RerankCandidateInputProto {
            doc_id: "d1".into(),
            text: "hello world. A second sentence here.".into(),
            ..Default::default()
        }],
        ..Default::default()
    }
}

fn dense_query(model: &str) -> AnalyzeQueryRequest {
    AnalyzeQueryRequest {
        query: "hello world".into(),
        dense_targets: vec![QueryDenseTargetProto { model: model.into(), ..Default::default() }],
        ..Default::default()
    }
}

/// Six model pools are held and each gets a full queue of interactive calls:
/// four through `AnalyzeQuery`, one through the rerank step of
/// `RerankAndHighlight`, and one through its highlight step, which queues on
/// a second pool after the rerank pool let the call through. That is 768
/// queued calls, more than Tokio's 512 blocking threads. A queued call that
/// parked a blocking thread (the BLOCK on #2370) could not even fill the
/// queues, and every other call that needs a blocking thread would wait
/// behind them for ever.
#[tokio::test]
async fn multi_pool_queued_analyze_and_rerank_never_starve_blocking_pool() {
    let rerank_bundle = make_rerank_bundle("PAIR_BERT");
    let rerank_cfg = |name: &str| ModelConfig {
        bundle: rerank_bundle.0.to_str().unwrap().to_string(),
        name: Some(name.into()),
        ..model_cfg(1)
    };
    let embed_cfg = |name: &str| ModelConfig { name: Some(name.into()), ..model_cfg(1) };
    // `r1` stays free: it only feeds calls on to the held highlight pool `h0`.
    let held_names = ["e0", "e1", "e2", "e3", "r0", "h0"];
    let models = vec![
        embed_cfg("e0"),
        embed_cfg("e1"),
        embed_cfg("e2"),
        embed_cfg("e3"),
        rerank_cfg("r0"),
        embed_cfg("h0"),
        rerank_cfg("r1"),
        embed_cfg("free"),
    ];

    let server = Server::start(loopback(), models, MAX, Serve::default()).await.unwrap();
    server.load().await.unwrap();
    let endpoint = format!("http://{}", server.local_addr());
    let shared_ch = tonic::transport::Endpoint::from_shared(endpoint).unwrap().connect().await.unwrap();

    let held_pools: Vec<_> = held_names.iter().map(|name| server.hold(name).unwrap()).collect();

    let mut handles = Vec::with_capacity(held_names.len() * MAX_QUEUE_DEPTH);
    for _ in 0..MAX_QUEUE_DEPTH {
        for name in &held_names[..4] {
            let mut client = AnalysisServiceClient::new(shared_ch.clone());
            let req = interactive(dense_query(name));
            handles.push(tokio::spawn(async move { client.analyze_query(req).await.map(|_| 0) }));
        }
        for (rerank, embed) in [("r0", ""), ("r1", "h0")] {
            let mut client = AnalysisServiceClient::new(shared_ch.clone());
            let req = interactive(rerank_req(rerank, embed));
            handles.push(tokio::spawn(async move {
                let out = client.rerank_and_highlight(req).await?.into_inner();
                Ok(out.results.iter().map(|r| r.highlights.len()).sum::<usize>())
            }));
        }
    }

    // Every held pool fills to its limit. Calls for `h0` get there one at a
    // time through the single `r1` session, so this also shows the rerank
    // step still runs while 640 other calls are queued.
    wait_for("every held pool's queue to fill", NO_ANSWER, || {
        held_names.iter().all(|name| {
            let (iq, bq) = server.priority_queue_depths(name).unwrap();
            iq + bq == MAX_QUEUE_DEPTH
        })
    })
    .await;

    // A plain ModelInfer on a held pool is refused as busy, and calls on
    // free pools are served, although 768 calls are queued.
    let mut ks = GrpcInferenceServiceClient::new(shared_ch.clone());
    let plain = within("a plain ModelInfer on a held pool", NO_ANSWER, ks.model_infer(texts_req("e0", &[b"plain"])));
    assert_eq!(plain.await.unwrap_err().code(), Code::Unavailable);

    let mut analysis = AnalysisServiceClient::new(shared_ch.clone());
    let free = within("AnalyzeQuery on a free pool", NO_ANSWER, analysis.analyze_query(dense_query("free")));
    assert_eq!(free.await.unwrap().into_inner().dense_results.len(), 1);
    let free = within("ModelInfer on a free pool", NO_ANSWER, ks.model_infer(texts_req("free", &[b"plain"])));
    assert_eq!(free.await.unwrap().into_inner().outputs.len(), 1);

    // One more queued call on a full pool is refused, not parked.
    let over = within("a call past the queue limit", NO_ANSWER, analysis.analyze_query(interactive(dense_query("e0"))));
    assert_eq!(over.await.unwrap_err().code(), Code::Unavailable);

    drop(held_pools);

    // Every queued call is then served, and the ones that waited on the
    // highlight pool come back with their highlights.
    let mut highlights = 0usize;
    for h in handles {
        highlights += within("a queued call after release", NO_ANSWER, h).await.unwrap().unwrap();
    }
    assert!(highlights >= MAX_QUEUE_DEPTH, "{highlights} highlights from {MAX_QUEUE_DEPTH} highlight calls");

    server.stop().await;
}
