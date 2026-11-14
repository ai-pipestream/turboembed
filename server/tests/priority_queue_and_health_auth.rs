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
use turbo_kserve::proto::result::{AnalyzeQueryRequest, QueryDenseTargetProto, RerankAndHighlightRequest};
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
    let wait_until = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        let (iq, bq) = server.priority_queue_depths(NAME).unwrap();
        if iq + bq == MAX_QUEUE_DEPTH {
            break;
        }
        assert!(tokio::time::Instant::now() < wait_until, "queue did not fill to {MAX_QUEUE_DEPTH}: ({iq}, {bq})");
        tokio::time::sleep(Duration::from_millis(2)).await;
    }

    // A plain call without x-turbo-priority must answer UNAVAILABLE immediately,
    // proving Tokio's blocking pool is not starved.
    let mut ks = GrpcInferenceServiceClient::new(shared_ch.clone());
    let plain_err =
        tokio::time::timeout(Duration::from_secs(1), ks.model_infer(Request::new(texts_req(NAME, &[b"plain"]))))
            .await
            .expect("plain call must not hang behind queued priority calls")
            .unwrap_err();
    assert_eq!(plain_err.code(), Code::Unavailable);

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
