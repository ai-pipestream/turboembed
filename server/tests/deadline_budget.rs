//! Real deadline and gRPC regressions for client timeout budgets.
mod common;

use std::sync::Arc;
use std::time::Duration;
use tonic::{Request, metadata::MetadataValue};
use turbo_kserve::deadline::Deadline;
use turbo_kserve::proto::ServerReadyRequest;

#[test]
fn positive_short_deadline_is_not_already_expired() {
    let d = Deadline::with_timeout(Duration::from_millis(15));
    assert!(!d.is_expired(), "a fresh 15 ms budget must allow work");
}

#[tokio::test]
async fn completed_deadline_releases_its_timer_state() {
    let d = Deadline::with_timeout(Duration::from_secs(3600));
    let weak = Arc::downgrade(d.cancel_flag());
    drop(d);
    tokio::task::yield_now().await;
    assert!(weak.upgrade().is_none(), "completed requests retain detached timer state");
}

#[tokio::test]
async fn ready_rpc_can_use_a_positive_short_budget() {
    let (s, _) = common::serve().await;
    let mut client = common::kserve(&s).await;
    assert!(client.server_ready(ServerReadyRequest {}).await.unwrap().into_inner().ready);
    let mut req = Request::new(ServerReadyRequest {});
    req.metadata_mut().insert("grpc-timeout", MetadataValue::from_static("15m"));
    let result = client.server_ready(req).await;
    s.stop().await;
    assert!(result.is_ok(), "ready RPC failed before processing: {result:?}");
}

#[tokio::test]
async fn clones_keep_the_full_budget_and_release_state_when_done() {
    let before = tokio::time::Instant::now();
    let d = Deadline::with_timeout(Duration::from_millis(50));
    assert!(d.stop_at().unwrap() >= before + Duration::from_millis(50));
    assert_eq!(d.stop_at(), d.hard_at());
    let clone = d.clone();
    let weak = Arc::downgrade(d.cancel_flag());
    drop(d);
    assert!(weak.upgrade().is_some());
    tokio::time::sleep_until(clone.stop_at().unwrap() + Duration::from_millis(5)).await;
    assert!(clone.is_expired());
    drop(clone);
    tokio::task::yield_now().await;
    assert!(weak.upgrade().is_none());
}

#[tokio::test]
async fn waiting_for_stream_header_expires_with_deadline_status() {
    let (s, _) = common::serve().await;
    // A tonic Channel starts its own grpc-timeout before sending the request,
    // yielding client-local CANCELLED. Inspect the server's wire status directly.
    let socket = tokio::net::TcpStream::connect(s.local_addr()).await.unwrap();
    let (mut sender, connection) = h2::client::handshake(socket).await.unwrap();
    let connection = tokio::spawn(connection);
    let req = tonic::codegen::http::Request::builder()
        .method("POST")
        .uri(format!("http://{}/turbo.result.v1.AnalysisService/AnalyzeStream", s.local_addr()))
        .header("content-type", "application/grpc")
        .header("te", "trailers")
        .header("grpc-timeout", "20m")
        .body(())
        .unwrap();
    let (response, body) = sender.send_request(req, false).unwrap();
    let response = tokio::time::timeout(Duration::from_secs(2), response).await.unwrap().unwrap();
    assert_eq!(response.headers().get("grpc-status").unwrap(), "4");
    drop(body);
    drop(response);
    drop(sender);
    connection.abort();
    s.stop().await;
}
