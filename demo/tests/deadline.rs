//! Deadline budgets through the real proxy and server, including upload.
mod common;

use std::time::Duration;

use prost::Message;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use turbo_kserve::Serve;
use turbo_kserve::proto::search::turbo_search_service_client::TurboSearchServiceClient;
use turbo_kserve::proto::search::{SetCollectionRequest, WatchCollectionRequest};
use turbo_kserve::proto::{ServerLiveRequest, ServerLiveResponse};

const LIVE: &str = "/inference.GRPCInferenceService/ServerLive";

#[tokio::test]
async fn slow_request_body_consumes_connect_deadline() {
    let server = common::server(&["tiny-static-bundle"], Serve::default()).await;
    let proxy = common::demo(&server, &[]).await;
    for protocol in [common::UNARY, common::STREAM] {
        let mut conn = TcpStream::connect(proxy.local_addr()).await.unwrap();
        // An unknown protobuf field is valid for the empty ServerLive request.
        let message = [0x78, 0];
        let body = if protocol == common::STREAM { common::envelope(0, &message) } else { message.to_vec() };
        let head = format!(
            "POST {LIVE} HTTP/1.1\r\nHost: demo\r\nConnection: close\r\nContent-Type: {protocol}\r\nConnect-Timeout-Ms: 25\r\nContent-Length: {}\r\n\r\n",
            body.len()
        );
        conn.write_all(head.as_bytes()).await.unwrap();
        tokio::time::sleep(Duration::from_millis(150)).await;
        let _ = conn.write_all(&body).await;
        let mut response = Vec::new();
        tokio::time::timeout(Duration::from_secs(3), conn.read_to_end(&mut response)).await.unwrap().unwrap();
        let response = String::from_utf8_lossy(&response);
        let status = if protocol == common::STREAM { "HTTP/1.1 200" } else { "HTTP/1.1 504" };
        assert!(response.starts_with(status), "expired {protocol} deadline: {response:?}");
        assert!(response.contains("deadline_exceeded"), "{response:?}");
    }
}

#[tokio::test]
async fn in_budget_requests_succeed_and_invalid_timeout_syntax_is_refused() {
    let server = common::server(&["tiny-static-bundle"], Serve::default()).await;
    let proxy = common::demo(&server, &[]).await;
    let msg = ServerLiveRequest {}.encode_to_vec();
    let reply =
        common::http(&proxy, "POST", LIVE, &[("Content-Type", common::UNARY), ("Connect-Timeout-Ms", "5000")], &msg)
            .await;
    assert_eq!(reply.status, 200);
    assert!(ServerLiveResponse::decode(reply.all().await.as_slice()).unwrap().live);
    for invalid in ["+1", "-1", "10000000000", "soon", ""] {
        let reply = common::http(
            &proxy,
            "POST",
            LIVE,
            &[("Content-Type", common::UNARY), ("Connect-Timeout-Ms", invalid)],
            &msg,
        )
        .await;
        assert_eq!(reply.status, 400, "{invalid}");
        assert!(String::from_utf8(reply.all().await).unwrap().contains("invalid_argument"));
    }
    let reply =
        common::http(&proxy, "POST", LIVE, &[("Content-Type", common::UNARY), ("Connect-Timeout-Ms", "0")], &msg).await;
    assert_eq!(reply.status, 504);
}

#[tokio::test]
async fn a_quiet_real_watch_stream_ends_at_the_connect_deadline() {
    let server = common::server(&["tiny-static-bundle"], Serve { admin: true, ..Serve::default() }).await;
    let proxy = common::demo(&server, &[]).await;
    let mut client = TurboSearchServiceClient::connect(format!("http://{}", server.local_addr())).await.unwrap();
    client
        .set_collection(SetCollectionRequest {
            collection_name: "deadline".into(),
            baseline_terms: vec!["text".into()],
            oov_alert_threshold: 0.1,
        })
        .await
        .unwrap();
    let body = common::envelope(
        0,
        &WatchCollectionRequest { collection_name: "deadline".into(), min_oov_ratio: 0.0 }.encode_to_vec(),
    );
    let mut reply = common::http(
        &proxy,
        "POST",
        "/turbo.search.v1.TurboSearchService/WatchCollection",
        &[("Content-Type", common::STREAM), ("Connect-Timeout-Ms", "100")],
        &body,
    )
    .await;
    assert_eq!(reply.status, 200);
    let (flags, body) = tokio::time::timeout(Duration::from_secs(3), reply.frame()).await.unwrap().unwrap();
    assert_eq!(flags, 2);
    assert!(String::from_utf8(body).unwrap().contains("deadline_exceeded"));
    assert!(reply.frame().await.is_none());
}
