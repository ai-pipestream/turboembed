//! A client cannot grow the gRPC metrics label set (#1300): a path that
//! names no method the protos define is recorded as service and method
//! `unknown`, so a flood of distinct junk paths leaves the series count and
//! the scrape size where they were.

mod common;

use std::time::Duration;

use common::*;
use turbo_kserve::proto::ServerReadyRequest;

/// POSTs one empty gRPC message to each path over one raw h2 connection,
/// waiting for each response's headers.
async fn post_all(server: &turbo_kserve::Server, paths: impl Iterator<Item = String>) {
    let socket = tokio::net::TcpStream::connect(server.local_addr()).await.unwrap();
    let (sender, connection) = h2::client::handshake(socket).await.unwrap();
    let connection = tokio::spawn(connection);
    for path in paths {
        let mut sender = sender.clone().ready().await.unwrap();
        let request = tonic::codegen::http::Request::builder()
            .method("POST")
            .uri(format!("http://{}{path}", server.local_addr()))
            .header("content-type", "application/grpc")
            .header("te", "trailers")
            .body(())
            .unwrap();
        let (response, mut body) = sender.send_request(request, false).unwrap();
        body.send_data(vec![0, 0, 0, 0, 0].into(), true).unwrap();
        let response = tokio::time::timeout(Duration::from_secs(5), response).await.unwrap().unwrap();
        drop(response);
    }
    connection.abort();
}

fn series(rendered: &str) -> usize {
    rendered.lines().filter(|l| l.starts_with("turbo_kserve_rpc_")).count()
}

#[tokio::test]
async fn junk_paths_do_not_add_metric_series() {
    let (s, _) = serve().await;
    let mut client = kserve(&s).await;
    client.server_ready(ServerReadyRequest {}).await.unwrap();
    // One call of each kind, so every series the flood could touch exists.
    post_all(
        &s,
        ["/junk.Svc/x".to_owned(), "/inference.GRPCInferenceService/Nope".to_owned(), "/".to_owned()].into_iter(),
    )
    .await;
    let before = s.metrics();
    assert!(before.contains("service=\"inference.GRPCInferenceService\",method=\"ServerReady\""), "{before}");
    assert!(before.contains("service=\"unknown\",method=\"unknown\""), "{before}");
    let long = "x".repeat(4000);
    let junk = (0..1000).map(|i| format!("/junk.Svc{i}/{long}"));
    let methods = (0..1000).map(|i| format!("/inference.GRPCInferenceService/Nope{i}"));
    post_all(&s, junk.chain(methods)).await;
    let after = s.metrics();
    s.stop().await;
    assert!(!after.contains("junk.Svc") && !after.contains("Nope"), "a junk path became a label");
    assert_eq!(series(&after), series(&before), "the flood added series");
    assert!(after.len() <= before.len() + 256, "scrape grew from {} to {} bytes", before.len(), after.len());
}
