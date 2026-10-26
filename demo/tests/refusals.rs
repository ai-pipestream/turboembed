//! What the demo refuses before calling the server: JSON and every other
//! content type, a method off the allowlist or unknown, a streaming
//! method over the unary protocol, compression, and a malformed envelope.

mod common;

use common::*;
use turbo_kserve::Serve;
use turbo_kserve::proto::ServerLiveResponse;
use turbo_kserve::proto::training::{ListTeachersRequest, ListTeachersResponse, TrainStaticModelRequest};

const LIVE: &str = "/inference.GRPCInferenceService/ServerLive";
const TRAIN: &str = "/turbo.training.v1.TurboTrainingService/TrainStaticModel";
const WATCH: &str = "/turbo.search.v1.TurboSearchService/WatchCollection";

fn text(b: Vec<u8>) -> String {
    String::from_utf8(b).unwrap()
}

#[tokio::test]
async fn json_and_other_content_types_are_415() {
    let s = server(&["tiny-bert-bundle"], Serve::default()).await;
    let d = demo(&s, &[]).await;
    for kind in [
        Some("application/json"),
        Some("application/connect+json"),
        Some("application/json; charset=utf-8"),
        Some("text/plain"),
        Some("application/grpc"),
        Some("application/grpc-web+proto"),
        None,
    ] {
        let headers: Vec<(&str, &str)> = kind.map(|k| ("Content-Type", k)).into_iter().collect();
        let r = http(&d, "POST", LIVE, &headers, b"{}").await;
        assert_eq!(r.status, 415, "{kind:?}");
        assert_eq!(r.header("accept-post"), Some("application/proto, application/connect+proto"));
    }
    // A content type's parameters and case do not matter.
    let r = http(&d, "POST", LIVE, &[("Content-Type", "Application/Proto; charset=x")], b"").await;
    assert_eq!(r.status, 200);
    // A streaming method has no unary form.
    let r = http(&d, "POST", WATCH, &[("Content-Type", UNARY)], b"").await;
    assert_eq!(r.status, 415);
}

#[tokio::test]
async fn a_method_off_the_allowlist_or_unknown_is_refused_in_either_protocol() {
    let s = server(&["tiny-bert-bundle"], Serve { admin: true, ..Serve::default() }).await;
    let d = demo(&s, &[]).await;
    let req = TrainStaticModelRequest::default();
    let (r, body) = unary(&d, "/turbo.search.v1.TurboSearchService/DeleteIndex", &req).await;
    assert_eq!(r.status, 403);
    assert!(text(body).starts_with(r#"{"code":"permission_denied""#));
    let (r, body) = unary(&d, "/turbo.search.v1.TurboSearchService/IndexDocuments", &req).await;
    assert_eq!(r.status, 501);
    assert!(text(body).starts_with(r#"{"code":"unimplemented""#));
    let (r, _) = unary(&d, "/grpc.health.v1.Health/Check", &req).await;
    assert_eq!(r.status, 501);

    let (msgs, end) = stream(&d, TRAIN, &req).await.frames::<ServerLiveResponse>().await;
    assert!(msgs.is_empty());
    assert!(end.starts_with(r#"{"error":{"code":"permission_denied""#), "{end}");
    let (_, end) =
        stream(&d, "/turbo.sink.v1.TurboSinkService/StreamDocuments", &req).await.frames::<ServerLiveResponse>().await;
    assert!(end.starts_with(r#"{"error":{"code":"unimplemented""#), "{end}");

    // Named with --allow, a write is forwarded.
    let d = demo(&s, &["turbo.search.v1.TurboSearchService/DeleteIndex"]).await;
    let (r, body) = unary(&d, "/turbo.search.v1.TurboSearchService/DeleteIndex", &req).await;
    assert_ne!(r.status, 403, "{}", text(body));
}

#[tokio::test]
async fn compression_and_malformed_envelopes_are_refused() {
    let s = server(&["tiny-bert-bundle"], Serve::default()).await;
    let d = demo(&s, &[]).await;
    let r = http(&d, "POST", LIVE, &[("Content-Type", UNARY), ("Content-Encoding", "gzip")], b"").await;
    assert_eq!(r.status, 501);
    let r = http(&d, "POST", LIVE, &[("Content-Type", UNARY), ("Content-Encoding", "identity")], b"").await;
    assert_eq!(r.status, 200);
    let r = http(&d, "POST", LIVE, &[("Content-Type", UNARY), ("Connect-Timeout-Ms", "soon")], b"").await;
    assert_eq!(r.status, 400);

    let teachers = "/turbo.training.v1.TurboTrainingService/ListTeachers";
    let _: ListTeachersResponse = ok(&d, teachers, &ListTeachersRequest {}).await;
    let headers = [("Content-Type", STREAM)];
    let cases: [(&[u8], &str); 5] = [
        (&[], "invalid_argument"),
        (&[0, 0, 0], "invalid_argument"),
        (&[0, 0, 0, 0, 2, 1], "invalid_argument"),
        (&[1, 0, 0, 0, 0], "unimplemented"),
        (&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "invalid_argument"),
    ];
    for (body, code) in cases {
        let (_, end) = http(&d, "POST", WATCH, &headers, body).await.frames::<ServerLiveResponse>().await;
        assert!(end.starts_with(&format!(r#"{{"error":{{"code":"{code}""#)), "{body:?}: {end}");
    }
    let compressed = [("Content-Type", STREAM), ("Connect-Content-Encoding", "gzip")];
    let (_, end) = http(&d, "POST", WATCH, &compressed, &envelope(0, b"")).await.frames::<ServerLiveResponse>().await;
    assert!(end.starts_with(r#"{"error":{"code":"unimplemented""#), "{end}");
}
