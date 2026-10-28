//! What an upload is refused, and how: before the server when it is not
//! allowed, unauthenticated, from another origin or not octet-stream; at
//! once when a frame or the body says it is over its limit; and the
//! server's own refusal with its status and `turbo-*` headers. Every error
//! is headers and no body.

mod common;
mod uploads;

use common::*;
use prost::Message;
use turbo_kserve::Serve;
use turbo_kserve::proto::sink::{DocumentSinkItem, DocumentSinkSummary};
use turbo_kserve::proto::vocabulary::ImportDictionaryHeader;
use turbo_kserve::sink::SinkReceiver;
use turbo_kserve_demo::limits::{FRAME_BYTES, MAX_FRAME_BYTES, MAX_STREAM_BYTES};
use turbo_kserve_demo::{Allow, Options};
use uploads::*;

fn allow(paths: &[&str]) -> Allow {
    Allow::with(&paths.iter().map(|p| p.to_string()).collect::<Vec<_>>()).unwrap()
}

#[tokio::test]
async fn an_upload_not_allowed_or_not_an_upload_never_reaches_the_server() {
    let d = demo_at(nowhere().await, &[]).await;
    for path in [IMPORT, LEARN, INDEX, SINK] {
        let a = put(d.local_addr(), path, "", b"").await;
        assert_eq!((a.status, a.header("grpc-status")), (403, Some("7")), "{path}");
        assert!(a.body.is_empty());
    }
    for path in ["/inference.GRPCInferenceService/ServerLive", "/turbo.result.v1.AnalysisService/AnalyzeStream"] {
        let a = put(d.local_addr(), path, "", b"").await;
        assert_eq!((a.status, a.header("grpc-status")), (501, Some("12")), "{path}");
    }
    // A Connect POST to an upload's path, and a GET on one.
    let r = http(&d, "POST", INDEX, &[("Content-Type", UNARY)], b"").await;
    assert_eq!(r.status, 501);
    let r = http(&d, "GET", &format!("/upload{INDEX}"), &[], b"").await;
    assert_eq!((r.status, r.header("allow")), (405, Some("PUT")));
    // Allowed, it goes to the server, which is not there.
    let d = demo_at(nowhere().await, &[INDEX]).await;
    assert_eq!(put(d.local_addr(), INDEX, "", b"").await.status, 503);
}

#[tokio::test]
async fn only_octet_stream_is_taken() {
    let d = demo_at(nowhere().await, &[IMPORT]).await;
    for kind in ["application/json", "multipart/form-data; boundary=x", "application/proto", "text/plain"] {
        let r = http(&d, "PUT", &format!("/upload{IMPORT}"), &[("Content-Type", kind)], b"{}").await;
        assert_eq!((r.status, r.header("accept")), (415, Some(OCTET)), "{kind}");
    }
}

#[tokio::test]
async fn the_gate_holds_for_uploads() {
    let options = Options {
        allow: allow(&[IMPORT]),
        token: Some("secret".into()),
        origins: vec!["http://app.example".into()],
        ..Options::default()
    };
    let d = demo_with(nowhere().await, options).await;
    let a = put(d.local_addr(), IMPORT, "", b"").await;
    assert_eq!((a.status, a.header("grpc-status")), (401, Some("16")));
    let a = put(d.local_addr(), IMPORT, "Authorization: Bearer wrong\r\n", b"").await;
    assert_eq!(a.status, 401);
    let a = put(d.local_addr(), IMPORT, "Authorization: Bearer secret\r\nOrigin: http://other.example\r\n", b"").await;
    assert_eq!(a.status, 403);
    let a = put(d.local_addr(), IMPORT, "Authorization: Bearer secret\r\nOrigin: http://app.example\r\n", b"").await;
    assert_eq!(a.status, 503);
    assert_eq!(a.header("access-control-allow-origin"), Some("http://app.example"));
    assert!(a.header("access-control-expose-headers").is_some_and(|h| h.contains("grpc-status")));
    let pre = |method: &'static str| {
        let headers = [("Origin", "http://app.example"), ("Access-Control-Request-Method", method)];
        let path = format!("/upload{IMPORT}");
        let d = &d;
        async move { http(d, "OPTIONS", &path, &headers, b"").await }
    };
    let r = pre("PUT").await;
    assert_eq!((r.status, r.header("access-control-allow-methods")), (204, Some("POST, PUT")));
    assert_eq!(pre("DELETE").await.status, 403);
}

#[tokio::test]
async fn a_body_or_frame_over_its_limit_is_refused_at_once() {
    let d = demo_at(nowhere().await, &[INDEX]).await;
    let p = Put::open(d.local_addr(), INDEX, "", Some(MAX_STREAM_BYTES + 1)).await;
    let a = p.answer(5).await;
    assert_eq!((a.status, a.header("grpc-status")), (413, Some("8")));

    let s = server(&["tiny-bert-bundle"], Serve { admin: true, ..Serve::default() }).await;
    let d = demo(&s, &[INDEX]).await;
    let mut p = Put::open(d.local_addr(), INDEX, "", Some(5 + MAX_FRAME_BYTES as u64 + 1)).await;
    let mut prefix = vec![0u8];
    prefix.extend_from_slice(&(MAX_FRAME_BYTES as u32 + 1).to_be_bytes());
    assert!(p.send(&prefix).await);
    let a = p.answer(5).await;
    assert_eq!((a.status, a.header("grpc-status")), (429, Some("8")), "{a:?}");

    // A compressed frame, and a body that ends inside a frame.
    for body in [envelope(1, b"x"), envelope(0, b"xyz")[..6].to_vec()] {
        let a = put(d.local_addr(), INDEX, "", &body).await;
        assert_eq!((a.status, a.header("grpc-status")), (400, Some("3")), "{a:?}");
    }
}

#[tokio::test]
async fn the_servers_refusal_keeps_its_status_and_turbo_headers() {
    let dir = scratch("upload-refused").join("vocabulary");
    let s = server(&["tiny-bert-bundle"], Serve { admin: true, vocabulary: Some(dir), ..Serve::default() }).await;
    let d = demo(&s, &[IMPORT]).await;
    let header = ImportDictionaryHeader {
        format_id: "plain-word-list".into(),
        name: "w".into(),
        byte_size: 100,
        sha256: "0".repeat(64),
        ..Default::default()
    };
    let body = [envelope(0, &header.encode_to_vec()), b"cat\n".to_vec()].concat();
    let a = put(d.local_addr(), IMPORT, "", &body).await;
    assert_eq!((a.status, a.header("grpc-status")), (400, Some("3")), "{a:?}");
    assert!(a.header("turbo-code").is_some() && a.header("grpc-message").is_some(), "{a:?}");
    assert!(a.body.is_empty());
}

/// An item whose message is `bytes` long.
fn item(sequence: u64, bytes: usize) -> Vec<u8> {
    let mut i = DocumentSinkItem { sequence, analyzed: Some(Default::default()), ..Default::default() };
    let base = i.encoded_len();
    for head in 2..=5 {
        i.doc_id = "d".repeat(bytes - base - head);
        if i.encoded_len() == bytes {
            return envelope(0, &i.encode_to_vec());
        }
    }
    panic!("no padding makes {bytes} bytes")
}

/// The 2 GiB cap in frames of `FRAME_BYTES`, chunked so only the frames
/// say how much: at the cap the upload is answered, one frame more is 413.
/// Four GiB cross loopback, so it runs only with TURBO_TEST_2GIB=1.
#[tokio::test]
async fn the_two_gib_cap_is_enforced_while_streaming() {
    if std::env::var("TURBO_TEST_2GIB").as_deref() != Ok("1") {
        eprintln!("skipped: set TURBO_TEST_2GIB=1");
        return;
    }
    let sink = Serve { sink_receiver: Some(SinkReceiver::new(|_| {})), ..Serve::default() };
    let s = server(&["tiny-bert-bundle"], sink).await;
    let d = demo(&s, &[SINK]).await;
    let n = MAX_STREAM_BYTES / (FRAME_BYTES as u64 + 5);
    for (count, want) in [(n, 200), (n + 1, 413)] {
        let mut p = Put::open(d.local_addr(), SINK, "", None).await;
        let mut sent = 0;
        while sent < count && p.chunk(&item(1000 + sent, FRAME_BYTES)).await {
            sent += 1;
        }
        p.send(b"0\r\n\r\n").await;
        let a = p.answer(120).await;
        assert_eq!(a.status, want, "{count} frames, {sent} sent: {a:?}");
        if want == 200 {
            assert_eq!(sent, count);
            let summary = DocumentSinkSummary::decode(&a.body[..]).unwrap();
            assert_eq!(summary.accepted_count, count, "{summary:?}");
        }
    }
}
