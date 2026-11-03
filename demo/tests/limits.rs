//! The demo's message limits (limits.rs): a request past axum's 2 MB
//! default reaches the server in either protocol, and one that says it is
//! over the request limit is refused before its body is read.

mod common;

use std::time::Duration;

use common::*;
use prost::Message;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use turbo_kserve::Serve;
use turbo_kserve::proto::vocabulary::{DownloadVocabularyRequest, VocabularyArtifactChunk};
use turbo_kserve::proto::{ServerLiveRequest, ServerLiveResponse};
use turbo_kserve_demo::limits::MAX_REQUEST_BYTES;

const LIVE: &str = "/inference.GRPCInferenceService/ServerLive";
const DOWNLOAD: &str = "/turbo.vocabulary.v1.TurboVocabularyService/DownloadVocabulary";

/// `msg` with an unknown field of `n` bytes, which the server skips.
fn padded(msg: &impl Message, n: usize) -> Vec<u8> {
    let mut out = msg.encode_to_vec();
    out.push((15 << 3) | 2);
    prost::encoding::encode_varint(n as u64, &mut out);
    out.resize(out.len() + n, b'p');
    out
}

#[tokio::test]
async fn a_request_over_two_mb_reaches_the_server_in_either_protocol() {
    let d = scratch("limits-vocab");
    let s = server(&["tiny-bert-bundle"], Serve { vocabulary: Some(d.join("vocabulary")), ..Serve::default() }).await;
    let demo = demo(&s, &[]).await;
    let body = padded(&ServerLiveRequest::default(), 3 << 20);
    let mut r = http(&demo, "POST", LIVE, &[("Content-Type", UNARY)], &body).await;
    assert_eq!(r.status, 200);
    while r.more().await {}
    assert!(ServerLiveResponse::decode(r.body.as_slice()).unwrap().live);

    let req = DownloadVocabularyRequest { artifact_id: "vocab-none".into() };
    let body = envelope(0, &padded(&req, 3 << 20));
    let r = http(&demo, "POST", DOWNLOAD, &[("Content-Type", STREAM)], &body).await;
    assert_eq!(r.status, 200);
    let (msgs, end) = r.frames::<VocabularyArtifactChunk>().await;
    // The server's own refusal, with its metadata, not the demo's.
    assert!(msgs.is_empty());
    assert!(!end.contains("resource_exhausted") && end.contains(r#""turbo-code":[""#), "{end}");
}

/// Sends `head` and `body` and reads the reply until `needle` or the end,
/// in at most five seconds: the demo answers without reading the rest.
async fn refused(demo: &turbo_kserve_demo::Demo, head: &str, body: &[u8], needle: &str) -> String {
    let mut conn = TcpStream::connect(demo.local_addr()).await.unwrap();
    conn.write_all(head.as_bytes()).await.unwrap();
    conn.write_all(body).await.unwrap();
    let mut got = Vec::new();
    let read = async {
        let mut buf = [0u8; 4096];
        while !String::from_utf8_lossy(&got).contains(needle) {
            match conn.read(&mut buf).await {
                Ok(0) | Err(_) => break,
                Ok(n) => got.extend_from_slice(&buf[..n]),
            }
        }
    };
    tokio::time::timeout(Duration::from_secs(5), read).await.expect("an answer before the body");
    String::from_utf8_lossy(&got).into_owned()
}

#[tokio::test]
async fn a_request_that_says_it_is_over_the_limit_is_refused_before_it_is_read() {
    let s = server(&["tiny-bert-bundle"], Serve::default()).await;
    let demo = demo(&s, &[]).await;
    let over = MAX_REQUEST_BYTES + 1;
    let head = format!("POST {LIVE} HTTP/1.1\r\nHost: demo\r\nContent-Type: {UNARY}\r\nContent-Length: {over}\r\n\r\n");
    let reply = refused(&demo, &head, b"", "resource_exhausted").await;
    assert!(reply.starts_with("HTTP/1.1 429"), "{reply}");
    assert!(reply.contains(&format!("the request is over {MAX_REQUEST_BYTES} bytes")), "{reply}");

    // Chunked, so no length up front: the envelope's says it.
    let head = format!(
        "POST {DOWNLOAD} HTTP/1.1\r\nHost: demo\r\nContent-Type: {STREAM}\r\nTransfer-Encoding: chunked\r\n\r\n"
    );
    let mut prefix = vec![0u8];
    prefix.extend_from_slice(&(over as u32).to_be_bytes());
    let chunk = [b"5\r\n".as_slice(), &prefix, b"\r\n"].concat();
    let reply = refused(&demo, &head, &chunk, "resource_exhausted").await;
    assert!(reply.starts_with("HTTP/1.1 200"), "{reply}");
    assert!(reply.contains(&format!("the request message is over {MAX_REQUEST_BYTES} bytes")), "{reply}");
}
