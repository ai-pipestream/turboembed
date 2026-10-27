//! The ceiling on one call's request: a client stream at the ceiling is
//! read to its end, and one frame more is refused RESOURCE_EXHAUSTED, with
//! the server holding a frame at a time, not the stream.

use prost::Message;
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES, DEFAULT_MAX_STREAM_BYTES, FRAME_BYTES};
use turbo_kserve::proto::result::AnalyzeStreamResponse;
use turbo_kserve::proto::sink::DocumentSinkItem;
use turbo_kserve::proto::sink::turbo_sink_service_client::TurboSinkServiceClient;
use turbo_kserve::sink::SinkReceiver;
use turbo_kserve::{Serve, Server};

/// An item whose message is `bytes` long, sequence under 16384.
fn item(sequence: u64, bytes: usize) -> DocumentSinkItem {
    let mut i = DocumentSinkItem {
        sequence,
        doc_id: String::new(),
        analyzed: Some(AnalyzeStreamResponse::default()),
        ..Default::default()
    };
    // The field's tag and length prefix take two to five bytes.
    let base = i.encoded_len();
    let pad = (2..=5).map(|head| bytes - base - head).find(|&p| {
        i.doc_id = "d".repeat(p);
        i.encoded_len() == bytes
    });
    assert!(pad.is_some(), "no padding makes {bytes} bytes");
    i
}

async fn sink(max_stream_bytes: Option<u64>) -> (Server, TurboSinkServiceClient<tonic::transport::Channel>) {
    let serve =
        Serve { admin: true, sink_receiver: Some(SinkReceiver::new(|_| {})), max_stream_bytes, ..Serve::default() };
    let s = Server::start("127.0.0.1:0".parse().unwrap(), Vec::new(), DEFAULT_MAX_MESSAGE_BYTES, serve).await.unwrap();
    let c = TurboSinkServiceClient::connect(format!("http://{}", s.local_addr()))
        .await
        .unwrap()
        .max_encoding_message_size(usize::MAX);
    (s, c)
}

/// Streams `n` items of `bytes` each, made as they are sent.
async fn send(
    c: &mut TurboSinkServiceClient<tonic::transport::Channel>,
    n: u64,
    bytes: usize,
) -> Result<u64, tonic::Status> {
    let items = tokio_stream::iter((0..n).map(move |k| item(1000 + k, bytes)));
    Ok(c.stream_documents(items).await?.into_inner().accepted_count)
}

#[tokio::test]
async fn a_stream_at_the_ceiling_is_read_and_one_frame_more_is_refused() {
    let bytes = 100_000;
    let frame = bytes as u64 + 5;
    let (s, mut c) = sink(Some(10 * frame)).await;
    assert_eq!(send(&mut c, 10, bytes).await.unwrap(), 10);
    let e = send(&mut c, 11, bytes).await.unwrap_err();
    assert_eq!(e.code(), tonic::Code::ResourceExhausted, "{e:?}");
    assert!(e.message().contains("max_stream_bytes"), "{e:?}");
    assert!(e.metadata().get("turbo-code").is_none());
    // The connection serves the next call.
    assert_eq!(send(&mut c, 1, bytes).await.unwrap(), 1);
    s.stop().await;
}

/// The default 2 GiB ceiling in frames of `FRAME_BYTES`, with the
/// process's peak resident memory well under the stream. Four GiB cross
/// loopback, so it runs only with TURBO_TEST_2GIB=1.
#[tokio::test]
async fn the_default_ceiling_streams_two_gib_in_bounded_memory() {
    if std::env::var("TURBO_TEST_2GIB").as_deref() != Ok("1") {
        eprintln!("skipped: set TURBO_TEST_2GIB=1");
        return;
    }
    let frame = FRAME_BYTES as u64 + 5;
    let n = DEFAULT_MAX_STREAM_BYTES / frame;
    let (s, mut c) = sink(None).await;
    assert_eq!(send(&mut c, n, FRAME_BYTES).await.unwrap(), n);
    let e = send(&mut c, n + 1, FRAME_BYTES).await.unwrap_err();
    assert_eq!(e.code(), tonic::Code::ResourceExhausted, "{e:?}");
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let hwm = status.lines().find_map(|l| l.strip_prefix("VmHWM:")).unwrap();
    let kib: u64 = hwm.trim().trim_end_matches("kB").trim().parse().unwrap();
    assert!(kib < 512 << 10, "peak resident {kib} KiB");
    s.stop().await;
}
