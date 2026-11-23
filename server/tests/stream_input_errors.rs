//! An AnalyzeStream whose input fails is answered with the input's own
//! status (issue #2517, F10). The reader used to answer every inbound
//! error as "the client cancelled the stream", so a document over the
//! server's message limit looked like the caller's own cancel and the
//! limit was never named.

mod common;

use common::*;
use tonic::Code;
use turbo_kserve::config::{Device, ModelConfig};
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::{Serve, Server};

/// The smallest message limit the tiny bundle's documents fit under.
const LIMIT: usize = 4096;

async fn serve_with_limit() -> (Server, Analysis) {
    let model = ModelConfig {
        bundle: tiny().to_str().unwrap().to_string(),
        name: None,
        device: Device::Index(cpu()),
        precision: 0,
        max_batch: 0,
        max_seq: 0,
        sessions: 1,
    };
    let s = Server::start("127.0.0.1:0".parse().unwrap(), vec![model], LIMIT, Serve::default()).await.unwrap();
    s.load().await.unwrap();
    let c = AnalysisServiceClient::connect(format!("http://{}", s.local_addr()))
        .await
        .unwrap()
        .max_encoding_message_size(usize::MAX)
        .max_decoding_message_size(usize::MAX);
    (s, c)
}

#[tokio::test]
async fn a_message_over_the_limit_names_the_limit_not_a_cancel() {
    let (s, mut c) = serve_with_limit().await;
    let mut stream = open_stream(&mut c, 0, 0, 0).await;
    stream.send("small", NAME, "a short text", None).await;
    stream.send("big", NAME, &"word ".repeat(LIMIT), None).await;
    let mut responses = stream.responses;
    let e = within("the stream's end", std::time::Duration::from_secs(10), async {
        loop {
            match responses.message().await {
                Ok(Some(_)) => continue,
                Ok(None) => panic!("a stream whose input failed ended OK"),
                Err(e) => break e,
            }
        }
    })
    .await;
    assert_ne!(e.code(), Code::Cancelled, "the server's own limit reported as the client's cancel: {e:?}");
    // docs/grpc.md: a message over --max-message-bytes is RESOURCE_EXHAUSTED
    // with no turbo-code on every service, a stream's trailers included;
    // OUT_OF_RANGE is TURBO_E_CAPACITY's.
    assert_eq!(e.code(), Code::ResourceExhausted, "{e:?}");
    assert!(e.metadata().get("turbo-code").is_none(), "{e:?}");
    assert!(
        e.message().contains("too large") || e.message().contains(&LIMIT.to_string()),
        "the limit is not named: {e:?}"
    );
    drop(stream.tx);
    let m = s.metrics();
    assert!(
        m.contains(r#"method="AnalyzeStream",code="RESOURCE_EXHAUSTED"} 1"#),
        "the metric records the status the client saw: {m}"
    );
    s.stop().await;
}
