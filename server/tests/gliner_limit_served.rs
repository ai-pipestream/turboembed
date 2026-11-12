//! A sentence over one of the GLiNER step's input limits, served (#2125).
//! The step fails as a capacity refusal: the step failure and its layer
//! errors carry TURBO_E_CAPACITY and kind LIMIT, the message names the
//! limit and both counts, and none of them quotes the document.
//!
//! The package is the GLiNER fixture, whose synthetic DeBERTa encoder has
//! 32 positions. The step never splits a sentence, so one sentence whose
//! pieces do not fit beside the schema prompt is over the encoder's
//! sequence, and one of more than 384 tokens is over the token limit.

#[path = "gliner_progressive/fixture.rs"]
mod fixture;

use tokio_stream::StreamExt as _;
use turbo_kserve::api::TURBO_E_CAPACITY;
use turbo_kserve::config::DEFAULT_MAX_MESSAGE_BYTES as MAX;
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::result::analyze_progressive_event::Event;
use turbo_kserve::proto::result::*;
use turbo_kserve::{Serve, Server};

async fn serve() -> (Server, AnalysisServiceClient<tonic::transport::Channel>) {
    let server =
        Server::start("127.0.0.1:0".parse().unwrap(), Vec::new(), MAX, Serve { admin: true, ..Serve::default() })
            .await
            .unwrap();
    let mut client = AnalysisServiceClient::connect(format!("http://{}", server.local_addr()))
        .await
        .unwrap()
        .max_decoding_message_size(MAX)
        .max_encoding_message_size(MAX);
    client
        .load_bundle(LoadBundleRequest {
            name: "gliner".into(),
            bundle: fixture::package().to_string_lossy().into_owned(),
            device: 0,
            sessions: 1,
            ..Default::default()
        })
        .await
        .unwrap();
    (server, client)
}

/// The GLiNER step's one failure for `text`, and every failed layer's
/// error from that failure and the complete graph.
async fn refused(
    client: &mut AnalysisServiceClient<tonic::transport::Channel>,
    text: String,
) -> (AnalysisStepFailure, Vec<LayerError>) {
    let request = AnalyzePipelineRequest {
        doc_id: "gliner-limit".into(),
        text,
        model: "gliner".into(),
        steps: vec![
            "turbo-sentdetect:rules".into(),
            "turbo-tokenize:uax29-untyped".into(),
            "turbo:subwords".into(),
            "turbo:gliner".into(),
        ],
        open_span: Some(OpenSpanSchema {
            labels: vec!["person".into(), "location".into()],
            threshold: Some(0.0),
            nested: false,
            prompts: Vec::new(),
        }),
        ..Default::default()
    };
    let mut stream = client.analyze_document_progressive(request).await.unwrap().into_inner();
    let (mut failures, mut complete) = (Vec::new(), None);
    while let Some(event) = stream.next().await {
        match event.unwrap().event {
            Some(Event::StepFailed(f)) => failures.push(f),
            Some(Event::Complete(done)) => complete = Some(done),
            _ => {}
        }
    }
    let [failure] = &failures[..] else { panic!("one failed step: {failures:?}") };
    assert_eq!(failure.step_id, "turbo:gliner");
    let complete = complete.expect("a complete graph");
    let failed = failure
        .layers
        .iter()
        .chain(&complete.layers)
        .filter(|l| l.status == LayerWireStatus::Failed as i32)
        .map(|l| l.error.clone().unwrap())
        .collect();
    (failure.clone(), failed)
}

#[tokio::test]
async fn a_sentence_over_a_gliner_limit_is_a_capacity_refusal_with_counts_only() {
    let (server, mut client) = serve().await;
    let secret = "Zanzibarquux";
    // One sentence each: 40 words against 32 positions, then 400 tokens
    // against the 384 one window holds.
    for (words, says) in [(40, "over its encoder max_sequence 32, got "), (400, "over its max_tokens 384, got 4")] {
        let text = format!("{secret} {}.", vec!["Ann met Bob"; words / 3].join(" "));
        let (failure, failed) = refused(&mut client, text).await;
        let wire = failure.failure.as_ref().unwrap();
        assert_eq!(wire.turbo_code, TURBO_E_CAPACITY, "{wire:?}");
        assert!(wire.message.contains(says), "{}", wire.message);
        let error = failure.error.as_ref().unwrap();
        assert_eq!((error.kind, error.turbo_code), (LayerErrorKind::Limit as i32, TURBO_E_CAPACITY));
        assert!(!failed.is_empty(), "{failure:?}");
        for e in &failed {
            assert_eq!((e.kind, e.turbo_code), (LayerErrorKind::Limit as i32, TURBO_E_CAPACITY), "{e:?}");
        }
        let shown = format!("{failure:?} {failed:?}");
        assert!(!shown.contains(secret) && !shown.contains("Ann"), "{shown}");
    }
    server.stop().await;
}
