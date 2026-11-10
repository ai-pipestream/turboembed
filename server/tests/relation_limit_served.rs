//! A document over the relation encoder's input limit, served (#2125).
//! The zero-shot relation step fails as a capacity refusal: the step
//! failure, its layer error and the complete graph's layer error all carry
//! TURBO_E_CAPACITY and kind LIMIT, the message names the limit and both
//! counts, and none of them quotes the document.
//!
//! The GLiNER2 package is a fixture: the synthetic DeBERTa encoder of the
//! GLiNER package fixture (32 positions) and the reduced GLiNER2 heads.

#[path = "gliner_progressive/fixture.rs"]
mod fixture;

use std::fs;
use std::path::{Path, PathBuf};

use tokio_stream::StreamExt as _;
use turbo_kserve::api::TURBO_E_CAPACITY;
use turbo_kserve::config::DEFAULT_MAX_MESSAGE_BYTES as MAX;
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::result::analyze_progressive_event::Event;
use turbo_kserve::proto::result::*;
use turbo_kserve::{Serve, Server};

fn relation_package() -> PathBuf {
    let entity = fixture::package();
    let dir = std::env::temp_dir().join(format!("turbo-kserve-gliner2-package-{}", std::process::id()));
    fs::create_dir_all(dir.join("encoder")).unwrap();
    for f in ["deberta.json", "weights.safetensors"] {
        fs::copy(entity.join("encoder").join(f), dir.join("encoder").join(f)).unwrap();
    }
    fs::copy(entity.join("spm.model"), dir.join("spm.model")).unwrap();
    let heads =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../analysis/relation/tests/fixtures/gliner2/reduced.safetensors");
    fs::copy(heads, dir.join("model.safetensors")).unwrap();
    fs::write(dir.join("config.json"), r#"{"counting_layer":"count_lstm_v2","max_width":8}"#).unwrap();
    fs::write(dir.join("added_tokens.json"), r#"{"[SEP_STRUCT]":5,"[SEP_TEXT]":6,"[P]":7,"[R]":8}"#).unwrap();
    dir
}

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
            name: "relations".into(),
            bundle: relation_package().to_string_lossy().into_owned(),
            device: 0,
            sessions: 1,
            ..Default::default()
        })
        .await
        .unwrap();
    (server, client)
}

#[tokio::test]
async fn a_document_over_the_encoder_limit_is_a_capacity_refusal_with_counts_only() {
    let (_server, mut client) = serve().await;
    let secret = "Zanzibarquux";
    let text = format!(
        "Mary flew from Paris to Berlin and then {secret} went on to London and Madrid and Rome and Vienna \
         and Prague and Lisbon and Dublin and Oslo before she came home to Paris."
    );
    let request = AnalyzePipelineRequest {
        doc_id: "relation-limit".into(),
        text,
        relation_model: "relations".into(),
        steps: vec![
            "turbo-sentdetect:rules".into(),
            "turbo-tokenize:uax29-untyped".into(),
            "turbo:entities".into(),
            "turbo:zero-shot-relations".into(),
        ],
        relations: Some(OpenRelationSchema {
            labels: vec![OpenRelationLabel { name: "near".into(), ..Default::default() }],
            threshold: Some(0.5),
            within_sentence: true,
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
    let wire = failure.failure.as_ref().unwrap();
    assert_eq!(wire.turbo_code, TURBO_E_CAPACITY, "{wire:?}");
    let message = &wire.message;
    assert!(message.contains("over its encoder max_sequence 32, got "), "{message}");
    let error = failure.error.as_ref().unwrap();
    assert_eq!((error.kind, error.turbo_code), (LayerErrorKind::Limit as i32, TURBO_E_CAPACITY));

    let complete = complete.expect("a complete graph");
    let failed: Vec<&LayerError> = failure
        .layers
        .iter()
        .chain(&complete.layers)
        .filter(|l| l.status == LayerWireStatus::Failed as i32)
        .map(|l| l.error.as_ref().unwrap())
        .collect();
    assert!(failed.len() >= 2, "{:?}", complete.layers);
    for e in failed {
        assert_eq!((e.kind, e.turbo_code), (LayerErrorKind::Limit as i32, TURBO_E_CAPACITY), "{e:?}");
    }
    let shown = format!("{failures:?}");
    assert!(!shown.contains(secret) && !shown.contains("Paris"), "{shown}");
}
