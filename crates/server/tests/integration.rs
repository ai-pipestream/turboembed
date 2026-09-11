//! End-to-end integration tests: boot the real server on an ephemeral port
//! with the mock backend and exercise the OIP surface over a real gRPC
//! connection — health, metadata, unary ModelInfer, bidi ModelStreamInfer,
//! and bearer auth.

use std::collections::HashMap;

use tonic::metadata::MetadataValue;
use tonic::transport::Channel;
use tonic::Request;

use inferstream_protocol::inference::grpc_inference_service_client::GrpcInferenceServiceClient;
use inferstream_protocol::inference::model_infer_request::InferInputTensor;
use inferstream_protocol::inference::{
    ModelInferRequest, ModelMetadataRequest, ModelReadyRequest, ServerLiveRequest,
    ServerReadyRequest,
};
use inferstream_protocol::tensor::{pack_bytes, unpack_bytes, unpack_fp32};
use inferstream_server::config::Config;

const BASE_CONFIG: &str = r#"
listen = "127.0.0.1:0"

[[models]]
name = "mock-embed"
backend = "mock"

[[models]]
name = "mock-generate"
backend = "mock"
"#;

const BEARER_CONFIG: &str = r#"
listen = "127.0.0.1:0"

[auth]
mode = "bearer"
bearer_tokens = ["integration-test-key"]

[[models]]
name = "mock-embed"
backend = "mock"
"#;

/// Boot the server on an ephemeral port; returns a connected client and a
/// guard that shuts the server down on drop.
async fn start_server(config_text: &str) -> (GrpcInferenceServiceClient<Channel>, ServerGuard) {
    let config = Config::from_toml(config_text).expect("test config parses");
    let (bound_tx, bound_rx) = tokio::sync::oneshot::channel();
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let handle = tokio::spawn(inferstream_server::serve(config, bound_tx, async move {
        let _ = shutdown_rx.await;
    }));
    let addr = bound_rx.await.expect("server reports bound address");
    let channel = Channel::from_shared(format!("http://{addr}"))
        .unwrap()
        .connect()
        .await
        .expect("client connects");
    (
        GrpcInferenceServiceClient::new(channel),
        ServerGuard {
            shutdown: Some(shutdown_tx),
            handle,
        },
    )
}

struct ServerGuard {
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    handle: tokio::task::JoinHandle<Result<(), inferstream_server::ServerError>>,
}

impl ServerGuard {
    async fn stop(mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        let _ = self.handle.await;
    }
}

fn text_request(model: &str, id: &str, text: &str) -> ModelInferRequest {
    ModelInferRequest {
        model_name: model.to_string(),
        id: id.to_string(),
        inputs: vec![InferInputTensor {
            name: "text".to_string(),
            datatype: "BYTES".to_string(),
            shape: vec![1],
            parameters: HashMap::new(),
            contents: None,
        }],
        raw_input_contents: vec![pack_bytes(&[text.as_bytes()])],
        ..Default::default()
    }
}

#[tokio::test]
async fn health_and_metadata() {
    let (mut client, guard) = start_server(BASE_CONFIG).await;

    let live = client
        .server_live(ServerLiveRequest {})
        .await
        .unwrap()
        .into_inner();
    assert!(live.live);

    let ready = client
        .server_ready(ServerReadyRequest {})
        .await
        .unwrap()
        .into_inner();
    assert!(ready.ready);

    let model_ready = client
        .model_ready(ModelReadyRequest {
            name: "mock-embed".into(),
            version: String::new(),
        })
        .await
        .unwrap()
        .into_inner();
    assert!(model_ready.ready);

    let missing = client
        .model_ready(ModelReadyRequest {
            name: "no-such-model".into(),
            version: String::new(),
        })
        .await
        .unwrap()
        .into_inner();
    assert!(!missing.ready);

    let metadata = client
        .model_metadata(ModelMetadataRequest {
            name: "mock-embed".into(),
            version: String::new(),
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(metadata.name, "mock-embed");
    assert_eq!(metadata.platform, "mock");
    assert_eq!(metadata.inputs[0].name, "text");
    assert_eq!(metadata.outputs[0].name, "embedding");
    assert_eq!(metadata.outputs[0].datatype, "FP32");

    guard.stop().await;
}

#[tokio::test]
async fn unary_model_infer_roundtrip() {
    let (mut client, guard) = start_server(BASE_CONFIG).await;

    let response = client
        .model_infer(text_request("mock-embed", "req-42", "hello inferstream"))
        .await
        .unwrap()
        .into_inner();

    assert_eq!(response.model_name, "mock-embed");
    assert_eq!(response.id, "req-42", "response must echo the request id");
    assert_eq!(response.outputs.len(), 1);
    assert_eq!(response.outputs[0].datatype, "FP32");
    let embedding = unpack_fp32(&response.raw_output_contents[0]).unwrap();
    assert_eq!(embedding.len() as i64, response.outputs[0].shape[0]);

    // Determinism across calls.
    let again = client
        .model_infer(text_request("mock-embed", "req-43", "hello inferstream"))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(response.raw_output_contents, again.raw_output_contents);

    // Unknown model surfaces NOT_FOUND.
    let error = client
        .model_infer(text_request("no-such-model", "x", "hi"))
        .await
        .unwrap_err();
    assert_eq!(error.code(), tonic::Code::NotFound);

    guard.stop().await;
}

#[tokio::test]
async fn bidi_model_stream_infer_roundtrip() {
    let (mut client, guard) = start_server(BASE_CONFIG).await;

    // Two requests multiplexed on one stream, plus one for a missing model.
    let requests = vec![
        text_request("mock-generate", "stream-a", "first prompt"),
        text_request("mock-generate", "stream-b", "second prompt"),
        text_request("no-such-model", "stream-c", "orphan"),
    ];
    let mut stream = client
        .model_stream_infer(tokio_stream::iter(requests))
        .await
        .unwrap()
        .into_inner();

    let mut chunks_by_id: HashMap<String, Vec<String>> = HashMap::new();
    let mut errors_by_id: HashMap<String, String> = HashMap::new();
    while let Some(message) = stream.message().await.unwrap() {
        let response = message.infer_response.expect("chunk carries a response");
        if message.error_message.is_empty() {
            let tokens = unpack_bytes(&response.raw_output_contents[0]).unwrap();
            chunks_by_id
                .entry(response.id.clone())
                .or_default()
                .push(String::from_utf8(tokens[0].clone()).unwrap());
        } else {
            errors_by_id.insert(response.id.clone(), message.error_message);
        }
    }

    // Mock backend emits 4 chunks per request; correlation is via request id.
    for id in ["stream-a", "stream-b"] {
        let tokens = &chunks_by_id[id];
        assert_eq!(tokens, &["tok-0", "tok-1", "tok-2", "tok-3"], "id={id}");
    }
    // The bad request failed per-request without tearing down the stream.
    assert!(errors_by_id["stream-c"].contains("not configured"));
    assert_eq!(chunks_by_id.len(), 2);

    guard.stop().await;
}

#[tokio::test]
async fn bearer_auth_gates_all_rpcs() {
    let (mut client, guard) = start_server(BEARER_CONFIG).await;

    // No token: rejected.
    let error = client
        .server_live(ServerLiveRequest {})
        .await
        .unwrap_err();
    assert_eq!(error.code(), tonic::Code::Unauthenticated);

    // Wrong token: rejected.
    let mut bad = Request::new(ServerLiveRequest {});
    bad.metadata_mut().insert(
        "authorization",
        MetadataValue::try_from("Bearer wrong-key").unwrap(),
    );
    let error = client.server_live(bad).await.unwrap_err();
    assert_eq!(error.code(), tonic::Code::Unauthenticated);

    // Correct token: accepted, including for inference.
    let token = MetadataValue::try_from("Bearer integration-test-key").unwrap();
    let mut live = Request::new(ServerLiveRequest {});
    live.metadata_mut().insert("authorization", token.clone());
    assert!(client.server_live(live).await.unwrap().into_inner().live);

    let mut infer = Request::new(text_request("mock-embed", "auth-1", "secured"));
    infer.metadata_mut().insert("authorization", token);
    let response = client.model_infer(infer).await.unwrap().into_inner();
    assert_eq!(response.id, "auth-1");

    guard.stop().await;
}
