//! OIP edge-case integration tests over a real gRPC connection: malformed
//! `ModelInfer` requests map onto gRPC error statuses, batch and empty-input
//! behavior follows the mock contract, metadata/readiness separate served
//! from unknown models, streaming responses echo the request id with final
//! flags, and bearer auth gates the streaming RPC as well as the unary one.

use std::collections::HashMap;

use tonic::metadata::MetadataValue;
use tonic::transport::Channel;
use tonic::Request;

use inferstream_protocol::inference::grpc_inference_service_client::GrpcInferenceServiceClient;
use inferstream_protocol::inference::infer_parameter::ParameterChoice;
use inferstream_protocol::inference::model_infer_request::InferInputTensor;
use inferstream_protocol::inference::{
    InferTensorContents, ModelInferRequest, ModelMetadataRequest, ModelReadyRequest,
};
use inferstream_protocol::tensor::{pack_bytes, pack_fp32, unpack_bytes, unpack_fp32};
use inferstream_server::config::Config;

/// Embedding dimension of `MockBackend::default()` — the only kind of model
/// these tests serve, so the constant is the mock contract, not a guess.
const MOCK_DIM: i64 = 8;

const BASE_CONFIG: &str = r#"
listen = "127.0.0.1:0"

[[models]]
name = "mock-embed"
backend = "mock"
"#;

const BEARER_CONFIG: &str = r#"
listen = "127.0.0.1:0"

[auth]
mode = "bearer"
bearer_tokens = ["oip-edge-key"]

[[models]]
name = "mock-embed"
backend = "mock"
"#;

/// Boot the server on an ephemeral port; returns a connected client and a
/// guard that shuts the server down on drop. Mirrors the harness in
/// `integration.rs` — each integration test file is its own crate, so the
/// helper is duplicated rather than shared.
async fn start_server(config_text: &str) -> (GrpcInferenceServiceClient<Channel>, ServerGuard) {
    let config = Config::from_toml(config_text).expect("test config parses");
    let registry = inferstream_server::build_registry(&config, &inferstream_server::mock_factory())
        .expect("registry builds");
    let (bound_tx, bound_rx) = tokio::sync::oneshot::channel();
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let handle = tokio::spawn(inferstream_server::serve(
        config,
        registry,
        bound_tx,
        async move {
            let _ = shutdown_rx.await;
        },
    ));
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

/// A `text` input tensor with the given datatype and a placeholder shape;
/// tests wire `raw_input_contents` (or `contents`) themselves.
fn text_tensor(datatype: &str) -> InferInputTensor {
    InferInputTensor {
        name: "text".to_string(),
        datatype: datatype.to_string(),
        shape: vec![1],
        parameters: HashMap::new(),
        contents: None,
    }
}

fn text_request(model: &str, id: &str, text: &str) -> ModelInferRequest {
    ModelInferRequest {
        model_name: model.to_string(),
        id: id.to_string(),
        inputs: vec![text_tensor("BYTES")],
        raw_input_contents: vec![pack_bytes(&[text.as_bytes()])],
        ..Default::default()
    }
}

#[tokio::test]
async fn missing_text_input_is_invalid_argument() {
    let (mut client, guard) = start_server(BASE_CONFIG).await;

    // No input tensors at all.
    let no_inputs = ModelInferRequest {
        model_name: "mock-embed".into(),
        id: "no-inputs".into(),
        ..Default::default()
    };
    let error = client.model_infer(no_inputs).await.unwrap_err();
    assert_eq!(error.code(), tonic::Code::InvalidArgument);
    assert!(
        error
            .message()
            .contains("expected an input tensor named \"text\""),
        "unexpected message: {}",
        error.message()
    );

    // A tensor under another name does not satisfy the contract either.
    let wrong_name = ModelInferRequest {
        model_name: "mock-embed".into(),
        id: "wrong-name".into(),
        inputs: vec![InferInputTensor {
            name: "prompt".into(),
            ..text_tensor("BYTES")
        }],
        raw_input_contents: vec![pack_bytes(&[b"hi".as_slice()])],
        ..Default::default()
    };
    let error = client.model_infer(wrong_name).await.unwrap_err();
    assert_eq!(error.code(), tonic::Code::InvalidArgument);
    assert!(
        error
            .message()
            .contains("expected an input tensor named \"text\""),
        "unexpected message: {}",
        error.message()
    );

    guard.stop().await;
}

#[tokio::test]
async fn wrong_datatype_is_invalid_argument() {
    let (mut client, guard) = start_server(BASE_CONFIG).await;

    let request = ModelInferRequest {
        model_name: "mock-embed".into(),
        id: "wrong-dtype".into(),
        inputs: vec![text_tensor("FP32")],
        raw_input_contents: vec![pack_fp32(&[1.0, 2.0])],
        ..Default::default()
    };
    let error = client.model_infer(request).await.unwrap_err();
    assert_eq!(error.code(), tonic::Code::InvalidArgument);
    assert!(
        error.message().contains("must be BYTES"),
        "unexpected message: {}",
        error.message()
    );

    guard.stop().await;
}

#[tokio::test]
async fn typed_contents_without_raw_is_invalid_argument() {
    let (mut client, guard) = start_server(BASE_CONFIG).await;

    // OIP permits typed `contents` as an alternative to raw bytes; the mock
    // contract requires the raw representation and must say so, not silently
    // ignore the typed payload.
    let via_contents = ModelInferRequest {
        model_name: "mock-embed".into(),
        id: "typed".into(),
        inputs: vec![InferInputTensor {
            contents: Some(InferTensorContents {
                bytes_contents: vec![b"hi".to_vec()],
                ..Default::default()
            }),
            ..text_tensor("BYTES")
        }],
        ..Default::default()
    };
    let error = client.model_infer(via_contents).await.unwrap_err();
    assert_eq!(error.code(), tonic::Code::InvalidArgument);
    assert!(
        error.message().contains("raw_input_contents"),
        "unexpected message: {}",
        error.message()
    );

    guard.stop().await;
}

#[tokio::test]
async fn malformed_bytes_payload_is_invalid_argument() {
    let (mut client, guard) = start_server(BASE_CONFIG).await;

    // Two trailing bytes: the 4-byte little-endian length prefix is cut off,
    // so the raw blob is not a valid packed BYTES element.
    let request = ModelInferRequest {
        model_name: "mock-embed".into(),
        id: "garbage".into(),
        inputs: vec![text_tensor("BYTES")],
        raw_input_contents: vec![b"hi".to_vec()],
        ..Default::default()
    };
    let error = client.model_infer(request).await.unwrap_err();
    assert_eq!(error.code(), tonic::Code::InvalidArgument);
    assert!(
        error.message().contains("malformed BYTES payload"),
        "unexpected message: {}",
        error.message()
    );

    guard.stop().await;
}

#[tokio::test]
async fn empty_batch_is_invalid_argument() {
    let (mut client, guard) = start_server(BASE_CONFIG).await;

    // A shape-[0] BYTES tensor: the raw payload unpacks to zero elements.
    let request = ModelInferRequest {
        model_name: "mock-embed".into(),
        id: "empty-batch".into(),
        inputs: vec![InferInputTensor {
            shape: vec![0],
            ..text_tensor("BYTES")
        }],
        raw_input_contents: vec![Vec::new()],
        ..Default::default()
    };
    let error = client.model_infer(request).await.unwrap_err();
    assert_eq!(error.code(), tonic::Code::InvalidArgument);
    assert!(
        error.message().contains("contained no elements"),
        "unexpected message: {}",
        error.message()
    );

    guard.stop().await;
}

#[tokio::test]
async fn batch_of_three_repeats_identical_rows() {
    let (mut client, guard) = start_server(BASE_CONFIG).await;

    let request = ModelInferRequest {
        model_name: "mock-embed".into(),
        id: "batch".into(),
        inputs: vec![InferInputTensor {
            shape: vec![3],
            ..text_tensor("BYTES")
        }],
        raw_input_contents: vec![pack_bytes(&[
            b"alpha".as_slice(),
            b"beta",
            b"alpha", // duplicate of the first: rows must match
        ])],
        ..Default::default()
    };
    let response = client.model_infer(request).await.unwrap().into_inner();

    let dim = MOCK_DIM as usize;
    let output = &response.outputs[0];
    assert_eq!(output.shape, vec![3, MOCK_DIM]);
    let values = unpack_fp32(&response.raw_output_contents[0]).unwrap();
    assert_eq!(values.len(), 3 * dim);
    assert_eq!(
        values[..dim],
        values[2 * dim..],
        "identical texts embed identically"
    );
    assert_ne!(
        values[..dim],
        values[dim..2 * dim],
        "different texts differ"
    );

    // A batch row must equal the unary embedding of the same text — the
    // batch path changes the shape, not the values.
    let single = client
        .model_infer(text_request("mock-embed", "single", "alpha"))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(single.outputs[0].shape, vec![MOCK_DIM]);
    assert_eq!(
        single.raw_output_contents[0],
        response.raw_output_contents[0][..4 * dim]
    );

    guard.stop().await;
}

#[tokio::test]
async fn empty_string_is_a_valid_single_text() {
    let (mut client, guard) = start_server(BASE_CONFIG).await;

    // Zero bytes of text is still one element; only a zero-element batch is
    // rejected. Guards against the empty-check over-reaching into content.
    let response = client
        .model_infer(text_request("mock-embed", "empty-str", ""))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(response.outputs[0].shape, vec![MOCK_DIM]);
    assert_eq!(response.raw_output_contents[0].len(), 4 * MOCK_DIM as usize);

    guard.stop().await;
}

#[tokio::test]
async fn metadata_and_ready_separate_served_from_unknown() {
    let (mut client, guard) = start_server(BASE_CONFIG).await;

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
    assert_eq!(metadata.versions, ["1"]);

    // Unknown model: metadata is an error status (unlike readiness, which
    // answers `ready: false` for the same name).
    let error = client
        .model_metadata(ModelMetadataRequest {
            name: "no-such-model".into(),
            version: String::new(),
        })
        .await
        .unwrap_err();
    assert_eq!(error.code(), tonic::Code::NotFound);

    // The mock is always ready; an explicit version must not break routing.
    let served = client
        .model_ready(ModelReadyRequest {
            name: "mock-embed".into(),
            version: "1".into(),
        })
        .await
        .unwrap()
        .into_inner();
    assert!(served.ready);

    let unknown = client
        .model_ready(ModelReadyRequest {
            name: "no-such-model".into(),
            version: String::new(),
        })
        .await
        .unwrap()
        .into_inner();
    assert!(!unknown.ready);

    guard.stop().await;
}

#[tokio::test]
async fn stream_infer_echoes_request_id_and_sets_final_flag() {
    let (mut client, guard) = start_server(BASE_CONFIG).await;

    let mut stream = client
        .model_stream_infer(tokio_stream::iter(vec![text_request(
            "mock-embed",
            "flags-1",
            "stream me",
        )]))
        .await
        .unwrap()
        .into_inner();

    let mut ids = Vec::new();
    let mut tokens = Vec::new();
    let mut finals = Vec::new();
    while let Some(message) = stream.message().await.unwrap() {
        assert!(
            message.error_message.is_empty(),
            "chunk failed: {}",
            message.error_message
        );
        let response = message.infer_response.expect("chunk carries a response");
        ids.push(response.id.clone());
        let elements = unpack_bytes(&response.raw_output_contents[0]).unwrap();
        tokens.push(String::from_utf8(elements[0].clone()).unwrap());
        finals.push(matches!(
            response
                .parameters
                .get("final")
                .and_then(|p| p.parameter_choice.as_ref()),
            Some(ParameterChoice::BoolParam(true))
        ));
    }

    assert_eq!(ids, ["flags-1"; 4], "every chunk echoes the request id");
    assert_eq!(tokens, ["tok-0", "tok-1", "tok-2", "tok-3"]);
    assert_eq!(
        finals,
        [false, false, false, true],
        "only the last chunk is final"
    );

    guard.stop().await;
}

#[tokio::test]
async fn stream_infer_with_no_requests_closes_cleanly() {
    let (mut client, guard) = start_server(BASE_CONFIG).await;

    let mut stream = client
        .model_stream_infer(tokio_stream::iter(Vec::<ModelInferRequest>::new()))
        .await
        .unwrap()
        .into_inner();
    assert!(
        stream.message().await.unwrap().is_none(),
        "an idle request stream must end without any response"
    );

    guard.stop().await;
}

#[tokio::test]
async fn stream_infer_reports_bad_request_without_tearing_down() {
    let (mut client, guard) = start_server(BASE_CONFIG).await;

    // A malformed request mid-stream fails as a per-request error chunk (id
    // echoed, no outputs); the follow-up good request still streams. The
    // error chunk and the good chunks are produced by independent tasks, so
    // collect them unordered and assert on the set, not the interleaving.
    let empty_batch = ModelInferRequest {
        model_name: "mock-embed".into(),
        id: "bad-batch".into(),
        inputs: vec![InferInputTensor {
            shape: vec![0],
            ..text_tensor("BYTES")
        }],
        raw_input_contents: vec![Vec::new()],
        ..Default::default()
    };
    let good = text_request("mock-embed", "good-req", "still works");

    let mut stream = client
        .model_stream_infer(tokio_stream::iter(vec![empty_batch, good]))
        .await
        .unwrap()
        .into_inner();

    let mut error_chunks: Vec<(String, String)> = Vec::new();
    let mut good_chunks = 0;
    while let Some(message) = stream.message().await.unwrap() {
        let response = message.infer_response.expect("chunk carries a response");
        if message.error_message.is_empty() {
            assert_eq!(response.id, "good-req");
            good_chunks += 1;
        } else {
            error_chunks.push((response.id, message.error_message));
        }
    }

    assert_eq!(error_chunks.len(), 1, "exactly one per-request error");
    let (error_id, error_message) = &error_chunks[0];
    assert_eq!(
        error_id, "bad-batch",
        "error chunk echoes the failing request id"
    );
    assert!(
        error_message.contains("contained no elements"),
        "unexpected error message: {error_message}"
    );
    assert_eq!(good_chunks, 4, "stream stayed up for the follow-up request");

    guard.stop().await;
}

#[tokio::test]
async fn bearer_auth_gates_model_stream_infer() {
    let (mut client, guard) = start_server(BEARER_CONFIG).await;

    // No token: the stream call itself is rejected before any request lands.
    let error = client
        .model_stream_infer(tokio_stream::iter(vec![text_request(
            "mock-embed",
            "noauth",
            "hi",
        )]))
        .await
        .unwrap_err();
    assert_eq!(error.code(), tonic::Code::Unauthenticated);

    // Correct token: the same stream flows normally.
    let token = MetadataValue::try_from("Bearer oip-edge-key").unwrap();
    let mut request = Request::new(tokio_stream::iter(vec![text_request(
        "mock-embed",
        "authed",
        "hi",
    )]));
    request.metadata_mut().insert("authorization", token);

    let mut stream = client
        .model_stream_infer(request)
        .await
        .unwrap()
        .into_inner();
    let mut chunks = 0;
    while let Some(message) = stream.message().await.unwrap() {
        assert!(message.error_message.is_empty());
        assert_eq!(message.infer_response.unwrap().id, "authed");
        chunks += 1;
    }
    assert_eq!(chunks, 4);

    guard.stop().await;
}
