//! End-to-end integration tests for the `inferstream.v1.InferstreamService`
//! extension: Tokenize / Detokenize / Embed / ListModels / Rerank over a real
//! gRPC connection against the mock backend, plus bearer-auth gating and the
//! OIP service still answering on the same endpoint.

use tonic::metadata::MetadataValue;
use tonic::transport::Channel;
use tonic::Request;

use inferstream_protocol::extension::inferstream_service_client::InferstreamServiceClient;
use inferstream_protocol::extension::{
    DetokenizeRequest, EmbedRequest, ListModelsRequest, RerankRequest, TokenIds, TokenizeRequest,
};
use inferstream_protocol::inference::grpc_inference_service_client::GrpcInferenceServiceClient;
use inferstream_protocol::inference::ServerMetadataRequest;
use inferstream_server::config::Config;

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
bearer_tokens = ["extension-test-key"]

[[models]]
name = "mock-embed"
backend = "mock"
"#;

async fn start_server(config_text: &str) -> (Channel, ServerGuard) {
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
        channel,
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

#[tokio::test]
async fn tokenize_detokenize_round_trip_over_grpc() {
    let (channel, guard) = start_server(BASE_CONFIG).await;
    let mut client = InferstreamServiceClient::new(channel);

    let response = client
        .tokenize(TokenizeRequest {
            model_name: "mock-embed".into(),
            texts: vec!["hello grpc".into(), "unicode ✓".into()],
            with_offsets: true,
            ..Default::default()
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(response.encodings.len(), 2);
    let first = &response.encodings[0];
    assert!(!first.input_ids.is_empty());
    assert_eq!(first.input_ids.len(), first.attention_mask.len());
    assert_eq!(first.input_ids.len(), first.tokens.len());
    assert_eq!(first.input_ids.len(), first.offsets.len());

    let decoded = client
        .detokenize(DetokenizeRequest {
            model_name: "mock-embed".into(),
            sequences: response
                .encodings
                .iter()
                .map(|e| TokenIds {
                    ids: e.input_ids.clone(),
                })
                .collect(),
            skip_special_tokens: true,
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(decoded.texts, vec!["hello grpc", "unicode ✓"]);

    guard.stop().await;
}

#[tokio::test]
async fn tokenize_batch_padding_over_grpc() {
    let (channel, guard) = start_server(BASE_CONFIG).await;
    let mut client = InferstreamServiceClient::new(channel);

    let response = client
        .tokenize(TokenizeRequest {
            model_name: "mock-embed".into(),
            texts: vec!["a".into(), "a much longer sentence".into()],
            pad_to_longest: true,
            ..Default::default()
        })
        .await
        .unwrap()
        .into_inner();
    let lens: Vec<usize> = response
        .encodings
        .iter()
        .map(|e| e.input_ids.len())
        .collect();
    assert_eq!(lens[0], lens[1], "batch padded to the longest sequence");
    assert_eq!(*response.encodings[0].attention_mask.last().unwrap(), 0);

    guard.stop().await;
}

#[tokio::test]
async fn tokenize_validates_input() {
    let (channel, guard) = start_server(BASE_CONFIG).await;
    let mut client = InferstreamServiceClient::new(channel);

    let empty = client
        .tokenize(TokenizeRequest {
            model_name: "mock-embed".into(),
            texts: vec![],
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert_eq!(empty.code(), tonic::Code::InvalidArgument);

    let missing = client
        .tokenize(TokenizeRequest {
            model_name: "no-such-model".into(),
            texts: vec!["x".into()],
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert_eq!(missing.code(), tonic::Code::NotFound);

    guard.stop().await;
}

#[tokio::test]
async fn embed_returns_typed_vectors() {
    let (channel, guard) = start_server(BASE_CONFIG).await;
    let mut client = InferstreamServiceClient::new(channel);

    let response = client
        .embed(EmbedRequest {
            model_name: "mock-embed".into(),
            texts: vec!["first".into(), "second".into(), "first".into()],
            pooling: "mean".into(),
            normalize: Some(true),
            truncate_to: 128,
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(response.dim, 8, "mock embedding dim");
    assert_eq!(response.embeddings.len(), 3);
    for embedding in &response.embeddings {
        assert_eq!(embedding.values.len(), 8);
    }
    assert_eq!(
        response.embeddings[0].values, response.embeddings[2].values,
        "same text embeds identically"
    );
    assert_ne!(response.embeddings[0].values, response.embeddings[1].values);

    // Single text also works (backend returns flat [d]).
    let single = client
        .embed(EmbedRequest {
            model_name: "mock-embed".into(),
            texts: vec!["solo".into()],
            ..Default::default()
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(single.embeddings.len(), 1);
    assert_eq!(single.dim, 8);

    guard.stop().await;
}

#[tokio::test]
async fn embed_validates_input() {
    let (channel, guard) = start_server(BASE_CONFIG).await;
    let mut client = InferstreamServiceClient::new(channel);

    let empty = client
        .embed(EmbedRequest {
            model_name: "mock-embed".into(),
            texts: vec![],
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert_eq!(empty.code(), tonic::Code::InvalidArgument);

    let missing = client
        .embed(EmbedRequest {
            model_name: "no-such-model".into(),
            texts: vec!["x".into()],
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert_eq!(missing.code(), tonic::Code::NotFound);

    guard.stop().await;
}

#[tokio::test]
async fn list_models_reports_backend_and_dims() {
    let (channel, guard) = start_server(BASE_CONFIG).await;
    let mut client = InferstreamServiceClient::new(channel);

    let response = client
        .list_models(ListModelsRequest {})
        .await
        .unwrap()
        .into_inner();
    assert_eq!(response.models.len(), 1);
    let model = &response.models[0];
    assert_eq!(model.name, "mock-embed");
    assert_eq!(model.backend, "mock");
    assert!(model.ready);
    assert_eq!(model.platform, "mock");
    assert_eq!(model.embedding_dim, 8);
    assert!(model.has_tokenizer, "mock implements the tokenize surface");

    guard.stop().await;
}

#[tokio::test]
async fn rerank_orders_by_score_and_honors_top_n() {
    let (channel, guard) = start_server(BASE_CONFIG).await;
    let mut client = InferstreamServiceClient::new(channel);

    let response = client
        .rerank(RerankRequest {
            model_name: "mock-embed".into(),
            query: "rust inference".into(),
            documents: vec![
                "cooking pasta".into(),         // 0 hits
                "rust inference server".into(), // 2 hits
                "some rust code".into(),        // 1 hit
            ],
            top_n: 0,
        })
        .await
        .unwrap()
        .into_inner();
    let order: Vec<u32> = response.results.iter().map(|r| r.index).collect();
    assert_eq!(order, vec![1, 2, 0], "descending score order");
    assert!(response.results[0].score > response.results[1].score);

    let top1 = client
        .rerank(RerankRequest {
            model_name: "mock-embed".into(),
            query: "rust inference".into(),
            documents: vec!["cooking".into(), "rust inference".into()],
            top_n: 1,
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(top1.results.len(), 1);
    assert_eq!(top1.results[0].index, 1);

    guard.stop().await;
}

#[tokio::test]
async fn rerank_validates_input() {
    let (channel, guard) = start_server(BASE_CONFIG).await;
    let mut client = InferstreamServiceClient::new(channel);

    let empty = client
        .rerank(RerankRequest {
            model_name: "mock-embed".into(),
            query: "q".into(),
            documents: vec![],
            top_n: 0,
        })
        .await
        .unwrap_err();
    assert_eq!(empty.code(), tonic::Code::InvalidArgument);

    guard.stop().await;
}

#[tokio::test]
async fn server_metadata_advertises_extension_and_both_services_answer() {
    let (channel, guard) = start_server(BASE_CONFIG).await;

    let mut oip = GrpcInferenceServiceClient::new(channel.clone());
    let metadata = oip
        .server_metadata(ServerMetadataRequest {})
        .await
        .unwrap()
        .into_inner();
    assert!(metadata.extensions.contains(&"inferstream.v1".to_string()));
    assert!(metadata
        .extensions
        .contains(&"model_stream_infer".to_string()));

    let mut ext = InferstreamServiceClient::new(channel);
    assert!(ext.list_models(ListModelsRequest {}).await.is_ok());

    guard.stop().await;
}

#[tokio::test]
async fn bearer_auth_gates_extension_rpcs() {
    let (channel, guard) = start_server(BEARER_CONFIG).await;
    let mut client = InferstreamServiceClient::new(channel);

    let denied = client.list_models(ListModelsRequest {}).await.unwrap_err();
    assert_eq!(denied.code(), tonic::Code::Unauthenticated);

    let token = MetadataValue::try_from("Bearer extension-test-key").unwrap();
    let mut request = Request::new(ListModelsRequest {});
    request.metadata_mut().insert("authorization", token);
    let allowed = client.list_models(request).await.unwrap().into_inner();
    assert_eq!(allowed.models.len(), 1);

    guard.stop().await;
}
