//! Prove the gRPC `Rerank` façade scores match library goldens.
//!
//! Skips when MiniLM-L6 weights are absent so `cargo test --workspace`
//! stays offline-green. `make test-turborerank` fetches then runs the
//! ignored suite, which fails if weights are missing.

use std::path::PathBuf;
use std::sync::Arc;

use inferstream_backend::Backend;
use inferstream_backend_turborerank::TurboRerankBackend;
use inferstream_protocol::extension::inferstream_service_client::InferstreamServiceClient;
use inferstream_protocol::extension::RerankRequest;
use inferstream_server::config::{BackendKind, Config, ModelConfig};
use inferstream_server::{build_registry, ServerError};
use serde::Deserialize;
use turborerank::{default_model_dir, weights_present, Activation, Device, Engine, Truncation};

const ALIAS: &str = "ms-marco-minilm-l6";
const QUERY: &str = "How many people live in Berlin?";
const REL: &str = "Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.";
const MID: &str = "Berlin is well known for its museums.";
const IRREL: &str = "New York City is famous for its pizza and bagels.";

#[derive(Deserialize)]
struct GoldenFile {
    texts: GoldenTexts,
    sigmoid: Vec<f32>,
    logits: Vec<f32>,
    #[serde(default)]
    atol: Option<f32>,
}

#[derive(Deserialize)]
struct GoldenTexts {
    query: String,
    documents: Vec<String>,
}

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/reference_rerank/ms_marco_minilm_l6_berlin.json")
}

fn load_golden() -> GoldenFile {
    let text = std::fs::read_to_string(golden_path()).expect("berlin golden");
    serde_json::from_str(&text).expect("berlin golden parses")
}

fn close(got: &[f32], want: &[f32], atol: f32) {
    assert_eq!(got.len(), want.len(), "score count");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= atol,
            "score[{i}] got={g} want={w} atol={atol}"
        );
    }
}

fn reject_word_overlap(scores: &[f32]) {
    let only_unit = scores
        .iter()
        .all(|s| (*s - 0.0).abs() < 1e-8 || (*s - 1.0).abs() < 1e-8 || (*s - 0.5).abs() < 1e-8);
    assert!(
        !only_unit,
        "FAKE: RPC scores look like the word-overlap mock {scores:?}"
    );
}

fn factory() -> impl inferstream_server::BackendFactory {
    move |model: &ModelConfig| -> Result<Arc<dyn Backend>, ServerError> {
        match model.backend {
            BackendKind::TurboRerank => {
                let backend = TurboRerankBackend::open_for_model(
                    &model.name,
                    model.device.as_deref(),
                    model.path.as_deref(),
                    model.max_batch_size,
                )
                .map_err(|e| ServerError::InvalidModelConfig {
                    model: model.name.clone(),
                    backend: model.backend.as_str(),
                    message: e.to_string(),
                })?;
                Ok(Arc::new(backend))
            }
            BackendKind::Mock => Err(ServerError::InvalidModelConfig {
                model: model.name.clone(),
                backend: model.backend.as_str(),
                message: format!(
                    "catalog CE alias {:?} refuses backend=mock; word-overlap \
                     is not MiniLM. Use backend = \"turborerank\"",
                    model.name
                ),
            }),
            _ => Err(inferstream_server::unsupported(
                model,
                "this test factory only constructs TurboRerank",
            )),
        }
    }
}

async fn start_ce_server() -> (
    tonic::transport::Channel,
    impl std::future::Future<Output = ()>,
) {
    let config = Config::from_toml(&format!(
        r#"
        listen = "127.0.0.1:0"

        [[models]]
        name = "{ALIAS}"
        backend = "turborerank"
        device = "cpu"
        path = "{}"
        max_batch_size = 32
        "#,
        default_model_dir().display()
    ))
    .expect("test config parses");
    let registry = build_registry(&config, &factory()).expect("CE registry builds");
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
    let channel = tonic::transport::Channel::from_shared(format!("http://{addr}"))
        .unwrap()
        .connect()
        .await
        .expect("client connects");
    let stop = async move {
        let _ = shutdown_tx.send(());
        let _ = handle.await;
    };
    (channel, stop)
}

#[test]
fn mock_backend_kind_cannot_satisfy_ce_alias() {
    let config = Config::from_toml(
        r#"
        [[models]]
        name = "ms-marco-minilm-l6"
        backend = "mock"
        "#,
    )
    .unwrap();
    let err = match build_registry(&config, &factory()) {
        Ok(_) => panic!("CE alias on mock must fail at startup"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("mock") || msg.contains("word-overlap") || msg.contains("turborerank"),
        "startup error must refuse mock-as-MiniLM, got {msg}"
    );
}

#[tokio::test]
async fn backend_rerank_matches_berlin_sigmoid_when_weights_present() {
    if !weights_present() {
        return;
    }
    let golden = load_golden();
    let atol = golden.atol.unwrap_or(0.002);
    let backend =
        TurboRerankBackend::open(ALIAS, Device::Cpu, Some(&default_model_dir()), Some(32))
            .expect("CPU MiniLM CE");
    let scores = backend
        .rerank(ALIAS, &golden.texts.query, &golden.texts.documents, false)
        .await
        .expect("rerank");
    reject_word_overlap(&scores);
    close(&scores, &golden.sigmoid, atol);
    assert!(
        scores[0] > scores[1] && scores[1] > scores[2],
        "input-order scores must stay monotonic: {scores:?}"
    );
    let raw = backend
        .rerank(ALIAS, &golden.texts.query, &golden.texts.documents, true)
        .await
        .expect("rerank raw_scores");
    close(&raw, &golden.logits, atol);
}

#[tokio::test]
async fn rpc_scores_match_berlin_and_honor_sort_top_n_when_weights_present() {
    if !weights_present() {
        return;
    }
    let golden = load_golden();
    let atol = golden.atol.unwrap_or(0.002);
    let (channel, stop) = start_ce_server().await;
    let mut client = InferstreamServiceClient::new(channel);

    let response = client
        .rerank(RerankRequest {
            model_name: ALIAS.into(),
            query: golden.texts.query.clone(),
            documents: golden.texts.documents.clone(),
            top_n: 0,
            return_documents: true,
            raw_scores: false,
        })
        .await
        .expect("Rerank RPC")
        .into_inner();

    assert_eq!(response.results.len(), 3);
    let mut by_index = vec![0.0f32; 3];
    for row in &response.results {
        by_index[row.index as usize] = row.score;
        assert_eq!(
            row.document, golden.texts.documents[row.index as usize],
            "return_documents echoes input"
        );
    }
    reject_word_overlap(&by_index);
    close(&by_index, &golden.sigmoid, atol);

    let raw_rpc = client
        .rerank(RerankRequest {
            model_name: ALIAS.into(),
            query: golden.texts.query.clone(),
            documents: golden.texts.documents.clone(),
            top_n: 0,
            return_documents: false,
            raw_scores: true,
        })
        .await
        .expect("Rerank RPC raw_scores")
        .into_inner();
    let mut raw_by_index = vec![0.0f32; 3];
    for row in &raw_rpc.results {
        raw_by_index[row.index as usize] = row.score;
    }
    close(&raw_by_index, &golden.logits, atol);

    let order: Vec<u32> = response.results.iter().map(|r| r.index).collect();
    assert_eq!(order, vec![0, 1, 2], "Berlin set is already descending");
    assert!(response.results[0].score > response.results[1].score);

    let top1 = client
        .rerank(RerankRequest {
            model_name: ALIAS.into(),
            query: QUERY.into(),
            documents: vec![IRREL.into(), REL.into(), MID.into()],
            top_n: 1,
            return_documents: false,
            raw_scores: false,
        })
        .await
        .expect("top_n RPC")
        .into_inner();
    assert_eq!(top1.results.len(), 1);
    assert_eq!(
        top1.results[0].index, 1,
        "most relevant Berlin pop doc is input index 1"
    );
    assert!(top1.results[0].document.is_empty());

    let oversize = client
        .rerank(RerankRequest {
            model_name: ALIAS.into(),
            query: QUERY.into(),
            documents: (0..33).map(|i| format!("doc {i}")).collect(),
            top_n: 0,
            return_documents: false,
            raw_scores: false,
        })
        .await
        .unwrap_err();
    assert_eq!(oversize.code(), tonic::Code::InvalidArgument);

    stop.await;
}

#[test]
#[ignore]
fn ignored_rpc_requires_weights() {
    assert!(
        weights_present(),
        "make fetch-rerankers before make test-turborerank"
    );
}

#[test]
fn library_sigmoid_is_the_rpc_activation() {
    if !weights_present() {
        return;
    }
    let engine =
        Engine::create_with_config(Device::Cpu, Some(&default_model_dir())).expect("CPU engine");
    engine
        .load_model(ALIAS)
        .unwrap_or_else(|e| panic!("load: {e} ({})", engine.last_error()));
    let lib = engine
        .score(
            Some(ALIAS),
            QUERY,
            &[REL, MID, IRREL],
            Truncation::LongestFirst,
            Activation::Sigmoid,
            0,
        )
        .unwrap();
    let golden = load_golden();
    close(&lib, &golden.sigmoid, golden.atol.unwrap_or(0.002));
}
