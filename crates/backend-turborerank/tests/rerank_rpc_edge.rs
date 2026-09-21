//! Edge-case coverage for the gRPC `Rerank` façade on the real CPU
//! cross-encoder (`ms-marco-minilm-l6`): the 32-document batch boundary,
//! empty query / empty document rows, unicode and binary documents, sort +
//! `top_n` truncation, bit-for-bit determinism, unknown-model statuses, and
//! the word-overlap rejection gate.
//!
//! Skips when MiniLM-L6 weights are absent so `cargo test --workspace`
//! stays offline-green, matching `rpc_berlin.rs`. `make test-turborerank`
//! fetches then runs the suite against real weights.

use std::sync::Arc;

use inferstream_backend::{Backend, BackendError};
use inferstream_backend_turborerank::{device_from_config, TurboRerankBackend};
use inferstream_protocol::extension::inferstream_service_client::InferstreamServiceClient;
use inferstream_protocol::extension::{RerankRequest, RerankResponse, RerankResult};
use inferstream_server::config::{BackendKind, Config, ModelConfig};
use inferstream_server::{build_registry, ServerError};
use tonic::transport::Channel;
use turborerank::{default_model_dir, weights_present, Device};

const ALIAS: &str = "ms-marco-minilm-l6";
const QUERY: &str = "What is the capital of France?";
const PARIS: &str = "Paris is the capital and most populous city of France.";
const LYON: &str = "Lyon is a major city in France, known for its cuisine.";
const CROISSANT: &str = "The croissant is a buttery French pastry of Austrian origin.";
const IRREL: &str = "The Sahara Desert covers much of North Africa.";

/// Local mirror of the crate-private `reject_mock_shaped`: the word-overlap
/// mock can only emit all-equal scores drawn from `{0.0, 0.5, 1.0}`. Used as
/// a post-condition on every successful response so a passing RPC proves
/// real cross-encoder scores clear the gate (the gate's reject path needs
/// the mock, which `open` refuses — see `mock_gate_blocked_at_open`).
fn mock_shaped(scores: &[f32]) -> bool {
    let all_equal = scores.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-8);
    let only_unit = scores
        .iter()
        .all(|s| (*s - 0.0).abs() < 1e-8 || (*s - 1.0).abs() < 1e-8 || (*s - 0.5).abs() < 1e-8);
    all_equal && only_unit
}

fn assert_not_mock_shaped(scores: &[f32]) {
    assert!(
        !mock_shaped(scores),
        "RPC returned word-overlap-shaped scores {scores:?}"
    );
}

fn assert_sorted_desc(results: &[RerankResult]) {
    for w in results.windows(2) {
        assert!(
            w[0].score >= w[1].score,
            "results not sorted descending: {} then {}",
            w[0].score,
            w[1].score
        );
    }
}

fn assert_index_permutation(results: &[RerankResult], n: usize) {
    let mut indices: Vec<u32> = results.iter().map(|r| r.index).collect();
    indices.sort_unstable();
    assert_eq!(
        indices,
        (0..n as u32).collect::<Vec<_>>(),
        "result indices must be a permutation of 0..{n}"
    );
}

fn scores_of(response: &RerankResponse) -> Vec<f32> {
    response.results.iter().map(|r| r.score).collect()
}

fn rerank_request(
    query: &str,
    documents: &[String],
    top_n: u32,
    return_documents: bool,
    raw_scores: bool,
) -> RerankRequest {
    RerankRequest {
        model_name: ALIAS.into(),
        query: query.into(),
        documents: documents.to_vec(),
        top_n,
        return_documents,
        raw_scores,
    }
}

async fn rpc_ok(
    client: &mut InferstreamServiceClient<Channel>,
    req: RerankRequest,
) -> RerankResponse {
    client.rerank(req).await.expect("Rerank RPC").into_inner()
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

async fn start_ce_server() -> (Channel, impl std::future::Future<Output = ()>) {
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
    let channel = Channel::from_shared(format!("http://{addr}"))
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

#[tokio::test]
async fn single_document_rerank_when_weights_present() {
    if !weights_present() {
        return;
    }
    let (channel, stop) = start_ce_server().await;
    let mut client = InferstreamServiceClient::new(channel);
    let docs: Vec<String> = vec![PARIS.into()];

    let response = rpc_ok(&mut client, rerank_request(QUERY, &docs, 0, true, false)).await;
    assert_eq!(response.results.len(), 1);
    let row = &response.results[0];
    assert_eq!(row.index, 0);
    assert_eq!(row.document, PARIS, "return_documents echoes the input doc");
    assert!(
        row.score.is_finite() && row.score > 0.0 && row.score < 1.0,
        "sigmoid(CLS logit) must stay inside (0, 1), got {}",
        row.score
    );
    assert_not_mock_shaped(&[row.score]);

    let bare = rpc_ok(&mut client, rerank_request(QUERY, &docs, 0, false, false)).await;
    assert!(
        bare.results[0].document.is_empty(),
        "return_documents=false leaves the document field empty"
    );
    stop.await;
}

#[tokio::test]
async fn batch_boundary_accepts_32_rejects_33_when_weights_present() {
    if !weights_present() {
        return;
    }
    let (channel, stop) = start_ce_server().await;
    let mut client = InferstreamServiceClient::new(channel);

    let docs: Vec<String> = (0..32)
        .map(|i| format!("France document {i}: Paris is the capital of France."))
        .collect();
    let response = rpc_ok(&mut client, rerank_request(QUERY, &docs, 0, false, false)).await;
    assert_eq!(
        response.results.len(),
        32,
        "exactly max_client_batch_size documents must score"
    );
    assert_index_permutation(&response.results, 32);
    assert_sorted_desc(&response.results);
    let scores = scores_of(&response);
    assert!(
        scores.iter().all(|s| s.is_finite() && *s > 0.0 && *s < 1.0),
        "sigmoid scores must be finite and inside (0, 1): {scores:?}"
    );
    assert_not_mock_shaped(&scores);

    let too_many: Vec<String> = (0..33)
        .map(|i| format!("France document {i}: Paris is the capital of France."))
        .collect();
    let err = client
        .rerank(rerank_request(QUERY, &too_many, 0, false, false))
        .await
        .unwrap_err();
    assert_eq!(err.code(), tonic::Code::InvalidArgument);
    assert!(
        err.message().contains("max_client_batch_size"),
        "oversize rejection must name the TEI limit, got: {}",
        err.message()
    );
    stop.await;
}

#[tokio::test]
async fn backend_enforces_configured_max_documents_when_weights_present() {
    if !weights_present() {
        return;
    }
    let dir = default_model_dir();
    let backend = TurboRerankBackend::open(ALIAS, Device::Cpu, Some(&dir), Some(4))
        .expect("CPU MiniLM CE with max_documents = 4");
    assert_eq!(backend.max_documents(), 4);

    let four: Vec<String> = (0..4)
        .map(|i| format!("doc {i} about Paris France"))
        .collect();
    let scores = backend
        .rerank(ALIAS, QUERY, &four, false)
        .await
        .expect("batch of 4 within max_documents");
    assert_eq!(scores.len(), 4);

    let five: Vec<String> = (0..5)
        .map(|i| format!("doc {i} about Paris France"))
        .collect();
    let err = backend
        .rerank(ALIAS, QUERY, &five, false)
        .await
        .expect_err("batch of 5 exceeds max_documents");
    match err {
        BackendError::InvalidRequest(message) => assert!(
            message.contains("max_client_batch_size"),
            "unexpected rejection message: {message}"
        ),
        other => panic!("expected BackendError::InvalidRequest, got {other:?}"),
    }
    let mut dest = Vec::new();
    let err = backend
        .rerank_into(ALIAS, QUERY, &five, false, &mut dest)
        .await
        .expect_err("rerank_into must enforce max_documents too");
    assert!(
        matches!(err, BackendError::InvalidRequest(_)),
        "expected BackendError::InvalidRequest, got {err:?}"
    );
}

#[tokio::test]
async fn empty_query_and_empty_document_rows_when_weights_present() {
    if !weights_present() {
        return;
    }
    let (channel, stop) = start_ce_server().await;
    let mut client = InferstreamServiceClient::new(channel);

    // `pack_text` rejects a row only when BOTH sides are empty; either side
    // alone packs as [CLS][SEP] on the empty side (native test_pack_empty_sides).
    let docs: Vec<String> = vec![PARIS.into(), LYON.into()];
    let response = rpc_ok(&mut client, rerank_request("", &docs, 0, false, false)).await;
    assert_eq!(
        response.results.len(),
        2,
        "empty query must still score every document"
    );
    assert_not_mock_shaped(&scores_of(&response));

    let empty_docs: Vec<String> = vec![String::new(), String::new()];
    let response = rpc_ok(
        &mut client,
        rerank_request(QUERY, &empty_docs, 0, false, false),
    )
    .await;
    assert_eq!(
        response.results.len(),
        2,
        "empty document rows pack as an empty doc side"
    );
    assert_not_mock_shaped(&scores_of(&response));

    let mixed: Vec<String> = vec![String::new(), PARIS.into()];
    let err = client
        .rerank(rerank_request("", &mixed, 0, false, false))
        .await
        .unwrap_err();
    assert_eq!(err.code(), tonic::Code::InvalidArgument);
    assert!(
        err.message().contains("empty"),
        "pack_text must name the both-empty row, got: {}",
        err.message()
    );
    stop.await;
}

#[tokio::test]
async fn unicode_nul_and_overlong_docs_when_weights_present() {
    if !weights_present() {
        return;
    }
    let (channel, stop) = start_ce_server().await;
    let mut client = InferstreamServiceClient::new(channel);

    let filler = (0..600)
        .map(|i| format!("the quick brown fox {i} jumps over the lazy dog"))
        .collect::<Vec<_>>()
        .join(" ");
    let docs: Vec<String> = vec![
        "Größte Stadt: Paris — Hauptstadt Frankreichs.".into(),
        "巴黎是法国的首都。".into(),
        "🥐🗼🚀 croissants, towers and rockets".into(),
        "cafe\u{0301} naïve résumé — combining accents".into(),
        "القاهرة عاصمة مصر وليست فرنسا".into(),
        "embedded\u{0}NUL byte survives the RPC round trip".into(),
        filler,
    ];
    let n = docs.len();

    let response = rpc_ok(&mut client, rerank_request(QUERY, &docs, 0, true, false)).await;
    assert_eq!(response.results.len(), n);
    assert_index_permutation(&response.results, n);
    assert_sorted_desc(&response.results);
    let scores = scores_of(&response);
    assert!(
        scores.iter().all(|s| s.is_finite() && *s > 0.0 && *s < 1.0),
        "unicode/binary docs must score finite sigmoid values: {scores:?}"
    );
    assert_not_mock_shaped(&scores);
    for row in &response.results {
        assert_eq!(
            row.document, docs[row.index as usize],
            "UTF-8 (incl. NUL) must echo byte-identical through the RPC"
        );
    }
    stop.await;
}

#[tokio::test]
async fn top_n_truncates_sorted_desc_when_weights_present() {
    if !weights_present() {
        return;
    }
    let (channel, stop) = start_ce_server().await;
    let mut client = InferstreamServiceClient::new(channel);
    let docs: Vec<String> = vec![PARIS.into(), LYON.into(), CROISSANT.into(), IRREL.into()];

    let full = rpc_ok(&mut client, rerank_request(QUERY, &docs, 0, true, false)).await;
    assert_eq!(full.results.len(), 4);
    assert_sorted_desc(&full.results);
    let full_order: Vec<u32> = full.results.iter().map(|r| r.index).collect();
    assert_eq!(
        full_order,
        vec![0, 1, 2, 3],
        "France docs must rank above the desert doc"
    );
    let full_scores = scores_of(&full);
    for w in full_scores.windows(2) {
        assert!(
            w[0] > w[1],
            "curated docs must score strictly descending: {full_scores:?}"
        );
    }

    let top2 = rpc_ok(&mut client, rerank_request(QUERY, &docs, 2, true, false)).await;
    assert_eq!(top2.results.len(), 2, "top_n truncates the sorted list");
    let top2_order: Vec<u32> = top2.results.iter().map(|r| r.index).collect();
    assert_eq!(
        top2_order,
        vec![0, 1],
        "top_n keeps the highest-ranked docs"
    );
    assert_eq!(
        top2.results[0].document, PARIS,
        "return_documents echoes on truncated results"
    );
    assert!(top2.results[0].score > top2.results[1].score);

    let overshoot = rpc_ok(
        &mut client,
        rerank_request(QUERY, &docs, 1000, false, false),
    )
    .await;
    assert_eq!(
        overshoot.results.len(),
        4,
        "top_n larger than the batch keeps every document"
    );
    stop.await;
}

#[tokio::test]
async fn identical_rpcs_are_bit_for_bit_deterministic_when_weights_present() {
    if !weights_present() {
        return;
    }
    let (channel, stop) = start_ce_server().await;
    let mut client = InferstreamServiceClient::new(channel);
    let docs: Vec<String> = vec![PARIS.into(), LYON.into(), CROISSANT.into(), IRREL.into()];

    let baseline = rpc_ok(&mut client, rerank_request(QUERY, &docs, 0, true, false)).await;
    for attempt in 1..3 {
        let again = rpc_ok(&mut client, rerank_request(QUERY, &docs, 0, true, false)).await;
        assert_eq!(
            baseline.results, again.results,
            "identical sigmoid RPC #{attempt} diverged"
        );
    }

    let raw_baseline = rpc_ok(&mut client, rerank_request(QUERY, &docs, 0, false, true)).await;
    let raw_again = rpc_ok(&mut client, rerank_request(QUERY, &docs, 0, false, true)).await;
    assert_eq!(
        raw_baseline.results, raw_again.results,
        "raw_scores (identity activation) must be deterministic too"
    );
    assert_ne!(
        scores_of(&baseline),
        scores_of(&raw_baseline),
        "sigmoid and identity activations must not coincide"
    );
    stop.await;
}

#[tokio::test]
async fn duplicate_documents_keep_input_order_on_ties_when_weights_present() {
    if !weights_present() {
        return;
    }
    let (channel, stop) = start_ce_server().await;
    let mut client = InferstreamServiceClient::new(channel);
    let docs: Vec<String> = vec![PARIS.into(), PARIS.into(), IRREL.into(), PARIS.into()];

    let response = rpc_ok(&mut client, rerank_request(QUERY, &docs, 0, false, false)).await;
    assert_eq!(response.results.len(), 4);
    let order: Vec<u32> = response.results.iter().map(|r| r.index).collect();
    assert_eq!(
        order,
        vec![0, 1, 3, 2],
        "equal scores keep input order (stable sort); desert doc ranks last"
    );
    let paris_score = response.results[0].score;
    for row in &response.results[..3] {
        assert_eq!(
            row.score, paris_score,
            "identical (query, doc) pairs must score identically"
        );
    }
    assert!(paris_score > response.results[3].score);
    assert_not_mock_shaped(&scores_of(&response));
    stop.await;
}

#[tokio::test]
async fn unknown_model_fails_not_found_and_documents_validate_first_when_weights_present() {
    if !weights_present() {
        return;
    }
    let (channel, stop) = start_ce_server().await;
    let mut client = InferstreamServiceClient::new(channel);
    let docs: Vec<String> = vec![PARIS.into()];

    let mut unknown = rerank_request(QUERY, &docs, 0, false, false);
    unknown.model_name = "no-such-ce".into();
    let err = client.rerank(unknown).await.unwrap_err();
    assert_eq!(err.code(), tonic::Code::NotFound);
    assert!(
        err.message().contains("no-such-ce"),
        "NotFound must name the missing model, got: {}",
        err.message()
    );

    let mut unknown_empty = rerank_request(QUERY, &[], 0, false, false);
    unknown_empty.model_name = "no-such-ce".into();
    let err = client.rerank(unknown_empty).await.unwrap_err();
    assert_eq!(
        err.code(),
        tonic::Code::InvalidArgument,
        "empty documents validate before model lookup"
    );

    let err = client
        .rerank(rerank_request(QUERY, &[], 0, false, false))
        .await
        .unwrap_err();
    assert_eq!(err.code(), tonic::Code::InvalidArgument);
    stop.await;
}

#[test]
fn mock_gate_blocked_at_open_when_weights_present() {
    if !weights_present() {
        return;
    }
    // `reject_mock_shaped` is crate-private and the real CPU engine can never
    // emit all-equal {0.0, 0.5, 1.0} scores, so the all-0 / all-0.5 / all-1
    // reject branch is only reachable if the word-overlap mock sits behind
    // this façade. `open` blocks that one layer earlier: the mock device and
    // the mock alias never construct a backend, so no RPC path can produce
    // mock-shaped scores. The gate's pass path (normal — even all-equal but
    // non-unit — scores accepted) is asserted on every successful rerank in
    // this file, including duplicate-document ties.
    let err = device_from_config(Some("mock")).expect_err("device=mock must be refused");
    assert!(
        matches!(err, BackendError::Unavailable(_)),
        "expected Unavailable, got {err:?}"
    );

    let err = open_err(ALIAS, Device::Mock);
    let msg = err.to_string().to_ascii_lowercase();
    assert!(
        msg.contains("mock"),
        "mock device refusal must name mock: {msg}"
    );
    assert!(
        !msg.contains("word-overlap as minilm") && !msg.contains("serving word-overlap"),
        "must never offer to serve mock scores as MiniLM: {msg}"
    );

    let err = open_err("mock", Device::Cpu);
    assert!(
        err.to_string().to_ascii_lowercase().contains("mock"),
        "mock alias refusal must name mock: {err}"
    );
}

fn open_err(alias: &str, device: Device) -> BackendError {
    let dir = default_model_dir();
    match TurboRerankBackend::open(alias, device, Some(&dir), None) {
        Ok(_) => panic!("expected {alias} on {device:?} to fail to open"),
        Err(e) => e,
    }
}
