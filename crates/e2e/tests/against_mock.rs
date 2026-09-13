//! Live harness against the in-process mock server (no GPU).
//!
//! Registers the same logical names the production suite uses so the
//! skip / required / generate / tokenize path is the one that runs on
//! Machine A / Machine B / Machine C.

use std::collections::HashSet;

use inferstream_e2e::{
    run_parity, run_suite, CatalogIndex, Matrix, Outcome, ParityConfig, ParityMode, SuiteConfig,
    SuiteFilter, Target,
};
use inferstream_server::config::Config;
use tonic::transport::Channel;

const MOCK_CONFIG: &str = r#"
listen = "127.0.0.1:0"

[[models]]
name = "minilm"
backend = "mock"

[[models]]
name = "default-llm"
backend = "mock"

[[models]]
name = "qwen-0.5b"
backend = "mock"
"#;

const NO_MINILM: &str = r#"
listen = "127.0.0.1:0"

[[models]]
name = "default-llm"
backend = "mock"
"#;

struct ServerGuard {
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    handle: tokio::task::JoinHandle<Result<(), inferstream_server::ServerError>>,
    addr: String,
}

impl ServerGuard {
    async fn stop(mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        let _ = self.handle.await;
    }
}

async fn start_server(config_text: &str) -> ServerGuard {
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
    // Touch the channel so a refused connection fails this helper, not the suite.
    let _ = Channel::from_shared(format!("http://{addr}"))
        .unwrap()
        .connect()
        .await
        .expect("client connects");
    ServerGuard {
        shutdown: Some(shutdown_tx),
        handle,
        addr: addr.to_string(),
    }
}

fn suite_config(addr: String) -> SuiteConfig {
    SuiteConfig {
        target: Target::Mock,
        addr,
        token: None,
        matrix: Matrix::builtin(),
        catalog: CatalogIndex::builtin().unwrap(),
        goldens_dir: None,
        only: HashSet::new(),
        filter: SuiteFilter::All,
        max_tokens: 16,
        cosine_min: 0.99,
    }
}

fn is_pass(outcome: Option<&Outcome>) -> bool {
    matches!(outcome, Some(Outcome::Pass { .. }))
}

fn is_skip(outcome: Option<&Outcome>) -> bool {
    matches!(outcome, Some(Outcome::Skip { .. }))
}

fn is_fail(outcome: Option<&Outcome>) -> bool {
    matches!(outcome, Some(Outcome::Fail { .. }))
}

#[tokio::test]
async fn same_suite_against_mock_logical_names() {
    let server = start_server(MOCK_CONFIG).await;
    let report = run_suite(suite_config(server.addr.clone()))
        .await
        .expect("suite runs");
    assert!(
        !report.failed(),
        "{}",
        report.format(Target::Mock, &server.addr)
    );
    assert!(is_pass(report.outcome("server-live")));
    assert!(is_pass(report.outcome("list-models")));
    assert!(is_pass(report.outcome("list-models:minilm")));
    assert!(is_pass(report.outcome("embed:minilm")));
    assert!(is_skip(report.outcome("embed:mpnet")), "mpnet not served");
    assert!(is_pass(report.outcome("tokenize:minilm")));
    assert!(is_pass(report.outcome("tokenize:default-llm")));
    assert!(is_pass(report.outcome("generate:default-llm")));
    assert!(is_pass(report.outcome("generate:qwen-0.5b")));
    assert!(is_skip(report.outcome("generate:qwen-7b")));
    server.stop().await;
}

#[tokio::test]
async fn missing_required_minilm_hard_fails() {
    let server = start_server(NO_MINILM).await;
    let report = run_suite(suite_config(server.addr.clone()))
        .await
        .expect("suite runs");
    assert!(report.failed());
    assert!(is_fail(report.outcome("list-models:minilm")));
    assert!(is_fail(report.outcome("embed:minilm")));
    server.stop().await;
}

#[tokio::test]
async fn apple_target_skips_mpnet_as_not_available_on_arch() {
    let server = start_server(MOCK_CONFIG).await;
    let mut config = suite_config(server.addr.clone());
    config.target = Target::Apple;
    // Apple matrix dim is 384; mock returns 8 — only exercise skip logic.
    config.filter = SuiteFilter::Embed;
    config.only = ["mpnet"].iter().map(|s| (*s).to_string()).collect();
    let report = run_suite(config).await.expect("suite runs");
    match report.outcome("embed:mpnet") {
        Some(Outcome::Skip { reason }) => {
            assert!(reason.contains("NotAvailableOnArch"), "reason was {reason}");
        }
        other => panic!("expected mpnet skip, got {other:?}"),
    }
    server.stop().await;
}

#[tokio::test]
async fn parity_goldens_write_and_compare_against_mock() {
    let server = start_server(MOCK_CONFIG).await;
    let tmp = tempfile::tempdir().expect("tmpdir");
    let mut config = ParityConfig {
        mode: ParityMode::Goldens { write: true },
        goldens_dir: tmp.path().to_path_buf(),
        aliases: vec!["minilm".into()],
        workspace: tmp.path().to_path_buf(),
        token: None,
        catalog: CatalogIndex::builtin().unwrap(),
        matrix: Matrix::builtin(),
        peers: [(Target::Mock, server.addr.clone())].into_iter().collect(),
        dumps: Default::default(),
        soak_limit: 0,
    };
    let written = run_parity(config.clone()).await.expect("write");
    assert!(
        !written.failed(),
        "{}",
        written.format(Target::Mock, "parity-write")
    );
    assert!(is_pass(written.outcome("parity-golden:minilm")));

    config.mode = ParityMode::Goldens { write: false };
    let compared = run_parity(config).await.expect("compare");
    assert!(
        !compared.failed(),
        "{}",
        compared.format(Target::Mock, "parity-compare")
    );
    assert!(is_pass(compared.outcome("parity-golden:minilm")));
    server.stop().await;
}

#[tokio::test]
async fn parity_cross_live_versus_dump() {
    let server = start_server(MOCK_CONFIG).await;
    let tmp = tempfile::tempdir().expect("tmpdir");
    let write = ParityConfig {
        mode: ParityMode::Goldens { write: true },
        goldens_dir: tmp.path().to_path_buf(),
        aliases: vec!["minilm".into()],
        workspace: tmp.path().to_path_buf(),
        token: None,
        catalog: CatalogIndex::builtin().unwrap(),
        matrix: Matrix::builtin(),
        peers: [(Target::Mock, server.addr.clone())].into_iter().collect(),
        dumps: Default::default(),
        soak_limit: 0,
    };
    let written = run_parity(write).await.expect("write");
    assert!(
        !written.failed(),
        "{}",
        written.format(Target::Mock, "write")
    );

    let src = tmp.path().join("mock/minilm.json");
    let dest_dir = tmp.path().join("nvidia");
    std::fs::create_dir_all(&dest_dir).unwrap();
    std::fs::copy(&src, dest_dir.join("minilm.json")).unwrap();

    let cross = ParityConfig {
        mode: ParityMode::Cross,
        goldens_dir: tmp.path().to_path_buf(),
        aliases: vec!["minilm".into()],
        workspace: tmp.path().to_path_buf(),
        token: None,
        catalog: CatalogIndex::builtin().unwrap(),
        matrix: Matrix::builtin(),
        peers: [(Target::Mock, server.addr.clone())].into_iter().collect(),
        dumps: [(Target::Nvidia, dest_dir)].into_iter().collect(),
        soak_limit: 0,
    };
    let report = run_parity(cross).await.expect("cross");
    assert!(
        !report.failed(),
        "{}",
        report.format(Target::Mock, "parity-cross")
    );
    assert!(is_pass(report.outcome("parity-cross:minilm:mock-nvidia")));
    server.stop().await;
}
