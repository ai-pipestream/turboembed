//! Harness-logic unit coverage: cosine / golden math, matrix parsing,
//! `decide()` edges, corpus fixtures, and one `run_suite` pass against the
//! in-process mock server driven by a tempdir matrix (mock-only aliases; no
//! GPU, no catalog-model correctness claims).

use std::collections::HashSet;
use std::path::PathBuf;

use inferstream_e2e::{
    corpus, decide,
    golden::{cosine, golden_cosine, load_golden, lookup_golden, GoldenEmbedding},
    run_suite, CatalogIndex, Decision, Matrix, Outcome, SkipReason, SuiteConfig, SuiteFilter,
    Target,
};
use inferstream_fetch::workspace_root;
use inferstream_protocol::extension::ModelInfo;
use inferstream_server::config::Config;
use tonic::transport::Channel;

// ---------------------------------------------------------------------------
// cosine / golden math
// ---------------------------------------------------------------------------

#[test]
fn cosine_boundary_cases() {
    let v = [0.2f32, -0.4, 0.9];
    assert!(
        (cosine(&v, &v) - 1.0).abs() < 1e-6,
        "identical vectors score 1.0"
    );
    let a = [1.0f32, 0.0, 0.0];
    let b = [0.0f32, 1.0, 0.0];
    assert!(cosine(&a, &b).abs() < 1e-6, "orthogonal vectors score 0.0");
    assert_eq!(
        cosine(&a, &[0.0f32, 1.0]),
        0.0,
        "length mismatch scores 0.0, not NaN"
    );
    assert_eq!(cosine(&[], &[]), 0.0, "empty vectors score 0.0");
    let zero = [0.0f32; 3];
    assert_eq!(cosine(&zero, &v), 0.0, "zero-norm vector scores 0.0");
    let opposite = [-0.2f32, 0.4, -0.9];
    assert!(
        (cosine(&v, &opposite) + 1.0).abs() < 1e-6,
        "opposite vectors score -1.0"
    );
}

#[test]
fn golden_cosine_full_vector_and_errors() {
    let g = GoldenEmbedding {
        model: None,
        text: "hello".into(),
        dim: 2,
        vector: Some(vec![0.6, 0.8]),
        vector_head: None,
        vector_tail: None,
    };
    assert!((golden_cosine(&[0.6, 0.8], &g).unwrap() - 1.0).abs() < 1e-6);
    assert!((golden_cosine(&[-0.6, -0.8], &g).unwrap() + 1.0).abs() < 1e-6);

    let dim_err = golden_cosine(&[1.0], &g).unwrap_err();
    assert!(dim_err.contains("dim mismatch"), "was {dim_err}");

    let wrong_len = GoldenEmbedding {
        model: None,
        text: "t".into(),
        dim: 2,
        vector: Some(vec![1.0]),
        vector_head: None,
        vector_tail: None,
    };
    let err = golden_cosine(&[1.0, 0.0], &wrong_len).unwrap_err();
    assert!(err.contains("vector length"), "was {err}");

    let no_vectors = GoldenEmbedding {
        model: None,
        text: "t".into(),
        dim: 2,
        vector: None,
        vector_head: None,
        vector_tail: None,
    };
    let err = golden_cosine(&[1.0, 0.0], &no_vectors).unwrap_err();
    assert!(err.contains("neither vector nor vector_head"), "was {err}");
}

#[test]
fn golden_cosine_head_tail_takes_min() {
    let g = GoldenEmbedding {
        model: None,
        text: "t".into(),
        dim: 4,
        vector: None,
        vector_head: Some(vec![1.0, 0.0]),
        vector_tail: Some(vec![0.0, 1.0]),
    };
    let aligned = golden_cosine(&[1.0, 0.0, 0.0, 1.0], &g).unwrap();
    assert!((aligned - 1.0).abs() < 1e-6);
    // Head is orthogonal while the tail matches: the score is the min.
    let head_off = golden_cosine(&[0.0, 1.0, 0.0, 1.0], &g).unwrap();
    assert!(head_off.abs() < 1e-6, "min must pick the orthogonal head");

    let head_only = GoldenEmbedding {
        model: None,
        text: "t".into(),
        dim: 1,
        vector: None,
        vector_head: Some(vec![1.0]),
        vector_tail: None,
    };
    let err = golden_cosine(&[1.0], &head_only).unwrap_err();
    assert!(err.contains("no vector_tail"), "was {err}");
}

/// Missing golden files are a *silent skip* of the golden check, never an
/// error; lookups are per-(arch, alias).
#[test]
fn golden_lookup_missing_alias_means_skip_not_error() {
    let tmp = tempfile::tempdir().expect("tmpdir");
    let goldens = tmp.path();
    assert!(lookup_golden(goldens, "mock", "minilm").is_none());
    assert!(lookup_golden(goldens, "intel", "minilm").is_none());

    let mock_dir = goldens.join("mock");
    std::fs::create_dir_all(&mock_dir).expect("mock golden dir");
    std::fs::write(
        mock_dir.join("minilm.json"),
        r#"{"model":"minilm","text":"hello","dim":2,"vector":[1.0,0.0]}"#,
    )
    .expect("write golden");

    let path = lookup_golden(goldens, "mock", "minilm").expect("golden found");
    let golden = load_golden(&path).expect("golden loads");
    assert!((golden_cosine(&[1.0, 0.0], &golden).unwrap() - 1.0).abs() < 1e-6);
    // Same alias under another arch (or another alias under mock) is missing.
    assert!(lookup_golden(goldens, "intel", "minilm").is_none());
    assert!(lookup_golden(goldens, "mock", "other-alias").is_none());
}

#[test]
fn load_golden_malformed_file_is_an_error() {
    let tmp = tempfile::tempdir().expect("tmpdir");
    let path = tmp.path().join("broken.json");
    std::fs::write(&path, "{ not json").expect("write broken golden");
    let err = load_golden(&path).expect_err("broken golden must not load");
    assert!(!err.is_empty());
}

// ---------------------------------------------------------------------------
// Matrix parsing
// ---------------------------------------------------------------------------

#[test]
fn from_path_parses_committed_matrix() {
    let path = workspace_root().join("testdata/e2e/matrix.json");
    let m = Matrix::from_path(&path).expect("committed matrix parses");
    assert_eq!(
        m,
        Matrix::builtin(),
        "from_path of the builtin file matches builtin()"
    );

    assert_eq!(m.embeds.len(), 13, "matrix.json embeds");
    assert_eq!(m.llms.len(), 3, "matrix.json llms");

    let minilm = m.embed("minilm").expect("minilm");
    assert!(minilm.required);
    assert_eq!(minilm.dim, 384);
    for arch in ["nvidia", "intel", "apple"] {
        assert!(minilm.arches.iter().any(|a| a == arch), "minilm on {arch}");
    }

    let mpnet = m.embed("mpnet").expect("mpnet");
    assert_eq!(mpnet.dim, 768);
    assert!(!mpnet.arches.iter().any(|a| a == "apple"));

    // Required flags: only minilm is required among embeds; every LLM is
    // optional (hosts soft-skip when they did not put it on `serve`).
    let required: Vec<&str> = m
        .embeds
        .iter()
        .filter(|e| e.required)
        .map(|e| e.alias.as_str())
        .collect();
    assert_eq!(required, ["minilm"]);
    assert!(m.llms.iter().all(|l| !l.required));
    assert!(m.embeds.iter().all(|e| e.dim > 0));

    // Arch targeting agrees with the file's arch lists.
    assert_eq!(m.embed_on_target(Target::Nvidia).count(), 13);
    assert_eq!(m.embed_on_target(Target::Apple).count(), 11);
    assert!(m
        .embed_on_target(Target::Apple)
        .all(|e| e.alias != "mpnet" && e.alias != "nomic-embed-text"));
    assert_eq!(m.llm_on_target(Target::Intel).count(), 3);
}

#[test]
fn from_path_rejects_bad_files() {
    let tmp = tempfile::tempdir().expect("tmpdir");

    let garbage = tmp.path().join("garbage.json");
    std::fs::write(&garbage, "not json at all").expect("write garbage");
    assert!(Matrix::from_path(&garbage).is_err());

    let bad_schema = tmp.path().join("bad-schema.json");
    std::fs::write(&bad_schema, r#"{"embeds":[{"alias":"x"}],"llms":[]}"#).expect("write");
    assert!(
        Matrix::from_path(&bad_schema).is_err(),
        "embed row without dim must not parse"
    );

    assert!(Matrix::from_path(tmp.path().join("does-not-exist.json")).is_err());
}

#[test]
fn from_path_roundtrips_a_minimal_matrix() {
    let tmp = tempfile::tempdir().expect("tmpdir");
    let path = tmp.path().join("matrix.json");
    std::fs::write(
        &path,
        r#"{
          "embeds": [
            {"alias": "a-embed", "dim": 2, "required": true, "arches": ["intel"]}
          ],
          "llms": [
            {"alias": "b-llm", "arches": ["nvidia", "apple"]}
          ]
        }"#,
    )
    .expect("write matrix");

    let m = Matrix::from_path(&path).expect("minimal matrix parses");
    let embed = m.embed("a-embed").expect("a-embed");
    assert_eq!(embed.dim, 2);
    assert!(embed.required);
    assert_eq!(embed.arches, ["intel"]);
    let llm = m.llm("b-llm").expect("b-llm");
    assert!(!llm.required, "required defaults to false");
    assert_eq!(llm.arches, ["nvidia", "apple"]);

    assert_eq!(m.embed_on_target(Target::Intel).count(), 1);
    assert_eq!(
        m.embed_on_target(Target::Nvidia).count(),
        0,
        "a-embed is intel-only"
    );
    assert_eq!(m.llm_on_target(Target::Apple).count(), 1);
}

// ---------------------------------------------------------------------------
// decide() edges
// ---------------------------------------------------------------------------

fn served_info(name: &str, ready: bool) -> ModelInfo {
    ModelInfo {
        name: name.into(),
        ready,
        embedding_dim: 8,
        ..Default::default()
    }
}

#[test]
fn decide_required_but_not_ready_fails() {
    let cat = CatalogIndex::builtin().expect("catalog");
    match decide(
        "minilm",
        true,
        Target::Mock,
        &cat,
        Some(&served_info("minilm", false)),
    ) {
        Decision::Fail(msg) => assert!(msg.contains("not ready"), "was {msg}"),
        other => panic!("expected Fail, got {other:?}"),
    }
}

#[test]
fn decide_optional_not_ready_skips() {
    let cat = CatalogIndex::builtin().expect("catalog");
    match decide(
        "minilm",
        false,
        Target::Mock,
        &cat,
        Some(&served_info("minilm", false)),
    ) {
        Decision::Skip(reason @ SkipReason::NotReady { .. }) => {
            assert!(reason.to_string().contains("not ready"));
        }
        other => panic!("expected NotReady skip, got {other:?}"),
    }
}

/// An alias the catalog has never heard of cannot be "not available on an
/// arch" — it lands on the plain not-served skip.
#[test]
fn decide_unknown_unserved_alias_skips_not_served() {
    let cat = CatalogIndex::builtin().expect("catalog");
    assert!(!cat.known("harness-only-alias"));
    match decide("harness-only-alias", false, Target::Intel, &cat, None) {
        Decision::Skip(SkipReason::NotServed { alias }) => {
            assert_eq!(alias, "harness-only-alias")
        }
        other => panic!("expected NotServed skip, got {other:?}"),
    }
}

#[test]
fn decide_not_available_on_arch_names_available_arches() {
    let cat = CatalogIndex::builtin().expect("catalog");
    match decide("mpnet", false, Target::Apple, &cat, None) {
        Decision::Skip(SkipReason::NotAvailableOnArch {
            alias,
            arch,
            available,
        }) => {
            assert_eq!(alias, "mpnet");
            assert_eq!(arch, "apple");
            assert_eq!(available, "nvidia, intel");
            let rendered = SkipReason::NotAvailableOnArch {
                alias,
                arch,
                available,
            }
            .to_string();
            assert!(rendered.contains("NotAvailableOnArch"), "was {rendered}");
            assert!(rendered.contains("apple"), "was {rendered}");
        }
        other => panic!("expected NotAvailableOnArch, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Corpus fixtures
// ---------------------------------------------------------------------------

#[test]
fn sts_micro_pairs_are_complete_and_well_formed() {
    let pairs = corpus::load_sts_micro();
    assert_eq!(pairs.len(), 12);
    let ids: HashSet<&str> = pairs.iter().map(|p| p.id.as_str()).collect();
    assert_eq!(ids.len(), pairs.len(), "ids must be unique");
    assert_eq!(pairs[0].id, "sts-0001");
    for pair in &pairs {
        assert!(!pair.text_a.trim().is_empty(), "{} empty text_a", pair.id);
        assert!(!pair.text_b.trim().is_empty(), "{} empty text_b", pair.id);
        assert!(
            (0.0..=5.0).contains(&pair.score),
            "{} score {} outside 0–5",
            pair.id,
            pair.score
        );
    }
}

#[test]
fn parse_sts_jsonl_skips_blanks_and_reports_bad_lines() {
    let ok = corpus::parse_sts_jsonl(
        "{\"id\":\"a\",\"score\":1.0,\"text_a\":\"x\",\"text_b\":\"y\"}\n\n   \n",
    )
    .expect("blank lines are skipped");
    assert_eq!(ok.len(), 1);
    assert_eq!(ok[0].id, "a");
    assert!(corpus::parse_sts_jsonl("")
        .expect("empty input parses")
        .is_empty());

    let err = corpus::parse_sts_jsonl(
        "{\"id\":\"a\",\"score\":1.0,\"text_a\":\"x\",\"text_b\":\"y\"}\nnot json\n",
    )
    .expect_err("malformed line must error");
    assert!(err.contains("line 2"), "was {err}");
}

// ---------------------------------------------------------------------------
// run_suite against the in-process mock with a tempdir matrix
// ---------------------------------------------------------------------------

const MOCK_MATRIX_CONFIG: &str = r#"
listen = "127.0.0.1:0"

[[models]]
name = "minilm"
backend = "mock"

[[models]]
name = "mock-extra"
backend = "mock"

[[models]]
name = "default-llm"
backend = "mock"
"#;

/// Deliberately gives "mock-extra" dim 16 while the mock serves 8: on
/// Target::Mock the served ListModels dim wins, so the suite must still pass.
const TEMP_MATRIX_JSON: &str = r#"
{
  "embeds": [
    {"alias": "minilm", "dim": 384, "required": true, "arches": ["nvidia", "intel", "apple"]},
    {"alias": "mock-extra", "dim": 16, "arches": ["nvidia", "intel", "apple"]}
  ],
  "llms": [
    {"alias": "default-llm", "arches": ["nvidia", "intel", "apple"]}
  ]
}
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

fn suite_config(addr: String, matrix: Matrix, goldens_dir: Option<PathBuf>) -> SuiteConfig {
    SuiteConfig {
        target: Target::Mock,
        addr,
        token: None,
        matrix,
        catalog: CatalogIndex::builtin().unwrap(),
        goldens_dir,
        only: HashSet::new(),
        filter: SuiteFilter::All,
        max_tokens: 16,
        cosine_min: 0.99,
    }
}

#[tokio::test]
async fn run_suite_with_tempdir_matrix_matches_mock_dims() {
    let server = start_server(MOCK_MATRIX_CONFIG).await;
    let tmp = tempfile::tempdir().expect("tmpdir");
    let matrix_path = tmp.path().join("matrix.json");
    std::fs::write(&matrix_path, TEMP_MATRIX_JSON).expect("write temp matrix");
    let matrix = Matrix::from_path(&matrix_path).expect("temp matrix parses");
    // An empty goldens dir exercises "golden missing → check skipped": the
    // embeds must pass without any golden files on disk.
    let goldens = tmp.path().join("goldens");
    std::fs::create_dir_all(&goldens).expect("goldens dir");

    let report = run_suite(suite_config(server.addr.clone(), matrix, Some(goldens)))
        .await
        .expect("suite runs");
    assert!(
        !report.failed(),
        "{}",
        report.format(Target::Mock, &server.addr)
    );
    assert_eq!(report.counts(), (8, 1, 0));

    // The mock-only alias embeds with the served dim (8), not the matrix dim.
    match report.outcome("embed:mock-extra") {
        Some(Outcome::Pass { detail }) => {
            assert!(detail.contains("dim=8"), "detail was {detail}");
            assert!(detail.contains("vectors=2"), "detail was {detail}");
        }
        other => panic!("expected embed:mock-extra pass, got {other:?}"),
    }
    assert!(matches!(
        report.outcome("embed:minilm"),
        Some(Outcome::Pass { .. })
    ));
    assert!(matches!(
        report.outcome("generate:default-llm"),
        Some(Outcome::Pass { .. })
    ));
    // The catalog CE alias is never on the mock serve list → soft skip.
    match report.outcome("rerank:ms-marco-minilm-l6") {
        Some(Outcome::Skip { reason }) => {
            assert!(reason.contains("not served"), "reason was {reason}")
        }
        other => panic!("expected rerank skip, got {other:?}"),
    }
    server.stop().await;
}
