//! Integration suite for `turbo-client` (`#2338`):
//! 1. Pure in-process lexical lattice + one-pass NLP + local `libturbo` bundle
//!    execution with bidirectional UTF-8 / UTF-16 / code-point source spans.
//! 2. Hybrid local lexical/NLP + pooled remote `turbo-kserve` gRPC offload with
//!    bearer auth, `x-turbo-priority: interactive`, and `EmbedCoalescer` batching.
//! 3. Multi-Tokio-runtime teardown and replacement on one shared `TurboClient`
//!    instance with automatic channel eviction and zero stale reactor errors.
//! 4. Deterministic `ClientError::FailedPrecondition` refusal on both
//!    `AnalysisFingerprint` and local/remote `BundleScoringFingerprint` mismatches.

#![cfg(all(feature = "local-core", feature = "grpc"))]

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use turbo_analysis::Span;
use turbo_analysis_layers::TokenArcKind;
use turbo_client::proto::result::{AnalyzeQueryRequest, QueryDenseTargetProto};
use turbo_client::{
    AnalysisFingerprint, BundleScoringFingerprint, BundleScoringSpec, ClientError, CoalesceConfig, FieldAnalysisConfig,
    IndexGenerationHandshake, LocalEncoderOptions, MultiResOptions, NormalizerKind, RemoteEndpointConfig,
    RemotePriority, StemmerKind, TermSourceKind, TokenizerKind, TurboClient,
};
use turbo_kserve::config::ModelConfig;
use turbo_kserve::{AuthConfig, Serve, Server};

fn tiny_bundle_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle")
}

struct TestServerGuard {
    rt: tokio::runtime::Runtime,
    server: Option<Server>,
}

impl TestServerGuard {
    fn local_addr(&self) -> SocketAddr {
        self.server.as_ref().expect("server running").local_addr()
    }
}

impl Drop for TestServerGuard {
    fn drop(&mut self) {
        if let Some(s) = self.server.take() {
            self.rt.block_on(s.stop());
        }
    }
}

fn start_test_server(bearer: Option<&str>) -> TestServerGuard {
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
    let model = ModelConfig::parse(&format!(
        "bundle={},name=tiny-bert-bundle,device=0,sessions=2",
        tiny_bundle_dir().display()
    ))
    .expect("parse ModelConfig");
    let mut serve = Serve::default();
    if let Some(tok) = bearer {
        serve.auth = Some(AuthConfig::from_tokens([tok], std::iter::empty::<&str>()).expect("parse AuthConfig"));
    }
    let server = rt.block_on(async move {
        let s =
            Server::start(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0), vec![model], 16 * 1024 * 1024, serve)
                .await
                .expect("start test turbo-kserve");
        s.load().await.expect("load test bundle");
        s
    });
    TestServerGuard { rt, server: Some(server) }
}

#[test]
fn in_process_lexical_nlp_and_local_bundle_preserve_bidirectional_offsets() {
    let field_cfg = FieldAnalysisConfig {
        normalizers: vec![
            NormalizerKind::StripInvisible,
            NormalizerKind::Dehyphenate,
            NormalizerKind::Nfkc,
            NormalizerKind::FullCaseFold,
            NormalizerKind::AccentFold,
        ],
        tokenizer: TokenizerKind::Uax29,
        stemmer: StemmerKind::Porter,
        term_source: TermSourceKind::Stems,
        sentence_language: "eng".into(),
        emit_lattice: true,
    };
    let expected_afp = field_cfg.fingerprint();
    let client = TurboClient::builder()
        .with_default_field(field_cfg)
        .with_handshake(IndexGenerationHandshake { analysis: Some(expected_afp), bundle: None })
        .with_local_embed_bundle(tiny_bundle_dir(), LocalEncoderOptions::default(), None)
        .build()
        .expect("build local TurboClient");

    let raw = "Caf\u{00E9} \u{1F680} micro-\nservices invoice $250.00 (5 kg) in Berlin on 2026-05-01. Second sentence.";
    let lex = client.lexical().analyze(raw).expect("in-process lexical analysis");
    assert_eq!(lex.fingerprint, expected_afp);
    assert!(lex.sentences.len() >= 2);

    // Dehyphenation lattice emits joined "microservic" stem plus split part arcs,
    // and every UTF-8 / UTF-16 / code-point span round-trips back to ViewId::SOURCE.
    assert!(lex.arcs.iter().any(|a| a.term.starts_with("microservic")));
    assert!(lex.arcs.iter().any(|a| a.kind == TokenArcKind::DehyphenatedPart));
    for arc in &lex.arcs {
        let (u8s, u8e) = arc.spans.utf8;
        assert!(raw.is_char_boundary(u8s as usize) && raw.is_char_boundary(u8e as usize));
        let rt_u16 = lex.utf16_to_source_utf8(raw, arc.spans.utf16.0, arc.spans.utf16.1).unwrap();
        let rt_cp = lex.code_point_to_source_utf8(raw, arc.spans.code_point.0, arc.spans.code_point.1).unwrap();
        assert_eq!(rt_u16, (u8s, u8e));
        assert_eq!(rt_cp, (u8s, u8e));
    }

    // Dual cased analysis preserves case in the twin while folding the primary.
    let dual = client.lexical().analyze_dual("Berlin Caf\u{00E9}").unwrap();
    assert!(dual.primary.arcs.iter().any(|a| a.term == "berlin"));
    assert!(dual.cased.arcs.iter().any(|a| a.term == "Berlin"));

    // One-pass document NLP extracts money, SI quantity, date, Berlin geo, quality, and sketch.
    let nlp = client.document_nlp().analyze(raw).expect("one-pass NLP");
    assert!(nlp.money.iter().any(|m| m.currency == "USD" && (m.amount_f64 - 250.0).abs() < 1e-6));
    assert!(nlp.quantities.iter().any(|q| !q.is_percent && q.value_f64 > 0.0));
    assert_eq!(nlp.document_date.as_ref().map(|d| d.iso.as_str()), Some("2026-05-01"));
    assert!(nlp.geo.iter().any(|g| g.canonical_name == "Berlin" && g.country_code == "DE"));
    assert!(nlp.quality.assessed);
    assert!(nlp.sketch.word_count >= 8 && !nlp.sketch.minhash.is_empty());

    // One-pass query analyzer extracts structured filters + BM25 terms with zero RPCs.
    let qplan = client.query_analyzer().analyze("invoices under $500 in Berlin").expect("one-pass query");
    assert!(!qplan.filters.is_empty());
    assert!(!qplan.fields.is_empty());

    // Local libturbo bundle embeds whole document + sentence spans in one pass.
    let sent_spans: Vec<Span> = lex.sentences.iter().map(|s| Span::new(s.spans.utf8.0, s.spans.utf8.1)).collect();
    let local_embed = client.local_embed().expect("local embed engine");
    let mres = local_embed
        .embed_document_multires(
            raw,
            &sent_spans,
            &MultiResOptions { emit_row_sha256: true, turboquant: Some((4, true)), ..Default::default() },
        )
        .expect("single-pass multires embed");
    assert_eq!(mres.full.rows, 1);
    assert!(mres.full.dim >= 8);
    assert!(!mres.full.turboquant_compact.is_empty());
    assert_eq!(mres.sentences.as_ref().map(|s| s.rows as usize), Some(sent_spans.len()));
}

#[test]
fn remote_offload_coalescing_and_multi_runtime_replacement_succeed_cleanly() {
    let server = start_test_server(Some("secret-ingest-query-token"));
    let endpoint = format!("http://{}", server.local_addr());

    let remote_cfg = RemoteEndpointConfig::new(endpoint)
        .with_bearer_token("secret-ingest-query-token")
        .with_priority(RemotePriority::Interactive)
        .with_timeout(Duration::from_secs(5));

    let client = Arc::new(
        TurboClient::builder()
            .with_remote_endpoint(remote_cfg, None)
            .with_embed_coalescer(
                "tiny-bert-bundle",
                Some("query".into()),
                CoalesceConfig { max_batch: 8, linger: Duration::from_millis(15), queue_capacity: 64 },
            )
            .build()
            .expect("build hybrid TurboClient"),
    );

    // Runtime #1: run concurrent coalesced queries + remote AnalyzeQuery, then drop rt1 completely.
    let rt1_vec = {
        let rt1 = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
        let out = rt1.block_on(async {
            let mut tasks = Vec::new();
            for i in 0..6 {
                let c = Arc::clone(&client);
                tasks.push(tokio::spawn(async move {
                    c.coalescer().unwrap().embed_query(&format!("distributed query {i}"), None).await.unwrap()
                }));
            }
            let mut rows = Vec::new();
            for t in tasks {
                rows.push(t.await.unwrap());
            }
            assert_eq!(rows.len(), 6);
            assert!(client.coalescer().unwrap().batches_dispatched() < 6);

            let aq = client
                .remote()
                .unwrap()
                .analyze_query(
                    AnalyzeQueryRequest {
                        query: "servers under $400 in Berlin".into(),
                        dense_targets: vec![QueryDenseTargetProto {
                            target_id: "body_vec".into(),
                            model: "tiny-bert-bundle".into(),
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                    None,
                )
                .await
                .unwrap();
            assert_eq!(aq.dense_results.len(), 1);
            assert!(!aq.filters.is_empty());
            assert_eq!(client.remote().unwrap().pool().active_channel_count(), 1);
            rows.remove(0)
        });
        drop(rt1);
        out
    };

    // Give the dropped runtime's task drop guard a brief moment to evict its channel.
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(client.remote().unwrap().pool().active_channel_count(), 0);

    // Runtime #2: reuse the exact same `TurboClient` on a brand-new Tokio runtime.
    let rt2 = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    let rt2_vec = rt2.block_on(async {
        let v = client.coalescer().unwrap().embed_query("distributed query 0", None).await.unwrap();
        assert_eq!(client.remote().unwrap().pool().active_channel_count(), 1);
        v
    });
    assert_eq!(rt1_vec.len(), rt2_vec.len());
    for (a, b) in rt1_vec.iter().zip(&rt2_vec) {
        assert!((a - b).abs() < 1e-5, "query vector drifted across runtime replacement: {a} vs {b}");
    }
}

#[test]
fn handshake_refuses_mismatched_analysis_and_local_or_remote_bundle_fingerprints() {
    let wrong_afp = AnalysisFingerprint::from_sha256([0xAA; 32]);
    let err = match TurboClient::builder()
        .with_handshake(IndexGenerationHandshake { analysis: Some(wrong_afp), bundle: None })
        .build()
    {
        Ok(_) => panic!("expected FailedPrecondition on analysis fingerprint mismatch"),
        Err(e) => e,
    };
    assert!(matches!(err, ClientError::FailedPrecondition(_)), "got {err:?}");

    let wrong_bfp = BundleScoringFingerprint::from_spec(&BundleScoringSpec {
        model_id: "other-model".into(),
        revision: "99".into(),
        manifest_sha256: "f".repeat(64),
        task: "embed".into(),
        dim: 1024,
        pooling: "cls".into(),
        normalize: "l2".into(),
    });
    let local_err = match TurboClient::builder()
        .with_handshake(IndexGenerationHandshake { analysis: None, bundle: Some(wrong_bfp) })
        .with_local_embed_bundle(tiny_bundle_dir(), LocalEncoderOptions::default(), None)
        .build()
    {
        Ok(_) => panic!("expected FailedPrecondition on local bundle fingerprint mismatch"),
        Err(e) => e,
    };
    assert!(matches!(local_err, ClientError::FailedPrecondition(_)), "got {local_err:?}");

    // Remote connect() verifies ModelMetadata and refuses mismatched BundleScoringFingerprint,
    // while accepting the genuine local bundle's BundleScoringFingerprint!
    let genuine_local = TurboClient::builder()
        .with_local_embed_bundle(tiny_bundle_dir(), LocalEncoderOptions::default(), None)
        .build()
        .unwrap();
    let genuine_bfp = genuine_local.local_embed().unwrap().fingerprint();

    let server = start_test_server(None);
    let endpoint = format!("http://{}", server.local_addr());
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    rt.block_on(async {
        let remote_err = match TurboClient::builder()
            .with_handshake(IndexGenerationHandshake { analysis: None, bundle: Some(wrong_bfp) })
            .with_remote_endpoint(
                RemoteEndpointConfig::new(endpoint.clone()),
                Some(("tiny-bert-bundle".into(), String::new())),
            )
            .connect()
            .await
        {
            Ok(_) => panic!("expected FailedPrecondition on remote bundle fingerprint mismatch"),
            Err(e) => e,
        };
        assert!(matches!(remote_err, ClientError::FailedPrecondition(_)), "got {remote_err:?}");

        let ok_client = TurboClient::builder()
            .with_handshake(IndexGenerationHandshake { analysis: None, bundle: Some(genuine_bfp) })
            .with_remote_endpoint(RemoteEndpointConfig::new(endpoint), Some(("tiny-bert-bundle".into(), String::new())))
            .connect()
            .await
            .expect("matching remote BundleScoringFingerprint should pass handshake");
        assert!(ok_client.remote().is_some());
    });
}

#[test]
fn accent_folded_decomposed_marks_keep_full_grapheme_source_spans() {
    let cfg = FieldAnalysisConfig {
        tokenizer: TokenizerKind::Simple,
        normalizers: vec![NormalizerKind::StripInvisible, NormalizerKind::AccentFold, NormalizerKind::FullCaseFold],
        stemmer: StemmerKind::None,
        term_source: TermSourceKind::Tokens,
        sentence_language: "eng".into(),
        emit_lattice: true,
    };
    let client = TurboClient::builder().with_default_field(cfg).build().unwrap();
    let src = "caf\u{00E9} cafe\u{0301} foo\u{200B}bar";
    let analyzed = client.lexical().analyze(src).unwrap();
    let terms_and_spans: Vec<(&str, u32, u32, (u32, u32))> =
        analyzed.arcs.iter().map(|a| (a.term.as_str(), a.pos, a.pos_len, a.spans.utf8)).collect();
    assert_eq!(terms_and_spans[0], ("cafe", 0, 1, (0, 5)));
    assert_eq!(terms_and_spans[1], ("cafe", 1, 1, (6, 12)));
    assert_eq!(terms_and_spans[2], ("foo", 2, 1, (13, 16)));
    assert_eq!(terms_and_spans[3], ("foobar", 2, 2, (13, 22)));
    assert_eq!(terms_and_spans[4], ("bar", 3, 1, (19, 22)));
}
