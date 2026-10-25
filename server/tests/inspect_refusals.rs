//! Capacity refusal, non-finite vector, multibyte page boundary, and
//! overlapping-subword tests for `InspectDocumentWindow` and `ProjectChunks`
//! (issue #1220).

use tonic::Code;
use turbo_analysis::export::export;
use turbo_analysis::{Document, Ix, IxRange, Key, ModelRef as CoreModelRef, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{Boundaries, Chunks, Cut, Sentences, Subwords, Tokens, Vectors};
use turbo_kserve::api::{TURBO_E_CAPACITY, TURBO_E_INVALID_ARGUMENT};
use turbo_kserve::config::{Device, ModelConfig};
use turbo_kserve::inspect::heatmap::MAX_HEATMAP_BUCKETS;
use turbo_kserve::inspect::project::MAX_STRATEGY_BUDGETS;
use turbo_kserve::inspect::window::MAX_PAGE_SIZE;
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::result::{BudgetSpec, DocumentWindowRequest, OffsetMode, ProjectChunksRequest};
use turbo_kserve::results::document::codecs;
use turbo_kserve::{Serve, Server};

fn test_model_config() -> ModelConfig {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().expect("workspace root");
    let bundle = root.join("testdata/tiny-bert-bundle");
    ModelConfig {
        bundle: bundle.to_string_lossy().into_owned(),
        name: Some("tiny".into()),
        device: Device::Index(0),
        precision: 0,
        max_batch: 0,
        max_seq: 0,
        sessions: 1,
    }
}

fn check_turbo_error(status: &tonic::Status, expected_grpc: Code, expected_turbo: i32) {
    assert_eq!(status.code(), expected_grpc, "unexpected gRPC code: {status:?}");
    let tc = status
        .metadata()
        .get("turbo-code")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<i32>().ok())
        .expect("turbo-code metadata header must be present");
    assert_eq!(tc, expected_turbo, "unexpected turbo-code on {status:?}");
}

#[tokio::test]
async fn inspect_and_project_refusals_and_multibyte_boundaries() {
    let srv = Server::start(
        "127.0.0.1:0".parse().unwrap(),
        vec![test_model_config()],
        8 * 1024 * 1024,
        Serve { admin: false, ..Serve::default() },
    )
    .await
    .expect("start server");
    let addr = srv.local_addr();
    let mut client = AnalysisServiceClient::connect(format!("http://{addr}")).await.expect("connect");

    // Build a document with multibyte UTF-8 ("Café 😀 world"), overlapping Subwords,
    // and Tokens whose ends are non-monotonic (Overlap::Any: token 0 spans 0..16,
    // token 1 spans 6..10, token 2 spans 11..16).
    let text = "Café 😀 world";
    let mut doc = Document::new(text);

    let mut sent_spans = Spans::<Sentences>::new(ViewId::SOURCE);
    sent_spans.push(Span::new(0, text.len() as u32));
    doc.add(&Key::DEFAULT, Sentences { spans: sent_spans }, Provenance::by("test")).unwrap();

    let mut tok_spans = Spans::<Tokens>::new(ViewId::SOURCE);
    tok_spans.push(Span::new(0, 16));
    tok_spans.push(Span::new(6, 10));
    tok_spans.push(Span::new(11, 16));
    doc.add(&Key::DEFAULT, Tokens { spans: tok_spans, sentence: Some(vec![Ix::new(0); 3]) }, Provenance::by("test"))
        .unwrap();

    let model_ref = CoreModelRef { id: "test-embed".into(), revision: "r1".into(), sha256: "0123456789abcdef".into() };
    let mut sub_spans = Spans::<Subwords>::new(ViewId::SOURCE);
    sub_spans.push(Span::new(0, 5));
    sub_spans.push(Span::new(6, 10));
    sub_spans.push(Span::new(6, 10));
    sub_spans.push(Span::new(11, 16));
    doc.add(
        &Key::DEFAULT,
        Subwords {
            spans: sub_spans,
            ids: vec![1, 2, 3, 4],
            token: Some(vec![Ix::new(0), Ix::new(1), Ix::new(1), Ix::new(2)]),
            model: model_ref.clone(),
        },
        Provenance::by("test"),
    )
    .unwrap();

    doc.add(
        &Key::DEFAULT,
        Boundaries { cost: vec![Boundaries::UNIT], reason: vec![Cut::Sentence] },
        Provenance::by("test"),
    )
    .unwrap();

    let graph_bytes = export(&doc, codecs()).unwrap().to_bytes().unwrap();

    // 1. Overlapping Tokens (non-monotonic ends) and Subwords are all preserved across pages.
    let win_p0 = client
        .inspect_document_window(DocumentWindowRequest {
            graph: graph_bytes.clone(),
            view: 0,
            offsets: OffsetMode::Utf8 as i32,
            page_size: 8, // cuts inside "😀" (bytes 6..10), snapping down to byte 6
            page: 0,
            filter_layers: vec!["opennlp:tokens".into(), "opennlp:subwords".into()],
            heatmap_buckets: 0,
            xray_target_view: None,
        })
        .await
        .unwrap()
        .into_inner();
    let win_p1 = client
        .inspect_document_window(DocumentWindowRequest {
            graph: graph_bytes.clone(),
            view: 0,
            offsets: OffsetMode::Utf8 as i32,
            page_size: 8,
            page: 1,
            filter_layers: vec!["opennlp:tokens".into(), "opennlp:subwords".into()],
            heatmap_buckets: 0,
            xray_target_view: None,
        })
        .await
        .unwrap()
        .into_inner();

    let w0 = win_p0.window.unwrap();
    let w1 = win_p1.window.unwrap();
    assert_eq!(w0.byte_end, w1.byte_start, "adjacent pages must not overlap at multibyte boundaries");
    // In page 1 (bytes 6..16), tokens 0, 1, and 2 must all appear despite token 0 having end = 16 > token 1's end = 10.
    let seen_tokens: std::collections::BTreeSet<u32> = win_p1
        .segments
        .iter()
        .flat_map(|s| s.annotations.iter().filter(|a| a.layer == "opennlp:tokens").map(|a| a.item))
        .collect();
    assert!(seen_tokens.contains(&0) && seen_tokens.contains(&1) && seen_tokens.contains(&2));

    // 2. Capacity refusals: page_size > MAX_PAGE_SIZE and heatmap_buckets > MAX_HEATMAP_BUCKETS.
    let over_page = client
        .inspect_document_window(DocumentWindowRequest {
            graph: graph_bytes.clone(),
            view: 0,
            offsets: OffsetMode::Utf8 as i32,
            page_size: MAX_PAGE_SIZE + 1,
            page: 0,
            filter_layers: Vec::new(),
            heatmap_buckets: 0,
            xray_target_view: None,
        })
        .await
        .unwrap_err();
    check_turbo_error(&over_page, Code::OutOfRange, TURBO_E_CAPACITY);

    let over_buckets = client
        .inspect_document_window(DocumentWindowRequest {
            graph: graph_bytes.clone(),
            view: 0,
            offsets: OffsetMode::Utf8 as i32,
            page_size: 16_000,
            page: 0,
            filter_layers: Vec::new(),
            heatmap_buckets: MAX_HEATMAP_BUCKETS + 1,
            xray_target_view: None,
        })
        .await
        .unwrap_err();
    check_turbo_error(&over_buckets, Code::OutOfRange, TURBO_E_CAPACITY);

    // 3. Default strategy matrix (empty strategy_budgets) and over-limit strategy_budgets refusal.
    let default_matrix_resp = client
        .project_chunks(ProjectChunksRequest {
            graph: graph_bytes.clone(),
            offsets: OffsetMode::CodePoint as i32,
            pca_dims: 2,
            max_stack_segments: 5,
            include_drift_curve: true,
            include_strategy_matrix: true,
            strategy_budgets: Vec::new(),
        })
        .await
        .unwrap()
        .into_inner();
    let matrix = default_matrix_resp.strategy_matrix.unwrap();
    let strat_names: Vec<&str> = matrix.strategies.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(strat_names, vec!["greedy_sentences", "structure_only", "semantic_dp"]);

    let too_many_budgets: Vec<BudgetSpec> = (0..=MAX_STRATEGY_BUDGETS)
        .map(|i| BudgetSpec {
            name: format!("b{i}"),
            max_tokens: 64,
            target_tokens: 48,
            size_weight: 1000,
            overhead: 1,
        })
        .collect();
    let over_budgets = client
        .project_chunks(ProjectChunksRequest {
            graph: graph_bytes.clone(),
            offsets: OffsetMode::Utf8 as i32,
            pca_dims: 2,
            max_stack_segments: 5,
            include_drift_curve: false,
            include_strategy_matrix: true,
            strategy_budgets: too_many_budgets,
        })
        .await
        .unwrap_err();
    check_turbo_error(&over_budgets, Code::OutOfRange, TURBO_E_CAPACITY);

    // 4. Non-finite (NaN) coordinate in Vectors<Chunks> is refused with InvalidArgument.
    let mut nan_doc = doc.clone();
    let mut chunk_spans = Spans::<Chunks>::new(ViewId::SOURCE);
    chunk_spans.push(Span::new(0, text.len() as u32));
    nan_doc
        .add(
            &Key::DEFAULT,
            Chunks {
                spans: chunk_spans,
                level: vec![0],
                parent: vec![None],
                sentences: vec![IxRange::new(0, 1)],
                cut: vec![Cut::DocumentStart],
                cost: vec![0],
                model_tokens: vec![3],
                over_budget: vec![false],
                prompt_overhead: 1,
            },
            Provenance::by("test"),
        )
        .unwrap();
    nan_doc
        .add(
            &Key::DEFAULT,
            Vectors::<Chunks>::new(2, vec![1.0, f32::NAN], Vec::new(), model_ref),
            Provenance::by("test"),
        )
        .unwrap();
    let nan_bytes = export(&nan_doc, codecs()).unwrap().to_bytes().unwrap();
    let nan_err = client
        .project_chunks(ProjectChunksRequest {
            graph: nan_bytes,
            offsets: OffsetMode::Utf8 as i32,
            pca_dims: 2,
            max_stack_segments: 5,
            include_drift_curve: false,
            include_strategy_matrix: false,
            strategy_budgets: Vec::new(),
        })
        .await
        .unwrap_err();
    check_turbo_error(&nan_err, Code::InvalidArgument, TURBO_E_INVALID_ARGUMENT);

    drop(client);
    srv.stop().await;
}
