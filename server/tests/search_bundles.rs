//! Real `embed_bundle`, `rerank_bundle`, admin gate, snapshot confinement,
//! and gRPC error code tests for `TurboSearchService` (`#1221`).

mod common;

use std::collections::HashMap;

use tonic::Code;
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::proto::result::CapabilitiesRequest;
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::search::query_node::Clause;
use turbo_kserve::proto::search::turbo_search_service_client::TurboSearchServiceClient;
use turbo_kserve::proto::search::*;
use turbo_kserve::{Serve, Server};

fn test_space(dim: u32) -> SpaceSpec {
    SpaceSpec {
        model: Some(ModelIdentity {
            id: "turbo/test-axis-16".into(),
            revision: "v1".into(),
            sha256: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".into(),
        }),
        dim,
        normalization: NormalizationMode::None as i32,
    }
}

fn vec16(axis: usize) -> Vec<f32> {
    let mut v = vec![0.1f32; 16];
    v[axis % 16] = 1.0;
    v
}

#[tokio::test]
async fn embed_bundle_and_rerank_bundle_with_real_models() {
    let rerank_dir = common::make_rerank_bundle("PAIR_BERT");
    let cpu = common::cpu();
    let models = vec![
        ModelConfig {
            bundle: common::tiny().to_str().unwrap().into(),
            name: Some("tiny-embed".into()),
            device: Device::Index(cpu),
            precision: 0,
            max_batch: 0,
            max_seq: 0,
            sessions: 1,
        },
        ModelConfig {
            bundle: rerank_dir.0.to_str().unwrap().into(),
            name: Some("tiny-rerank".into()),
            device: Device::Index(cpu),
            precision: 0,
            max_batch: 0,
            max_seq: 0,
            sessions: 1,
        },
    ];
    let srv = Server::start("127.0.0.1:0".parse().unwrap(), models, MAX, Serve::default()).await.unwrap();
    srv.load().await.unwrap();
    let addr = srv.local_addr();
    let mut analysis = AnalysisServiceClient::connect(format!("http://{addr}")).await.unwrap();
    let caps = analysis.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
    let embed_cap = caps.bundles.iter().find(|b| b.name == "tiny-embed").unwrap();
    let mref = embed_cap.model.as_ref().unwrap();
    let embed_space = SpaceSpec {
        model: Some(ModelIdentity {
            id: mref.id.clone(),
            revision: mref.revision.clone(),
            sha256: mref.sha256.clone(),
        }),
        dim: embed_cap.widths[0],
        normalization: NormalizationMode::Unit as i32,
    };

    let mut client = TurboSearchServiceClient::connect(format!("http://{addr}")).await.unwrap();

    // Refuses embed_bundle when SpaceSpec model identity does not match the loaded bundle
    let bad_bundle_spec = client
        .create_index(CreateIndexRequest {
            spec: Some(IndexSpec {
                index_name: "bad-bundle-idx".into(),
                space: Some(test_space(16)),
                metric: DistanceMetric::Cosine as i32,
                storage: VectorStorageKind::FlatF32 as i32,
                quant_bits: 0,
                quant_calibrate: false,
                collection_name: String::new(),
                embed_bundle: "tiny-embed".into(),
                chunk_budget_tokens: 16,
            }),
            load_from_path: String::new(),
        })
        .await
        .unwrap_err();
    assert_eq!(bad_bundle_spec.code(), Code::InvalidArgument);

    client
        .create_index(CreateIndexRequest {
            spec: Some(IndexSpec {
                index_name: "live-embed".into(),
                space: Some(embed_space.clone()),
                metric: DistanceMetric::Cosine as i32,
                storage: VectorStorageKind::FlatF32 as i32,
                quant_bits: 0,
                quant_calibrate: false,
                collection_name: String::new(),
                embed_bundle: "tiny-embed".into(),
                chunk_budget_tokens: 6,
            }),
            load_from_path: String::new(),
        })
        .await
        .unwrap();

    // Ingest raw text documents (including whitespace-only text) without precomputed vectors
    let docs = vec![
        IndexDocumentRequest {
            index_name: "live-embed".into(),
            doc_id: "d1".into(),
            text: "Alpha beta gamma delta. Epsilon zeta eta theta.".into(),
            graph: Vec::new(),
            chunks: Vec::new(),
            attributes: HashMap::new(),
        },
        IndexDocumentRequest {
            index_name: "live-embed".into(),
            doc_id: "d-ws".into(),
            text: "   \n\t  ".into(),
            graph: Vec::new(),
            chunks: Vec::new(),
            attributes: HashMap::new(),
        },
    ];
    let ing = client.index_documents(tokio_stream::iter(docs)).await.unwrap().into_inner();
    assert_eq!(ing.indexed_documents, 2);
    assert!(ing.indexed_chunks >= 2);

    // Query using SemanticClause.query_text + cross-encoder rerank_bundle
    let res = client
        .search(SearchRequest {
            index_name: "live-embed".into(),
            natural_query: "alpha beta".into(),
            query_ast: Some(QueryNode {
                clause: Some(Clause::Semantic(SemanticClause {
                    query_vector: Vec::new(),
                    query_space: None,
                    query_text: "alpha beta".into(),
                    top_k: 4,
                })),
            }),
            query_vector: Vec::new(),
            query_space: None,
            top_k: 4,
            parent_expansion: None,
            rerank_bundle: "tiny-rerank".into(),
        })
        .await
        .unwrap()
        .into_inner();
    assert!(!res.hits.is_empty());
    assert!(res.hits[0].attribution.as_ref().unwrap().matched_spans.iter().any(|s| s.source == "rerank"));

    // Reranking without natural_query or query_text is refused with InvalidArgument
    let no_q_rerank = client
        .search(SearchRequest {
            index_name: "live-embed".into(),
            natural_query: String::new(),
            query_ast: Some(QueryNode {
                clause: Some(Clause::Term(TermClause { terms: vec!["alpha".into()], mode: 0, top_k: 2 })),
            }),
            query_vector: Vec::new(),
            query_space: None,
            top_k: 2,
            parent_expansion: None,
            rerank_bundle: "tiny-rerank".into(),
        })
        .await
        .unwrap_err();
    assert_eq!(no_q_rerank.code(), Code::InvalidArgument);

    srv.stop().await;
}

#[tokio::test]
async fn admin_gate_snapshot_confinement_and_error_codes() {
    let snap_dir = std::env::temp_dir().join(format!("turbo-search-sec-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&snap_dir);
    std::fs::create_dir_all(&snap_dir).unwrap();
    let default_models = || {
        vec![ModelConfig {
            bundle: common::tiny().to_str().unwrap().into(),
            name: None,
            device: Device::Select,
            precision: 0,
            max_batch: 0,
            max_seq: 0,
            sessions: 1,
        }]
    };
    let srv_no_admin =
        Server::start("127.0.0.1:0".parse().unwrap(), default_models(), MAX, Serve { admin: false, catalog: None })
            .await
            .unwrap();
    let mut c_no_admin =
        TurboSearchServiceClient::connect(format!("http://{}", srv_no_admin.local_addr())).await.unwrap();

    assert_eq!(
        c_no_admin
            .persist_index(PersistIndexRequest { index_name: "x".into(), path: "a.snap".into() })
            .await
            .unwrap_err()
            .code(),
        Code::Unimplemented
    );
    assert_eq!(
        c_no_admin
            .create_index(CreateIndexRequest { spec: None, load_from_path: "a.snap".into() })
            .await
            .unwrap_err()
            .code(),
        Code::Unimplemented
    );
    srv_no_admin.stop().await;

    let srv =
        Server::start("127.0.0.1:0".parse().unwrap(), default_models(), MAX, Serve { admin: true, catalog: None })
            .await
            .unwrap();
    srv.set_search_snapshot_dir(&snap_dir);
    let mut c = TurboSearchServiceClient::connect(format!("http://{}", srv.local_addr())).await.unwrap();

    // Missing index, alias, and collection return NotFound
    assert_eq!(
        c.delete_index(DeleteIndexRequest { index_name: "missing".into() }).await.unwrap_err().code(),
        Code::NotFound
    );
    assert_eq!(
        c.delete_alias(DeleteAliasRequest { alias: "missing".into() }).await.unwrap_err().code(),
        Code::NotFound
    );
    assert_eq!(
        c.get_collection(GetCollectionRequest { collection_name: "missing".into() }).await.unwrap_err().code(),
        Code::NotFound
    );

    c.create_index(CreateIndexRequest {
        spec: Some(IndexSpec {
            index_name: "sec-idx".into(),
            space: Some(test_space(16)),
            metric: DistanceMetric::Cosine as i32,
            storage: VectorStorageKind::FlatF32 as i32,
            ..Default::default()
        }),
        load_from_path: String::new(),
    })
    .await
    .unwrap();

    // Duplicate CreateIndex returns AlreadyExists
    let dup = c
        .create_index(CreateIndexRequest {
            spec: Some(IndexSpec { index_name: "sec-idx".into(), space: Some(test_space(16)), ..Default::default() }),
            load_from_path: String::new(),
        })
        .await
        .unwrap_err();
    assert_eq!(dup.code(), Code::AlreadyExists);

    // Path traversal / separators in PersistIndex and CreateIndex.load_from_path refused with InvalidArgument
    for bad_path in ["../escape.snap", "/etc/evil.snap", "sub/file.snap", ".."] {
        assert_eq!(
            c.persist_index(PersistIndexRequest { index_name: "sec-idx".into(), path: bad_path.into() })
                .await
                .unwrap_err()
                .code(),
            Code::InvalidArgument
        );
        assert_eq!(
            c.create_index(CreateIndexRequest { spec: None, load_from_path: bad_path.into() })
                .await
                .unwrap_err()
                .code(),
            Code::InvalidArgument
        );
    }

    // Corrupt snapshot file with huge untrusted count fails fast with DataLoss (not OOM)
    let mut corrupt = b"TSRCHIX\x01".to_vec();
    corrupt.extend_from_slice(&[0u8; 24]);
    std::fs::write(snap_dir.join("corrupt.snap"), &corrupt).unwrap();
    assert_eq!(
        c.create_index(CreateIndexRequest { spec: None, load_from_path: "corrupt.snap".into() })
            .await
            .unwrap_err()
            .code(),
        Code::DataLoss
    );

    // Mid-codepoint span, dimension mismatch, omitted query_space, and Reindex into different space without embed_bundle
    let bad_span_doc = IndexDocumentRequest {
        index_name: "sec-idx".into(),
        doc_id: "utf8-doc".into(),
        text: "élan".into(),
        graph: Vec::new(),
        chunks: vec![ChunkInput { span_start: 1, span_end: 5, ..Default::default() }],
        attributes: HashMap::new(),
    };
    assert_eq!(
        c.index_documents(tokio_stream::iter(vec![bad_span_doc])).await.unwrap_err().code(),
        Code::InvalidArgument
    );

    let bad_dim_doc = IndexDocumentRequest {
        index_name: "sec-idx".into(),
        doc_id: "dim-doc".into(),
        text: "hello".into(),
        graph: Vec::new(),
        chunks: vec![ChunkInput { span_start: 0, span_end: 5, dense_vector: vec![1.0; 7], ..Default::default() }],
        attributes: HashMap::new(),
    };
    assert_eq!(
        c.index_documents(tokio_stream::iter(vec![bad_dim_doc])).await.unwrap_err().code(),
        Code::InvalidArgument
    );

    let good_doc = IndexDocumentRequest {
        index_name: "sec-idx".into(),
        doc_id: "ok-doc".into(),
        text: "hello world".into(),
        graph: Vec::new(),
        chunks: vec![ChunkInput { span_start: 0, span_end: 11, dense_vector: vec16(0), ..Default::default() }],
        attributes: HashMap::new(),
    };
    c.index_documents(tokio_stream::iter(vec![good_doc])).await.unwrap();

    // Omitted query_space on raw query_vector is refused
    let no_qspace = c
        .search(SearchRequest {
            index_name: "sec-idx".into(),
            query_vector: vec16(0),
            query_space: None,
            top_k: 5,
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert_eq!(no_qspace.code(), Code::InvalidArgument);

    // Reindex into a different model space without embed_bundle is refused
    let mut other_space = test_space(16);
    other_space.model.as_mut().unwrap().id = "other/encoder".into();
    let bad_reindex = c
        .reindex(ReindexRequest {
            source_index: "sec-idx".into(),
            target_spec: Some(IndexSpec {
                index_name: "sec-idx-other".into(),
                space: Some(other_space),
                metric: DistanceMetric::Cosine as i32,
                storage: VectorStorageKind::FlatF32 as i32,
                ..Default::default()
            }),
            swap_alias: String::new(),
        })
        .await
        .unwrap_err();
    assert_eq!(bad_reindex.code(), Code::InvalidArgument);

    srv.stop().await;
    let _ = std::fs::remove_dir_all(&snap_dir);
}
