//! Real `embed_bundle`, `rerank_bundle`, admin gate, snapshot confinement,
//! and gRPC error code tests for `TurboSearchService` (`#1221`).

mod common;

use std::collections::HashMap;
use std::time::Duration;

use tokio_stream::StreamExt;
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

fn default_models() -> Vec<ModelConfig> {
    vec![ModelConfig {
        bundle: common::tiny().to_str().unwrap().into(),
        name: None,
        device: Device::Select,
        precision: 0,
        max_batch: 0,
        max_seq: 0,
        sessions: 1,
    }]
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
            ..Default::default()
        },
        IndexDocumentRequest {
            index_name: "live-embed".into(),
            doc_id: "d-ws".into(),
            text: "   \n\t  ".into(),
            graph: Vec::new(),
            chunks: Vec::new(),
            attributes: HashMap::new(),
            ..Default::default()
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
    let srv_no_admin = Server::start(
        "127.0.0.1:0".parse().unwrap(),
        default_models(),
        MAX,
        Serve { admin: false, ..Serve::default() },
    )
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
        Server::start("127.0.0.1:0".parse().unwrap(), default_models(), MAX, Serve { admin: true, ..Serve::default() })
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
        ..Default::default()
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
        ..Default::default()
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
        ..Default::default()
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

#[tokio::test]
async fn empty_server_listings_collection_deletion_and_watchers() {
    let srv = Server::start("127.0.0.1:0".parse().unwrap(), default_models(), MAX, Serve::default()).await.unwrap();
    let mut c = TurboSearchServiceClient::connect(format!("http://{}", srv.local_addr())).await.unwrap();

    // Empty server state for ListIndexes, ListAliases, ListCollections, and DeleteCollection
    assert!(c.list_indexes(ListIndexesRequest {}).await.unwrap().into_inner().indexes.is_empty());
    assert!(c.list_aliases(ListAliasesRequest {}).await.unwrap().into_inner().aliases.is_empty());
    assert!(c.list_collections(ListCollectionsRequest {}).await.unwrap().into_inner().collections.is_empty());
    for missing in ["", "missing-col"] {
        let err = c.delete_collection(DeleteCollectionRequest { collection_name: missing.into() }).await.unwrap_err();
        assert_eq!(err.code(), Code::NotFound);
        assert_eq!(err.metadata().get("turbo-code").unwrap(), "1280");
    }

    for (ix_name, col_name) in [("idx-b", "col-b"), ("idx-a", "col-a")] {
        let spec = IndexSpec {
            index_name: ix_name.into(),
            space: Some(test_space(16)),
            collection_name: col_name.into(),
            ..Default::default()
        };
        c.create_index(CreateIndexRequest { spec: Some(spec), load_from_path: String::new() }).await.unwrap();
    }
    c.set_alias(SetAliasRequest { alias: "alias-z".into(), index_name: "idx-a".into() }).await.unwrap();
    c.set_alias(SetAliasRequest { alias: "alias-a".into(), index_name: "idx-b".into() }).await.unwrap();

    let ixs = c.list_indexes(ListIndexesRequest {}).await.unwrap().into_inner().indexes;
    assert_eq!(ixs.iter().map(|i| i.index_name.as_str()).collect::<Vec<_>>(), vec!["idx-a", "idx-b"]);
    let aliases = c.list_aliases(ListAliasesRequest {}).await.unwrap().into_inner().aliases;
    assert_eq!(
        aliases,
        vec![
            AliasEntry { alias: "alias-a".into(), index_name: "idx-b".into() },
            AliasEntry { alias: "alias-z".into(), index_name: "idx-a".into() },
        ]
    );

    // DeleteIndex removes idx-b from ListIndexes and prunes alias-a from ListAliases
    c.delete_index(DeleteIndexRequest { index_name: "idx-b".into() }).await.unwrap();
    assert_eq!(c.list_indexes(ListIndexesRequest {}).await.unwrap().into_inner().indexes.len(), 1);
    assert_eq!(
        c.list_aliases(ListAliasesRequest {}).await.unwrap().into_inner().aliases,
        vec![AliasEntry { alias: "alias-z".into(), index_name: "idx-a".into() }]
    );

    // Configure col-a with high oov_alert_threshold (0.90) and col-b with 0.0
    for (name, term, thresh) in [("col-a", "alpha", 0.90), ("col-b", "beta", 0.0)] {
        let req = SetCollectionRequest {
            collection_name: name.into(),
            baseline_terms: vec![term.into()],
            oov_alert_threshold: thresh,
        };
        c.set_collection(req).await.unwrap();
    }
    let cols = c.list_collections(ListCollectionsRequest {}).await.unwrap().into_inner().collections;
    assert_eq!(cols.iter().map(|x| x.collection_name.as_str()).collect::<Vec<_>>(), vec!["col-a", "col-b"]);

    let watch_req = WatchCollectionRequest { collection_name: "col-a".into(), min_oov_ratio: 0.40 };
    let mut watch_a = c.watch_collection(watch_req).await.unwrap().into_inner();

    // First doc has oov_ratio = 0.50 < 0.90 threshold, so watch_a is not notified before deletion
    let doc1 = IndexDocumentRequest {
        index_name: "idx-a".into(),
        doc_id: "d1".into(),
        text: "alpha beta".into(),
        ..Default::default()
    };
    c.index_documents(tokio_stream::iter(vec![doc1])).await.unwrap();
    assert!(tokio::time::timeout(Duration::from_millis(60), watch_a.next()).await.is_err());

    // Delete col-a: removed from ListCollections, watcher stays open without spurious event,
    // and subsequent indexing into idx-a recreates col-a with default threshold 0.0 and notifies watch_a
    c.delete_collection(DeleteCollectionRequest { collection_name: "col-a".into() }).await.unwrap();
    let cols_after = c.list_collections(ListCollectionsRequest {}).await.unwrap().into_inner().collections;
    assert_eq!(cols_after.iter().map(|x| x.collection_name.as_str()).collect::<Vec<_>>(), vec!["col-b"]);
    assert!(tokio::time::timeout(Duration::from_millis(60), watch_a.next()).await.is_err());

    let doc2 = IndexDocumentRequest {
        index_name: "idx-a".into(),
        doc_id: "d2".into(),
        text: "alpha gamma".into(),
        ..Default::default()
    };
    c.index_documents(tokio_stream::iter(vec![doc2])).await.unwrap();
    let ev = tokio::time::timeout(Duration::from_secs(2), watch_a.next()).await.unwrap().unwrap().unwrap();
    assert_eq!(ev.collection_name, "col-a");
    assert_eq!(ev.source_index, "idx-a");
    assert_eq!(ev.report.unwrap().total_units, 1);

    drop(watch_a);
    srv.stop().await;
}
