//! End-to-end integration tests for `TurboSearchService` (issue `#1221`).

mod common;

use std::collections::HashMap;
use std::time::Duration;

use tokio_stream::StreamExt;
use tonic::Code;
use tonic::transport::Channel;
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::proto::search::query_node::Clause;
use turbo_kserve::proto::search::turbo_search_service_client::TurboSearchServiceClient;
use turbo_kserve::proto::search::*;
use turbo_kserve::{Serve, Server};

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
async fn turbo_search_service_end_to_end() {
    let snap_dir = std::env::temp_dir().join(format!("turbo-search-snaps-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&snap_dir);
    std::fs::create_dir_all(&snap_dir).unwrap();
    let srv = Server::start("127.0.0.1:0".parse().unwrap(), default_models(), MAX, Serve { admin: true })
        .await
        .expect("start server");
    srv.set_search_snapshot_dir(&snap_dir);
    let addr = srv.local_addr();
    let channel = Channel::from_shared(format!("http://{addr}")).unwrap().connect().await.unwrap();
    let mut client = TurboSearchServiceClient::new(channel);

    let providers = client.list_search_providers(ListSearchProvidersRequest {}).await.unwrap().into_inner().providers;
    assert!(providers.iter().any(|p| p.id == "turbo-chunker:bm25-posting"));
    assert!(providers.iter().any(|p| p.id == "turbo-vector-index:turboquant"));
    assert!(providers.iter().any(|p| p.id == "turbo-vector-index:token-maxsim"));

    let col = client
        .set_collection(SetCollectionRequest {
            collection_name: "finance-corpus".into(),
            baseline_terms: vec!["neural".into(), "retrieval".into(), "pipeline".into()],
            oov_alert_threshold: 0.10,
        })
        .await
        .unwrap()
        .into_inner()
        .collection
        .unwrap();
    assert_eq!(col.collection_name, "finance-corpus");

    let mut watch_stream = client
        .watch_collection(WatchCollectionRequest { collection_name: "finance-corpus".into(), min_oov_ratio: 0.10 })
        .await
        .unwrap()
        .into_inner();
    let mut strict_watch = client
        .watch_collection(WatchCollectionRequest { collection_name: "finance-corpus".into(), min_oov_ratio: 0.99 })
        .await
        .unwrap()
        .into_inner();

    for (name, storage, bits, cal) in
        [("idx-flat", VectorStorageKind::FlatF32, 0, false), ("idx-quant", VectorStorageKind::TurboQuant, 4, true)]
    {
        client
            .create_index(CreateIndexRequest {
                spec: Some(IndexSpec {
                    index_name: name.into(),
                    space: Some(test_space(16)),
                    metric: DistanceMetric::Cosine as i32,
                    storage: storage as i32,
                    quant_bits: bits,
                    quant_calibrate: cal,
                    collection_name: "finance-corpus".into(),
                    embed_bundle: String::new(),
                    chunk_budget_tokens: 32,
                }),
                load_from_path: String::new(),
            })
            .await
            .unwrap();
    }

    client.set_alias(SetAliasRequest { alias: "active".into(), index_name: "idx-flat".into() }).await.unwrap();

    let doc1_text = "Neural retrieval pipeline in Paris for $1200. Hybrid search combines BM25 and SPLADE.";
    let doc2_text = "Neural sparse retrieval pipeline in Berlin. TurboQuant compresses dense vectors.";

    let make_docs = |target: &str| {
        vec![
            IndexDocumentRequest {
                index_name: target.into(),
                doc_id: "doc-1".into(),
                text: doc1_text.into(),
                graph: Vec::new(),
                chunks: vec![
                    ChunkInput {
                        chunk_id: "doc-1#0".into(),
                        level_kind: "sentence".into(),
                        span_start: 0,
                        span_end: 45,
                        parent_start: 0,
                        parent_end: doc1_text.len() as u32,
                        has_parent: true,
                        dense_vector: vec16(0),
                        sparse_terms: vec![SparseEntryProto {
                            term_id: 101,
                            term_text: "paris".into(),
                            weight: 2.5,
                            span_start: 29,
                            span_end: 34,
                            has_span: true,
                        }],
                        token_vectors: vec![
                            TokenVectorInput { values: vec16(0), span_start: 0, span_end: 6, has_span: true },
                            TokenVectorInput { values: vec16(1), span_start: 29, span_end: 34, has_span: true },
                        ],
                        attributes: HashMap::from([("region".into(), "eu".into())]),
                    },
                    ChunkInput {
                        chunk_id: "doc-1#1".into(),
                        level_kind: "sentence".into(),
                        span_start: 46,
                        span_end: doc1_text.len() as u32,
                        parent_start: 0,
                        parent_end: doc1_text.len() as u32,
                        has_parent: true,
                        dense_vector: vec16(2),
                        sparse_terms: vec![SparseEntryProto {
                            term_id: 202,
                            term_text: "splade".into(),
                            weight: 3.0,
                            span_start: 78,
                            span_end: 84,
                            has_span: true,
                        }],
                        token_vectors: vec![TokenVectorInput {
                            values: vec16(2),
                            span_start: 78,
                            span_end: 84,
                            has_span: true,
                        }],
                        attributes: HashMap::from([("region".into(), "eu".into())]),
                    },
                ],
                attributes: HashMap::new(),
            },
            IndexDocumentRequest {
                index_name: target.into(),
                doc_id: "doc-2".into(),
                text: doc2_text.into(),
                graph: Vec::new(),
                chunks: vec![ChunkInput {
                    chunk_id: "doc-2#0".into(),
                    level_kind: "sentence".into(),
                    span_start: 0,
                    span_end: 43,
                    parent_start: 0,
                    parent_end: doc2_text.len() as u32,
                    has_parent: true,
                    dense_vector: vec16(3),
                    sparse_terms: vec![SparseEntryProto {
                        term_id: 303,
                        term_text: "berlin".into(),
                        weight: 2.0,
                        span_start: 36,
                        span_end: 42,
                        has_span: true,
                    }],
                    token_vectors: vec![TokenVectorInput {
                        values: vec16(3),
                        span_start: 36,
                        span_end: 42,
                        has_span: true,
                    }],
                    attributes: HashMap::from([("region".into(), "de".into())]),
                }],
                attributes: HashMap::new(),
            },
        ]
    };

    let idx_res = client.index_documents(tokio_stream::iter(make_docs("active"))).await.unwrap().into_inner();
    assert_eq!(idx_res.index_name, "idx-flat");
    assert_eq!(idx_res.indexed_documents, 2);
    assert_eq!(idx_res.indexed_chunks, 3);
    client.index_documents(tokio_stream::iter(make_docs("idx-quant"))).await.unwrap();

    let ev = tokio::time::timeout(Duration::from_secs(2), watch_stream.next()).await.unwrap().unwrap().unwrap();
    assert_eq!(ev.collection_name, "finance-corpus");
    assert!(ev.report.unwrap().oov_ratio > 0.10);
    assert!(tokio::time::timeout(Duration::from_millis(80), strict_watch.next()).await.is_err());

    // Natural-language query with extracted filter ($1200) narrows results to doc-1#0 only
    let nl_res = client
        .search(SearchRequest {
            index_name: "active".into(),
            natural_query: "neural retrieval in Paris for $1200".into(),
            query_ast: None,
            query_vector: vec16(0),
            query_space: Some(test_space(16)),
            top_k: 5,
            parent_expansion: Some(ParentExpansionSpec { enabled: true, max_parent_tokens: 100 }),
            rerank_bundle: String::new(),
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(nl_res.hits.len(), 1);
    assert_eq!(nl_res.hits[0].chunk_id, "doc-1#0");
    assert_eq!(nl_res.hits[0].parent_text, doc1_text);
    assert!(!nl_res.document_heat.is_empty());

    // Phrase query with slop = 0 vs slop = 1
    for (slop, want_hits) in [(0, 1), (1, 2)] {
        let pr = client
            .search(SearchRequest {
                index_name: "active".into(),
                natural_query: String::new(),
                query_ast: Some(QueryNode {
                    clause: Some(Clause::Phrase(PhraseClause {
                        terms: vec!["neural".into(), "retrieval".into()],
                        slop,
                        top_k: 5,
                    })),
                }),
                query_vector: Vec::new(),
                query_space: None,
                top_k: 5,
                parent_expansion: None,
                rerank_bundle: String::new(),
            })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(pr.hits.len(), want_hits);
    }

    // Compound QueryNode AST (RRF over TurboQuant + Sparse + LateInteraction + Boost + Filter)
    let mut q_toks = vec16(0);
    q_toks.extend(vec16(1));
    let compound_res = client
        .search(SearchRequest {
            index_name: "idx-quant".into(),
            natural_query: String::new(),
            query_ast: Some(QueryNode {
                clause: Some(Clause::Join(JoinClause {
                    op: JoinOperator::And as i32,
                    fusion: FusionStrategy::ReciprocalRank as i32,
                    rrf_k: 60.0,
                    children: vec![
                        QueryNode {
                            clause: Some(Clause::Semantic(SemanticClause {
                                query_vector: vec16(0),
                                query_space: Some(test_space(16)),
                                query_text: String::new(),
                                top_k: 5,
                            })),
                        },
                        QueryNode {
                            clause: Some(Clause::Sparse(SparseClause {
                                query_terms: vec![SparseEntryProto {
                                    term_id: 101,
                                    term_text: "paris".into(),
                                    weight: 1.5,
                                    span_start: 0,
                                    span_end: 0,
                                    has_span: false,
                                }],
                                top_k: 5,
                            })),
                        },
                        QueryNode {
                            clause: Some(Clause::Boost(Box::new(BoostClause {
                                boost: 2.0,
                                child: Some(Box::new(QueryNode {
                                    clause: Some(Clause::LateInteraction(LateInteractionClause {
                                        query_token_vectors: q_toks,
                                        query_space: Some(test_space(16)),
                                        top_k: 5,
                                    })),
                                })),
                            }))),
                        },
                        QueryNode {
                            clause: Some(Clause::Filter(FilterClause {
                                doc_id: "doc-1".into(),
                                level_kind: "sentence".into(),
                                attributes: HashMap::from([("region".into(), "eu".into())]),
                                contains_text: "Paris".into(),
                            })),
                        },
                    ],
                })),
            }),
            query_vector: Vec::new(),
            query_space: None,
            top_k: 5,
            parent_expansion: None,
            rerank_bundle: String::new(),
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(compound_res.hits.len(), 1);
    let sources: Vec<&str> =
        compound_res.hits[0].attribution.as_ref().unwrap().matched_spans.iter().map(|s| s.source.as_str()).collect();
    for want in ["turboquant", "sparse", "maxsim", "filter"] {
        assert!(sources.contains(&want), "missing attribution source {want}");
    }

    // PersistIndex + reload via CreateIndex(load_from_path) + SealIndex + Reindex with alias swap
    let persisted = client
        .persist_index(PersistIndexRequest { index_name: "idx-quant".into(), path: "quant.snap".into() })
        .await
        .unwrap()
        .into_inner();
    assert!(persisted.bytes_written > 0);

    let reloaded = client
        .create_index(CreateIndexRequest {
            spec: Some(IndexSpec { index_name: "idx-quant-reloaded".into(), ..Default::default() }),
            load_from_path: "quant.snap".into(),
        })
        .await
        .unwrap()
        .into_inner()
        .summary
        .unwrap();
    assert_eq!(reloaded.chunk_count, 3);

    let dup_reload = client
        .create_index(CreateIndexRequest {
            spec: Some(IndexSpec { index_name: "idx-quant-reloaded".into(), ..Default::default() }),
            load_from_path: "quant.snap".into(),
        })
        .await
        .unwrap_err();
    assert_eq!(dup_reload.code(), Code::AlreadyExists);

    let sealed = client.seal_index(SealIndexRequest { index_name: "idx-flat".into() }).await.unwrap().into_inner();
    assert!(sealed.summary.unwrap().sealed);
    assert_eq!(
        client.index_documents(tokio_stream::iter(make_docs("idx-flat"))).await.unwrap_err().code(),
        Code::FailedPrecondition
    );

    let reindexed = client
        .reindex(ReindexRequest {
            source_index: "idx-flat".into(),
            target_spec: Some(IndexSpec {
                index_name: "idx-v2".into(),
                space: Some(test_space(16)),
                metric: DistanceMetric::Cosine as i32,
                storage: VectorStorageKind::TurboQuant as i32,
                quant_bits: 4,
                quant_calibrate: false,
                collection_name: "finance-corpus".into(),
                embed_bundle: String::new(),
                chunk_budget_tokens: 32,
            }),
            swap_alias: "active".into(),
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(reindexed.swapped_alias, "active");
    assert_eq!(reindexed.summary.unwrap().chunk_count, 3);

    drop(watch_stream);
    drop(strict_watch);
    drop(client);
    srv.stop().await;
    let _ = std::fs::remove_dir_all(&snap_dir);
}
