//! End-to-end integration tests for `TurboSearchService` (issue `#1221`).

use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use tokio_stream::StreamExt;
use tonic::transport::Channel;
use turbo_kserve::config::{Device, ModelConfig};
use turbo_kserve::proto::search::query_node::Clause;
use turbo_kserve::proto::search::turbo_search_service_client::TurboSearchServiceClient;
use turbo_kserve::proto::search::*;
use turbo_kserve::{Serve, Server};

fn test_space(dim: u32) -> SpaceSpec {
    SpaceSpec {
        model: Some(ModelIdentity {
            id: "intfloat/multilingual-e5-small".into(),
            revision: "main".into(),
            sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".into(),
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
    let tiny = Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/tiny-bert-bundle");
    let models = vec![ModelConfig {
        bundle: tiny.to_str().unwrap().to_string(),
        name: None,
        device: Device::Select,
        precision: 0,
        max_batch: 0,
        max_seq: 0,
        sessions: 1,
    }];
    let srv = Server::start("127.0.0.1:0".parse().unwrap(), models, 16 << 20, Serve { admin: false })
        .await
        .expect("start server");
    let addr = srv.local_addr();
    let channel = Channel::from_shared(format!("http://{addr}")).unwrap().connect().await.unwrap();
    let mut client = TurboSearchServiceClient::new(channel);

    // 1. Provider discovery
    let providers = client.list_search_providers(ListSearchProvidersRequest {}).await.unwrap().into_inner().providers;
    assert!(providers.iter().any(|p| p.id == "turbo-chunker:bm25-posting"));
    assert!(providers.iter().any(|p| p.id == "turbo-vector-index:turboquant"));
    assert!(providers.iter().any(|p| p.id == "turbo-vector-index:token-maxsim"));

    // 2. Collection setup + WatchCollection stream
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

    // 3. Create FlatF32 index and TurboQuant (4-bit) index
    client
        .create_index(CreateIndexRequest {
            spec: Some(IndexSpec {
                index_name: "idx-flat".into(),
                space: Some(test_space(16)),
                metric: DistanceMetric::Cosine as i32,
                storage: VectorStorageKind::FlatF32 as i32,
                quant_bits: 0,
                quant_calibrate: false,
                collection_name: "finance-corpus".into(),
                embed_bundle: String::new(),
                chunk_budget_tokens: 32,
            }),
            load_from_path: String::new(),
        })
        .await
        .unwrap();

    client
        .create_index(CreateIndexRequest {
            spec: Some(IndexSpec {
                index_name: "idx-quant".into(),
                space: Some(test_space(16)),
                metric: DistanceMetric::Cosine as i32,
                storage: VectorStorageKind::TurboQuant as i32,
                quant_bits: 4,
                quant_calibrate: true,
                collection_name: "finance-corpus".into(),
                embed_bundle: String::new(),
                chunk_budget_tokens: 32,
            }),
            load_from_path: String::new(),
        })
        .await
        .unwrap();

    // Set alias `active` -> `idx-flat`
    client.set_alias(SetAliasRequest { alias: "active".into(), index_name: "idx-flat".into() }).await.unwrap();

    // 4. Index documents into `active` (resolves to `idx-flat`) and `idx-quant`
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
    assert!(idx_res.drift.unwrap().oov_ratio > 0.10);

    client.index_documents(tokio_stream::iter(make_docs("idx-quant"))).await.unwrap();

    // Verify WatchCollection emitted live OOV drift event
    let ev = tokio::time::timeout(Duration::from_secs(2), watch_stream.next())
        .await
        .expect("watch event timeout")
        .expect("stream open")
        .expect("event ok");
    assert_eq!(ev.collection_name, "finance-corpus");
    let rep = ev.report.unwrap();
    assert!(rep.oov_ratio > 0.10);
    assert!(!rep.new_terms.is_empty());

    // 5. Natural-language query with turbo-query filter extraction + ParentExpansion
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
    assert_eq!(nl_res.resolved_index, "idx-flat");
    assert!(!nl_res.analyzed_query_terms.is_empty());
    assert!(nl_res.extracted_filters.iter().any(|f| f.kind == "place" || f.kind == "money"));
    assert_eq!(nl_res.hits[0].chunk_id, "doc-1#0");
    assert_eq!(nl_res.hits[0].parent_text, doc1_text);
    assert!(!nl_res.document_heat.is_empty());

    // 6. Phrase query with slop = 0 vs slop = 1
    let exact_phrase = client
        .search(SearchRequest {
            index_name: "active".into(),
            natural_query: String::new(),
            query_ast: Some(QueryNode {
                clause: Some(Clause::Phrase(PhraseClause {
                    terms: vec!["neural".into(), "retrieval".into()],
                    slop: 0,
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
    assert_eq!(exact_phrase.hits.len(), 1);
    assert_eq!(exact_phrase.hits[0].chunk_id, "doc-1#0");

    let slop_phrase = client
        .search(SearchRequest {
            index_name: "active".into(),
            natural_query: String::new(),
            query_ast: Some(QueryNode {
                clause: Some(Clause::Phrase(PhraseClause {
                    terms: vec!["neural".into(), "retrieval".into()],
                    slop: 1,
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
    assert_eq!(slop_phrase.hits.len(), 2);

    // 7. Compound QueryNode AST (RRF over TurboQuant + Sparse + LateInteraction + Boost + Filter)
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
    let top_hit = &compound_res.hits[0];
    assert_eq!(top_hit.chunk_id, "doc-1#0");
    let sources: Vec<&str> =
        top_hit.attribution.as_ref().unwrap().matched_spans.iter().map(|s| s.source.as_str()).collect();
    assert!(sources.contains(&"turboquant"));
    assert!(sources.contains(&"sparse"));
    assert!(sources.contains(&"maxsim"));
    assert!(sources.contains(&"filter"));

    // 8. Space mismatch refusal
    let mut wrong_space = test_space(16);
    wrong_space.model.as_mut().unwrap().id = "other/model".into();
    let bad_space_err = client
        .search(SearchRequest {
            index_name: "idx-quant".into(),
            natural_query: String::new(),
            query_ast: None,
            query_vector: vec16(0),
            query_space: Some(wrong_space),
            top_k: 5,
            parent_expansion: None,
            rerank_bundle: String::new(),
        })
        .await
        .expect_err("mismatched model space must fail");
    assert_eq!(bad_space_err.code(), tonic::Code::InvalidArgument);

    // 9. PersistIndex + reload via CreateIndex(load_from_path) + SealIndex + Reindex with alias swap
    let snap_file = std::env::temp_dir().join(format!("turbo_search_snap_{}.bin", std::process::id()));
    let snap_str = snap_file.to_str().unwrap().to_owned();
    let persisted = client
        .persist_index(PersistIndexRequest { index_name: "idx-quant".into(), path: snap_str.clone() })
        .await
        .unwrap()
        .into_inner();
    assert!(persisted.bytes_written > 0);

    let reloaded = client
        .create_index(CreateIndexRequest {
            spec: Some(IndexSpec { index_name: "idx-quant-reloaded".into(), ..Default::default() }),
            load_from_path: snap_str.clone(),
        })
        .await
        .unwrap()
        .into_inner()
        .summary
        .unwrap();
    assert_eq!(reloaded.chunk_count, 3);
    let _ = std::fs::remove_file(&snap_file);

    // Seal `idx-flat` and verify further writes fail
    let sealed = client.seal_index(SealIndexRequest { index_name: "idx-flat".into() }).await.unwrap().into_inner();
    assert!(sealed.summary.unwrap().sealed);
    let write_sealed = client.index_documents(tokio_stream::iter(make_docs("idx-flat"))).await;
    assert_eq!(write_sealed.unwrap_err().code(), tonic::Code::FailedPrecondition);

    // Reindex `idx-flat` -> `idx-v2` (TurboQuant) and swap alias `active`
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

    let aliases = client.list_aliases(ListAliasesRequest {}).await.unwrap().into_inner().aliases;
    assert!(aliases.iter().any(|a| a.alias == "active" && a.index_name == "idx-v2"));
    drop(watch_stream);
    drop(client);
    srv.stop().await;
}
