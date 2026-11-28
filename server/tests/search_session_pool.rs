//! Regression tests for session-pool queueing, deadline cancellation, and
//! rerank batch chunking across `Search`, `IndexDocuments`, and
//! `ProjectChunks` (`#2609`).

mod common;

use std::collections::HashMap;
use std::time::Duration;

use tonic::metadata::MetadataValue;
use tonic::{Code, Request};
use turbo_analysis::export::export;
use turbo_analysis::{Document, Ix, Key, ModelRef, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{Sentences, Vectors};
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::result::{CapabilitiesRequest, OffsetMode, ProjectChunksRequest};
use turbo_kserve::proto::search::query_node::Clause;
use turbo_kserve::proto::search::turbo_search_service_client::TurboSearchServiceClient;
use turbo_kserve::proto::search::*;
use turbo_kserve::results::document::codecs;
use turbo_kserve::{PRIORITY_HEADER, Serve, Server};

const NO_ANSWER: Duration = Duration::from_secs(60);

async fn wire_deadline_status<M: prost::Message>(server: &Server, path: &str, msg: M, timeout_hdr: &str) -> String {
    let socket = tokio::net::TcpStream::connect(server.local_addr()).await.unwrap();
    let (mut sender, conn) = h2::client::handshake(socket).await.unwrap();
    let conn = tokio::spawn(conn);
    let request = tonic::codegen::http::Request::builder()
        .method("POST")
        .uri(format!("http://{}{path}", server.local_addr()))
        .header("content-type", "application/grpc")
        .header("te", "trailers")
        .header("grpc-timeout", timeout_hdr)
        .body(())
        .unwrap();
    let encoded = msg.encode_to_vec();
    let mut frame = vec![0];
    frame.extend_from_slice(&u32::try_from(encoded.len()).unwrap().to_be_bytes());
    frame.extend_from_slice(&encoded);
    let (response, mut body) = sender.send_request(request, false).unwrap();
    body.send_data(frame.into(), true).unwrap();
    let resp = tokio::time::timeout(Duration::from_secs(5), response).await.unwrap().unwrap();
    let mut status = resp.headers().get("grpc-status").map(|v| v.to_str().unwrap().to_owned());
    let mut resp_body = resp.into_body();
    if status.is_none()
        && let Ok(Some(trailers)) = resp_body.trailers().await
    {
        status = trailers.get("grpc-status").map(|v| v.to_str().unwrap().to_owned());
    }
    conn.abort();
    status.expect("server returned grpc-status")
}

fn make_rerank_bundle_with_max_batch(max_batch: u32) -> common::BundleDir {
    let dir = common::make_rerank_bundle("PAIR_BERT");
    let manifest_path = dir.0.join("manifest.json");
    let mut m: serde_json::Value = serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    m["rerank"]["max_batch"] = serde_json::Value::from(max_batch);
    std::fs::write(&manifest_path, serde_json::to_vec_pretty(&m).unwrap()).unwrap();
    dir
}

fn build_sentence_vector_graph(model: ModelRef, dim: u32) -> Vec<u8> {
    let sentences = ["Alpha beta gamma. ", "Delta epsilon zeta."];
    let text = sentences.concat();
    let mut doc = Document::new(text.as_str());
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut at = 0u32;
    for s in sentences {
        spans.push(Span::new(at, at + s.trim_end().len() as u32));
        at += s.len() as u32;
    }
    doc.add(&Key::DEFAULT, Sentences { spans }, Provenance::by("test")).unwrap();
    let mut data = vec![0.0f32; dim as usize * 2];
    data[0] = 1.0;
    data[1] = 1.0;
    let inv = (0.5f32).sqrt();
    data[dim as usize] = inv;
    data[dim as usize + 1] = -inv;
    let vecs = Vectors::<Sentences>::new(dim, data, Vec::<Ix<Sentences>>::new(), model);
    doc.add(&Key::DEFAULT, vecs, Provenance::by("test")).unwrap();
    export(&doc, codecs()).unwrap().to_bytes().unwrap()
}

fn model_cfg(path: &std::path::Path, name: &str, max_batch: u32) -> ModelConfig {
    let device = {
        #[cfg(feature = "cuda")]
        {
            Device::Select
        }
        #[cfg(not(feature = "cuda"))]
        {
            Device::Index(common::cpu())
        }
    };
    ModelConfig {
        bundle: path.to_str().unwrap().into(),
        name: Some(name.into()),
        device,
        precision: 0,
        max_batch,
        max_seq: 0,
        sessions: 1,
    }
}

#[tokio::test]
async fn search_rerank_chunks_candidates_when_top_k_exceeds_max_batch() {
    let rerank_dir = make_rerank_bundle_with_max_batch(8);
    let models = vec![model_cfg(&rerank_dir.0, "rerank-batch2", 2), model_cfg(&rerank_dir.0, "rerank-batch8", 8)];
    let srv = Server::start("127.0.0.1:0".parse().unwrap(), models, MAX, Serve::default()).await.unwrap();
    srv.load().await.unwrap();

    let mut client = TurboSearchServiceClient::connect(format!("http://{}", srv.local_addr())).await.unwrap();
    let space = SpaceSpec {
        model: Some(ModelIdentity {
            id: "turbo/test-rerank-space".into(),
            revision: "v1".into(),
            sha256: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".into(),
        }),
        dim: 8,
        normalization: NormalizationMode::Unit as i32,
    };
    client
        .create_index(CreateIndexRequest {
            spec: Some(IndexSpec {
                index_name: "rerank-idx".into(),
                space: Some(space),
                metric: DistanceMetric::Cosine as i32,
                storage: VectorStorageKind::FlatF32 as i32,
                chunk_budget_tokens: 64,
                ..Default::default()
            }),
            load_from_path: String::new(),
        })
        .await
        .unwrap();

    // Five matching documents (d1 and d3 share identical text to verify deterministic tie-breaking).
    let passages = [
        ("d0", "alpha network latency routing protocol packet"),
        ("d1", "alpha exact tie candidate passage text"),
        ("d2", "alpha consensus quorum state machine log"),
        ("d3", "alpha exact tie candidate passage text"),
        ("d4", "alpha alpha distributed storage checkpoint wal"),
    ];
    let docs: Vec<IndexDocumentRequest> = passages
        .iter()
        .map(|(id, text)| IndexDocumentRequest {
            index_name: "rerank-idx".into(),
            doc_id: (*id).into(),
            text: (*text).into(),
            ..Default::default()
        })
        .collect();
    let ing = client.index_documents(tokio_stream::iter(docs)).await.unwrap().into_inner();
    assert_eq!(ing.indexed_documents, 5);
    assert_eq!(ing.indexed_chunks, 5);

    let search_with = |bundle: &str| SearchRequest {
        index_name: "rerank-idx".into(),
        natural_query: "alpha routing storage".into(),
        top_k: 5,
        rerank_bundle: bundle.into(),
        ..Default::default()
    };

    let ref_resp = client.search(search_with("rerank-batch8")).await.unwrap().into_inner();
    assert_eq!(ref_resp.hits.len(), 5);

    let chunked_resp = client.search(search_with("rerank-batch2")).await.unwrap().into_inner();
    assert_eq!(chunked_resp.hits.len(), 5);

    for (got, want) in chunked_resp.hits.iter().zip(&ref_resp.hits) {
        assert_eq!(got.chunk_id, want.chunk_id);
        assert_eq!(got.doc_id, want.doc_id);
        assert!((got.score - want.score).abs() < 1e-5, "{} vs {}", got.score, want.score);
        let got_rr = got
            .attribution
            .as_ref()
            .unwrap()
            .matched_spans
            .iter()
            .find(|s| s.source == "rerank")
            .expect("rerank attribution span");
        assert_eq!(got_rr.label, "rerank-batch2");
        assert_eq!(got_rr.score.to_bits(), got.score.to_bits());
    }
    let heat_ids = |r: &SearchResponse| -> Vec<String> { r.document_heat.iter().map(|h| h.doc_id.clone()).collect() };
    assert_eq!(heat_ids(&chunked_resp), heat_ids(&ref_resp));

    srv.stop().await;
}

#[tokio::test]
async fn search_index_and_project_chunks_queue_on_single_session_and_honor_deadline() {
    let rerank_dir = make_rerank_bundle_with_max_batch(4);
    let models = vec![model_cfg(&common::tiny(), "tiny-embed", 2), model_cfg(&rerank_dir.0, "tiny-rerank", 2)];
    let srv = Server::start("127.0.0.1:0".parse().unwrap(), models, MAX, Serve::default()).await.unwrap();
    srv.load().await.unwrap();
    let endpoint = format!("http://{}", srv.local_addr());
    let shared_ch = tonic::transport::Endpoint::from_shared(endpoint).unwrap().connect().await.unwrap();

    let mut analysis = AnalysisServiceClient::new(shared_ch.clone());
    let caps = analysis.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
    let embed_cap = caps.bundles.iter().find(|b| b.name == "tiny-embed").unwrap();
    let mproto = embed_cap.model.as_ref().unwrap();
    let dim = embed_cap.widths[0];
    let embed_space = SpaceSpec {
        model: Some(ModelIdentity {
            id: mproto.id.clone(),
            revision: mproto.revision.clone(),
            sha256: mproto.sha256.clone(),
        }),
        dim,
        normalization: NormalizationMode::Unit as i32,
    };
    let reindex_space = embed_space.clone();
    let graph_bytes = build_sentence_vector_graph(
        ModelRef { id: mproto.id.clone(), revision: mproto.revision.clone(), sha256: mproto.sha256.clone() },
        dim,
    );

    let mut search_client = TurboSearchServiceClient::new(shared_ch.clone());
    search_client
        .create_index(CreateIndexRequest {
            spec: Some(IndexSpec {
                index_name: "pool-idx".into(),
                space: Some(embed_space),
                metric: DistanceMetric::Cosine as i32,
                storage: VectorStorageKind::FlatF32 as i32,
                embed_bundle: "tiny-embed".into(),
                chunk_budget_tokens: 64,
                ..Default::default()
            }),
            load_from_path: String::new(),
        })
        .await
        .unwrap();

    let seed_docs: Vec<IndexDocumentRequest> = (0..3)
        .map(|i| IndexDocumentRequest {
            index_name: "pool-idx".into(),
            doc_id: format!("seed-{i}"),
            text: format!("alpha beta seed document {i} with distinct retrieval terms"),
            attributes: HashMap::new(),
            ..Default::default()
        })
        .collect();
    search_client.index_documents(tokio_stream::iter(seed_docs)).await.unwrap();

    let make_semantic_search = || SearchRequest {
        index_name: "pool-idx".into(),
        natural_query: "alpha beta".into(),
        query_ast: Some(QueryNode {
            clause: Some(Clause::Semantic(SemanticClause {
                query_text: "alpha beta retrieval".into(),
                top_k: 2,
                ..Default::default()
            })),
        }),
        top_k: 2,
        rerank_bundle: "tiny-rerank".into(),
        ..Default::default()
    };

    // 1. Hold tiny-embed's single session while launching concurrent Search,
    //    IndexDocuments, and ProjectChunks requests; all must queue rather than
    //    failing with BUSY, then succeed once released.
    let held_embed = srv.hold("tiny-embed").unwrap();
    let mut tasks = Vec::new();
    for i in 0..3 {
        let mut sc = TurboSearchServiceClient::new(shared_ch.clone());
        let req = make_semantic_search();
        tasks.push(tokio::spawn(async move {
            let r = sc.search(req).await.map_err(|e| format!("search {i}: {e}"))?.into_inner();
            assert!(!r.hits.is_empty());
            Ok::<_, String>(())
        }));

        let mut ic = TurboSearchServiceClient::new(shared_ch.clone());
        let doc = IndexDocumentRequest {
            index_name: "pool-idx".into(),
            doc_id: format!("queued-doc-{i}"),
            text: format!("alpha queued document {i}. Second chunk sentence here."),
            ..Default::default()
        };
        tasks.push(tokio::spawn(async move {
            let r = ic.index_documents(tokio_stream::iter(vec![doc])).await.map_err(|e| format!("index {i}: {e}"))?;
            assert_eq!(r.into_inner().indexed_documents, 1);
            Ok::<_, String>(())
        }));

        let mut ac = AnalysisServiceClient::new(shared_ch.clone());
        let g = graph_bytes.clone();
        tasks.push(tokio::spawn(async move {
            let r = ac
                .project_chunks(ProjectChunksRequest {
                    graph: g,
                    offsets: OffsetMode::Utf8 as i32,
                    pca_dims: 2,
                    query: Some(format!("alpha query {i}")),
                    ..Default::default()
                })
                .await
                .map_err(|e| format!("project_chunks {i}: {e}"))?
                .into_inner();
            assert_eq!(r.query.expect("placed query").model_name, "tiny-embed");
            Ok::<_, String>(())
        }));
    }

    common::wait_for("all 9 requests queued on tiny-embed", NO_ANSWER, || {
        let (iq, bq) = srv.priority_queue_depths("tiny-embed").unwrap();
        iq == 6 && bq == 3
    })
    .await;

    drop(held_embed);
    let mut errs = Vec::new();
    for t in tasks {
        if let Err(e) = common::within("queued task completion", NO_ANSWER, t).await.unwrap() {
            errs.push(e);
        }
    }
    assert!(errs.is_empty(), "queued embed tasks failed: {errs:?}");

    // 2. Hold tiny-rerank's single session while launching concurrent reranked
    //    Search calls (top_k = 5 > max_batch = 2); all queue and succeed.
    let held_rerank = srv.hold("tiny-rerank").unwrap();
    let mut rerank_tasks = Vec::new();
    for i in 0..3 {
        let mut sc = TurboSearchServiceClient::new(shared_ch.clone());
        rerank_tasks.push(tokio::spawn(async move {
            let r = sc
                .search(SearchRequest {
                    index_name: "pool-idx".into(),
                    natural_query: "alpha".into(),
                    top_k: 5,
                    rerank_bundle: "tiny-rerank".into(),
                    ..Default::default()
                })
                .await
                .map_err(|e| format!("rerank search {i}: {e}"))?
                .into_inner();
            assert_eq!(r.hits.len(), 5);
            Ok::<_, String>(())
        }));
    }

    common::wait_for("all 3 rerank searches queued on tiny-rerank", NO_ANSWER, || {
        let (iq, bq) = srv.priority_queue_depths("tiny-rerank").unwrap();
        iq == 3 && bq == 0
    })
    .await;

    drop(held_rerank);
    for t in rerank_tasks {
        common::within("queued rerank completion", NO_ANSWER, t).await.unwrap().unwrap();
    }

    // 3. When grpc-timeout expires while waiting for a held session slot, each
    //    RPC returns server grpc-status 4 (DEADLINE_EXCEEDED) and leaks no session.
    let held_embed = srv.hold("tiny-embed").unwrap();

    assert_eq!(
        wire_deadline_status(&srv, "/turbo.search.v1.TurboSearchService/Search", make_semantic_search(), "20m").await,
        "4"
    );

    let timeout_doc = IndexDocumentRequest {
        index_name: "pool-idx".into(),
        doc_id: "timeout-doc".into(),
        text: "alpha timeout document".into(),
        ..Default::default()
    };
    assert_eq!(
        wire_deadline_status(&srv, "/turbo.search.v1.TurboSearchService/IndexDocuments", timeout_doc, "20m").await,
        "4"
    );

    let proj_timeout = ProjectChunksRequest {
        graph: graph_bytes,
        offsets: OffsetMode::Utf8 as i32,
        pca_dims: 2,
        query: Some("alpha timeout query".into()),
        ..Default::default()
    };
    assert_eq!(
        wire_deadline_status(&srv, "/turbo.result.v1.AnalysisService/ProjectChunks", proj_timeout, "20m").await,
        "4"
    );

    drop(held_embed);

    let held_rerank = srv.hold("tiny-rerank").unwrap();
    let rerank_timeout = SearchRequest {
        index_name: "pool-idx".into(),
        natural_query: "alpha".into(),
        top_k: 4,
        rerank_bundle: "tiny-rerank".into(),
        ..Default::default()
    };
    assert_eq!(
        wire_deadline_status(&srv, "/turbo.search.v1.TurboSearchService/Search", rerank_timeout, "20m").await,
        "4"
    );
    drop(held_rerank);

    // Invalid x-turbo-priority metadata is rejected with INVALID_ARGUMENT.
    let mut bad_prio = Request::new(make_semantic_search());
    bad_prio.metadata_mut().insert(PRIORITY_HEADER, MetadataValue::from_static("urgent"));
    assert_eq!(search_client.search(bad_prio).await.unwrap_err().code(), Code::InvalidArgument);

    // 4. Multi-clause SemanticClause Join (> max_batch = 2 distinct query_texts)
    //    and Reindex with smaller chunk budget (> max_batch chunks) both succeed.
    let multi_q = SearchRequest {
        index_name: "pool-idx".into(),
        query_ast: Some(QueryNode {
            clause: Some(Clause::Join(JoinClause {
                op: JoinOperator::Or as i32,
                fusion: FusionStrategy::ReciprocalRank as i32,
                rrf_k: 60.0,
                children: ["alpha query one", "beta query two", "retrieval query three"]
                    .into_iter()
                    .map(|q| QueryNode {
                        clause: Some(Clause::Semantic(SemanticClause {
                            query_text: q.into(),
                            top_k: 3,
                            ..Default::default()
                        })),
                    })
                    .collect(),
            })),
        }),
        top_k: 5,
        rerank_bundle: "tiny-rerank".into(),
        ..Default::default()
    };
    assert_eq!(search_client.search(multi_q).await.unwrap().into_inner().hits.len(), 5);

    let held_for_reindex = srv.hold("tiny-embed").unwrap();
    let mut rc = TurboSearchServiceClient::new(shared_ch.clone());
    let reindex_req = ReindexRequest {
        source_index: "pool-idx".into(),
        target_spec: Some(IndexSpec {
            index_name: "pool-idx-v2".into(),
            space: Some(reindex_space),
            metric: DistanceMetric::Cosine as i32,
            storage: VectorStorageKind::FlatF32 as i32,
            embed_bundle: "tiny-embed".into(),
            chunk_budget_tokens: 8,
            ..Default::default()
        }),
        swap_alias: String::new(),
    };
    let reindex_task = tokio::spawn(async move { rc.reindex(reindex_req).await });
    common::wait_for("reindex queued on tiny-embed", NO_ANSWER, || {
        srv.priority_queue_depths("tiny-embed").unwrap() == (0, 1)
    })
    .await;
    drop(held_for_reindex);
    let reindexed = common::within("reindex completion", NO_ANSWER, reindex_task).await.unwrap().unwrap().into_inner();
    assert!(reindexed.summary.unwrap().chunk_count >= 6);

    // Sessions remain healthy and un-leaked after deadline cancellations.
    drop(srv.hold("tiny-embed").expect("tiny-embed session released"));
    drop(srv.hold("tiny-rerank").expect("tiny-rerank session released"));
    assert!(!search_client.search(make_semantic_search()).await.unwrap().into_inner().hits.is_empty());

    srv.stop().await;
}
