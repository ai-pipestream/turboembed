//! Server-streaming Connect calls through the demo: each server message in
//! its envelope as the server sent it, then the end-of-stream envelope
//! with the call's status and the server's turbo-code and turbo-field.

mod common;

use std::collections::HashMap;
use std::time::Duration;

use common::*;
use turbo_kserve::Serve;
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::result::analyze_progressive_event::Event;
use turbo_kserve::proto::result::{AnalyzePipelineRequest, AnalyzeProgressiveEvent};
use turbo_kserve::proto::search::turbo_search_service_client::TurboSearchServiceClient;
use turbo_kserve::proto::search::*;
use turbo_kserve::proto::vocabulary::{DownloadVocabularyRequest, VocabularyArtifactChunk};

const NAME: &str = "tiny-bert-bundle";

#[tokio::test]
async fn analyze_document_progressive_streams_the_servers_events() {
    let s = server(&[NAME], Serve::default()).await;
    let d = demo(&s, &[]).await;
    let req = AnalyzePipelineRequest {
        doc_id: "doc".into(),
        text: "# Report\n\nOur team in Paris evaluated retrieval. They published benchmarks.\n".into(),
        profile: "minimal-v1".into(),
        ..Default::default()
    };
    let path = "/turbo.result.v1.AnalysisService/AnalyzeDocumentProgressive";
    let (got, end) = stream(&d, path, &req).await.frames::<AnalyzeProgressiveEvent>().await;
    assert_eq!(end, "{}");
    let mut grpc = AnalysisServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    let mut want = Vec::new();
    let mut st = grpc.analyze_document_progressive(req).await.unwrap().into_inner();
    while let Some(e) = st.message().await.unwrap() {
        want.push(e);
    }
    assert!(got.len() >= 2);
    assert_eq!(got.len(), want.len());
    assert!(matches!(got[0].event, Some(Event::Started(_))));
    let (Some(Event::Complete(g)), Some(Event::Complete(w))) =
        (&got.last().unwrap().event, &want.last().unwrap().event)
    else {
        panic!("no Complete event")
    };
    assert_eq!(g.graph, w.graph);

    // The server's refusal of the stream: a profile with no model to run.
    let req = AnalyzePipelineRequest { profile: "no-such-profile".into(), ..req_text() };
    let (msgs, end) = stream(&d, path, &req).await.frames::<AnalyzeProgressiveEvent>().await;
    assert!(msgs.is_empty());
    assert!(end.starts_with(r#"{"error":{"code":""#), "{end}");
}

fn req_text() -> AnalyzePipelineRequest {
    AnalyzePipelineRequest { doc_id: "d".into(), text: "Hello world.".into(), ..Default::default() }
}

fn vec16(axis: usize) -> Vec<f32> {
    let mut v = vec![0.1f32; 16];
    v[axis % 16] = 1.0;
    v
}

#[tokio::test]
async fn watch_collection_streams_events_as_documents_are_indexed() {
    let s = server(&[NAME], Serve { admin: true, ..Serve::default() }).await;
    let d = demo(&s, &[]).await;
    let mut c = TurboSearchServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    let col = SetCollectionRequest {
        collection_name: "corpus".into(),
        baseline_terms: vec!["neural".into()],
        oov_alert_threshold: 0.1,
    };
    c.set_collection(col).await.unwrap();
    let spec = IndexSpec {
        index_name: "idx".into(),
        space: Some(SpaceSpec {
            model: Some(ModelIdentity {
                id: "turbo/test-axis-16".into(),
                revision: "v1".into(),
                sha256: "0".repeat(64),
            }),
            dim: 16,
            normalization: NormalizationMode::None as i32,
        }),
        metric: DistanceMetric::Cosine as i32,
        storage: VectorStorageKind::FlatF32 as i32,
        collection_name: "corpus".into(),
        chunk_budget_tokens: 32,
        ..Default::default()
    };
    c.create_index(CreateIndexRequest { spec: Some(spec), load_from_path: String::new() }).await.unwrap();

    let path = "/turbo.search.v1.TurboSearchService/WatchCollection";
    let mut watch =
        stream(&d, path, &WatchCollectionRequest { collection_name: "corpus".into(), min_oov_ratio: 0.0 }).await;
    // The watch is open at the server before the document goes in.
    tokio::time::sleep(Duration::from_millis(200)).await;
    let text = "Sparse retrieval pipeline in Berlin.";
    let doc = IndexDocumentRequest {
        index_name: "idx".into(),
        doc_id: "doc-1".into(),
        text: text.into(),
        chunks: vec![ChunkInput {
            chunk_id: "doc-1#0".into(),
            level_kind: "sentence".into(),
            span_end: text.len() as u32,
            dense_vector: vec16(0),
            sparse_terms: vec![SparseEntryProto {
                term_id: 7,
                term_text: "berlin".into(),
                weight: 1.0,
                ..Default::default()
            }],
            attributes: HashMap::new(),
            ..Default::default()
        }],
        ..Default::default()
    };
    c.index_documents(futures_util::stream::iter([doc])).await.unwrap();
    let (flags, payload) = tokio::time::timeout(Duration::from_secs(5), watch.frame()).await.unwrap().unwrap();
    assert_eq!(flags, 0);
    let ev = <CollectionEvent as prost::Message>::decode(payload.as_slice()).unwrap();
    assert_eq!((ev.collection_name.as_str(), ev.source_index.as_str()), ("corpus", "idx"));

    // The server's refusal of a watch, with its turbo metadata.
    let (msgs, end) = stream(&d, path, &WatchCollectionRequest::default()).await.frames::<CollectionEvent>().await;
    assert!(msgs.is_empty());
    assert!(end.starts_with(r#"{"error":{"code":"invalid_argument","message":"#), "{end}");
    assert!(end.contains(r#""metadata":{"turbo-code":["256"],"turbo-field":["#), "{end}");
}

#[tokio::test]
async fn download_vocabulary_refusal_ends_the_stream_with_the_servers_metadata() {
    let d = scratch("vocab");
    let s = server(&[NAME], Serve { vocabulary: Some(d.join("vocabulary")), ..Serve::default() }).await;
    let demo = demo(&s, &[]).await;
    let path = "/turbo.vocabulary.v1.TurboVocabularyService/DownloadVocabulary";
    let req = DownloadVocabularyRequest { artifact_id: "vocab-none".into() };
    let (msgs, end) = stream(&demo, path, &req).await.frames::<VocabularyArtifactChunk>().await;
    assert!(msgs.is_empty());
    assert!(end.starts_with(r#"{"error":{"code":""#) && end.contains(r#""metadata":{"turbo-code":[""#), "{end}");
}
