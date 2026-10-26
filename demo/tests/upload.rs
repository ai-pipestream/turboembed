//! Uploads through the demo to an in-process server: a dictionary whose
//! first 4 MiB chunk reaches the server before the body ends, a corpus learned, and an
//! index batch, each over axum's 2 MB default, answered with the RPC's
//! binary reply.

mod common;
mod uploads;

use std::collections::HashMap;
use std::time::{Duration, Instant};

use common::*;
use prost::Message;
use turbo_kserve::Serve;
use turbo_kserve::proto::search::turbo_search_service_client::TurboSearchServiceClient;
use turbo_kserve::proto::search::*;
use turbo_kserve::proto::vocabulary::*;
use uploads::*;

fn vocabulary(name: &str) -> Serve {
    Serve { admin: true, vocabulary: Some(scratch(name).join("vocabulary")), ..Serve::default() }
}

#[tokio::test]
async fn a_dictionary_streams_to_the_server_as_it_arrives() {
    let s = server(&["tiny-bert-bundle"], vocabulary("upload-import")).await;
    let relay = Relay::start(s.local_addr()).await;
    let d = demo_at(relay.addr, &[IMPORT]).await;
    let words: Vec<u8> = (0..800_000).flat_map(|i| format!("w{i:07}\n").into_bytes()).collect();
    let header = ImportDictionaryHeader {
        format_id: "plain-word-list".into(),
        language: String::new(),
        name: "words".into(),
        byte_size: words.len() as u64,
        sha256: turbo_sha256::sha256(&words).to_hex(),
        has_header_row: false,
    };
    let body = [envelope(0, &header.encode_to_vec()), words.clone()].concat();
    let before = relay.sent();
    let mut p = Put::open(d.local_addr(), IMPORT, "", Some(body.len() as u64)).await;
    let (first, rest) = body.split_at(6 << 20);
    assert!(p.send(first).await);
    // A 4 MiB chunk reaches the server while the rest is still unsent.
    let started = Instant::now();
    while relay.sent() - before < 4 << 20 {
        assert!(started.elapsed() < Duration::from_secs(30), "only {} bytes reached the server", relay.sent());
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(p.send(rest).await);
    let a = p.answer(60).await;
    assert_eq!(a.status, 200, "{a:?}");
    assert_eq!(a.header("content-type"), Some("application/proto"));
    let got = DictionaryArtifactDescriptor::decode(a.body.as_slice()).unwrap();
    assert_eq!((got.entry_count, got.byte_size, got.name.as_str()), (800_000, words.len() as u64, "words"));
}

#[tokio::test]
async fn a_corpus_is_learned_from_a_header_and_document_frames() {
    let s = server(&["tiny-bert-bundle"], vocabulary("upload-learn")).await;
    let d = demo(&s, &[LEARN]).await;
    let header =
        LearnVocabularyHeader { min_term_frequency: 2, lowercase: true, language: "en".into(), ..Default::default() };
    let mut body = envelope(0, &header.encode_to_vec());
    let text = "The New York office opened in Boston. ".repeat(20_000);
    for _ in 0..3 {
        body.extend(envelope(0, &LearnVocabularyDocument { text: text.clone() }.encode_to_vec()));
    }
    assert!(body.len() > 2 << 20);
    let a = put(d.local_addr(), LEARN, "", &body).await;
    assert_eq!(a.status, 200, "{a:?}");
    let got = VocabularyArtifactDescriptor::decode(a.body.as_slice()).unwrap();
    assert_eq!((got.documents, got.language.as_str()), (3, "en"));
    assert!(got.term_count > 0);
}

fn vec16(axis: usize) -> Vec<f32> {
    let mut v = vec![0.1f32; 16];
    v[axis % 16] = 1.0;
    v
}

#[tokio::test]
async fn an_index_batch_over_two_mb_is_indexed() {
    let s = server(&["tiny-bert-bundle"], Serve { admin: true, ..Serve::default() }).await;
    let mut c = TurboSearchServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
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
        chunk_budget_tokens: 32,
        ..Default::default()
    };
    c.create_index(CreateIndexRequest { spec: Some(spec), load_from_path: String::new() }).await.unwrap();
    let d = demo(&s, &[INDEX]).await;
    let text = "Sparse retrieval in Berlin. ".repeat(4_000);
    let mut body = Vec::new();
    for i in 0..30 {
        let doc = IndexDocumentRequest {
            index_name: "idx".into(),
            doc_id: format!("doc-{i}"),
            text: text.clone(),
            chunks: vec![ChunkInput {
                chunk_id: format!("doc-{i}#0"),
                level_kind: "sentence".into(),
                span_end: text.len() as u32,
                dense_vector: vec16(i),
                attributes: HashMap::new(),
                ..Default::default()
            }],
            ..Default::default()
        };
        body.extend(envelope(0, &doc.encode_to_vec()));
    }
    assert!(body.len() > 2 << 20);
    let a = put(d.local_addr(), INDEX, "", &body).await;
    assert_eq!(a.status, 200, "{a:?}");
    let got = IndexDocumentsResponse::decode(a.body.as_slice()).unwrap();
    assert_eq!((got.index_name.as_str(), got.indexed_documents), ("idx", 30));
}
