//! Integration tests for named, versioned analysis profiles (`#1302`) and the
//! end-to-end `normalize -> tokenize -> subwords -> chunker -> embed`
//! progressive pipeline with bi-directional cross-view offset tracking (`#1325`).

mod common;

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use common::*;
use serde_json::{Value, json};
use tokio_stream::StreamExt as _;
use tonic::Code;
use turbo::bundle::sha256_hex;
use turbo_analysis::export::{DocumentGraph, import};
use turbo_analysis::{CodePointOffset, Ix, Key, Layer, Utf16Offset, View, ViewId};
use turbo_analysis_layers::{Chunks, Entities, Levels, Sentences, SparseTerms, Structure, Subwords, Tokens, Vectors};
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::progressive::{DEFAULT_V1, MINIMAL_V1, RETRIEVAL_V1};
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::result::analyze_progressive_event::Event;
use turbo_kserve::proto::result::*;
use turbo_kserve::results::document::codecs;
use turbo_kserve::{Serve, Server};

struct TempBundle(PathBuf);

impl Drop for TempBundle {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn next_temp_dir(tag: &str) -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("turbo-kserve-1325-{}-{tag}-{id}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    dir
}

fn splade_bundle(tag: &str) -> TempBundle {
    let dir = next_temp_dir(tag);
    let src_dir = tiny();
    for f in ["tokenizer.json", "reference/reference.safetensors"] {
        fs::create_dir_all(dir.join(f).parent().unwrap()).unwrap();
        fs::copy(src_dir.join(f), dir.join(f)).unwrap();
    }
    let src = fs::read(src_dir.join("weights/model.safetensors")).unwrap();
    let n = u64::from_le_bytes(src[..8].try_into().unwrap()) as usize;
    let mut header: serde_json::Map<String, Value> = serde_json::from_slice(&src[8..8 + n]).unwrap();
    let mut data = src[8 + n..].to_vec();
    let lin = |start: f32, end: f32, count: usize| {
        (0..count)
            .flat_map(|i| {
                let t = i as f32 / (count - 1) as f32;
                (start + (end - start) * t).to_le_bytes()
            })
            .collect::<Vec<u8>>()
    };
    let tensors = [
        ("mlm.transform.weight", vec![32, 32], lin(-0.2, 0.3, 1024)),
        ("mlm.transform.bias", vec![32], lin(-0.1, 0.1, 32)),
        ("mlm.transform.LayerNorm.weight", vec![32], lin(0.8, 1.2, 32)),
        ("mlm.transform.LayerNorm.bias", vec![32], lin(-0.05, 0.05, 32)),
        ("mlm.decoder.bias", vec![30522], lin(-0.06, 0.06, 30522)),
    ];
    for (name, shape, bytes) in &tensors {
        let at = data.len();
        data.extend_from_slice(bytes);
        header.insert((*name).into(), json!({ "dtype": "F32", "shape": shape, "data_offsets": [at, data.len()] }));
    }
    let mut h = serde_json::to_vec(&header).unwrap();
    while !h.len().is_multiple_of(8) {
        h.push(b' ');
    }
    let mut weights = (h.len() as u64).to_le_bytes().to_vec();
    weights.extend(h);
    weights.extend(data);
    fs::create_dir_all(dir.join("weights")).unwrap();
    fs::write(dir.join("weights/model.safetensors"), &weights).unwrap();

    let mut m: Value = serde_json::from_slice(&fs::read(src_dir.join("manifest.json")).unwrap()).unwrap();
    m["embed"]["sparse"] = json!({ "pooling": "SPARSE_POOLING_MAXLOG" });
    let tn = m["artifacts"][0]["tensor_names"].as_object_mut().unwrap();
    tn.insert("mlm_transform_weight".into(), json!("mlm.transform.weight"));
    tn.insert("mlm_transform_bias".into(), json!("mlm.transform.bias"));
    tn.insert("mlm_transform_ln_weight".into(), json!("mlm.transform.LayerNorm.weight"));
    tn.insert("mlm_transform_ln_bias".into(), json!("mlm.transform.LayerNorm.bias"));
    tn.insert("mlm_decoder_bias".into(), json!("mlm.decoder.bias"));
    let w = m["files"].as_array_mut().unwrap().iter_mut().find(|f| f["path"] == "weights/model.safetensors").unwrap();
    w["size"] = json!(weights.len());
    w["sha256"] = json!(sha256_hex(&weights));
    fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&m).unwrap()).unwrap();
    TempBundle(dir)
}

async fn serve_models(models: Vec<ModelConfig>) -> (Server, AnalysisServiceClient<tonic::transport::Channel>) {
    let server = Server::start("127.0.0.1:0".parse().unwrap(), models, MAX, Serve { admin: true, ..Serve::default() })
        .await
        .unwrap();
    server.load().await.unwrap();
    let client = AnalysisServiceClient::connect(format!("http://{}", server.local_addr()))
        .await
        .unwrap()
        .max_decoding_message_size(MAX)
        .max_encoding_message_size(MAX);
    (server, client)
}

#[tokio::test]
async fn capabilities_lists_versioned_profiles_with_steps_layers_and_bundles() {
    let (server, mut client) = serve().await;
    let caps = client.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
    assert_eq!(caps.profiles.len(), 3);

    for (proto, spec) in caps.profiles.iter().zip([DEFAULT_V1, RETRIEVAL_V1, MINIMAL_V1]) {
        assert_eq!(proto.name, spec.name);
        assert_eq!(proto.version, spec.version);
        assert_eq!(proto.steps, spec.steps);
        assert_eq!(proto.layers, spec.layers);
        assert_eq!(proto.required_bundles, spec.required_bundles);
    }

    server.stop().await;
}

#[tokio::test]
async fn each_named_profile_runs_from_single_request_and_produces_exact_listed_layers() {
    let splade = splade_bundle("profiles");
    let models = vec![ModelConfig {
        bundle: splade.0.to_str().unwrap().to_string(),
        name: Some("splade-embed".into()),
        device: Device::Index(0),
        precision: 0,
        max_batch: 8,
        max_seq: 64,
        sessions: 1,
    }];
    let (server, mut client) = serve_models(models).await;
    let text = "# European Report\n\nOur team in Paris and Berlin evaluated neural retrieval models. They published thirty-two benchmarks across London.\n";

    for spec in [MINIMAL_V1, DEFAULT_V1, RETRIEVAL_V1] {
        let req = AnalyzePipelineRequest {
            doc_id: format!("doc-{}", spec.name),
            text: text.into(),
            source_bytes: Vec::new(),
            offsets: OffsetMode::Utf8 as i32,
            profile: spec.name.into(),
            steps: Vec::new(),
            model: String::new(),
            initial_graph: Vec::new(),
        };
        let mut stream = client.analyze_document_progressive(req).await.unwrap().into_inner();
        let mut events = Vec::new();
        while let Some(ev) = stream.next().await {
            events.push(ev.unwrap());
        }

        // Started + N steps + Complete.
        assert_eq!(events.len(), spec.steps.len() + 2, "profile {}", spec.name);
        for (i, ev) in events.iter().enumerate() {
            assert_eq!(ev.sequence, (i + 1) as u64);
        }

        let Some(Event::Complete(complete)) = &events.last().unwrap().event else {
            panic!("expected Complete event for profile {}", spec.name);
        };
        let produced_layers: Vec<&str> = complete.layers.iter().map(|l| l.layer.as_str()).collect();
        assert_eq!(produced_layers, spec.layers, "profile {} layer set mismatch", spec.name);
        for l in &complete.layers {
            assert_eq!(l.status, LayerWireStatus::Ok as i32, "layer {} failed in {}", l.layer, spec.name);
        }

        let graph = DocumentGraph::from_bytes(&complete.graph).unwrap();
        let doc = import(&graph, codecs()).unwrap();
        assert!(doc.get::<Structure>().is_some());
        assert!(doc.get::<Sentences>().is_some());
        assert!(doc.get::<Tokens>().is_some());

        if spec.name != "minimal-v1" {
            let chunks = doc.get::<Chunks>().unwrap();
            let vecs = doc.get::<Vectors<Chunks>>().unwrap();
            assert_eq!(vecs.len(), chunks.len());
            assert_eq!(vecs.dim, 32);
            assert!(doc.get::<Entities>().is_some());
            assert!(doc.get::<Subwords>().is_some());
            assert!(doc.get::<Levels>().is_some());
        }
        if spec.name == "retrieval-v1" {
            let sparse = doc.get::<SparseTerms<View>>().unwrap();
            assert_eq!(sparse.len(), 1);
            assert!(!sparse.term.is_empty());
        }
    }

    server.stop().await;
}

#[tokio::test]
async fn missing_bundle_for_profile_is_refused_naming_the_missing_bundle() {
    // 1. Server with zero loaded models refuses `default-v1` naming `TASK_EMBED`.
    let (empty_server, mut empty_client) = serve_models(Vec::new()).await;
    let err = empty_client
        .analyze_document_progressive(AnalyzePipelineRequest {
            doc_id: "no-embed".into(),
            text: "Hello world.".into(),
            source_bytes: Vec::new(),
            offsets: OffsetMode::Utf8 as i32,
            profile: "default-v1".into(),
            steps: Vec::new(),
            model: String::new(),
            initial_graph: Vec::new(),
        })
        .await
        .unwrap_err();
    assert_eq!(err.code(), Code::NotFound);
    assert!(err.message().contains("TASK_EMBED"), "expected 'TASK_EMBED' in refusal: {}", err.message());
    empty_server.stop().await;

    // 2. Server with `tiny-bert-bundle` (no MLM head) refuses `retrieval-v1` naming `embed.sparse`.
    let (dense_server, mut dense_client) = serve().await;
    let err = dense_client
        .analyze_document_progressive(AnalyzePipelineRequest {
            doc_id: "no-sparse".into(),
            text: "Hello world.".into(),
            source_bytes: Vec::new(),
            offsets: OffsetMode::Utf8 as i32,
            profile: "retrieval-v1".into(),
            steps: Vec::new(),
            model: String::new(),
            initial_graph: Vec::new(),
        })
        .await
        .unwrap_err();
    assert_eq!(err.code(), Code::NotFound);
    assert!(err.message().contains("embed.sparse"), "expected 'embed.sparse' in refusal: {}", err.message());

    // 3. Requesting an unloaded model name with `default-v1` names both the model and `TASK_EMBED`.
    let err = dense_client
        .analyze_document_progressive(AnalyzePipelineRequest {
            doc_id: "missing-named-model".into(),
            text: "Hello world.".into(),
            source_bytes: Vec::new(),
            offsets: OffsetMode::Utf8 as i32,
            profile: "default-v1".into(),
            steps: Vec::new(),
            model: "nonexistent-embed-bundle".into(),
            initial_graph: Vec::new(),
        })
        .await
        .unwrap_err();
    assert_eq!(err.code(), Code::NotFound);
    assert!(err.message().contains("nonexistent-embed-bundle"), "got: {}", err.message());
    assert!(err.message().contains("TASK_EMBED"), "got: {}", err.message());

    dense_server.stop().await;
}

#[tokio::test]
async fn end_to_end_normalize_tokenize_subwords_chunk_embed_projects_bidirectional_source_offsets() {
    let splade = splade_bundle("e2e-norm");
    let models = vec![ModelConfig {
        bundle: splade.0.to_str().unwrap().to_string(),
        name: Some("splade-embed".into()),
        device: Device::Index(0),
        precision: 0,
        max_batch: 8,
        max_seq: 64,
        sessions: 1,
    }];
    let (server, mut client) = serve_models(models).await;

    // Contains multi-byte UTF-8, surrogate-pair emoji (U+1F680), NFKC ligature (U+FB01 -> "fi"),
    // zero-width space (U+200B), and collapsed whitespace runs.
    let source_text = "# Caf\u{00E9} \u{1F680} Report\n\nThe \u{FB01}rst\u{200B}   flight from Paris to Tokyo landed safely.   Second sentence with na\u{00EF}ve r\u{00E9}sum\u{00E9} data.\n";

    let steps = vec![
        "turbo-structure:blocks".into(),
        "turbo:normalize".into(),
        "turbo-structure:project".into(),
        "turbo-sentdetect:rules".into(),
        "turbo-tokenize:uax29-untyped".into(),
        "turbo-geo:names".into(),
        "turbo:subwords".into(),
        "turbo:structure-boundaries".into(),
        "turbo:chunker".into(),
        "turbo:level-builder".into(),
        "turbo:sentence-vectors".into(),
        "turbo:chunk-vectors".into(),
        "turbo:sparse".into(),
    ];

    let req = AnalyzePipelineRequest {
        doc_id: "doc-norm-e2e".into(),
        text: source_text.into(),
        source_bytes: Vec::new(),
        offsets: OffsetMode::Utf8 as i32,
        profile: String::new(),
        steps,
        model: "splade-embed".into(),
        initial_graph: Vec::new(),
    };

    let mut stream = client.analyze_document_progressive(req).await.unwrap().into_inner();
    let mut events = Vec::new();
    while let Some(ev) = stream.next().await {
        events.push(ev.unwrap());
    }

    let Some(Event::Complete(complete)) = &events.last().unwrap().event else {
        panic!("expected Complete event");
    };
    assert_eq!(complete.views.len(), 2, "expected SOURCE view and normalized child view");

    let graph = DocumentGraph::from_bytes(&complete.graph).unwrap();
    let doc = import(&graph, codecs()).unwrap();
    let norm_view = doc.get_key(&Key::<View>::named("normalized")).unwrap();
    assert_eq!(norm_view.parent(), Some(ViewId::SOURCE));
    assert!(
        norm_view.text().contains("first flight"),
        "NFKC + Invisible + Whitespace normalized: {:?}",
        norm_view.text()
    );

    let sents = doc.get::<Sentences>().unwrap();
    let toks = doc.get::<Tokens>().unwrap();
    let subs = doc.get::<Subwords>().unwrap();
    let ents = doc.get::<Entities>().unwrap();
    let chunks = doc.get::<Chunks>().unwrap();
    let levels = doc.get::<Levels>().unwrap();
    let sent_vecs = doc.get::<Vectors<Sentences>>().unwrap();
    let chunk_vecs = doc.get::<Vectors<Chunks>>().unwrap();
    let sparse = doc.get::<SparseTerms<View>>().unwrap();

    assert_eq!(sents.spans.view(), ViewId::SOURCE);
    assert_eq!(toks.spans.view(), ViewId::SOURCE);
    assert_eq!(subs.spans.view(), ViewId::SOURCE);
    assert_eq!(ents.spans.view(), ViewId::SOURCE);
    assert_eq!(chunks.spans.view(), ViewId::SOURCE);
    assert_eq!(levels.spans.view(), ViewId::SOURCE);
    assert_eq!(sent_vecs.len(), sents.len());
    assert_eq!(chunk_vecs.len(), chunks.len());
    assert_eq!(sparse.len(), 1);

    // Verify bi-directional projection between ViewId::SOURCE and normalized child view.
    let norm_toks = doc.project_layer(toks, norm_view.id()).unwrap();
    let roundtrip_toks = doc.project_layer(&norm_toks, ViewId::SOURCE).unwrap();
    assert_eq!(roundtrip_toks.len(), toks.len());
    for i in 0..toks.len() {
        let ix = Ix::new(i as u32);
        let sp = toks.spans.get(ix);
        assert_eq!(roundtrip_toks.spans.get(ix), sp);
        assert!(source_text.get(sp.range()).is_some());
    }

    // Verify InspectDocumentGraph and RoundTripTypedDocument across UTF-8, UTF-16, and CodePoint modes.
    let offsets_table = doc.offsets(ViewId::SOURCE);
    for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
        let typed = client
            .inspect_document_graph(InspectDocumentGraphRequest {
                doc_id: "doc-norm-e2e".into(),
                graph: complete.graph.clone().into(),
                view: 0,
                offsets: mode as i32,
            })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(typed.raw_text, source_text);

        for layer in &typed.layers {
            for ann in &layer.annotations {
                for sp in &ann.spans {
                    let (b_start, b_end) = match mode {
                        OffsetMode::Utf8 => (sp.start, sp.end),
                        OffsetMode::Utf16 => (
                            offsets_table.from_utf16(source_text, Utf16Offset(sp.start)).unwrap(),
                            offsets_table.from_utf16(source_text, Utf16Offset(sp.end)).unwrap(),
                        ),
                        OffsetMode::CodePoint => (
                            offsets_table.from_code_point(source_text, CodePointOffset(sp.start)).unwrap(),
                            offsets_table.from_code_point(source_text, CodePointOffset(sp.end)).unwrap(),
                        ),
                    };
                    assert!(source_text.get(b_start as usize..b_end as usize).is_some());
                    match mode {
                        OffsetMode::Utf8 => {}
                        OffsetMode::Utf16 => {
                            assert_eq!(offsets_table.to_utf16(source_text, b_start).0, sp.start);
                            assert_eq!(offsets_table.to_utf16(source_text, b_end).0, sp.end);
                        }
                        OffsetMode::CodePoint => {
                            assert_eq!(offsets_table.to_code_point(source_text, b_start).0, sp.start);
                            assert_eq!(offsets_table.to_code_point(source_text, b_end).0, sp.end);
                        }
                    }
                }
            }
        }

        let echoed = client.round_trip_typed_document(typed.clone()).await.unwrap().into_inner();
        assert_eq!(echoed, typed);
    }

    server.stop().await;
}
