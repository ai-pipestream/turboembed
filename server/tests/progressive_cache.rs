//! Regression and lifecycle tests for progressive annotator caching on
//! served bundles (issue `#2608`):
//! - Warm `AnalyzeDocumentProgressive` calls reuse `Loaded`'s model and
//!   session pool plus cached tokenizer/head state without re-reading the
//!   bundle from disk.
//! - Layer provenance, `ModelRef`, `Execution`, and `receipt.config_digest`
//!   match standalone `open` annotators byte for byte.
//! - Concurrent progressive calls serialize cleanly on a 1-session pool and
//!   leave the session idle afterwards.
//! - `UnloadBundle` drops cached progressive state, and a subsequent
//!   `LoadBundle` serves the replacement bundle's weights and provenance.

mod common;
#[path = "progressive_heads/bundle.rs"]
mod head_bundle;

use std::fs;
use std::path::Path;

use common::{Analysis, cpu};
use serde_json::{Value, json};
use tokio_stream::StreamExt as _;
use tonic::Code;
use turbo::bundle::sha256_hex;
use turbo_analysis::export::{DocumentGraph, export, import};
use turbo_analysis::{Analyzer, Document, Key};
use turbo_analysis_embed::{
    ChunkVectors, Device as EncoderDevice, EncoderOptions, SentenceVectors, SparseAnnotator, SparseOptions,
    SubwordAnnotator, VectorAnnotator, VectorOptions,
};
use turbo_analysis_layers::{Chunks, Entities, PosTags, Sentences, SparseTerms, Structure, Subwords, Tokens, Vectors};
use turbo_chunker::{Budget, Chunker, Costs, StructureBoundaries};
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::progressive::heads::{GroupStep, HeadGroup, HeadStep};
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::result::analyze_progressive_event::Event;
use turbo_kserve::proto::result::*;
use turbo_kserve::results::document::codecs;
use turbo_kserve::{Serve, Server};
use turbo_sat::{SatAnnotator, SatOptions};
use turbo_scratch::Scratch;
use turbo_structure::{StructureAnnotator, Syntax};
use turbo_tokenize::uax29::UntypedWordTokenizer;

const DOC: &str = "# Field Report\n\nAnn met Bob in Paris. The quiet old dog slept by the river.";

const FULL_STEPS: &[&str] = &[
    "turbo-structure:blocks",
    "turbo:sat",
    "turbo-tokenize:uax29-untyped",
    "turbo:subwords",
    "turbo:structure-boundaries",
    "turbo:chunker",
    "turbo:sentence-vectors",
    "turbo:chunk-vectors",
    "turbo:sparse",
    "turbo-tagger:pos",
    "turbo-tagger:entities",
    "turbo-tagger:dependencies",
    "turbo-tagger:lemmas",
];

/// Builds a tiny bundle equipped with SPLADE (`mlm.*`), SaT (`classifier.*`),
/// and the four shared token/dependency/lemma heads (`heads.*`).
fn rich_bundle(tag: &str, model_id: &str, revision: &str, weight_bias: f32) -> Scratch {
    head_bundle::bundle(tag, |m| {
        m["model"]["id"] = json!(model_id);
        m["model"]["revision"] = json!(revision);
        m["embed"]["sparse"] = json!({ "pooling": "SPARSE_POOLING_MAXLOG" });
        m["classifier"] = json!({ "kind": "CLASSIFIER_TOKEN", "labels": ["newline"] });
        let tn = m["artifacts"][0]["tensor_names"].as_object_mut().unwrap();
        tn.insert("mlm_transform_weight".into(), json!("mlm.transform.weight"));
        tn.insert("mlm_transform_bias".into(), json!("mlm.transform.bias"));
        tn.insert("mlm_transform_ln_weight".into(), json!("mlm.transform.LayerNorm.weight"));
        tn.insert("mlm_transform_ln_bias".into(), json!("mlm.transform.LayerNorm.bias"));
        tn.insert("mlm_decoder_bias".into(), json!("mlm.decoder.bias"));
        tn.insert("classifier_weight".into(), json!("classifier.weight"));
        tn.insert("classifier_bias".into(), json!("classifier.bias"));
        let _ = weight_bias;
    })
}

fn append_extra_tensors(dir: &Path, weight_bias: f32) {
    let weights_path = dir.join("weights/model.safetensors");
    let src = fs::read(&weights_path).unwrap();
    let n = u64::from_le_bytes(src[..8].try_into().unwrap()) as usize;
    let mut header: serde_json::Map<String, Value> = serde_json::from_slice(&src[8..8 + n]).unwrap();
    let mut data = src[8 + n..].to_vec();

    let lin = |start: f32, end: f32, count: usize| {
        (0..count)
            .flat_map(|i| {
                let t = if count <= 1 { 0.0 } else { i as f32 / (count - 1) as f32 };
                (start + (end - start) * t + weight_bias).to_le_bytes()
            })
            .collect::<Vec<u8>>()
    };
    let extras = [
        ("mlm.transform.weight", vec![32u64, 32], lin(-0.2, 0.3, 1024)),
        ("mlm.transform.bias", vec![32], lin(-0.1, 0.1, 32)),
        ("mlm.transform.LayerNorm.weight", vec![32], lin(0.8, 1.2, 32)),
        ("mlm.transform.LayerNorm.bias", vec![32], lin(-0.05, 0.05, 32)),
        ("mlm.decoder.bias", vec![30522], lin(-0.06, 0.06, 30522)),
        ("classifier.weight", vec![1, 32], lin(-0.4, 0.4, 32)),
        ("classifier.bias", vec![1], vec![(0.15f32 + weight_bias).to_le_bytes()].into_iter().flatten().collect()),
    ];
    for (name, shape, bytes) in extras {
        let start = data.len();
        data.extend_from_slice(&bytes);
        header.insert(name.into(), json!({ "dtype": "F32", "shape": shape, "data_offsets": [start, data.len()] }));
    }
    let mut head = serde_json::to_vec(&header).unwrap();
    head.resize(head.len().next_multiple_of(8), b' ');
    let mut out = (head.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&head);
    out.extend_from_slice(&data);
    fs::write(&weights_path, &out).unwrap();

    let manifest_path = dir.join("manifest.json");
    let mut m: Value = serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    let w = m["files"].as_array_mut().unwrap().iter_mut().find(|f| f["path"] == "weights/model.safetensors").unwrap();
    w["size"] = json!(out.len());
    w["sha256"] = json!(sha256_hex(&out));
    fs::write(&manifest_path, serde_json::to_vec_pretty(&m).unwrap()).unwrap();
}

fn build_rich_bundle(tag: &str, model_id: &str, revision: &str, weight_bias: f32) -> Scratch {
    let dir = rich_bundle(tag, model_id, revision, weight_bias);
    append_extra_tensors(&dir, weight_bias);
    dir
}

fn req_for(model: &str, text: &str, steps: &[&str]) -> AnalyzePipelineRequest {
    AnalyzePipelineRequest {
        doc_id: "cache-doc".into(),
        text: text.into(),
        offsets: OffsetMode::Utf8 as i32,
        steps: steps.iter().map(|&s| s.to_owned()).collect(),
        model: model.into(),
        ..Default::default()
    }
}

async fn collect_progressive(
    client: &mut Analysis,
    req: AnalyzePipelineRequest,
) -> (DocumentGraphResponse, ProcessingReceipt) {
    let mut stream = client.analyze_document_progressive(req).await.unwrap().into_inner();
    let mut events = Vec::new();
    while let Some(item) = stream.next().await {
        events.push(item.unwrap());
    }
    let Some(Event::Complete(complete)) = events[events.len() - 2].event.clone() else {
        panic!("expected Complete event");
    };
    for layer in &complete.layers {
        assert_eq!(layer.status, LayerWireStatus::Ok as i32, "layer {} failed: {:?}", layer.layer, layer.error);
    }
    let Some(Event::Receipt(receipt)) = events[events.len() - 1].event.clone() else {
        panic!("expected terminal Receipt event");
    };
    (complete, receipt)
}

fn standalone_full_graph(dir: &Path, max_batch: u32, max_seq: u32) -> Vec<u8> {
    let eo = EncoderOptions { device: EncoderDevice::Index(cpu()), max_batch, max_seq, ..EncoderOptions::default() };
    let sat = SatAnnotator::open(dir, &SatOptions { encoder: eo.clone(), ..SatOptions::default() }).unwrap();
    let sub = SubwordAnnotator::open(dir).unwrap();
    let va = std::sync::Arc::new(
        VectorAnnotator::open(dir, &VectorOptions { encoder: eo.clone(), ..VectorOptions::default() }).unwrap(),
    );
    let max = va.width();
    let target = (max / 2).max(1);
    let budget = Budget { max, target, size_weight: target.saturating_mul(2), overhead: va.overhead() };
    let sa = SparseAnnotator::open(dir, &SparseOptions { encoder: eo.clone(), ..SparseOptions::default() }).unwrap();
    let head_steps = [HeadStep::Pos, HeadStep::Entities, HeadStep::Dependencies, HeadStep::Lemmas];
    let group = HeadGroup::open(dir, &head_steps, &eo).unwrap();

    let analyzer = Analyzer::builder()
        .add(StructureAnnotator::new(Syntax::Markdown))
        .add(sat)
        .add(UntypedWordTokenizer::default())
        .add(sub)
        .add(StructureBoundaries { costs: Costs::default() })
        .add(Chunker { budget })
        .add(SentenceVectors(va.clone()))
        .add(ChunkVectors(va))
        .add(sa)
        .add(GroupStep::<PosTags>::new(group.clone()))
        .add(GroupStep::<Entities>::new(group.clone()))
        .add(GroupStep::<turbo_analysis_layers::Dependencies<Tokens>>::new(group.clone()))
        .add(GroupStep::<turbo_analysis_layers::Lemmas>::new(group))
        .build()
        .unwrap();

    let mut doc = Document::new(DOC);
    for r in analyzer.run(std::slice::from_mut(&mut doc)) {
        r.unwrap();
    }
    export(&doc, codecs()).unwrap().to_bytes().unwrap()
}

#[tokio::test]
async fn warm_progressive_reuses_loaded_cache_after_bundle_deletion_and_unloads_cleanly() {
    let dir_v1 = build_rich_bundle("prog-cache-v1", "tiny/prog-v1", "rev-1", 0.0);
    let expected_graph = standalone_full_graph(&dir_v1, 8, 64);

    let cfg = ModelConfig {
        bundle: dir_v1.to_str().unwrap().to_owned(),
        name: Some("prog".into()),
        device: Device::Index(cpu()),
        precision: 0,
        max_batch: 8,
        max_seq: 64,
        sessions: 1,
    };
    let server =
        Server::start("127.0.0.1:0".parse().unwrap(), vec![cfg], MAX, Serve { admin: true, ..Serve::default() })
            .await
            .unwrap();
    server.load().await.unwrap();
    let addr = format!("http://{}", server.local_addr());
    let mut client = AnalysisServiceClient::connect(addr.clone())
        .await
        .unwrap()
        .max_decoding_message_size(MAX)
        .max_encoding_message_size(MAX);

    // Cold call warms ProgressiveCache over Loaded's single pooled session.
    let (cold_complete, cold_receipt) = collect_progressive(&mut client, req_for("prog", DOC, FULL_STEPS)).await;
    assert_eq!(cold_complete.graph, expected_graph, "cached progressive run must match standalone open byte for byte");

    // Delete weights, tokenizer, and manifest on disk so any disk reload fails.
    fs::remove_file(dir_v1.join("weights/model.safetensors")).unwrap();
    fs::remove_file(dir_v1.join("tokenizer.json")).unwrap();
    fs::remove_file(dir_v1.join("manifest.json")).unwrap();
    assert!(!dir_v1.join("weights/model.safetensors").exists());

    // Warm call with the exact same pipeline succeeds from memory and preserves receipt digests.
    let (warm_complete, warm_receipt) = collect_progressive(&mut client, req_for("prog", DOC, FULL_STEPS)).await;
    assert_eq!(warm_complete.graph, cold_complete.graph);
    assert_eq!(warm_receipt.config_digest, cold_receipt.config_digest);
    assert_eq!(warm_receipt.result, cold_receipt.result);

    // Varying HeadStep subsets reuse the single cached WordEncoder and cached GroupHeads
    // without touching disk or leaking cross-request features.
    for subset in [
        &["turbo-sentdetect:rules", "turbo-tokenize:uax29-untyped", "turbo-tagger:pos"][..],
        &["turbo-sentdetect:rules", "turbo-tokenize:uax29-untyped", "turbo-tagger:entities", "turbo-tagger:lemmas"][..],
    ] {
        let (sub_complete, _) =
            collect_progressive(&mut client, req_for("prog", "Alice visited Rome yesterday.", subset)).await;
        let doc = import(&DocumentGraph::from_bytes(&sub_complete.graph).unwrap(), codecs()).unwrap();
        assert!(doc.get::<Tokens>().is_some());
    }

    // Concurrent progressive calls against a 1-session pool queue cleanly without TURBO_E_BUSY
    // and return the session to idle when done.
    let mut handles = Vec::new();
    for i in 0..6 {
        let mut c = client.clone();
        let text = format!("# Doc {i}\n\nAnn met Bob in Paris on trip {i}.");
        handles
            .push(tokio::spawn(async move { collect_progressive(&mut c, req_for("prog", &text, FULL_STEPS)).await }));
    }
    for h in handles {
        let (done, _) = h.await.unwrap();
        let doc = import(&DocumentGraph::from_bytes(&done.graph).unwrap(), codecs()).unwrap();
        assert!(doc.get::<Structure>().is_some());
        assert!(doc.get::<Sentences>().is_some());
        assert!(doc.get::<Subwords>().is_some());
        assert!(doc.get::<Chunks>().is_some());
        assert!(doc.get::<Vectors<Sentences>>().is_some());
        assert!(doc.get::<Vectors<Chunks>>().is_some());
        assert!(doc.get::<SparseTerms<turbo_analysis::View>>().is_some());
        assert!(doc.get::<PosTags>().is_some());
        assert!(doc.get::<Entities>().is_some());
    }
    drop(server.hold("prog").expect("single session returns to idle after concurrent progressive runs"));

    // UnloadBundle drops Served and its ProgressiveCache; subsequent calls fail NotFound.
    client.unload_bundle(UnloadBundleRequest { name: "prog".into() }).await.unwrap();
    let err = client.analyze_document_progressive(req_for("prog", DOC, FULL_STEPS)).await.unwrap_err();
    assert_eq!(err.code(), Code::NotFound);

    // Attempting to reload the deleted directory fails rather than serving stale state.
    let reload_deleted = client
        .load_bundle(LoadBundleRequest {
            name: "prog".into(),
            bundle: dir_v1.to_str().unwrap().to_owned(),
            device: cpu(),
            sessions: 1,
            max_batch: 8,
            max_seq: 64,
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert_ne!(reload_deleted.code(), Code::Ok);

    // Loading a replacement bundle under the same name serves the new model's provenance and weights.
    let dir_v2 = build_rich_bundle("prog-cache-v2", "tiny/prog-v2", "rev-2", 0.25);
    client
        .load_bundle(LoadBundleRequest {
            name: "prog".into(),
            bundle: dir_v2.to_str().unwrap().to_owned(),
            device: cpu(),
            sessions: 1,
            max_batch: 8,
            max_seq: 64,
            ..Default::default()
        })
        .await
        .unwrap();

    let (v2_complete, _) = collect_progressive(&mut client, req_for("prog", DOC, FULL_STEPS)).await;
    assert_ne!(v2_complete.graph, cold_complete.graph, "replacement bundle must not serve old cached annotators");
    let v2_doc = import(&DocumentGraph::from_bytes(&v2_complete.graph).unwrap(), codecs()).unwrap();
    let vec_prov = v2_doc.provenance(Key::<Vectors<Chunks>>::DEFAULT.id()).unwrap();
    let mref = vec_prov.model.as_ref().expect("vector layer records ModelRef");
    assert_eq!(mref.id, "tiny/prog-v2");
    assert_eq!(mref.revision, "rev-2");

    server.stop().await;
}

#[cfg(feature = "cuda")]
#[tokio::test]
async fn cuda_warm_progressive_reuses_gpu_session_after_weights_deleted() {
    let dir = build_rich_bundle("prog-cache-cuda", "tiny/prog-cuda", "rev-gpu", 0.0);
    let cfg = ModelConfig {
        bundle: dir.to_str().unwrap().to_owned(),
        name: Some("prog-gpu".into()),
        device: Device::Select,
        precision: 0,
        max_batch: 8,
        max_seq: 64,
        sessions: 1,
    };
    let server =
        Server::start("127.0.0.1:0".parse().unwrap(), vec![cfg], MAX, Serve { admin: true, ..Serve::default() })
            .await
            .unwrap();
    server.load().await.unwrap();
    let mut client = AnalysisServiceClient::connect(format!("http://{}", server.local_addr()))
        .await
        .unwrap()
        .max_decoding_message_size(MAX)
        .max_encoding_message_size(MAX);

    let (cold, cold_receipt) = collect_progressive(&mut client, req_for("prog-gpu", DOC, FULL_STEPS)).await;
    let cold_doc = import(&DocumentGraph::from_bytes(&cold.graph).unwrap(), codecs()).unwrap();
    let exec = cold_doc
        .provenance(Key::<Vectors<Chunks>>::DEFAULT.id())
        .unwrap()
        .execution
        .as_ref()
        .expect("GPU execution recorded");
    assert_eq!(exec.backend, "cuda");

    fs::remove_file(dir.join("weights/model.safetensors")).unwrap();
    fs::remove_file(dir.join("tokenizer.json")).unwrap();
    fs::remove_file(dir.join("manifest.json")).unwrap();

    for _ in 0..5 {
        let (warm, warm_receipt) = collect_progressive(&mut client, req_for("prog-gpu", DOC, FULL_STEPS)).await;
        assert_eq!(warm.graph, cold.graph);
        assert_eq!(warm_receipt.config_digest, cold_receipt.config_digest);
    }
    drop(server.hold("prog-gpu").expect("GPU session returned to idle pool"));
    server.stop().await;
}
