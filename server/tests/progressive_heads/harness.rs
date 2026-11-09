//! A server over sealed tiny head bundles and the progressive request
//! round trip, for the head step tests.

#![allow(dead_code)]

use std::path::Path;

use tokio_stream::StreamExt as _;
use turbo_analysis::Document;
use turbo_analysis::export::{DocumentGraph, import};
use turbo_analysis_embed::{Device as EncoderDevice, EncoderOptions};
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::result::analyze_progressive_event::Event;
use turbo_kserve::proto::result::*;
use turbo_kserve::results::document::codecs;
use turbo_kserve::{Serve, Server};

use crate::common::{Analysis, cpu};

/// Pieces a row, special tokens included, for the multi-window tests.
pub const MAX_SEQ: u32 = 12;

pub fn model(bundle: &Path, name: &str, max_seq: u32) -> ModelConfig {
    ModelConfig {
        bundle: bundle.to_str().unwrap().to_string(),
        name: Some(name.into()),
        device: Device::Index(cpu()),
        precision: 0,
        max_batch: 0,
        max_seq,
        sessions: 1,
    }
}

/// The encoder options a model served with `max_seq` runs on.
pub fn encoder_options(max_seq: u32) -> EncoderOptions {
    EncoderOptions { device: EncoderDevice::Index(cpu()), max_seq, ..EncoderOptions::default() }
}

pub async fn serve_models(models: Vec<ModelConfig>) -> (Server, Analysis) {
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

pub fn request(text: &str, profile: &str, steps: &[&str]) -> AnalyzePipelineRequest {
    AnalyzePipelineRequest {
        binding: None,
        doc_id: "heads".into(),
        text: text.into(),
        source_bytes: Vec::new(),
        offsets: OffsetMode::Utf8 as i32,
        profile: profile.into(),
        steps: steps.iter().map(|s| s.to_string()).collect(),
        model: String::new(),
        initial_graph: Vec::new(),
        structure_seed: None,
        pii_output: None,
        operation_key: String::new(),
        pii_redaction: None,
        vocabulary: None,
    }
}

pub async fn run(client: &mut Analysis, req: AnalyzePipelineRequest) -> Vec<AnalyzeProgressiveEvent> {
    let mut stream = client.analyze_document_progressive(req).await.unwrap().into_inner();
    let mut events = Vec::new();
    while let Some(ev) = stream.next().await {
        events.push(ev.unwrap());
    }
    events
}

pub fn started(events: &[AnalyzeProgressiveEvent]) -> Vec<String> {
    let Some(Event::Started(started)) = &events[0].event else { panic!("no Started event") };
    let mut ids = started.step_ids.clone();
    ids.sort_unstable();
    ids
}

/// The Complete event, every layer of it written.
pub fn complete(events: &[AnalyzeProgressiveEvent]) -> &DocumentGraphResponse {
    let Some(Event::Complete(complete)) = &events[events.len() - 2].event else { panic!("no Complete event") };
    for l in &complete.layers {
        assert_eq!(l.status, LayerWireStatus::Ok as i32, "layer {} failed: {:?}", l.layer, l.error);
    }
    complete
}

pub fn document(complete: &DocumentGraphResponse) -> Document {
    import(&DocumentGraph::from_bytes(&complete.graph).unwrap(), codecs()).unwrap()
}

/// The (layer, instance) pairs the Complete event lists.
pub fn layers(complete: &DocumentGraphResponse) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> =
        complete.layers.iter().map(|l| (l.layer.clone(), l.instance.clone())).collect();
    out.sort_unstable();
    out
}
