//! The same document analyzed by a running `turbo-kserve` on the CPU.
//!
//! The client imports docling's structure (the one step a converter's
//! output needs on the client), sends it as `initial_graph` to
//! `AnalyzeDocumentProgressive`, and the server runs the rest: normalizing,
//! sentences, tokens, the model's subwords, place names, structural
//! boundaries, the chunker, levels and a vector per chunk. The finished
//! graph then goes back to the server for its heatmaps
//! (`InspectDocumentWindow`) and its chunk projection (`ProjectChunks`).

use tonic::transport::Channel;
use turbo_analysis::export::{DocumentGraph, export};
use turbo_analysis::{Document, Key, Provenance};
use turbo_analysis_layers::Structure;
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::result::analyze_progressive_event::Event;
use turbo_kserve::proto::result::*;
use turbo_kserve::results::document::codecs;

use crate::docling::{self, Docling};

/// The name `turbo-kserve` serves the tiny test bundle under.
pub const MODEL: &str = "tiny-bert-bundle";

/// The server pipeline, in step ids `AnalyzeDocumentProgressive` takes.
pub const STEPS: [&str; 9] = [
    "turbo:normalize:whitespace",
    "turbo-sentdetect:rules",
    "turbo-tokenize:uax29-untyped",
    "turbo:subwords",
    "turbo-geo:names",
    "turbo:structure-boundaries",
    "turbo:chunker",
    "turbo:levels",
    "turbo:chunk-vectors",
];

/// How many equal-width buckets each heatmap lane is binned into.
pub const HEAT_BUCKETS: u32 = 16;

/// The largest answer the client decodes, the server's default budget.
const MAX_MESSAGE: usize = 64 << 20;

/// The docling document read and its structure as `initial_graph` bytes:
/// the source text and one `Structure` layer, the default instance, over
/// it.
pub fn seed(json: &str) -> Result<(Docling, Vec<u8>), String> {
    let (d, structure) = docling::parse(json)?;
    let mut doc = Document::new(d.text.as_str());
    doc.add(&Key::<Structure>::DEFAULT, structure, Provenance::by("docling:import")).map_err(|e| e.to_string())?;
    let graph = export(&doc, codecs()).map_err(|e| e.to_string())?;
    let bytes = graph.to_bytes().map_err(|e| e.to_string())?;
    Ok((d, bytes))
}

/// What the server answered.
pub struct Remote {
    /// The steps in the order they finished.
    pub completed: Vec<String>,
    /// The analyzed document's graph.
    pub graph: Vec<u8>,
    /// The source view's window with every lane binned into `HEAT_BUCKETS`.
    pub window: DocumentWindowResponse,
    /// The chunk projection, a point per chunk with its level.
    pub chunks: ProjectChunksResponse,
}

/// A client of the server at `addr` (`http://host:port`).
pub async fn connect(addr: &str) -> Result<AnalysisServiceClient<Channel>, String> {
    let c = AnalysisServiceClient::connect(addr.to_owned()).await.map_err(|e| format!("{addr}: {e}"))?;
    Ok(c.max_decoding_message_size(MAX_MESSAGE).max_encoding_message_size(MAX_MESSAGE))
}

fn status(what: &str, s: tonic::Status) -> String {
    let code = s.metadata().get("turbo-code").and_then(|v| v.to_str().ok()).unwrap_or("-");
    format!("{what}: {:?} (turbo-code {code}): {}", s.code(), s.message())
}

/// Runs the server pipeline over `seed` and asks for the heat and chunks
/// of the result, every span in `offsets`.
pub async fn run(
    client: &mut AnalysisServiceClient<Channel>,
    seed: Vec<u8>,
    offsets: OffsetMode,
) -> Result<Remote, String> {
    let req = AnalyzePipelineRequest {
        doc_id: "report".into(),
        offsets: offsets as i32,
        steps: STEPS.iter().map(|s| (*s).to_owned()).collect(),
        model: MODEL.into(),
        initial_graph: seed,
        ..Default::default()
    };
    let mut events = client.analyze_document_progressive(req).await.map_err(|s| status("analyze", s))?.into_inner();
    let mut completed = Vec::new();
    let mut graph = None;
    while let Some(ev) = events.message().await.map_err(|s| status("analyze", s))? {
        match ev.event {
            Some(Event::LayersReady(b)) => completed.push(b.completed_step),
            Some(Event::StepFailed(f)) => {
                let why = f.failure.map(|x| x.message).unwrap_or_default();
                return Err(format!("step {} failed: {why}", f.step_id));
            }
            Some(Event::Complete(c)) => graph = Some(c.graph.to_vec()),
            Some(Event::Started(_)) | None => {}
        }
    }
    let graph = graph.ok_or("the stream ended without the document")?;
    let window = DocumentWindowRequest {
        graph: graph.clone(),
        view: 0,
        offsets: offsets as i32,
        heatmap_buckets: HEAT_BUCKETS,
        ..Default::default()
    };
    let window = client.inspect_document_window(window).await.map_err(|s| status("window", s))?.into_inner();
    let chunks = ProjectChunksRequest { graph: graph.clone(), offsets: offsets as i32, ..Default::default() };
    let chunks = client.project_chunks(chunks).await.map_err(|s| status("chunks", s))?.into_inner();
    Ok(Remote { completed, graph, window, chunks })
}

/// The answered graph, decoded.
pub fn document(remote: &Remote) -> Result<Document, String> {
    let graph = DocumentGraph::from_bytes(&remote.graph).map_err(|e| e.to_string())?;
    turbo_analysis::export::import(&graph, codecs()).map_err(|e| e.to_string())
}
