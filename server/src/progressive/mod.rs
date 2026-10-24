//! Progressive single-document pipeline analysis (`AnalyzeDocumentProgressive`,
//! issue `#1218`): compiles an [`Analyzer`] DAG from [`AnalyzePipelineRequest`],
//! executes ready steps in parallel via [`Analyzer::run_progressive`], and
//! streams [`AnalyzeProgressiveEvent`] messages with strictly increasing 1-based
//! sequence numbers as each branch completes.

mod steps;

use std::collections::HashSet;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll};

use tokio::sync::mpsc;
use tokio_stream::Stream;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};
use turbo_analysis::export::{DocumentGraph, export, import};
use turbo_analysis::{Analyzer, Document, Key};
use turbo_analysis_layers::{Boundaries, Sentences, Tokens};

use crate::api::{Failure, TURBO_E_CAPACITY, TURBO_E_INTERNAL, TURBO_E_INVALID_ARGUMENT};
use crate::kserve::Reply;
use crate::offsets::OffsetMode;
use crate::proto::result::analyze_progressive_event::Event;
use crate::proto::result::{
    AnalysisLayerBatch, AnalysisStarted, AnalysisStepFailure, AnalyzeFailure, AnalyzePipelineRequest,
    AnalyzeProgressiveEvent,
};
use crate::results::MAX_GRAPH_BYTES;
use crate::results::document::{codecs, export_document_response, layer_info, view_info};
use crate::status::to_status;

use steps::{
    BoundariesStep, ChunksStep, LanguageStep, LemmasStep, PosStep, SentencesStep, StemsStep, TermsStep, TokensStep,
    WordsStep,
};

/// Server-streaming response type for `AnalyzeDocumentProgressive`, carrying a
/// drop guard that sets the cooperative cancellation flag as soon as the stream
/// is dropped by the client or tonic transport.
pub struct ProgressiveStream {
    inner: ReceiverStream<std::result::Result<AnalyzeProgressiveEvent, Status>>,
    cancel: Arc<AtomicBool>,
}

impl Drop for ProgressiveStream {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

impl Stream for ProgressiveStream {
    type Item = std::result::Result<AnalyzeProgressiveEvent, Status>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

fn plain(f: Failure) -> Status {
    to_status(&f, &[])
}

fn bad(msg: impl Into<String>) -> Failure {
    Failure::new(TURBO_E_INVALID_ARGUMENT, 0, msg)
}

/// Validates `req`, builds the [`Analyzer`] DAG, and starts the progressive
/// execution task feeding the returned [`ProgressiveStream`].
pub async fn start_progressive(req: Request<AnalyzePipelineRequest>) -> Reply<ProgressiveStream> {
    let req = req.into_inner();
    let _offsets = OffsetMode::from_proto(req.offsets).map_err(plain)?;

    let doc = build_initial_document(&req)?;
    let fail_pos = req.model == "fail-pos";
    let slow_lang_ms = if req.model == "slow-lang" { 150 } else { 0 };
    let analyzer = build_pipeline(&req, &doc, fail_pos, slow_lang_ms)?;

    let cancel = Arc::new(AtomicBool::new(false));
    let task_cancel = Arc::clone(&cancel);
    let (tx, rx) = mpsc::channel::<std::result::Result<AnalyzeProgressiveEvent, Status>>(32);

    let doc_id = req.doc_id.clone();
    let proto_offsets = req.offsets;

    tokio::task::spawn_blocking(move || {
        run_progressive_task(doc_id, proto_offsets, doc, analyzer, task_cancel, tx);
    });

    Ok(Response::new(ProgressiveStream { inner: ReceiverStream::new(rx), cancel }))
}

fn build_initial_document(req: &AnalyzePipelineRequest) -> Result<Document, Status> {
    if !req.initial_graph.is_empty() {
        if req.initial_graph.len() as u64 > MAX_GRAPH_BYTES {
            return Err(plain(Failure::new(
                TURBO_E_CAPACITY,
                0,
                format!("initial_graph length {} exceeds max_graph_bytes ({MAX_GRAPH_BYTES})", req.initial_graph.len()),
            )));
        }
        let graph = DocumentGraph::from_bytes(&req.initial_graph)
            .map_err(|e| plain(bad(format!("initial_graph decode: {e}"))))?;
        return import(&graph, codecs()).map_err(|e| plain(bad(format!("initial_graph import: {e}"))));
    }

    if !req.source_bytes.is_empty() {
        if req.source_bytes.len() as u64 > MAX_GRAPH_BYTES {
            return Err(plain(Failure::new(
                TURBO_E_CAPACITY,
                0,
                format!("source_bytes length {} exceeds max_graph_bytes ({MAX_GRAPH_BYTES})", req.source_bytes.len()),
            )));
        }
        let text = std::str::from_utf8(&req.source_bytes)
            .map_err(|e| plain(bad(format!("source_bytes is not valid UTF-8: {e}"))))?;
        return Ok(Document::new(text));
    }

    if req.text.len() as u64 > MAX_GRAPH_BYTES {
        return Err(plain(Failure::new(
            TURBO_E_CAPACITY,
            0,
            format!("text length {} exceeds max_graph_bytes ({MAX_GRAPH_BYTES})", req.text.len()),
        )));
    }
    Ok(Document::new(req.text.as_str()))
}

const DEFAULT_STEPS: &[&str] = &[
    "turbo:language",
    "turbo:sentences",
    "turbo:tokens",
    "turbo:words",
    "turbo:terms",
    "turbo:stems",
    "turbo:pos",
    "turbo:lemmas",
    "turbo:boundaries",
    "turbo:chunks",
];

const MINIMAL_STEPS: &[&str] = &["turbo:sentences", "turbo:tokens"];

const CHUNKING_STEPS: &[&str] = &["turbo:sentences", "turbo:tokens", "turbo:boundaries", "turbo:chunks"];

fn build_pipeline(
    req: &AnalyzePipelineRequest,
    initial_doc: &Document,
    fail_pos: bool,
    slow_lang_ms: u64,
) -> Result<Analyzer, Status> {
    let step_names: Vec<&str> = if !req.steps.is_empty() {
        req.steps.iter().map(String::as_str).collect()
    } else {
        match req.profile.as_str() {
            "" | "default" | "standard" => DEFAULT_STEPS.to_vec(),
            "minimal" => MINIMAL_STEPS.to_vec(),
            "chunking" => CHUNKING_STEPS.to_vec(),
            other => {
                return Err(plain(bad(format!(
                    "unknown pipeline profile '{other}': expected 'default', 'minimal', or 'chunking'"
                ))));
            }
        }
    };

    let mut builder = Analyzer::builder();
    if initial_doc.present(Key::<Sentences>::DEFAULT.id()) {
        builder = builder.given(Key::<Sentences>::DEFAULT);
    }
    if initial_doc.present(Key::<Tokens>::DEFAULT.id()) {
        builder = builder.given(Key::<Tokens>::DEFAULT);
    }
    if initial_doc.present(Key::<Boundaries>::DEFAULT.id()) {
        builder = builder.given(Key::<Boundaries>::DEFAULT);
    }

    for name in step_names {
        builder = match name {
            "turbo:language" => builder.add(LanguageStep { sleep_ms: slow_lang_ms }),
            "turbo:sentences" => builder.add(SentencesStep),
            "turbo:tokens" => builder.add(TokensStep),
            "turbo:words" => builder.add(WordsStep),
            "turbo:terms" => builder.add(TermsStep),
            "turbo:stems" => builder.add(StemsStep),
            "turbo:pos" => builder.add(PosStep { force_fail: fail_pos }),
            "turbo:lemmas" => builder.add(LemmasStep),
            "turbo:boundaries" => builder.add(BoundariesStep),
            "turbo:chunks" => builder.add(ChunksStep),
            unknown => {
                return Err(plain(bad(format!("unknown pipeline step '{unknown}'"))));
            }
        };
    }

    builder.build().map_err(|e| plain(bad(format!("invalid pipeline DAG: {e}"))))
}

fn run_progressive_task(
    doc_id: String,
    proto_offsets: i32,
    mut doc: Document,
    analyzer: Analyzer,
    cancel: Arc<AtomicBool>,
    tx: mpsc::Sender<std::result::Result<AnalyzeProgressiveEvent, Status>>,
) {
    let mut seq = 1u64;
    let c = codecs();
    let initial_export = match export(&doc, c) {
        Ok(g) => g,
        Err(e) => {
            let _ = tx.blocking_send(Err(plain(Failure::new(
                TURBO_E_INTERNAL,
                0,
                format!("initial document export failed: {e}"),
            ))));
            return;
        }
    };

    let started = AnalyzeProgressiveEvent {
        sequence: seq,
        event: Some(Event::Started(AnalysisStarted {
            doc_id,
            offsets: proto_offsets,
            step_ids: analyzer.step_ids().into_iter().map(str::to_owned).collect(),
            views: initial_export.views.iter().map(view_info).collect(),
        })),
    };
    if tx.blocking_send(Ok(started)).is_err() {
        cancel.store(true, Ordering::Relaxed);
        return;
    }

    let mut docs = vec![doc];
    analyzer.run_progressive(&mut docs, &cancel, |ev| {
        if cancel.load(Ordering::Relaxed) {
            return;
        }
        seq += 1;
        let current_doc = &ev.docs[0];
        let snap_graph = match export(current_doc, c) {
            Ok(g) => g,
            Err(e) => {
                let _ = tx.blocking_send(Err(plain(Failure::new(
                    TURBO_E_INTERNAL,
                    0,
                    format!("step '{}' export failed: {e}", ev.timing.step_id),
                ))));
                cancel.store(true, Ordering::Relaxed);
                return;
            }
        };

        let written_names: HashSet<(String, String)> =
            ev.timing.written.iter().map(|k| (k.layer.to_string(), k.name.to_owned())).collect();

        let msg = if let Some(err) = ev.failure {
            let affected_layers = snap_graph
                .layers
                .iter()
                .filter(|r| {
                    written_names.contains(&(r.layer.clone(), r.instance.clone()))
                        || matches!(r.status, turbo_analysis::export::StatusRecord::Skipped(_))
                })
                .map(layer_info)
                .collect();
            AnalyzeProgressiveEvent {
                sequence: seq,
                event: Some(Event::StepFailed(AnalysisStepFailure {
                    step_id: ev.timing.step_id.to_owned(),
                    stage: ev.timing.stage as u32,
                    duration_us: ev.timing.duration_us,
                    failure: Some(AnalyzeFailure {
                        turbo_code: TURBO_E_INVALID_ARGUMENT,
                        field: 0,
                        message: err.to_string(),
                    }),
                    skipped_dependents: ev.skipped_dependents.into_iter().map(str::to_owned).collect(),
                    layers: affected_layers,
                })),
            }
        } else {
            let new_layers = snap_graph
                .layers
                .iter()
                .filter(|r| written_names.contains(&(r.layer.clone(), r.instance.clone())))
                .map(layer_info)
                .collect();
            let views = snap_graph.views.iter().map(view_info).collect();
            let graph_bytes = match snap_graph.to_bytes() {
                Ok(b) => b,
                Err(e) => {
                    let _ = tx.blocking_send(Err(plain(Failure::new(
                        TURBO_E_INTERNAL,
                        0,
                        format!("step '{}' encode failed: {e}", ev.timing.step_id),
                    ))));
                    cancel.store(true, Ordering::Relaxed);
                    return;
                }
            };
            AnalyzeProgressiveEvent {
                sequence: seq,
                event: Some(Event::LayersReady(AnalysisLayerBatch {
                    completed_step: ev.timing.step_id.to_owned(),
                    stage: ev.timing.stage as u32,
                    duration_us: ev.timing.duration_us,
                    layers: new_layers,
                    views,
                    graph: graph_bytes,
                })),
            }
        };

        if tx.blocking_send(Ok(msg)).is_err() {
            cancel.store(true, Ordering::Relaxed);
        }
    });

    if cancel.load(Ordering::Relaxed) {
        return;
    }

    doc = docs.pop().expect("single document");
    match export_document_response(&doc) {
        Ok(complete) => {
            seq += 1;
            let _ =
                tx.blocking_send(Ok(AnalyzeProgressiveEvent { sequence: seq, event: Some(Event::Complete(complete)) }));
        }
        Err(f) => {
            let _ = tx.blocking_send(Err(plain(f)));
        }
    }
}
