//! Progressive single-document pipeline analysis (`AnalyzeDocumentProgressive`,
//! issue `#1218`): compiles an [`Analyzer`] DAG from [`AnalyzePipelineRequest`],
//! executes ready steps in parallel via [`Analyzer::run_progressive`], and
//! streams [`AnalyzeProgressiveEvent`] messages with strictly increasing 1-based
//! sequence numbers as each branch completes.

mod normalize;
pub mod profiles;
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
use turbo_analysis::{Analyzer, Document};

use crate::api::{
    Failure, TURBO_E_CAPACITY, TURBO_E_INTERNAL, TURBO_E_INVALID_ARGUMENT, TURBO_E_INVALID_STATE, TURBO_E_PANIC,
};
use crate::deadline::{Deadline, deadline_status};
use crate::kserve::Reply;
use crate::metrics::{DocGuard, StreamGuard};
use crate::offsets::OffsetMode;
use crate::proto::result::analyze_progressive_event::Event;
use crate::proto::result::{
    AnalysisLayerBatch, AnalysisStarted, AnalysisStepFailure, AnalyzeFailure, AnalyzePipelineRequest,
    AnalyzeProgressiveEvent,
};
use crate::results::MAX_GRAPH_BYTES;
use crate::results::document::{codecs, export_document_response, layer_info, view_info};
use crate::state::State;
use crate::status::to_status;

pub use profiles::{
    AnalysisProfile, DEFAULT_V1, MINIMAL_V1, PROFILES, RETRIEVAL_V1, find_profile, profile_capabilities,
};
pub use steps::compile_analyzer;
use steps::{CHUNKING_STEPS, DEFAULT_STEPS, MINIMAL_STEPS, compile_analyzer_with_state};

/// Server-streaming response type for `AnalyzeDocumentProgressive`, carrying a
/// drop guard that sets the cooperative cancellation flag as soon as the stream
/// is dropped by the client or tonic transport.
pub struct ProgressiveStream {
    inner: ReceiverStream<std::result::Result<AnalyzeProgressiveEvent, Status>>,
    cancel: Arc<AtomicBool>,
    _stream_guard: Option<StreamGuard>,
    _doc_guard: Option<DocGuard>,
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

fn step_error_code(err: &turbo_analysis::Error) -> i32 {
    match err {
        turbo_analysis::Error::Annotator { detail, .. } if detail.starts_with("panic:") => TURBO_E_PANIC,
        turbo_analysis::Error::Annotator { .. } => TURBO_E_INTERNAL,
        turbo_analysis::Error::Missing(_)
        | turbo_analysis::Error::Duplicate(_)
        | turbo_analysis::Error::Clash(_)
        | turbo_analysis::Error::Misaligned { .. }
        | turbo_analysis::Error::Unaligned { .. }
        | turbo_analysis::Error::Pipeline(_) => TURBO_E_INVALID_STATE,
        turbo_analysis::Error::Bounds { .. }
        | turbo_analysis::Error::Invalid { .. }
        | turbo_analysis::Error::Offset { .. }
        | turbo_analysis::Error::Source(_) => TURBO_E_INVALID_ARGUMENT,
        turbo_analysis::Error::Skipped(_) => 0,
    }
}

/// Validates `req`, builds the [`Analyzer`] DAG, and starts the progressive
/// execution task feeding the returned [`ProgressiveStream`].
pub(crate) async fn start_progressive(state: &State, req: Request<AnalyzePipelineRequest>) -> Reply<ProgressiveStream> {
    let deadline = Deadline::from_request(&req)?;
    deadline.check()?;
    let req = req.into_inner();
    let offsets = OffsetMode::from_proto(req.offsets).map_err(plain)?;

    let doc = build_initial_document(&req)?;
    let analyzer = build_pipeline(state, &req, &doc)?;

    let cancel = Arc::clone(deadline.cancel_flag());
    let task_cancel = Arc::clone(&cancel);
    let task_deadline = deadline.clone();
    let (tx, rx) = mpsc::channel::<std::result::Result<AnalyzeProgressiveEvent, Status>>(32);

    let doc_id = req.doc_id.clone();
    let proto_offsets = offsets.to_proto();
    let (stream_guard, doc_guard) = (Some(state.metrics.track_stream()), Some(state.metrics.track_docs(1, None)));

    tokio::task::spawn_blocking(move || {
        run_progressive_task(doc_id, proto_offsets, doc, analyzer, task_cancel, task_deadline, tx);
    });

    Ok(Response::new(ProgressiveStream {
        inner: ReceiverStream::new(rx),
        cancel,
        _stream_guard: stream_guard,
        _doc_guard: doc_guard,
    }))
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

fn build_pipeline(state: &State, req: &AnalyzePipelineRequest, initial_doc: &Document) -> Result<Analyzer, Status> {
    let (step_names, profile_name, required_bundles): (Vec<&str>, &str, &[&str]) = if !req.steps.is_empty() {
        (req.steps.iter().map(String::as_str).collect(), req.profile.as_str(), &[])
    } else if let Some(p) = find_profile(req.profile.as_str()) {
        (p.steps.to_vec(), p.name, p.required_bundles)
    } else {
        match req.profile.as_str() {
            "" | "default" | "standard" => (DEFAULT_STEPS.to_vec(), req.profile.as_str(), &[]),
            "minimal" => (MINIMAL_STEPS.to_vec(), req.profile.as_str(), &[]),
            "chunking" => (CHUNKING_STEPS.to_vec(), req.profile.as_str(), &[]),
            other => {
                return Err(plain(bad(format!(
                    "unknown pipeline profile '{other}': expected 'default-v1', 'retrieval-v1', 'minimal-v1', 'default', 'minimal', or 'chunking'"
                ))));
            }
        }
    };

    compile_analyzer_with_state(initial_doc, &step_names, Some(state), &req.model, profile_name, required_bundles)
        .map_err(plain)
}

fn run_progressive_task(
    doc_id: String,
    proto_offsets: i32,
    mut doc: Document,
    analyzer: Analyzer,
    cancel: Arc<AtomicBool>,
    deadline: Deadline,
    tx: mpsc::Sender<std::result::Result<AnalyzeProgressiveEvent, Status>>,
) {
    if deadline.is_expired() {
        let _ = tx.blocking_send(Err(deadline_status()));
        return;
    }
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
                        turbo_code: step_error_code(&err),
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
            if graph_bytes.len() as u64 > MAX_GRAPH_BYTES {
                let _ = tx.blocking_send(Err(plain(Failure::new(
                    TURBO_E_CAPACITY,
                    0,
                    format!(
                        "step '{}' graph snapshot length {} exceeds max_graph_bytes ({MAX_GRAPH_BYTES})",
                        ev.timing.step_id,
                        graph_bytes.len()
                    ),
                ))));
                cancel.store(true, Ordering::Relaxed);
                return;
            }
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

    if deadline.is_expired() {
        let _ = tx.blocking_send(Err(deadline_status()));
        return;
    }
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
