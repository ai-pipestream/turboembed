//! Progressive single-document pipeline analysis (`AnalyzeDocumentProgressive`,
//! issue `#1218`): compiles an [`Analyzer`] DAG from [`AnalyzePipelineRequest`],
//! executes ready steps in parallel via [`Analyzer::run_progressive`], and
//! streams [`AnalyzeProgressiveEvent`] messages with strictly increasing 1-based
//! sequence numbers as each branch completes.

mod config;
mod normalize;
pub mod profiles;
pub mod seed;
#[cfg(test)]
mod step_error_tests;
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

use crate::api::{Failure, TURBO_E_CAPACITY, TURBO_E_INTERNAL, TURBO_E_INVALID_ARGUMENT};
use crate::deadline::{Deadline, deadline_status};
use crate::kserve::Reply;
use crate::metrics::{DocGuard, StreamGuard};
use crate::offsets::OffsetMode;
use crate::proto::result::analyze_progressive_event::Event;
use crate::proto::result::{
    AnalysisLayerBatch, AnalysisStarted, AnalysisStepFailure, AnalyzeFailure, AnalyzePipelineRequest,
    AnalyzeProgressiveEvent, DocumentGraphResponse, InputBinding, LayerErrorKind,
};
use crate::results::MAX_GRAPH_BYTES;
use crate::results::disclosure::{Disclosure, PiiOutputPolicy, holds_pii, plans_pii};
use crate::results::document::{codecs, export_document_response, layer_info, view_info};
use crate::results::receipt::error::{kind_name, of_error};
use crate::results::receipt::{self, Operation, check_operation_key};
use crate::results::typed::identity::{bound, check_binding, input_identity, result_identity};
use crate::sink::SinkForwarder;
use crate::state::State;
use crate::status::to_status;

use crate::pii_policy;
pub use profiles::{
    ANNOTATE_V1, ANNOTATE_V2, AnalysisProfile, DEFAULT_V1, MINIMAL_V1, PROFILES, RETRIEVAL_V1, find_profile,
    profile_capabilities,
};
pub(crate) use steps::{
    CHUNKING_STEPS, DEFAULT_STEPS, MINIMAL_STEPS, StepOptions, compile_analyzer_with_state, compile_pipeline,
};
pub use steps::{compile_analyzer, compile_analyzer_with_redaction, heads};

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

/// Validates `req`, builds the [`Analyzer`] DAG, and starts the progressive
/// execution task feeding the returned [`ProgressiveStream`]. With `sink`,
/// the complete graph the client was handed is then forwarded to it, after
/// the reply has ended: a full queue drops and counts it.
pub(crate) async fn start_progressive(
    state: &State,
    req: Request<AnalyzePipelineRequest>,
    sink: Option<Arc<SinkForwarder>>,
    pii_policy: PiiOutputPolicy,
) -> Reply<ProgressiveStream> {
    let deadline = Deadline::from_request(&req)?;
    deadline.check()?;
    let req = req.into_inner();
    let offsets = OffsetMode::from_proto(req.offsets).map_err(plain)?;
    check_operation_key(&req.operation_key).map_err(plain)?;
    check_binding(req.binding.as_ref(), "binding").map_err(plain)?;
    let disclosure = Disclosure::decide(req.pii_output.as_ref(), pii_policy).map_err(plain)?;

    let doc = build_initial_document(&req, offsets)?;
    let analyzer = build_pipeline(state, &req, &doc)?;
    let mut op = Operation::start(&req.operation_key);
    op.requested_steps = analyzer.step_ids().into_iter().map(str::to_owned).collect();

    let cancel = Arc::clone(deadline.cancel_flag());
    let task_cancel = Arc::clone(&cancel);
    let task_deadline = deadline.clone();
    let (tx, rx) = mpsc::channel::<std::result::Result<AnalyzeProgressiveEvent, Status>>(32);

    let doc_id = req.doc_id.clone();
    let proto_offsets = offsets.to_proto();
    let (stream_guard, doc_guard) = (Some(state.metrics.track_stream()), Some(state.metrics.track_docs(1, None)));

    let (sink_doc_id, binding) = (doc_id.clone(), req.binding.clone());
    tokio::task::spawn_blocking(move || {
        // The task owns the stream's one sender, so the reply has ended
        // when it returns.
        let answer = Answer { doc_id, offsets: proto_offsets, disclosure, binding: binding.clone() };
        let delivered = run_progressive_task(answer, doc, analyzer, task_cancel, task_deadline, tx, op);
        if let (Some(sink), Some((exported, doc))) = (sink, delivered) {
            // A refusal is counted in the forwarder's stats.
            let _ = sink.offer_exported(&sink_doc_id, exported, &doc, binding.as_ref());
        }
    });

    Ok(Response::new(ProgressiveStream {
        inner: ReceiverStream::new(rx),
        cancel,
        _stream_guard: stream_guard,
        _doc_guard: doc_guard,
    }))
}

fn build_initial_document(req: &AnalyzePipelineRequest, offsets: OffsetMode) -> Result<Document, Status> {
    if let Some(seed) = &req.structure_seed {
        if !req.initial_graph.is_empty() || !req.source_bytes.is_empty() || !req.text.is_empty() {
            return Err(plain(bad(
                "structure_seed carries the text: leave text, source_bytes and initial_graph empty",
            )));
        }
        if seed.text.len() as u64 > MAX_GRAPH_BYTES {
            return Err(plain(Failure::new(
                TURBO_E_CAPACITY,
                0,
                format!("structure_seed text length {} exceeds max_graph_bytes ({MAX_GRAPH_BYTES})", seed.text.len()),
            )));
        }
        return seed::seed_document(seed, offsets).map_err(plain);
    }

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
                    "unknown pipeline profile '{other}': expected 'default-v1', 'retrieval-v1', 'minimal-v1', 'annotate-v1', 'annotate-v2', 'default', 'minimal', or 'chunking'"
                ))));
            }
        }
    };

    let redaction = match &req.pii_redaction {
        None => None,
        Some(sel) => Some(
            pii_policy::select(sel, &state.pii_policies, &*state.pii_resolver)
                .map_err(|e| plain(bad(e.to_string())))?,
        ),
    };
    let open_span = config::open_span(req.open_span.as_ref()).map_err(plain)?;
    let relations = config::relations(req.relations.as_ref());
    compile_analyzer_with_state(
        initial_doc,
        &step_names,
        Some(state),
        &req.model,
        profile_name,
        required_bundles,
        StepOptions {
            redaction: redaction.as_ref(),
            open_span,
            relations,
            relation_model: &req.relation_model,
            vocabulary: req.vocabulary.as_ref(),
        },
    )
    .map_err(plain)
}

/// Sends the terminal receipt of `op` over `doc`: `result` is the
/// identity of the complete graph the client was handed, absent when it
/// was handed none, and `binding` the caller's, copied as given. Returns
/// whether the client took it.
fn send_receipt(
    tx: &mpsc::Sender<std::result::Result<AnalyzeProgressiveEvent, Status>>,
    sequence: u64,
    (op, binding): (&Operation, Option<&InputBinding>),
    doc: &Document,
    result: Option<crate::proto::result::ResultIdentity>,
) -> bool {
    let receipt = receipt::build(op, doc, input_identity(doc, binding), result, &[]);
    tx.blocking_send(Ok(AnalyzeProgressiveEvent { sequence, event: Some(Event::Receipt(receipt)) })).is_ok()
}

/// What the answer is about and how it is told: the client's document id
/// and offset unit, the PII disclosure its request asked for, and its
/// binding of the input.
struct Answer {
    doc_id: String,
    offsets: i32,
    disclosure: Disclosure,
    binding: Option<InputBinding>,
}

fn run_progressive_task(
    answer: Answer,
    mut doc: Document,
    analyzer: Analyzer,
    cancel: Arc<AtomicBool>,
    deadline: Deadline,
    tx: mpsc::Sender<std::result::Result<AnalyzeProgressiveEvent, Status>>,
    mut op: Operation,
) -> Option<(DocumentGraphResponse, Document)> {
    let Answer { doc_id, offsets: proto_offsets, disclosure, binding } = answer;
    // Whether snapshots may carry graph bytes is decided for the request
    // before anything streams: a pipeline that will write PII findings
    // withholds them from the first snapshot on, since the text is the
    // same before the PII step runs as after (#1904).
    let snapshot_graphs = disclosure.full_text || !(holds_pii(&doc) || plans_pii(&analyzer));
    if deadline.is_expired() {
        let _ = tx.blocking_send(Err(deadline_status()));
        return None;
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
            return None;
        }
    };

    let started = AnalyzeProgressiveEvent {
        sequence: seq,
        event: Some(Event::Started(AnalysisStarted {
            doc_id,
            offsets: proto_offsets,
            step_ids: analyzer.step_ids().into_iter().map(str::to_owned).collect(),
            views: initial_export.views.iter().map(|r| view_info(&initial_export, r)).collect(),
        })),
    };
    if tx.blocking_send(Ok(started)).is_err() {
        cancel.store(true, Ordering::Relaxed);
        return None;
    }

    let mut docs = vec![doc];
    analyzer.run_progressive(&mut docs, &cancel, |ev| {
        if cancel.load(Ordering::Relaxed) {
            return;
        }
        seq += 1;
        op.executed_steps.push(ev.timing.step_id.to_owned());
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
                .map(|r| layer_info(&snap_graph, r))
                .collect();
            let error = of_error(&err);
            let kind = LayerErrorKind::try_from(error.kind).unwrap_or_default();
            AnalyzeProgressiveEvent {
                sequence: seq,
                event: Some(Event::StepFailed(AnalysisStepFailure {
                    step_id: ev.timing.step_id.to_owned(),
                    stage: ev.timing.stage as u32,
                    duration_us: ev.timing.duration_us,
                    failure: Some(AnalyzeFailure {
                        turbo_code: error.turbo_code,
                        field: 0,
                        // A code name, never the error's text: an annotator's
                        // detail can quote the document (issue #1908). An
                        // input over a limit says which and by how much,
                        // counts only (#2125, #2134).
                        message: match &err {
                            turbo_analysis::Error::OverLimit { limit, max, got, .. } => {
                                format!("step '{}' failed: over its {limit} {max}, got {got}", ev.timing.step_id)
                            }
                            _ => format!("step '{}' failed: {}", ev.timing.step_id, kind_name(kind)),
                        },
                    }),
                    skipped_dependents: ev.skipped_dependents.into_iter().map(str::to_owned).collect(),
                    layers: affected_layers,
                    error: Some(error),
                })),
            }
        } else {
            let new_layers = snap_graph
                .layers
                .iter()
                .filter(|r| written_names.contains(&(r.layer.clone(), r.instance.clone())))
                .map(|r| layer_info(&snap_graph, r))
                .collect();
            let views = snap_graph.views.iter().map(|r| view_info(&snap_graph, r)).collect();
            let graph_bytes = if !snapshot_graphs {
                Vec::new()
            } else {
                match snap_graph.to_bytes() {
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

    doc = docs.pop().expect("single document");
    if deadline.is_expired() {
        // The deadline cancelled the run: what ran is on the receipt, and
        // no result was handed over.
        op.cancelled = true;
        let _ = send_receipt(&tx, seq + 1, (&op, binding.as_ref()), &doc, None);
        let _ = tx.blocking_send(Err(deadline_status()));
        return None;
    }
    if cancel.load(Ordering::Relaxed) {
        return None;
    }

    // The sink is offered the whole export and applies its own disclosure
    // (docs/grpc-sink.md); the client is answered under the request's.
    let full = export_document_response(&doc, &Disclosure::FULL);
    let answered = if disclosure.graph(&doc) { full.clone() } else { export_document_response(&doc, &disclosure) };
    match full.and_then(|full| answered.map(|complete| (full, complete))) {
        Ok((full, mut complete)) => {
            // The client's findings projection carries the request's binding.
            complete.findings = complete.findings.map(|f| bound(f, binding.as_ref()));
            seq += 1;
            // The result is the complete graph whether or not the client
            // was handed its bytes.
            let result = result_identity(&full.graph);
            // What the sink is offered: the bytes, layers and views of the
            // export, and the document they were exported from.
            let exported = DocumentGraphResponse {
                graph: full.graph,
                layers: full.layers,
                views: full.views,
                source_text: String::new(),
                findings: None,
            };
            let event = AnalyzeProgressiveEvent { sequence: seq, event: Some(Event::Complete(complete)) };
            if tx.blocking_send(Ok(event)).is_err() {
                return None;
            }
            // The terminal receipt; a client that left before it holds an
            // incomplete stream, but it was handed the complete graph.
            let _ = send_receipt(&tx, seq + 1, (&op, binding.as_ref()), &doc, Some(result));
            Some((exported, doc))
        }
        Err(f) => {
            let _ = tx.blocking_send(Err(plain(f)));
            None
        }
    }
}
