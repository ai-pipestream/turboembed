//! One stream's reader (issue #722): the request messages after the
//! header, admitted through the advertised queue caps and validated.
//! Every message, a failure included, takes its queue room before it
//! reaches the engine, so the caps bound failures and results alike; a
//! document over a byte cap refuses at once, naming the limit, without
//! waiting for room it can never take. A document past the caps does
//! not error the stream: the reader waits, and stops waiting the
//! moment the stream dies, so a fast client is held by backpressure
//! alone and a cancelled stream leaks nothing it had not yet handed
//! over.

use std::collections::VecDeque;
use std::sync::Arc;

use tokio::sync::{Semaphore, mpsc};
use tonic::{Status, Streaming};

use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT, TURBO_E_INVALID_UTF8};
use crate::proto::result::analyze_stream_request::Kind;
use crate::proto::result::{AnalyzeDocument, AnalyzeOutput, AnalyzeStreamRequest};
use crate::state::State;
use crate::stream::handler;
use crate::stream::run::Doc;
use crate::stream::{Dead, Envelope, Room, Shared, admit, admit_refusal, bound_retained_id, over_byte_cap, over_bytes};

/// Validates one document the way the single-document call validates
/// its model and options, and builds the work the scheduler queues.
/// The refusal is the document's own: it crosses as a failed answer,
/// tagged with the document's id, holding the room it was admitted
/// with, and the stream continues.
fn check(state: &State, d: AnalyzeDocument, room: Room) -> Result<Doc, Box<(String, Failure, Room)>> {
    let id = d.id.clone();
    let checked = (|d: AnalyzeDocument, room: Room| -> Result<Doc, (Failure, Room)> {
        if d.id.is_empty() {
            return Err((Failure::new(TURBO_E_INVALID_ARGUMENT, 0, "id is empty"), room));
        }
        if d.model.is_empty() {
            return Err((Failure::new(TURBO_E_INVALID_ARGUMENT, 0, "model is empty"), room));
        }
        let (served, loaded) = match state.ready(&d.model, "") {
            Ok(v) => v,
            Err(f) => return Err((f, room)),
        };
        if loaded.task != turbo::TURBO_TASK_EMBED {
            return Err((Failure::new(crate::api::TURBO_E_UNSUPPORTED_TASK, 0, "the model's task is not embed"), room));
        }
        let doc_opts = match handler::doc_options(d.output, d.offsets, d.options.as_ref()) {
            Ok(o) => o,
            Err(f) => return Err((f, room)),
        };
        if doc_opts.output != AnalyzeOutput::Vectors {
            if std::str::from_utf8(&d.text).is_err() {
                return Err((Failure::new(TURBO_E_INVALID_UTF8, 0, "document text is not valid UTF-8"), room));
            }
            let embed = match loaded.embed() {
                Ok(e) => e,
                Err(f) => return Err((f, room)),
            };
            if let Err(f) = embed.validate_options(doc_opts.output, &doc_opts.embed, doc_opts.sparse_top_k) {
                return Err((f, room));
            }
        }
        let batch_limit = loaded.slots[0].info.max_batch;
        let version = loaded.revision.clone();
        Ok(Doc {
            seq: 0,
            id: d.id,
            text: d.text,
            opts: doc_opts.embed,
            output: doc_opts.output,
            offsets: doc_opts.offsets,
            sparse_top_k: doc_opts.sparse_top_k,
            sparse_floor: doc_opts.sparse_floor,
            token_stride: doc_opts.token_stride,
            loaded,
            model: served.name.clone(),
            version,
            batch_limit,
            room,
        })
    })(d, room);
    checked.map_err(|(f, room)| Box::new((id, f, room)))
}

/// Reads and admits until the input closes, the client cancels, or the
/// engine dies. Dropping the sender ends the engine's input; the
/// staged document and the backlog go with it.
pub(crate) async fn run(
    state: Arc<State>,
    shared: Arc<Shared>,
    stream_items: Arc<Semaphore>,
    stream_bytes: Arc<Semaphore>,
    mut inbound: Streaming<AnalyzeStreamRequest>,
    work: mpsc::Sender<Envelope>,
    dead: Arc<Dead>,
) {
    let mut backlog: VecDeque<AnalyzeDocument> = VecDeque::new();
    let mut input_closed = false;
    loop {
        if dead.is_set() {
            return;
        }
        let document = match backlog.pop_front() {
            Some(d) => d,
            None if input_closed => return,
            None => match inbound.message().await {
                Ok(Some(r)) => match r.kind {
                    Some(Kind::Document(d)) => d,
                    Some(Kind::Header(_)) => {
                        let _ = work
                            .send(Envelope::Abort(Status::invalid_argument(
                                "the header must be the stream's first message",
                            )))
                            .await;
                        return;
                    }
                    None => {
                        let _ =
                            work.send(Envelope::Abort(Status::invalid_argument("a request message has no kind"))).await;
                        return;
                    }
                },
                // The client half-closed: the documents in flight and
                // queued still answer.
                Ok(None) => {
                    input_closed = true;
                    continue;
                }
                // The client cancelled or went away: what is queued for
                // this stream is dropped, not drained.
                Err(_) => {
                    let _ = work.try_send(Envelope::Abort(Status::cancelled("the client cancelled the stream")));
                    return;
                }
            },
        };
        let id = document.id.clone();
        let bytes = document.text.len() + document.id.len() + document.model.len();
        // A document over a byte cap refuses at once, naming the limit,
        // but its refusal is admitted like anything else: one item and
        // the id's bytes, held until the answer is sent. Without that
        // room a flood of oversized documents would pile into the
        // ordered buffer past every cap. The retained id is bounded to
        // the stream's byte cap first, so the refusal holds exactly the
        // id bytes its room charges.
        if let Some((limit, _)) = over_byte_cap(bytes) {
            let failure = over_bytes(bytes, limit);
            let id = bound_retained_id(id);
            // The refused document is dropped before the refusal waits on
            // admission: while it waits, nothing here holds the document's
            // text or the id's original over-cap allocation.
            drop(document);
            let mut admit =
                Box::pin(admit_refusal(shared.clone(), stream_items.clone(), stream_bytes.clone(), id.len()));
            let Some(room) = wait_room(&mut admit, &dead).await else { return };
            if work.send(Envelope::Failed { id, failure, room: Some(room) }).await.is_err() {
                return;
            }
            continue;
        }
        let mut admit = Box::pin(admit(shared.clone(), stream_items.clone(), stream_bytes.clone(), bytes));
        let Some(room) = wait_room(&mut admit, &dead).await else { return };
        // Every admitted message holds its room from here on, the
        // failures included: the room frees where the answer frees.
        match check(&state, document, room) {
            Ok(doc) => {
                if work.send(Envelope::Doc(doc)).await.is_err() {
                    return;
                }
            }
            Err(failed) => {
                let (id, failure, room) = *failed;
                if work.send(Envelope::Failed { id, failure, room: Some(room) }).await.is_err() {
                    return;
                }
            }
        }
    }
}

/// Waits out an admission, or until the stream dies: an over-cap or
/// flooded admission never outlives the stream.
async fn wait_room(admit: &mut (impl std::future::Future<Output = Room> + Unpin), dead: &Arc<Dead>) -> Option<Room> {
    if dead.is_set() {
        return None;
    }
    tokio::select! {
        room = &mut *admit => Some(room),
        _ = dead.wait() => None,
    }
}
