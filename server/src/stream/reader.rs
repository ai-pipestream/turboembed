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
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use tokio::sync::{Semaphore, mpsc};
use tonic::{Status, Streaming};

use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT};
use crate::proto::result::analyze_stream_request::Kind;
use crate::proto::result::{AnalyzeDocument, AnalyzeStreamRequest};
use crate::state::State;
use crate::stream::handler;
use crate::stream::run::Doc;
use crate::stream::{Envelope, Room, Shared, admit, over_byte_cap, over_bytes};

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
        let opts = match handler::options(d.options.as_ref()) {
            Ok(o) => o,
            Err(f) => return Err((f, room)),
        };
        let batch_limit = loaded.slots[0].info.max_batch;
        let version = loaded.revision.clone();
        Ok(Doc { seq: 0, id: d.id, text: d.text, opts, loaded, model: served.name.clone(), version, batch_limit, room })
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
    dead: Arc<AtomicBool>,
) {
    let mut backlog: VecDeque<AnalyzeDocument> = VecDeque::new();
    let mut input_closed = false;
    loop {
        if dead.load(Ordering::SeqCst) {
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
        // A document over a byte cap refuses at once, naming the limit:
        // admitting it would wait for room it can never take.
        if let Some((limit, _)) = over_byte_cap(bytes) {
            let failure = over_bytes(bytes, limit);
            if work.send(Envelope::Failed { id, failure, room: None }).await.is_err() {
                return;
            }
            continue;
        }
        let mut admit = Box::pin(admit(shared.clone(), stream_items.clone(), stream_bytes.clone(), bytes));
        let room = loop {
            if dead.load(Ordering::SeqCst) {
                return;
            }
            let room = tokio::select! {
                room = &mut admit => Some(room),
                _ = tokio::time::sleep(Duration::from_millis(5)) => None,
            };
            match room {
                Some(room) => break room,
                None => continue,
            }
        };
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
