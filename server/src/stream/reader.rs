//! One stream's reader (issue #722): the request messages after the
//! header, validated and admitted through the advertised queue caps.
//! A document past the caps does not error the stream: the reader
//! waits, and stops waiting the moment the stream dies, so a fast
//! client is held by backpressure alone and a cancelled stream leaks
//! nothing it had not yet handed over.

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
use crate::stream::{Envelope, Shared, admit};

/// Validates one document the way the single-document call validates
/// its model and options, and builds the work the scheduler queues.
/// The refusal is the document's own: it crosses as a failed answer,
/// tagged with the document's id, and the stream continues.
fn check(state: &State, d: AnalyzeDocument) -> Result<Doc, (String, Failure)> {
    let id = d.id.clone();
    let checked = (|d: AnalyzeDocument| {
        if d.id.is_empty() {
            return Err(Failure::new(TURBO_E_INVALID_ARGUMENT, 0, "id is empty"));
        }
        if d.model.is_empty() {
            return Err(Failure::new(TURBO_E_INVALID_ARGUMENT, 0, "model is empty"));
        }
        let (served, loaded) = state.ready(&d.model, "")?;
        let opts = handler::options(d.options.as_ref())?;
        let batch_limit = loaded.slots[0].info.max_batch;
        let version = loaded.revision.clone();
        Ok(Doc { seq: 0, id: d.id, text: d.text, opts, loaded, model: served.name.clone(), version, batch_limit })
    })(d);
    checked.map_err(|f| (id, f))
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
    let mut backlog: VecDeque<AnalyzeStreamRequest> = VecDeque::new();
    let mut input_closed = false;
    loop {
        if dead.load(Ordering::SeqCst) {
            return;
        }
        let request = match backlog.pop_front() {
            Some(r) => r,
            None if input_closed => return,
            None => match inbound.message().await {
                Ok(Some(r)) => r,
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
        let doc = match request.kind {
            Some(Kind::Document(d)) => d,
            Some(Kind::Header(_)) => {
                let _ = work
                    .send(Envelope::Abort(Status::invalid_argument("the header must be the stream's first message")))
                    .await;
                return;
            }
            None => {
                let _ = work.send(Envelope::Abort(Status::invalid_argument("a request message has no kind"))).await;
                return;
            }
        };
        let doc = match check(&state, doc) {
            Ok(doc) => doc,
            Err((id, failure)) => {
                if work.send(Envelope::Failed { id, failure }).await.is_err() {
                    return;
                }
                continue;
            }
        };
        let bytes = doc.text.len() + doc.id.len() + doc.model.len();
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
        if work.send(Envelope::Doc(doc, room)).await.is_err() {
            return;
        }
    }
}
