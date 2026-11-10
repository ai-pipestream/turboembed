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
use crate::proto::result::{AnalyzeDocument, AnalyzeOutput, AnalyzeStreamRequest, InputBinding};
use crate::results::typed::identity::check_binding;
use crate::state::State;
use crate::stream::handler;
use crate::stream::run::Doc;
use crate::stream::{
    Dead, Envelope, MAX_QUEUED_BYTES_PER_STREAM, Room, Shared, admit, admit_refusal, bound_retained_id, over_byte_cap,
    over_bytes,
};

/// A document refused before it ran: its id, its binding as given, why,
/// and the room it holds.
type Refused = (String, Option<InputBinding>, Failure, Room);

/// Validates one document the way the single-document call validates
/// its model and options, and builds the work the scheduler queues.
/// The refusal is the document's own: it crosses as a failed answer,
/// tagged with the document's id, holding the room it was admitted
/// with, and the stream continues.
fn check(state: &State, d: AnalyzeDocument, mut room: Room) -> Result<Doc, Box<Refused>> {
    let id = d.id.clone();
    let checked: Result<_, Failure> = (|| {
        if d.id.is_empty() {
            return Err(Failure::new(TURBO_E_INVALID_ARGUMENT, 0, "id is empty"));
        }
        check_binding(d.binding.as_ref(), "binding")?;
        if d.model.is_empty() && d.output != AnalyzeOutput::DocumentGraph as i32 {
            return Err(Failure::new(TURBO_E_INVALID_ARGUMENT, 0, "model is empty"));
        }
        let resolved_model = if d.model.is_empty() {
            None
        } else if d.output == AnalyzeOutput::DocumentGraph as i32 {
            let served = state.ready_served(&d.model)?;
            let loaded = served.loaded.get().cloned();
            if let Some(l) = &loaded {
                room.bind_device(l.device);
            }
            Some((served, loaded))
        } else {
            let (served, loaded) = state.ready(&d.model, "")?;
            room.bind_device(loaded.device);
            if loaded.task != turbo::TURBO_TASK_EMBED {
                return Err(Failure::new(crate::api::TURBO_E_UNSUPPORTED_TASK, 0, "the model's task is not embed"));
            }
            Some((served, Some(loaded)))
        };
        let doc_opts = handler::doc_options(d.output, d.offsets, d.options.as_ref())?;
        let default_vocab = crate::vocabulary::ResolvedVocabulary::default();
        let base_vocab = resolved_model.as_ref().map_or(&default_vocab, |(served, _)| &served.vocabulary);
        let vocabulary = state.vocabulary_resolver.resolve_merged(base_vocab, d.vocabulary.as_ref())?;
        if doc_opts.output == AnalyzeOutput::DocumentGraph {
            if std::str::from_utf8(&d.text).is_err() {
                return Err(Failure::new(TURBO_E_INVALID_UTF8, 0, "document text is not valid UTF-8"));
            }
        } else if !matches!(doc_opts.output, AnalyzeOutput::Vectors | AnalyzeOutput::VectorCodes) {
            if std::str::from_utf8(&d.text).is_err() {
                return Err(Failure::new(TURBO_E_INVALID_UTF8, 0, "document text is not valid UTF-8"));
            }
            let (_, Some(loaded)) = resolved_model.as_ref().expect("model required for embed outputs") else {
                unreachable!()
            };
            let embed = loaded.embed()?;
            embed.validate_options(doc_opts.output, &doc_opts.embed, doc_opts.sparse_top_k)?;
        }
        if let Some(r) = &doc_opts.int8_ranges {
            let Some((_, Some(loaded))) = resolved_model.as_ref() else {
                return Err(Failure::new(TURBO_E_INVALID_ARGUMENT, 8, "int8_ranges requires a loaded embedding model"));
            };
            // The library reads 2 x the resolved width: the table must be that long.
            let o = &doc_opts.embed;
            let width = if o.output_dim == 0 { loaded.model.info()?.dim } else { o.output_dim };
            if r.len() != 2 * width as usize {
                return Err(Failure::new(
                    TURBO_E_INVALID_ARGUMENT,
                    8,
                    format!(
                        "int8_ranges holds {} values and the vectors are {width} wide: 2 x {width} needed",
                        r.len()
                    ),
                ));
            }
        }
        Ok((resolved_model, doc_opts, vocabulary))
    })();
    let (resolved_model, doc_opts, vocabulary) = match checked {
        Ok(v) => v,
        Err(f) => return Err(Box::new((id, d.binding, f, room))),
    };
    let (loaded, model, version, batch_limit) = match resolved_model {
        Some((served, Some(loaded))) => {
            let batch_limit = loaded.slots[0].info.max_batch;
            let version = loaded.revision.clone();
            let model = served.name.clone();
            (Some(loaded), model, version, batch_limit)
        }
        Some((served, None)) => {
            let version = served
                .gliner
                .get()
                .map(|g| g.identity().revision.clone())
                .or_else(|| served.relations.get().map(|r| r.model().revision.clone()))
                .unwrap_or_default();
            (None, served.name.clone(), version, crate::stream::MAX_INFLIGHT_ITEMS_PER_STREAM as u32)
        }
        None => (None, String::new(), String::new(), crate::stream::MAX_INFLIGHT_ITEMS_PER_STREAM as u32),
    };
    Ok(Doc {
        seq: 0,
        id: d.id,
        binding: d.binding,
        text: d.text,
        opts: doc_opts.embed,
        int8_ranges: doc_opts.int8_ranges,
        output: doc_opts.output,
        offsets: doc_opts.offsets,
        sparse_top_k: doc_opts.sparse_top_k,
        sparse_floor: doc_opts.sparse_floor,
        token_stride: doc_opts.token_stride,
        vocabulary,
        loaded,
        model,
        version,
        batch_limit,
        room,
    })
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
        let mut document = match backlog.pop_front() {
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
        let binding_len = document.binding.as_ref().map_or(0, prost::Message::encoded_len);
        let vocab_len = document.vocabulary.as_ref().map_or(0, prost::Message::encoded_len);
        let bytes = document.text.len() + document.id.len() + document.model.len() + binding_len + vocab_len;
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
            // The binding is echoed only when it fits the room the id took.
            let binding = document.binding.take().filter(|_| binding_len + id.len() <= MAX_QUEUED_BYTES_PER_STREAM);
            // The refused document is dropped before the refusal waits on
            // admission: while it waits, nothing here holds the document's
            // text or the id's original over-cap allocation.
            drop(document);
            let held = id.len() + binding.as_ref().map_or(0, |_| binding_len);
            let mut admit = Box::pin(admit_refusal(shared.clone(), stream_items.clone(), stream_bytes.clone(), held));
            let Some(room) = wait_room(&mut admit, &dead).await else { return };
            if work.send(Envelope::Failed { id, binding, failure, room: Some(room) }).await.is_err() {
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
                let (id, binding, failure, room) = *failed;
                if work.send(Envelope::Failed { id, binding, failure, room: Some(room) }).await.is_err() {
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
