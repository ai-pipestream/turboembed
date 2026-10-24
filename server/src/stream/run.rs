//! One micro-batch on one held session (issue #722): the same write,
//! run and read the single-document call makes, over the batch's texts,
//! then the result's rows handed back one per document. A batch whose
//! write or run fails comes back whole so the scheduler can isolate the
//! failing document by running each one alone, exactly as the
//! single-document call would answer it.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tokio::sync::OwnedSemaphorePermit;
use turbo::{TURBO_PLACE_HOST, TURBO_PLACE_PINNED, TURBO_PLACE_SHARED, turbo_embed_options, turbo_text};

use crate::api::{Failure, Output, Result, TURBO_E_INTERNAL, TURBO_E_INVALID_UTF8, TURBO_E_UNSUPPORTED_OPTION};
use crate::offsets::OffsetMode;
use crate::proto::result::AnalyzeOutput;
use crate::proto::result::analyze_stream_response::Outcome as ProtoOutcome;
use crate::state::{Held, Loaded};
use crate::stream::{BatchKey, Room, Shared};

/// One admitted, queued document: everything a batch run needs, the
/// model load held alive by its Arc from admission through the run, so
/// an unload mid-stream never drops a session from under it, and the
/// queue room it holds until its answer is sent or the stream dies.
pub(crate) struct Doc {
    pub(crate) seq: u64,
    pub(crate) id: String,
    pub(crate) text: Vec<u8>,
    pub(crate) opts: turbo_embed_options,
    pub(crate) output: AnalyzeOutput,
    pub(crate) offsets: OffsetMode,
    pub(crate) sparse_top_k: u32,
    pub(crate) sparse_floor: f32,
    pub(crate) token_stride: u32,
    pub(crate) loaded: Arc<Loaded>,
    pub(crate) model: String,
    pub(crate) version: String,
    pub(crate) batch_limit: u32,
    pub(crate) room: Room,
}

/// The computed payload for one document in a batch.
pub(crate) enum RowPayload {
    Vector(Vec<u8>),
    Outcome(Box<ProtoOutcome>),
}

/// One document's answer: its row or its failure, with the queue room
/// and the window permit the document still holds; both free where the
/// answer frees, on the stream's cancellation too.
pub(crate) struct Row {
    pub(crate) seq: u64,
    pub(crate) id: String,
    pub(crate) payload: RowPayload,
    pub(crate) room: Room,
    pub(crate) window: OwnedSemaphorePermit,
}

/// A batch that failed whole, to be retried one document at a time: the
/// documents with their rooms, and their window permits.
pub(crate) struct Whole {
    pub(crate) failure: Failure,
    pub(crate) docs: Vec<Doc>,
    pub(crate) permits: Vec<OwnedSemaphorePermit>,
}

/// A batch's answer: the metadata once, then one row per document, in
/// the batch's order.
pub(crate) struct Outcome {
    pub(crate) model: String,
    pub(crate) version: String,
    pub(crate) dim: u32,
    pub(crate) rows: std::result::Result<Vec<Row>, Whole>,
}

/// The grouping key: the same load, output kind, offset mode, and options.
/// The load's identity, not the served name: an unload and a reload under
/// one name are different loads and never batch together.
pub(crate) fn key(d: &Doc) -> BatchKey {
    let o = &d.opts;
    (
        Arc::as_ptr(&d.loaded) as usize,
        d.output as i32,
        d.offsets.to_proto(),
        o.truncate,
        o.max_tokens,
        o.prompt_role,
        o.normalize,
        o.pooling,
        o.output_dim,
        d.sparse_top_k,
        d.sparse_floor.to_bits(),
        d.token_stride,
    )
}

/// Runs one batch on the held session, one write, one run and one read.
/// Counts the model execution just before the run; a batch aborted
/// before it starts never counts.
pub(crate) fn run(
    shared: &Shared,
    dead: &AtomicBool,
    held: Held,
    docs: Vec<Doc>,
    opts: &turbo_embed_options,
    permits: Vec<OwnedSemaphorePermit>,
) -> Outcome {
    let (model, version) = (docs[0].model.clone(), docs[0].version.clone());
    let output = docs[0].output;
    if output != AnalyzeOutput::Vectors {
        return run_structured(shared, dead, held, docs, opts, permits, model, version);
    }
    let fail = |failure, docs, permits| Outcome {
        model: model.clone(),
        version: version.clone(),
        dim: 0,
        rows: Err(Whole { failure, docs, permits }),
    };
    let texts: Vec<turbo_text> =
        docs.iter().map(|d| turbo_text { ptr: d.text.as_ptr().cast(), len: d.text.len() as u64 }).collect();
    let slot = held.slot();
    if let Err(f) = slot.session.write_text(&texts, opts) {
        return fail(f, docs, permits);
    }
    shared.runs.fetch_add(1, Ordering::Relaxed);
    let result = match slot.session.run() {
        Ok(r) => r,
        Err(f) => return fail(f, docs, permits),
    };
    drop(texts);
    let (dim, data) = match read(&result, docs.len()) {
        Ok(v) => v,
        Err(f) => return fail(f, docs, permits),
    };
    let row = dim as usize * 4;
    let mut permits = permits.into_iter();
    let rows: Vec<Row> = data
        .chunks_exact(row)
        .zip(docs)
        .map(|(c, d)| Row {
            seq: d.seq,
            id: d.id,
            payload: RowPayload::Vector(c.to_vec()),
            room: d.room,
            window: permits.next().expect("one window permit per document"),
        })
        .collect();
    Outcome { model, version, dim, rows: Ok(rows) }
}

#[allow(clippy::too_many_arguments)]
fn run_structured(
    shared: &Shared,
    dead: &AtomicBool,
    held: Held,
    docs: Vec<Doc>,
    opts: &turbo_embed_options,
    permits: Vec<OwnedSemaphorePermit>,
    model: String,
    version: String,
) -> Outcome {
    let fail = |failure, docs, permits| Outcome {
        model: model.clone(),
        version: version.clone(),
        dim: 0,
        rows: Err(Whole { failure, docs, permits }),
    };
    if docs[0].loaded.embed.is_none() {
        return fail(Failure::new(TURBO_E_UNSUPPORTED_OPTION, 0, "the model has no embed companion"), docs, permits);
    }
    let loaded = docs[0].loaded.clone();
    let embed = loaded.embed.as_ref().expect("embed companion verified");
    let mut texts: Vec<&str> = Vec::with_capacity(docs.len());
    for d in &docs {
        match std::str::from_utf8(&d.text) {
            Ok(s) => texts.push(s),
            Err(_) => {
                return fail(Failure::new(TURBO_E_INVALID_UTF8, 0, "document text is not valid UTF-8"), docs, permits);
            }
        }
    }
    let slot = held.slot();
    let offsets = docs[0].offsets;
    let computed = match docs[0].output {
        AnalyzeOutput::SparseTerms => {
            let cut = turbo::sparse::SparseOptions { top_k: docs[0].sparse_top_k, floor: docs[0].sparse_floor };
            embed.run_sparse_batch(shared, dead, slot, &texts, opts, offsets, cut)
        }
        AnalyzeOutput::TokenVectors => {
            embed.run_token_vectors_batch(shared, dead, slot, &texts, opts, offsets, docs[0].token_stride)
        }
        AnalyzeOutput::Vectors => unreachable!(),
    };
    drop(texts);
    let outcomes = match computed {
        Ok(o) => o,
        Err(f) => return fail(f, docs, permits),
    };
    let mut permits = permits.into_iter();
    let rows = outcomes
        .into_iter()
        .zip(docs)
        .map(|(out, d)| Row {
            seq: d.seq,
            id: d.id,
            payload: RowPayload::Outcome(Box::new(out)),
            room: d.room,
            window: permits.next().expect("one window permit per document"),
        })
        .collect();
    Outcome { model, version, dim: 0, rows: Ok(rows) }
}

/// The result's whole bytes and dim, read the way the single-document
/// call reads them, whatever placement the backend produced.
fn read(result: &Output, want_rows: usize) -> Result<(u32, Vec<u8>)> {
    let first = result.info()?;
    let data = match first.placement {
        TURBO_PLACE_HOST | TURBO_PLACE_PINNED | TURBO_PLACE_SHARED => {
            let buffer = result.buffer()?;
            let ptr = buffer.host_ptr()?;
            let bytes = first.bytes as usize;
            if bytes == 0 {
                Vec::new()
            } else {
                let v = unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), bytes) }.to_vec();
                drop(buffer);
                v
            }
        }
        _ => result.read(first.bytes)?,
    };
    let info = result.info()?;
    if info.batch as usize != want_rows {
        return Err(Failure::new(
            TURBO_E_INTERNAL,
            0,
            format!("the result carries {} rows for {} documents", info.batch, want_rows),
        ));
    }
    Ok((info.dim, data))
}
