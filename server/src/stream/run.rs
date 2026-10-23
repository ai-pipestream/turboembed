//! One micro-batch on one held session (issue #722): the same write,
//! run and read the single-document call makes, over the batch's texts,
//! then the result's rows handed back one per document. A batch whose
//! write or run fails comes back whole so the scheduler can isolate the
//! failing document by running each one alone, exactly as the
//! single-document call would answer it.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use tokio::sync::OwnedSemaphorePermit;
use turbo::{TURBO_PLACE_HOST, TURBO_PLACE_PINNED, TURBO_PLACE_SHARED, turbo_embed_options, turbo_text};

use crate::api::{Failure, Output, Result, TURBO_E_INTERNAL};
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
    pub(crate) loaded: Arc<Loaded>,
    pub(crate) model: String,
    pub(crate) version: String,
    pub(crate) batch_limit: u32,
    pub(crate) room: Room,
}

/// One document's answer: its row or its failure, with the queue room
/// and the window permit the document still holds; both free where the
/// answer frees, on the stream's cancellation too.
pub(crate) struct Row {
    pub(crate) seq: u64,
    pub(crate) id: String,
    pub(crate) data: Vec<u8>,
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

/// The grouping key: the same load, the same options. The load's
/// identity, not the served name: an unload and a reload under one name
/// are different loads and never batch together.
pub(crate) fn key(d: &Doc) -> BatchKey {
    let o = &d.opts;
    (Arc::as_ptr(&d.loaded) as usize, o.truncate, o.max_tokens, o.prompt_role, o.normalize, o.pooling, o.output_dim)
}

/// Runs one batch on the held session, one write, one run and one read.
/// Counts the model execution just before the run; a batch aborted
/// before it starts never counts.
pub(crate) fn run(
    shared: &Shared,
    held: Held,
    docs: Vec<Doc>,
    opts: &turbo_embed_options,
    permits: Vec<OwnedSemaphorePermit>,
) -> Outcome {
    let (model, version) = (docs[0].model.clone(), docs[0].version.clone());
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
            data: c.to_vec(),
            room: d.room,
            window: permits.next().expect("one window permit per document"),
        })
        .collect();
    Outcome { model, version, dim, rows: Ok(rows) }
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
