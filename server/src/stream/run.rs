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
use crate::stream::Shared;

/// One admitted, queued document: everything a batch run needs, the
/// model load held alive by its Arc from admission through the run, so
/// an unload mid-stream never drops a session from under it.
pub(crate) struct Doc {
    pub(crate) seq: u64,
    pub(crate) id: String,
    pub(crate) text: Vec<u8>,
    pub(crate) opts: turbo_embed_options,
    pub(crate) loaded: Arc<Loaded>,
    pub(crate) model: String,
    pub(crate) version: String,
    pub(crate) batch_limit: u32,
}

/// One document's row, with the id the answer is tagged by.
pub(crate) type Row = (u64, String, Vec<u8>);

/// A batch's answer: the metadata once, then one row per document, in
/// the batch's order. `rows` is `Err` only when the batch came back
/// whole, to be retried one document at a time. `window` carries the
/// executed documents' window permits back: they free where the answer
/// frees, on the stream's cancellation too.
pub(crate) struct Outcome {
    pub(crate) model: String,
    pub(crate) version: String,
    pub(crate) dim: u32,
    pub(crate) rows: std::result::Result<Vec<Row>, (Failure, Vec<Doc>)>,
    pub(crate) window: Vec<OwnedSemaphorePermit>,
}

/// The grouping key: the same model, the same options. Documents batch
/// together only under one key.
pub(crate) fn key(d: &Doc) -> crate::stream::engine::BatchKey {
    let o = &d.opts;
    (d.model.clone(), o.truncate, o.max_tokens, o.prompt_role, o.normalize, o.pooling, o.output_dim)
}

/// Runs one batch on the held session, one write, one run and one read.
/// Counts the model execution just before the run; a batch aborted
/// before it starts never counts.
pub(crate) fn run(
    shared: &Shared,
    held: Held,
    docs: Vec<Doc>,
    opts: &turbo_embed_options,
    window: Vec<OwnedSemaphorePermit>,
) -> Outcome {
    let (model, version) = (docs[0].model.clone(), docs[0].version.clone());
    let fail = |rows, window| Outcome { model: model.clone(), version: version.clone(), dim: 0, rows, window };
    let texts: Vec<turbo_text> =
        docs.iter().map(|d| turbo_text { ptr: d.text.as_ptr().cast(), len: d.text.len() as u64 }).collect();
    let slot = held.slot();
    if let Err(f) = slot.session.write_text(&texts, opts) {
        return fail(Err((f, docs)), window);
    }
    shared.runs.fetch_add(1, Ordering::Relaxed);
    let result = match slot.session.run() {
        Ok(r) => r,
        Err(f) => return fail(Err((f, docs)), window),
    };
    drop(texts);
    let (dim, data) = match read(&result, docs.len()) {
        Ok(v) => v,
        Err(f) => return fail(Err((f, docs)), window),
    };
    let row = dim as usize * 4;
    let rows: Vec<Row> = data.chunks_exact(row).zip(docs).map(|(c, d)| (d.seq, d.id, c.to_vec())).collect();
    Outcome { model, version, dim, rows: Ok(rows), window }
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
