//! The processing receipt (issue #1908): what one attempt at one
//! operation did, kept apart from the immutable result it references.
//! The receipt binds the caller's operation key, a fresh server attempt
//! id, the exact input and, when there is one, the result identity, with
//! a status that never defaults to complete: UNSPECIFIED is 0 and never
//! written. Per layer it records the assessed source byte ranges, the
//! detector asked for and the ones that ran, truncations and a sanitized
//! error. The server keeps no state for the key and promises no
//! exactly-once execution; the caller rejects a reused key with
//! different input.

pub mod error;
pub mod stream;

use std::any::TypeId;
use std::collections::hash_map::RandomState;
use std::hash::BuildHasher;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use turbo_analysis::{Document, Key, KeyId, View};

use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT};
use crate::proto::result::{
    ByteRange, DetectorRef, InputIdentity, LayerReceipt, LayerWireStatus, ProcessingReceipt, ProcessingStatus,
    ResultIdentity, TruncationReason,
};

/// The version of the receipt's messages and rules.
pub const RECEIPT_SCHEMA_VERSION: u32 = 1;

/// The longest operation key accepted, in bytes.
pub const MAX_OPERATION_KEY_BYTES: usize = 128;

/// Refuses an operation key that is not at most
/// [`MAX_OPERATION_KEY_BYTES`] of printable ASCII. The refusal never
/// quotes the key.
pub fn check_operation_key(key: &str) -> Result<(), Failure> {
    if key.len() > MAX_OPERATION_KEY_BYTES || !key.bytes().all(|b| b.is_ascii_graphic()) {
        return Err(Failure::new(
            TURBO_E_INVALID_ARGUMENT,
            0,
            format!("operation_key must be at most {MAX_OPERATION_KEY_BYTES} printable ASCII bytes without spaces"),
        ));
    }
    Ok(())
}

/// A fresh attempt id: 32 lowercase hexadecimal digits, unique within a
/// process by a counter and across processes by a seed drawn once from
/// the standard library's per-process random hash keys. Not a secret.
pub fn new_attempt_id() -> String {
    static SEED: OnceLock<(RandomState, RandomState)> = OnceLock::new();
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let (a, b) = SEED.get_or_init(|| (RandomState::new(), RandomState::new()));
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    format!("{:016x}{:016x}", a.hash_one(n), b.hash_one(n) ^ n)
}

/// What a producer knows about a layer's coverage that the document does
/// not record: the source byte ranges it assessed and why it stopped
/// short. A layer without one assessed its whole text.
#[derive(Clone, Debug, Default)]
pub struct Coverage {
    pub layer: String,
    pub instance: String,
    pub assessed: Vec<ByteRange>,
    pub truncations: Vec<TruncationReason>,
}

/// One attempt at one operation, as the server ran it.
#[derive(Clone, Debug, Default)]
pub struct Operation {
    pub key: String,
    pub attempt_id: String,
    pub requested_steps: Vec<String>,
    pub executed_steps: Vec<String>,
    pub cancelled: bool,
}

impl Operation {
    /// An operation under the caller's `key` with a fresh attempt id.
    pub fn start(key: &str) -> Operation {
        Operation { key: key.to_owned(), attempt_id: new_attempt_id(), ..Operation::default() }
    }
}

/// `ranges` as the proto promises them: each clamped to `[0, len)`, empty
/// ones dropped, sorted, and overlapping or touching ones merged, so the
/// result is sorted and disjoint whatever a producer reported.
pub fn normalize_ranges(ranges: &[ByteRange], len: u32) -> Vec<ByteRange> {
    let mut out: Vec<ByteRange> = ranges
        .iter()
        .map(|r| ByteRange { start: r.start.min(len), end: r.end.min(len) })
        .filter(|r| r.start < r.end)
        .collect();
    out.sort_unstable_by_key(|r| (r.start, r.end));
    let mut merged: Vec<ByteRange> = Vec::with_capacity(out.len());
    for r in out {
        match merged.last_mut() {
            Some(last) if r.start <= last.end => last.end = last.end.max(r.end),
            _ => merged.push(r),
        }
    }
    merged
}

/// Whether normalized `assessed` ranges cover all of `[0, len)`.
pub fn covers_whole(assessed: &[ByteRange], len: u32) -> bool {
    match assessed {
        [] => len == 0,
        [only] => only.start == 0 && only.end == len,
        _ => false,
    }
}

fn layer_receipt(doc: &Document, key: KeyId, source_len: u32, coverage: &[Coverage]) -> LayerReceipt {
    let (wire, error) = error::layer_status(doc, key);
    let layer = key.layer.to_string();
    let detector = doc
        .provenance(key)
        .filter(|p| !p.annotator.is_empty())
        .map(|p| DetectorRef { id: p.annotator.to_owned(), version: p.version.to_owned() });
    let is_view = key.ty == TypeId::of::<View>();
    let item_count = if is_view { 0 } else { doc.layer_len_by_name(&layer, key.name).unwrap_or(0) as u32 };
    let whole = if source_len == 0 { Vec::new() } else { vec![ByteRange { start: 0, end: source_len }] };
    let cover = coverage.iter().find(|c| c.layer == layer && c.instance == key.name);
    let (status, assessed, truncations) = match wire {
        LayerWireStatus::Ok => match cover {
            Some(c) => {
                let assessed = normalize_ranges(&c.assessed, source_len);
                let cuts: Vec<i32> =
                    c.truncations.iter().filter(|t| **t != TruncationReason::Unspecified).map(|t| *t as i32).collect();
                let partial = !cuts.is_empty() || !covers_whole(&assessed, source_len);
                let status = if partial { ProcessingStatus::Partial } else { ProcessingStatus::Complete };
                (status, assessed, cuts)
            }
            None => (ProcessingStatus::Complete, whole, Vec::new()),
        },
        LayerWireStatus::Skipped => (ProcessingStatus::NotAssessed, Vec::new(), Vec::new()),
        LayerWireStatus::Failed => (ProcessingStatus::Failed, Vec::new(), Vec::new()),
        LayerWireStatus::Unspecified => (ProcessingStatus::Unspecified, Vec::new(), Vec::new()),
    };
    let executed = if wire == LayerWireStatus::Ok { detector.iter().cloned().collect() } else { Vec::new() };
    LayerReceipt {
        layer,
        instance: key.name.to_owned(),
        status: status as i32,
        item_count,
        assessed,
        requested: detector.into_iter().collect(),
        executed,
        truncations,
        error,
        provenance_digest: doc.provenance_digest(key).map(|d| d.as_bytes().to_vec()),
    }
}

/// The document status from its layers: CANCELLED for a cancelled
/// operation, FAILED when any layer failed, NOT_ASSESSED when nothing
/// past the source view was assessed, PARTIAL when any layer is partial
/// or not assessed or a requested step did not run, else COMPLETE.
pub fn document_status(op: &Operation, layers: &[LayerReceipt]) -> ProcessingStatus {
    if op.cancelled {
        return ProcessingStatus::Cancelled;
    }
    let source = Key::<View>::DEFAULT.id();
    let source_layer = source.layer.to_string();
    let work: Vec<ProcessingStatus> = layers
        .iter()
        .filter(|l| !(l.layer == source_layer && l.instance == source.name))
        .map(|l| ProcessingStatus::try_from(l.status).unwrap_or(ProcessingStatus::Unspecified))
        .collect();
    if work.iter().any(|s| matches!(s, ProcessingStatus::Failed | ProcessingStatus::Unspecified)) {
        return ProcessingStatus::Failed;
    }
    let assessed = work.iter().filter(|s| matches!(s, ProcessingStatus::Complete | ProcessingStatus::Partial)).count();
    if assessed == 0 {
        return ProcessingStatus::NotAssessed;
    }
    let missing_step = op.requested_steps.iter().any(|s| !op.executed_steps.contains(s));
    if missing_step || assessed < work.len() || work.contains(&ProcessingStatus::Partial) {
        return ProcessingStatus::Partial;
    }
    ProcessingStatus::Complete
}

/// The receipt of `op` over `doc`, bound to `input` and, when the attempt
/// produced one, `result`. `coverage` is what producers report beyond
/// the document; empty means every complete layer assessed its whole text.
pub fn build(
    op: &Operation,
    doc: &Document,
    input: InputIdentity,
    result: Option<ResultIdentity>,
    coverage: &[Coverage],
) -> ProcessingReceipt {
    let source_len = input.source_text_len;
    let layers: Vec<LayerReceipt> = doc.layers().map(|k| layer_receipt(doc, k, source_len, coverage)).collect();
    let status = document_status(op, &layers);
    ProcessingReceipt {
        receipt_schema_version: RECEIPT_SCHEMA_VERSION,
        operation_key: op.key.clone(),
        attempt_id: op.attempt_id.clone(),
        input: Some(input),
        result,
        status: status as i32,
        item_count: layers.iter().map(|l| u64::from(l.item_count)).sum(),
        layers,
        requested_steps: op.requested_steps.clone(),
        executed_steps: op.executed_steps.clone(),
        config_digest: doc.processing_config_digest().as_bytes().to_vec(),
    }
}
