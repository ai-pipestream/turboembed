//! Serving `TASK_EMBED` sparse/token-vector outputs, `TASK_RERANK`, and
//! `TASK_EXTRACTIVE_QA` bundles over gRPC.

pub(crate) mod embed;
mod multires;
mod qa;
mod rerank;

use crate::api::Failure;
use std::path::Path;
use turbo::{TURBO_TASK_EMBED, TURBO_TASK_EXTRACTIVE_QA, TURBO_TASK_RERANK};

pub use embed::EmbedModel;
pub(crate) use qa::infer_qa;
pub use qa::{QaModel, qa_options};
pub use rerank::rerank_options;
pub(crate) use rerank::{infer_rerank, score_rerank_batches};

/// Peeks at a bundle directory's manifest task before runtime creation,
/// so the device is selected for the bundle's own task. A manifest that
/// cannot be read fails with the bundle's error: the model load reads the
/// same manifest and would fail the same way, and a default task would
/// select a device for a task the bundle may not have (#2517).
pub fn peek_task(bundle_path: &str) -> Result<u32, Failure> {
    let bundle =
        turbo::bundle::Bundle::open(Path::new(bundle_path)).map_err(|e| Failure::new(e.code, e.field, e.message))?;
    Ok(match bundle.manifest.task.as_str() {
        "TASK_RERANK" => TURBO_TASK_RERANK,
        "TASK_EXTRACTIVE_QA" => TURBO_TASK_EXTRACTIVE_QA,
        _ => TURBO_TASK_EMBED,
    })
}
