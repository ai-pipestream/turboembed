//! Serving `TASK_RERANK` and `TASK_EXTRACTIVE_QA` bundles over KServe v2.

mod qa;
mod rerank;

use std::path::Path;
use turbo::{TURBO_TASK_EMBED, TURBO_TASK_EXTRACTIVE_QA, TURBO_TASK_RERANK};

pub(crate) use qa::infer_qa;
pub use qa::{QaModel, qa_options};
pub(crate) use rerank::infer_rerank;
pub use rerank::rerank_options;

/// Peeks at a bundle directory's manifest task before runtime creation.
/// Returns `TURBO_TASK_EMBED` if the manifest cannot be opened so the
/// C runtime's own loader reports the underlying error.
pub fn peek_task(bundle_path: &str) -> u32 {
    let Ok(bundle) = turbo::bundle::Bundle::open(Path::new(bundle_path)) else {
        return TURBO_TASK_EMBED;
    };
    match bundle.manifest.task.as_str() {
        "TASK_RERANK" => TURBO_TASK_RERANK,
        "TASK_EXTRACTIVE_QA" => TURBO_TASK_EXTRACTIVE_QA,
        _ => TURBO_TASK_EMBED,
    }
}
