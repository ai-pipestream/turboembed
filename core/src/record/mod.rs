//! Benchmark records, as docs/benchmarks.md describes them: written by
//! the turbo-bench tool, one JSON file per measurement, and read here to
//! decide whether a capability cell is SUPPORTED. The records committed in
//! benchmarks/records/ are compiled into the library by build.rs, so it
//! opens no file at run time.

use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::bundle::sha256_hex;
use crate::{
    TURBO_DTYPE_BF16, TURBO_DTYPE_F16, TURBO_DTYPE_F32, TURBO_PRECISION_EXACT, TURBO_PRECISION_FASTEST,
    TURBO_PRECISION_MODEL, TURBO_TASK_EMBED,
};

/// The schema version this library writes and reads.
pub const RECORD_VERSION: u32 = 1;

/// The longest file name a record may have: turbo_capability.benchmark
/// holds 96 bytes with its NUL.
pub const NAME_MAX: usize = 95;

/// Text no record may hold anywhere: the start of a user's home directory
/// on Linux (root's included) and on macOS. A record is published, and the
/// tool writes each host path in it as a placeholder (docs/benchmarks.md).
pub const HOST_PATHS: [&str; 6] = ["/home/", "/root/", "/var/home/", "/Users/", "C:\\Users\\", "C:/Users/"];

/// The kinds of token rows a record may be measured on.
pub const ROWS_MIXED: &str = "ROWS_MIXED";
pub const ROWS_DENSE: &str = "ROWS_DENSE";

fn rows_mixed() -> String {
    ROWS_MIXED.into()
}

/// The environment variables that change what a backend runs, each with
/// its backend: a record names those that were set in library.settings.
///
/// TURBO_CUDA_TUNED and TURBO_CUDA_CHOICES, and the levelzero and metal
/// backends' TURBO_<BACKEND>_TUNED and TURBO_<BACKEND>_CHOICES, are not
/// read from the environment: they are the session's turbo_session_info.tuned and
/// choices, present in every record of a backend that reports choices, so
/// a record made with nothing set still says which kernels ran, and the
/// CHOICES variable set to that string forces them back.
///
/// TURBO_NPU_GRAPH_FORMAT is the same kind of report: the graph format
/// the device selected (`NGRAPH_LITE` or `NATIVE`), from the session's
/// choices. It is not an environment override. TURBO_NPU_GRAPH_INPUT is
/// the loaded artifact's graph input (`INPUT_TOKEN_IDS` or
/// `INPUT_EMBEDDINGS`), written by the bench from that artifact.
pub const LIBRARY_VARS: [(&str, &str); 26] = [
    ("cpu", "TURBO_CPU_THREADS"),
    ("cuda", "TURBO_CUDA_TILE"),
    ("cuda", "TURBO_CUDA_SK_STEPS"),
    ("cuda", "TURBO_CUDA_ATTENTION"),
    ("cuda", "TURBO_CUDA_CUBLAS"),
    ("cuda", "TURBO_CUDA_LAYER_NORM"),
    ("cuda", "TURBO_CUDA_POOL"),
    ("cuda", "TURBO_CUDA_TF32"),
    ("cuda", "TURBO_CUDA_F16_ACCUMULATE"),
    ("cuda", "TURBO_CUDA_GELU"),
    ("cuda", "TURBO_CUDA_RESIDUAL"),
    ("cuda", "TURBO_CUDA_PRODUCT"),
    ("cuda", "TURBO_AUTOTUNE"),
    ("cuda", "TURBO_AUTOTUNE_BUDGET_MS"),
    ("cuda", "TURBO_CUDA_TUNED"),
    ("cuda", "TURBO_CUDA_CHOICES"),
    ("levelzero", "TURBO_AUTOTUNE"),
    ("levelzero", "TURBO_AUTOTUNE_BUDGET_MS"),
    ("levelzero", "TURBO_LEVELZERO_TUNED"),
    ("levelzero", "TURBO_LEVELZERO_CHOICES"),
    ("metal", "TURBO_AUTOTUNE"),
    ("metal", "TURBO_AUTOTUNE_BUDGET_MS"),
    ("metal", "TURBO_METAL_TUNED"),
    ("metal", "TURBO_METAL_CHOICES"),
    ("npu", "TURBO_NPU_GRAPH_FORMAT"),
    ("npu", "TURBO_NPU_GRAPH_INPUT"),
];

/// The reason a cell without any record for it gives.
pub const NO_RECORD: &str = "no benchmark record for this cell";

/// The reference programs a record may name, each with its role: the
/// vendors' fastest kernel paths (TensorRT for NVIDIA GPUs, OpenVINO for
/// Intel GPUs and for the NPU, HailoRT for Hailo, MLX for Apple GPUs) and
/// the embedding server.
pub const REFERENCES: [(&str, &str); 5] = [
    ("text-embeddings-inference", "end_to_end"),
    ("tensorrt", "kernel"),
    ("openvino", "kernel"),
    ("hailort", "kernel"),
    ("mlx", "kernel"),
];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Record {
    pub record_version: u32,
    /// When the measurement finished, UTC, `YYYY-MM-DDTHH:MM:SSZ`.
    pub recorded_at: String,
    pub machine: Machine,
    pub device: Device,
    pub library: Library,
    /// `TASK_EMBED`.
    pub task: String,
    /// `PRECISION_MODEL`, `PRECISION_FASTEST` or `PRECISION_EXACT`: what the
    /// session asked for.
    pub precision: String,
    /// `DTYPE_F32`, `DTYPE_F16`, `DTYPE_BF16` or `DTYPE_I8`: what it
    /// computed in, as turbo_session_get_info reported it.
    pub compute_dtype: String,
    pub bundle: BundleId,
    pub rows: Rows,
    pub timing: Timing,
    pub conformance: Conformance,
    /// Every reference program the tool knows for this backend, measured
    /// or with the reason it was not.
    pub references: Vec<ReferenceRun>,
    /// timing.p50_ms over the p50 of the fastest measured reference; null
    /// when none was measured.
    pub speed_ratio: Option<f64>,
    /// The name of that reference.
    pub speed_reference: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Machine {
    /// turbo_device_info.arch of the measured device: the label the record
    /// is filed under.
    pub arch: String,
    /// The host processor's name, as the CPU device reports it, for
    /// context; empty when this build lists no CPU device.
    pub host_cpu: String,
    /// The operating system the tool was built for.
    pub os: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Device {
    /// turbo_device_info.backend.
    pub backend: String,
    /// `DEVICE_CPU`, `DEVICE_GPU`, `DEVICE_IGPU` or `DEVICE_NPU`.
    pub kind: String,
    pub name: String,
    pub vendor: String,
    pub driver_version: String,
    pub runtime_version: String,
    pub memory_total: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Library {
    /// The version number of turbo_version(): `0.1.0`.
    pub version: String,
    /// turbo_version() whole: the version and the backends linked.
    pub build: String,
    /// The commit the library was built from, 40 hex.
    pub commit: String,
    /// The branches of origin that contain the commit, as the tree's
    /// remote-tracking refs showed them when the record was made.
    pub pushed_to: Vec<String>,
    /// Each of LIBRARY_VARS for the backend that was set when the record
    /// was made, as `NAME=value`, in that order; empty when none was, or
    /// in a record made before the field was.
    #[serde(default)]
    pub settings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleId {
    pub model_id: String,
    pub revision: String,
    pub manifest_sha256: String,
    pub artifact_sha256: String,
    pub tokenizer_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rows {
    /// `ROWS_MIXED`: the reference cases that fit seq, cycled, each
    /// padded; `ROWS_DENSE`: every row a case of at least seq tokens, cut
    /// to seq the way the bundle truncates, so no token is padding. A
    /// record made before rows had a kind is mixed, the only kind then.
    /// A mixed pass whose frame batch is 1 lists every fitting case,
    /// each its own `[1, seq]` frame, so `cases` is longer than `batch`.
    #[serde(default = "rows_mixed")]
    pub kind: String,
    pub batch: u32,
    pub seq: u32,
    /// Mask entries of 1 across the measured rows.
    pub live_tokens: u64,
    /// Which reference case each measured row is, in row order. One per
    /// row of the single batch, or one per frame when `batch` is 1 and
    /// the pass cycles every fitting case.
    pub cases: Vec<u32>,
    /// docs/benchmarks.md, "Token rows": the hash of exactly what was
    /// written.
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Timing {
    /// Runs made and not timed.
    pub warmup: u32,
    /// Runs timed.
    pub iterations: u32,
    pub p50_ms: f64,
    pub p99_ms: f64,
    pub mean_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    /// Rows embedded per second over the timed runs.
    pub rows_per_second: f64,
    /// Token positions the library computed per run: each row's through
    /// its last live token, since its backends pack the rows and skip the
    /// padding after them. npu and hailo count every position of each compiled
    /// frame in the pass (one frame is batch x seq; a frame of batch 1
    /// that cycles cases counts one frame per case). Beside
    /// rows.live_tokens, and a reference's computed_tokens, so a padded
    /// and a packed time are not read as the same work. Null only in a
    /// record made before the field was.
    #[serde(default)]
    pub computed_tokens: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Conformance {
    /// Vectors compared with the bundle's fp32 reference.
    pub rows: u32,
    pub min_cosine: f64,
    pub max_abs_diff: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceRun {
    /// `tensorrt`, `openvino` or `text-embeddings-inference` (REFERENCES).
    pub name: String,
    /// `kernel` for tensorrt and openvino, `end_to_end` for
    /// text-embeddings-inference: what it is compared on.
    pub role: String,
    /// The container image, as `name@sha256:<64 hex>`; empty only when it
    /// was not run and none was named.
    pub pinned: String,
    /// The version the program reported of itself; empty when it did not run.
    pub version: String,
    /// Every external command the tool ran for it, each as its argv.
    pub commands: Vec<Vec<String>>,
    /// What the tool did around the commands, in a line.
    pub procedure: String,
    pub measured: Option<Measured>,
    /// Why it was not measured; null when it was.
    pub not_run: Option<String>,
    /// text-embeddings-inference on a GPU, profiled: its requests' time
    /// in GPU kernels, which no header of its gives. Absent when it was
    /// not profiled, and in a record made before the field was.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kernel_time: Option<KernelTime>,
}

/// The GPU time of a reference's requests, as a profiler traced them.
/// Each profiled request is sent alone, with the GPU left idle before the
/// next, so a request's kernels are the trace's activity between two
/// idle gaps.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KernelTime {
    /// The profiler as it named itself (`nsys --version`).
    pub profiler: String,
    /// Requests profiled.
    pub requests: u64,
    /// Per request, the kernels' durations summed.
    pub p50_ms: f64,
    pub p99_ms: f64,
    /// Per request, the memory copies' and sets' durations summed.
    pub copy_p50_ms: f64,
    /// Per request, the kernels launched (p50).
    pub kernels_p50: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Measured {
    /// Timed runs, as the program or the tool counted them.
    pub iterations: u64,
    pub p50_ms: f64,
    pub p99_ms: f64,
    pub rows_per_second: f64,
    /// Its lowest cosine against the bundle's reference, when its vectors
    /// were seen; null when the program does not return them.
    pub min_cosine: Option<f64>,
    /// Token positions it computed per run: every padded row of the pass
    /// for a kernel on the static shape (one frame is batch x seq; a
    /// batch-1 cycle is one frame per case); null when that cannot be
    /// known from outside it, or in a record made before the field was.
    #[serde(default)]
    pub computed_tokens: Option<u64>,
}

/// What a compute dtype must reach against the fp32 reference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tolerance {
    pub min_cosine: f64,
    /// None: not bounded.
    pub max_abs_diff: Option<f64>,
}

/// The tolerance for a TURBO_DTYPE_* value, or None where there is no
/// floor yet.
pub fn tolerance(dtype: u32) -> Option<Tolerance> {
    match dtype {
        TURBO_DTYPE_F32 => Some(Tolerance { min_cosine: 0.9999, max_abs_diff: Some(1e-4) }),
        TURBO_DTYPE_F16 | TURBO_DTYPE_BF16 => Some(Tolerance { min_cosine: 0.999, max_abs_diff: None }),
        crate::TURBO_DTYPE_I8 => Some(Tolerance { min_cosine: 0.93, max_abs_diff: None }),
        _ => None,
    }
}

/// A TURBO_DTYPE_* value as a record names it.
pub fn dtype_name(dtype: u32) -> Option<&'static str> {
    match dtype {
        TURBO_DTYPE_F32 => Some("DTYPE_F32"),
        TURBO_DTYPE_F16 => Some("DTYPE_F16"),
        TURBO_DTYPE_BF16 => Some("DTYPE_BF16"),
        crate::TURBO_DTYPE_I8 => Some("DTYPE_I8"),
        _ => None,
    }
}

/// The TURBO_DTYPE_* value of a record's name.
fn dtype_value(name: &str) -> Option<u32> {
    [TURBO_DTYPE_F32, TURBO_DTYPE_F16, TURBO_DTYPE_BF16, crate::TURBO_DTYPE_I8]
        .into_iter()
        .find(|&d| dtype_name(d) == Some(name))
}

pub fn precision_name(precision: u32) -> Option<&'static str> {
    match precision {
        TURBO_PRECISION_MODEL => Some("PRECISION_MODEL"),
        TURBO_PRECISION_FASTEST => Some("PRECISION_FASTEST"),
        TURBO_PRECISION_EXACT => Some("PRECISION_EXACT"),
        _ => None,
    }
}

pub fn task_name(task: u32) -> Option<&'static str> {
    (task == TURBO_TASK_EMBED).then_some("TASK_EMBED")
}

/// The version number of this build, which records must name.
pub fn library_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

mod decisions;
mod names;
#[cfg(test)]
mod tests;
mod timing;
mod validate;
pub use decisions::{Cell, Verdict, decide, decide_embedded, decide_embedded_for, embedded, pinned};
pub use names::file_name;
#[cfg(test)]
use names::slug;
use names::{is_hex, setting};
pub use timing::{fastest, speed};
use timing::{pass_rows, positive};
