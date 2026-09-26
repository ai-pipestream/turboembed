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
pub const HOST_PATHS: [&str; 4] = ["/home/", "/root/", "/var/home/", "/Users/"];

/// The kinds of token rows a record may be measured on.
pub const ROWS_MIXED: &str = "ROWS_MIXED";
pub const ROWS_DENSE: &str = "ROWS_DENSE";

fn rows_mixed() -> String {
    ROWS_MIXED.into()
}

/// The environment variables that change what a backend runs, each with
/// its backend: a record names those that were set in library.settings.
///
/// TURBO_CUDA_TUNED and TURBO_CUDA_CHOICES are not read from the
/// environment: they are the session's turbo_session_info.tuned and
/// choices, present in every record of a backend that reports choices, so
/// a record made with nothing set still says which kernels ran, and
/// TURBO_CUDA_CHOICES set to that string forces them back.
pub const LIBRARY_VARS: [(&str, &str); 16] = [
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
];

/// The reason a cell without any record for it gives.
pub const NO_RECORD: &str = "no benchmark record for this cell";

/// The reference programs a record may name, each with its role: the
/// vendors' fastest kernel paths (TensorRT for NVIDIA GPUs, OpenVINO for
/// Intel GPUs) and the fastest known embedding server.
pub const REFERENCES: [(&str, &str); 3] =
    [("text-embeddings-inference", "end_to_end"), ("tensorrt", "kernel"), ("openvino", "kernel")];

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
    #[serde(default = "rows_mixed")]
    pub kind: String,
    pub batch: u32,
    pub seq: u32,
    /// Mask entries of 1 across the batch.
    pub live_tokens: u64,
    /// Which reference case each row is, in row order.
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
    /// padding after them. Beside rows.live_tokens, and a reference's
    /// computed_tokens, so a padded and a packed time are not read as the
    /// same work. Null only in a record made before the field was.
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
    /// Token positions it computed per run: batch x seq for a kernel on
    /// the padded rows; null when that cannot be known from outside it,
    /// or in a record made before the field was.
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

/// Lower-case ASCII letters and digits, anything else one `-`, none at
/// either end.
fn slug(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.is_empty() && !out.ends_with('-') {
            out.push('-');
        }
    }
    out.trim_end_matches('-').to_owned()
}

/// A record's file name, from its contents alone:
///
/// `<machine>.<backend>.<task>.<precision>[-dense].<model>-<manifest>.<commit>.json`
///
/// machine is the arch label, and for a CPU the arch label and the first 8
/// hex of the SHA-256 of the processor's name, since a CPU record is filed
/// under both; task and precision are the enum names without their
/// prefix; model is the last part of the model id, at most 32 bytes;
/// manifest is the first 8 hex of the bundle's manifest hash, commit the
/// first 12 of the library's; `-dense` marks dense rows, so a mixed and a
/// dense record of one commit are both kept. Longer than NAME_MAX is an
/// error.
pub fn file_name(r: &Record) -> Result<String, String> {
    let mut machine = slug(&r.machine.arch);
    if r.device.kind == "DEVICE_CPU" {
        machine = format!("{machine}-{}", &sha256_hex(r.device.name.as_bytes())[..8]);
    }
    let bare = |s: &str, prefix: &str| slug(s.strip_prefix(prefix).unwrap_or(s));
    let mut model = slug(r.bundle.model_id.rsplit('/').next().unwrap_or(""));
    model.truncate(32);
    let model = model.trim_end_matches('-');
    let manifest = r.bundle.manifest_sha256.get(..8).unwrap_or("");
    let commit = r.library.commit.get(..12).unwrap_or("");
    let dense = if r.rows.kind == ROWS_DENSE { "-dense" } else { "" };
    let name = format!(
        "{machine}.{}.{}.{}{dense}.{model}-{manifest}.{commit}.json",
        slug(&r.device.backend),
        bare(&r.task, "TASK_"),
        bare(&r.precision, "PRECISION_"),
    );
    if name.len() > NAME_MAX {
        return Err(format!("the record's name {name} is {} bytes, over {NAME_MAX}", name.len()));
    }
    Ok(name)
}

fn is_hex(s: &str, len: usize) -> bool {
    s.len() == len && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Finite and above zero.
fn positive(v: f64) -> bool {
    v.is_finite() && v > 0.0
}

/// The fastest measured reference: its name and p50.
pub fn fastest(references: &[ReferenceRun]) -> Option<(&str, f64)> {
    references
        .iter()
        .filter_map(|r| r.measured.as_ref().map(|m| (r.name.as_str(), m.p50_ms)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
}

/// speed_ratio and speed_reference as the references give them.
pub fn speed(p50_ms: f64, references: &[ReferenceRun]) -> (Option<f64>, Option<String>) {
    match fastest(references) {
        Some((name, p50)) => (Some(p50_ms / p50), Some(name.to_owned())),
        None => (None, None),
    }
}

impl Record {
    /// Parse a record read from `name`, and check that it is well formed:
    /// its name is the one its contents give, every hash and number is
    /// one a measurement could have produced, each reference was either
    /// measured or says why not, speed_ratio is what the references
    /// give, and no text in it is a path under a home directory.
    pub fn parse(name: &str, bytes: &[u8]) -> Result<Record, String> {
        let r: Record = serde_json::from_slice(bytes).map_err(|e| format!("{name}: {e}"))?;
        r.check().map_err(|e| format!("{name}: {e}"))?;
        let want = file_name(&r).map_err(|e| format!("{name}: {e}"))?;
        if name != want {
            return Err(format!("{name}: its contents name it {want}"));
        }
        Ok(r)
    }

    pub fn check(&self) -> Result<(), String> {
        if self.record_version != RECORD_VERSION {
            return Err(format!("record_version {} is not {RECORD_VERSION}", self.record_version));
        }
        let t = self.recorded_at.as_bytes();
        let shape = "dddd-dd-ddTdd:dd:ddZ".as_bytes();
        if t.len() != shape.len()
            || !t.iter().zip(shape).all(|(c, s)| if *s == b'd' { c.is_ascii_digit() } else { c == s })
        {
            return Err(format!("recorded_at {:?} is not YYYY-MM-DDTHH:MM:SSZ", self.recorded_at));
        }
        if self.machine.arch.is_empty() || self.device.backend.is_empty() {
            return Err("machine.arch and device.backend must not be empty".into());
        }
        if !["DEVICE_CPU", "DEVICE_GPU", "DEVICE_IGPU", "DEVICE_NPU"].contains(&self.device.kind.as_str()) {
            return Err(format!("device.kind {:?} is not a DEVICE_* value", self.device.kind));
        }
        if !is_hex(&self.library.commit, 40) {
            return Err(format!("library.commit {:?} is not 40 lowercase hex", self.library.commit));
        }
        if self.library.pushed_to.is_empty() {
            return Err("library.pushed_to is empty: the commit was on no branch of origin".into());
        }
        if self.library.version.is_empty() || !self.library.build.starts_with(&self.library.version) {
            return Err(format!(
                "library.build {:?} does not start with {:?}",
                self.library.build, self.library.version
            ));
        }
        let vars: Vec<&str> = LIBRARY_VARS.iter().filter(|(b, _)| *b == self.device.backend).map(|&(_, v)| v).collect();
        let mut at = 0;
        for s in &self.library.settings {
            let name = s.split_once('=').map(|(n, _)| n);
            match name.and_then(|n| vars[at..].iter().position(|v| *v == n)) {
                Some(i) => at += i + 1,
                None => {
                    return Err(format!(
                        "library.settings {s:?} is not NAME=value, NAME one of {vars:?} for {}, each once and in \
                         that order",
                        self.device.backend
                    ));
                }
            }
        }
        if task_name(TURBO_TASK_EMBED) != Some(self.task.as_str()) {
            return Err(format!("task {:?} is not TASK_EMBED", self.task));
        }
        if ![TURBO_PRECISION_MODEL, TURBO_PRECISION_FASTEST, TURBO_PRECISION_EXACT]
            .iter()
            .any(|&p| precision_name(p) == Some(self.precision.as_str()))
        {
            return Err(format!("precision {:?} is not a PRECISION_* value", self.precision));
        }
        if dtype_value(&self.compute_dtype).is_none() {
            return Err(format!("compute_dtype {:?} is not a DTYPE_* value", self.compute_dtype));
        }
        let b = &self.bundle;
        for (what, h) in
            [("manifest", &b.manifest_sha256), ("artifact", &b.artifact_sha256), ("tokenizer", &b.tokenizer_sha256)]
        {
            if !is_hex(h, 64) {
                return Err(format!("bundle.{what}_sha256 {h:?} is not 64 lowercase hex"));
            }
        }
        if b.model_id.is_empty() {
            return Err("bundle.model_id is empty".into());
        }
        let rows = &self.rows;
        if rows.batch == 0 || rows.seq == 0 || rows.cases.len() != rows.batch as usize || !is_hex(&rows.sha256, 64) {
            return Err("rows: batch and seq must be above 0, cases one per row, sha256 64 lowercase hex".into());
        }
        if rows.live_tokens < rows.batch as u64 || rows.live_tokens > rows.batch as u64 * rows.seq as u64 {
            return Err(format!("rows.live_tokens {} does not fit {} x {}", rows.live_tokens, rows.batch, rows.seq));
        }
        match rows.kind.as_str() {
            ROWS_MIXED => {}
            ROWS_DENSE if rows.live_tokens == rows.batch as u64 * rows.seq as u64 => {}
            ROWS_DENSE => {
                return Err(format!(
                    "rows: dense, yet {} of {} tokens are live",
                    rows.live_tokens,
                    rows.batch as u64 * rows.seq as u64
                ));
            }
            k => return Err(format!("rows.kind {k:?} is not {ROWS_MIXED} or {ROWS_DENSE}")),
        }
        let slots = rows.live_tokens..=rows.batch as u64 * rows.seq as u64;
        if let Some(n) = self.timing.computed_tokens
            && !slots.contains(&n)
        {
            return Err(format!(
                "timing.computed_tokens {n} is not between the live tokens and batch x seq, {slots:?}"
            ));
        }
        let t = &self.timing;
        if t.iterations == 0
            || ![t.p50_ms, t.p99_ms, t.mean_ms, t.min_ms, t.max_ms, t.rows_per_second].into_iter().all(positive)
            || !(t.min_ms <= t.p50_ms && t.p50_ms <= t.p99_ms && t.p99_ms <= t.max_ms)
        {
            return Err("timing: iterations above 0, every figure finite and above 0, min <= p50 <= p99 <= max".into());
        }
        let c = &self.conformance;
        if c.rows == 0
            || !(c.min_cosine.is_finite() && (-1.0..=1.0 + 1e-6).contains(&c.min_cosine))
            || !(c.max_abs_diff.is_finite() && c.max_abs_diff >= 0.0)
        {
            return Err("conformance: rows above 0, min_cosine in [-1, 1], max_abs_diff finite and not negative".into());
        }
        for r in &self.references {
            if !REFERENCES.contains(&(r.name.as_str(), r.role.as_str())) {
                return Err(format!("reference {:?} with role {:?} is not one of {REFERENCES:?}", r.name, r.role));
            }
            if let Some(m) = &r.measured
                && let Some(n) = m.computed_tokens
                && !slots.contains(&n)
            {
                return Err(format!(
                    "reference {}: computed_tokens {n} is not between the live tokens and batch x seq, {slots:?}",
                    r.name
                ));
            }
            if let Some(m) = &r.measured
                && let Some(c) = m.min_cosine
                && !(c.is_finite() && (-1.0..=1.0 + 1e-6).contains(&c))
            {
                return Err(format!("reference {}: min_cosine {c} is not in [-1, 1]", r.name));
            }
            match (&r.measured, &r.not_run) {
                (Some(m), None) => {
                    if m.iterations == 0
                        || ![m.p50_ms, m.p99_ms, m.rows_per_second].into_iter().all(positive)
                        || m.p99_ms < m.p50_ms
                        || r.version.is_empty()
                        || r.commands.is_empty()
                    {
                        return Err(format!(
                            "reference {}: a measurement has iterations, a version, its commands, \
                             and figures above 0 with p50 <= p99",
                            r.name
                        ));
                    }
                }
                (None, Some(why)) if !why.is_empty() => {}
                _ => return Err(format!("reference {}: exactly one of measured and not_run", r.name)),
            }
            // A program disabled before an image was named has none.
            if (r.measured.is_some() || !r.pinned.is_empty()) && pinned(&r.pinned).is_none() {
                return Err(format!("reference {}: {:?} is not name@sha256:<64 hex>", r.name, r.pinned));
            }
        }
        let text = serde_json::to_string(self).map_err(|e| e.to_string())?;
        if let Some(home) = HOST_PATHS.iter().find(|h| text.contains(*h)) {
            return Err(format!(
                "the record holds a host path ({home}...): a path on the machine that made it is written as a \
                 placeholder"
            ));
        }
        let (ratio, by) = speed(t.p50_ms, &self.references);
        let same = match (ratio, self.speed_ratio) {
            (Some(a), Some(b)) => (a - b).abs() <= 1e-12 * a.abs(),
            (None, None) => true,
            _ => false,
        };
        if !same || by != self.speed_reference {
            return Err(format!(
                "speed_ratio {:?} by {:?} is not what the references give, {ratio:?} by {by:?}",
                self.speed_ratio, self.speed_reference
            ));
        }
        Ok(())
    }
}

/// The digest of a container image pinned as `name@sha256:<64 hex>`,
/// where the name is lower-case letters, digits and `._/:-`, starting
/// with a letter or digit (so it can never be read as an option).
pub fn pinned(image: &str) -> Option<&str> {
    let (name, digest) = image.split_once("@sha256:")?;
    let mut b = name.bytes();
    let first = b.next()?;
    let ok = |c: u8| c.is_ascii_lowercase() || c.is_ascii_digit();
    (ok(first) && b.all(|c| ok(c) || b"._/:-".contains(&c)) && is_hex(digest, 64)).then_some(digest)
}

/// One capability cell as the core asks about it.
#[derive(Debug, Clone, Copy)]
pub struct Cell<'a> {
    /// turbo_device_info.arch.
    pub arch: &'a str,
    /// turbo_device_info.name: part of the key for a CPU.
    pub name: &'a str,
    pub cpu: bool,
    pub backend: &'a str,
    pub task: u32,
    pub precision: u32,
    /// The compute dtype the backend says the precision resolves to.
    pub dtype: u32,
    /// This build's version number.
    pub version: &'a str,
    /// The operating system this build is for, std::env::consts::OS.
    pub os: &'a str,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    Supported {
        benchmark: String,
        cosine_floor: f64,
        speed_ratio: f64,
    },
    /// Why not: no record, or what the best one lacks.
    Not(String),
}

impl Record {
    /// Whether the record is for the cell: measured on mixed rows, with
    /// the same arch label, operating system, backend, task and precision,
    /// and for a CPU the same processor. Mixed rows are what a server sees,
    /// so only they back a capability and its speed_ratio; a dense record
    /// is kept and parsed as reference evidence and backs nothing.
    pub fn is_for(&self, cell: &Cell) -> bool {
        self.rows.kind == ROWS_MIXED
            && self.machine.arch == cell.arch
            && self.machine.os == cell.os
            && self.device.backend == cell.backend
            && task_name(cell.task) == Some(self.task.as_str())
            && precision_name(cell.precision) == Some(self.precision.as_str())
            && (!cell.cpu || (self.device.kind == "DEVICE_CPU" && self.device.name == cell.name))
    }

    /// Why a record for the cell does not back SUPPORTED, or None when it
    /// does: it is for this library version, computed in the dtype the
    /// backend resolves the precision to, reached that dtype's tolerance,
    /// and measured at least one reference program.
    pub fn falls_short(&self, cell: &Cell) -> Option<String> {
        if self.library.version != cell.version {
            return Some(format!("recorded with library {}, this is {}", self.library.version, cell.version));
        }
        let Some(tol) = dtype_value(&self.compute_dtype).and_then(tolerance) else {
            return Some(format!("{} has no conformance floor", self.compute_dtype));
        };
        if dtype_value(&self.compute_dtype) != Some(cell.dtype) {
            return Some(format!("computed in {}, the cell computes in dtype {}", self.compute_dtype, cell.dtype));
        }
        let c = &self.conformance;
        if c.min_cosine < tol.min_cosine {
            return Some(format!("min cosine {} is under {}", c.min_cosine, tol.min_cosine));
        }
        if let Some(most) = tol.max_abs_diff
            && c.max_abs_diff > most
        {
            return Some(format!("max abs diff {:e} is over {most:e}", c.max_abs_diff));
        }
        if !self.references.iter().any(|r| r.measured.is_some() && REFERENCES.iter().any(|&(n, _)| n == r.name)) {
            return Some("no reference program measured".into());
        }
        None
    }
}

/// The verdict of `records` on a cell. Of the records for it, those that
/// back SUPPORTED are ranked by recorded_at, then name, and the newest
/// wins. When none backs it, the reason is the newest record's.
pub fn decide<'a>(records: impl IntoIterator<Item = (&'a str, &'a Record)>, cell: &Cell) -> Verdict {
    let mut matching: Vec<(&str, &Record)> = records.into_iter().filter(|(_, r)| r.is_for(cell)).collect();
    matching.sort_by(|a, b| (&a.1.recorded_at, a.0).cmp(&(&b.1.recorded_at, b.0)));
    if let Some((name, r)) = matching.iter().rev().find(|(_, r)| r.falls_short(cell).is_none()) {
        return Verdict::Supported {
            benchmark: (*name).to_owned(),
            cosine_floor: r.conformance.min_cosine,
            speed_ratio: r.speed_ratio.unwrap_or(0.0),
        };
    }
    match matching.last() {
        Some((name, r)) => Verdict::Not(format!("{name}: {}", r.falls_short(cell).unwrap_or_default())),
        None => Verdict::Not(NO_RECORD.into()),
    }
}

// EMBEDDED: every file name and its text, from benchmarks/records/.
include!(concat!(env!("OUT_DIR"), "/records.rs"));

/// The records compiled into this build, each parsed, or why it does not.
pub fn embedded() -> &'static [(&'static str, Result<Record, String>)] {
    static PARSED: OnceLock<Vec<(&'static str, Result<Record, String>)>> = OnceLock::new();
    PARSED.get_or_init(|| EMBEDDED.iter().map(|(name, text)| (*name, Record::parse(name, text.as_bytes()))).collect())
}

/// The verdict of the compiled-in records on a cell. A record that does
/// not parse backs nothing; a test keeps the committed ones parsing.
pub fn decide_embedded(cell: &Cell) -> Verdict {
    decide(embedded().iter().filter_map(|(n, r)| r.as_ref().ok().map(|r| (*n, r))), cell)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_committed_record_parses_under_its_own_name() {
        for (name, r) in embedded() {
            if let Err(e) = r {
                panic!("benchmarks/records/{name}: {e}");
            }
        }
    }

    #[test]
    fn slugs_are_lower_case_letters_digits_and_single_dashes() {
        assert_eq!(slug("all-MiniLM-L6-v2"), "all-minilm-l6-v2");
        assert_eq!(slug("  Intel(R) Xeon(R)  CPU @ 2.10GHz "), "intel-r-xeon-r-cpu-2-10ghz");
        assert_eq!(slug("x86_64"), "x86-64");
    }

    #[test]
    fn the_tolerances_are_the_documented_ones() {
        assert_eq!(tolerance(TURBO_DTYPE_F32), Some(Tolerance { min_cosine: 0.9999, max_abs_diff: Some(1e-4) }));
        assert_eq!(tolerance(TURBO_DTYPE_F16), Some(Tolerance { min_cosine: 0.999, max_abs_diff: None }));
        assert_eq!(tolerance(TURBO_DTYPE_BF16), tolerance(TURBO_DTYPE_F16));
        assert_eq!(tolerance(crate::TURBO_DTYPE_I32), None);
        assert_eq!(dtype_value("DTYPE_I8"), Some(crate::TURBO_DTYPE_I8));
        assert_eq!(tolerance(crate::TURBO_DTYPE_I8), None, "int8 is recorded and has no floor");
    }

    #[test]
    fn an_image_is_pinned_by_digest_under_a_plain_name() {
        let d = "0".repeat(64);
        assert_eq!(pinned(&format!("nvcr.io/nvidia/tensorrt@sha256:{d}")), Some(d.as_str()));
        assert_eq!(pinned(&format!("localhost:5000/tei@sha256:{d}")), Some(d.as_str()));
        for bad in ["-v/:/x", "", "Upper/case", "a b", "a;b", "a@b"] {
            assert_eq!(pinned(&format!("{bad}@sha256:{d}")), None, "{bad:?}");
        }
        assert_eq!(pinned("nvcr.io/nvidia/tensorrt:25.01"), None);
        assert_eq!(pinned(&format!("tei@sha256:{}", "A".repeat(64))), None);
    }
}
