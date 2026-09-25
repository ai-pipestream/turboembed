//! TensorRT, the kernel reference: trtexec in NVIDIA's TensorRT container,
//! pinned by digest, building an engine from the ONNX file the bundle
//! carries and timing it on the same token rows, on the same GPU.
//!
//! The library never executes ONNX; a reference program may. A bundle
//! with no FORMAT_ONNX artifact gives a record that says trtexec could
//! not run. The ONNX graph is the encoder as exported upstream: it stops
//! at the hidden states, so trtexec's time has no pooling or
//! normalization in it, and its inputs are named and typed as the export
//! made them (the tool's options say which).
//!
//! F32 builds from the upstream graph with TF32 off. F16 and BF16 build
//! from the bundle's graph converted to that dtype, with --stronglyTyped:
//! TensorRT 11 removed weak typing, and --fp16 and --bf16 with it, and a
//! strongly typed build is the same on TensorRT 10.

use std::fs;
use std::path::{Path, PathBuf};

use turbo::bundle::Bundle;
use turbo::manifest::Dtype;
use turbo::record::{Measured, ReferenceRun};
use turbo::{TURBO_DTYPE_BF16, TURBO_DTYPE_F16, TURBO_DTYPE_F32};

use crate::docker::{self, Log, Ran, argv};
use crate::measure::{Measurement, Rows};
use crate::{Result, onnx};

pub const NAME: &str = "tensorrt";

#[derive(Debug, Clone)]
pub struct TensorRt {
    /// `name@sha256:<64 hex>`.
    pub image: String,
    /// trtexec inside the image.
    pub trtexec: String,
    /// The ONNX inputs for ids, mask and types, in that order.
    pub inputs: [String; 3],
    /// `int64` or `int32`: the element type of those inputs.
    pub input_dtype: String,
    /// trtexec's --warmUp, in milliseconds.
    pub warmup_ms: u32,
    /// Where the input files are written for the container to read.
    pub work: PathBuf,
}

/// The ONNX graph trtexec builds for a compute dtype, as its artifact's
/// compute_dtype (None: the upstream graph), and trtexec's precision
/// flags: F32 from the upstream graph with TF32 off, as the library
/// computes; F16 and BF16 from the graph converted to that dtype, strongly
/// typed, so every layer runs in the type the graph gives it.
pub fn precision(compute_dtype: u32) -> std::result::Result<(Option<Dtype>, Vec<String>), String> {
    match compute_dtype {
        TURBO_DTYPE_F32 => Ok((None, argv(&["--noTF32"]))),
        TURBO_DTYPE_F16 => Ok((Some(Dtype::F16), argv(&["--stronglyTyped"]))),
        TURBO_DTYPE_BF16 => Ok((Some(Dtype::Bf16), argv(&["--stronglyTyped"]))),
        d => Err(format!("trtexec has no build for compute dtype {d}")),
    }
}

/// The ONNX input names, each of `[A-Za-z0-9_.]+` and neither `.` nor
/// `..` (onnx::check_inputs).
pub fn check_inputs(inputs: &[String]) -> Result<()> {
    onnx::check_inputs("--tensorrt-inputs", inputs)
}

/// The rows as the raw little-endian files --loadInputs reads, in `dtype`.
pub fn input_bytes(values: &[i32], dtype: &str) -> Result<Vec<u8>> {
    onnx::input_bytes("--tensorrt-input-dtype", values, dtype)
}

/// The `docker run` of trtexec: no network, no pulls, the bundle and the
/// inputs mounted read-only.
#[allow(clippy::too_many_arguments)]
pub fn run_argv(
    t: &TensorRt,
    bundle: &Path,
    work: &Path,
    onnx: &str,
    gpu: u32,
    rows: &Rows,
    iterations: u32,
    precision: &[String],
) -> Vec<String> {
    let shape = format!("{}x{}", rows.batch, rows.seq);
    let shapes: Vec<String> = t.inputs.iter().map(|n| format!("{n}:{shape}")).collect();
    let loads: Vec<String> = t.inputs.iter().map(|n| format!("{n}:/work/{n}.bin")).collect();
    let mut a = argv(&[
        "docker",
        "run",
        "--rm",
        "--pull",
        "never",
        "--network",
        "none",
        "--gpus",
        &format!("device={gpu}"),
        "--mount",
        &format!("type=bind,src={},dst=/bundle,readonly", bundle.display()),
        "--mount",
        &format!("type=bind,src={},dst=/work,readonly", work.display()),
        &t.image,
        &t.trtexec,
        &format!("--onnx=/bundle/{onnx}"),
        &format!("--shapes={}", shapes.join(",")),
        &format!("--loadInputs={}", loads.join(",")),
        &format!("--warmUp={}", t.warmup_ms),
        &format!("--iterations={iterations}"),
        "--duration=0",
        "--percentile=99",
    ]);
    a.extend(precision.iter().cloned());
    a
}

/// What trtexec's performance summary says.
#[derive(Debug, Clone, PartialEq)]
pub struct Summary {
    pub version: String,
    /// Queries timed; each query is the whole batch.
    pub queries: u64,
    pub qps: f64,
    /// Latency: H2D, GPU compute and D2H of one query, in ms.
    pub latency_median: f64,
    pub latency_p99: f64,
    pub compute_median: f64,
    pub compute_p99: f64,
}

/// `name = <value> ms` in a summary line.
fn value(line: &str, name: &str) -> Option<f64> {
    let at = line.find(&format!("{name} = "))? + name.len() + 3;
    line[at..].split_whitespace().next()?.trim_end_matches(',').parse().ok()
}

/// The line whose text after its `[I] ` tag starts with `tag`.
fn line<'a>(out: &'a str, tag: &str) -> Option<&'a str> {
    out.lines().find(|l| l.split_once("[I] ").is_some_and(|(_, rest)| rest.starts_with(tag)))
}

/// Parse trtexec's output. It must end in `&&&& PASSED`, name its
/// TensorRT version, and give the summary's Throughput, Latency and GPU
/// Compute Time lines with median and percentile(99%), and the number of
/// queries timed.
pub fn parse(out: &str) -> Result<Summary> {
    if !out.lines().any(|l| l.starts_with("&&&& PASSED")) {
        return Err("trtexec did not report &&&& PASSED".into());
    }
    let missing = |what: &str| format!("trtexec output has no {what}");
    let version = line(out, "TensorRT version: ")
        .and_then(|l| l.rsplit("TensorRT version: ").next())
        .map(|v| v.trim().to_owned())
        .ok_or_else(|| missing("TensorRT version line"))?;
    let queries = line(out, "Timing trace has ")
        .and_then(|l| l.split("Timing trace has ").nth(1)?.split_whitespace().next()?.parse().ok())
        .ok_or_else(|| missing("Timing trace line"))?;
    let qps = line(out, "Throughput: ")
        .and_then(|l| l.split("Throughput: ").nth(1)?.split_whitespace().next()?.parse().ok())
        .ok_or_else(|| missing("Throughput line"))?;
    let latency = line(out, "Latency: ").ok_or_else(|| missing("Latency line"))?;
    let compute = line(out, "GPU Compute Time: ").ok_or_else(|| missing("GPU Compute Time line"))?;
    let get = |l: &str, n: &str| value(l, n).ok_or_else(|| format!("trtexec line {l:?} has no {n}"));
    Ok(Summary {
        version,
        queries,
        qps,
        latency_median: get(latency, "median")?,
        latency_p99: get(latency, "percentile(99%)")?,
        compute_median: get(compute, "median")?,
        compute_p99: get(compute, "percentile(99%)")?,
    })
}

fn not_run(image: &str, log: Log, procedure: &str, why: String) -> ReferenceRun {
    ReferenceRun {
        name: NAME.into(),
        role: "kernel".into(),
        pinned: image.into(),
        version: String::new(),
        commands: log.commands,
        procedure: procedure.into(),
        measured: None,
        not_run: Some(why),
    }
}

/// The bundle's ONNX file for the session's compute dtype, or why there
/// is none to run.
pub fn onnx_file(m: &Measurement) -> std::result::Result<String, String> {
    let (dtype, _) = precision(m.compute_dtype)?;
    onnx::file(&m.manifest, dtype, "for trtexec to build an engine from")
}

/// The tag trtexec's error lines carry.
pub const ERROR_TAGS: [&str; 1] = ["[E] "];

/// Build and time the engine on `gpu`, the device's CUDA ordinal. A thing
/// trtexec cannot do for this bundle, trtexec failing to build or run the
/// engine included, is a record that says so, with trtexec's first error
/// line; a failure of docker is an error.
pub fn run(t: &TensorRt, m: &Measurement, gpu: u32, iterations: u32) -> Result<ReferenceRun> {
    let image = docker::check_pinned("--tensorrt-image", &t.image)?;
    let procedure = format!(
        "trtexec builds an engine from the bundle's ONNX file and times {iterations} queries of the batch's \
         [{}, {}] rows loaded from files; p50 and p99 are its Latency (H2D, GPU compute, D2H) median and \
         percentile(99%)",
        m.rows.batch, m.rows.seq
    );
    let log = Log::default();
    let onnx = match onnx_file(m) {
        Ok(f) => f,
        Err(why) => return Ok(not_run(image, log, &procedure, why)),
    };
    let precision = match precision(m.compute_dtype) {
        Ok((_, flags)) => flags,
        Err(why) => return Ok(not_run(image, log, &procedure, why)),
    };
    let procedure = format!("{procedure}; the engine is built from {onnx} with {}", precision.join(" "));
    // The file the manifest lists, checked against its hash.
    Bundle::open(&m.bundle_dir).and_then(|b| b.read_verified(&onnx)).map_err(|e| e.message)?;
    let mut log = log;
    docker::require_image(&mut log, image)?;

    check_inputs(&t.inputs)?;
    let work = t.work.join(format!("turbo-bench-trtexec-{}", std::process::id()));
    fs::create_dir_all(&work).map_err(|e| format!("{}: {e}", work.display()))?;
    let result = (|| {
        onnx::write_inputs(&work, &t.inputs, &t.input_dtype, "--tensorrt-input-dtype", &m.rows)?;
        let work = fs::canonicalize(&work).map_err(|e| format!("{}: {e}", work.display()))?;
        let cmd = run_argv(t, &m.bundle_dir, &work, &onnx, gpu, &m.rows, iterations, &precision);
        let shown = run_argv(
            t,
            Path::new(docker::BUNDLE),
            Path::new(docker::WORK),
            &onnx,
            gpu,
            &m.rows,
            iterations,
            &precision,
        );
        match log.run_program(&cmd, shown, &ERROR_TAGS)? {
            Ran::Done(out) => parse(&out).map(Ok),
            Ran::Failed(why) => Ok(Err(format!("trtexec {why}"))),
        }
    })();
    let _ = fs::remove_dir_all(&work);
    let s = match result? {
        Ok(s) => s,
        Err(why) => return Ok(not_run(image, log, &procedure, why)),
    };
    Ok(ReferenceRun {
        name: NAME.into(),
        role: "kernel".into(),
        pinned: image.into(),
        version: s.version,
        commands: log.commands,
        procedure: format!(
            "{procedure}; its GPU Compute Time was median {} ms, percentile(99%) {} ms",
            s.compute_median, s.compute_p99
        ),
        measured: Some(Measured {
            iterations: s.queries,
            p50_ms: s.latency_median,
            p99_ms: s.latency_p99,
            rows_per_second: s.qps * m.rows.batch as f64,
            min_cosine: None,
            computed_tokens: Some(m.rows.padded_tokens()),
        }),
        not_run: None,
    })
}
