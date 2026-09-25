//! OpenVINO, the kernel reference for Intel GPUs: its `benchmark_app` in
//! an OpenVINO container, pinned by digest, compiling the ONNX file the
//! bundle carries for the GPU and timing it on the same token rows, one
//! request at a time.
//!
//! The library never executes ONNX; a reference program may. A bundle
//! with no FORMAT_ONNX artifact gives a record that says benchmark_app
//! could not run. As for trtexec, the graph stops at the hidden states,
//! so its time has no pooling or normalization in it.
//!
//! benchmark_app reports one latency percentile per run
//! (`-latency_percentile`, the median by default), so the tool runs it
//! twice with the same arguments: once for the median, once for the 99th
//! percentile.

use std::fs;
use std::path::{Path, PathBuf};

use turbo::bundle::Bundle;
use turbo::record::{Measured, ReferenceRun};
use turbo::{TURBO_DTYPE_F16, TURBO_DTYPE_F32};

use crate::docker::{self, Log, argv};
use crate::measure::{Measurement, Rows};
use crate::{Result, onnx};

pub const NAME: &str = "openvino";

/// The percentiles the two runs report.
pub const PERCENTILES: [u32; 2] = [50, 99];

#[derive(Debug, Clone)]
pub struct OpenVino {
    /// `name@sha256:<64 hex>`.
    pub image: String,
    /// benchmark_app inside the image.
    pub benchmark_app: String,
    /// The ONNX inputs for ids, mask and types, in that order.
    pub inputs: [String; 3],
    /// `int64` or `int32`: the element type of those inputs.
    pub input_dtype: String,
    /// The host's DRI directory, handed to the container whole.
    pub dri: PathBuf,
    /// Where the input files are written for the container to read.
    pub work: PathBuf,
}

/// benchmark_app's `-infer_precision` for a compute dtype: the GPU plugin
/// computes in f32 or f16.
pub fn infer_precision(compute_dtype: u32) -> std::result::Result<&'static str, String> {
    match compute_dtype {
        TURBO_DTYPE_F32 => Ok("f32"),
        TURBO_DTYPE_F16 => Ok("f16"),
        d => Err(format!("OpenVINO's GPU plugin has no inference precision for compute dtype {d}")),
    }
}

/// The group that owns the render nodes under `dri`, which the container
/// is added to so it may open them; an error when there is none.
pub fn render_group(dri: &Path) -> Result<u32> {
    let mut nodes: Vec<PathBuf> = fs::read_dir(dri)
        .map_err(|e| format!("{}: {e}", dri.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with("renderD"))
        .map(|e| e.path())
        .collect();
    nodes.sort();
    let node = nodes.first().ok_or_else(|| format!("{}: no render node (renderD*) for the GPU", dri.display()))?;
    gid(node)
}

#[cfg(unix)]
fn gid(path: &Path) -> Result<u32> {
    use std::os::unix::fs::MetadataExt;
    fs::metadata(path).map(|m| m.gid()).map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(not(unix))]
fn gid(path: &Path) -> Result<u32> {
    Err(format!("{}: a render node's group is read on unix only", path.display()))
}

/// The `docker run` of benchmark_app: no network, no pulls, the GPU's
/// DRI nodes with their group, the bundle and the inputs mounted
/// read-only; the ONNX file compiled for the GPU at the rows' static
/// shape, in the compute dtype, with the latency hint, one synchronous
/// request, and `iterations` timed inferences after its warm-up one.
#[allow(clippy::too_many_arguments)]
pub fn run_argv(
    o: &OpenVino,
    bundle: &Path,
    work: &Path,
    onnx: &str,
    render_gid: u32,
    rows: &Rows,
    iterations: u32,
    precision: &str,
    percentile: u32,
) -> Vec<String> {
    let shape = format!("[{},{}]", rows.batch, rows.seq);
    let shapes: Vec<String> = o.inputs.iter().map(|n| format!("{n}{shape}")).collect();
    let files: Vec<String> = o.inputs.iter().map(|n| format!("{n}:/work/{n}.bin")).collect();
    argv(&[
        "docker",
        "run",
        "--rm",
        "--pull",
        "never",
        "--network",
        "none",
        "--device",
        &o.dri.display().to_string(),
        "--group-add",
        &render_gid.to_string(),
        "--mount",
        &format!("type=bind,src={},dst=/bundle,readonly", bundle.display()),
        "--mount",
        &format!("type=bind,src={},dst=/work,readonly", work.display()),
        &o.image,
        &o.benchmark_app,
        "-m",
        &format!("/bundle/{onnx}"),
        "-d",
        "GPU",
        "-hint",
        "latency",
        "-api",
        "sync",
        "-nireq",
        "1",
        "-niter",
        &iterations.to_string(),
        "-shape",
        &shapes.join(","),
        "-i",
        &files.join(","),
        "-infer_precision",
        precision,
        "-latency_percentile",
        &percentile.to_string(),
    ])
}

/// What benchmark_app's report says.
#[derive(Debug, Clone, PartialEq)]
pub struct Report {
    /// The OpenVINO build, from the `Build` line under `OpenVINO:`.
    pub version: String,
    /// Inferences timed; each is the whole batch.
    pub count: u64,
    /// Their wall time, in ms.
    pub duration_ms: f64,
    /// The latency at the run's percentile, in ms.
    pub latency_ms: f64,
    pub average_ms: f64,
    /// As benchmark_app computes it: frames of its batch dimension a
    /// second.
    pub throughput_fps: f64,
}

/// A line's text after benchmark_app's `[ INFO ] ` tag, trimmed.
fn info_lines(out: &str) -> impl Iterator<Item = &str> {
    out.lines().filter_map(|l| l.trim_start().strip_prefix("[ INFO ]")).map(str::trim)
}

/// `<value> ms` or `<value> us`, in ms: benchmark_app prints latencies in
/// microseconds when their average is under a millisecond.
fn latency(text: &str) -> Option<f64> {
    let mut it = text.split_whitespace();
    let v: f64 = it.next()?.parse().ok()?;
    match it.next()? {
        "ms" => Some(v),
        "us" => Some(v / 1000.0),
        _ => None,
    }
}

/// Parse benchmark_app's report of a run with `-latency_percentile
/// <percentile>`: the `Build` line under `OpenVINO:`, `Count:`,
/// `Duration:`, the `Latency:` block's percentile line (`Median:` for 50)
/// and `Average:`, and `Throughput:`. benchmark_app's Python and C++
/// forms both print these, each after an `[ INFO ]` tag.
pub fn parse(out: &str, percentile: u32) -> Result<Report> {
    let missing = |what: &str| format!("benchmark_app output has no {what}");
    let lines: Vec<&str> = info_lines(out).collect();
    let after = |tag: &str| lines.iter().find_map(|l| l.strip_prefix(tag)).map(str::trim);
    let version = lines
        .iter()
        .skip_while(|l| **l != "OpenVINO:")
        .find_map(|l| l.strip_prefix("Build"))
        .map(|v| v.trim_start_matches(['.', ' ', ':']).trim().to_owned())
        .filter(|v| !v.is_empty())
        .ok_or_else(|| missing("Build line under OpenVINO:"))?;
    let count = after("Count:")
        .and_then(|v| v.strip_suffix("iterations")?.trim().parse().ok())
        .ok_or_else(|| missing("Count line"))?;
    let duration_ms = after("Duration:")
        .and_then(|v| v.strip_suffix("ms")?.trim().parse().ok())
        .ok_or_else(|| missing("Duration line"))?;
    let block: Vec<&str> = lines.iter().skip_while(|l| **l != "Latency:").skip(1).take(4).copied().collect();
    let tag = if percentile == 50 { "Median:".to_owned() } else { format!("{percentile} percentile:") };
    let in_block = |tag: &str| block.iter().find_map(|l| l.strip_prefix(tag)).and_then(latency);
    let latency_ms = in_block(&tag).ok_or_else(|| missing(&format!("{tag} line in its Latency block")))?;
    let average_ms = in_block("Average:").ok_or_else(|| missing("Average: line in its Latency block"))?;
    let throughput_fps = after("Throughput:")
        .and_then(|v| v.strip_suffix("FPS")?.trim().parse().ok())
        .ok_or_else(|| missing("Throughput line"))?;
    if count == 0 {
        return Err("benchmark_app timed no inference".into());
    }
    Ok(Report { version, count, duration_ms, latency_ms, average_ms, throughput_fps })
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

/// The bundle's ONNX file, or why there is none to run.
pub fn onnx_file(m: &Measurement) -> std::result::Result<String, String> {
    onnx::file(&m.manifest, "for benchmark_app to compile")
}

/// The measured reference from the median run and the 99th percentile
/// run of the same arguments.
pub fn measured(image: &str, log: Log, procedure: &str, p50: Report, p99: Report, batch: u32) -> Result<ReferenceRun> {
    if p50.version != p99.version || p50.count != p99.count {
        return Err(format!(
            "benchmark_app's two runs differ: OpenVINO {} and {}, {} and {} iterations",
            p50.version, p99.version, p50.count, p99.count
        ));
    }
    if p99.latency_ms < p50.latency_ms {
        return Err(format!(
            "benchmark_app's 99th percentile run gave {} ms, under its median run's {} ms: the two runs are not \
             comparable; run again",
            p99.latency_ms, p50.latency_ms
        ));
    }
    Ok(ReferenceRun {
        name: NAME.into(),
        role: "kernel".into(),
        pinned: image.into(),
        version: p50.version,
        commands: log.commands,
        procedure: format!(
            "{procedure}; the median run took {} ms for {} inferences, average {} ms, and reported {} FPS; \
             the 99th percentile run averaged {} ms and reported {} FPS",
            p50.duration_ms, p50.count, p50.average_ms, p50.throughput_fps, p99.average_ms, p99.throughput_fps
        ),
        measured: Some(Measured {
            iterations: p50.count,
            p50_ms: p50.latency_ms,
            p99_ms: p99.latency_ms,
            rows_per_second: p50.count as f64 * batch as f64 / (p50.duration_ms / 1000.0),
            min_cosine: None,
        }),
        not_run: None,
    })
}

/// Compile the bundle's ONNX file for the GPU and time it, twice. A thing
/// benchmark_app cannot do for this bundle is a record that says so; a
/// failure of docker or of benchmark_app is an error.
pub fn run(o: &OpenVino, m: &Measurement, iterations: u32) -> Result<ReferenceRun> {
    let image = docker::check_pinned("--openvino-image", &o.image)?;
    let procedure = format!(
        "benchmark_app compiles the bundle's ONNX file for the GPU at the batch's static [{}, {}] shape and times \
         {iterations} synchronous inferences of the rows loaded from files, one request, after its one warm-up \
         inference; it runs twice with the same arguments, -latency_percentile 50 then 99, and p50 and p99 are \
         those runs' latencies; rows per second is the median run's count times the batch over its duration",
        m.rows.batch, m.rows.seq
    );
    let log = Log::default();
    let model = match onnx_file(m) {
        Ok(f) => f,
        Err(why) => return Ok(not_run(image, log, &procedure, why)),
    };
    let precision = match infer_precision(m.compute_dtype) {
        Ok(p) => p,
        Err(why) => return Ok(not_run(image, log, &procedure, why)),
    };
    // The file the manifest lists, checked against its hash.
    Bundle::open(&m.bundle_dir).and_then(|b| b.read_verified(&model)).map_err(|e| e.message)?;
    onnx::check_inputs("--openvino-inputs", &o.inputs)?;
    let gid = render_group(&o.dri)?;
    let mut log = log;
    docker::require_image(&mut log, image)?;

    let work = o.work.join(format!("turbo-bench-openvino-{}", std::process::id()));
    fs::create_dir_all(&work).map_err(|e| format!("{}: {e}", work.display()))?;
    let result = (|| {
        onnx::write_inputs(&work, &o.inputs, &o.input_dtype, "--openvino-input-dtype", &m.rows)?;
        let work = fs::canonicalize(&work).map_err(|e| format!("{}: {e}", work.display()))?;
        let argv = |bundle: &Path, work: &Path, p: u32| {
            run_argv(o, bundle, work, &model, gid, &m.rows, iterations, precision, p)
        };
        let (bundle, shown) = (Path::new(docker::BUNDLE), Path::new(docker::WORK));
        let [a, b] = PERCENTILES;
        let p50 = parse(&log.run_as(&argv(&m.bundle_dir, &work, a), argv(bundle, shown, a))?, a)?;
        let p99 = parse(&log.run_as(&argv(&m.bundle_dir, &work, b), argv(bundle, shown, b))?, b)?;
        Ok::<_, String>((p50, p99))
    })();
    let _ = fs::remove_dir_all(&work);
    let (p50, p99) = result?;
    measured(image, log, &procedure, p50, p99, m.rows.batch)
}
