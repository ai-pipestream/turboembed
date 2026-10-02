//! OpenVINO, the kernel reference for Intel GPUs and for the NPU: its
//! `benchmark_app` in an OpenVINO container, pinned by digest, timing the
//! same token rows, one request at a time. Level Zero measures the GPU
//! (`-d GPU`, the DRI nodes) from the bundle's ONNX file. The NPU
//! reference measures the NPU (`-d NPU`) from the static OpenVINO IR
//! (`model.xml` and `model.bin`), never from `onnx/model.onnx`. On Linux
//! the container is given the accel device node. On a machine whose
//! container cannot see the NPU driver (the Windows intel-npu host),
//! `benchmark_app` installed with OpenVINO runs in place of the container
//! (`--openvino-bin`), pinned by the binary's SHA-256.
//!
//! The library never executes ONNX and never links OpenVINO. The NPU
//! product path is the Level Zero graph extension and FORMAT_OPENVINO_IR.
//! A bundle with no file for the device's reference gives a record that
//! says benchmark_app could not run. As for trtexec, the graph stops at
//! the hidden states, so its time has no pooling or normalization in it.
//!
//! benchmark_app reports one latency percentile per run
//! (`-latency_percentile`, the median by default), so the tool runs it
//! twice with the same arguments: once for the median, once for the 99th
//! percentile.

use std::fs;
use std::path::{Path, PathBuf};

use turbo::bundle::{Bundle, sha256_hex};
use turbo::manifest::{Format, GraphInput, Manifest};
use turbo::record::{Measured, ReferenceRun};
use turbo::{TURBO_DTYPE_F16, TURBO_DTYPE_F32};

use crate::docker::{self, Log, Ran, argv};
use crate::measure::{Measurement, Rows};
use crate::{Result, onnx};

pub const NAME: &str = "openvino";

/// benchmark_app's `-d` for the Level Zero GPU.
pub const DEVICE_GPU: &str = "GPU";

/// benchmark_app's `-d` for the NPU.
pub const DEVICE_NPU: &str = "NPU";

/// The NPU device node handed to the container when `--openvino-accel`
/// is omitted. Another node is that flag. Several NPUs are still
/// refused: `-d NPU` is OpenVINO's first device, and there is no
/// per-device index on this command.
pub const DEFAULT_ACCEL: &str = "/dev/accel/accel0";

/// The machine's benchmark_app, as a recorded command names it.
pub const BENCHMARK_APP_BIN: &str = "<benchmark-app>";

/// The percentiles the two runs report.
pub const PERCENTILES: [u32; 2] = [50, 99];

/// The machine's benchmark_app as a record pins it: its file's SHA-256.
pub fn native_pin(bin: &Path) -> Result<String> {
    let bytes = fs::read(bin).map_err(|e| format!("--openvino-bin {}: {e}", bin.display()))?;
    Ok(format!("benchmark-app@sha256:{}", sha256_hex(&bytes)))
}

#[derive(Debug, Clone)]
pub struct OpenVino {
    /// `name@sha256:<64 hex>`.
    pub image: String,
    /// benchmark_app inside the image.
    pub benchmark_app: String,
    /// The ONNX inputs for ids, mask and types, in that order; two when
    /// the graph takes no types.
    pub inputs: Vec<String>,
    /// `int64` or `int32`: the element type of those inputs.
    pub input_dtype: String,
    /// `GPU` or `NPU`: benchmark_app's `-d`. The backend sets it.
    pub device: String,
    /// The host's DRI directory, handed to the container for a GPU run.
    pub dri: PathBuf,
    /// The host's NPU device node, handed to the container for an NPU run.
    pub accel: PathBuf,
    /// Where the input files are written for the container to read.
    pub work: PathBuf,
    /// benchmark_app installed on the machine (`--openvino-bin`), run in
    /// place of an image; `image` is then empty.
    pub binary: Option<PathBuf>,
}

/// benchmark_app's `-infer_precision` for a compute dtype: the plugin
/// computes in f32 or f16.
pub fn infer_precision(device: &str, compute_dtype: u32) -> std::result::Result<&'static str, String> {
    match compute_dtype {
        TURBO_DTYPE_F32 => Ok("f32"),
        TURBO_DTYPE_F16 => Ok("f16"),
        d => Err(format!("OpenVINO's {device} plugin has no inference precision for compute dtype {d}")),
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

/// The group that owns the NPU device node, which the container is added
/// to so it may open it.
pub fn accel_group(accel: &Path) -> Result<u32> {
    if !accel.exists() {
        return Err(format!("{}: no NPU device node for the container", accel.display()));
    }
    gid(accel)
}

/// benchmark_app's own arguments, on paths as the program sees them.
#[allow(clippy::too_many_arguments)]
fn flags(
    o: &OpenVino,
    model: &str,
    rows: &Rows,
    iterations: u32,
    precision: &str,
    percentile: u32,
    input_dir: &str,
) -> Vec<String> {
    let shape = format!("[{},{}]", rows.batch, rows.seq);
    let shapes: Vec<String> = o.inputs.iter().map(|n| format!("{n}{shape}")).collect();
    let files: Vec<String> = o.inputs.iter().map(|n| format!("{n}:{input_dir}/{n}.bin")).collect();
    argv(&[
        "-m",
        model,
        "-d",
        &o.device,
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

/// The `docker run` of benchmark_app: no network, no pulls, the device
/// node with its group (DRI for the GPU, the accel node for the NPU),
/// the bundle and the inputs mounted read-only; the ONNX file compiled
/// for `o.device` at the rows' static shape, in the compute dtype, with
/// the latency hint, one synchronous request, and `iterations` timed
/// inferences after its warm-up one.
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
    let node = if o.device == DEVICE_NPU { o.accel.display().to_string() } else { o.dri.display().to_string() };
    let mut a = argv(&[
        "docker",
        "run",
        "--rm",
        "--pull",
        "never",
        "--network",
        "none",
        "--device",
        &node,
        "--group-add",
        &render_gid.to_string(),
        "--mount",
        &format!("type=bind,src={},dst=/bundle,readonly", bundle.display()),
        "--mount",
        &format!("type=bind,src={},dst=/work,readonly", work.display()),
        &o.image,
        &o.benchmark_app,
    ]);
    a.extend(flags(o, &format!("/bundle/{onnx}"), rows, iterations, precision, percentile, "/work"));
    a
}

/// The machine's benchmark_app run in place of the container: the same
/// arguments on the bundle's and the inputs' directories as they are.
#[allow(clippy::too_many_arguments)]
pub fn native_argv(
    o: &OpenVino,
    bin: &str,
    bundle: &Path,
    work: &Path,
    onnx: &str,
    rows: &Rows,
    iterations: u32,
    precision: &str,
    percentile: u32,
) -> Vec<String> {
    let mut a = argv(&[bin]);
    a.extend(flags(
        o,
        &format!("{}/{onnx}", bundle.display()),
        rows,
        iterations,
        precision,
        percentile,
        &work.display().to_string(),
    ));
    a
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
/// forms both print these, each after an `[ INFO ]` tag, except the C++
/// form's lines of the version block.
pub fn parse(out: &str, percentile: u32) -> Result<Report> {
    let missing = |what: &str| format!("benchmark_app output has no {what}");
    let lines: Vec<&str> = info_lines(out).collect();
    let after = |tag: &str| lines.iter().find_map(|l| l.strip_prefix(tag)).map(str::trim);
    // The C++ form prints the version block untagged after its first line.
    let version = out
        .lines()
        .map(|l| l.trim_start().strip_prefix("[ INFO ]").unwrap_or(l).trim())
        .skip_while(|l| *l != "OpenVINO:")
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

/// A home directory in a program's own error, written as `<home>` so a
/// record does not name the account that ran it. Commands are already
/// recorded with placeholders; this is the failure text.
pub fn redact_user_paths(text: &str) -> String {
    let prefixes =
        ["/home/", "/root/", "/var/home/", "/Users/", "C:\\Users\\", "C:/Users/", "c:\\users\\", "c:/users/"];
    let mut out = text.to_owned();
    let mut i = 0;
    while i < out.len() {
        let rest = &out[i..];
        let found =
            prefixes.iter().filter_map(|p| rest.to_ascii_lowercase().find(&p.to_ascii_lowercase()).map(|at| (at, *p)));
        let Some((at, prefix)) = found.min_by_key(|(at, _)| *at) else { break };
        let start = i + at;
        let after = start + prefix.len();
        let name_end = out[after..]
            .char_indices()
            .find(|(_, c)| *c == '/' || *c == '\\')
            .map(|(n, _)| after + n)
            .unwrap_or(out.len());
        out.replace_range(start..name_end, "<home>");
        i = start + "<home>".len();
    }
    out
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
        not_run: Some(redact_user_paths(&why)),
    }
}

/// The file `benchmark_app -m` is given. GPU compiles the upstream ONNX
/// file. NPU compiles the static token-id IR (the xml; the bin sits
/// beside it). The dynamic ONNX file is not the NPU input.
pub fn model_file(device: &str, manifest: &Manifest) -> std::result::Result<String, String> {
    if device == DEVICE_NPU {
        let ir = manifest.artifacts.iter().find(|a| {
            a.format == Format::OpenvinoIr
                && a.graph_input == GraphInput::TokenIds
                && a.backends.iter().any(|b| b == "npu")
        });
        match ir.map(|a| a.files.as_slice()) {
            Some([xml, ..]) if xml.ends_with(".xml") => Ok(xml.clone()),
            Some(_) => Err("the npu OpenVINO IR's first file is not the xml benchmark_app -d NPU compiles".into()),
            None => {
                Err("the bundle carries no static OpenVINO IR (FORMAT_OPENVINO_IR, INPUT_TOKEN_IDS, backend npu) for \
                 benchmark_app -d NPU. The NPU plugin is not given onnx/model.onnx"
                    .into())
            }
        }
    } else {
        onnx::file(manifest, None, "for benchmark_app to compile")
    }
}

/// The measured reference from the median run and the 99th percentile
/// run of the same arguments, on `rows` at their static shape.
pub fn measured(image: &str, log: Log, procedure: &str, p50: Report, p99: Report, rows: &Rows) -> Result<ReferenceRun> {
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
            rows_per_second: p50.count as f64 * rows.row_count() as f64 / (p50.duration_ms / 1000.0),
            min_cosine: None,
            computed_tokens: Some(rows.padded_tokens()),
        }),
        not_run: None,
    })
}

/// One pass over a batch-1 cycle: each pair is one case's median run and
/// its 99th percentile run, in case order. p50 and p99 are the sums of
/// those latencies. A single pair is `measured` unchanged.
pub fn measured_frames(
    image: &str,
    log: Log,
    procedure: &str,
    pairs: &[(Report, Report)],
    rows: &Rows,
) -> Result<ReferenceRun> {
    if pairs.len() != rows.frames() || pairs.is_empty() {
        return Err(format!("benchmark_app ran {} frames and the rows have {}", pairs.len(), rows.frames()));
    }
    let (first, _) = &pairs[0];
    for (p50, p99) in pairs {
        if p50.version != first.version
            || p99.version != first.version
            || p50.count != first.count
            || p99.count != first.count
        {
            return Err(format!(
                "benchmark_app's frames differ: OpenVINO {} and {}, {} and {} iterations",
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
    }
    if pairs.len() == 1 {
        return measured(image, log, procedure, pairs[0].0.clone(), pairs[0].1.clone(), rows);
    }
    let p50_ms: f64 = pairs.iter().map(|(p, _)| p.latency_ms).sum();
    let p99_ms: f64 = pairs.iter().map(|(_, p)| p.latency_ms).sum();
    let duration_ms: f64 = pairs.iter().map(|(p, _)| p.duration_ms).sum();
    let count = first.count;
    let n = rows.row_count() as f64;
    Ok(ReferenceRun {
        name: NAME.into(),
        role: "kernel".into(),
        pinned: image.into(),
        version: first.version.clone(),
        commands: log.commands,
        procedure: format!(
            "{procedure}; {n} cases, each its own [{}, {}] request: median latencies sum to {p50_ms} ms over {count} \
             inferences a case (durations sum to {duration_ms} ms), and the 99th percentile latencies sum to {p99_ms} ms",
            rows.batch, rows.seq
        ),
        measured: Some(Measured {
            iterations: count,
            p50_ms,
            p99_ms,
            rows_per_second: count as f64 * n / (duration_ms / 1000.0),
            min_cosine: None,
            computed_tokens: Some(rows.padded_tokens()),
        }),
        not_run: None,
    })
}

/// The tag benchmark_app's error lines carry.
pub const ERROR_TAGS: [&str; 1] = ["[ ERROR ] "];

/// Compile the bundle's ONNX file for `o.device` and time it, twice. A
/// thing benchmark_app cannot do for this bundle, benchmark_app failing
/// to compile or run the model included, is a record that says so, with
/// its first error line; a failure of docker, or of finding the device
/// node, is an error.
pub fn run(o: &OpenVino, m: &Measurement, iterations: u32) -> Result<ReferenceRun> {
    if o.device != DEVICE_GPU && o.device != DEVICE_NPU {
        return Err(format!("benchmark_app device {:?} is not {DEVICE_GPU} or {DEVICE_NPU}", o.device));
    }
    let pinned = match &o.binary {
        Some(bin) => native_pin(bin)?,
        None => docker::check_pinned("--openvino-image", &o.image)?.to_owned(),
    };
    let image = pinned.as_str();
    let own = if o.binary.is_some() { ", the machine's own (--openvino-bin)," } else { "" };
    let graph = if o.device == DEVICE_NPU { "static OpenVINO IR" } else { "ONNX file" };
    let frames = m.rows.frames();
    let procedure = if frames == 1 {
        format!(
            "benchmark_app{own} compiles the bundle's {graph} for the {} at the batch's static [{}, {}] shape and times \
             {iterations} synchronous inferences of the rows loaded from files, one request, after its one warm-up \
             inference; it runs twice with the same arguments, -latency_percentile 50 then 99, and p50 and p99 are \
             those runs' latencies; rows per second is the median run's count times the batch over its duration",
            o.device, m.rows.batch, m.rows.seq
        )
    } else {
        format!(
            "benchmark_app{own} compiles the bundle's {graph} for the {} at the frame's static [{}, {}] shape and times \
             {iterations} synchronous inferences of each of {frames} fitting cases, each case padded to seq in its own \
             [{}, {}] request, after that request's one warm-up inference; each case runs twice, -latency_percentile \
             50 then 99; p50 and p99 are the sums of those cases' latencies, one pass over the case set; rows per \
             second is the median runs' count times the case count over the sum of their durations",
            o.device, m.rows.batch, m.rows.seq, m.rows.batch, m.rows.seq
        )
    };
    let log = Log::default();
    let model = match model_file(&o.device, &m.manifest) {
        Ok(f) => f,
        Err(why) => return Ok(not_run(image, log, &procedure, why)),
    };
    let precision = match infer_precision(&o.device, m.compute_dtype) {
        Ok(p) => p,
        Err(why) => return Ok(not_run(image, log, &procedure, why)),
    };
    // The file the manifest lists, checked against its hash.
    Bundle::open(&m.bundle_dir).and_then(|b| b.read_verified(&model)).map_err(|e| e.message)?;
    onnx::check_inputs("--openvino-inputs", &o.inputs)?;
    let gid = if o.binary.is_some() {
        None
    } else if o.device == DEVICE_NPU {
        Some(accel_group(&o.accel)?)
    } else {
        Some(render_group(&o.dri)?)
    };
    let mut log = log;
    if o.binary.is_none() {
        docker::require_image(&mut log, image)?;
    }

    let work = o.work.join(format!("turbo-bench-openvino-{}", std::process::id()));
    fs::create_dir_all(&work).map_err(|e| format!("{}: {e}", work.display()))?;
    let result = (|| {
        let (bundle, shown_bundle) = (&m.bundle_dir, Path::new(docker::BUNDLE));
        let gid = gid.unwrap_or(0);
        // One frame keeps the single input directory. A cycle writes each
        // case under frame-<n> and mounts that directory, so every request
        // stays [batch, seq] and the recorded command names the case.
        let mut run_pair =
            |dir: &Path, shown: &Path, frame_rows: &Rows| -> Result<std::result::Result<(Report, Report), String>> {
                let argv_for = |p: u32| match &o.binary {
                    Some(bin) => (
                        native_argv(
                            o,
                            &bin.display().to_string(),
                            bundle,
                            dir,
                            &model,
                            frame_rows,
                            iterations,
                            precision,
                            p,
                        ),
                        native_argv(
                            o,
                            BENCHMARK_APP_BIN,
                            shown_bundle,
                            shown,
                            &model,
                            frame_rows,
                            iterations,
                            precision,
                            p,
                        ),
                    ),
                    None => (
                        run_argv(o, bundle, dir, &model, gid, frame_rows, iterations, precision, p),
                        run_argv(o, shown_bundle, shown, &model, gid, frame_rows, iterations, precision, p),
                    ),
                };
                let [a, b] = PERCENTILES;
                let mut timed = |p: u32| {
                    let (cmd, shown) = argv_for(p);
                    match log.run_program(&cmd, shown, &ERROR_TAGS)? {
                        Ran::Done(out) => parse(&out, p).map(Ok),
                        Ran::Failed(why) => Ok(Err(format!("benchmark_app {why}"))),
                    }
                };
                let p50 = match timed(a)? {
                    Ok(r) => r,
                    Err(why) => return Ok(Err(why)),
                };
                Ok(timed(b)?.map(|p99| (p50, p99)))
            };
        let mut pairs = Vec::with_capacity(frames);
        for f in 0..frames {
            let (dir, shown, frame_rows) = if frames == 1 {
                onnx::write_inputs(&work, &o.inputs, &o.input_dtype, "--openvino-input-dtype", &m.rows)?;
                let dir = fs::canonicalize(&work).map_err(|e| format!("{}: {e}", work.display()))?;
                (dir, PathBuf::from(docker::WORK), m.rows.frame(0))
            } else {
                let dir = Rows::frame_dir(&work, f);
                fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
                let frame_rows = m.rows.frame(f);
                onnx::write_inputs(&dir, &o.inputs, &o.input_dtype, "--openvino-input-dtype", &frame_rows)?;
                let dir = fs::canonicalize(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
                (dir, PathBuf::from(format!("{}/frame-{f}", docker::WORK)), frame_rows)
            };
            match run_pair(&dir, &shown, &frame_rows)? {
                Ok(pair) => pairs.push(pair),
                Err(why) => return Ok(Err(why)),
            }
        }
        Ok::<_, String>(Ok(pairs))
    })();
    let _ = fs::remove_dir_all(&work);
    let pairs = match result? {
        Ok(r) => r,
        Err(why) => return Ok(not_run(image, log, &procedure, why)),
    };
    measured_frames(image, log, &procedure, &pairs, &m.rows)
}
