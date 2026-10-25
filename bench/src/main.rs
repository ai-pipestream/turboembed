//! turbo-bench: measure the library on a device and write a benchmark
//! record (docs/benchmarks.md).

mod cli;
mod qa_cmd;
mod references;
mod rerank_cmd;
mod static_cmd;
use references::Given;

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::SystemTime;

use cli::{Opts, USAGE};
use turbo::record::{self, Record};
use turbo::{TURBO_PRECISION_EXACT, TURBO_PRECISION_FASTEST, TURBO_PRECISION_MODEL};
use turbo_bench::measure::{self, Plan, RowKind};
use turbo_bench::openvino::{self, OpenVino};
use turbo_bench::tei::{self, Tei};
use turbo_bench::tensorrt::{self, TensorRt};
use turbo_bench::{Result, git};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("turbo-bench: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &[String]) -> Result<()> {
    match args.first().map(String::as_str) {
        Some("record") => record_cmd(&args[1..]),
        Some("ranges") => ranges_cmd(&args[1..]),
        Some("rerank") => rerank_cmd::run(&args[1..]),
        Some("postag") => turbo_bench::postag::cmd(&args[1..]),
        Some("lattice") => turbo_bench::throughput::lattice_cmd(&args[1..]),
        Some("grammar") => turbo_bench::throughput::grammar_cmd(&args[1..]),
        Some("qa") => qa_cmd::run(&args[1..]),
        Some("static") => static_cmd::run(&args[1..]),
        Some("check") if args.len() > 1 => {
            for a in &args[1..] {
                check(Path::new(a))?;
            }
            Ok(())
        }
        _ => Err(USAGE.into()),
    }
}

fn record_cmd(args: &[String]) -> Result<()> {
    let mut o = Opts::parse(args)?;
    let bundle = o.take("--bundle").ok_or("--bundle is required")?;
    let precision = match o.take("--precision").as_deref().unwrap_or("model") {
        "model" => TURBO_PRECISION_MODEL,
        "fastest" => TURBO_PRECISION_FASTEST,
        "exact" => TURBO_PRECISION_EXACT,
        p => return Err(format!("--precision {p}: model, fastest or exact")),
    };
    let plan = Plan {
        bundle: PathBuf::from(bundle),
        device: o.take("--device").unwrap_or_else(|| "cpu".into()),
        precision,
        batch: o.number("--batch")?,
        seq: o.number("--seq")?,
        rows: RowKind::parse(o.take("--rows").as_deref().unwrap_or("mixed"))?,
        warmup: o.number("--warmup")?.unwrap_or(20),
        iterations: o.number("--iterations")?.unwrap_or(200),
    };
    let sysfs = Path::new(turbo_bench::cpus::SYSFS_CPUS);
    let cpus = o.take("--cpus").map(|l| turbo_bench::cpus::Cpus::parse(&l, sysfs)).transpose()?;
    let repo = o.take("--repo").map_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join(".."), PathBuf::from);
    let out = o.take("--out");
    let work = o.take("--work").map_or_else(std::env::temp_dir, PathBuf::from);
    let no_hailort = o.flag("--no-hailort");
    let hailort =
        o.take("--hailort-bin").map(|binary| turbo_bench::hailo::HailoRt { binary: binary.into(), work: work.clone() });
    let no_tei = o.flag("--no-tei");
    let tei = match (o.take("--tei-image"), o.take("--tei-bin"), o.take("--tei-model")) {
        (Some(image), None, Some(model)) => {
            Some(Tei { image, model_dir: model.into(), cpus: cpus.clone(), binary: None })
        }
        (None, Some(bin), Some(model)) => {
            Some(Tei { image: String::new(), model_dir: model.into(), cpus: cpus.clone(), binary: Some(bin.into()) })
        }
        (None, None, None) => None,
        (Some(_), Some(_), _) => return Err("--tei-image and --tei-bin are two ways to run TEI; give one".into()),
        _ => return Err("--tei-model goes with --tei-image or --tei-bin".into()),
    };
    let no_trt = o.flag("--no-tensorrt");
    let inputs = onnx_inputs(o.take("--tensorrt-inputs"), "--tensorrt-inputs")?;
    tensorrt::check_inputs(&inputs)?;
    let trt = match (o.take("--tensorrt-image"), o.take("--tensorrt-bin")) {
        (Some(_), Some(_)) => {
            return Err("--tensorrt-image and --tensorrt-bin are two ways to run trtexec; give one".into());
        }
        (None, None) => None,
        (image, bin) => Some(TensorRt {
            image: image.unwrap_or_default(),
            trtexec: String::new(),
            inputs,
            input_dtype: String::new(),
            warmup_ms: 0,
            work: work.clone(),
            binary: bin.map(PathBuf::from),
        }),
    };
    let trt = match trt {
        Some(mut t) => {
            t.trtexec = match (o.take("--trtexec"), &t.binary) {
                (Some(_), Some(_)) => {
                    return Err("--trtexec names trtexec inside the image; --tensorrt-bin is the binary itself".into());
                }
                (given, _) => given.unwrap_or_else(|| "trtexec".into()),
            };
            t.input_dtype = o.take("--tensorrt-input-dtype").unwrap_or_else(|| "int64".into());
            t.warmup_ms = o.number("--tensorrt-warmup-ms")?.unwrap_or(1000);
            tensorrt::input_bytes(&[], &t.input_dtype)?;
            Some(t)
        }
        None => None,
    };
    let no_ov = o.flag("--no-openvino");
    let ov_inputs = onnx_inputs(o.take("--openvino-inputs"), "--openvino-inputs")?;
    turbo_bench::onnx::check_inputs("--openvino-inputs", &ov_inputs)?;
    let benchmark_app = o.take("--benchmark-app");
    let benchmark_app_given = benchmark_app.is_some();
    let accel = o.take("--openvino-accel");
    let accel_given = accel.is_some();
    let ov = match (o.take("--openvino-image"), o.take("--openvino-bin")) {
        (Some(_), Some(_)) => {
            return Err("--openvino-image and --openvino-bin are two ways to run benchmark_app; give one".into());
        }
        (None, None) => None,
        (image, bin) => {
            if bin.is_some() && benchmark_app.is_some() {
                return Err(
                    "--benchmark-app names benchmark_app inside the image; --openvino-bin is the binary itself".into(),
                );
            }
            let input_dtype = o.take("--openvino-input-dtype").unwrap_or_else(|| "int64".into());
            turbo_bench::onnx::input_bytes("--openvino-input-dtype", &[], &input_dtype)?;
            Some(OpenVino {
                image: image.unwrap_or_default(),
                benchmark_app: benchmark_app.unwrap_or_else(|| "benchmark_app".into()),
                inputs: ov_inputs,
                input_dtype,
                device: String::new(),
                dri: PathBuf::from("/dev/dri"),
                accel: accel.map_or_else(|| PathBuf::from(openvino::DEFAULT_ACCEL), PathBuf::from),
                work: work.clone(),
                binary: bin.map(PathBuf::from),
            })
        }
    };
    if ov.is_none() && benchmark_app_given {
        return Err("--benchmark-app names benchmark_app inside the image; give --openvino-image".into());
    }
    if accel_given && ov.as_ref().is_none_or(|v| v.binary.is_some()) {
        return Err("--openvino-accel names the NPU device node handed to the container; give --openvino-image".into());
    }
    if let Some(unknown) = o.0.keys().next() {
        return Err(format!("{unknown}: not an option here\n{USAGE}"));
    }
    if (no_hailort && hailort.is_some())
        || (no_tei && tei.is_some())
        || (no_trt && trt.is_some())
        || (no_ov && ov.is_some())
    {
        return Err("a reference program is both named and disabled".into());
    }
    if let Some(t) = &tei {
        match &t.binary {
            // Read now, so a wrong path fails before anything is measured.
            Some(bin) => {
                tei::native_pin(bin)?;
                if cpus.is_some() {
                    return Err("--cpus pins TEI's container; the native router (--tei-bin) takes none".into());
                }
            }
            None => {
                turbo_bench::docker::check_pinned("--tei-image", &t.image)?;
            }
        }
    }
    if let Some(t) = &trt {
        match &t.binary {
            // Read now, so a wrong path fails before anything is measured.
            Some(bin) => {
                tensorrt::native_pin(bin)?;
            }
            None => {
                turbo_bench::docker::check_pinned("--tensorrt-image", &t.image)?;
            }
        }
    }
    if let Some(v) = &ov {
        match &v.binary {
            Some(bin) => {
                openvino::native_pin(bin)?;
            }
            None => {
                turbo_bench::docker::check_pinned("--openvino-image", &v.image)?;
            }
        }
    }
    // Before the runtime starts a thread, which would not inherit it.
    if let Some(c) = &cpus {
        turbo_bench::cpus::pin(c)?;
    }
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    turbo_bench::check_not_testdata(&plan.bundle, &[workspace.join("testdata"), repo.join("testdata")])?;
    let rt = turbo_bench::api::Runtime::create()?;
    let backend = turbo_bench::api::field(&rt.device_info(rt.find(&plan.device)?)?.backend);
    drop(rt);
    let given = Given {
        tei: (tei.is_some(), no_tei),
        trt: (trt.is_some(), no_trt),
        ov: (ov.is_some(), no_ov),
        hailo: (hailort.is_some(), no_hailort),
    };
    given.check(&backend)?;
    // The native router is the reference where no container reaches the
    // GPU; elsewhere TEI runs from its image.
    if tei.as_ref().is_some_and(|t| t.binary.is_some()) && backend != "metal" {
        return Err(format!("--tei-bin is TEI's native router for metal; give --tei-image for {backend}"));
    }
    if ov.as_ref().is_some_and(|v| v.binary.is_some()) && backend != "npu" {
        return Err(format!(
            "--openvino-bin is benchmark_app on the machine for npu; give --openvino-image for {backend}"
        ));
    }
    if accel_given && backend != "npu" {
        return Err(format!("--openvino-accel names the NPU device node; it is only for npu, not {backend}"));
    }

    // Refused before anything is measured, and checked again after.
    let before = git::provenance(&repo)?;
    turbo_bench::check_build(&before.commit, turbo_bench::BUILD_COMMIT, turbo_bench::BUILD_CHANGES)?;
    let m = measure::measure(&plan)?;
    if let Some(why) = measure::npu_embeddings_speed(&m.backend(), &m.settings, tei.is_some(), ov.is_some()) {
        return Err(why.into());
    }
    eprintln!(
        "{} {}: p50 {:.4} ms, p99 {:.4} ms over {} runs of [{}, {}] {}, {} live tokens, {} computed; min cosine \
         {}, max abs diff {:e}",
        m.backend(),
        turbo_bench::api::field(&m.device.name),
        m.timing.p50_ms,
        m.timing.p99_ms,
        m.timing.iterations,
        m.rows.batch,
        m.rows.seq,
        m.rows.kind.name(),
        m.rows.live_tokens(),
        m.timing.computed_tokens.unwrap_or(0),
        m.conformance.min_cosine,
        m.conformance.max_abs_diff
    );
    let references = references::run(&m, &plan, tei, trt, ov, &given, hailort)?;
    let after = git::provenance(&repo)?;
    if after != before {
        return Err(format!("the working tree moved from {} to {} during the run", before.commit, after.commit));
    }
    let r = turbo_bench::record(&m, &after, references, turbo_bench::utc(SystemTime::now()))?;
    let dir = out.map_or_else(|| Path::new(&after.top).join(git::RECORDS_DIR), PathBuf::from);
    let path = turbo_bench::write(&r, &dir)?;
    eprint!("{}", turbo_bench::report(&r));
    println!("{}", path.display());
    Ok(())
}

/// `turbo-bench ranges`: rows, ranges and one row per chunk, measured on
/// one device and written as a range record.
fn ranges_cmd(args: &[String]) -> Result<()> {
    let mut o = Opts::parse(args)?;
    let bundle = o.take("--bundle").ok_or("--bundle is required")?;
    let precision = match o.take("--precision").as_deref().unwrap_or("model") {
        "model" => TURBO_PRECISION_MODEL,
        "fastest" => TURBO_PRECISION_FASTEST,
        "exact" => TURBO_PRECISION_EXACT,
        p => return Err(format!("--precision {p}: model, fastest or exact")),
    };
    let plan = turbo_bench::ranges::RangePlan {
        bundle: PathBuf::from(bundle),
        device: o.take("--device").unwrap_or_else(|| "cpu".into()),
        precision,
        batch: o.number("--batch")?,
        seq: o.number("--seq")?,
        chunk: o.number("--chunk")?.unwrap_or(64),
        warmup: o.number("--warmup")?.unwrap_or(20),
        iterations: o.number("--iterations")?.unwrap_or(200),
    };
    let repo = o.take("--repo").map_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join(".."), PathBuf::from);
    let out = o.take("--out");
    if let Some(unknown) = o.0.keys().next() {
        return Err(format!("{unknown}: not a ranges option\n{USAGE}"));
    }
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    turbo_bench::check_not_testdata(&plan.bundle, &[workspace.join("testdata"), repo.join("testdata")])?;
    // Refused before anything is measured, and checked again after.
    let before = git::provenance(&repo)?;
    turbo_bench::check_build(&before.commit, turbo_bench::BUILD_COMMIT, turbo_bench::BUILD_CHANGES)?;
    let r = turbo_bench::ranges::measure(&plan, &before, turbo_bench::utc(SystemTime::now()))?;
    let after = git::provenance(&repo)?;
    if after != before {
        return Err(format!("the working tree moved from {} to {} during the run", before.commit, after.commit));
    }
    let dir = out.map_or_else(|| Path::new(&after.top).join(turbo_bench::ranges::RANGES_DIR), PathBuf::from);
    let path = turbo_bench::ranges::write(&r, &dir)?;
    eprintln!(
        "{} {}: [{}, {}], chunks of {}: rows p50 {:.4} ms, ranges p50 {:.4} ms ({:.3}x rows), one row per chunk p50 \
         {:.4} ms ({:.3}x ranges); whole-row range max abs diff {:e}, chunk min cosine {:.4}",
        r.device.backend,
        r.device.name,
        r.shape.batch,
        r.shape.seq,
        r.shape.chunk,
        r.rows.p50_ms,
        r.ranges.p50_ms,
        r.ranges_over_rows,
        r.chunk_rows.p50_ms,
        r.chunk_rows_over_ranges,
        r.whole_row_max_abs_diff,
        r.chunk_min_cosine
    );
    println!("{}", path.display());
    Ok(())
}

/// The input names for ids, mask and types, from a comma-separated list
/// or the default; two when the graph takes no types.
fn onnx_inputs(list: Option<String>, option: &str) -> Result<Vec<String>> {
    let list = list.unwrap_or_else(|| "input_ids,attention_mask,token_type_ids".into());
    let v: Vec<String> = list.split(',').map(str::to_owned).collect();
    if v.len() != 2 && v.len() != 3 {
        return Err(format!("{option} names the inputs for ids, mask and types, or for ids and mask alone"));
    }
    Ok(v)
}

/// Parse a record under its file name and say whether it backs SUPPORTED
/// for its own cell in this build.
fn check(path: &Path) -> Result<()> {
    let name = path.file_name().and_then(|n| n.to_str()).ok_or_else(|| format!("{}: no file name", path.display()))?;
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let r = Record::parse(name, &bytes)?;
    let cell = record::Cell {
        arch: &r.machine.arch,
        name: &r.device.name,
        cpu: r.device.kind == "DEVICE_CPU",
        backend: &r.device.backend,
        task: turbo::TURBO_TASK_EMBED,
        precision: [TURBO_PRECISION_MODEL, TURBO_PRECISION_FASTEST, TURBO_PRECISION_EXACT]
            .into_iter()
            .find(|&p| record::precision_name(p) == Some(r.precision.as_str()))
            .unwrap_or(TURBO_PRECISION_MODEL),
        dtype: [turbo::TURBO_DTYPE_F32, turbo::TURBO_DTYPE_F16, turbo::TURBO_DTYPE_BF16]
            .into_iter()
            .find(|&d| record::dtype_name(d) == Some(r.compute_dtype.as_str()))
            .unwrap_or(0),
        version: record::library_version(),
        os: &r.machine.os,
        graph_format: r.library.settings.iter().find_map(|s| s.strip_prefix("TURBO_NPU_GRAPH_FORMAT=")),
    };
    // Only mixed rows back a capability; a dense record is a measurement.
    if r.rows.kind != record::ROWS_MIXED {
        println!("{name}: does not back SUPPORTED: rows.kind {}; only {} rows do", r.rows.kind, record::ROWS_MIXED);
        return Ok(());
    }
    match r.falls_short(&cell) {
        None => println!("{name}: backs SUPPORTED, speed_ratio {:?}", r.speed_ratio),
        Some(why) => println!("{name}: does not back SUPPORTED: {why}"),
    }
    Ok(())
}
