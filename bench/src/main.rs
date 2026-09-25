//! turbo-bench: measure the library on a device and write a benchmark
//! record (docs/benchmarks.md).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::SystemTime;

use turbo::record::{self, Record, ReferenceRun};
use turbo::{TURBO_PRECISION_EXACT, TURBO_PRECISION_FASTEST, TURBO_PRECISION_MODEL};
use turbo_bench::measure::{self, Measurement, Plan};
use turbo_bench::tei::{self, Tei};
use turbo_bench::tensorrt::{self, TensorRt};
use turbo_bench::{Result, git};

const USAGE: &str = "\
usage:
  turbo-bench record --bundle <dir> [options]
      measure the library on one device, run the reference programs on
      the same token rows, and write the record
  turbo-bench check <record.json>...
      parse records and say whether each backs SUPPORTED for its cell

record options:
  --device <index|backend>     the runtime device (default cpu)
  --precision <p>              model, fastest or exact (default model)
  --batch <n>, --seq <n>       the rows' shape (default: 32 or the model's
                               max_batch if smaller; the longest reference
                               case that fits)
  --warmup <n>                 untimed runs first (default 20)
  --iterations <n>             timed runs (default 200)
  --repo <dir>                 the git working tree the library was built
                               from (default: the one this tool was built in)
  --out <dir>                  where the record goes (default
                               <repo>/benchmarks/records)
  --work <dir>                 scratch for reference inputs (default: the
                               system's temporary directory)
  --tei-image <name@sha256:..> text-embeddings-inference, pinned
  --tei-model <dir>            the model in the upstream layout
  --no-tei                     record that TEI was not run
  --tensorrt-image <name@sha256:..>
                               NVIDIA's TensorRT container, pinned (cuda)
  --tensorrt-inputs <a,b,c>    the ONNX inputs for ids, mask and types
                               (default input_ids,attention_mask,token_type_ids)
  --tensorrt-input-dtype <t>   int64 or int32 (default int64)
  --tensorrt-warmup-ms <n>     trtexec --warmUp (default 1000)
  --trtexec <path>             trtexec in the image (default trtexec)
  --no-tensorrt                record that TensorRT was not run";

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
        Some("check") if args.len() > 1 => {
            for a in &args[1..] {
                check(Path::new(a))?;
            }
            Ok(())
        }
        _ => Err(USAGE.into()),
    }
}

/// `--name value` pairs and bare `--flag`s.
struct Opts(BTreeMap<String, Option<String>>);

const FLAGS: [&str; 2] = ["--no-tei", "--no-tensorrt"];

impl Opts {
    fn parse(args: &[String]) -> Result<Opts> {
        let mut m = BTreeMap::new();
        let mut it = args.iter();
        while let Some(a) = it.next() {
            if !a.starts_with("--") {
                return Err(format!("{a}: not an option\n{USAGE}"));
            }
            let v = if FLAGS.contains(&a.as_str()) {
                None
            } else {
                Some(it.next().ok_or_else(|| format!("{a} needs a value"))?.clone())
            };
            if m.insert(a.clone(), v).is_some() {
                return Err(format!("{a} is given twice"));
            }
        }
        Ok(Opts(m))
    }

    fn take(&mut self, name: &str) -> Option<String> {
        self.0.remove(name).flatten()
    }

    fn flag(&mut self, name: &str) -> bool {
        self.0.remove(name).is_some()
    }

    fn number(&mut self, name: &str) -> Result<Option<u32>> {
        self.take(name).map(|v| v.parse().map_err(|_| format!("{name} {v}: not a number"))).transpose()
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
        warmup: o.number("--warmup")?.unwrap_or(20),
        iterations: o.number("--iterations")?.unwrap_or(200),
    };
    let repo = o.take("--repo").map_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join(".."), PathBuf::from);
    let out = o.take("--out");
    let work = o.take("--work").map_or_else(std::env::temp_dir, PathBuf::from);
    let no_tei = o.flag("--no-tei");
    let tei = match (o.take("--tei-image"), o.take("--tei-model")) {
        (Some(image), Some(model)) => Some(Tei { image, model_dir: model.into() }),
        (None, None) => None,
        _ => return Err("--tei-image and --tei-model go together".into()),
    };
    let no_trt = o.flag("--no-tensorrt");
    let inputs = o.take("--tensorrt-inputs").unwrap_or_else(|| "input_ids,attention_mask,token_type_ids".into());
    let inputs: Vec<String> = inputs.split(',').map(str::to_owned).collect();
    let inputs: [String; 3] =
        inputs.try_into().map_err(|_| "--tensorrt-inputs names three inputs: ids, mask, types".to_owned())?;
    let trt = o.take("--tensorrt-image").map(|image| TensorRt {
        image,
        trtexec: String::new(),
        inputs,
        input_dtype: String::new(),
        warmup_ms: 0,
        work: work.clone(),
    });
    let trt = match trt {
        Some(mut t) => {
            t.trtexec = o.take("--trtexec").unwrap_or_else(|| "trtexec".into());
            t.input_dtype = o.take("--tensorrt-input-dtype").unwrap_or_else(|| "int64".into());
            t.warmup_ms = o.number("--tensorrt-warmup-ms")?.unwrap_or(1000);
            tensorrt::input_bytes(&[], &t.input_dtype)?;
            Some(t)
        }
        None => None,
    };
    if let Some(unknown) = o.0.keys().next() {
        return Err(format!("{unknown}: not an option here\n{USAGE}"));
    }
    if (no_tei && tei.is_some()) || (no_trt && trt.is_some()) {
        return Err("a reference program is both named and disabled".into());
    }
    if let Some(t) = &tei {
        turbo_bench::docker::check_pinned("--tei-image", &t.image)?;
    }
    if let Some(t) = &trt {
        turbo_bench::docker::check_pinned("--tensorrt-image", &t.image)?;
    }

    // Refused before anything is measured, and checked again after.
    let before = git::provenance(&repo)?;
    let m = measure::measure(&plan)?;
    eprintln!(
        "{} {}: p50 {:.4} ms, p99 {:.4} ms over {} runs of [{}, {}]; min cosine {}, max abs diff {:e}",
        m.backend(),
        turbo_bench::api::field(&m.device.name),
        m.timing.p50_ms,
        m.timing.p99_ms,
        m.timing.iterations,
        m.rows.batch,
        m.rows.seq,
        m.conformance.min_cosine,
        m.conformance.max_abs_diff
    );
    let references = references(&m, &plan, tei, no_tei, trt, no_trt)?;
    let after = git::provenance(&repo)?;
    if after != before {
        return Err(format!("the working tree moved from {} to {} during the run", before.commit, after.commit));
    }
    let r = turbo_bench::record(&m, &after, references, turbo_bench::utc(SystemTime::now()))?;
    let dir = out.map_or_else(|| Path::new(&after.top).join(git::RECORDS_DIR), PathBuf::from);
    let path = turbo_bench::write(&r, &dir)?;
    println!("{}", path.display());
    Ok(())
}

/// A reference program disabled on the command line.
fn disabled(name: &str, role: &str, flag: &str) -> ReferenceRun {
    ReferenceRun {
        name: name.into(),
        role: role.into(),
        pinned: String::new(),
        version: String::new(),
        commands: Vec::new(),
        procedure: String::new(),
        measured: None,
        not_run: Some(format!("disabled on the command line ({flag})")),
    }
}

/// Every reference program for the device's backend: TEI and TensorRT
/// for cuda, TEI's CPU image for the CPU, none for another backend. Each
/// is named or disabled on the command line; neither is an error.
fn references(
    m: &Measurement,
    plan: &Plan,
    tei: Option<Tei>,
    no_tei: bool,
    trt: Option<TensorRt>,
    no_trt: bool,
) -> Result<Vec<ReferenceRun>> {
    let backend = m.backend();
    let (wants_tei, wants_trt) = match backend.as_str() {
        "cuda" => (true, true),
        "cpu" => (true, false),
        _ => (false, false),
    };
    let gpu = (backend == "cuda").then_some(m.device.ordinal);
    // docker numbers GPUs in the driver's (PCI bus) order, CUDA fastest
    // first unless told otherwise: with more than one, they must agree.
    let named = tei.is_some() || trt.is_some();
    if gpu.is_some() && named && std::env::var("CUDA_DEVICE_ORDER").as_deref() != Ok("PCI_BUS_ID") {
        let rt = turbo_bench::api::Runtime::create()?;
        let mut cuda = 0;
        for i in 0..rt.device_count()? {
            cuda += u32::from(turbo_bench::api::field(&rt.device_info(i)?.backend) == "cuda");
        }
        if cuda > 1 {
            return Err(format!(
                "{cuda} CUDA devices are listed: set CUDA_DEVICE_ORDER=PCI_BUS_ID so the device measured is \
                 the one docker gives the reference program"
            ));
        }
    }
    let mut out = Vec::new();
    match (wants_tei, tei, no_tei) {
        (true, Some(t), _) => out.push(tei::run(&t, m, gpu, plan.warmup, plan.iterations)?),
        (true, None, true) => out.push(disabled(tei::NAME, "end_to_end", "--no-tei")),
        (true, None, false) => {
            return Err(format!("{backend}: TEI is a reference here: give --tei-image and --tei-model, or --no-tei"));
        }
        (false, Some(_), _) | (false, None, true) => {
            return Err(format!("{backend}: TEI is not a reference for this backend"));
        }
        (false, None, false) => {}
    }
    match (wants_trt, trt, no_trt) {
        (true, Some(t), _) => out.push(tensorrt::run(&t, m, m.device.ordinal, plan.iterations)?),
        (true, None, true) => out.push(disabled(tensorrt::NAME, "kernel", "--no-tensorrt")),
        (true, None, false) => {
            return Err(format!("{backend}: TensorRT is a reference here: give --tensorrt-image, or --no-tensorrt"));
        }
        (false, Some(_), _) | (false, None, true) => {
            return Err(format!("{backend}: TensorRT is not a reference for this backend"));
        }
        (false, None, false) => {}
    }
    Ok(out)
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
    };
    match r.falls_short(&cell) {
        None => println!("{name}: backs SUPPORTED, speed_ratio {:?}", r.speed_ratio),
        Some(why) => println!("{name}: does not back SUPPORTED: {why}"),
    }
    Ok(())
}
