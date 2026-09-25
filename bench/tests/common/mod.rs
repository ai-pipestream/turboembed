//! What the tool's tests share: the small sealed bundle, git working trees
//! made in temporary directories, and one real measurement of the CPU
//! backend on the bundle.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use turbo::TURBO_PRECISION_MODEL;
use turbo::record::{self, Measured, Record, ReferenceRun};
use turbo_bench::git::Provenance;
use turbo_bench::measure::{Measurement, Plan, measure};

pub fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

pub fn tiny_bundle() -> PathBuf {
    workspace().join("testdata/tiny-bert-bundle")
}

/// A copy of the small bundle at `dir`, outside testdata/, where the tool
/// takes it as it would any bundle.
pub fn bundle_copy(dir: &Path) -> PathBuf {
    fn copy(from: &Path, to: &Path) {
        std::fs::create_dir_all(to).unwrap();
        for e in std::fs::read_dir(from).unwrap() {
            let e = e.unwrap();
            if e.file_type().unwrap().is_dir() {
                copy(&e.path(), &to.join(e.file_name()));
            } else {
                std::fs::copy(e.path(), to.join(e.file_name())).unwrap();
            }
        }
    }
    copy(&tiny_bundle(), dir);
    dir.to_owned()
}

/// A fresh, empty directory, removed with everything in it when this goes.
pub struct Scratch(PathBuf);

impl std::ops::Deref for Scratch {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A fresh, empty directory for `name`.
pub fn scratch(name: &str) -> Scratch {
    let d = std::env::temp_dir().join(format!("turbo-bench-test-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    Scratch(d)
}

/// git in `dir`, with an identity and no signing of its own, so the
/// user's configuration changes nothing; its output, or a panic.
pub fn git(dir: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["-c", "user.name=turbo-bench test", "-c", "user.email=test@invalid", "-c", "commit.gpgsign=false"])
        .args(args)
        .output()
        .unwrap();
    assert!(out.status.success(), "git {args:?}: {}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).trim().to_owned()
}

/// A working tree under `root` with one commit on main, pushed to a bare
/// repository that is its origin. Returns the working tree.
pub fn pushed_repo(root: &Path) -> PathBuf {
    let origin = root.join("origin.git");
    let work = root.join("work");
    std::fs::create_dir_all(&origin).unwrap();
    std::fs::create_dir_all(&work).unwrap();
    git(&origin, &["init", "--quiet", "--bare", "--initial-branch=main"]);
    git(&work, &["init", "--quiet", "--initial-branch=main"]);
    std::fs::write(work.join("README"), "a library\n").unwrap();
    git(&work, &["add", "README"]);
    git(&work, &["commit", "--quiet", "-m", "first"]);
    git(&work, &["remote", "add", "origin", origin.to_str().unwrap()]);
    git(&work, &["push", "--quiet", "origin", "main"]);
    work
}

/// The reference programs' names, as records give them.
pub const TEI: &str = turbo_bench::tei::NAME;
pub const TRT: &str = turbo_bench::tensorrt::NAME;
pub const OV: &str = turbo_bench::openvino::NAME;

/// The CPU backend on the small bundle, measured once for every test
/// here.
pub fn cpu_measurement() -> &'static Measurement {
    static M: OnceLock<Measurement> = OnceLock::new();
    M.get_or_init(|| {
        let plan = Plan {
            bundle: tiny_bundle(),
            device: "cpu".into(),
            precision: TURBO_PRECISION_MODEL,
            batch: None,
            seq: None,
            warmup: 3,
            iterations: 30,
        };
        measure(&plan).unwrap_or_else(|e| panic!("{e}"))
    })
}

/// The provenance of a pushed repository made for the test.
pub fn provenance(name: &str) -> Provenance {
    let root = scratch(name);
    turbo_bench::git::provenance(&pushed_repo(&root)).unwrap()
}

/// A record of the real CPU measurement with `references`.
pub fn cpu_record(name: &str, references: Vec<ReferenceRun>) -> Record {
    turbo_bench::record(cpu_measurement(), &provenance(name), references, "2026-01-02T03:04:05Z".into())
        .unwrap_or_else(|e| panic!("{e}"))
}

/// A reference entry for the SUPPORTED rule's tests. No reference program
/// runs here, so its figures are derived from the real CPU measurement
/// (twice its latency, half its rate) and it names an image nobody
/// pulls; it is never written where the core reads records.
pub fn measured_reference(name: &str) -> ReferenceRun {
    let m = cpu_measurement();
    let role = record::REFERENCES.iter().find(|r| r.0 == name).unwrap_or_else(|| panic!("{name}: no such reference")).1;
    ReferenceRun {
        name: name.into(),
        role: role.into(),
        pinned: format!("example.invalid/{name}@sha256:{}", "0".repeat(64)),
        version: "0.0.0".into(),
        commands: vec![vec!["turbo-bench".into(), "record".into()]],
        procedure: "figures derived from the library's own measurement, for the rule's tests".into(),
        measured: Some(Measured {
            iterations: m.timing.iterations as u64,
            p50_ms: m.timing.p50_ms * 2.0,
            p99_ms: m.timing.p99_ms * 2.0,
            rows_per_second: m.timing.rows_per_second / 2.0,
            min_cosine: Some(m.conformance.min_cosine),
        }),
        not_run: None,
    }
}

/// trtexec's output in the form TensorRT's samples print it: the version
/// line from trtexec.cpp, and the prolog and performance summary from
/// sampleReporting.cpp, with --percentile=99.
pub const TRTEXEC_OUT: &str = "\
&&&& RUNNING TensorRT.trtexec [TensorRT v100300] [b17] # trtexec --onnx=/bundle/onnx/model.onnx --percentile=99
[09/25/2026-10:00:00] [I] === Model Options ===
[09/25/2026-10:00:00] [I] Format: ONNX
[09/25/2026-10:00:00] [I] TensorRT version: 10.3.0
[09/25/2026-10:00:40] [I] Warmup completed 2710 queries over 1000 ms
[09/25/2026-10:00:40] [I] Timing trace has 200 queries over 0.0741 s
[09/25/2026-10:00:40] [I]
[09/25/2026-10:00:40] [I] === Trace details ===
[09/25/2026-10:00:40] [I] === Performance summary ===
[09/25/2026-10:00:40] [I] Throughput: 2699.06 qps
[09/25/2026-10:00:40] [I] Latency: min = 0.36377 ms, max = 0.52124 ms, mean = 0.374511 ms, median = 0.372559 ms, percentile(99%) = 0.412598 ms
[09/25/2026-10:00:40] [I] Enqueue Time: min = 0.0114746 ms, max = 0.0491943 ms, mean = 0.0136421 ms, median = 0.0130615 ms, percentile(99%) = 0.0249023 ms
[09/25/2026-10:00:40] [I] H2D Latency: min = 0.0107422 ms, max = 0.0244141 ms, mean = 0.0117093 ms, median = 0.0115967 ms, percentile(99%) = 0.0170898 ms
[09/25/2026-10:00:40] [I] GPU Compute Time: min = 0.339966 ms, max = 0.48999 ms, mean = 0.350241 ms, median = 0.348389 ms, percentile(99%) = 0.385986 ms
[09/25/2026-10:00:40] [I] D2H Latency: min = 0.0112305 ms, max = 0.0161133 ms, mean = 0.0125591 ms, median = 0.0124512 ms, percentile(99%) = 0.0146484 ms
[09/25/2026-10:00:40] [I] Total Host Walltime: 0.0741 s
[09/25/2026-10:00:40] [I] Total GPU Compute Time: 0.0700482 s
&&&& PASSED TensorRT.trtexec [TensorRT v100300] [b17] # trtexec --onnx=/bundle/onnx/model.onnx --percentile=99
";

/// benchmark_app's report in the form its Python tool prints it
/// (tools/benchmark_tool/openvino/tools/benchmark: main.py, benchmark.py's
/// print_version_info, utils/utils.py's next_step, and the logging format
/// `[ %(levelname)s ] %(message)s`), for `-latency_percentile 50`.
pub const BENCHMARK_APP_OUT: &str = "\
[Step 1/11] Parsing and validating input arguments
[ INFO ] Parsing input parameters
[Step 2/11] Loading OpenVINO Runtime
[ INFO ] OpenVINO:
[ INFO ] Build ................................. 2025.3.0-19807-44526285f24-releases/2025/3
[ INFO ] 
[ INFO ] Device info:
[ INFO ] GPU
[ INFO ] Build ................................. 2025.3.0-19807-44526285f24-releases/2025/3
[ INFO ] 
[ INFO ] 
[Step 3/11] Setting device configuration
[Step 4/11] Reading model files
[ INFO ] Loading model files
[ INFO ] Read model took 38.91 ms
[Step 5/11] Resizing model to match image sizes and given batch
[ INFO ] Model batch size: 1
[Step 6/11] Configuring input of the model
[Step 7/11] Loading the model to the device
[ INFO ] Compile model took 2140.27 ms
[Step 8/11] Querying optimal runtime parameters
[Step 9/11] Creating infer requests and preparing input tensors
[Step 10/11] Measuring performance (Start inference synchronously, limits: 200 iterations)
[ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
[ INFO ] First inference took 11.84 ms
[Step 11/11] Dumping statistics report
[ INFO ] Execution Devices:['GPU']
[ INFO ] Count:            200 iterations
[ INFO ] Duration:         412.64 ms
[ INFO ] Latency:
[ INFO ]    Median:        2.03 ms
[ INFO ]    Average:       2.05 ms
[ INFO ]    Min:           1.95 ms
[ INFO ]    Max:           2.71 ms
[ INFO ] Throughput:   484.68 FPS
";

/// The same run with `-latency_percentile 99` and an average under a
/// millisecond, which benchmark_app prints in microseconds.
pub fn p99_out() -> String {
    BENCHMARK_APP_OUT
        .replace("   Median:        2.03 ms", "   99 percentile:     961.20 us")
        .replace("   Average:       2.05 ms", "   Average:       904.33 us")
        .replace("   Min:           1.95 ms", "   Min:           880.10 us")
        .replace("   Max:           2.71 ms", "   Max:           1210.52 us")
}
