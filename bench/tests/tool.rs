//! The tool end to end: turbo-bench run as a program on the CPU backend
//! with the small bundle, reference programs disabled, writing its record
//! to a temporary directory.

mod common;

use std::path::Path;
use std::process::{Command, Output};

use common::*;
use turbo::bundle::sha256_hex;
use turbo::record::{self, Cell, Record, Verdict};
use turbo::{TURBO_DTYPE_F32, TURBO_PRECISION_MODEL, TURBO_TASK_EMBED};
use turbo_bench::api::{Runtime, field};
use turbo_bench::measure::{Reference, Rows};

fn tool(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_turbo-bench")).args(args).output().unwrap()
}

fn record_args<'a>(repo: &'a str, out: &'a str, bundle: &'a str) -> Vec<&'a str> {
    vec![
        "record",
        "--bundle",
        bundle,
        "--device",
        "cpu",
        "--repo",
        repo,
        "--out",
        out,
        "--warmup",
        "3",
        "--iterations",
        "25",
        "--batch",
        "12",
        "--no-tei",
    ]
}

#[test]
fn a_cpu_record_is_measured_and_without_a_reference_backs_nothing() {
    let root = scratch("tool-cpu");
    let repo = pushed_repo(&root);
    let out = root.join("records");
    let bundle = tiny_bundle();
    let o = tool(&record_args(repo.to_str().unwrap(), out.to_str().unwrap(), bundle.to_str().unwrap()));
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let path = String::from_utf8(o.stdout).unwrap().trim().to_owned();
    let path = Path::new(&path);
    assert_eq!(path.parent().unwrap(), out);
    let name = path.file_name().unwrap().to_str().unwrap();
    let r = Record::parse(name, &std::fs::read(path).unwrap()).unwrap();

    // Each field from where it is measured or read, not from the command line.
    let rt = Runtime::create().unwrap();
    let cpu = rt.device_info(rt.find("cpu").unwrap()).unwrap();
    assert_eq!(r.device.name, field(&cpu.name));
    assert_eq!(r.machine.arch, field(&cpu.arch));
    assert_eq!(r.device.memory_total, cpu.memory_total);
    assert_eq!(r.library.commit, git(&repo, &["rev-parse", "HEAD"]));
    assert_eq!(r.library.pushed_to, vec!["origin/main".to_owned()]);
    assert_eq!(r.library.build, turbo_bench::api::version());
    let manifest = std::fs::read(bundle.join("manifest.json")).unwrap();
    assert_eq!(r.bundle.manifest_sha256, sha256_hex(&manifest));
    let tokenizer = std::fs::read(bundle.join("tokenizer.json")).unwrap();
    assert_eq!(r.bundle.tokenizer_sha256, sha256_hex(&tokenizer));
    let weights = std::fs::read(bundle.join("weights/model.safetensors")).unwrap();
    assert_eq!(r.bundle.artifact_sha256, sha256_hex(&weights));
    let b = turbo::bundle::Bundle::open(&bundle).unwrap();
    let rows = Rows::build(&Reference::read(&b).unwrap(), 0, 12, r.rows.seq).unwrap();
    assert_eq!((r.rows.batch, r.rows.sha256.as_str()), (12, rows.sha256().as_str()));
    assert_eq!((r.timing.warmup, r.timing.iterations), (3, 25));
    assert!(r.timing.p50_ms > 0.0 && r.timing.p50_ms <= r.timing.p99_ms, "{:?}", r.timing);
    assert!(r.conformance.min_cosine >= 0.9999 && r.conformance.max_abs_diff <= 1e-4, "{:?}", r.conformance);
    assert_eq!(r.references.len(), 1, "TEI is the CPU's reference, and it was disabled");
    assert_eq!(r.references[0].name, "text-embeddings-inference");
    assert_eq!(r.references[0].not_run.as_deref(), Some("disabled on the command line (--no-tei)"));
    assert_eq!((r.speed_ratio, r.speed_reference.as_deref()), (None, None));

    // Such a record backs no cell.
    let (arch, cpu_name) = (field(&cpu.arch), field(&cpu.name));
    let cell = Cell {
        arch: &arch,
        name: &cpu_name,
        cpu: true,
        backend: "cpu",
        task: TURBO_TASK_EMBED,
        precision: TURBO_PRECISION_MODEL,
        dtype: TURBO_DTYPE_F32,
        version: record::library_version(),
    };
    assert_eq!(record::decide([(name, &r)], &cell), Verdict::Not(format!("{name}: no reference program measured")));
    let o = tool(&["check", path.to_str().unwrap()]);
    assert!(o.status.success());
    assert_eq!(
        String::from_utf8(o.stdout).unwrap(),
        format!("{name}: does not back SUPPORTED: no reference program measured\n")
    );

    // The same commit, machine and bundle again: never replaced.
    let o = tool(&record_args(repo.to_str().unwrap(), out.to_str().unwrap(), bundle.to_str().unwrap()));
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("a record is never replaced"));
}

#[test]
fn the_tool_refuses_a_dirty_tree_before_measuring() {
    let root = scratch("tool-dirty");
    let repo = pushed_repo(&root);
    std::fs::write(repo.join("README"), "edited\n").unwrap();
    let out = root.join("records");
    let o = tool(&record_args(repo.to_str().unwrap(), out.to_str().unwrap(), tiny_bundle().to_str().unwrap()));
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("the working tree is not clean"));
    assert!(!out.exists(), "nothing is written");
}

#[test]
fn the_tool_wants_each_reference_named_or_disabled() {
    let root = scratch("tool-references");
    let repo = pushed_repo(&root);
    let (repo, out, bundle) = (repo.to_str().unwrap().to_owned(), root.join("o"), tiny_bundle());
    let mut args = record_args(&repo, out.to_str().unwrap(), bundle.to_str().unwrap());
    args.pop();
    let o = tool(&args);
    assert!(
        String::from_utf8_lossy(&o.stderr)
            .contains("cpu: TEI is a reference here: give --tei-image and --tei-model, or --no-tei")
    );
    let mut args = record_args(&repo, out.to_str().unwrap(), bundle.to_str().unwrap());
    args.push("--no-tensorrt");
    let o = tool(&args);
    assert!(String::from_utf8_lossy(&o.stderr).contains("cpu: TensorRT is not a reference for this backend"));
    let mut args = record_args(&repo, out.to_str().unwrap(), bundle.to_str().unwrap());
    args.pop();
    args.extend(["--tei-image", "ghcr.io/huggingface/text-embeddings-inference:cpu-1.8", "--tei-model", "."]);
    let o = tool(&args);
    assert!(String::from_utf8_lossy(&o.stderr).contains("is not pinned as name@sha256:<64 hex>"), "{o:?}");
    assert!(!out.exists());
}
