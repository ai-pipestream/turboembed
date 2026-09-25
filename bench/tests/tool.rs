//! The tool as a program: its refusals, on git trees made for the test,
//! and, when this checkout is a clean commit on a branch of origin (the
//! only tree the tool will name), a record made end to end on the CPU
//! backend with the small bundle, TEI disabled, into a temporary
//! directory.

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

/// Why this checkout cannot be recorded from, or None when it can: the
/// tool names only the commit it was built from, clean and pushed.
fn unrecordable() -> Option<String> {
    let p = match turbo_bench::git::provenance(&workspace()) {
        Ok(p) => p,
        Err(e) => return Some(e),
    };
    turbo_bench::check_build(&p.commit, turbo_bench::BUILD_COMMIT, turbo_bench::BUILD_CHANGES).err()
}

#[test]
fn a_cpu_record_is_measured_and_without_a_reference_backs_nothing() {
    if let Some(why) = unrecordable() {
        // CI on main sets this: there the checkout is a pushed commit, and
        // a skip would hide that the tool cannot record from it.
        if std::env::var("TURBO_BENCH_REQUIRE_RECORD").as_deref() == Ok("1") {
            panic!("TURBO_BENCH_REQUIRE_RECORD=1, and this checkout cannot be recorded from: {why}");
        }
        eprintln!("skipped: this checkout cannot be recorded from: {why}");
        return;
    }
    let root = scratch("tool-cpu");
    let repo = workspace().canonicalize().unwrap();
    let out = root.join("records");
    let bundle = bundle_copy(&root.join("bundle"));
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
    assert_eq!(r.library.commit, turbo_bench::BUILD_COMMIT);
    assert_eq!(r.library.pushed_to, turbo_bench::git::provenance(&repo).unwrap().pushed_to);
    assert_eq!(r.machine.os, std::env::consts::OS);
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
        os: std::env::consts::OS,
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
fn the_tool_refuses_a_tree_it_was_not_built_from() {
    if turbo_bench::BUILD_COMMIT.is_empty() {
        // Built outside git (from a tarball): the tool refuses every tree
        // for that, which the_build_is_named_only_when_it_is_the_commit_and_clean covers.
        eprintln!("skipped: this build is from no git commit, so there is no other tree to refuse");
        return;
    }
    let root = scratch("tool-other-tree");
    let repo = pushed_repo(&root);
    let out = root.join("records");
    let bundle = bundle_copy(&root.join("bundle"));
    let o = tool(&record_args(repo.to_str().unwrap(), out.to_str().unwrap(), bundle.to_str().unwrap()));
    assert!(!o.status.success());
    let e = String::from_utf8_lossy(&o.stderr);
    let head = git(&repo, &["rev-parse", "HEAD"]);
    assert!(
        e.contains(&format!(
            "this tool and its library were built from {}, and --repo is at {head}",
            turbo_bench::BUILD_COMMIT
        )),
        "{e}"
    );
    assert!(!out.exists(), "nothing is measured or written");
}

#[test]
fn the_build_is_named_only_when_it_is_the_commit_and_clean() {
    let c = "0123456789abcdef0123456789abcdef01234567";
    turbo_bench::check_build(c, c, "").unwrap();
    let other = "f".repeat(40);
    assert!(turbo_bench::check_build(c, &other, "").unwrap_err().contains("were built from ffff"));
    let e = turbo_bench::check_build(c, c, " M core/src/lib.rs").unwrap_err();
    assert!(e.contains("with changes in the working tree ( M core/src/lib.rs)"), "{e}");
    assert!(turbo_bench::check_build(c, "", "").unwrap_err().contains("not built in a git working tree"));
}

#[test]
fn the_tool_refuses_a_bundle_from_testdata() {
    let root = scratch("tool-testdata");
    let repo = pushed_repo(&root);
    let out = root.join("records");
    let o = tool(&record_args(repo.to_str().unwrap(), out.to_str().unwrap(), tiny_bundle().to_str().unwrap()));
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("a test fixture under"), "{o:?}");
    // Nor through a link, or a path that only reaches it with `..`.
    let link = root.join("linked");
    std::os::unix::fs::symlink(tiny_bundle(), &link).unwrap();
    let dotted = workspace().join("bench/../testdata/tiny-bert-bundle");
    for b in [link, dotted] {
        let o = tool(&record_args(repo.to_str().unwrap(), out.to_str().unwrap(), b.to_str().unwrap()));
        assert!(String::from_utf8_lossy(&o.stderr).contains("a test fixture under"), "{o:?}");
    }
    assert!(!out.exists());
}

#[test]
fn the_tool_refuses_a_dirty_tree_before_measuring() {
    let root = scratch("tool-dirty");
    let repo = pushed_repo(&root);
    std::fs::write(repo.join("README"), "edited\n").unwrap();
    let out = root.join("records");
    let bundle = bundle_copy(&root.join("bundle"));
    let o = tool(&record_args(repo.to_str().unwrap(), out.to_str().unwrap(), bundle.to_str().unwrap()));
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("the working tree is not clean"));
    assert!(!out.exists(), "nothing is written");
}

#[test]
fn the_tool_wants_each_reference_named_or_disabled() {
    let root = scratch("tool-references");
    let repo = pushed_repo(&root);
    let (repo, out, bundle) = (repo.to_str().unwrap().to_owned(), root.join("o"), bundle_copy(&root.join("b")));
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
