//! `turbo-train-head summarize`: one table from a set of `receipt.json`
//! files, with the peer rows (#1830).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use turbo::bundle::sha256_hex;

const COMMIT: &str = "9e9e8102836cbedb7120105f02785ba3dba00b05";

fn records() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("records")
}

/// A fresh directory `<name>` under a per-process, per-suite parent, so a
/// head row's label (its run directory's name) is `name` itself, the same
/// on every machine and run.
fn scratch(suite: &str, name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("turbo-summarize-1830-{}", std::process::id())).join(suite).join(name);
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

/// Stages a run directory: `receipt.json` from a committed record with
/// `commit` set and, where the record lists outputs, files whose digests
/// the receipt then records.
fn stage(suite: &str, name: &str, record: (&str, &str), commit: &str) -> PathBuf {
    let d = scratch(suite, name);
    let p = records().join(record.0).join(record.1);
    let mut v: serde_json::Value =
        serde_json::from_slice(&fs::read(&p).unwrap()).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    v["commit"] = serde_json::Value::String(commit.into());
    let mut outputs = Vec::new();
    for o in v["outputs"].as_array().cloned().unwrap_or_default() {
        let file = o["path"].as_str().unwrap();
        let bytes = format!("{suite}:{name}:{file}").into_bytes();
        fs::write(d.join(file), &bytes).unwrap();
        outputs.push(serde_json::json!({"path": file, "size": bytes.len(), "sha256": sha256_hex(&bytes)}));
    }
    v["outputs"] = serde_json::Value::Array(outputs);
    fs::write(d.join("receipt.json"), serde_json::to_vec_pretty(&v).unwrap()).unwrap();
    d
}

fn path(p: &Path) -> String {
    p.to_str().unwrap().to_owned()
}

fn summarize(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_turbo-train-head")).arg("summarize").args(args).output().unwrap()
}

fn stdout(o: &std::process::Output) -> String {
    assert!(o.status.success(), "stderr: {}", String::from_utf8_lossy(&o.stderr));
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn stderr(o: &std::process::Output) -> String {
    assert!(!o.status.success(), "expected failure, stdout: {}", String::from_utf8_lossy(&o.stdout));
    String::from_utf8_lossy(&o.stderr).into_owned()
}

fn at(text: &str, needle: &str) -> usize {
    text.find(needle).unwrap_or_else(|| panic!("{needle:?} not in:\n{text}"))
}

fn round4(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

#[test]
fn summarize_groups_two_corpora_with_peer_rows() {
    let head = stage("group", "masc-pos-B", ("masc-pos-1649", "head-B.json"), COMMIT);
    let base = stage("group", "majority", ("masc-pos-1649", "majority.json"), COMMIT);
    let wnut = stage("group", "wnut17", ("open-span-1735", "wnut17.json"), COMMIT);
    let raw = stage("group", "spacy-raw", ("masc-pos-1649", "spacy-trf-raw.json"), COMMIT);
    let mapped = stage("group", "spacy-mapped", ("masc-pos-1649", "spacy-trf-mapped.json"), COMMIT);
    let (h, b, w, r, m) = (path(&head), path(&base), path(&wnut), path(&raw), path(&mapped));
    let text = stdout(&summarize(&[h.as_str(), b.as_str(), w.as_str(), "--peer", r.as_str(), "--peer", m.as_str()]));
    assert!(text.contains("# masc-pos (tags, accuracy)"), "{text}");
    assert!(text.contains("# wnut17 (bio, span_f1)"), "{text}");
    // The numbers match the README tables of both records.
    for number in [
        "0.9581",
        "+0.0617 [+0.0595, +0.0639]",
        "0.8964",
        "0.8753",
        "-0.0211 [-0.0248, -0.0170]",
        "0.8853",
        "-0.0110 [-0.0147, -0.0073]",
        "0.2708",
        "32 run / best 32",
        "7 run / best 2",
        "head.safetensors",
    ] {
        assert!(text.contains(number), "{number} not in:\n{text}");
    }
    // Head variants first, the baseline beneath them, the peers last.
    let head_at = at(&text, "masc-pos-B");
    let base_at = at(&text, "majority baseline");
    let raw_at = at(&text, "spacy-3.8.7-en_core_web_trf-3.8.0 |");
    let mapped_at = at(&text, "spacy-3.8.7-en_core_web_trf-3.8.0 (mapped)");
    assert!(head_at < base_at && base_at < raw_at && raw_at < mapped_at, "{text}");
}

#[test]
fn summarize_json_out_scrubs_host() {
    let head = stage("scrub", "masc-pos-B", ("masc-pos-1649", "head-B.json"), COMMIT);
    let base = stage("scrub", "majority", ("masc-pos-1649", "majority.json"), COMMIT);
    let out = scratch("scrub", "out");
    let _ = fs::remove_dir_all(&out);
    let (h, b, o) = (path(&head), path(&base), path(&out));
    let text = stdout(&summarize(&[h.as_str(), b.as_str(), "--format", "json", "--out", o.as_str()]));
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["schema"], serde_json::Value::String("turboembed.trainer.summary/1".into()));
    assert_eq!(v["corpora"].as_array().unwrap().len(), 1);
    let rows = v["corpora"][0]["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["label"], serde_json::Value::String("masc-pos-B".into()), "the run dir name, no pid");
    assert_eq!(round4(rows[0]["model"].as_f64().unwrap()), 0.9581);
    assert_eq!(round4(rows[0]["baseline"].as_f64().unwrap()), 0.8964);
    assert_eq!(rows[0]["epochs_run"].as_u64().unwrap(), 32);
    assert_eq!(rows[0]["epochs_kept"].as_u64().unwrap(), 32);
    // epochs_kept is the receipt's 1-based best_epoch (#1804), so it lies in
    // 1..=epochs_run and the final epoch can be the kept one.
    let staged: serde_json::Value = serde_json::from_slice(&fs::read(head.join("receipt.json")).unwrap()).unwrap();
    assert_eq!(rows[0]["epochs_kept"], staged["training"]["best_epoch"]);
    let (run, kept) = (rows[0]["epochs_run"].as_u64().unwrap(), rows[0]["epochs_kept"].as_u64().unwrap());
    assert!((1..=run).contains(&kept), "kept {kept} outside 1..={run}");
    assert!(rows[1].get("model").is_none(), "the baseline row has no model score");
    let receipt = fs::read(head.join("receipt.json")).unwrap();
    assert_eq!(rows[0]["receipt_sha256"], serde_json::Value::String(sha256_hex(&receipt)));
    let written = fs::read(out.join("summary.json")).unwrap();
    assert_eq!(written.as_slice(), text.as_bytes(), "stdout matches the file");
    assert!(!String::from_utf8_lossy(&written).contains("host"), "the host field is scrubbed");
    assert!(out.join("summary.md").is_file());
    assert!(fs::read(out.join("summary.md")).unwrap().starts_with(b"# masc-pos (tags, accuracy)"));
}

#[test]
fn check_fails_on_mismatched_commits() {
    let a = stage("commits", "a", ("masc-pos-1649", "head-B.json"), COMMIT);
    let b = stage("commits", "b", ("masc-pos-1649", "majority.json"), "88cecf90f1158bcc018cd19a2e4204699e78c147");
    let err = stderr(&summarize(&["--check", path(&a).as_str(), path(&b).as_str()]));
    assert!(err.contains("differs"), "{err}");
    assert!(err.contains(COMMIT), "{err}");
}

#[test]
fn check_verifies_output_digests() {
    let h = stage("digests", "masc-pos-B", ("masc-pos-1649", "head-B.json"), COMMIT);
    let ok = summarize(&["--check", path(&h).as_str()]);
    assert!(ok.status.success(), "stderr: {}", String::from_utf8_lossy(&ok.stderr));
    fs::write(h.join("head.safetensors"), b"changed").unwrap();
    let err = stderr(&summarize(&["--check", path(&h).as_str()]));
    assert!(err.contains("sha256"), "{err}");
    assert!(err.contains("head.safetensors"), "{err}");
}

#[test]
fn check_verifies_a_colocated_summary() {
    let h = stage("stale", "masc-pos-B", ("masc-pos-1649", "head-B.json"), COMMIT);
    let row = |digest: String| {
        serde_json::json!({
            "label": "masc-pos-B", "kind": "head",
            "dir": ".", "receipt": "receipt.json",
            "receipt_sha256": digest, "commit": COMMIT,
            "model": 0.9581, "baseline": 0.8964, "wall_s": 1.0, "outputs": [],
        })
    };
    fs::write(
        h.join("summary.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema": "turboembed.trainer.summary/1", "command": ["summarize", "."],
            "commit": COMMIT,
            "corpora": [{"corpus": "masc-pos", "task": "tags", "metric": "accuracy", "rows": [row("0".repeat(64))]}],
        }))
        .unwrap(),
    )
    .unwrap();
    let err = stderr(&summarize(&["--check", path(&h).as_str()]));
    assert!(err.contains("summary says"), "{err}");
    let good = sha256_hex(&fs::read(h.join("receipt.json")).unwrap());
    fs::write(
        h.join("summary.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema": "turboembed.trainer.summary/1", "command": ["summarize", "."],
            "commit": COMMIT,
            "corpora": [{"corpus": "masc-pos", "task": "tags", "metric": "accuracy", "rows": [row(good)]}],
        }))
        .unwrap(),
    )
    .unwrap();
    let ok = summarize(&["--check", path(&h).as_str()]);
    assert!(ok.status.success(), "stderr: {}", String::from_utf8_lossy(&ok.stderr));
}

/// Copies the files of one directory (no subdirectories) into `to`.
fn copy_dir(from: &Path, to: &Path) {
    fs::create_dir_all(to).unwrap();
    for e in fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        fs::copy(e.path(), to.join(e.file_name())).unwrap();
    }
}

#[test]
fn check_verifies_the_copy_it_was_given() {
    // A run directory with its own summary, written with absolute paths
    // on the command line, as a trainer would run it.
    let h = stage("copy", "masc-pos-B", ("masc-pos-1649", "head-B.json"), COMMIT);
    stdout(&summarize(&[path(&h).as_str(), "--out", path(&h).as_str()]));
    let written = fs::read_to_string(h.join("summary.json")).unwrap();
    let tmp = std::env::temp_dir();
    let tmp = tmp.to_str().unwrap().trim_end_matches(['/', '\\']);
    assert!(!written.contains(tmp), "summary.json names a machine path:\n{written}");
    let v: serde_json::Value = serde_json::from_str(&written).unwrap();
    assert_eq!(v["command"], serde_json::json!(["summarize", "."]), "{written}");
    let row = &v["corpora"][0]["rows"][0];
    assert_eq!((row["dir"].as_str(), row["receipt"].as_str()), (Some("."), Some("receipt.json")), "{written}");

    // Copy it to a models folder, then delete the original: the copy
    // still checks, against itself.
    let models = scratch("copy", "models").join("masc-pos-B");
    copy_dir(&h, &models);
    fs::remove_dir_all(&h).unwrap();
    let ok = summarize(&["--check", path(&models).as_str()]);
    assert!(ok.status.success(), "stderr: {}", String::from_utf8_lossy(&ok.stderr));

    // A corrupted copied receipt fails through the summary.
    let receipt = models.join("receipt.json");
    let good = fs::read(&receipt).unwrap();
    let mut bad = good.clone();
    let digit = bad.iter().position(|b| b.is_ascii_digit()).unwrap();
    bad[digit] = if bad[digit] == b'9' { b'8' } else { bad[digit] + 1 };
    fs::write(&receipt, &bad).unwrap();
    let err = stderr(&summarize(&["--check", path(&models).as_str()]));
    assert!(err.contains("summary says"), "{err}");
    assert!(err.contains(path(&receipt).as_str()), "the error names the copy's receipt: {err}");

    // So does a corrupted copied output.
    fs::write(&receipt, &good).unwrap();
    fs::write(models.join("head.safetensors"), b"changed").unwrap();
    let err = stderr(&summarize(&["--check", path(&models).as_str()]));
    assert!(err.contains("head.safetensors"), "{err}");
}

#[test]
fn check_refuses_a_summary_with_absolute_paths() {
    let h = stage("absolute", "masc-pos-B", ("masc-pos-1649", "head-B.json"), COMMIT);
    let digest = sha256_hex(&fs::read(h.join("receipt.json")).unwrap());
    let row = serde_json::json!({
        "label": "masc-pos-B", "kind": "head",
        "dir": path(&h), "receipt": path(&h.join("receipt.json")),
        "receipt_sha256": digest, "commit": COMMIT,
        "model": 0.9581, "baseline": 0.8964, "wall_s": 1.0, "outputs": [],
    });
    let summary = serde_json::json!({
        "schema": "turboembed.trainer.summary/1", "command": ["summarize", "."], "commit": COMMIT,
        "corpora": [{"corpus": "masc-pos", "task": "tags", "metric": "accuracy", "rows": [row]}],
    });
    fs::write(h.join("summary.json"), serde_json::to_vec_pretty(&summary).unwrap()).unwrap();
    let err = stderr(&summarize(&["--check", path(&h).as_str()]));
    assert!(err.contains("absolute path"), "{err}");
}

#[test]
fn summary_paths_are_relative_to_the_out_directory() {
    let head = stage("relative", "masc-pos-B", ("masc-pos-1649", "head-B.json"), COMMIT);
    let base = stage("relative", "majority", ("masc-pos-1649", "majority.json"), COMMIT);
    let out = scratch("relative", "out");
    let (h, b, o) = (path(&head), path(&base), path(&out));
    stdout(&summarize(&[h.as_str(), "--peer", b.as_str(), "--out", o.as_str(), "--check"]));
    let v: serde_json::Value = serde_json::from_slice(&fs::read(out.join("summary.json")).unwrap()).unwrap();
    assert_eq!(v["command"], serde_json::json!(["summarize", "../masc-pos-B", "--peer", "../majority", "--check"]));
    let rows = v["corpora"][0]["rows"].as_array().unwrap();
    assert_eq!(rows[0]["dir"], serde_json::json!("../masc-pos-B"));
    assert_eq!(rows[0]["receipt"], serde_json::json!("../masc-pos-B/receipt.json"));
    // The out directory and the runs move together: the summary still checks.
    let moved = scratch("relative-moved", "x");
    for name in ["masc-pos-B", "majority", "out"] {
        copy_dir(&out.parent().unwrap().join(name), &moved.join(name));
    }
    fs::remove_dir_all(out.parent().unwrap()).unwrap();
    let ok = summarize(&["--check", path(&moved.join("masc-pos-B")).as_str(), path(&moved.join("majority")).as_str()]);
    assert!(ok.status.success(), "stderr: {}", String::from_utf8_lossy(&ok.stderr));
}
