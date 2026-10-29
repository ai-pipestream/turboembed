//! Every tested model is pinned: the bundles checked in, the bundles the
//! records name, the upstream files of every bundle recipe, and every
//! model a test loads from a path it is given (it goes through
//! turbo_pins::require).

use std::fs;
use std::path::{Path, PathBuf};

use turbo_pins::{Kind, is_hex64, lookup, sha256_hex};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

/// Every regular file under `dir`, skipping build output and dot
/// directories.
fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy();
        if path.is_dir() {
            if !(name.starts_with('.') || name == "target" || name == "node_modules") {
                walk(&path, out);
            }
        } else if path.is_file() {
            out.push(path);
        }
    }
}

fn files(under: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    walk(&root().join(under), &mut out);
    out.sort();
    out
}

fn rel(path: &Path) -> String {
    path.strip_prefix(root()).unwrap().to_string_lossy().replace('\\', "/")
}

fn pinned_as(hex: &str, kind: Kind) -> bool {
    lookup(hex).is_some_and(|p| p.kind == kind)
}

#[test]
fn every_checked_in_bundle_is_pinned_by_its_manifest() {
    let mut missing = Vec::new();
    let mut seen = 0;
    for f in files("") {
        if f.file_name().unwrap() != "manifest.json" {
            continue;
        }
        let bytes = fs::read(&f).unwrap();
        if !String::from_utf8_lossy(&bytes).contains("\"bundle_version\"") {
            continue;
        }
        seen += 1;
        let hex = sha256_hex(&bytes);
        if !pinned_as(&hex, Kind::Bundle) {
            missing.push(format!("{}  bundle  {}", turbo_pins::line_key(&hex), rel(f.parent().unwrap())));
        }
    }
    assert!(seen >= 6, "found {seen} bundles");
    assert!(missing.is_empty(), "bundles with no pin in analysis/pins/models.txt:\n{}", missing.join("\n"));
}

/// Every `"manifest_sha256": "<hex>"` of the records: the bundle a
/// benchmark, evaluation or receipt was made with.
#[test]
fn every_bundle_a_record_names_is_pinned() {
    const KEY: &str = "\"manifest_sha256\": \"";
    let mut missing = Vec::new();
    let mut seen = 0;
    for dir in ["benchmarks", "docs", "analysis/eval/records", "testdata"] {
        for f in files(dir) {
            let ext = f.extension().and_then(|e| e.to_str()).unwrap_or("");
            if ext != "json" && ext != "jsonl" {
                continue;
            }
            let text = fs::read_to_string(&f).unwrap();
            for (at, _) in text.match_indices(KEY) {
                let Some(hex) = text.get(at + KEY.len()..at + KEY.len() + 64).filter(|h| is_hex64(h)) else {
                    continue;
                };
                seen += 1;
                if !pinned_as(hex, Kind::Bundle) {
                    missing.push(format!("{hex} in {}", rel(&f)));
                }
            }
        }
    }
    assert!(seen > 100, "found {seen} record digests");
    missing.dedup();
    assert!(missing.is_empty(), "record bundles with no pin in analysis/pins/models.txt:\n{}", missing.join("\n"));
}

/// A recipe names its upstream at a commit and every file it fetches by
/// SHA-256, so the bundle fetch refuses anything else.
#[test]
fn every_bundle_recipe_pins_its_commit_and_every_upstream_file() {
    let mut bad = Vec::new();
    let mut seen = 0;
    for f in files("bundle/recipes") {
        if f.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        seen += 1;
        let recipe: serde_json::Value = serde_json::from_slice(&fs::read(&f).unwrap()).unwrap();
        let commit = recipe.pointer("/manifest/model/source/commit").and_then(|c| c.as_str()).unwrap_or("");
        if commit.len() != 40 || !commit.bytes().all(|b| b.is_ascii_hexdigit()) {
            bad.push(format!("{}: source commit {commit:?}", rel(&f)));
        }
        for u in recipe["upstream"].as_array().into_iter().flatten() {
            let sha = u["sha256"].as_str().unwrap_or("");
            if !is_hex64(sha) {
                bad.push(format!("{}: upstream {} has no SHA-256", rel(&f), u["path"]));
            }
        }
    }
    assert!(seen >= 30, "found {seen} recipes");
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

/// The environment variables that name a model for a test: a bundle, a
/// package, a checkpoint, a head, or a directory of models.
fn names_a_model(var: &str) -> bool {
    const MARKS: [&str; 8] = [
        "BUNDLE",
        "CHECKPOINT",
        "PACKAGE",
        "_HEAD",
        "OPENNLP_MODELS",
        "GOLDENS_MODELS",
        "SENTENCEPIECE_MODELS",
        "BILSTM_MODEL",
    ];
    (var.starts_with("TURBO_") || var.starts_with("GLINER") || var.starts_with("GLIREL"))
        && !var.starts_with("TURBO_E_")
        && !var.contains("REQUIRE")
        && (MARKS.iter().any(|m| var.contains(m)) || var == "GLIREL_SPM")
}

/// Sources that read such a variable and need no turbo_pins call, each for
/// the reason given.
const EXEMPT: [(&str, &str); 4] = [
    ("server/src/config.rs", "the server's own configuration of the bundles it serves, not a test"),
    ("core/tests/sentencepiece.rs", "pins each model it reads by SHA-512 in FETCHED and checks it at load"),
    (
        "docs/receipts/document-trace/main.rs",
        "a receipt built against a fixed older checkout; reference.json records its bundle's digest, pinned",
    ),
    (
        "analysis/postag/tests/bilstm_ewt.rs",
        "trains the model it then scores when TURBO_BILSTM_MODEL does not exist yet; the model is its own output",
    ),
];

/// The quoted upper case names in `text`: a `"`, then upper case letters,
/// digits and `_`, then a `"`.
fn quoted_names(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    for (at, _) in text.match_indices('"') {
        let name = &bytes[at + 1..];
        let len = name.iter().take_while(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || **b == b'_').count();
        if len > 6 && name.get(len) == Some(&b'"') {
            out.push(&text[at + 1..at + 1 + len]);
        }
    }
    out
}

#[test]
fn every_test_that_loads_a_model_from_a_path_checks_its_pin() {
    let mut bad = Vec::new();
    let mut seen = 0;
    for f in files("") {
        if f.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let name = rel(&f);
        if name.starts_with("analysis/pins/") || EXEMPT.iter().any(|(f, _)| *f == name) {
            continue;
        }
        let text = fs::read_to_string(&f).unwrap();
        let mut vars: Vec<&str> = quoted_names(&text).into_iter().filter(|v| names_a_model(v)).collect();
        if vars.is_empty() {
            continue;
        }
        seen += 1;
        if !text.contains("turbo_pins::") {
            vars.sort();
            vars.dedup();
            bad.push(format!("{name}: reads {} and never calls turbo_pins", vars.join(", ")));
        }
    }
    assert!(seen >= 40, "found {seen} sources");
    assert!(
        bad.is_empty(),
        "a test that loads a model from a path must pass it through turbo_pins::require:\n{}",
        bad.join("\n")
    );
}

#[test]
fn the_lint_knows_a_model_variable_from_a_flag() {
    for yes in ["TURBO_TEST_BUNDLE", "TURBO_TEST_RERANK_BUNDLES", "GLINER_PACKAGE", "TURBO_PRAGMATICS_HEAD"] {
        assert!(names_a_model(yes), "{yes}");
    }
    for no in
        ["TURBO_TEST_REQUIRE_OPENNLP_MODELS", "TURBO_E_BUNDLE_NOT_FOUND", "TURBO_TEST_DEVICE", "CARGO_MANIFEST_DIR"]
    {
        assert!(!names_a_model(no), "{no}");
    }
}
