//! Regression guard over the retired shared-layer namespace (`LEGACY_NAMESPACE`
//! in `src/ids.rs`, #1773, #1834).
//!
//! ## Scope
//!
//! Active committed source, tests, scripts, and documentation must not spell
//! any layer ID in the retired namespace: whether its suffix is one of the
//! historical aliases in [`legacy_ids`] or a previously unseen name such as
//! `<LEGACY_NAMESPACE>future-layer`, and whether the ID appears bare, with an
//! `@instance` suffix, or nested inside a generic layer name such as
//! `turbo:categories<<LEGACY_NAMESPACE>sentences>`.
//!
//! Intentional namespace-only compatibility checks (such as
//! `id.starts_with(LEGACY_NAMESPACE)` with no suffix after the colon), Rust
//! `::` module paths, struct field type annotations (`field: Type`), and outer
//! URIs, Maven coordinates, or filesystem paths (`urn:...`, `org.apache....`,
//! `/path:ro`) do not name a layer ID and are permitted.
//!
//! ## Exception Policy
//!
//! Broad directory-prefix exemptions are forbidden so a newly added source
//! file under `analysis/eval/records/`, `benchmarks/records/`, or
//! `docs/receipts/`, or a file sharing the prefix of `analysis/core/src/ids.rs`,
//! is always scanned:
//!
//! 1. [`EXACT_EXCEPTIONS`] lists individual committed files by exact path,
//!    each paired with its stated reason: the centralized compatibility
//!    mapping module (`analysis/core/src/ids.rs`), the binary pre-rename graph
//!    fixture (`analysis/layers/tests/fixtures/legacy-layer-ids.tgrf`), the
//!    negative reader/exporter refusal tests that assert malformed or
//!    retired-namespace inputs are rejected, and two frozen historical receipt
//!    files under `docs/receipts/`.
//! 2. [`HISTORICAL_RECORD_DIRS`] permits only `.json` files under
//!    `analysis/eval/records/` and `benchmarks/records/` to retain known
//!    historical layer or model IDs from [`legacy_ids`] and
//!    [`legacy_model_ids`]. Non-JSON files under those directories and
//!    previously unseen retired-namespace suffixes are still rejected.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use turbo_analysis::ids::{LEGACY_NAMESPACE, legacy_ids, legacy_model_ids};

/// Exact file paths exempt from the active-source layer-ID scan, each paired
/// with its reason. Matched by exact path equality (`path == entry`), never by
/// prefix.
const EXACT_EXCEPTIONS: &[(&str, &str)] = &[
    ("analysis/core/src/ids.rs", "Centralized legacy-to-current layer and model ID compatibility mapping table"),
    (
        "analysis/layers/tests/fixtures/legacy-layer-ids.tgrf",
        "Stored pre-rename DocumentGraph binary fixture for reader canonicalization tests",
    ),
    (
        "analysis/goldens/tests/edge_cases/rows.rs",
        "Golden reader negative test refusing a mismatched legacy-namespace block end line",
    ),
    (
        "analysis/goldens/tests/goldens.rs",
        "Golden reader negative test refusing an unknown legacy-namespace layer header",
    ),
    (
        "analysis/goldens/tests/reader_edge_cases.rs",
        "Golden reader negative test refusing unknown legacy-namespace layer shapes",
    ),
    ("bundle/tests/postag.rs", "Bundle exporter negative test refusing a retired-namespace model ID"),
    ("docs/receipts/document-trace/main.rs", "Frozen historical document-trace receipt harness"),
    ("docs/receipts/progressive-snapshot-adoption-1249.md", "Frozen historical #1249 progressive-snapshot receipt"),
];

/// Directories whose committed `.json` files hold pre-rename evaluation or
/// benchmark records. Only `.json` files under these directories may contain
/// known historical IDs from [`legacy_ids`] or [`legacy_model_ids`]; any
/// source file (`.rs`, `.md`, scripts) or unseen retired-namespace ID under
/// these directories is still rejected.
const HISTORICAL_RECORD_DIRS: &[(&str, &str)] = &[
    ("analysis/eval/records/", "Pre-rename evaluation record JSON files"),
    ("benchmarks/records/", "Pre-rename benchmark record JSON files"),
];

fn is_exact_exception(path: &str) -> bool {
    EXACT_EXCEPTIONS.iter().any(|&(entry, _)| path == entry)
}

fn is_historical_record_json(path: &str) -> bool {
    Path::new(path).extension().is_some_and(|ext| ext == "json")
        && HISTORICAL_RECORD_DIRS.iter().any(|&(dir, _)| path.starts_with(dir))
}

/// Whether `prefix` (the text immediately before `LEGACY_NAMESPACE`) continues
/// an outer identifier, URI scheme, Maven coordinate, or filesystem path
/// rather than starting a bare or nested layer ID. Escaped control characters
/// in string literals (`\t`, `\n`, `\r`) are field delimiters, not identifier
/// prefixes.
fn continues_outer_token(prefix: &str) -> bool {
    let mut chars = prefix.chars().rev();
    let Some(prev) = chars.next() else { return false };
    if matches!(prev, 't' | 'n' | 'r') && chars.next() == Some('\\') {
        return false;
    }
    prev.is_alphanumeric() || matches!(prev, '_' | '-' | ':' | '.' | '/')
}

/// Extracts every `<LEGACY_NAMESPACE><suffix>` identifier in `text`, in order
/// of appearance. A bare namespace prefix with no `<suffix>` (such as
/// `starts_with(LEGACY_NAMESPACE)`), a Rust `::` path, or a `field: Type`
/// annotation has an empty suffix and is not a layer ID.
fn legacy_namespace_ids(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for (at, _) in text.match_indices(LEGACY_NAMESPACE) {
        if continues_outer_token(&text[..at]) {
            continue;
        }
        let tail = &text[at + LEGACY_NAMESPACE.len()..];
        let suffix_len = tail
            .char_indices()
            .find(|&(_, c)| !(c.is_alphanumeric() || matches!(c, '-' | '_')))
            .map_or(tail.len(), |(i, _)| i);
        if suffix_len == 0 {
            continue;
        }
        let id = format!("{LEGACY_NAMESPACE}{}", &tail[..suffix_len]);
        if !out.contains(&id) {
            out.push(id);
        }
    }
    out
}

fn known_tables() -> (BTreeMap<String, &'static str>, BTreeSet<String>) {
    let layers: BTreeMap<String, &'static str> = legacy_ids().collect();
    let models: BTreeSet<String> = legacy_model_ids().map(|(old, _)| old).collect();
    (layers, models)
}

/// Scans one file's text under the policy for `path` and returns any
/// violations formatted as `<path>: <id> (...)`.
fn scan_file_text(
    path: &str,
    text: &str,
    known_layers: &BTreeMap<String, &'static str>,
    known_models: &BTreeSet<String>,
) -> Vec<String> {
    if is_exact_exception(path) {
        return Vec::new();
    }
    let historical_json = is_historical_record_json(path);
    let mut violations = Vec::new();
    for id in legacy_namespace_ids(text) {
        if historical_json && (known_layers.contains_key(&id) || known_models.contains(&id)) {
            continue;
        }
        match known_layers.get(&id) {
            Some(new) => violations.push(format!("{path}: {id} (now {new})")),
            None => violations.push(format!("{path}: {id} (retired namespace {LEGACY_NAMESPACE})")),
        }
    }
    violations
}

fn list_tree_paths(root: &Path) -> Vec<String> {
    if root.join(".git").exists() {
        let out = std::process::Command::new("git")
            .current_dir(root)
            .args(["ls-files", "-z"])
            .output()
            .expect("git ls-files");
        assert!(out.status.success(), "git could not list the tree");
        let stdout = String::from_utf8(out.stdout).expect("UTF-8 paths");
        return stdout.split('\0').filter(|p| !p.is_empty()).map(str::to_owned).collect();
    }
    let mut out = Vec::new();
    collect_dir_paths(root, root, &mut out);
    out.sort();
    out
}

fn collect_dir_paths(root: &Path, dir: &Path, out: &mut Vec<String>) {
    let Ok(rd) = fs::read_dir(dir) else { return };
    for entry in rd.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_dir_paths(root, &path, out);
        } else if path.is_file()
            && let Ok(rel) = path.strip_prefix(root)
        {
            out.push(rel.to_string_lossy().replace('\\', "/"));
        }
    }
}

/// Scans `root` under the guard's exception policy and returns
/// `(text_files_scanned, violations)`.
fn scan_root(root: &Path) -> (usize, Vec<String>) {
    let (known_layers, known_models) = known_tables();
    let mut seen = 0;
    let mut found = Vec::new();
    for path in list_tree_paths(root) {
        if is_exact_exception(&path) {
            continue;
        }
        // Binary files (graph fixtures, models) are not source text.
        let Ok(text) = fs::read_to_string(root.join(&path)) else { continue };
        seen += 1;
        found.extend(scan_file_text(&path, &text, &known_layers, &known_models));
    }
    (seen, found)
}

#[test]
fn exact_exceptions_and_historical_dirs_are_well_formed_and_present() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for &(entry, reason) in EXACT_EXCEPTIONS {
        assert!(!entry.trim().is_empty(), "empty exception path");
        assert!(!entry.ends_with('/'), "directory prefix {entry} is not permitted in EXACT_EXCEPTIONS");
        assert!(!reason.trim().is_empty(), "exception {entry} has no stated reason");
        assert!(root.join(entry).is_file(), "exception path {entry} does not exist");
    }
    for &(dir, reason) in HISTORICAL_RECORD_DIRS {
        assert!(dir.ends_with('/'), "historical record directory {dir} must end with '/'");
        assert!(!reason.trim().is_empty(), "historical record directory {dir} has no stated reason");
        assert!(root.join(dir).is_dir(), "historical record directory {dir} does not exist");
    }
}

#[test]
fn no_committed_source_writes_a_legacy_layer_id() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let (seen, found) = scan_root(&root);
    assert!(seen > 1000, "only {seen} files scanned");
    assert!(found.is_empty(), "legacy layer ids in committed sources:\n  {}", found.join("\n  "));
}

#[test]
fn scanner_rejects_known_and_unseen_legacy_ids_and_new_sources_in_record_dirs() {
    let dir = std::env::temp_dir().join(format!("turbo-legacy-layer-guard-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let write = |rel: &str, content: &str| {
        let p = dir.join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, content).unwrap();
    };

    let known_tok = format!("{LEGACY_NAMESPACE}tokens");
    let known_sent = format!("{LEGACY_NAMESPACE}sentences");
    let known_sub = format!("{LEGACY_NAMESPACE}subwords");
    let unseen = format!("{LEGACY_NAMESPACE}future-layer");
    let unseen_nested = format!("{LEGACY_NAMESPACE}unseen_ edge");

    // 1. Positive controls: current `turbo:` IDs (bare, nested, @instance),
    // intentional namespace-only compatibility checks, Rust `::` paths,
    // struct fields, URIs, paths, Maven coordinates, and exact exceptions.
    write(
        "analysis/core/src/current_positive.rs",
        &format!(
            "const A: &str = \"turbo:sentences\";\n\
             const B: &str = \"turbo:categories<turbo:sentences>\";\n\
             const C: &str = \"turbo:edges<turbo:entities, turbo:tokens>\";\n\
             const D: &str = \"turbo:pos@gold\";\n\
             fn is_retired(id: &str) -> bool {{ id.starts_with(\"{LEGACY_NAMESPACE}\") }}\n\
             pub use opennlp::OpenNlpBpe;\n\
             struct Holder {{ opennlp: String }}\n\
             const DTD: &str = \"urn:{LEGACY_NAMESPACE}unused-dtd\";\n\
             const VOL: &str = \"/work:/opennlp:ro\";\n\
             const MAVEN: &str = \"org.apache.opennlp:opennlp-tools:2.5.4\";\n\
             const EMBEDDED: &str = \"acme{known_tok}\";\n"
        ),
    );
    write("analysis/core/src/ids.rs", &format!("const LEGACY_MAPPED: &str = \"{known_tok}\";\n"));
    write("docs/receipts/document-trace/main.rs", &format!("const TRACE: &str = \"{known_sent}\";\n"));
    write("analysis/eval/records/v1-172/cpu/scifact.json", &format!("{{\"analyzer\": [\"{known_sub}\"]}}\n"));
    write("benchmarks/records/postag/sample.json", &format!("{{\"id\": \"{LEGACY_NAMESPACE}BilstmPOSModel\"}}\n"));

    let (seen, clean_violations) = scan_root(&dir);
    assert_eq!(seen, 3, "exact exceptions are skipped before reading; 3 non-exempt files are scanned");
    assert!(clean_violations.is_empty(), "unexpected violations on positive controls: {clean_violations:?}");

    // 2. Negative controls:
    // - Known historical layer ID in active source (bare and nested).
    // - Previously unseen legacy-namespace layer ID in active source (bare and nested).
    // - New Rust source files under each formerly exempt directory
    //   (`analysis/eval/records/`, `benchmarks/records/`, `docs/receipts/`).
    // - Files sharing the prefix of `analysis/core/src/ids.rs` or an exact receipt exception.
    // - Unseen legacy-namespace layer ID inside a historical `.json` record.
    write("analysis/core/src/known_bare.rs", &format!("const L: &str = \"{known_tok}\";\n"));
    write("analysis/core/src/unseen_bare.rs", &format!("const L: &str = \"{unseen}\";\n"));
    write(
        "analysis/core/src/nested_ids.rs",
        &format!(
            "const N1: &str = \"turbo:categories<{known_sent}>\";\n\
             const N2: &str = \"turbo:edges<turbo:entities, {unseen_nested}>\";\n\
             const N3: &str = \"{LEGACY_NAMESPACE}dependencies<turbo:words>\";\n\
             const N4: &str = \"{LEGACY_NAMESPACE}pos@gold\";\n\
             const TSV: &str = \"layer\\t{unseen}\\ttext\";\n"
        ),
    );
    write("analysis/eval/records/new_source.rs", &format!("const R: &str = \"{known_sub}\";\n"));
    write("benchmarks/records/new_source.rs", &format!("const R: &str = \"{known_tok}\";\n"));
    write("docs/receipts/new_source.rs", &format!("const R: &str = \"{unseen}\";\n"));
    write("analysis/core/src/ids.rs.extra", &format!("const P: &str = \"{known_tok}\";\n"));
    write("analysis/core/src/ids_extra.rs", &format!("const P: &str = \"{unseen}\";\n"));
    write("docs/receipts/document-trace/main.rs.extra", &format!("const P: &str = \"{known_sent}\";\n"));
    write("analysis/eval/records/v1-172/cpu/unseen.json", &format!("{{\"analyzer\": [\"{unseen}\"]}}\n"));

    let (_, violations) = scan_root(&dir);
    assert_eq!(
        violations,
        vec![
            format!("analysis/core/src/ids.rs.extra: {known_tok} (now turbo:tokens)"),
            format!("analysis/core/src/ids_extra.rs: {unseen} (retired namespace {LEGACY_NAMESPACE})"),
            format!("analysis/core/src/known_bare.rs: {known_tok} (now turbo:tokens)"),
            format!("analysis/core/src/nested_ids.rs: {known_sent} (now turbo:sentences)"),
            format!(
                "analysis/core/src/nested_ids.rs: {LEGACY_NAMESPACE}unseen_ (retired namespace {LEGACY_NAMESPACE})"
            ),
            format!("analysis/core/src/nested_ids.rs: {LEGACY_NAMESPACE}dependencies (now turbo:dependencies)"),
            format!("analysis/core/src/nested_ids.rs: {LEGACY_NAMESPACE}pos (now turbo:pos)"),
            format!("analysis/core/src/nested_ids.rs: {unseen} (retired namespace {LEGACY_NAMESPACE})"),
            format!("analysis/core/src/unseen_bare.rs: {unseen} (retired namespace {LEGACY_NAMESPACE})"),
            format!("analysis/eval/records/new_source.rs: {known_sub} (now turbo:subwords)"),
            format!("analysis/eval/records/v1-172/cpu/unseen.json: {unseen} (retired namespace {LEGACY_NAMESPACE})"),
            format!("benchmarks/records/new_source.rs: {known_tok} (now turbo:tokens)"),
            format!("docs/receipts/document-trace/main.rs.extra: {known_sent} (now turbo:sentences)"),
            format!("docs/receipts/new_source.rs: {unseen} (retired namespace {LEGACY_NAMESPACE})"),
        ]
    );

    fs::remove_dir_all(&dir).unwrap();
}
