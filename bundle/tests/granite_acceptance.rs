use std::fs;
use std::path::Path;

use serde_json::Value;
use turbo_bundle::{fetch, recipe::Recipe};

#[test]
fn granite_fetch_requires_acceptance_even_with_no_files_to_download() {
    let root = std::env::temp_dir().join(format!("granite-acceptance-{}", std::process::id()));
    fs::create_dir(&root).unwrap();
    let dir = root.join("upstream");
    let mut recipe = Recipe::load(Path::new("ibm-granite/granite-embedding-30m-english")).unwrap();
    // Exercise the real acceptance/provenance path without a network or
    // model download. The notice and pinned model identity are unchanged.
    recipe.upstream.clear();
    let error = fetch::fetch(&recipe, &dir, false).unwrap_err();
    assert!(error.contains("--accept-terms"), "{error}");
    assert!(!dir.exists(), "no destination files before acceptance");
    let notice = recipe.notice.as_deref().unwrap();
    assert!(notice.contains("CC BY-SA") && notice.contains("not distributed"));
    fetch::fetch(&recipe, &dir, true).unwrap();
    let record: Value = serde_json::from_slice(&fs::read(dir.join(fetch::PROVENANCE)).unwrap()).unwrap();
    assert_eq!(record["terms_accepted"], true);
    assert_eq!(record["notice"], notice);
    assert_eq!(record["revision"], "9b5b096411652ec1189c68fcfb90d0a82c5b45af");
    assert_eq!(record["license"], "Apache-2.0");
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn cli_fetch_and_make_require_consent_for_path_and_catalogue_id() {
    let root = std::env::temp_dir().join(format!("granite-cli-{}", std::process::id()));
    fs::create_dir(&root).unwrap();
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("recipes/granite-embedding-30m-english.json");
    for (i, recipe) in [Path::new("ibm-granite/granite-embedding-30m-english"), path.as_path()].iter().enumerate() {
        for action in ["fetch", "make"] {
            let upstream = root.join(format!("upstream-{i}-{action}"));
            let bundle = root.join(format!("bundle-{i}-{action}"));
            let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_turbo-bundle"));
            command.arg(action).arg(recipe).arg(&upstream);
            if action == "make" {
                command.arg(&bundle);
            }
            let result = command.output().unwrap();
            assert!(!result.status.success());
            let error = String::from_utf8_lossy(&result.stderr);
            assert!(error.contains("--accept-terms"), "{error}");
            assert!(!upstream.exists() && !bundle.exists());
        }
    }
    fs::remove_dir_all(root).unwrap();
}
