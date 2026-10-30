use std::fs;
use std::path::Path;

use turbo_bundle::{fetch, recipe::Recipe};

#[test]
fn nomic_fetch_requires_acceptance_for_its_ms_marco_lineage() {
    let root = std::env::temp_dir().join(format!("nomic-acceptance-{}", std::process::id()));
    fs::create_dir(&root).unwrap();
    let dir = root.join("upstream");
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("recipes/nomic-embed-text-v1.5.json");
    let mut recipe = Recipe::load(&path).unwrap();
    let notice = recipe.notice.clone().expect("the nomic recipe carries a notice");
    assert!(notice.contains("MS MARCO") && notice.contains("non-commercial"), "{notice}");
    // No network or download: the refusal comes before any file is fetched.
    recipe.upstream.clear();
    let error = fetch::fetch(&recipe, &dir, false).unwrap_err();
    assert!(error.contains("--accept-terms"), "{error}");
    assert!(!dir.exists(), "no destination files before acceptance");
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn nomic_cli_fetch_and_make_require_consent() {
    let root = std::env::temp_dir().join(format!("nomic-cli-{}", std::process::id()));
    fs::create_dir(&root).unwrap();
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("recipes/nomic-embed-text-v1.5.json");
    for action in ["fetch", "make"] {
        let upstream = root.join(format!("upstream-{action}"));
        let bundle = root.join(format!("bundle-{action}"));
        let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_turbo-bundle"));
        command.arg(action).arg(&path).arg(&upstream);
        if action == "make" {
            command.arg(&bundle);
        }
        let result = command.output().unwrap();
        assert!(!result.status.success());
        let error = String::from_utf8_lossy(&result.stderr);
        assert!(error.contains("--accept-terms"), "{error}");
        assert!(!upstream.exists() && !bundle.exists());
    }
    fs::remove_dir_all(root).unwrap();
}
