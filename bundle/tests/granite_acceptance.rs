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
