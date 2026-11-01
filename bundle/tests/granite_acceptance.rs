use std::fs;
use std::path::Path;

use serde_json::Value;
use turbo_bundle::{fetch, licence::Consent, recipe::Recipe};
use turbo_zoo::{Acceptances, Licence, Requirement, Sha256, Shown, UserId};

const GRANITE_TERMS: &str = "https://huggingface.co/ibm-granite/granite-embedding-30m-english/blob/9b5b096411652ec1189c68fcfb90d0a82c5b45af/README.md";

#[test]
fn granite_fetch_requires_acceptance_even_with_no_files_to_download() {
    let root = std::env::temp_dir().join(format!("granite-acceptance-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir(&root).unwrap();
    let dir = root.join("upstream");
    let mut recipe = Recipe::load(Path::new("ibm-granite/granite-embedding-30m-english")).unwrap();
    // Exercise the real acceptance/provenance path without a network or
    // model download. The notice and pinned model identity are unchanged.
    recipe.upstream.clear();
    let error = fetch::fetch(&recipe, &dir, false).unwrap_err();
    assert!(error.contains("--accept-terms") && error.contains(GRANITE_TERMS), "{error}");
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
fn granite_fetch_succeeds_without_flag_once_accepted_on_the_page_and_fails_when_not() {
    let root = std::env::temp_dir().join(format!("granite-page-store-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(root.join("home")).unwrap();
    let store_path = root.join("home/acceptances.json");
    let alice: UserId = "alice".parse().unwrap();
    let mut recipe = Recipe::load(Path::new("ibm-granite/granite-embedding-30m-english")).unwrap();
    recipe.upstream.clear();

    // Without an acceptance in the store, fetch fails naming the terms address.
    let no_flag = Consent { flag: false, store: Some((store_path.clone(), alice.clone())) };
    let error = fetch::fetch_with(&recipe, &root.join("up-refused"), &no_flag).unwrap_err();
    assert!(error.contains(GRANITE_TERMS) && error.contains("--accept-terms"), "{error}");
    assert!(!root.join("up-refused").exists());

    // Record the acceptance as the Licences page does (accept_shown with text_sha256).
    let req = Requirement { licence: Licence::RestrictedPublic, text: GRANITE_TERMS.into() };
    let shown = Shown { artifact: None, text_sha256: Some(Sha256::of(GRANITE_TERMS.as_bytes())) };
    let mut acceptances = Acceptances::default();
    acceptances.accept_shown(&alice, &req, &shown, 1_792_000_000).unwrap();
    acceptances.save(&store_path).unwrap();

    // Now fetch_with and the CLI both succeed without --accept-terms.
    fetch::fetch_with(&recipe, &root.join("up-accepted"), &no_flag).unwrap();
    assert!(root.join("up-accepted").join(fetch::PROVENANCE).is_file());

    let recipe_file = root.join("granite-empty-upstream.json");
    let mut raw: Value = serde_json::from_str(
        &fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("recipes/granite-embedding-30m-english.json"))
            .unwrap(),
    )
    .unwrap();
    raw["manifest"]["artifacts"][0].as_object_mut().unwrap().remove("produced_by");
    raw["upstream"] = Value::Array(Vec::new());
    fs::write(&recipe_file, serde_json::to_vec_pretty(&raw).unwrap()).unwrap();

    let cli_up = root.join("up-cli-accepted");
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_turbo-bundle"))
        .env("TURBO_ZOO_HOME", root.join("home"))
        .env("USER", "alice")
        .arg("fetch")
        .arg(&recipe_file)
        .arg(&cli_up)
        .output()
        .unwrap();
    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    assert!(cli_up.join(fetch::PROVENANCE).is_file());

    // Revoking the acceptance makes CLI fetch without --accept-terms fail again with the terms address.
    acceptances.revoke(&alice, &req);
    acceptances.save(&store_path).unwrap();
    let cli_revoked = root.join("up-cli-revoked");
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_turbo-bundle"))
        .env("TURBO_ZOO_HOME", root.join("home"))
        .env("USER", "alice")
        .arg("fetch")
        .arg(&recipe_file)
        .arg(&cli_revoked)
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains(GRANITE_TERMS) && stderr.contains("--accept-terms"), "{stderr}");
    assert!(!cli_revoked.exists());

    fs::remove_dir_all(root).unwrap();
}

#[test]
fn cli_fetch_and_make_require_consent_for_path_and_catalogue_id() {
    let root = std::env::temp_dir().join(format!("granite-cli-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir(&root).unwrap();
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("recipes/granite-embedding-30m-english.json");
    for (i, recipe) in [Path::new("ibm-granite/granite-embedding-30m-english"), path.as_path()].iter().enumerate() {
        for action in ["fetch", "make"] {
            let upstream = root.join(format!("upstream-{i}-{action}"));
            let bundle = root.join(format!("bundle-{i}-{action}"));
            let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_turbo-bundle"));
            command.env("TURBO_ZOO_HOME", root.join("empty-zoo-home")).env("USER", "alice");
            command.arg(action).arg(recipe).arg(&upstream);
            if action == "make" {
                command.arg(&bundle);
            }
            let result = command.output().unwrap();
            assert!(!result.status.success());
            let error = String::from_utf8_lossy(&result.stderr);
            assert!(error.contains("--accept-terms") && error.contains(GRANITE_TERMS), "{error}");
            assert!(!upstream.exists() && !bundle.exists());
        }
    }
    fs::remove_dir_all(root).unwrap();
}
