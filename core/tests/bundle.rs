//! A real bundle, with its reference produced by the upstream pipeline:
//! set TURBO_TEST_BUNDLE to its directory and run with --ignored. Without
//! the variable the tests fail rather than passing.

mod common;

use std::path::PathBuf;

use common::*;
use serde_json::Value;
use turbo::*;

fn dir() -> PathBuf {
    std::env::var_os("TURBO_TEST_BUNDLE").expect("TURBO_TEST_BUNDLE is not set").into()
}

#[test]
#[ignore = "needs a real bundle directory in TURBO_TEST_BUNDLE"]
fn a_real_bundle_loads_and_matches_its_reference_ids() {
    // turbo_tokenizer_create runs loader rules 1 to 5, which include
    // encoding every reference case and comparing the ids exactly.
    let tok = Tok::create(&dir()).unwrap_or_else(|e| panic!("{e:?}"));
    let info = tok.info();
    assert!(info.vocab_size > 0 && info.max_seq > 0);
}

#[test]
#[ignore = "needs a real bundle directory in TURBO_TEST_BUNDLE"]
fn a_real_bundle_loads_on_the_cpu_as_its_manifest_says() {
    let dir = dir();
    let bytes = std::fs::read(dir.join("manifest.json")).unwrap();
    let m: Value = serde_json::from_slice(&bytes).unwrap();
    let hash = |path: &Value| {
        let files = m["files"].as_array().unwrap();
        files.iter().find(|f| f["path"] == *path).unwrap()["sha256"].as_str().unwrap().to_owned()
    };
    let art = m["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["backends"].as_array().unwrap().iter().any(|b| b == "cpu"));
    let art = art.expect("an artifact for the cpu");
    let l = Loaded::load(&dir).unwrap_or_else(|e| panic!("{e:?}"));
    let info = l.info();
    let e = &m["embed"];
    assert_eq!(info.task, TURBO_TASK_EMBED);
    assert_eq!(info.dim as u64, e["dim"].as_u64().unwrap());
    assert_eq!(info.max_seq as u64, e["max_seq"].as_u64().unwrap());
    assert_eq!(info.max_batch as u64, e["max_batch"].as_u64().unwrap());
    assert_eq!(info.dtype, TURBO_DTYPE_F32);
    assert_eq!(field(&info.model_id), m["model"]["id"].as_str().unwrap());
    assert_eq!(field(&info.revision), m["model"]["revision"].as_str().unwrap());
    assert_eq!(field(&info.manifest_sha256), sha256_hex(&bytes));
    assert_eq!(field(&info.tokenizer_sha256), hash(&m["tokenizer"]["file"]));
    assert_eq!(field(&info.artifact_sha256), hash(&art["files"][0]));
    assert_eq!(field(&info.prefix_query), e["prefix_query"].as_str().unwrap_or_default());
    assert_eq!(field(&info.prefix_document), e["prefix_document"].as_str().unwrap_or_default());
    // The CPU reads the weights in the core's one verified copy.
    let ModelWeights { files, held } = unsafe { model_weights(l.m) }.unwrap();
    let layers = m["architecture"]["layers"].as_u64().unwrap() as usize;
    let held = held.unwrap();
    assert_eq!(held.len(), 5 + 16 * layers);
    assert!(held.iter().all(|&p| files.iter().any(|f| f.as_ptr_range().contains(&(p as *const u8)))));
}
