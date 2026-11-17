//! The small QA and rerank bundles the tool's tests measure: the small
//! BERT bundle's encoder with the head and the reference cases of
//! testdata/tiny-qa-reference and testdata/tiny-rerank-reference, written
//! under a directory the test gives. Shared by the unit tests under src/
//! and the tests of the built tool.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use turbo::bundle::sha256_hex;

/// The extractive QA bundle, at `dir`/bundle.
pub fn tiny_qa_bundle(dir: &Path) -> PathBuf {
    let bundle = dir.join("bundle");
    std::fs::create_dir_all(bundle.join("weights")).unwrap();
    std::fs::create_dir_all(bundle.join("reference")).unwrap();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let tiny = root.join("testdata/tiny-bert-bundle");
    let reference = root.join("testdata/tiny-qa-reference");
    let spec: Value = serde_json::from_slice(&std::fs::read(reference.join("cases.json")).unwrap()).unwrap();
    let mut manifest: Value = serde_json::from_slice(&std::fs::read(tiny.join("manifest.json")).unwrap()).unwrap();
    manifest.as_object_mut().unwrap().remove("embed");
    manifest["model"]["id"] = "turbo/test-tiny-qa".into();
    manifest["task"] = "TASK_EXTRACTIVE_QA".into();
    manifest["qa"] = spec["qa"].clone();
    manifest["classifier"] = json!({"kind":"CLASSIFIER_TOKEN","labels":["start","end"]});
    manifest["artifacts"][0]["files"] = json!(["weights/model.safetensors", "weights/head.safetensors"]);
    manifest["artifacts"][0]["tensor_names"]["classifier_weight"] = "qa_outputs.weight".into();
    manifest["artifacts"][0]["tensor_names"]["classifier_bias"] = "qa_outputs.bias".into();
    let mut cases = spec["cases"].clone();
    for c in cases.as_array_mut().unwrap() {
        c["prompt_role"] = "PROMPT_NONE".into();
    }
    manifest["reference"]["cases"] = cases;
    manifest["reference"]["produced_by"]["tool"] = "transformers".into();
    let mut files = Vec::new();
    for (src, dest) in [
        (tiny.join("tokenizer.json"), "tokenizer.json"),
        (tiny.join("weights/model.safetensors"), "weights/model.safetensors"),
        (reference.join("head.safetensors"), "weights/head.safetensors"),
        (reference.join("reference.safetensors"), "reference/reference.safetensors"),
    ] {
        let b = std::fs::read(src).unwrap();
        std::fs::write(bundle.join(dest), &b).unwrap();
        files.push(json!({"path": dest, "size": b.len(), "sha256": sha256_hex(&b)}));
    }
    manifest["files"] = files.into();
    std::fs::write(bundle.join("manifest.json"), serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
    bundle
}

/// The rerank bundle, at `dir`/bundle.
pub fn tiny_rerank_bundle(dir: &Path) -> PathBuf {
    let bundle = dir.join("bundle");
    std::fs::create_dir_all(bundle.join("weights")).unwrap();
    std::fs::create_dir_all(bundle.join("reference")).unwrap();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let tiny = root.join("testdata/tiny-bert-bundle");
    let reference = root.join("testdata/tiny-rerank-reference");
    let mut manifest: Value = serde_json::from_slice(&std::fs::read(tiny.join("manifest.json")).unwrap()).unwrap();
    manifest.as_object_mut().unwrap().remove("embed");
    manifest["task"] = "TASK_RERANK".into();
    manifest["rerank"] = json!({"max_seq": 32, "max_batch": 4, "query_tokens": 8, "pair_template": "PAIR_BERT"});
    manifest["classifier"] = json!({"kind": "CLASSIFIER_SEQUENCE", "labels": ["LABEL_0"]});
    manifest["artifacts"][0]["files"] = json!(["weights/model.safetensors", "weights/head.safetensors"]);
    let names = manifest["artifacts"][0]["tensor_names"].as_object_mut().unwrap();
    for (k, v) in [
        ("pooler_weight", "bert.pooler.dense.weight"),
        ("pooler_bias", "bert.pooler.dense.bias"),
        ("classifier_weight", "classifier.weight"),
        ("classifier_bias", "classifier.bias"),
    ] {
        names.insert(k.into(), v.into());
    }
    manifest["reference"]["cases"] =
        serde_json::from_slice(&std::fs::read(reference.join("cases.json")).unwrap()).unwrap();
    let mut files = Vec::new();
    for (src, dest) in [
        (tiny.join("tokenizer.json"), "tokenizer.json"),
        (tiny.join("weights/model.safetensors"), "weights/model.safetensors"),
        (reference.join("head.safetensors"), "weights/head.safetensors"),
        (reference.join("reference.safetensors"), "reference/reference.safetensors"),
    ] {
        let b = std::fs::read(src).unwrap();
        std::fs::write(bundle.join(dest), &b).unwrap();
        files.push(json!({"path": dest, "size": b.len(), "sha256": sha256_hex(&b)}));
    }
    manifest["files"] = files.into();
    std::fs::write(bundle.join("manifest.json"), serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
    bundle
}
