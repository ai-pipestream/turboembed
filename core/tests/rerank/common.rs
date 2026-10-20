#![allow(dead_code)]
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};
use turbo::{
    bundle::sha256_hex,
    safetensors::{Dtype, File},
};
static SERIAL: AtomicUsize = AtomicUsize::new(0);
pub struct Fixture {
    pub dir: PathBuf,
    pub manifest: Value,
}
impl Fixture {
    pub fn new(root: &Path) -> Self {
        let dir = root.join("target/rerank-tests").join(format!(
            "{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(dir.join("weights")).unwrap();
        fs::create_dir_all(dir.join("reference")).unwrap();
        let tiny = root.join("testdata/tiny-bert-bundle");
        let reference = root.join("testdata/tiny-rerank-reference");
        let mut manifest: Value = serde_json::from_slice(&fs::read(tiny.join("manifest.json")).unwrap()).unwrap();
        manifest.as_object_mut().unwrap().remove("embed");
        manifest["task"] = "TASK_RERANK".into();
        manifest["rerank"] = json!({"max_seq":32,"max_batch":4,"query_tokens":8,"pair_template":"PAIR_BERT"});
        manifest["classifier"] = json!({"kind":"CLASSIFIER_SEQUENCE","labels":["LABEL_0"]});
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
            serde_json::from_slice(&fs::read(reference.join("cases.json")).unwrap()).unwrap();
        let mut files = Vec::new();
        for (src, dest) in [
            (tiny.join("tokenizer.json"), "tokenizer.json"),
            (tiny.join("weights/model.safetensors"), "weights/model.safetensors"),
            (reference.join("head.safetensors"), "weights/head.safetensors"),
            (reference.join("reference.safetensors"), "reference/reference.safetensors"),
        ] {
            let b = fs::read(src).unwrap();
            fs::write(dir.join(dest), &b).unwrap();
            files.push(json!({"path":dest,"size":b.len(),"sha256":sha256_hex(&b)}));
        }
        manifest["files"] = files.into();
        let f = Self { dir, manifest };
        f.save();
        f
    }
    pub fn save(&self) {
        fs::write(self.dir.join("manifest.json"), serde_json::to_vec_pretty(&self.manifest).unwrap()).unwrap();
    }
    pub fn reference(&self) -> Vec<f32> {
        let b = fs::read(self.dir.join("reference/reference.safetensors")).unwrap();
        File::parse("ref", &b).unwrap().get("scores", Dtype::F32, 2).unwrap().f32s()
    }
    pub fn cases(&self) -> Vec<(String, String)> {
        self.manifest["reference"]["cases"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| (c["text"].as_str().unwrap().into(), c["passage"].as_str().unwrap().into()))
            .collect()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}
