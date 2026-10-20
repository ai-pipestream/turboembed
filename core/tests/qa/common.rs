#![allow(dead_code)]
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};
use turbo::bundle::sha256_hex;
static SERIAL: AtomicUsize = AtomicUsize::new(0);
pub struct Fixture {
    pub dir: PathBuf,
    pub manifest: Value,
}
impl Fixture {
    pub fn new(root: &Path) -> Self {
        let dir = root.join("target/qa-tests").join(format!(
            "{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(dir.join("weights")).unwrap();
        fs::create_dir_all(dir.join("reference")).unwrap();
        let tiny = root.join("testdata/tiny-bert-bundle");
        let reference = root.join("testdata/tiny-qa-reference");
        let spec: Value = serde_json::from_slice(&fs::read(reference.join("cases.json")).unwrap()).unwrap();
        let mut manifest: Value = serde_json::from_slice(&fs::read(tiny.join("manifest.json")).unwrap()).unwrap();
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
    pub fn refresh(&mut self, name: &str) {
        let bytes = fs::read(self.dir.join(name)).unwrap();
        let file = self.manifest["files"].as_array_mut().unwrap().iter_mut().find(|f| f["path"] == name).unwrap();
        file["size"] = (bytes.len() as u64).into();
        file["sha256"] = sha256_hex(&bytes).into();
        self.save();
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}
