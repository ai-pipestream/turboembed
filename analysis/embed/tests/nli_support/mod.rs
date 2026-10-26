//! A sequence-classification NLI bundle made from the tiny test bundle:
//! its real tokenizer and encoder weights, plus a fixed classifier head
//! with the given outcome labels. Shared by the NLI scorer tests.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::{Value, json};
use turbo::bundle::sha256_hex;

pub struct TempDir(pub PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct Tensor {
    name: String,
    dtype: &'static str,
    shape: Vec<u64>,
    data: Vec<u8>,
}

fn tiny_bundle() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle")
}

fn safetensors_file(tensors: &[Tensor]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    for t in tensors {
        let end = offset + t.data.len();
        header.insert(
            t.name.clone(),
            json!({
                "dtype": t.dtype,
                "shape": t.shape,
                "data_offsets": [offset, end]
            }),
        );
        offset = end;
    }
    let mut hdr_bytes = serde_json::to_vec(&Value::Object(header)).unwrap();
    while !hdr_bytes.len().is_multiple_of(8) {
        hdr_bytes.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + hdr_bytes.len() + offset);
    out.extend_from_slice(&(hdr_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&hdr_bytes);
    for t in tensors {
        out.extend_from_slice(&t.data);
    }
    out
}

fn linspace_f32(start: f32, end: f32, count: usize) -> Vec<u8> {
    (0..count)
        .flat_map(|i| {
            let t = i as f32 / (count.saturating_sub(1).max(1)) as f32;
            (start + (end - start) * t).to_le_bytes()
        })
        .collect()
}

fn slice_f32(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[allow(dead_code)]
pub fn make_nli_bundle(name: &str, labels: &[&str], edit: impl FnOnce(&mut Value)) -> TempDir {
    make_nli_bundle_with(name, labels, None, edit)
}

/// As `make_nli_bundle`, with the classifier's weight `[labels, 32]` and
/// bias `[labels]` given in place of the fixed ones.
#[allow(dead_code)]
pub fn make_nli_bundle_with(
    name: &str,
    labels: &[&str],
    head: Option<(Vec<f32>, Vec<f32>)>,
    edit: impl FnOnce(&mut Value),
) -> TempDir {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("turbo-nli-{}-{name}-{id}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    for f in ["tokenizer.json", "reference/reference.safetensors"] {
        fs::create_dir_all(dir.join(f).parent().unwrap()).unwrap();
        fs::copy(tiny_bundle().join(f), dir.join(f)).unwrap();
    }
    let src_bytes = fs::read(tiny_bundle().join("weights/model.safetensors")).unwrap();
    let n = u64::from_le_bytes(src_bytes[..8].try_into().unwrap()) as usize;
    let raw_header: serde_json::Map<String, Value> = serde_json::from_slice(&src_bytes[8..8 + n]).unwrap();
    let data = &src_bytes[8 + n..];
    let mut tensors = Vec::new();
    for (k, v) in raw_header {
        if k == "__metadata__" {
            continue;
        }
        let shape: Vec<u64> = v["shape"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap()).collect();
        let offs = v["data_offsets"].as_array().unwrap();
        let (a, b) = (offs[0].as_u64().unwrap() as usize, offs[1].as_u64().unwrap() as usize);
        tensors.push(Tensor { name: k, dtype: "F32", shape, data: data[a..b].to_vec() });
    }
    let num_labels = labels.len();
    tensors.push(Tensor {
        name: "classifier.weight".into(),
        dtype: "F32",
        shape: vec![num_labels as u64, 32],
        data: match &head {
            Some((w, _)) => slice_f32(w),
            None => linspace_f32(0.35, -0.25, num_labels * 32),
        },
    });
    let biases: Vec<f32> = match head {
        Some((_, b)) => b,
        None => (0..num_labels).map(|i| (i as f32 - 1.0) * 0.07).collect(),
    };
    tensors.push(Tensor {
        name: "classifier.bias".into(),
        dtype: "F32",
        shape: vec![num_labels as u64],
        data: slice_f32(&biases),
    });
    let weights_bytes = safetensors_file(&tensors);
    fs::create_dir_all(dir.join("weights")).unwrap();
    fs::write(dir.join("weights/model.safetensors"), &weights_bytes).unwrap();

    let mut m: Value = serde_json::from_slice(&fs::read(tiny_bundle().join("manifest.json")).unwrap()).unwrap();
    m["classifier"] = json!({
        "kind": "CLASSIFIER_SEQUENCE",
        "labels": labels,
    });
    let tn = m["artifacts"][0]["tensor_names"].as_object_mut().unwrap();
    tn.insert("pooler_weight".into(), json!("pooler.dense.weight"));
    tn.insert("pooler_bias".into(), json!("pooler.dense.bias"));
    tn.insert("classifier_weight".into(), json!("classifier.weight"));
    tn.insert("classifier_bias".into(), json!("classifier.bias"));
    edit(&mut m);
    let files = m["files"].as_array_mut().unwrap();
    let w_entry = files.iter_mut().find(|f| f["path"] == "weights/model.safetensors").unwrap();
    w_entry["size"] = json!(weights_bytes.len());
    w_entry["sha256"] = json!(sha256_hex(&weights_bytes));
    fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&m).unwrap()).unwrap();
    TempDir(dir)
}
