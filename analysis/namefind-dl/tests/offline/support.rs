use std::fs;

use serde_json::{Value, json};
use turbo::bundle::sha256_hex;

fn tiny() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle")
}

/// The tiny bundle with a zero-weight head of `labels`, the bias `BIAS`.
fn write_bundle(name: &str, labels: &[&str; 3]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("turbo-namefind-dl-{}-{name}", std::process::id()));
    let tmp = dir.clone();
    let _ = fs::remove_dir_all(&tmp);
    for f in ["tokenizer.json", "reference/reference.safetensors"] {
        fs::create_dir_all(tmp.join(f).parent().unwrap()).unwrap();
        fs::copy(tiny().join(f), tmp.join(f)).unwrap();
    }
    let src = fs::read(tiny().join("weights/model.safetensors")).unwrap();
    let n = u64::from_le_bytes(src[..8].try_into().unwrap()) as usize;
    let mut header: serde_json::Map<String, Value> = serde_json::from_slice(&src[8..8 + n]).unwrap();
    header.remove("__metadata__");
    let mut data = src[8 + n..].to_vec();
    let mut add = |name: &str, shape: &[usize], values: &[f32]| {
        let start = data.len();
        data.extend(values.iter().flat_map(|v| v.to_le_bytes()));
        header.insert(name.into(), json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}));
    };
    add("classifier.weight", &[3, 32], &[0.0; 96]);
    add("classifier.bias", &[3], &BIAS);
    let mut head = serde_json::to_vec(&header).unwrap();
    head.resize(head.len().next_multiple_of(8), b' ');
    let mut weights = (head.len() as u64).to_le_bytes().to_vec();
    weights.extend(head);
    weights.extend(data);
    fs::create_dir_all(tmp.join("weights")).unwrap();
    fs::write(tmp.join("weights/model.safetensors"), &weights).unwrap();
    let mut m: Value = serde_json::from_slice(&fs::read(tiny().join("manifest.json")).unwrap()).unwrap();
    m["classifier"] = json!({"kind": "CLASSIFIER_TOKEN", "labels": labels});
    let names = m["artifacts"][0]["tensor_names"].as_object_mut().unwrap();
    names.insert("classifier_weight".into(), json!("classifier.weight"));
    names.insert("classifier_bias".into(), json!("classifier.bias"));
    let files = m["files"].as_array_mut().unwrap();
    let w = files.iter_mut().find(|f| f["path"] == "weights/model.safetensors").unwrap();
    w["size"] = json!(weights.len());
    w["sha256"] = json!(sha256_hex(&weights));
    fs::write(tmp.join("manifest.json"), serde_json::to_vec_pretty(&m).unwrap()).unwrap();
    dir
}
