//! The harness embedder against a bundle's own reference vectors, for a
//! mean-pooled and a CLS-pooled bundle: each case without a prompt role,
//! embedded through `TurboEmbedder`, within 0.999 cosine of the vector
//! the bundle's reference model wrote. The bundles are the caller's:
//! `TURBO_EVAL_BUNDLES` names their directories, separated by commas, and
//! `TURBO_EVAL_DEVICE` the device (default cpu). Without them the test
//! has nothing to check and says so.

use std::path::Path;

use turbo::safetensors::{Dtype, File};
use turbo_analysis_embed::{Device, Precision};
use turbo_eval::embedder::{Embedder, Role, TurboEmbedder};

fn device() -> Device {
    match std::env::var("TURBO_EVAL_DEVICE").as_deref() {
        Ok("cpu") | Err(_) => Device::Cpu,
        Ok(name) => Device::Backend(name.into()),
    }
}

fn check(dir: &Path) {
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    let reference = &manifest["reference"];
    let bytes = std::fs::read(dir.join(reference["file"].as_str().unwrap())).unwrap();
    let file = File::parse("reference", &bytes).unwrap();
    let embeddings = file.get("embeddings", Dtype::F32, 2).unwrap().f32s();
    let mut e = TurboEmbedder::open(dir, device(), Precision::Model).unwrap();
    let dim = e.dim();
    let mut checked = 0;
    for (i, case) in reference["cases"].as_array().unwrap().iter().enumerate() {
        if case["prompt_role"] != "PROMPT_NONE" {
            continue;
        }
        let text = case["text"].as_str().unwrap();
        let mut out = Vec::new();
        e.embed(&[text], Role::Document, &mut out).unwrap();
        let want = &embeddings[i * dim..(i + 1) * dim];
        let norm = |v: &[f32]| v.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>().sqrt();
        let dot: f64 = out.iter().zip(want).map(|(&a, &b)| f64::from(a) * f64::from(b)).sum();
        let cosine = dot / (norm(&out) * norm(want));
        assert!(cosine > 0.999, "{}: case {i} {text:?}: cosine {cosine}", dir.display());
        checked += 1;
    }
    assert!(checked > 0, "{}: no case without a prompt role", dir.display());
}

#[test]
fn the_harness_embeds_as_the_bundle_s_reference_does() {
    let Ok(dirs) = std::env::var("TURBO_EVAL_BUNDLES") else {
        eprintln!("TURBO_EVAL_BUNDLES is not set: no bundle to check");
        return;
    };
    for d in dirs.split(',').filter(|d| !d.is_empty()) {
        check(Path::new(d));
    }
}
