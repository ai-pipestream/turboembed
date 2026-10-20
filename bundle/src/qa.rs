//! Verification of sealed extractive-QA bundles against their token logits.
use crate::{Result, recipe::Recipe};
use serde_json::{Value, json};
use std::path::Path;
use turbo::{
    bundle::Bundle,
    qa::ExtractiveQa,
    safetensors::{Dtype, File},
};

pub fn cases(recipe: &Recipe) -> Result<Value> {
    let m = &recipe.manifest;
    let q = &m["qa"];
    let cases = m["reference"]["cases"].as_array().ok_or("QA reference.cases: missing")?;
    for (i, c) in cases.iter().enumerate() {
        if c["text"].as_str().is_none() || c["context"].as_str().is_none() {
            return Err(format!("QA reference case {i} requires text and context"));
        }
    }
    Ok(json!({"task":"TASK_EXTRACTIVE_QA", "qa": q, "cases": cases}))
}

pub fn verify(path: &Path, bundle: &Bundle) -> Result<()> {
    let mut session = ExtractiveQa::open(path, 1).map_err(|e| e.message)?;
    let bytes = bundle.read_verified(&bundle.manifest.reference.file).map_err(|e| e.message)?;
    let file = File::parse("qa reference", &bytes).map_err(|e| e.message)?;
    let wanted = file.get("logits", Dtype::F32, 2).map_err(|e| e.message)?.f32s();
    let mut actual = Vec::new();
    for case in &bundle.manifest.reference.cases {
        let context = case.context.as_deref().ok_or("QA reference case requires context")?;
        for w in session.logits(&case.text, context).map_err(|e| e.message)? {
            for (start, end) in w.start.into_iter().zip(w.end) {
                actual.extend([start, end]);
            }
        }
    }
    if actual.len() != wanted.len() || actual.iter().zip(&wanted).any(|(&a, &b)| (a - b).abs() > 1e-4) {
        return Err("QA reference logits differ from CPU inference by more than 1e-4".into());
    }
    Ok(())
}
