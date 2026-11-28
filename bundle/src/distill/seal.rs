//! Sealing and checking a staged static bundle on the CPU backend.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use turbo::bundle::Bundle;
use turbo::manifest::{PromptRole, Truncation};
use turbo::safetensors::{self, Dtype};
use turbo::tokenizer::{Encode, Tokenizer};

use super::math::{cosine, round4, static_vector, top1};
use super::quality::quality_texts;
use super::{EMBEDDINGS, paths, report_path, spec};
use crate::Result;
use crate::api;
use crate::recipe::Recipe;

/// Seal a staged bundle: the table's report fills `static_embedding` and
/// the artifact's `produced_by`, and the tool writes the reference
/// (static_reference.rs). Then the core loads the bundle on the CPU, and
/// its vectors must match the reference's and the tool's own for the
/// quality texts.
pub fn seal(recipe: &mut Recipe, bundle: &Path) -> Result<()> {
    spec(recipe)?;
    let p = paths(recipe)?;
    if bundle.join("manifest.json").exists() {
        return Err(format!("{} already holds a bundle; make it into an empty directory", bundle.display()));
    }
    let read = |path: &Path| -> Result<Value> {
        serde_json::from_slice(&fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?)
            .map_err(|e| format!("{}: {e}", path.display()))
    };
    let table_report_path = report_path(bundle, &p.weights);
    let table_report = read(&table_report_path)?;
    let m = &mut recipe.manifest;
    m["static_embedding"]["distilled_from"] = table_report["distilled_from"].clone();
    m["static_embedding"]["quality"] = table_report["quality"].clone();
    m["static_embedding"]["vocab_size"] = table_report["vocab_size"].clone();
    let mut pb = json!({});
    for key in ["tool", "tool_version", "container", "args", "reproducible"] {
        pb[key] = table_report[key].clone();
    }
    m["artifacts"][0]["produced_by"] = pb;
    // The report leaves the bundle, which lists every file in it; it is
    // put back if the bundle does not seal, so the run can be repeated.
    let kept: Vec<(PathBuf, Vec<u8>)> = [table_report_path]
        .into_iter()
        .map(|path| {
            let bytes = fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            Ok((path, bytes))
        })
        .collect::<Result<_>>()?;
    for (path, _) in &kept {
        fs::remove_file(path).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    let sealed = crate::static_reference::write(m, bundle).and_then(|produced_by| {
        m["reference"]["produced_by"] = produced_by;
        crate::seal::seal_manifest(m.clone(), bundle)?;
        check(bundle, &table_report["quality"])
    });
    if let Err(e) = sealed {
        let _ = fs::remove_file(bundle.join("manifest.json"));
        let _ = fs::remove_file(bundle.join(&p.reference));
        for (path, bytes) in &kept {
            crate::fetch::write_atomic(path, bytes)?;
        }
        return Err(e);
    }
    println!("{}: verified", bundle.display());
    Ok(())
}

/// The sealed bundle on the library's CPU device: every reference case
/// to the bit (static_reference::check), and the quality texts within the
/// F32 tolerance (docs/conformance.md) of the tool's own arithmetic in
/// F64, ranking to the same static_top1.
pub fn check(bundle: &Path, quality: &Value) -> Result<()> {
    crate::static_reference::check(bundle)?;
    let b = Bundle::open(bundle).map_err(|e| e.message)?;
    let m = &b.manifest;
    let st = m.static_embedding.as_ref().ok_or("static_embedding: missing")?;
    let k = m.embed().dim as usize;
    let rt = api::Runtime::create()?;
    let ctx = rt.cpu()?;
    let model = ctx.load(bundle)?;
    let info = model.info()?;
    let session = model.session(info.max_batch, info.max_seq)?;
    let o = api::options(0, 0);
    let mut got = vec![0f32; k];
    // The library's vectors are rounded as the table's dtype rounds them;
    // the tool's own are not.
    let close = |what: &str, got: &[f32], want: &[f64]| -> Result<()> {
        let g: Vec<f64> = got.iter().map(|&v| v as f64).collect();
        let cos = cosine(&g, want);
        let diff = g.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f64, f64::max);
        if cos < 0.9999 || diff > 1e-3 {
            return Err(format!("{what}: cosine {cos}, largest difference {diff} from the library's vector"));
        }
        Ok(())
    };

    // The table as stored, and the tool's arithmetic on it.
    let art = &m.artifacts[0];
    let wbytes = b.read_verified(&art.files[0]).map_err(|e| e.message)?;
    let wf = safetensors::File::parse(&art.files[0], &wbytes).map_err(|e| e.message)?;
    let floats = |name: &str| -> Result<Vec<f32>> {
        let t = wf.tensor(name).ok_or(format!("{}: no {name}", art.files[0]))?;
        Ok(match t.dtype {
            Dtype::F32 => t.f32s(),
            Dtype::F16 => t.data.as_chunks::<2>().0.iter().map(|&c| half::f16::from_le_bytes(c).to_f32()).collect(),
            _ => return Err(format!("{}: {name} is {}", art.files[0], t.dtype_name)),
        })
    };
    let table = floats(EMBEDDINGS)?;
    let tok = Tokenizer::load(&b).map_err(|e| e.message)?;
    let quality_file = st.quality.as_ref().ok_or("static_embedding.quality: missing")?;
    let texts = quality_texts(&bundle.join(&quality_file.texts))?;
    let mut lib = Vec::with_capacity(texts.len());
    // The bundle's own tokenizer applies the static rules itself.
    let e = Encode {
        add_special_tokens: true,
        truncation: Truncation::Right,
        max_tokens: st.max_length,
        prompt: PromptRole::None,
    };
    for (i, t) in texts.iter().enumerate() {
        let ids = tok.encode(&t.text, e).map_err(|e| e.message)?;
        let mine = static_vector(&table, k, &ids);
        session.texts(&[t.text.as_str()], &o, &mut got)?;
        close(&format!("quality text {i}"), &got, &mine)?;
        lib.push(got.iter().map(|&v| v as f64).collect::<Vec<f64>>());
    }
    let groups: Vec<u32> = texts.iter().map(|t| t.group).collect();
    let recorded = quality["static_top1"].as_f64().ok_or("quality.static_top1: missing")?;
    let measured = round4(top1(&lib, &groups));
    if measured != recorded {
        return Err(format!("the library's vectors rank static_top1 {measured}, the distillation measured {recorded}"));
    }
    Ok(())
}
