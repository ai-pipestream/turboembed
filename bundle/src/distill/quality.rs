//! The quality texts a distillation is measured on, and the ids and
//! vectors each model gives them.

use std::fs;
use std::path::Path;

use serde::Deserialize;
use serde_json::Value;
use turbo::bundle::Bundle;
use turbo::manifest::{PromptRole, Truncation};
use turbo::tokenizer::{Encode, Tokenizer};

use crate::Result;
use crate::api;

/// One text of the quality set.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct QualityText {
    pub(super) text: String,
    pub(super) group: u32,
}

pub(super) fn quality_texts(path: &Path) -> Result<Vec<QualityText>> {
    let s = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let texts: Vec<QualityText> = s
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| serde_json::from_str(l).map_err(|e| format!("{} line {}: {e}", path.display(), i + 1)))
        .collect::<Result<_>>()?;
    if texts.len() < 3 {
        return Err(format!("{}: too few texts to rank", path.display()));
    }
    Ok(texts)
}

/// The base model's own vectors for the texts, as its bundle says.
pub(super) fn base_vectors(
    model: &api::Model,
    info: &turbo::turbo_model_info,
    texts: &[QualityText],
) -> Result<Vec<Vec<f64>>> {
    let d = info.dim as usize;
    let session = model.session(info.max_batch, info.max_seq)?;
    let o = api::options(0, 0);
    let mut out = Vec::with_capacity(texts.len());
    let mut buf = vec![0f32; info.max_batch as usize * d];
    for chunk in texts.chunks(info.max_batch as usize) {
        let t: Vec<&str> = chunk.iter().map(|q| q.text.as_str()).collect();
        session.texts(&t, &o, &mut buf[..t.len() * d])?;
        out.extend(buf[..t.len() * d].chunks_exact(d).map(|v| v.iter().map(|&x| x as f64).collect()));
    }
    Ok(out)
}

/// A text's ids for the static model, from the base tokenizer, as
/// a static model takes them: the text cut to `max_length` times the median
/// entry length in characters, encoded with no special tokens and cut to
/// `max_length` tokens, then the unknown token dropped.
pub(super) fn static_ids(tok: &Tokenizer, median: usize, text: &str, max_length: u32) -> Result<Vec<i32>> {
    let cut: String = text.chars().take(max_length as usize * median).collect();
    let e = Encode {
        add_special_tokens: false,
        truncation: Truncation::Right,
        max_tokens: max_length,
        prompt: PromptRole::None,
    };
    let mut ids = tok.encode(&cut, e).map_err(|e| e.message)?;
    ids.retain(|&id| id != tok.unk_id);
    Ok(ids)
}

/// Each text's ids through the tokenizer of the bundle staged in
/// `bundle`, whose manifest is `m` without `files`: the static rules the
/// library applies, which a tokenizer with learned terms changes. The
/// manifest is written for the tokenizer to load and removed again, as
/// the reference writer does (static_reference.rs).
pub(super) fn staged_ids(m: &Value, bundle: &Path, texts: &[QualityText]) -> Result<Vec<Vec<i32>>> {
    let reference = m["reference"]["file"].as_str().ok_or("manifest.reference.file: missing")?;
    let mut staged = m.clone();
    staged["reference"]["produced_by"] = serde_json::json!({
        "tool": "turbo-bundle", "tool_version": "staged", "container": "host", "args": [], "reproducible": true,
    });
    // Not measured yet: that is what the ids are for.
    if let Some(st) = staged["static_embedding"].as_object_mut() {
        st.remove("quality");
    }
    crate::seal::write_manifest(&staged, bundle, Some(reference))?;
    let made = (|| {
        let b = Bundle::open(bundle).map_err(|e| e.message)?;
        let tok = Tokenizer::unchecked(&b).map_err(|e| e.message)?;
        let st = b.manifest.static_embedding.as_ref().ok_or("static_embedding: missing")?;
        let e = Encode {
            add_special_tokens: true,
            truncation: Truncation::Right,
            max_tokens: st.max_length,
            prompt: PromptRole::None,
        };
        texts.iter().map(|t| tok.encode(&t.text, e).map_err(|e| e.message)).collect::<Result<Vec<_>>>()
    })();
    let _ = fs::remove_file(bundle.join("manifest.json"));
    made
}
