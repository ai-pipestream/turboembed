//! A static model distilled from a bundle: one vector per vocabulary
//! entry, so a text's vector is a lookup and a mean (docs/static.md).
//!
//! The base bundle's encoder, run by the library at exact F32 precision on
//! the chosen device, embeds every vocabulary entry alone, between the base
//! template's special tokens, mean-pooled over the row with no
//! normalization. Entries the recipe's `skip` pattern matches whole are left
//! out (BERT's `[unused..]` ids, which no text gives). The rest are reduced
//! to `pca_dims` by PCA, each row centred by the mean of the kept rows and
//! projected on the covariance's leading eigenvectors, then weighted by
//! Zipf's law: the entry of rank r among the kept ones, in id order, is
//! taken to occur with probability p_r proportional to 1 / (r + 2), and its
//! row is scaled by sif / (sif + p_r). That is Model2Vec's distillation (its
//! `distill` with mean pooling and no vocabulary of its own), written here
//! so the library that serves the base model is the one that distils it. As
//! Model2Vec does, the special tokens a text never gives ([CLS], [SEP],
//! [MASK]: every special token but the padding and the unknown one) are left
//! out too. An entry left out keeps a row of zeros, so the table has a row
//! per id of the base tokenizer, and the file holds the table alone: no
//! weights.
//!
//! The cost is measured on the texts the recipe carries, with groups of
//! texts that mean the same thing: how often each text's nearest other text
//! is of its group, with the base model and with the static one, and how
//! the two models' similarities rank against each other over every pair.

use std::path::{Path, PathBuf};

mod skip;
use serde::Deserialize;
use serde_json::json;
use skip::SkipRule;
use turbo::bundle::Bundle;
use turbo::manifest::{Manifest, SpecialRole};
use turbo::tokenizer::Tokenizer;
use turbo::{TURBO_EMBED_STAGE_ENCODE, TURBO_NORMALIZE_NONE, TURBO_POOLING_MEAN};

use crate::Result;
use crate::api;
use crate::recipe::Recipe;

pub use crate::api::Device;

/// The recipe's `distill` block.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    /// The width of the static vectors: at most the base model's.
    pub pca_dims: u32,
    /// Model2Vec's SIF coefficient, in (0, 1).
    pub sif_coefficient: f64,
    /// A whole-token `PREFIX\d+SUFFIX` rule with ASCII digits and literal
    /// edges (escape ASCII punctuation with a backslash). Matching rows
    /// are left out of the PCA and ranks, with zero vector and weight.
    pub skip: String,
    /// What the table is stored in: "F32" or "F16".
    pub dtype: String,
}

/// The table's tensor name in the weights file, as Model2Vec names it.
pub(super) const EMBEDDINGS: &str = "embeddings";

/// Where the recipe puts things.
pub(super) struct Paths {
    pub(super) weights: String,
    pub(super) tokenizer: String,
    pub(super) texts: String,
    pub(super) reference: String,
}

pub(super) fn paths(recipe: &Recipe) -> Result<Paths> {
    let m = &recipe.manifest;
    let arts = m["artifacts"].as_array().ok_or("manifest.artifacts: missing")?;
    let [art] = arts.as_slice() else {
        return Err("a static recipe has one artifact, its table".into());
    };
    let files = art["files"].as_array().ok_or("manifest.artifacts[0].files: missing")?;
    let [weights] = files.as_slice() else {
        return Err("a static recipe's artifact is one file".into());
    };
    let names = &art["tensor_names"];
    if *names != json!({ "static_embeddings": EMBEDDINGS }) {
        return Err(format!(
            "manifest.artifacts[0].tensor_names: the tool writes static_embeddings {EMBEDDINGS:?} alone"
        ));
    }
    if art.get("produced_by").is_some() {
        return Err("manifest.artifacts[0].produced_by: written by the tool, not the recipe".into());
    }
    let s = |p: &str| recipe.str_at(p).map(str::to_owned);
    let p = Paths {
        weights: weights.as_str().ok_or("manifest.artifacts[0].files[0]: not a string")?.to_owned(),
        tokenizer: s("/tokenizer/file")?,
        texts: s("/static_embedding/quality/texts")?,
        reference: s("/reference/file")?,
    };
    for f in [&p.weights, &p.tokenizer, &p.texts, &p.reference] {
        crate::check_rel(f)?;
    }
    Ok(p)
}

pub(super) fn spec(recipe: &Recipe) -> Result<&Spec> {
    let s = recipe.distill.as_ref().ok_or("the recipe has no distill block")?;
    if !(s.sif_coefficient > 0.0 && s.sif_coefficient < 1.0) {
        return Err(format!("distill.sif_coefficient {} is not in (0, 1)", s.sif_coefficient));
    }
    if s.pca_dims == 0 {
        return Err("distill.pca_dims: must be over 0".into());
    }
    if s.dtype != "F32" && s.dtype != "F16" {
        return Err(format!("distill.dtype {:?}: F32 or F16", s.dtype));
    }
    Ok(s)
}

/// Write the static bundle's files into `bundle` on the CPU: the base's
/// tokenizer file, the table, the quality texts, and beside the table its
/// report, which [`seal`] reads.
pub fn stage(recipe: &Recipe, base_dir: &Path, bundle: &Path) -> Result<()> {
    stage_on(recipe, base_dir, bundle, &[], &Device::Cpu, &mut progress::print).map(|_| ())
}

/// What a staged distillation ran and added beyond the base's vocabulary.
#[derive(Debug, Default)]
pub struct Staged {
    /// The backend that executed the base encoder (`"cpu"`, `"cuda"`, ...).
    pub backend: String,
    /// Where `TURBO_EMBED_STAGE_ENCODE` ran (`TURBO_STAGE_HOST` or `TURBO_STAGE_DEVICE`).
    pub encode_stage: u32,
    /// The learned terms' entries, as ids from the base's vocab_size up.
    pub added: Vec<String>,
    /// The learned terms left out, and why.
    pub refused: Vec<(String, Refusal)>,
}

/// [`stage`], with `terms` added to the vocabulary (`extra.rs`) and each
/// step reported to `observe`.
pub fn stage_with(
    recipe: &Recipe,
    base_dir: &Path,
    bundle: &Path,
    terms: &[String],
    observe: &mut dyn FnMut(Progress),
) -> Result<Staged> {
    stage_on(recipe, base_dir, bundle, terms, &Device::Cpu, observe)
}

/// [`stage_with`], running the base encoder on `device` with no fallback to
/// another device.
pub fn stage_on(
    recipe: &Recipe,
    base_dir: &Path,
    bundle: &Path,
    terms: &[String],
    device: &Device,
    observe: &mut dyn FnMut(Progress),
) -> Result<Staged> {
    let spec = spec(recipe)?;
    let p = paths(recipe)?;
    if bundle.join("manifest.json").exists() {
        return Err(format!("{} already holds a bundle; make it into an empty directory", bundle.display()));
    }
    let base = Bundle::open(base_dir).map_err(|e| format!("base bundle: {}", e.message))?;
    let bm = &base.manifest;
    let arch =
        bm.architecture.as_ref().ok_or("base bundle: no architecture; a static model is distilled from an encoder")?;
    let base_tok = Tokenizer::load(&base).map_err(|e| format!("base bundle: {}", e.message))?;
    same_tokenizer(recipe, bm)?;
    let tok_bytes = base.read_verified(&bm.tokenizer.file).map_err(|e| e.message)?;
    let vocab_size = arch.vocab_size as usize;
    let added = extra::plan(&base_tok, bm, terms)?;
    let tok_out = match added.entries.is_empty() {
        true => tok_bytes.clone(),
        false => extra::extend(&tok_bytes, vocab_size, &added.entries)?,
    };
    crate::fetch::write_atomic(&bundle.join(&p.tokenizer), &tok_out)?;
    crate::seal::stage(recipe, base_dir, bundle)?;
    let declared = recipe.manifest["static_embedding"]["vocab_size"].as_u64();
    if declared != Some(vocab_size as u64) {
        return Err(format!("static_embedding.vocab_size {declared:?} is not the base's {vocab_size}"));
    }
    let h = arch.hidden as usize;
    let k = spec.pca_dims as usize;
    if k > h {
        return Err(format!("distill.pca_dims {k} is over the base model's width {h}"));
    }
    if recipe.manifest["embed"]["dim"].as_u64() != Some(k as u64) {
        return Err(format!("embed.dim is not distill.pca_dims {k}"));
    }

    // Which entries are kept.
    let vocab = vocabulary(&tok_bytes, vocab_size)?;
    let skip = SkipRule::parse(&spec.skip)?;
    let unused_special = |id: usize| {
        bm.tokenizer
            .special_tokens
            .iter()
            .any(|s| s.id as usize == id && !matches!(s.role, SpecialRole::Pad | SpecialRole::Unk))
    };
    let kept: Vec<usize> = (0..vocab_size)
        .filter(|&id| !unused_special(id) && vocab[id].as_ref().is_some_and(|t| !skip.is_match(t)))
        .collect();
    if kept.len() <= k {
        return Err(format!("{} entries are kept, and PCA to {k} needs more", kept.len()));
    }
    let (n_added, total) = (added.entries.len(), vocab_size + added.entries.len());
    observe(Progress::Kept { kept: kept.len(), vocab_size, added: n_added, refused: added.refused.len() });

    // Every kept entry through the base encoder on `device`.
    let rt = api::Runtime::create()?;
    let ctx = rt.open(device)?;
    let model = ctx.load(base_dir)?;
    let info = model.info()?;
    let n = kept.len() + n_added;
    let (mut rows, entry_run) =
        embed_entries(&model, &info, bm, &kept, h, &mut |done| observe(Progress::Embedded { done, total: n }))?;
    let backend = api::field(&entry_run.backend);
    let encode_stage = entry_run.stage[TURBO_EMBED_STAGE_ENCODE];
    rows.extend(extra::embed(&model, &info, bm, &added.pieces, h, |done| {
        observe(Progress::Embedded { done: kept.len() + done, total: n })
    })?);

    // PCA, then Zipf weights by rank: the kept entries in id order, then
    // the learned terms in theirs.
    let (projected, explained) = pca(&rows, n, h, k);
    observe(Progress::Reduced { pca_dims: k, explained_variance: explained });
    let zipf: f64 = (0..n).map(|r| 1.0 / (r as f64 + 2.0)).sum();
    let mut table = vec![0f32; total * k];
    let ids = kept.iter().copied().chain(vocab_size..total);
    for (r, id) in ids.enumerate() {
        let p = (1.0 / (r as f64 + 2.0)) / zipf;
        let w = spec.sif_coefficient / (spec.sif_coefficient + p);
        for (t, v) in table[id * k..(id + 1) * k].iter_mut().zip(&projected[r * k..(r + 1) * k]) {
            // Rounded once, from F64, to what is stored.
            *t = if spec.dtype == "F16" { half::f16::from_f64(v * w).to_f32() } else { (v * w) as f32 };
        }
    }
    let file = write_table(&table, total, k, &spec.dtype);
    crate::fetch::write_atomic(&bundle.join(&p.weights), &file)?;

    // The cost, on the quality texts.
    let texts = quality_texts(&bundle.join(&p.texts))?;
    let max_length = recipe.manifest["static_embedding"]["max_length"]
        .as_u64()
        .ok_or("manifest.static_embedding.max_length: missing")? as u32;
    let median = median_chars(&vocab);
    let (base_vecs, quality_run) = base_vectors(&model, &info, &texts)?;
    if api::field(&quality_run.backend) != backend || quality_run.stage[TURBO_EMBED_STAGE_ENCODE] != encode_stage {
        return Err("quality texts did not run on the base encoder's device".into());
    }
    let static_vecs: Vec<Vec<f64>> = match n_added {
        0 => texts
            .iter()
            .map(|t| Ok(static_vector(&table, k, &static_ids(&base_tok, median, &t.text, max_length)?)))
            .collect::<Result<_>>()?,
        _ => {
            let mut m = recipe.manifest.clone();
            m["static_embedding"]["vocab_size"] = json!(total);
            quality::staged_ids(&m, bundle, &texts)?.iter().map(|ids| static_vector(&table, k, ids)).collect()
        }
    };
    let groups: Vec<u32> = texts.iter().map(|t| t.group).collect();
    let quality = json!({
        "texts": p.texts,
        "base_top1": round4(top1(&base_vecs, &groups)),
        "static_top1": round4(top1(&static_vecs, &groups)),
        "similarity_spearman": round4(spearman(&base_vecs, &static_vecs)),
    });
    observe(Progress::Measured {
        base_top1: quality["base_top1"].as_f64().unwrap_or(0.0),
        static_top1: quality["static_top1"].as_f64().unwrap_or(0.0),
        similarity_spearman: quality["similarity_spearman"].as_f64().unwrap_or(0.0),
    });

    let report = json!({
        "tool": "turbo-bundle distill",
        "tool_version": format!("{} (libturbo {})", env!("CARGO_PKG_VERSION"), turbo_version()),
        "container": "host",
        "args": [
            format!("base={}@{}", bm.model.id, base.manifest_sha256),
            format!("backend={backend}"),
            format!("rows={}", template_text(bm)),
            "pooling=mean".to_owned(),
            format!("skip={}", spec.skip),
            format!("pca_dims={k}"),
            format!("explained_variance={explained:.4}"),
            format!("sif_coefficient={}", spec.sif_coefficient),
            format!("dtype={}", spec.dtype),
            format!("learned_terms={n_added}"),
        ],
        "reproducible": false,
        "distilled_from": {
            "model_id": bm.model.id,
            "revision": bm.model.revision,
            "manifest_sha256": base.manifest_sha256,
        },
        "quality": quality,
        "vocab_size": total,
    });
    let report_path = report_path(bundle, &p.weights);
    crate::fetch::write_atomic(&report_path, &serde_json::to_vec_pretty(&report).unwrap())?;
    Ok(Staged { backend, encode_stage, added: added.entries, refused: added.refused })
}

pub(crate) fn turbo_version() -> String {
    unsafe { std::ffi::CStr::from_ptr(turbo::turbo_version()) }.to_string_lossy().into_owned()
}

pub(super) fn report_path(bundle: &Path, weights: &str) -> PathBuf {
    bundle.join(format!("{weights}.report.json"))
}

/// The static tokenizer block is the base's, with no special tokens
/// around the text.
fn same_tokenizer(recipe: &Recipe, bm: &Manifest) -> Result<()> {
    let mut want = serde_json::to_value(&bm.tokenizer).unwrap();
    want["template"] = json!(["$TEXT"]);
    let mut got = recipe.manifest["tokenizer"].clone();
    // The file may sit at another path.
    got["file"] = want["file"].clone();
    if got != want {
        return Err(format!(
            "manifest.tokenizer is not the base bundle's with template [\"$TEXT\"]: the base has {want}"
        ));
    }
    Ok(())
}

/// The base template with the entry in $TEXT's place, as text.
fn template_text(bm: &Manifest) -> String {
    bm.tokenizer
        .template
        .iter()
        .map(|s| if s == "$TEXT" { "<entry>" } else { s.as_str() })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Each kept entry as the base model sees it alone: the template's
/// special tokens around it, mean-pooled, not normalized. [kept, h].
fn embed_entries(
    model: &api::Model,
    info: &turbo::turbo_model_info,
    bm: &Manifest,
    kept: &[usize],
    h: usize,
    done: &mut dyn FnMut(usize),
) -> Result<(Vec<f64>, turbo::turbo_result_info)> {
    let id_of = |s: &str| bm.tokenizer.special_tokens.iter().find(|t| t.content == s).map(|t| t.id as i32);
    let template: Vec<Option<i32>> = bm
        .tokenizer
        .template
        .iter()
        .map(|s| if s == "$TEXT" { Ok(None) } else { id_of(s).map(Some).ok_or(format!("template {s:?}")) })
        .collect::<Result<_>>()?;
    let seq = template.len();
    let batch = info.max_batch as usize;
    let session = model.session(info.max_batch, seq as u32)?;
    let o = api::options(TURBO_POOLING_MEAN, TURBO_NORMALIZE_NONE);
    let mut out = Vec::with_capacity(kept.len() * h);
    let mut vecs = vec![0f32; batch * h];
    let mut last = None;
    for (c, chunk) in kept.chunks(batch).enumerate() {
        let ids: Vec<i32> = chunk.iter().flat_map(|&id| template.iter().map(move |t| t.unwrap_or(id as i32))).collect();
        let mask = vec![1i32; ids.len()];
        let n = chunk.len();
        last = Some(session.tokens(n as u32, seq as u32, &ids, &mask, &o, &mut vecs[..n * h])?);
        out.extend(vecs[..n * h].iter().map(|&v| v as f64));
        let completed = out.len() / h;
        if c % 64 == 63 || completed == kept.len() {
            done(completed);
        }
    }
    Ok((out, last.ok_or("kept entries: empty")?))
}

mod extra;
pub mod from_base;
mod math;
pub mod progress;
mod quality;
mod seal;
#[cfg(test)]
mod tests;

pub use extra::Refusal;
pub use progress::Progress;
use quality::{base_vectors, quality_texts, static_ids};
pub use seal::{check, seal};

use math::{median_chars, pca, round4, spearman, static_vector, top1, vocabulary, write_table};
