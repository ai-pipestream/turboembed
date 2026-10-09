//! A static model distilled from a bundle: one vector per vocabulary
//! entry, so a text's vector is a lookup and a mean (docs/static.md).
//!
//! The base bundle's encoder, run by the library on the CPU, embeds every
//! vocabulary entry alone, between the base template's special tokens,
//! mean-pooled over the row with no normalization. Entries the recipe's
//! `skip` pattern matches whole are left out (BERT's `[unused..]` ids,
//! which no text gives). The rest are reduced to `pca_dims` by PCA, each
//! row centred by the mean of the kept rows and projected on the
//! covariance's leading eigenvectors, then weighted by Zipf's law: the
//! entry of rank r among the kept ones, in id order, is taken to occur with
//! probability p_r proportional to 1 / (r + 2), and its row is scaled by
//! sif / (sif + p_r). That is Model2Vec's distillation (its `distill` with
//! mean pooling and no vocabulary of its own), written here so the library
//! that serves the base model is the one that distils it.
//!
//! The table has a weight per token besides: 0 for the unknown token and
//! for every entry left out, 1 for the rest. A 0 leaves the token out of a
//! text's mean, as Model2Vec leaves the unknown token out.
//!
//! The cost is measured on the texts the recipe carries, with groups of
//! texts that mean the same thing: how often each text's nearest other text
//! is of its group, with the base model and with the static one, and how
//! the two models' similarities rank against each other over every pair.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use regex::Regex;
use serde::Deserialize;
use serde_json::{Value, json};
use turbo::bundle::Bundle;
use turbo::manifest::{Manifest, PromptRole, Truncation};
use turbo::safetensors::{self, Dtype};
use turbo::tokenizer::{Encode, Tokenizer};
use turbo::{TURBO_NORMALIZE_NONE, TURBO_POOLING_MEAN};

use crate::Result;
use crate::api;
use crate::recipe::Recipe;

/// The recipe's `distill` block.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    /// The width of the static vectors: at most the base model's.
    pub pca_dims: u32,
    /// Model2Vec's SIF coefficient, in (0, 1).
    pub sif_coefficient: f64,
    /// A regular expression; a vocabulary entry it matches whole is left
    /// out of the PCA and the ranks, its row zero and its weight 0.
    pub skip: String,
    /// What the table is stored in: "F32" or "F16".
    pub dtype: String,
}

/// The table's tensor names in the weights file.
const EMBEDDINGS: &str = "embeddings";
const WEIGHTS: &str = "weights";

/// One text of the quality set.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct QualityText {
    text: String,
    group: u32,
}

/// Where the recipe puts things.
struct Paths {
    weights: String,
    tokenizer: String,
    texts: String,
    reference: String,
}

fn paths(recipe: &Recipe) -> Result<Paths> {
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
    if names["static_embeddings"] != EMBEDDINGS || names["static_weights"] != WEIGHTS {
        return Err(format!(
            "manifest.artifacts[0].tensor_names: the tool writes static_embeddings {EMBEDDINGS:?} and \
             static_weights {WEIGHTS:?}"
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

fn spec(recipe: &Recipe) -> Result<&Spec> {
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

/// Write the static bundle's files into `bundle`: the base's tokenizer
/// file, the table, the quality texts, and beside the table its report,
/// which `seal` reads; and `reference/cases.json`, what the reference
/// script reads.
pub fn stage(recipe: &Recipe, base_dir: &Path, bundle: &Path) -> Result<()> {
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
    crate::fetch::write_atomic(&bundle.join(&p.tokenizer), &tok_bytes)?;
    crate::seal::stage(recipe, base_dir, bundle)?;
    let vocab_size = arch.vocab_size as usize;
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
    let skip = Regex::new(&format!("^(?:{})$", spec.skip)).map_err(|e| format!("distill.skip: {e}"))?;
    let kept: Vec<usize> =
        (0..vocab_size).filter(|&id| vocab[id].as_ref().is_some_and(|t| !skip.is_match(t))).collect();
    if kept.len() <= k {
        return Err(format!("{} entries are kept, and PCA to {k} needs more", kept.len()));
    }
    let unk = base_tok.unk_id as usize;
    println!(
        "distill: {} of {vocab_size} entries kept, {} left out by {:?}",
        kept.len(),
        vocab_size - kept.len(),
        spec.skip
    );

    // Every kept entry through the base encoder.
    let rt = api::Runtime::create()?;
    let ctx = rt.cpu()?;
    let model = ctx.load(base_dir)?;
    let info = model.info()?;
    let rows = embed_entries(&model, &info, bm, &kept, h)?;

    // PCA, then Zipf weights by rank among the kept entries.
    let (projected, explained) = pca(&rows, kept.len(), h, k);
    let n = kept.len();
    let total: f64 = (0..n).map(|r| 1.0 / (r as f64 + 2.0)).sum();
    let mut table = vec![0f32; vocab_size * k];
    let mut weights = vec![0f32; vocab_size];
    for (r, &id) in kept.iter().enumerate() {
        let p = (1.0 / (r as f64 + 2.0)) / total;
        let w = spec.sif_coefficient / (spec.sif_coefficient + p);
        for (t, v) in table[id * k..(id + 1) * k].iter_mut().zip(&projected[r * k..(r + 1) * k]) {
            *t = (v * w) as f32;
        }
        weights[id] = if id == unk { 0.0 } else { 1.0 };
    }
    // As stored: what the library reads, and what quality is measured on.
    if spec.dtype == "F16" {
        for v in table.iter_mut().chain(weights.iter_mut()) {
            *v = half::f16::from_f32(*v).to_f32();
        }
    }
    let file = write_table(&table, &weights, vocab_size, k, &spec.dtype);
    crate::fetch::write_atomic(&bundle.join(&p.weights), &file)?;

    // The cost, on the quality texts.
    let texts = quality_texts(&bundle.join(&p.texts))?;
    let max_seq = recipe.manifest["embed"]["max_seq"].as_u64().ok_or("manifest.embed.max_seq: missing")? as u32;
    let base_vecs = base_vectors(&model, &info, &texts)?;
    let static_vecs: Vec<Vec<f64>> = texts
        .iter()
        .map(|t| {
            let ids = static_ids(&base_tok, &t.text, max_seq)?;
            Ok(static_vector(&table, &weights, k, &ids))
        })
        .collect::<Result<_>>()?;
    let groups: Vec<u32> = texts.iter().map(|t| t.group).collect();
    let quality = json!({
        "texts": p.texts,
        "base_top1": round4(top1(&base_vecs, &groups)),
        "static_top1": round4(top1(&static_vecs, &groups)),
        "similarity_spearman": round4(spearman(&base_vecs, &static_vecs)),
    });
    println!("distill: quality {quality}");

    let report = json!({
        "tool": "turbo-bundle distill",
        "tool_version": format!("{} (libturbo {})", env!("CARGO_PKG_VERSION"), turbo_version()),
        "container": "host",
        "args": [
            format!("base={}@{}", bm.model.id, base.manifest_sha256),
            format!("rows={}", template_text(bm)),
            "pooling=mean".to_owned(),
            format!("skip={}", spec.skip),
            format!("pca_dims={k}"),
            format!("explained_variance={explained:.4}"),
            format!("sif_coefficient={}", spec.sif_coefficient),
            format!("dtype={}", spec.dtype),
        ],
        "reproducible": false,
        "distilled_from": {
            "model_id": bm.model.id,
            "revision": bm.model.revision,
            "manifest_sha256": base.manifest_sha256,
        },
        "quality": quality,
    });
    let report_path = report_path(bundle, &p.weights);
    crate::fetch::write_atomic(&report_path, &serde_json::to_vec_pretty(&report).unwrap())?;

    // What the reference script reads.
    let mut cases = crate::reference::cases(recipe)?;
    cases["weights"] = json!(p.weights);
    cases["tokenizer"] = json!(p.tokenizer);
    cases["unk_id"] = json!(base_tok.unk_id);
    cases["pad_id"] = json!(base_tok.fill_id());
    crate::fetch::write_atomic(&cases_path(bundle), &serde_json::to_vec_pretty(&cases).unwrap())?;
    Ok(())
}

fn turbo_version() -> String {
    unsafe { std::ffi::CStr::from_ptr(turbo::turbo_version()) }.to_string_lossy().into_owned()
}

fn report_path(bundle: &Path, weights: &str) -> PathBuf {
    bundle.join(format!("{weights}.report.json"))
}

fn cases_path(bundle: &Path) -> PathBuf {
    bundle.join("reference/cases.json")
}

fn round4(v: f64) -> f64 {
    (v * 1e4).round() / 1e4
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

/// Each id's entry in the tokenizer file, for ids below `vocab_size`;
/// None for an id the file does not name.
fn vocabulary(tokenizer_json: &[u8], vocab_size: usize) -> Result<Vec<Option<String>>> {
    let v: Value = serde_json::from_slice(tokenizer_json).map_err(|e| format!("tokenizer file: {e}"))?;
    let mut out = vec![None; vocab_size];
    match &v["model"]["vocab"] {
        Value::Object(map) => {
            for (token, id) in map {
                let id = id.as_u64().ok_or("tokenizer file: a vocab id is not a number")? as usize;
                if id < vocab_size {
                    out[id] = Some(token.clone());
                }
            }
        }
        Value::Array(pieces) => {
            for (id, p) in pieces.iter().enumerate().take(vocab_size) {
                out[id] = Some(p[0].as_str().ok_or("tokenizer file: a piece is not a string")?.to_owned());
            }
        }
        _ => return Err("tokenizer file: model.vocab is neither a map nor a list".into()),
    }
    Ok(out)
}

/// Each kept entry as the base model sees it alone: the template's
/// special tokens around it, mean-pooled, not normalized. [kept, h].
fn embed_entries(
    model: &api::Model,
    info: &turbo::turbo_model_info,
    bm: &Manifest,
    kept: &[usize],
    h: usize,
) -> Result<Vec<f64>> {
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
    for (c, chunk) in kept.chunks(batch).enumerate() {
        let ids: Vec<i32> = chunk.iter().flat_map(|&id| template.iter().map(move |t| t.unwrap_or(id as i32))).collect();
        let mask = vec![1i32; ids.len()];
        let n = chunk.len();
        session.tokens(n as u32, seq as u32, &ids, &mask, &o, &mut vecs[..n * h])?;
        out.extend(vecs[..n * h].iter().map(|&v| v as f64));
        if c % 64 == 63 {
            println!("distill: {} of {} entries embedded", (c + 1) * batch, kept.len());
        }
    }
    Ok(out)
}

/// PCA of `rows` [n, h] to `k` dims: the rows centred on their mean and
/// projected on the covariance's k leading eigenvectors, each signed so
/// its largest entry is positive. Returns [n, k] and the share of the
/// variance kept.
fn pca(rows: &[f64], n: usize, h: usize, k: usize) -> (Vec<f64>, f64) {
    let mut mean = vec![0f64; h];
    for r in rows.chunks_exact(h) {
        for (m, v) in mean.iter_mut().zip(r) {
            *m += v;
        }
    }
    for m in &mut mean {
        *m /= n as f64;
    }
    let centred: Vec<f64> = rows.chunks_exact(h).flat_map(|r| r.iter().zip(&mean).map(|(v, m)| v - m)).collect();
    // The covariance, upper triangle by rows of the data, split over
    // threads by blocks of data rows and summed in a fixed order.
    let threads = std::thread::available_parallelism().map_or(1, |t| t.get()).min(64);
    let per = n.div_ceil(threads);
    let parts: Vec<Vec<f64>> = std::thread::scope(|s| {
        let handles: Vec<_> = centred
            .chunks(per * h)
            .map(|block| {
                s.spawn(move || {
                    let mut c = vec![0f64; h * h];
                    for r in block.chunks_exact(h) {
                        for i in 0..h {
                            let ri = r[i];
                            let row = &mut c[i * h..(i + 1) * h];
                            for j in i..h {
                                row[j] += ri * r[j];
                            }
                        }
                    }
                    c
                })
            })
            .collect();
        handles.into_iter().map(|t| t.join().unwrap()).collect()
    });
    let mut cov = vec![0f64; h * h];
    for p in &parts {
        for (c, v) in cov.iter_mut().zip(p) {
            *c += v;
        }
    }
    for i in 0..h {
        for j in i..h {
            let v = cov[i * h + j] / (n as f64 - 1.0);
            cov[i * h + j] = v;
            cov[j * h + i] = v;
        }
    }
    let (values, vectors) = jacobi(&mut cov, h);
    let mut order: Vec<usize> = (0..h).collect();
    order.sort_by(|&a, &b| values[b].total_cmp(&values[a]));
    let total: f64 = values.iter().map(|v| v.max(0.0)).sum();
    let kept: f64 = order[..k].iter().map(|&i| values[i].max(0.0)).sum();
    // Component c is column order[c] of `vectors`.
    let mut comps = vec![0f64; k * h];
    for (c, &col) in order[..k].iter().enumerate() {
        let comp = &mut comps[c * h..(c + 1) * h];
        for (i, x) in comp.iter_mut().enumerate() {
            *x = vectors[i * h + col];
        }
        let big = comp.iter().copied().fold(0f64, |a, b| if b.abs() > a.abs() { b } else { a });
        if big < 0.0 {
            for x in comp.iter_mut() {
                *x = -*x;
            }
        }
    }
    let mut out = vec![0f64; n * k];
    for (r, o) in centred.chunks_exact(h).zip(out.chunks_exact_mut(k)) {
        for (c, x) in o.iter_mut().enumerate() {
            *x = comps[c * h..(c + 1) * h].iter().zip(r).map(|(a, b)| a * b).sum();
        }
    }
    (out, if total > 0.0 { kept / total } else { 0.0 })
}

/// The eigenvalues and eigenvectors (as columns of an [n, n] matrix) of
/// the symmetric matrix `a`, by cyclic Jacobi rotations. `a` is
/// overwritten.
fn jacobi(a: &mut [f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let norm: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    for _sweep in 0..100 {
        let off: f64 = (0..n).flat_map(|p| (p + 1..n).map(move |q| (p, q))).map(|(p, q)| a[p * n + q].powi(2)).sum();
        if off.sqrt() <= 1e-15 * norm {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq.abs() <= 1e-300 {
                    continue;
                }
                let (app, aqq) = (a[p * n + p], a[q * n + q]);
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // Columns p and q, then rows p and q.
                for k in 0..n {
                    let (akp, akq) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    ((0..n).map(|i| a[i * n + i]).collect(), v)
}

/// The table as a safetensors file: `embeddings` [vocab, k] then `weights`
/// [vocab], in `dtype`, the header padded to a multiple of 8.
fn write_table(table: &[f32], weights: &[f32], vocab: usize, k: usize, dtype: &str) -> Vec<u8> {
    let bytes = |v: &[f32]| -> Vec<u8> {
        if dtype == "F16" {
            v.iter().flat_map(|&x| half::f16::from_f32(x).to_le_bytes()).collect()
        } else {
            v.iter().flat_map(|&x| x.to_le_bytes()).collect()
        }
    };
    let (e, w) = (bytes(table), bytes(weights));
    let header = json!({
        EMBEDDINGS: { "dtype": dtype, "shape": [vocab, k], "data_offsets": [0, e.len()] },
        WEIGHTS: { "dtype": dtype, "shape": [vocab], "data_offsets": [e.len(), e.len() + w.len()] },
    });
    let mut h = serde_json::to_vec(&header).unwrap();
    while h.len() % 8 != 0 {
        h.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + h.len() + e.len() + w.len());
    out.extend((h.len() as u64).to_le_bytes());
    out.extend(h);
    out.extend(e);
    out.extend(w);
    out
}

fn quality_texts(path: &Path) -> Result<Vec<QualityText>> {
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
fn base_vectors(model: &api::Model, info: &turbo::turbo_model_info, texts: &[QualityText]) -> Result<Vec<Vec<f64>>> {
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

/// A text's ids for the static model: the base tokenizer with no special
/// tokens, cut on the right at `max_seq`.
fn static_ids(tok: &Tokenizer, text: &str, max_seq: u32) -> Result<Vec<i32>> {
    let e = Encode {
        add_special_tokens: false,
        truncation: Truncation::Right,
        max_tokens: max_seq,
        prompt: PromptRole::None,
    };
    tok.encode(text, e).map_err(|e| e.message)
}

/// The static vector of `ids` in f64, L2-normalized: the mean of weight
/// times row over the ids whose weight is not 0.
fn static_vector(table: &[f32], weights: &[f32], k: usize, ids: &[i32]) -> Vec<f64> {
    let mut acc = vec![0f64; k];
    let mut n = 0;
    for &id in ids {
        let id = id as usize;
        let w = weights[id] as f64;
        if w == 0.0 {
            continue;
        }
        n += 1;
        for (a, &v) in acc.iter_mut().zip(&table[id * k..(id + 1) * k]) {
            *a += w * v as f64;
        }
    }
    if n > 0 {
        for a in &mut acc {
            *a /= n as f64;
        }
    }
    let norm = acc.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
    acc.iter().map(|v| v / norm).collect()
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

/// The share of texts whose nearest other text, by cosine, is of their
/// group; the first of equals is the nearest.
fn top1(vecs: &[Vec<f64>], groups: &[u32]) -> f64 {
    let n = vecs.len();
    let hits = (0..n)
        .filter(|&i| {
            let mut best = (f64::NEG_INFINITY, i);
            for j in (0..n).filter(|&j| j != i) {
                let c = cosine(&vecs[i], &vecs[j]);
                if c > best.0 {
                    best = (c, j);
                }
            }
            groups[best.1] == groups[i]
        })
        .count();
    hits as f64 / n as f64
}

/// Spearman's rank correlation of the two models' cosines over every
/// pair of texts, ties sharing their mean rank.
fn spearman(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let n = a.len();
    let pairs: Vec<(usize, usize)> = (0..n).flat_map(|i| (i + 1..n).map(move |j| (i, j))).collect();
    let xa: Vec<f64> = pairs.iter().map(|&(i, j)| cosine(&a[i], &a[j])).collect();
    let xb: Vec<f64> = pairs.iter().map(|&(i, j)| cosine(&b[i], &b[j])).collect();
    pearson(&ranks(&xa), &ranks(&xb))
}

fn ranks(x: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..x.len()).collect();
    order.sort_by(|&i, &j| x[i].total_cmp(&x[j]));
    let mut r = vec![0f64; x.len()];
    let mut i = 0;
    while i < order.len() {
        let mut j = i;
        while j + 1 < order.len() && x[order[j + 1]] == x[order[i]] {
            j += 1;
        }
        let mean = (i + j) as f64 / 2.0;
        for &o in &order[i..=j] {
            r[o] = mean;
        }
        i = j + 1;
    }
    r
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let cov: f64 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let va: f64 = a.iter().map(|x| (x - ma).powi(2)).sum();
    let vb: f64 = b.iter().map(|y| (y - mb).powi(2)).sum();
    cov / (va * vb).sqrt()
}

/// Run the reference script in the recipe's pinned container on the
/// staged bundle, and put the reference file and its report in it.
pub fn reference(recipe: &Recipe, bundle: &Path) -> Result<()> {
    let container = recipe.str_at("/reference/produced_by/container")?;
    let image = crate::reference::present(container)?;
    let p = paths(recipe)?;
    let work = crate::reference::scratch(bundle)?;
    fs::copy(cases_path(bundle), work.join("cases.json")).map_err(|e| format!("cases: {e}"))?;
    let abs = |p: &Path| fs::canonicalize(p).map_err(|e| format!("{}: {e}", p.display()));
    let mut cmd = Command::new("docker");
    cmd.args(["run", "--rm", "--network", "none", "--entrypoint", "python", "--mount"])
        .arg(format!("type=bind,src={},dst=/model,readonly", abs(bundle)?.display()))
        .arg("--mount")
        .arg(format!("type=bind,src={},dst=/work", abs(&work)?.display()));
    if let Some(user) = crate::reference::current_user() {
        cmd.args(["--user", &user]);
    }
    cmd.arg(&image).args([
        "/static_reference.py",
        "/model",
        "/work/cases.json",
        "/work/reference.safetensors",
        "/work/produced_by.json",
    ]);
    let out = cmd.output().map_err(|e| format!("docker: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "the reference container failed:\n{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let bytes = fs::read(work.join("reference.safetensors")).map_err(|e| format!("reference output: {e}"))?;
    crate::fetch::write_atomic(&bundle.join(&p.reference), &bytes)?;
    let mut reported: Value = serde_json::from_slice(
        &fs::read(work.join("produced_by.json")).map_err(|e| format!("produced_by output: {e}"))?,
    )
    .map_err(|e| format!("produced_by output: {e}"))?;
    reported["container"] = json!(container);
    crate::fetch::write_atomic(
        &bundle.join(format!("{}.report.json", p.reference)),
        &serde_json::to_vec_pretty(&reported).unwrap(),
    )?;
    let _ = fs::remove_dir_all(&work);
    Ok(())
}

/// Seal a staged bundle: the table's report fills `static_embedding` and
/// the artifact's `produced_by`, the reference's report the reference's.
/// Then the core loads the bundle on the CPU, and its vectors must match
/// the reference's and the tool's own for the quality texts.
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
    let reference_report_path = bundle.join(format!("{}.report.json", p.reference));
    if !bundle.join(&p.reference).is_file() {
        return Err(format!(
            "{} is not in the bundle: run static_reference.py on the staged bundle (docs/static.md)",
            p.reference
        ));
    }
    let reference_report = read(&reference_report_path)?;
    let container = reference_report["container"]
        .as_str()
        .filter(|c| !c.is_empty() && *c != "host")
        .ok_or("the reference report names no container")?;
    let produced_by = crate::reference::produced_by(&reference_report, container)?;

    let m = &mut recipe.manifest;
    m["static_embedding"]["distilled_from"] = table_report["distilled_from"].clone();
    m["static_embedding"]["quality"] = table_report["quality"].clone();
    let mut pb = json!({});
    for key in ["tool", "tool_version", "container", "args", "reproducible"] {
        pb[key] = table_report[key].clone();
    }
    m["artifacts"][0]["produced_by"] = pb;
    m["reference"]["produced_by"] = produced_by;
    // The reports leave the bundle, which lists every file in it; they are
    // put back if it does not seal, so the run can be repeated.
    let kept: Vec<(PathBuf, Vec<u8>)> = [table_report_path, reference_report_path, cases_path(bundle)]
        .into_iter()
        .map(|path| {
            let bytes = fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            Ok((path, bytes))
        })
        .collect::<Result<_>>()?;
    for (path, _) in &kept {
        fs::remove_file(path).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    let sealed = crate::seal::seal_manifest(m.clone(), bundle).and_then(|()| check(bundle, &table_report["quality"]));
    if let Err(e) = sealed {
        let _ = fs::remove_file(bundle.join("manifest.json"));
        for (path, bytes) in &kept {
            crate::fetch::write_atomic(path, bytes)?;
        }
        return Err(e);
    }
    println!("{}: verified", bundle.display());
    Ok(())
}

/// The sealed bundle on the library's CPU device: every reference case
/// within the F32 tolerance of the reference (docs/conformance.md), and
/// the quality texts within it of the tool's own arithmetic, ranking to
/// the same static_top1.
pub fn check(bundle: &Path, quality: &Value) -> Result<()> {
    let b = Bundle::open(bundle).map_err(|e| e.message)?;
    let m = &b.manifest;
    let st = m.static_embedding.as_ref().ok_or("not a static bundle")?;
    let k = m.embed().dim as usize;
    let rt = api::Runtime::create()?;
    let ctx = rt.cpu()?;
    let model = ctx.load(bundle)?;
    let info = model.info()?;
    let session = model.session(info.max_batch, info.max_seq)?;
    let o = api::options(0, 0);
    let close = |what: &str, got: &[f32], want: &[f64]| -> Result<()> {
        let g: Vec<f64> = got.iter().map(|&v| v as f64).collect();
        let cos = cosine(&g, want);
        let diff = g.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f64, f64::max);
        if cos < 0.9999 || diff > 1e-4 {
            return Err(format!("{what}: cosine {cos}, largest difference {diff} from the library's vector"));
        }
        Ok(())
    };

    let bytes = b.read_verified(&m.reference.file).map_err(|e| e.message)?;
    let refs = safetensors::File::parse(&m.reference.file, &bytes).map_err(|e| e.message)?;
    let want = refs.get("embeddings", Dtype::F32, 2).map_err(|e| e.message)?.f32s();
    let mut got = vec![0f32; k];
    for (i, c) in m.reference.cases.iter().enumerate() {
        let role = match c.prompt_role {
            PromptRole::None => 0,
            PromptRole::Query => turbo::TURBO_PROMPT_QUERY,
            PromptRole::Document => turbo::TURBO_PROMPT_DOCUMENT,
        };
        let mut oc = o;
        oc.prompt_role = role;
        session.texts(&[c.text.as_str()], &oc, &mut got)?;
        let w: Vec<f64> = want[i * k..(i + 1) * k].iter().map(|&v| v as f64).collect();
        close(&format!("reference case {i}"), &got, &w)?;
    }

    // The table as stored, and the tool's arithmetic on it.
    let art = &m.artifacts[0];
    let wbytes = b.read_verified(&art.files[0]).map_err(|e| e.message)?;
    let wf = safetensors::File::parse(&art.files[0], &wbytes).map_err(|e| e.message)?;
    let floats = |name: &str| -> Result<Vec<f32>> {
        let t = wf.tensor(name).ok_or(format!("{}: no {name}", art.files[0]))?;
        Ok(match t.dtype {
            Dtype::F32 => t.f32s(),
            Dtype::F16 => t.data.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            _ => return Err(format!("{}: {name} is {}", art.files[0], t.dtype_name)),
        })
    };
    let (table, weights) = (floats(EMBEDDINGS)?, floats(WEIGHTS)?);
    let tok = Tokenizer::load(&b).map_err(|e| e.message)?;
    let texts = quality_texts(&bundle.join(&st.quality.texts))?;
    let mut lib = Vec::with_capacity(texts.len());
    for (i, t) in texts.iter().enumerate() {
        let ids = static_ids(&tok, &t.text, m.embed().max_seq)?;
        let mine = static_vector(&table, &weights, k, &ids);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobi_finds_the_eigenpairs_of_a_symmetric_matrix() {
        // A = Q diag(5, 2, -1) Q^T for a rotation Q.
        let (c, s) = (0.6f64, 0.8f64);
        let q = [c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0];
        let d = [5.0, 2.0, -1.0];
        let mut a = vec![0f64; 9];
        for i in 0..3 {
            for j in 0..3 {
                a[i * 3 + j] = (0..3).map(|k| q[i * 3 + k] * d[k] * q[j * 3 + k]).sum();
            }
        }
        let orig = a.clone();
        let (vals, vecs) = jacobi(&mut a, 3);
        let mut sorted = vals.clone();
        sorted.sort_by(|a, b| b.total_cmp(a));
        for (g, w) in sorted.iter().zip(d) {
            assert!((g - w).abs() < 1e-12, "{sorted:?}");
        }
        // A v = lambda v for every column.
        for col in 0..3 {
            for i in 0..3 {
                let av: f64 = (0..3).map(|k| orig[i * 3 + k] * vecs[k * 3 + col]).sum();
                assert!((av - vals[col] * vecs[i * 3 + col]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn pca_keeps_the_direction_of_most_variance() {
        // Points along (1, 1, 0) with a little noise on z: the first
        // component is (1, 1, 0) / sqrt 2, and it holds almost all variance.
        let rows: Vec<f64> = (0..50)
            .flat_map(|i| {
                let t = i as f64 - 24.5;
                [t + 3.0, t - 1.0, 0.01 * (i % 3) as f64]
            })
            .collect();
        let (proj, explained) = pca(&rows, 50, 3, 1);
        assert!(explained > 0.999, "{explained}");
        let r2 = std::f64::consts::SQRT_2;
        for (i, p) in proj.iter().enumerate() {
            let t = i as f64 - 24.5;
            assert!((p - t * r2).abs() < 0.02, "row {i}: {p} vs {}", t * r2);
        }
    }

    #[test]
    fn spearman_ranks_ties_by_their_mean() {
        assert_eq!(ranks(&[3.0, 1.0, 3.0, 2.0]), vec![2.5, 0.0, 2.5, 1.0]);
        let a = vec![vec![1.0, 0.0], vec![0.9, 0.1], vec![0.0, 1.0]];
        assert!((spearman(&a, &a) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn the_table_file_is_what_the_core_reads() {
        let file = write_table(&[1.0, -2.0, 0.5, 4.0], &[1.0, 0.0], 2, 2, "F16");
        let f = safetensors::File::parse("t", &file).unwrap();
        let e = f.tensor(EMBEDDINGS).unwrap();
        assert_eq!(e.shape, [2, 2]);
        assert_eq!(e.dtype, Dtype::F16);
        let header = u64::from_le_bytes(file[..8].try_into().unwrap());
        assert_eq!(header % 8, 0);
        assert_eq!(f.tensor(WEIGHTS).unwrap().shape, [2]);
    }
}
