//! A static bundle's reference, computed by the tool itself: no container
//! and no other program.
//!
//! A case's ids are the core tokenizer's, as the bundle runs it: the
//! static rules (docs/static.md) at `static_embedding.max_length`. Its
//! vector is computed here from the table as stored, one value at a time,
//! by the rules docs/static.md states: each id's row (the row the mapping
//! names, when the table has one) times the id's weight (no product
//! without weights), summed in id order and divided by the count, in F64
//! for an I8 or F64 table or F64 weights and in F32 otherwise, the mean
//! rounded to F32, and to F16 for an F16 table; then, under NORMALIZE_L2,
//! divided by its norm plus 1e-32, the squares summed pairwise, and
//! rounded to F16 again for an F16 table. No ids give the zero vector.
//!
//! This is plain code apart from the library's: the library's vectors
//! must equal it to the bit (`check`), whichever of its kernels runs.

use std::fs;
use std::path::Path;

use serde_json::{Value, json};
use turbo::bundle::Bundle;
use turbo::manifest::{Manifest, Normalize, Pooling, PromptRole, TensorRole, Truncation};
use turbo::safetensors::{self, Dtype, Tensor};
use turbo::tokenizer::{Encode, Tokenizer};

use crate::Result;
use crate::recipe::Recipe;

/// The table as stored, read value by value.
struct Table<'a> {
    rows: &'a Tensor<'a>,
    dim: usize,
    weights: Option<Vec<f64>>,
    mapping: Option<Vec<usize>>,
    /// The sum runs in F64.
    wide: bool,
}

impl Table<'_> {
    /// Value j of row `row`, exact.
    fn at64(&self, row: usize, j: usize) -> f64 {
        let i = row * self.dim + j;
        let d = self.rows.data;
        let two = |i: usize| u16::from_le_bytes([d[2 * i], d[2 * i + 1]]);
        match self.rows.dtype {
            Dtype::F32 => f32::from_le_bytes(d[4 * i..4 * i + 4].try_into().unwrap()) as f64,
            Dtype::F16 => half::f16::from_bits(two(i)).to_f64(),
            Dtype::Bf16 => half::bf16::from_bits(two(i)).to_f64(),
            Dtype::F64 => f64::from_le_bytes(d[8 * i..8 * i + 8].try_into().unwrap()),
            _ => d[i] as i8 as f64,
        }
    }

    /// The mean of the rows of `ids`, the arithmetic in the module comment.
    fn mean(&self, ids: &[i32]) -> Vec<f32> {
        let n = ids.len();
        let mut out = vec![0f32; self.dim];
        if n == 0 {
            return out;
        }
        for (j, o) in out.iter_mut().enumerate() {
            if self.wide {
                let mut acc = 0f64;
                for &id in ids {
                    let v = self.at64(self.row_of(id), j);
                    acc += match &self.weights {
                        Some(w) => v * w[id as usize],
                        None => v,
                    };
                }
                *o = (acc / n as f64) as f32;
            } else {
                let mut acc = 0f32;
                for &id in ids {
                    // Every stored value but an F64 one is an F32 exactly.
                    let v = self.at64(self.row_of(id), j) as f32;
                    acc += match &self.weights {
                        Some(w) => v * w[id as usize] as f32,
                        None => v,
                    };
                }
                *o = acc / n as f32;
            }
        }
        out
    }

    fn row_of(&self, id: i32) -> usize {
        self.mapping.as_ref().map_or(id as usize, |m| m[id as usize])
    }
}

/// The sum of the squares of `x`, each rounded to F32, summed pairwise:
/// under 8 values one by one; up to 128 in eight running sums, added as a
/// tree, then the rest one by one; above that, the two halves (the first
/// a multiple of 8) apart.
fn pairwise_squares(x: &[f32]) -> f32 {
    let n = x.len();
    if n < 8 {
        return x.iter().fold(0f32, |s, v| s + v * v);
    }
    if n > 128 {
        let half = n / 2 - (n / 2) % 8;
        return pairwise_squares(&x[..half]) + pairwise_squares(&x[half..]);
    }
    let mut r: Vec<f32> = x[..8].iter().map(|v| v * v).collect();
    let whole = n - n % 8;
    for block in x[8..whole].as_chunks::<8>().0 {
        for (s, v) in r.iter_mut().zip(block) {
            *s += v * v;
        }
    }
    let tree = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    x[whole..].iter().fold(tree, |s, v| s + v * v)
}

fn round_f16(x: &mut [f32]) {
    for v in x {
        *v = half::f16::from_f32(*v).to_f32();
    }
}

/// A staged static bundle's vector for each of `texts` (each with its
/// prompt role), and its ids, from the tokenizer and the table.
fn vectors(m: &Manifest, bundle: &Bundle, tok: &Tokenizer, texts: &[(&str, PromptRole)]) -> Result<Rows> {
    let st = m.static_embedding.as_ref().ok_or("not a static bundle")?;
    let art = m.artifacts.first().ok_or("manifest.artifacts: empty")?;
    let file = &art.files[0];
    let bytes = bundle.read_verified(file).map_err(|e| e.message)?;
    let sf = safetensors::File::parse(file, &bytes).map_err(|e| e.message)?;
    let name = |role: TensorRole| art.tensor_names.get(&role).map(String::as_str);
    let tensor = |n: &str| sf.tensor(n).ok_or_else(|| format!("{file}: no tensor {n}"));
    let rows = tensor(name(TensorRole::StaticEmbeddings).ok_or("artifacts[0].tensor_names: no static_embeddings")?)?;
    if !matches!(rows.dtype, Dtype::F32 | Dtype::F16 | Dtype::Bf16 | Dtype::F64 | Dtype::I8) || rows.shape.len() != 2 {
        return Err(format!("{file}: the table is {} {:?}", rows.dtype_name, rows.shape));
    }
    let dim = rows.shape[1] as usize;
    let weights = match name(TensorRole::StaticWeights) {
        None => None,
        Some(n) => {
            let w = tensor(n)?;
            let d = w.data;
            let v: Vec<f64> = match w.dtype {
                Dtype::F32 => d.as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c) as f64).collect(),
                Dtype::F16 => d.as_chunks::<2>().0.iter().map(|&c| half::f16::from_le_bytes(c).to_f64()).collect(),
                Dtype::F64 => d.as_chunks::<8>().0.iter().map(|&c| f64::from_le_bytes(c)).collect(),
                _ => return Err(format!("{file}: weights are {}", w.dtype_name)),
            };
            Some((v, w.dtype == Dtype::F64))
        }
    };
    let mapping = match name(TensorRole::StaticMapping) {
        None => None,
        Some(n) => {
            let t = tensor(n)?;
            let v: Vec<usize> = match t.dtype {
                Dtype::I32 => t.i32s().into_iter().map(|x| x as usize).collect(),
                Dtype::I64 => t.data.as_chunks::<8>().0.iter().map(|&c| i64::from_le_bytes(c) as usize).collect(),
                _ => return Err(format!("{file}: the mapping is {}", t.dtype_name)),
            };
            Some(v)
        }
    };
    let table = Table {
        rows,
        dim,
        wide: matches!(rows.dtype, Dtype::I8 | Dtype::F64) || weights.as_ref().is_some_and(|w| w.1),
        weights: weights.map(|w| w.0),
        mapping,
    };
    let embed = m.embed();
    if embed.pooling != Pooling::Mean {
        return Err("the tool computes a static bundle's reference under POOLING_MEAN only".into());
    }
    let l2 = embed.normalize == Normalize::L2;
    let half = rows.dtype == Dtype::F16;
    let opts = Encode {
        add_special_tokens: true,
        truncation: Truncation::Right,
        max_tokens: st.max_length,
        prompt: PromptRole::None,
    };
    let mut out = Rows { ids: Vec::new(), vectors: Vec::new() };
    for &(text, prompt) in texts {
        let ids = tok.encode(text, Encode { prompt, ..opts }).map_err(|e| e.message)?;
        let mut v = table.mean(&ids);
        if half {
            round_f16(&mut v);
        }
        if l2 {
            let norm = pairwise_squares(&v).sqrt() + 1e-32;
            for x in &mut v {
                *x /= norm;
            }
            if half {
                round_f16(&mut v);
            }
        }
        out.ids.push(ids);
        out.vectors.push(v);
    }
    Ok(out)
}

struct Rows {
    ids: Vec<Vec<i32>>,
    vectors: Vec<Vec<f32>>,
}

/// Write the reference file of the static bundle staged in `bundle`, whose
/// manifest is `m` without `files`, and return its `produced_by`. The
/// manifest is written for the tokenizer to load and removed again; the
/// caller seals the bundle.
pub fn write(m: &Value, bundle: &Path) -> Result<Value> {
    let file = m["reference"]["file"].as_str().ok_or("manifest.reference.file: missing")?.to_owned();
    crate::check_rel(&file)?;
    let produced_by = json!({
        "tool": "turbo-bundle",
        "tool_version": format!("{} (libturbo {})", env!("CARGO_PKG_VERSION"), crate::distill::turbo_version()),
        "container": "host",
        "args": ["static reference", "ids: the core tokenizer", "vectors: the table as stored"],
        "reproducible": true,
    });
    let mut staged = m.clone();
    staged["reference"]["produced_by"] = produced_by.clone();
    let manifest = bundle.join("manifest.json");
    crate::seal::write_manifest(&staged, bundle, Some(&file))?;
    let made = (|| {
        let b = Bundle::open(bundle).map_err(|e| e.message)?;
        let tok = Tokenizer::unchecked(&b).map_err(|e| e.message)?;
        let cases: Vec<(&str, _)> =
            b.manifest.reference.cases.iter().map(|c| (c.text.as_str(), c.prompt_role)).collect();
        let rows = vectors(&b.manifest, &b, &tok, &cases)?;
        Ok::<_, String>((rows, tok.pad_id, b.manifest.embed().dim as usize))
    })();
    let _ = fs::remove_file(&manifest);
    let (rows, pad, dim) = made?;
    crate::fetch::write_atomic(&bundle.join(&file), &safetensors_file(&rows, pad, dim))?;
    Ok(produced_by)
}

/// Make a static bundle from upstream files already staged in `bundle`:
/// check the upstream config against the recipe, write the reference,
/// seal, and check the library against the reference.
pub fn make(recipe: &Recipe, upstream: &Path, bundle: &Path) -> Result<()> {
    check_config(recipe, upstream)?;
    let produced_by = write(&recipe.manifest, bundle)?;
    crate::seal::seal(recipe, bundle, produced_by, Vec::new())?;
    check(bundle)
}

/// A Model2Vec model's `config.json`, when the recipe fetches one, must
/// say what the manifest says: `normalize` (absent is false) as
/// NORMALIZE_L2, and `max_length` (absent is 512) as
/// `static_embedding.max_length`.
fn check_config(recipe: &Recipe, upstream: &Path) -> Result<()> {
    let Some(u) = recipe.upstream.iter().find(|u| u.path == "config.json") else {
        return Ok(());
    };
    let path = upstream.join(&u.path);
    let config: Value = serde_json::from_slice(&fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let m = &recipe.manifest;
    let normalize = config["normalize"].as_bool().unwrap_or(false);
    if normalize != (m["embed"]["normalize"] == "NORMALIZE_L2") {
        return Err(format!("config.json says normalize {normalize}, the manifest {}", m["embed"]["normalize"]));
    }
    let max_length = config["max_length"].as_u64().unwrap_or(512);
    if Some(max_length) != m["static_embedding"]["max_length"].as_u64() {
        return Err(format!(
            "config.json says max_length {max_length}, the manifest {}",
            m["static_embedding"]["max_length"]
        ));
    }
    Ok(())
}

/// The sealed static bundle on the library's CPU device: each reference
/// case, embedded alone and all of them in one batch, gives the
/// reference's vector to the bit.
pub fn check(bundle: &Path) -> Result<()> {
    let b = Bundle::open(bundle).map_err(|e| e.message)?;
    let m = &b.manifest;
    let dim = m.embed().dim as usize;
    let bytes = b.read_verified(&m.reference.file).map_err(|e| e.message)?;
    let refs = safetensors::File::parse(&m.reference.file, &bytes).map_err(|e| e.message)?;
    let want = refs.get("embeddings", Dtype::F32, 2).map_err(|e| e.message)?.f32s();
    let rt = crate::api::Runtime::create()?;
    let ctx = rt.cpu()?;
    let model = ctx.load(bundle)?;
    let info = model.info()?;
    let session = model.session(info.max_batch, info.max_seq)?;
    let o = crate::api::options(0, 0);
    let role = |r: PromptRole| match r {
        PromptRole::None => 0,
        PromptRole::Query => turbo::TURBO_PROMPT_QUERY,
        PromptRole::Document => turbo::TURBO_PROMPT_DOCUMENT,
    };
    let mut got = vec![0f32; dim];
    for (i, c) in m.reference.cases.iter().enumerate() {
        session.texts(
            &[c.text.as_str()],
            &turbo::turbo_embed_options { prompt_role: role(c.prompt_role), ..o },
            &mut got,
        )?;
        let w = &want[i * dim..(i + 1) * dim];
        if let Some(j) = (0..dim).find(|&j| got[j].to_bits() != w[j].to_bits()) {
            return Err(format!(
                "reference case {i}: value {j} is {} in the library, {} in the reference",
                got[j], w[j]
            ));
        }
    }
    // A batch gives each text the vector it gets alone.
    let plain: Vec<usize> =
        (0..m.reference.cases.len()).filter(|&i| m.reference.cases[i].prompt_role == PromptRole::None).collect();
    for chunk in plain.chunks(info.max_batch as usize) {
        let texts: Vec<&str> = chunk.iter().map(|&i| m.reference.cases[i].text.as_str()).collect();
        let mut all = vec![0f32; texts.len() * dim];
        session.texts(&texts, &o, &mut all)?;
        for (row, &i) in all.chunks_exact(dim).zip(chunk) {
            if row.iter().zip(&want[i * dim..(i + 1) * dim]).any(|(a, b)| a.to_bits() != b.to_bits()) {
                return Err(format!("reference case {i}: its vector in a batch is not its vector alone"));
            }
        }
    }
    Ok(())
}

/// `ids` [n, width] padded with `pad`, `lengths` [n], `embeddings` [n, dim]
/// as one safetensors file.
fn safetensors_file(rows: &Rows, pad: i32, dim: usize) -> Vec<u8> {
    let n = rows.ids.len();
    let width = rows.ids.iter().map(Vec::len).max().unwrap_or(0).max(1);
    let mut ids = Vec::with_capacity(n * width * 4);
    for r in &rows.ids {
        for p in 0..width {
            ids.extend(r.get(p).copied().unwrap_or(pad).to_le_bytes());
        }
    }
    let lengths: Vec<u8> = rows.ids.iter().flat_map(|r| (r.len() as i32).to_le_bytes()).collect();
    let emb: Vec<u8> = rows.vectors.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
    let (a, b) = (ids.len(), ids.len() + lengths.len());
    let header = json!({
        "ids": { "dtype": "I32", "shape": [n, width], "data_offsets": [0, a] },
        "lengths": { "dtype": "I32", "shape": [n], "data_offsets": [a, b] },
        "embeddings": { "dtype": "F32", "shape": [n, dim], "data_offsets": [b, b + emb.len()] },
    });
    let mut h = serde_json::to_vec(&header).unwrap();
    while !h.len().is_multiple_of(8) {
        h.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + h.len() + b + emb.len());
    out.extend((h.len() as u64).to_le_bytes());
    out.extend(h);
    out.extend(ids);
    out.extend(lengths);
    out.extend(emb);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pairwise sum against the sum in F64 of the same F32 squares:
    /// close, and exact on values whose squares add exactly.
    #[test]
    fn the_pairwise_sum_adds_every_square_once() {
        for n in [0, 1, 7, 8, 9, 63, 128, 129, 300, 1024] {
            let x: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) / 4.0).collect();
            let want: f64 = x.iter().map(|v| (v * v) as f64).sum();
            assert_eq!(pairwise_squares(&x) as f64, want, "{n}");
        }
    }
}
