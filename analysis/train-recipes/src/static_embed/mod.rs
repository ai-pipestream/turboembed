//! A permissively trained static (table) encoder: the table starts from
//! BERT-base-uncased's token embedding matrix (Apache-2.0 weights, counted at
//! their own licence as elsewhere in `docs/licences.md`), centered, reduced by
//! PCA and SIF-scaled the way `turbo-bundle distill` initializes a table, and is
//! then trained by InfoNCE on permissive sentence pairs: adjacent sentences of
//! FineWeb documents (ODC-By) and of multi-sentence Tatoeba CC0 rows. No
//! teacher scores and no fine-tuned checkpoint are involved after
//! initialization, so the trained table's lineage is the teacher weights plus
//! the named corpora.
//!
//! The sentence vector is the mean of its token rows, L2-normalized, exactly
//! the static runtime's pooling, so what is trained is what ships.
//! Tokenization is the core tokenizer's, loaded from the staged bundle.

use std::path::Path;

use turbo::bundle::Bundle;
use turbo::manifest::{PromptRole, Truncation};
use turbo::tokenizer::{Encode, Tokenizer};

use crate::error::Error;

pub mod bundle;
pub mod data;

/// Training knobs, all recorded on the card.
#[derive(Clone, Debug)]
pub struct Config {
    pub dim: usize,
    pub batch: usize,
    pub steps: usize,
    pub lr: f32,
    pub temperature: f32,
    pub max_tokens: u32,
    pub seed: u64,
    pub sif_coefficient: f64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            dim: 256,
            batch: 256,
            steps: 4000,
            lr: 1e-3,
            temperature: 0.05,
            max_tokens: 64,
            seed: 879,
            sif_coefficient: 1e-4,
        }
    }
}

/// SplitMix64: the run's only randomness (batch shuffles), seeded
/// from the config so a rerun at the same inputs trains identically.
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Read a finite, nonempty F32 teacher matrix through the core's
/// checked safetensors parser before PCA can index its rows.
pub fn read_tensor(bytes: &[u8], name: &str) -> Result<(Vec<f32>, Vec<usize>), Error> {
    use turbo::safetensors::{Dtype, File};
    let fail = |e: turbo::status::Error| Error::Model(e.message);
    let file = File::parse("teacher safetensors", bytes).map_err(fail)?;
    let tensor = file.get(name, Dtype::F32, 2).map_err(fail)?;
    let shape = tensor
        .shape
        .iter()
        .map(|&d| usize::try_from(d))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| Error::Model("teacher dimensions exceed address space".into()))?;
    if shape.contains(&0) {
        return Err(Error::Model("teacher matrix must be nonempty".into()));
    }
    let values = tensor.f32s();
    if values.iter().any(|v| !v.is_finite()) {
        return Err(Error::Model("teacher matrix must contain finite values".into()));
    }
    Ok((values, shape))
}

/// Centered top-`dim` principal components of the [rows x cols]
/// matrix, as [cols x dim] (column-major by component), by cyclic
/// Jacobi eigen decomposition of the covariance.
fn pca_components(x: &[f32], rows: usize, cols: usize, dim: usize) -> (Vec<f32>, Vec<f32>) {
    let mut mean = vec![0.0f64; cols];
    for r in 0..rows {
        for c in 0..cols {
            mean[c] += x[r * cols + c] as f64;
        }
    }
    for m in &mut mean {
        *m /= rows as f64;
    }
    // Covariance in f64, rows in parallel chunks merged in order.
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get().min(16));
    let chunk_rows = rows.div_ceil(threads);
    let parts: Vec<Vec<f64>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let (x, mean) = (x, &mean);
                s.spawn(move || {
                    let mut cov = vec![0.0f64; cols * cols];
                    let lo = t * chunk_rows;
                    let hi = ((t + 1) * chunk_rows).min(rows);
                    let mut row = vec![0.0f64; cols];
                    for r in lo..hi {
                        for c in 0..cols {
                            row[c] = x[r * cols + c] as f64 - mean[c];
                        }
                        for i in 0..cols {
                            if row[i] == 0.0 {
                                continue;
                            }
                            for j in i..cols {
                                cov[i * cols + j] += row[i] * row[j];
                            }
                        }
                    }
                    cov
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut cov = vec![0.0f64; cols * cols];
    for p in &parts {
        for (a, b) in cov.iter_mut().zip(p) {
            *a += b;
        }
    }
    for i in 0..cols {
        for j in i + 1..cols {
            cov[j * cols + i] = cov[i * cols + j];
        }
    }
    // Cyclic Jacobi rotations until the off-diagonal mass is spent.
    let mut v = vec![0.0f64; cols * cols];
    for i in 0..cols {
        v[i * cols + i] = 1.0;
    }
    for _ in 0..12 {
        let mut off = 0.0f64;
        for i in 0..cols {
            for j in i + 1..cols {
                off += cov[i * cols + j] * cov[i * cols + j];
            }
        }
        if off.sqrt() <= 1e-9 * (rows as f64) {
            break;
        }
        for p in 0..cols {
            for q in p + 1..cols {
                let apq = cov[p * cols + q];
                if apq.abs() < 1e-12 {
                    continue;
                }
                let tau = (cov[q * cols + q] - cov[p * cols + p]) / (2.0 * apq);
                let t = tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for k in 0..cols {
                    let (kp, kq) = (cov[k * cols + p], cov[k * cols + q]);
                    cov[k * cols + p] = c * kp - s * kq;
                    cov[k * cols + q] = s * kp + c * kq;
                }
                for k in 0..cols {
                    let (pk, qk) = (cov[p * cols + k], cov[q * cols + k]);
                    cov[p * cols + k] = c * pk - s * qk;
                    cov[q * cols + k] = s * pk + c * qk;
                }
                for k in 0..cols {
                    let (kp, kq) = (v[k * cols + p], v[k * cols + q]);
                    v[k * cols + p] = c * kp - s * kq;
                    v[k * cols + q] = s * kp + c * kq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..cols).collect();
    order.sort_by(|&a, &b| cov[b * cols + b].partial_cmp(&cov[a * cols + a]).unwrap());
    let mut comps = vec![0.0f32; cols * dim];
    for (k, &src) in order.iter().take(dim).enumerate() {
        for r in 0..cols {
            comps[k * cols + r] = v[r * cols + src] as f32;
        }
    }
    (comps, mean.iter().map(|&m| m as f32).collect())
}

/// The initial table: teacher rows centered, projected on the top
/// components, and SIF-scaled by vocabulary rank (the Zipf weighting
/// of `turbo-bundle distill`), as [vocab x dim].
pub fn initial_table(teacher: &[f32], vocab: usize, in_dim: usize, dim: usize, sif: f64) -> Vec<f32> {
    let (comps, mean) = pca_components(teacher, vocab, in_dim, dim);
    let zipf: f64 = (0..vocab).map(|r| 1.0 / (r as f64 + 2.0)).sum();
    let mut table = vec![0.0f32; vocab * dim];
    for r in 0..vocab {
        let p = (1.0 / (r as f64 + 2.0)) / zipf;
        let w = (sif / (sif + p)) as f32;
        let row = &teacher[r * in_dim..][..in_dim];
        for k in 0..dim {
            let mut dot = 0.0f32;
            for c in 0..in_dim {
                dot += (row[c] - mean[c]) * comps[k * in_dim + c];
            }
            table[r * dim + k] = dot * w;
        }
    }
    table
}

/// Sentences of one document: split on line breaks and sentence
/// terminators, trimmed, 20 to 400 chars. Training pairs tolerate
/// the noise of this split; the tokenizer later decides the tokens.
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for para in text.split('\n') {
        let para = para.trim();
        let mut start = 0;
        let bytes = para.as_bytes();
        for i in 1..bytes.len().saturating_sub(1) {
            if matches!(bytes[i], b'.' | b'?' | b'!') && bytes[i + 1] == b' ' {
                push_sentence(&mut out, &para[start..=i]);
                start = i + 2;
            }
        }
        if start < para.len() {
            push_sentence(&mut out, &para[start..]);
        }
    }
    out
}

fn push_sentence(out: &mut Vec<String>, s: &str) {
    let s = s.trim();
    if (20..=400).contains(&s.chars().count())
        && s.chars().filter(|c| c.is_alphabetic()).count() * 2 >= s.chars().count()
    {
        out.push(s.to_owned());
    }
}

/// Token ids of every sentence of every document, by the bundle's
/// tokenizer with no special tokens, sentences that come out empty
/// or over `max_tokens` dropped.
pub fn tokenize_docs(tok: &Tokenizer, docs: &[Vec<String>], max_tokens: u32) -> Result<Vec<Vec<Vec<i32>>>, Error> {
    let opts =
        Encode { add_special_tokens: false, truncation: Truncation::Right, max_tokens: 4096, prompt: PromptRole::None };
    let mut out = Vec::with_capacity(docs.len());
    for doc in docs {
        let mut sents = Vec::new();
        for s in doc {
            let ids = tok.encode(s, opts).map_err(|e| Error::Model(e.message))?;
            if !ids.is_empty() && ids.len() as u32 <= max_tokens {
                sents.push(ids);
            }
        }
        if !sents.is_empty() {
            out.push(sents);
        }
    }
    Ok(out)
}

/// The tokenizer of the bundle at `dir`, unchecked (its reference may
/// still hold the placeholder table's vectors while staging).
pub fn tokenizer_of(dir: &Path) -> Result<Tokenizer, Error> {
    let b = Bundle::open(dir).map_err(|e| Error::Model(e.message))?;
    Tokenizer::unchecked(&b).map_err(|e| Error::Model(e.message))
}

fn encode_mean(table: &[f32], dim: usize, ids: &[i32], out: &mut [f32]) {
    out.fill(0.0);
    for &id in ids {
        let row = &table[id as usize * dim..][..dim];
        for (o, &w) in out.iter_mut().zip(row) {
            *o += w;
        }
    }
    let n = ids.len() as f32;
    let mut norm = 0.0f32;
    for o in out.iter_mut() {
        *o /= n;
        norm += *o * *o;
    }
    let inv = 1.0 / norm.sqrt().max(1e-12);
    for o in out.iter_mut() {
        *o *= inv;
    }
}

/// One training example: a sentence's ids and the next sentence
/// of its document.
struct Pair {
    a: Vec<i32>,
    b: Vec<i32>,
}

/// Adjacent-sentence pairs only. A 200-step smoke whose pairs were
/// token-dropout views of single sentences scored below the
/// untrained initialized table on scifact (nDCG@10 0.284 against
/// 0.335, document chunker), so subset views are not a pair source:
/// they reward identity under deletion, not topical alignment.
fn make_pairs(docs: &[Vec<Vec<i32>>]) -> Vec<Pair> {
    let mut pairs = Vec::new();
    for doc in docs {
        for (i, s) in doc.iter().enumerate() {
            if i + 1 < doc.len() {
                pairs.push(Pair { a: s.clone(), b: doc[i + 1].clone() });
            }
        }
    }
    pairs
}

/// Train `table` in place by bidirectional InfoNCE with in-batch
/// negatives and Adam. Returns (first-window mean loss, last-window
/// mean loss) over windows of 100 steps.
pub fn train(table: &mut [f32], vocab: usize, docs: &[Vec<Vec<i32>>], cfg: &Config) -> (f64, f64) {
    let dim = cfg.dim;
    let mut rng = Rng(cfg.seed);
    let pairs = make_pairs(docs);
    if pairs.is_empty() {
        return (0.0, 0.0);
    }
    let mut order: Vec<usize> = (0..pairs.len()).collect();
    let mut m = vec![0.0f32; vocab * dim];
    let mut v = vec![0.0f32; vocab * dim];
    let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
    let mut losses: Vec<f64> = Vec::new();
    let mut ea = vec![0.0f32; cfg.batch * dim];
    let mut eb = vec![0.0f32; cfg.batch * dim];
    let mut grad = vec![0.0f32; vocab * dim];
    for t in 0..cfg.batch * dim {
        ea[t] = 0.0;
        eb[t] = 0.0;
    }
    for step in 1..=cfg.steps {
        if (step - 1) % (pairs.len() / cfg.batch).max(1) == 0 {
            for i in (1..order.len()).rev() {
                order.swap(i, rng.below(i + 1));
            }
        }
        let start = ((step - 1) * cfg.batch) % pairs.len();
        let idx: Vec<usize> = (0..cfg.batch).map(|k| order[(start + k) % order.len()]).collect();
        for (slot, &pi) in idx.iter().enumerate() {
            encode_mean(table, dim, &pairs[pi].a, &mut ea[slot * dim..][..dim]);
            encode_mean(table, dim, &pairs[pi].b, &mut eb[slot * dim..][..dim]);
        }
        // Similarities, both directions' cross entropy, and gradients
        // wrt the sentence vectors.
        let nb = cfg.batch;
        let mut ga = vec![0.0f32; nb * dim];
        let mut gb = vec![0.0f32; nb * dim];
        let mut loss = 0.0f64;
        for dir in 0..2 {
            let (src, dst, gsrc, gdst) =
                if dir == 0 { (&ea, &eb, &mut ga, &mut gb) } else { (&eb, &ea, &mut gb, &mut ga) };
            for i in 0..nb {
                let mut sims = vec![0.0f32; nb];
                let mut mx = f32::NEG_INFINITY;
                for j in 0..nb {
                    let mut d = 0.0f32;
                    for k in 0..dim {
                        d += src[i * dim + k] * dst[j * dim + k];
                    }
                    sims[j] = d / cfg.temperature;
                    mx = mx.max(sims[j]);
                }
                let mut sum = 0.0f32;
                for s in sims.iter_mut() {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                loss += (mx + sum.ln() - {
                    let mut d = 0.0f32;
                    for k in 0..dim {
                        d += src[i * dim + k] * dst[i * dim + k];
                    }
                    d / cfg.temperature
                }) as f64;
                for j in 0..nb {
                    let p = sims[j] / sum - if j == i { 1.0 } else { 0.0 };
                    let g = p / cfg.temperature / (2 * nb) as f32;
                    for k in 0..dim {
                        gsrc[i * dim + k] += g * dst[j * dim + k];
                        gdst[j * dim + k] += g * src[i * dim + k];
                    }
                }
            }
        }
        losses.push(loss / (2 * nb) as f64);
        // Through mean pooling and L2 normalization into token rows.
        for (slot, &pi) in idx.iter().enumerate() {
            for (ids, e, g) in [(&pairs[pi].a, &ea, &ga), (&pairs[pi].b, &eb, &gb)] {
                let s = &e[slot * dim..][..dim];
                let gs = &g[slot * dim..][..dim];
                let dot: f32 = s.iter().zip(gs).map(|(a, b)| a * b).sum();
                // |mean| before normalization is not tracked; the
                // normalization Jacobian uses the unit vector and the
                // mean length is folded into the 1/n mean gradient
                // by recomputing the unnormalized mean norm.
                let mut raw = vec![0.0f32; dim];
                for &id in ids.iter() {
                    for (o, &w) in raw.iter_mut().zip(&table[id as usize * dim..][..dim]) {
                        *o += w;
                    }
                }
                let len = (raw.iter().map(|x| x * x).sum::<f32>() / (ids.len() * ids.len()) as f32).sqrt().max(1e-12);
                for &id in ids.iter() {
                    let row = &mut grad[id as usize * dim..][..dim];
                    for k in 0..dim {
                        row[k] += (gs[k] - s[k] * dot) / len / ids.len() as f32;
                    }
                }
            }
        }
        // Adam over rows the batch touched.
        let bc1 = 1.0 - b1.powi(step as i32);
        let bc2 = 1.0 - b2.powi(step as i32);
        let mut touched: Vec<i32> =
            idx.iter().flat_map(|&pi| pairs[pi].a.iter().chain(&pairs[pi].b).copied()).collect();
        touched.sort_unstable();
        touched.dedup();
        for id in touched {
            let base = id as usize * dim;
            for k in 0..dim {
                let g = grad[base + k];
                if g == 0.0 {
                    continue;
                }
                m[base + k] = b1 * m[base + k] + (1.0 - b1) * g;
                v[base + k] = b2 * v[base + k] + (1.0 - b2) * g * g;
                let mh = m[base + k] / bc1;
                let vh = v[base + k] / bc2;
                table[base + k] -= cfg.lr * mh / (vh.sqrt() + eps);
                grad[base + k] = 0.0;
            }
        }
    }
    let win = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len().max(1) as f64;
    let n = losses.len().min(100);
    (win(&losses[..n]), win(&losses[losses.len().saturating_sub(100)..]))
}

/// A minimal safetensors writer: header JSON then raw F32 data, the
/// container the bundle tool and the core read.
pub fn write_safetensors(tensors: &[(&str, &[usize], Vec<f32>)]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, shape, values) in tensors {
        let lo = data.len();
        for v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        header.insert(
            name.to_string(),
            serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [lo, data.len()]}),
        );
    }
    let mut header = serde_json::to_string(&serde_json::Value::Object(header)).unwrap();
    // The core reader requires the header length to be a multiple
    // of 8; JSON permits the trailing spaces that pad it.
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&data);
    out
}

#[cfg(test)]
mod tests;
