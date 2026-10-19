//! PCA, Jacobi eigendecomposition, safetensors table serialization, and
//! ranking metrics for static model distillation.

use serde_json::{Value, json};

use super::EMBEDDINGS;
use crate::Result;

pub(super) fn round4(v: f64) -> f64 {
    (v * 1e4).round() / 1e4
}

/// Each id's entry in the tokenizer file, for ids below `vocab_size`;
/// None for an id the file does not name.
pub(super) fn vocabulary(tokenizer_json: &[u8], vocab_size: usize) -> Result<Vec<Option<String>>> {
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

/// PCA of `rows` [n, h] to `k` dims: the rows centred on their mean and
/// projected on the covariance's k leading eigenvectors, each signed so
/// its largest entry is positive. Returns [n, k] and the share of the
/// variance kept.
pub(super) fn pca(rows: &[f64], n: usize, h: usize, k: usize) -> (Vec<f64>, f64) {
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
pub(super) fn jacobi(a: &mut [f64], n: usize) -> (Vec<f64>, Vec<f64>) {
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

/// The table as a safetensors file: `embeddings` [vocab, k], in `dtype`,
/// the header padded to a multiple of 8.
pub(super) fn write_table(table: &[f32], vocab: usize, k: usize, dtype: &str) -> Vec<u8> {
    let e: Vec<u8> = if dtype == "F16" {
        table.iter().flat_map(|&x| half::f16::from_f32(x).to_le_bytes()).collect()
    } else {
        table.iter().flat_map(|&x| x.to_le_bytes()).collect()
    };
    let header = json!({
        EMBEDDINGS: { "dtype": dtype, "shape": [vocab, k], "data_offsets": [0, e.len()] },
    });
    let mut h = serde_json::to_vec(&header).unwrap();
    while !h.len().is_multiple_of(8) {
        h.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + h.len() + e.len());
    out.extend((h.len() as u64).to_le_bytes());
    out.extend(h);
    out.extend(e);
    out
}

/// The median length of the vocabulary's entries in characters, rounded
/// down.
pub(super) fn median_chars(vocab: &[Option<String>]) -> usize {
    let mut lens: Vec<usize> = vocab.iter().flatten().map(|t| t.chars().count()).collect();
    lens.sort_unstable();
    match lens.len() {
        0 => 0,
        n if n % 2 == 1 => lens[n / 2],
        n => (lens[n / 2 - 1] + lens[n / 2]) / 2,
    }
}

/// The static vector of `ids` in f64, L2-normalized: the mean of the rows;
/// zeros for no ids.
pub(super) fn static_vector(table: &[f32], k: usize, ids: &[i32]) -> Vec<f64> {
    let mut acc = vec![0f64; k];
    for &id in ids {
        let id = id as usize;
        for (a, &v) in acc.iter_mut().zip(&table[id * k..(id + 1) * k]) {
            *a += v as f64;
        }
    }
    if !ids.is_empty() {
        for a in &mut acc {
            *a /= ids.len() as f64;
        }
    }
    let norm = acc.iter().map(|v| v * v).sum::<f64>().sqrt() + 1e-32;
    acc.iter().map(|v| v / norm).collect()
}

pub(super) fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

/// The share of texts whose nearest other text, by cosine, is of their
/// group; the first of equals is the nearest.
pub(super) fn top1(vecs: &[Vec<f64>], groups: &[u32]) -> f64 {
    let n = vecs.len();
    if n < 2 {
        return 0.0;
    }
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
pub(super) fn spearman(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
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
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let cov: f64 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let va: f64 = a.iter().map(|x| (x - ma).powi(2)).sum();
    let vb: f64 = b.iter().map(|y| (y - mb).powi(2)).sum();
    if va <= 0.0 || vb <= 0.0 { 0.0 } else { cov / (va * vb).sqrt() }
}

#[cfg(test)]
mod tests {
    use turbo::safetensors::{self, Dtype};

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
    fn spearman_ranks_ties_by_their_mean_and_handles_zero_variance() {
        assert_eq!(ranks(&[3.0, 1.0, 3.0, 2.0]), vec![2.5, 0.0, 2.5, 1.0]);
        let a = vec![vec![1.0, 0.0], vec![0.9, 0.1], vec![0.0, 1.0]];
        assert!((spearman(&a, &a) - 1.0).abs() < 1e-12);
        let constant = vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![1.0, 0.0]];
        assert_eq!(spearman(&constant, &a), 0.0);
        assert_eq!(spearman(&a, &constant), 0.0);
        assert_eq!(pearson(&[], &[]), 0.0);
        assert_eq!(pearson(&[2.0, 2.0, 2.0], &[1.0, 2.0, 3.0]), 0.0);
        assert_eq!(top1(&[], &[]), 0.0);
        assert_eq!(top1(&[vec![1.0, 0.0]], &[0]), 0.0);
    }

    #[test]
    fn the_table_file_is_what_the_core_reads() {
        let file = write_table(&[1.0, -2.0, 0.5, 4.0], 2, 2, "F16");
        let f = safetensors::File::parse("t", &file).unwrap();
        let e = f.tensor(EMBEDDINGS).unwrap();
        assert_eq!(e.shape, [2, 2]);
        assert_eq!(e.dtype, Dtype::F16);
        let header = u64::from_le_bytes(file[..8].try_into().unwrap());
        assert_eq!(header % 8, 0);
        assert!(f.tensor("weights").is_none());
    }
}
