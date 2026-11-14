//! Deterministic spherical k-means Voronoi centroid training and multi-probe
//! shard routing (`#2346`).
//!
//! Trained centroids serialize directly to the typed `TCENT001` binary format
//! via [`LabelCentroids`], so `TurboVocabularyService` and `analysis/zoo`
//! artifacts share a single wire representation.

use turbo_analysis::{Fnv1a, ModelRef, Tag, TagSet};

use crate::centroids::{CentroidNorm, LabelCentroids};

/// Training options for spherical k-means Voronoi centroids (`#2346`).
#[derive(Clone, Debug)]
pub struct VoronoiOptions {
    /// Target number of Voronoi cells (`k >= 1`).
    pub num_centroids: usize,
    /// Maximum Lloyd refinement iterations (`0` defaults to `25`).
    pub max_iters: usize,
    ///Deterministic seed for k-means++ cosine initialization.
    pub seed: u64,
    /// TagSet name stored in the `TCENT001` header.
    pub tag_set_name: String,
    /// Softmax temperature stored in the `TCENT001` header (`> 0`).
    pub temperature: f64,
}

impl Default for VoronoiOptions {
    fn default() -> Self {
        Self { num_centroids: 8, max_iters: 25, seed: 42, tag_set_name: "voronoi_shards".to_string(), temperature: 0.1 }
    }
}

/// Trained spherical k-means Voronoi centroids (`#2346`).
#[derive(Clone, Debug)]
pub struct VoronoiCentroids {
    centroids: LabelCentroids,
    cluster_counts: Vec<u64>,
    mean_cosine: f64,
}

impl VoronoiCentroids {
    /// Underlying `TCENT001`-compatible [`LabelCentroids`].
    pub fn label_centroids(&self) -> &LabelCentroids {
        &self.centroids
    }

    /// Number of Voronoi cells (`k`).
    pub fn num_centroids(&self) -> usize {
        self.centroids.tags().len()
    }

    /// Vector dimension (`dim`).
    pub fn dim(&self) -> u32 {
        self.centroids.dim()
    }

    /// Sample assignment counts per Voronoi cell from training.
    pub fn cluster_counts(&self) -> &[u64] {
        &self.cluster_counts
    }

    /// Mean cosine similarity of training vectors to their assigned centroid.
    pub fn mean_cosine(&self) -> f64 {
        self.mean_cosine
    }

    /// Serializes to `TCENT001` bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.centroids.to_bytes()
    }

    /// Loads Voronoi routing centroids from `TCENT001` bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let centroids = LabelCentroids::from_bytes(bytes)?;
        let k = centroids.tags().len();
        Ok(Self { centroids, cluster_counts: vec![0; k], mean_cosine: 0.0 })
    }

    /// Routes `vector` to the top-`probes` Voronoi cells by cosine similarity,
    /// returning `(cell_index, cosine)` sorted descending by cosine and
    /// tie-broken by `cell_index` ascending.
    pub fn route_top_probes(&self, vector: &[f32], probes: usize) -> Result<Vec<(u32, f32)>, String> {
        let dim = self.centroids.dim() as usize;
        if vector.len() != dim {
            return Err(format!("query vector has dim {}, expected {dim}", vector.len()));
        }
        let unit = l2_normalize_f64(vector)?;
        let k = self.num_centroids();
        let take = probes.clamp(1, k);
        let mut scored: Vec<(u32, f32)> = (0..k)
            .map(|c| {
                let cent = self.centroids.centroid_of(Tag(c as u16));
                let dot: f64 = unit.iter().zip(cent.iter()).map(|(&a, &b)| a * f64::from(b)).sum();
                (c as u32, dot as f32)
            })
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        scored.truncate(take);
        Ok(scored)
    }
}

/// Trains deterministic spherical k-means Voronoi centroids (`#2346`) over a
/// row-major `[num_samples, dim]` sample matrix.
pub fn train_voronoi_centroids(
    model: ModelRef,
    dim: usize,
    samples: &[f32],
    opts: &VoronoiOptions,
) -> Result<VoronoiCentroids, String> {
    if dim == 0 {
        return Err("Voronoi centroid training requires dim >= 1".into());
    }
    if opts.num_centroids == 0 {
        return Err("Voronoi centroid training requires num_centroids >= 1".into());
    }
    if !samples.len().is_multiple_of(dim) {
        return Err(format!("sample buffer length {} is not a multiple of dim {dim}", samples.len()));
    }
    let n = samples.len() / dim;
    if n < opts.num_centroids {
        return Err(format!(
            "need at least {} sample vectors to train {} centroids",
            opts.num_centroids, opts.num_centroids
        ));
    }
    let k = opts.num_centroids;
    let max_iters = if opts.max_iters == 0 { 25 } else { opts.max_iters };

    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        rows.push(l2_normalize_f64(&samples[i * dim..(i + 1) * dim])?);
    }

    // Deterministic spherical k-means++ initialization using angular distance
    // `max(0, 1 - cos(x, c))`.
    let mut rng = Fnv1a::new().u64(opts.seed).u64(n as u64).u64(k as u64).finish() | 1;
    let first = (next_u64(&mut rng) as usize) % n;
    let mut centers: Vec<Vec<f64>> = Vec::with_capacity(k);
    centers.push(rows[first].clone());
    let mut min_dist = vec![2.0f64; n];

    for _ in 1..k {
        let last = centers.last().unwrap();
        let mut total_w = 0.0f64;
        for (i, row) in rows.iter().enumerate() {
            let cos = dot_f64(row, last).clamp(-1.0, 1.0);
            let d = (1.0 - cos).max(0.0);
            if d < min_dist[i] {
                min_dist[i] = d;
            }
            total_w += min_dist[i];
        }
        let chosen = if total_w <= 1e-12 {
            (next_u64(&mut rng) as usize) % n
        } else {
            let target = unit_f64(&mut rng) * total_w;
            let mut acc = 0.0f64;
            let mut pick = n - 1;
            for (i, &w) in min_dist.iter().enumerate() {
                acc += w;
                if acc >= target {
                    pick = i;
                    break;
                }
            }
            pick
        };
        centers.push(rows[chosen].clone());
    }

    let mut assign = vec![0usize; n];
    let mut counts = vec![0u64; k];
    for _ in 0..max_iters {
        let mut changed = false;
        counts.fill(0);
        let mut sums = vec![vec![0.0f64; dim]; k];
        for (i, row) in rows.iter().enumerate() {
            let mut best_c = 0usize;
            let mut best_cos = f64::NEG_INFINITY;
            for (c, center) in centers.iter().enumerate() {
                let cos = dot_f64(row, center);
                if cos > best_cos {
                    best_cos = cos;
                    best_c = c;
                }
            }
            if assign[i] != best_c {
                assign[i] = best_c;
                changed = true;
            }
            counts[best_c] += 1;
            for (s, &v) in sums[best_c].iter_mut().zip(row.iter()) {
                *s += v;
            }
        }
        for c in 0..k {
            if counts[c] == 0 {
                let fallback = (next_u64(&mut rng) as usize) % n;
                centers[c].clone_from(&rows[fallback]);
            } else {
                let norm = sums[c].iter().map(|v| v * v).sum::<f64>().sqrt();
                if norm > 1e-12 {
                    for (dst, &s) in centers[c].iter_mut().zip(sums[c].iter()) {
                        *dst = s / norm;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }

    let mut cos_sum = 0.0f64;
    counts.fill(0);
    for (i, row) in rows.iter().enumerate() {
        let c = assign[i];
        counts[c] += 1;
        cos_sum += dot_f64(row, &centers[c]);
    }
    let mean_cosine = cos_sum / (n as f64);

    let labels: Vec<String> = (0..k).map(|c| format!("cell_{c}")).collect();
    let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
    let tag_name = if opts.tag_set_name.trim().is_empty() { "voronoi_shards" } else { opts.tag_set_name.trim() };
    let tags = TagSet::new(tag_name, &label_refs);
    let mut flat = Vec::with_capacity(k * dim);
    for center in &centers {
        for &v in center {
            flat.push(v as f32);
        }
    }
    let centroids = LabelCentroids::from_centroids(model, dim as u32, tags, flat, opts.temperature, CentroidNorm::L2)?;
    Ok(VoronoiCentroids { centroids, cluster_counts: counts, mean_cosine })
}

fn l2_normalize_f64(v: &[f32]) -> Result<Vec<f64>, String> {
    let mut sq = 0.0f64;
    let mut out = Vec::with_capacity(v.len());
    for (i, &x) in v.iter().enumerate() {
        if !x.is_finite() {
            return Err(format!("element {i} is non-finite"));
        }
        let xf = f64::from(x);
        sq += xf * xf;
        out.push(xf);
    }
    let norm = sq.sqrt();
    if norm <= 1e-12 || !norm.is_finite() {
        return Err("zero-norm vector cannot be normalized for spherical Voronoi routing".into());
    }
    for x in &mut out {
        *x /= norm;
    }
    Ok(out)
}

fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn next_u64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn unit_f64(state: &mut u64) -> f64 {
    ((next_u64(state) >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trains_deterministic_voronoi_cells_and_round_trips_tcent001() {
        let mut samples = Vec::new();
        for i in 0..12 {
            let jitter = (i as f32) * 0.01;
            samples.extend_from_slice(&[1.0, jitter, 0.0, 0.0]);
            samples.extend_from_slice(&[0.0, 1.0, jitter, 0.0]);
            samples.extend_from_slice(&[0.0, 0.0, 1.0, jitter]);
        }
        let model = ModelRef { id: "tiny-bert".into(), revision: "r1".into(), sha256: "abc".into() };
        let opts = VoronoiOptions { num_centroids: 3, max_iters: 30, seed: 7, ..VoronoiOptions::default() };
        let a = train_voronoi_centroids(model.clone(), 4, &samples, &opts).unwrap();
        let b = train_voronoi_centroids(model, 4, &samples, &opts).unwrap();
        assert_eq!(a.to_bytes(), b.to_bytes());
        assert!(a.mean_cosine() > 0.99);
        assert_eq!(a.cluster_counts().iter().sum::<u64>(), 36);

        let decoded = VoronoiCentroids::from_bytes(&a.to_bytes()).unwrap();
        let probes = decoded.route_top_probes(&[0.98, 0.02, 0.0, 0.0], 2).unwrap();
        assert_eq!(probes.len(), 2);
        assert!(probes[0].1 > 0.99);
        assert!(probes[1].1 < 0.20);
    }
}
