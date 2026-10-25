//! Deterministic 2D/3D PCA coordinate projection over `Vectors<L>` with
//! sign-canonicalized eigenvectors and adjacent non-empty cosine calculation
//! (`ProjectChunks`, issue #1220).

use turbo_analysis::{Ix, Layer, Span};
use turbo_analysis_layers::Vectors;

use crate::offsets::Offsetter;
use crate::proto::result::{ProjectedPointProto, VectorProjectionProto};
use crate::results::model_to_proto;

const JACOBI_MAX_SWEEPS: usize = 64;
const JACOBI_TOL: f64 = 1e-12;

/// Projects `vectors` (aligned with `spans` and `levels`) into `pca_dims` (2
/// or 3) principal components, returning [`VectorProjectionProto`].
pub(crate) fn compute_pca<L: Layer>(
    vectors: &Vectors<L>,
    spans: &[Span],
    levels: &[u8],
    text: &str,
    offsetter: &Offsetter,
    pca_dims: u32,
) -> VectorProjectionProto {
    let n = spans.len();
    let d = vectors.dim as usize;
    let k = pca_dims as usize;
    let model = Some(model_to_proto(&vectors.model));
    if n == 0 || d == 0 {
        return VectorProjectionProto { model, pca_dims, explained_variance: vec![0.0; k], points: Vec::new() };
    }

    let present: Vec<usize> = (0..n).filter(|&i| !vectors.is_empty(Ix::new(i as u32))).collect();
    let m = present.len();
    let mut coords = vec![[0.0_f32; 3]; n];
    let mut explained = vec![0.0_f32; k];

    if m >= 2 {
        // Mean-center present rows in f64.
        let mut mean = vec![0.0_f64; d];
        for &row_idx in &present {
            let row = vectors.row(Ix::new(row_idx as u32));
            for (mu, &val) in mean.iter_mut().zip(row) {
                *mu += f64::from(val);
            }
        }
        let inv_m = 1.0 / (m as f64);
        for mu in &mut mean {
            *mu *= inv_m;
        }

        let mut centered = vec![0.0_f64; m * d];
        for (p_idx, &row_idx) in present.iter().enumerate() {
            let row = vectors.row(Ix::new(row_idx as u32));
            let dst = &mut centered[p_idx * d..(p_idx + 1) * d];
            for (c, (&val, &mu)) in dst.iter_mut().zip(row.iter().zip(&mean)) {
                *c = f64::from(val) - mu;
            }
        }

        // Use dual [m, m] Gram matrix X_c * X_c^T when m <= d, or direct [d, d]
        // covariance X_c^T * X_c when m > d.
        let (eigvals, axes) = if m <= d { gram_pca(&centered, m, d, k) } else { cov_pca(&centered, m, d, k) };

        let total_var: f64 = eigvals.iter().copied().filter(|&v| v > 0.0).sum();
        for c_idx in 0..k.min(axes.len()) {
            let ev = eigvals[c_idx].max(0.0);
            explained[c_idx] = if total_var > 0.0 { (ev / total_var) as f32 } else { 0.0 };
            let axis = &axes[c_idx];
            for (p_idx, &row_idx) in present.iter().enumerate() {
                let row_c = &centered[p_idx * d..(p_idx + 1) * d];
                let proj: f64 = row_c.iter().zip(axis).map(|(&x, &a)| x * a).sum();
                coords[row_idx][c_idx] = proj as f32;
            }
        }
    }

    let adjacent = compute_adjacent_cosines(vectors, n);
    let mut points = Vec::with_capacity(n);
    for i in 0..n {
        let (start, end) = offsetter.to_wire(text, spans[i]);
        let empty = vectors.is_empty(Ix::new(i as u32));
        points.push(ProjectedPointProto {
            item: i as u32,
            start,
            end,
            level: u32::from(levels.get(i).copied().unwrap_or(0)),
            x: coords[i][0],
            y: coords[i][1],
            z: if pca_dims == 3 { coords[i][2] } else { 0.0 },
            empty,
            adjacent_cosine: adjacent[i],
        });
    }

    VectorProjectionProto { model, pca_dims, explained_variance: explained, points }
}

fn gram_pca(centered: &[f64], m: usize, d: usize, k: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut gram = vec![0.0_f64; m * m];
    for i in 0..m {
        let ri = &centered[i * d..(i + 1) * d];
        for j in i..m {
            let rj = &centered[j * d..(j + 1) * d];
            let dot: f64 = ri.iter().zip(rj).map(|(&a, &b)| a * b).sum();
            gram[i * m + j] = dot;
            gram[j * m + i] = dot;
        }
    }
    let (eigvals, eigvecs) = jacobi_eigen(&mut gram, m);
    let mut axes = Vec::with_capacity(k);
    for c in 0..k.min(m) {
        let lambda = eigvals[c];
        let mut axis = vec![0.0_f64; d];
        if lambda > JACOBI_TOL {
            let u = &eigvecs[c];
            for (p_idx, &u_val) in u.iter().enumerate() {
                let row_c = &centered[p_idx * d..(p_idx + 1) * d];
                for (ax, &x) in axis.iter_mut().zip(row_c) {
                    *ax += u_val * x;
                }
            }
            normalize_and_canonicalize(&mut axis);
        }
        axes.push(axis);
    }
    (eigvals, axes)
}

fn cov_pca(centered: &[f64], m: usize, d: usize, k: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut cov = vec![0.0_f64; d * d];
    for p_idx in 0..m {
        let row = &centered[p_idx * d..(p_idx + 1) * d];
        for i in 0..d {
            let xi = row[i];
            for j in i..d {
                cov[i * d + j] += xi * row[j];
            }
        }
    }
    for i in 0..d {
        for j in (i + 1)..d {
            cov[j * d + i] = cov[i * d + j];
        }
    }
    let (eigvals, mut eigvecs) = jacobi_eigen(&mut cov, d);
    eigvecs.truncate(k.min(d));
    for axis in &mut eigvecs {
        normalize_and_canonicalize(axis);
    }
    (eigvals, eigvecs)
}

fn normalize_and_canonicalize(axis: &mut [f64]) {
    let norm = axis.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in axis.iter_mut() {
            *x /= norm;
        }
    }
    // Sign-canonicalize: make the largest-magnitude coordinate positive (break
    // ties by lowest coordinate index).
    let mut best_idx = 0usize;
    let mut best_abs = -1.0_f64;
    for (idx, &x) in axis.iter().enumerate() {
        let a = x.abs();
        if a > best_abs + 1e-12 {
            best_abs = a;
            best_idx = idx;
        }
    }
    if axis.get(best_idx).is_some_and(|&v| v < 0.0) {
        for x in axis.iter_mut() {
            *x = -*x;
        }
    }
}

/// Cyclic Jacobi eigendecomposition of symmetric `n x n` matrix `a`, returning
/// `(eigenvalues_desc, eigenvectors_desc)`.
fn jacobi_eigen(a: &mut [f64], n: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut v = vec![0.0_f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _sweep in 0..JACOBI_MAX_SWEEPS {
        let mut max_off = 0.0_f64;
        for p in 0..n {
            for q in (p + 1)..n {
                max_off = max_off.max(a[p * n + q].abs());
            }
        }
        if max_off <= JACOBI_TOL {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() <= JACOBI_TOL {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                a[p * n + p] = app - t * apq;
                a[q * n + q] = aqq + t * apq;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;
                for r in 0..n {
                    if r != p && r != q {
                        let arp = a[r * n + p];
                        let arq = a[r * n + q];
                        a[r * n + p] = c * arp - s * arq;
                        a[p * n + r] = a[r * n + p];
                        a[r * n + q] = s * arp + c * arq;
                        a[q * n + r] = a[r * n + q];
                    }
                    let vrp = v[r * n + p];
                    let vrq = v[r * n + q];
                    v[r * n + p] = c * vrp - s * vrq;
                    v[r * n + q] = s * vrp + c * vrq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| a[j * n + j].total_cmp(&a[i * n + i]).then_with(|| i.cmp(&j)));
    let eigvals = order.iter().map(|&col| a[col * n + col]).collect();
    let eigvecs = order.iter().map(|&col| (0..n).map(|row| v[row * n + col]).collect()).collect();
    (eigvals, eigvecs)
}

fn compute_adjacent_cosines<L: Layer>(vectors: &Vectors<L>, n: usize) -> Vec<Option<f32>> {
    let mut out = vec![None; n];
    for (i, slot) in out.iter_mut().enumerate().take(n.saturating_sub(1)) {
        let ix_a = Ix::new(i as u32);
        let ix_b = Ix::new((i + 1) as u32);
        if vectors.is_empty(ix_a) || vectors.is_empty(ix_b) {
            continue;
        }
        let ra = vectors.row(ix_a);
        let rb = vectors.row(ix_b);
        let (mut dot, mut na, mut nb) = (0.0_f64, 0.0_f64, 0.0_f64);
        for (&a, &b) in ra.iter().zip(rb) {
            let (af, bf) = (f64::from(a), f64::from(b));
            dot += af * bf;
            na += af * af;
            nb += bf * bf;
        }
        if na > 0.0 && nb > 0.0 {
            *slot = Some((dot / (na.sqrt() * nb.sqrt())).clamp(-1.0, 1.0) as f32);
        }
    }
    out
}
