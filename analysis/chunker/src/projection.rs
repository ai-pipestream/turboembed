//! Deterministic 2D/3D PCA projection of a `Vectors<L>` layer and the
//! boundary drift curve over sentence vectors, for inspecting chunk
//! and sentence geometry without leaving the base library.
//!
//! The projection is computed in `f64` over the non-empty rows only:
//! mean-centering, then the leading eigenvectors of the covariance by
//! Householder tridiagonalization and implicit QL (`eigen.rs`). When there are fewer rows than dimensions
//! (the usual case: tens to hundreds of chunks at 384 to 1024
//! dimensions), the eigendecomposition runs on the `[m, m]` Gram
//! matrix instead and the coordinates come from its eigenvectors
//! scaled by the singular values, which is the same projection up to
//! rounding. Every reduction sums sequentially in stored order and
//! every eigenvector is signed so its largest-magnitude coordinate
//! (the first, on a tie) is positive, so the coordinates are
//! bit-identical for a given input on any machine.
//!
//! Rows listed in `Vectors::empty` take no part in the mean or the
//! covariance, project to zero coordinates, and yield `None` adjacent
//! cosines. With zero or one non-empty rows every coordinate and
//! explained-variance ratio is zero: there is no direction of variance
//! to keep.

use std::marker::PhantomData;

mod eigen;
use eigen::symmetric_eigen;

use turbo_analysis::math::{dot_f64, sum_sq_f64};
use turbo_analysis::{Ix, Layer, ModelRef};
use turbo_analysis_layers::{Cut, Sentences, Vectors};

use crate::drift::{Drift, cosines, score_series};
use crate::sums::{Rows, checked_rows};

/// How many coordinates a projection carries per row.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ProjectionDims {
    D2 = 2,
    D3 = 3,
}

impl ProjectionDims {
    pub fn get(self) -> usize {
        self as usize
    }
}

/// The 2D or 3D PCA coordinates of every row of a `Vectors<L>`.
/// `coords` is row-major `[rows, dims]` in layer row order, zeros for
/// the rows in `empty`. `explained_variance[c]` is the share of the
/// total variance carried by component `c` for the first three
/// components (zero past the spectrum the rows span), independent of
/// `dims`. `adjacent_cosine` has one entry per row: entry `i` is the
/// cosine between the original vectors of rows `i` and `i + 1`, or
/// `None` for the last row, when either row is empty, or when either
/// vector has zero norm.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorProjection<L> {
    pub model: ModelRef,
    pub dims: ProjectionDims,
    pub coords: Vec<f32>,
    pub empty: Vec<Ix<L>>,
    pub explained_variance: [f32; 3],
    pub adjacent_cosine: Vec<Option<f32>>,
    pub _of: PhantomData<fn() -> L>,
}

impl<L: Layer> VectorProjection<L> {
    /// Row `i`'s coordinates.
    pub fn row(&self, i: Ix<L>) -> &[f32] {
        let d = self.dims.get();
        &self.coords[i.get() * d..(i.get() + 1) * d]
    }

    /// Computes the projection, refusing an invalid `vectors` layer
    /// and any non-finite value in a non-empty row. Refuses projected
    /// coordinates that cannot be represented as finite `f32` values.
    pub fn compute(vectors: &Vectors<L>, dims: ProjectionDims) -> Result<VectorProjection<L>, String> {
        vectors.check()?;
        let present: Vec<usize> = (0..vectors.len()).filter(|&i| !vectors.is_empty(Ix::new(i as u32))).collect();
        Self::project(vectors, dims, &present, present.len() < vectors.dim as usize)
    }

    fn project(
        vectors: &Vectors<L>,
        dims: ProjectionDims,
        present: &[usize],
        dual: bool,
    ) -> Result<VectorProjection<L>, String> {
        let (n, d) = (vectors.len(), vectors.dim as usize);
        let k = dims.get();
        let mut out = VectorProjection {
            model: vectors.model.clone(),
            dims,
            coords: vec![0.0; n * k],
            empty: vectors.empty.clone(),
            explained_variance: [0.0; 3],
            adjacent_cosine: adjacent_cosines(vectors),
            _of: PhantomData,
        };
        let mut centred = Vec::with_capacity(present.len() * d);
        for &i in present {
            let row = vectors.row(Ix::new(i as u32));
            if let Some(at) = row.iter().position(|x| !x.is_finite()) {
                return Err(format!("row {i} coordinate {at} is not finite"));
            }
            centred.extend(row.iter().map(|&x| x as f64));
        }
        let m = present.len();
        if m < 2 || d == 0 {
            return Ok(out);
        }
        let mut mean = vec![0f64; d];
        for r in centred.chunks_exact(d) {
            for (mu, v) in mean.iter_mut().zip(r) {
                *mu += v;
            }
        }
        for mu in &mut mean {
            *mu /= m as f64;
        }
        for r in centred.chunks_exact_mut(d) {
            for (v, mu) in r.iter_mut().zip(&mean) {
                *v -= mu;
            }
        }
        let (mut values, vecs, comps) =
            if dual { dual_basis(&centred, m, d) } else { covariance_basis(&centred, m, d) };
        let mut order: Vec<usize> = (0..values.len()).collect();
        order.sort_by(|&a, &b| values[b].total_cmp(&values[a]).then(a.cmp(&b)));
        for v in &mut values {
            *v = v.max(0.0);
        }
        let total: f64 = values.iter().sum();
        if total > 0.0 {
            for (c, slot) in out.explained_variance.iter_mut().enumerate() {
                if let Some(&col) = order.get(c) {
                    *slot = (values[col] / total) as f32;
                }
            }
        }
        // coords[row][c] = centred row . component c, where the dual
        // path's components are Gram eigenvectors and the product is
        // sqrt(lambda) * v[row] instead.
        for (r, &row_ix) in present.iter().enumerate() {
            let centred_row = &centred[r * d..(r + 1) * d];
            for c in 0..k {
                let Some(&col) = order.get(c) else { break };
                let v = if dual {
                    values[col].sqrt() * vecs[r * m + col] * comps[c]
                } else {
                    centred_row.iter().zip(&comps[c * d..(c + 1) * d]).map(|(a, b)| a * b).sum::<f64>()
                };
                let coordinate = v as f32;
                if !coordinate.is_finite() {
                    return Err(format!("row {row_ix} projection coordinate {c} is not finite"));
                }
                out.coords[row_ix * k + c] = coordinate;
            }
        }
        Ok(out)
    }
}

fn adjacent_cosines<L: Layer>(vectors: &Vectors<L>) -> Vec<Option<f32>> {
    let n = vectors.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let v = if i + 1 >= n || vectors.is_empty(Ix::new(i as u32)) || vectors.is_empty(Ix::new(i as u32 + 1)) {
            None
        } else {
            let (a, b) = (vectors.row(Ix::new(i as u32)), vectors.row(Ix::new(i as u32 + 1)));
            let (na, nb) = (sum_sq_f64(a), sum_sq_f64(b));
            if na == 0.0 || nb == 0.0 { None } else { Some((dot_f64(a, b) / (na.sqrt() * nb.sqrt())) as f32) }
        };
        out.push(v);
    }
    out
}

/// The sign that makes a component's largest-magnitude coordinate
/// positive: -1 when that coordinate (the first, on a tie) is negative.
fn canonical_sign(comp: &[f64]) -> f64 {
    let big = comp.iter().copied().fold(0f64, |a, b| if b.abs() > a.abs() { b } else { a });
    if big < 0.0 { -1.0 } else { 1.0 }
}

/// Covariance path: eigenvalues, eigenvectors, and the first three
/// sign-canonicalized components packed by rank then coordinate.
fn covariance_basis(centred: &[f64], m: usize, d: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut cov = vec![0f64; d * d];
    for r in centred.chunks_exact(d) {
        for i in 0..d {
            for j in i..d {
                cov[i * d + j] += r[i] * r[j];
            }
        }
    }
    for i in 0..d {
        for j in i..d {
            let v = cov[i * d + j] / (m as f64 - 1.0);
            cov[i * d + j] = v;
            cov[j * d + i] = v;
        }
    }
    let (values, vecs) = symmetric_eigen(&cov, d);
    let mut order: Vec<usize> = (0..d).collect();
    order.sort_by(|&a, &b| values[b].total_cmp(&values[a]).then(a.cmp(&b)));
    let keep = 3.min(d);
    let mut comps = vec![0f64; keep * d];
    for (c, &col) in order.iter().take(keep).enumerate() {
        let comp: Vec<f64> = (0..d).map(|i| vecs[i * d + col]).collect();
        let s = canonical_sign(&comp);
        for (i, x) in comp.into_iter().enumerate() {
            comps[c * d + i] = s * x;
        }
    }
    (values, vecs, comps)
}

/// Dual Gram path: eigenvalues of X X^T, its eigenvectors (columns),
/// and per-rank canonical signs (from the feature-space component
/// X^T v / sqrt(lambda)) packed by rank.
fn dual_basis(centred: &[f64], m: usize, d: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut gram = vec![0f64; m * m];
    for i in 0..m {
        for j in i..m {
            let v: f64 = centred[i * d..(i + 1) * d].iter().zip(&centred[j * d..(j + 1) * d]).map(|(a, b)| a * b).sum();
            gram[i * m + j] = v;
            gram[j * m + i] = v;
        }
    }
    let (values, vecs) = symmetric_eigen(&gram, m);
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| values[b].total_cmp(&values[a]).then(a.cmp(&b)));
    let mut signs = vec![1.0f64; 3];
    for (c, &col) in order.iter().take(3).enumerate() {
        let lam = values[col].max(0.0);
        if lam > 0.0 {
            // u = X^T v / sqrt(lambda): only its sign pattern matters,
            // and the positive scale does not change it.
            let mut u = vec![0f64; d];
            for r in 0..m {
                let w = vecs[r * m + col];
                for (ui, x) in u.iter_mut().zip(&centred[r * d..(r + 1) * d]) {
                    *ui += w * x;
                }
            }
            signs[c] = canonical_sign(&u);
        }
    }
    (values, vecs, signs)
}

/// The drift computation of [`crate::drift`] with every intermediate
/// series kept, per sentence gap (one entry per sentence, the last for
/// the gap after it), for inspection and tuning tools.
///
/// `raw_cosine_micro` is the windowed cosine in millionths (`None`
/// where either side has fewer than `window` sentences with a vector),
/// `smoothed_cosine_micro` its moving average over the scored gaps,
/// and `depth_score_micro` the score the cost is taken from before the
/// floor: the TextTiling depth under [`crate::Score::Depth`], the
/// distance under [`crate::Score::Distance`], and 0 at unscored gaps.
/// `cost` and `reason` are exactly what [`crate::drift`] returns for
/// the same inputs, bit for bit, because both come from the one
/// scoring routine.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundaryDriftCurve {
    pub raw_cosine_micro: Vec<Option<i64>>,
    pub smoothed_cosine_micro: Vec<Option<i64>>,
    pub depth_score_micro: Vec<i64>,
    pub cost: Vec<i32>,
    pub reason: Vec<Cut>,
}

impl BoundaryDriftCurve {
    /// Computes the curve for `vectors` under `drift`, the same inputs
    /// [`crate::DriftBoundaries`] scores. Retains [`Rows::of`]'s permissive
    /// input handling; use [`Self::try_compute`] to validate the layer.
    pub fn compute(vectors: &Vectors<Sentences>, drift: &Drift) -> BoundaryDriftCurve {
        Self::from_rows(&Rows::of(vectors), drift)
    }

    /// Validates vector shape and empty-row metadata, then refuses any
    /// non-finite coordinate before computing the same drift series.
    /// Finite zero-norm and explicitly empty rows keep their gap slots.
    pub fn try_compute(vectors: &Vectors<Sentences>, drift: &Drift) -> Result<BoundaryDriftCurve, String> {
        Ok(Self::from_rows(&checked_rows(vectors)?, drift))
    }

    fn from_rows(rows: &Rows<'_>, drift: &Drift) -> BoundaryDriftCurve {
        let raw = cosines(rows, drift.window);
        let (smoothed_cosine_micro, depth_score_micro, cost, reason) = score_series(&raw, drift);
        BoundaryDriftCurve { raw_cosine_micro: raw, smoothed_cosine_micro, depth_score_micro, cost, reason }
    }

    /// The gaps reasoned `Cut::TopicShift`: the selected valley
    /// bottoms, in gap order.
    pub fn topic_shift_gaps(&self) -> impl Iterator<Item = usize> + '_ {
        self.reason.iter().enumerate().filter_map(|(g, &r)| (r == Cut::TopicShift).then_some(g))
    }

    /// The gap of least cost, the earliest of equals, or `None` when
    /// there are no gaps.
    pub fn cheapest_gap(&self) -> Option<usize> {
        (0..self.cost.len()).min_by_key(|&g| (self.cost[g], g))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_analysis_layers::Chunks;

    fn vectors(dim: u32, rows: &[&[f32]], empty: &[u32]) -> Vectors<Chunks> {
        let mut data = Vec::new();
        for r in rows {
            data.extend_from_slice(r);
        }
        Vectors::new(
            dim,
            data,
            empty.iter().map(|&i| Ix::new(i)).collect(),
            ModelRef { id: "t".into(), revision: "0".into(), sha256: "0".into() },
        )
    }

    #[test]
    fn dual_and_covariance_paths_agree() {
        // 4 rows in 6 dims: dual is the production path; force both.
        let rows: Vec<Vec<f32>> = (0..4)
            .map(|i| (0..6).map(|j| ((i * 7 + j * 3) as f32 * 0.37).sin() + (i as f32) * 0.11 * j as f32).collect())
            .collect();
        let v = vectors(6, &rows.iter().map(|r| &r[..]).collect::<Vec<_>>(), &[]);
        let present: Vec<usize> = (0..4).collect();
        let dual = VectorProjection::<Chunks>::project(&v, ProjectionDims::D3, &present, true).unwrap();
        let cov = VectorProjection::<Chunks>::project(&v, ProjectionDims::D3, &present, false).unwrap();
        for (a, b) in dual.coords.iter().zip(&cov.coords) {
            assert!((a - b).abs() < 1e-5, "dual {a} vs covariance {b}");
        }
        for c in 0..3 {
            assert!((dual.explained_variance[c] - cov.explained_variance[c]).abs() < 1e-5);
        }
        let kept: f32 = dual.explained_variance.iter().sum();
        assert!(kept > 0.9 && kept <= 1.0 + 1e-6, "3 components of 4 points keep nearly all variance: {kept}");
    }
}
