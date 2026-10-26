//! Closed-form ridge distillation of the SPLATE adapter (#913): the
//! linear map from ColBERT token vectors to the SPLADE teacher's
//! per-token vocabulary activations, fit by `trained.rs`.

use super::posting::SplateAdapter;
use super::trained::TrainedConfig;
use crate::Result;

/// One fit token: its ColBERT vector and its teacher target terms
/// (`relu(logit)` weights over the BERT vocabulary).
pub(crate) struct FitToken {
    pub x: Vec<f32>,
    pub y: Vec<(u32, f32)>,
}

/// Closed-form ridge fit of the SPLATE adapter on the fit tokens:
/// `W, b = argmin sum_t ||W x_t + b - y_t||^2 + lambda ||W||^2` over
/// the `adapter_terms` heaviest teacher terms; every other vocabulary
/// row stays at zero weight with bias -30, so it never fires. Returns
/// the adapter, the covered term count, and the fit MSE per target.
pub(crate) fn fit_adapter(
    fit: &[FitToken],
    vocab: usize,
    dim: usize,
    cfg: &TrainedConfig,
) -> Result<(SplateAdapter, usize, f64)> {
    if fit.is_empty() {
        return Err("no aligned fit tokens for the SPLATE adapter".into());
    }
    let mut mass = vec![0.0f64; vocab];
    for t in fit {
        for &(term, w) in &t.y {
            mass[term as usize] += f64::from(w);
        }
    }
    let mut terms: Vec<u32> = (0..vocab as u32).filter(|&t| mass[t as usize] > 0.0).collect();
    terms.sort_by(|&a, &b| mass[b as usize].total_cmp(&mass[a as usize]).then(a.cmp(&b)));
    terms.truncate(cfg.adapter_terms);
    let k = terms.len();
    let mut col = vec![usize::MAX; vocab];
    for (c, &t) in terms.iter().enumerate() {
        col[t as usize] = c;
    }
    let p = dim + 1;
    let mut g = vec![0.0f64; p * p];
    let mut c = vec![0.0f64; p * k];
    let mut aug = vec![0.0f64; p];
    for t in fit {
        for (i, &x) in t.x.iter().enumerate() {
            aug[i] = f64::from(x);
        }
        aug[dim] = 1.0;
        for i in 0..p {
            for j in i..p {
                g[i * p + j] += aug[i] * aug[j];
            }
        }
        for &(term, w) in &t.y {
            let cc = col[term as usize];
            if cc != usize::MAX {
                for i in 0..p {
                    c[i * k + cc] += aug[i] * f64::from(w);
                }
            }
        }
    }
    for i in 0..p {
        for j in 0..i {
            g[i * p + j] = g[j * p + i];
        }
        g[i * p + i] += if i == dim { 0.0 } else { cfg.ridge_lambda };
    }
    let w = cholesky_solve(&g, &c, p, k)?;
    let mut weights = vec![0.0f32; vocab * dim];
    let mut bias = vec![-30.0f32; vocab];
    for (cc, &term) in terms.iter().enumerate() {
        let row = term as usize * dim;
        for i in 0..dim {
            weights[row + i] = w[i * k + cc] as f32;
        }
        bias[term as usize] = w[dim * k + cc] as f32;
    }
    let mut sse = 0.0f64;
    let mut n_targets = 0usize;
    for t in fit {
        let mut pred = vec![0.0f64; k];
        for cc in 0..k {
            let mut s = w[dim * k + cc];
            for i in 0..dim {
                s += w[i * k + cc] * f64::from(t.x[i]);
            }
            pred[cc] = s;
        }
        let mut seen = vec![false; k];
        for &(term, target) in &t.y {
            let cc = col[term as usize];
            if cc != usize::MAX {
                seen[cc] = true;
                sse += (pred[cc] - f64::from(target)).powi(2);
                n_targets += 1;
            }
        }
        for (cc, &s) in seen.iter().enumerate() {
            if !s {
                sse += pred[cc] * pred[cc];
                n_targets += 1;
            }
        }
    }
    let mse = if n_targets > 0 { sse / n_targets as f64 } else { 0.0 };
    Ok((SplateAdapter::new(weights, bias, vocab, dim)?, k, mse))
}

/// Solves `G W = C` for symmetric positive definite `G` (`p` by `p`)
/// and `k` right-hand sides, by Cholesky factorization.
fn cholesky_solve(g: &[f64], c: &[f64], p: usize, k: usize) -> Result<Vec<f64>> {
    let mut l = vec![0.0f64; p * p];
    for i in 0..p {
        for j in 0..=i {
            let mut s = g[i * p + j];
            for m in 0..j {
                s -= l[i * p + m] * l[j * p + m];
            }
            if i == j {
                if s <= 0.0 {
                    return Err(format!("adapter fit Gram matrix is not positive definite at {i}"));
                }
                l[i * p + i] = s.sqrt();
            } else {
                l[i * p + j] = s / l[j * p + j];
            }
        }
    }
    let mut y = c.to_vec();
    for rhs in 0..k {
        for i in 0..p {
            let mut s = y[i * k + rhs];
            for m in 0..i {
                s -= l[i * p + m] * y[m * k + rhs];
            }
            y[i * k + rhs] = s / l[i * p + i];
        }
        for i in (0..p).rev() {
            let mut s = y[i * k + rhs];
            for m in (i + 1)..p {
                s -= l[m * p + i] * y[m * k + rhs];
            }
            y[i * k + rhs] = s / l[i * p + i];
        }
    }
    Ok(y)
}

#[cfg(test)]
mod tests {
    use turbo::sparse::SparseOptions as SparseCut;

    use super::*;

    #[test]
    fn ridge_fit_recovers_a_known_linear_map() {
        // Teacher: term 5 fires as 2*x0 + 1, term 9 as relu-free x1 - 0.5
        // (kept positive on the samples), everything else silent.
        let dim = 4;
        let vocab = 16;
        let mut fit = Vec::new();
        for i in 0..64 {
            let x = vec![(i % 7) as f32 * 0.25, (i % 5) as f32 * 0.3 + 0.8, 0.1, -0.2];
            fit.push(FitToken { y: vec![(5, 2.0 * x[0] + 1.0), (9, x[1] - 0.5)], x });
        }
        let cfg = TrainedConfig { adapter_terms: 8, ridge_lambda: 1e-6, ..TrainedConfig::default() };
        let (adapter, terms, mse) = fit_adapter(&fit, vocab, dim, &cfg).expect("fit");
        assert_eq!(terms, 2);
        assert!(mse < 1e-6, "mse {mse}");
        let cut = SparseCut { top_k: 4, floor: 0.0 };
        let out = adapter.encode_tokens(&[0.5, 1.1, 0.1, -0.2], &cut).expect("encode");
        let ids: Vec<u32> = out.iter().map(|t| t.term).collect();
        assert_eq!(ids, vec![5, 9], "{out:?}");
    }

    #[test]
    fn cholesky_solve_solves_a_small_system() {
        // G = [[4, 1], [1, 3]], C = [1, 2] -> W = [1/11, 7/11].
        let w = cholesky_solve(&[4.0, 1.0, 1.0, 3.0], &[1.0, 2.0], 2, 1).expect("solve");
        assert!((w[0] - 1.0 / 11.0).abs() < 1e-12, "{w:?}");
        assert!((w[1] - 7.0 / 11.0).abs() < 1e-12, "{w:?}");
    }
}
