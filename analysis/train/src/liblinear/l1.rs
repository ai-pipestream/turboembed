//! `L1R_LR`: L1-regularized logistic regression by liblinear's newGLMNET
//! (`Linear.solve_l1r_lr`; Yuan, Ho and Lin, 2012): an outer Newton
//! iteration whose quadratic model is minimized by coordinate descent over
//! the features, with shrinking at both levels and a line search.
//!
//! The coordinate descent shuffles the active features with the
//! `java.util.Random` of the training's `Parameter`. liblinear-java draws
//! the swap partner of position `j` from `0..active - j`, where the C code
//! draws it from `j..active`; the Java draw is kept, as it decides the
//! order of the updates and so the weights.

use turbo_jvm::random::JavaRandom;

use super::problem::{Rows, axpy, java_max, java_min};
use super::{exp, log};

const MAX_NEWTON_ITER: usize = 100;
const MAX_NUM_LINESEARCH: usize = 20;
const NU: f64 = 1e-12;
const SIGMA: f64 = 0.01;

/// The problem by feature: `columns.row(j)` lists the instances that have
/// feature `j`, with its values. `y` is `+1` or `-1` per instance, `cp`
/// and `cn` the costs of the positive and negative instances. Starts from
/// zero and leaves the solution in `w`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve(
    columns: &Rows,
    y: &[f64],
    w: &mut [f64],
    cp: f64,
    cn: f64,
    eps: f64,
    max_iter: usize,
    random: &mut JavaRandom,
) {
    let l = y.len();
    let w_size = columns.len();
    let lf = l as f64;
    let positive: Vec<bool> = y.iter().map(|&v| v > 0.0).collect();
    let cost = |i: usize| if positive[i] { cp } else { cn };

    let mut index: Vec<u32> = (0..w_size as u32).collect();
    let mut hdiag = vec![0.0; w_size];
    let mut grad = vec![0.0; w_size];
    let mut wpd = vec![0.0; w_size];
    let mut xjneg_sum = vec![0.0; w_size];
    let mut x_td = vec![0.0; l];
    let mut exp_wtx = vec![0.0; l];
    let mut exp_wtx_new = vec![0.0; l];
    let mut tau = vec![0.0; l];
    let mut d = vec![0.0; l];

    w.fill(0.0);
    let mut w_norm = 0.0;
    for j in 0..w_size {
        w_norm += w[j].abs();
        wpd[j] = w[j];
        let (rows, values) = columns.row(j);
        for (&ind, &val) in rows.iter().zip(values) {
            let ind = ind as usize;
            exp_wtx[ind] += w[j] * val;
            if !positive[ind] {
                xjneg_sum[j] += cost(ind) * val;
            }
        }
    }
    for i in 0..l {
        exp_wtx[i] = exp(exp_wtx[i]);
        let tau_tmp = 1.0 / (1.0 + exp_wtx[i]);
        tau[i] = cost(i) * tau_tmp;
        d[i] = cost(i) * exp_wtx[i] * tau_tmp * tau_tmp;
    }

    let mut gnorm1_init = -1.0;
    let mut gmax_old = f64::INFINITY;
    let mut inner_eps = 1.0;
    let mut newton_iter = 0;
    while newton_iter < MAX_NEWTON_ITER {
        let mut gmax_new = 0.0f64;
        let mut gnorm1_new = 0.0;
        let mut active_size = w_size;

        let mut s = 0;
        while s < active_size {
            let j = index[s] as usize;
            hdiag[j] = NU;
            let mut tmp = 0.0;
            let (rows, values) = columns.row(j);
            for (&ind, &val) in rows.iter().zip(values) {
                let ind = ind as usize;
                hdiag[j] += val * val * d[ind];
                tmp += val * tau[ind];
            }
            grad[j] = -tmp + xjneg_sum[j];

            let gp = grad[j] + 1.0;
            let gn = grad[j] - 1.0;
            let mut violation = 0.0;
            if w[j] == 0.0 {
                if gp < 0.0 {
                    violation = -gp;
                } else if gn > 0.0 {
                    violation = gn;
                } else if gp > gmax_old / lf && gn < -gmax_old / lf {
                    // Outer level shrinking.
                    active_size -= 1;
                    index.swap(s, active_size);
                    continue;
                }
            } else if w[j] > 0.0 {
                violation = gp.abs();
            } else {
                violation = gn.abs();
            }
            gmax_new = java_max(gmax_new, violation);
            gnorm1_new += violation;
            s += 1;
        }

        if newton_iter == 0 {
            gnorm1_init = gnorm1_new;
        }
        if gnorm1_new <= eps * gnorm1_init {
            break;
        }

        let mut iter = 0;
        let mut qp_gmax_old = f64::INFINITY;
        let mut qp_active_size = active_size;
        x_td.fill(0.0);

        // Minimizes the quadratic model over wpd.
        while iter < max_iter {
            let mut qp_gmax_new = 0.0f64;
            let mut qp_gnorm1_new = 0.0;
            for j in 0..qp_active_size {
                let i = random.next_int_bounded((qp_active_size - j) as i32) as usize;
                index.swap(i, j);
            }

            let mut s = 0;
            while s < qp_active_size {
                let j = index[s] as usize;
                let h = hdiag[j];
                let mut g = grad[j] + (wpd[j] - w[j]) * NU;
                let (rows, values) = columns.row(j);
                for (&ind, &val) in rows.iter().zip(values) {
                    let ind = ind as usize;
                    g += val * d[ind] * x_td[ind];
                }

                let gp = g + 1.0;
                let gn = g - 1.0;
                let mut violation = 0.0;
                if wpd[j] == 0.0 {
                    if gp < 0.0 {
                        violation = -gp;
                    } else if gn > 0.0 {
                        violation = gn;
                    } else if gp > qp_gmax_old / lf && gn < -qp_gmax_old / lf {
                        // Inner level shrinking.
                        qp_active_size -= 1;
                        index.swap(s, qp_active_size);
                        continue;
                    }
                } else if wpd[j] > 0.0 {
                    violation = gp.abs();
                } else {
                    violation = gn.abs();
                }

                // The solution of the one-variable problem.
                let mut z = if gp < h * wpd[j] {
                    -gp / h
                } else if gn > h * wpd[j] {
                    -gn / h
                } else {
                    -wpd[j]
                };
                qp_gmax_new = java_max(qp_gmax_new, violation);
                qp_gnorm1_new += violation;

                s += 1;
                if z.abs() < 1.0e-12 {
                    continue;
                }
                z = java_min(java_max(z, -10.0), 10.0);
                wpd[j] += z;
                axpy(z, (rows, values), &mut x_td);
            }

            iter += 1;

            if qp_gnorm1_new <= inner_eps * gnorm1_init {
                // Inner stopping.
                if qp_active_size == active_size {
                    break;
                }
                // Active set reactivation.
                qp_active_size = active_size;
                qp_gmax_old = f64::INFINITY;
                continue;
            }
            qp_gmax_old = qp_gmax_new;
        }

        let mut delta = 0.0;
        let mut w_norm_new = 0.0;
        for j in 0..w_size {
            delta += grad[j] * (wpd[j] - w[j]);
            if wpd[j] != 0.0 {
                w_norm_new += wpd[j].abs();
            }
        }
        delta += w_norm_new - w_norm;

        let mut negsum_xtd = 0.0;
        for i in 0..l {
            if !positive[i] {
                negsum_xtd += cost(i) * x_td[i];
            }
        }

        let mut num_linesearch = 0;
        while num_linesearch < MAX_NUM_LINESEARCH {
            let mut cond = w_norm_new - w_norm + negsum_xtd - SIGMA * delta;
            for i in 0..l {
                let exp_xtd = exp(x_td[i]);
                exp_wtx_new[i] = exp_wtx[i] * exp_xtd;
                cond += cost(i) * log((1.0 + exp_wtx_new[i]) / (exp_xtd + exp_wtx_new[i]));
            }

            if cond <= 0.0 {
                w_norm = w_norm_new;
                w.copy_from_slice(&wpd);
                for i in 0..l {
                    exp_wtx[i] = exp_wtx_new[i];
                    let tau_tmp = 1.0 / (1.0 + exp_wtx[i]);
                    tau[i] = cost(i) * tau_tmp;
                    d[i] = cost(i) * exp_wtx[i] * tau_tmp * tau_tmp;
                }
                break;
            }
            w_norm_new = 0.0;
            for j in 0..w_size {
                wpd[j] = (w[j] + wpd[j]) * 0.5;
                if wpd[j] != 0.0 {
                    w_norm_new += wpd[j].abs();
                }
            }
            delta *= 0.5;
            negsum_xtd *= 0.5;
            for v in &mut x_td {
                *v *= 0.5;
            }
            num_linesearch += 1;
        }

        // Too many line search steps: recomputes exp(w'x) from w.
        if num_linesearch >= MAX_NUM_LINESEARCH {
            exp_wtx.fill(0.0);
            for (j, &wj) in w[..w_size].iter().enumerate() {
                if wj == 0.0 {
                    continue;
                }
                axpy(wj, columns.row(j), &mut exp_wtx);
            }
            for v in &mut exp_wtx {
                *v = exp(*v);
            }
        }

        if iter == 1 {
            inner_eps *= 0.25;
        }
        newton_iter += 1;
        gmax_old = gmax_new;
    }
}
