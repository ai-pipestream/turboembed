//! `L2R_LR_DUAL`: liblinear's coordinate descent on the dual of
//! L2-regularized logistic regression (`Linear.solve_l2r_lr_dual`,
//! algorithm 5 of Yu, Huang and Lin, 2011), with a Newton method on each
//! one-variable sub-problem.
//!
//! The instances are visited in an order shuffled every outer iteration
//! with the `java.util.Random` of the training's `Parameter`: each
//! `Parameter` has its own, seeded with zero, and the weight vectors of a
//! multi-class model draw from it in turn.

use turbo_jvm::random::JavaRandom;

use super::log;
use super::problem::{Rows, axpy, dot, java_max, java_min, nrm2_sq};

/// Runs at most `max_iter` outer iterations and returns how many ran;
/// `w` holds the primal solution on return. `y` is `+1` or `-1` per
/// instance, `cp` and `cn` the upper bounds of the positive and negative
/// instances' dual variables.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve(
    x: &Rows,
    y: &[f64],
    w: &mut [f64],
    cp: f64,
    cn: f64,
    eps: f64,
    max_iter: usize,
    random: &mut JavaRandom,
) -> usize {
    const MAX_INNER_ITER: usize = 100;
    const ETA: f64 = 0.1;
    let l = x.len();
    let mut x_tx = vec![0.0; l];
    let mut index: Vec<u32> = (0..l as u32).collect();
    // alpha[2i] is the variable, alpha[2i + 1] its distance to the bound.
    let mut alpha = vec![0.0; 2 * l];
    let yi: Vec<i8> = y.iter().map(|&v| if v > 0.0 { 1 } else { -1 }).collect();
    let upper_bound = |i: usize| if yi[i] > 0 { cp } else { cn };
    let mut innereps = 1e-2;
    let innereps_min = java_min(1e-8, eps);

    for i in 0..l {
        alpha[2 * i] = java_min(0.001 * upper_bound(i), 1e-8);
        alpha[2 * i + 1] = upper_bound(i) - alpha[2 * i];
    }
    w.fill(0.0);
    for i in 0..l {
        let row = x.row(i);
        x_tx[i] = nrm2_sq(row.1);
        axpy(f64::from(yi[i]) * alpha[2 * i], row, w);
    }

    let mut iter = 0;
    while iter < max_iter {
        for i in 0..l {
            let j = i + random.next_int_bounded((l - i) as i32) as usize;
            index.swap(i, j);
        }
        let mut newton_iter = 0usize;
        let mut gmax = 0.0f64;
        for &i in &index {
            let i = i as usize;
            let y_i = f64::from(yi[i]);
            let c = upper_bound(i);
            let row = x.row(i);
            let a = x_tx[i];
            let b = y_i * dot(w, row);

            // Minimizes g_1(z) or g_2(z), whichever has its minimum inside.
            let (mut ind1, mut ind2, mut sign) = (2 * i, 2 * i + 1, 1.0);
            if 0.5 * a * (alpha[ind2] - alpha[ind1]) + b < 0.0 {
                ind1 = 2 * i + 1;
                ind2 = 2 * i;
                sign = -1.0;
            }

            // g_t(z) = z log(z) + (C - z) log(C - z) + a (z - alpha_old)^2 / 2 + sign b (z - alpha_old)
            let alpha_old = alpha[ind1];
            let mut z = alpha_old;
            if c - z < 0.5 * c {
                z *= 0.1;
            }
            let mut gp = a * (z - alpha_old) + sign * b + log(z / (c - z));
            gmax = java_max(gmax, gp.abs());

            // Newton's method on the sub-problem.
            let mut inner_iter = 0;
            while inner_iter <= MAX_INNER_ITER {
                if gp.abs() < innereps {
                    break;
                }
                let gpp = a + c / (c - z) / z;
                let tmpz = z - gp / gpp;
                if tmpz <= 0.0 {
                    z *= ETA;
                } else {
                    // tmpz is in (0, C).
                    z = tmpz;
                }
                gp = a * (z - alpha_old) + sign * b + log(z / (c - z));
                newton_iter += 1;
                inner_iter += 1;
            }

            if inner_iter > 0 {
                alpha[ind1] = z;
                alpha[ind2] = c - z;
                axpy(sign * (z - alpha_old) * y_i, row, w);
            }
        }

        iter += 1;
        if gmax < eps {
            break;
        }
        if newton_iter <= l / 10 {
            innereps = java_max(innereps_min, 0.1 * innereps);
        }
    }
    iter
}
