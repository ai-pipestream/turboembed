//! `Linear.train` and `train_one` for the logistic regression solvers,
//! and the trained model as liblinear-java's `Model` holds it.

use turbo_jvm::random::JavaRandom;

use super::problem::{Rows, group_classes};
use super::{SolverType, dual, l1, newton};

/// `Parameter.max_iters`, the default the add-on keeps: the Newton
/// iterations of the primal solvers and the outer iterations of `L1R_LR`.
const MAX_ITERS: usize = 1000;
/// `train_one`'s `dual_solver_max_iter`: the dual solver's outer
/// iterations before it hands over to the primal Newton method.
const DUAL_MAX_ITER: usize = 300;

/// A trained linear model, liblinear-java's `Model`: the solver, the class
/// labels in liblinear's order, the feature count, the bias and the
/// weights.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearModel {
    pub(crate) solver: SolverType,
    pub(crate) label: Vec<i32>,
    pub(crate) nr_feature: usize,
    pub(crate) bias: f64,
    /// `w[i * nr_w + j]`: feature `i` (zero based, the bias feature last
    /// when there is one) for weight vector `j`.
    pub(crate) w: Vec<f64>,
}

impl LinearModel {
    pub fn solver(&self) -> SolverType {
        self.solver
    }

    /// `nr_class`: the classes the training data had.
    pub fn nr_class(&self) -> usize {
        self.label.len()
    }

    /// The class labels (the outcome ids) in the order liblinear numbered
    /// them, by first occurrence in the training data. Probabilities come
    /// out in this order.
    pub fn labels(&self) -> &[i32] {
        &self.label
    }

    /// `nr_feature`: the features without the bias feature.
    pub fn nr_feature(&self) -> usize {
        self.nr_feature
    }

    /// The bias feature's value; below zero when there is no bias feature.
    pub fn bias(&self) -> f64 {
        self.bias
    }

    /// The weight vectors per feature: one for a two class model, one per
    /// class otherwise.
    pub fn nr_w(&self) -> usize {
        if self.label.len() == 2 && self.solver != SolverType::McsvmCs { 1 } else { self.label.len() }
    }

    /// The features with a weight: the bias feature counts when there is
    /// one.
    pub fn w_size(&self) -> usize {
        self.nr_feature + usize::from(self.bias >= 0.0)
    }

    /// The weights, `w_size() * nr_w()` of them, feature by feature.
    pub fn weights(&self) -> &[f64] {
        &self.w
    }
}

/// `Linear.train` for classification by a logistic regression solver.
/// `x` holds the instances in index order with one based feature indexes
/// made zero based, the bias feature included; `y` their outcome ids; `n`
/// is `prob.n`, the features with the bias feature.
pub(crate) fn train(x: &Rows, y: &[u32], n: usize, bias: f64, solver: SolverType, c: f64, eps: f64) -> LinearModel {
    let classes = group_classes(y);
    let nr_class = classes.label.len();
    let nr_feature = if bias >= 0.0 { n - 1 } else { n };
    // Parameter.random: each Parameter has its own generator, seeded with
    // zero, which every class's training draws from in turn.
    let mut random = JavaRandom::new(0);

    // The instances grouped by class: sub_prob.x.
    let mut sub = Rows::with_capacity(x.len(), 0);
    for &i in &classes.perm {
        let (index, value) = x.row(i as usize);
        for (&j, &v) in index.iter().zip(value) {
            sub.push(j, v);
        }
        sub.end_row();
    }
    let columns = (solver == SolverType::L1rLr).then(|| sub.transpose(n));
    let l = sub.len();
    let mut sub_y = vec![0.0; l];
    let one = |sub_y: &[f64], w: &mut [f64], cp: f64, cn: f64, random: &mut JavaRandom| {
        train_one(&sub, columns.as_ref(), sub_y, w, solver, cp, cn, eps, random);
    };

    let w = if nr_class == 2 {
        let mut w = vec![0.0; n];
        let e0 = classes.start[0] + classes.count[0];
        sub_y[..e0].fill(1.0);
        sub_y[e0..].fill(-1.0);
        one(&sub_y, &mut w, c, c, &mut random);
        w
    } else {
        let mut model_w = vec![0.0; n * nr_class];
        let mut w = vec![0.0; n];
        for i in 0..nr_class {
            let si = classes.start[i];
            let ei = si + classes.count[i];
            sub_y.fill(-1.0);
            sub_y[si..ei].fill(1.0);
            w.fill(0.0);
            one(&sub_y, &mut w, c, c, &mut random);
            for j in 0..n {
                model_w[j * nr_class + i] = w[j];
            }
        }
        model_w
    };
    LinearModel { solver, label: classes.label, nr_feature, bias, w }
}

/// `Linear.train_one`: one weight vector, the positive class against the
/// rest.
#[allow(clippy::too_many_arguments)]
fn train_one(
    x: &Rows,
    columns: Option<&Rows>,
    y: &[f64],
    w: &mut [f64],
    solver: SolverType,
    cp: f64,
    cn: f64,
    eps: f64,
    random: &mut JavaRandom,
) {
    let l = y.len();
    let mut c = vec![0.0; l];
    let mut pos = 0usize;
    for (ci, &yi) in c.iter_mut().zip(y) {
        if yi > 0.0 {
            pos += 1;
            *ci = cp;
        } else {
            *ci = cn;
        }
    }
    let neg = l - pos;
    let mut primal_solver_tol = eps * pos.min(neg).max(1) as f64 / l as f64;
    let n = w.len();
    match solver {
        SolverType::L2rLr => {
            let mut fun = newton::LrFunction::new(x, y, &c, n);
            newton::newton(&mut fun, w, primal_solver_tol, MAX_ITERS);
        }
        SolverType::L1rLr => {
            let columns = columns.expect("the transposed problem of L1R_LR");
            l1::solve(columns, y, w, cp, cn, primal_solver_tol, MAX_ITERS, random);
        }
        SolverType::L2rLrDual => {
            let iter = dual::solve(x, y, w, cp, cn, eps, DUAL_MAX_ITER, random);
            if iter >= DUAL_MAX_ITER {
                // The dual solver did not converge: the primal Newton
                // method continues from its solution, with a tolerance a
                // tenth of the one derived from eps.
                primal_solver_tol *= 0.1;
                let mut fun = newton::LrFunction::new(x, y, &c, n);
                newton::newton(&mut fun, w, primal_solver_tol, MAX_ITERS);
            }
        }
        other => unreachable!("{} is refused before training", other.name()),
    }
}
