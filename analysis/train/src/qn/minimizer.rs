//! `QNMinimizer`, `UpdateInfo`, and `LineSearch`.

use crate::error::{Error, Result};

use super::generic::Objective;
use super::{
    ARMIJO, CONVERGE_TOLERANCE, MIN_STEP_SIZE, QnIteration, QnParams, QnStop, REL_GRAD_NORM_TOL, SHRINK, inner_product,
    l1_norm, l2_norm, max_one,
};

/// `LineSearchResult`: the two ends of the latest step.
pub(super) struct Search {
    pub(super) step_size: f64,
    pub(super) value_at_curr: f64,
    pub(super) value_at_next: f64,
    pub(super) grad_at_curr: Vec<f64>,
    pub(super) grad_at_next: Vec<f64>,
    /// With L1: the pseudo gradient at `next_point`, recomputed in place
    /// after each step.
    pub(super) pseudo_grad: Vec<f64>,
    pub(super) curr_point: Vec<f64>,
    pub(super) next_point: Vec<f64>,
    /// With L1: the orthant of the step.
    pub(super) sign: Vec<f64>,
    pub(super) fct_eval_count: i32,
}

impl Search {
    /// `getFuncChangeRate`.
    fn change_rate(&self) -> f64 {
        (self.value_at_curr - self.value_at_next) / self.value_at_curr
    }
}

/// `QNMinimizer.UpdateInfo`: the last `m` corrections `S` (points) and `Y`
/// (gradients) with `rho = 1 / (s . y)`. The JVM shifts the array
/// references when it discards the oldest, which aliases slots (see the
/// module documentation); `slot[i]` is which array position `i` refers to.
pub(super) struct UpdateInfo {
    m: usize,
    k: usize,
    pub(super) slot: Vec<usize>,
    pub(super) s: Vec<Vec<f64>>,
    pub(super) y: Vec<Vec<f64>>,
    pub(super) rho: Vec<f64>,
    alpha: Vec<f64>,
}

impl UpdateInfo {
    pub(super) fn new(m: usize, dimension: usize) -> UpdateInfo {
        UpdateInfo {
            m,
            k: 0,
            slot: (0..m).collect(),
            s: vec![vec![0.0; dimension]; m],
            y: vec![vec![0.0; dimension]; m],
            rho: vec![0.0; m],
            alpha: vec![0.0; m],
        }
    }

    /// `update(lsr)`.
    pub(super) fn update(&mut self, search: &Search) {
        let position = if self.k < self.m {
            self.k
        } else {
            for i in 0..self.m - 1 {
                self.slot[i] = self.slot[i + 1];
                self.rho[i] = self.rho[i + 1];
            }
            self.m - 1
        };
        let (s, y) = (&mut self.s[self.slot[position]], &mut self.y[self.slot[position]]);
        let mut sy = 0.0;
        for j in 0..s.len() {
            s[j] = search.next_point[j] - search.curr_point[j];
            y[j] = search.grad_at_next[j] - search.grad_at_curr[j];
            sy += s[j] * y[j];
        }
        self.rho[position] = 1.0 / sy;
        if self.k < self.m {
            self.k += 1;
        }
    }

    /// `computeDirection`: the two loop recursion over the gradient in
    /// `direction`, negated at the end.
    fn direction(&mut self, direction: &mut [f64]) {
        for i in (0..self.k).rev() {
            let s = &self.s[self.slot[i]];
            let y = &self.y[self.slot[i]];
            self.alpha[i] = self.rho[i] * inner_product(s, direction);
            for (d, &yj) in direction.iter_mut().zip(y) {
                *d -= self.alpha[i] * yj;
            }
        }
        for i in 0..self.k {
            let s = &self.s[self.slot[i]];
            let y = &self.y[self.slot[i]];
            let beta = self.rho[i] * inner_product(y, direction);
            for (d, &sj) in direction.iter_mut().zip(s) {
                *d += sj * (self.alpha[i] - beta);
            }
        }
        for d in direction.iter_mut() {
            *d = -*d;
        }
    }
}

/// `QNMinimizer.minimize` in progress.
pub(super) struct Minimizer<'f> {
    function: &'f dyn Objective,
    params: QnParams,
    pub(super) search: Search,
    updates: UpdateInfo,
    direction: Vec<f64>,
    /// Scratch for the function: one sum per outcome, the gradient, and
    /// with several threads one chunk's gradient.
    sums: Vec<f64>,
    gradient: Vec<f64>,
    chunk_grad: Vec<f64>,
}

impl<'f> Minimizer<'f> {
    pub(super) fn new(function: &'f dyn Objective, params: &QnParams) -> Minimizer<'f> {
        let dimension = function.dimension();
        let threads = function.chunks();
        Minimizer {
            function,
            params: *params,
            search: Search {
                step_size: 0.0,
                value_at_curr: 0.0,
                value_at_next: 0.0,
                grad_at_curr: vec![0.0; dimension],
                grad_at_next: vec![0.0; dimension],
                pseudo_grad: vec![0.0; if params.l1() { dimension } else { 0 }],
                curr_point: vec![0.0; dimension],
                next_point: vec![0.0; dimension],
                sign: vec![0.0; if params.l1() { dimension } else { 0 }],
                fct_eval_count: 0,
            },
            updates: UpdateInfo::new(params.updates as usize, dimension),
            direction: vec![0.0; dimension],
            sums: vec![0.0; function.scratch()],
            gradient: vec![0.0; dimension],
            chunk_grad: if threads > 1 { vec![0.0; dimension] } else { Vec::new() },
        }
    }

    fn value_at(&mut self, x: &[f64]) -> f64 {
        self.function.value_at(x, &mut self.sums)
    }

    /// The gradient at `x` into `out`.
    fn gradient_at(&mut self, x: &[f64], out: &mut [f64]) {
        self.function.gradient_at(x, &mut self.sums, &mut self.gradient, &mut self.chunk_grad);
        out.copy_from_slice(&self.gradient);
    }

    /// `computePseudoGrad`: the pseudo gradient of the L1 penalized
    /// objective at `x` with gradient `g`, into `pg`.
    fn pseudo_gradient(l1_cost: f64, x: &[f64], g: &[f64], pg: &mut [f64]) {
        for ((p, &xi), &gi) in pg.iter_mut().zip(x).zip(g) {
            // At zero, the one sided derivative that points downhill.
            *p = if xi < 0.0 || (xi == 0.0 && gi > l1_cost) {
                gi - l1_cost
            } else if xi > 0.0 || gi < -l1_cost {
                gi + l1_cost
            } else {
                0.0
            };
        }
    }

    /// The loop of `minimize`: returns why it stopped.
    pub(super) fn minimize(&mut self, observe: &mut dyn FnMut(&QnIteration, &[f64])) -> Result<QnStop> {
        let l1 = self.params.l1();
        // The origin (next_point starts as it), its value and gradient, in
        // the search's own buffers.
        let origin = std::mem::take(&mut self.search.next_point);
        let mut grad = std::mem::take(&mut self.search.grad_at_next);
        let mut value = self.value_at(&origin);
        self.gradient_at(&origin, &mut grad);
        if l1 {
            value += self.params.l1_cost * l1_norm(&origin);
            let mut pseudo = std::mem::take(&mut self.search.pseudo_grad);
            Minimizer::pseudo_gradient(self.params.l1_cost, &origin, &grad, &mut pseudo);
            self.search.pseudo_grad = pseudo;
        }
        self.search.value_at_next = value;
        self.search.grad_at_next = grad;
        self.search.next_point = origin;

        let mut initial_step = 1.0 / l2_norm(if l1 { &self.search.pseudo_grad } else { &self.search.grad_at_next });
        for iteration in 1..=self.params.iterations {
            self.direction.copy_from_slice(if l1 { &self.search.pseudo_grad } else { &self.search.grad_at_next });
            self.updates.direction(&mut self.direction);
            if l1 {
                for (d, &pg) in self.direction.iter_mut().zip(&self.search.pseudo_grad) {
                    if *d * pg >= 0.0 {
                        *d = 0.0;
                    }
                }
                self.constrained_line_search(initial_step)?;
                let mut pseudo = std::mem::take(&mut self.search.pseudo_grad);
                Minimizer::pseudo_gradient(
                    self.params.l1_cost,
                    &self.search.next_point,
                    &self.search.grad_at_next,
                    &mut pseudo,
                );
                self.search.pseudo_grad = pseudo;
            } else {
                self.line_search(initial_step)?;
            }
            self.updates.update(&self.search);
            let stop = self.converged();
            observe(
                &QnIteration {
                    iteration,
                    value: self.search.value_at_next,
                    change_rate: self.search.change_rate(),
                    step_size: self.search.step_size,
                    fct_evals: self.search.fct_eval_count,
                    stop,
                },
                &self.search.next_point,
            );
            if let Some(stop) = stop {
                return Ok(stop);
            }
            initial_step = 1.0;
        }
        Ok(QnStop::Iterations)
    }

    /// `LineSearch.doLineSearch`: backtracking from `initial_step` along
    /// `direction` until the Armijo condition holds, then the gradient at
    /// the new point.
    fn line_search(&mut self, initial_step: f64) -> Result<()> {
        let mut step = initial_step;
        let mut count = self.search.fct_eval_count;
        let dir_gradient = inner_product(&self.direction, &self.search.grad_at_next);
        let cached = ARMIJO * dir_gradient;
        let mut next = std::mem::take(&mut self.search.curr_point);
        let value_at_next;
        loop {
            for ((n, &x), &d) in next.iter_mut().zip(&self.search.next_point).zip(&self.direction) {
                *n = x + d * step;
            }
            let value = self.value_at(&next);
            count += 1;
            if value <= self.search.value_at_next + cached * step {
                value_at_next = value;
                break;
            }
            if !step.is_finite() || step <= 0.0 {
                return Err(Error::LineSearch);
            }
            step *= SHRINK;
        }
        let mut grad = std::mem::take(&mut self.search.grad_at_curr);
        self.gradient_at(&next, &mut grad);
        self.set_all(step, value_at_next, grad, next, count);
        Ok(())
    }

    /// `LineSearch.doConstrainedLineSearch`: the step projected onto the
    /// orthant of the current point (its pseudo gradient's where a
    /// coordinate is zero), with the L1 term in the value and the
    /// sufficient decrease measured against the pseudo gradient.
    fn constrained_line_search(&mut self, initial_step: f64) -> Result<()> {
        let mut step = initial_step;
        let mut count = self.search.fct_eval_count;
        for ((sign, &x), &pg) in self.search.sign.iter_mut().zip(&self.search.next_point).zip(&self.search.pseudo_grad)
        {
            *sign = if x == 0.0 { -pg } else { x };
        }
        let mut next = std::mem::take(&mut self.search.curr_point);
        let value_at_next;
        loop {
            for ((n, &x), &d) in next.iter_mut().zip(&self.search.next_point).zip(&self.direction) {
                *n = x + d * step;
            }
            for (n, &sign) in next.iter_mut().zip(&self.search.sign) {
                if *n * sign <= 0.0 {
                    *n = 0.0;
                }
            }
            let value = self.value_at(&next) + self.params.l1_cost * l1_norm(&next);
            count += 1;
            let mut dir_gradient = 0.0;
            for ((&n, &x), &pg) in next.iter().zip(&self.search.next_point).zip(&self.search.pseudo_grad) {
                dir_gradient += (n - x) * pg;
            }
            if value <= self.search.value_at_next + ARMIJO * dir_gradient {
                value_at_next = value;
                break;
            }
            if !step.is_finite() || step <= 0.0 {
                return Err(Error::LineSearch);
            }
            step *= SHRINK;
        }
        let mut grad = std::mem::take(&mut self.search.grad_at_curr);
        self.gradient_at(&next, &mut grad);
        self.set_all(step, value_at_next, grad, next, count);
        Ok(())
    }

    /// `LineSearchResult.setAll`: the old next becomes the current, the
    /// new point the next.
    fn set_all(&mut self, step: f64, value_at_next: f64, grad_at_next: Vec<f64>, next: Vec<f64>, count: i32) {
        let s = &mut self.search;
        s.step_size = step;
        s.value_at_curr = s.value_at_next;
        s.value_at_next = value_at_next;
        s.grad_at_curr = std::mem::replace(&mut s.grad_at_next, grad_at_next);
        s.curr_point = std::mem::replace(&mut s.next_point, next);
        s.fct_eval_count = count;
    }

    /// `isConverged`, as the test that fired.
    fn converged(&self) -> Option<QnStop> {
        let s = &self.search;
        if s.change_rate() < CONVERGE_TOLERANCE {
            return Some(QnStop::ChangeRate);
        }
        let x_norm = max_one(l2_norm(&s.next_point));
        let grad_norm = l2_norm(if self.params.l1() { &s.pseudo_grad } else { &s.grad_at_next });
        if grad_norm / x_norm < REL_GRAD_NORM_TOL {
            return Some(QnStop::GradientNorm);
        }
        if s.step_size < MIN_STEP_SIZE {
            return Some(QnStop::StepSize);
        }
        if s.fct_eval_count > self.params.max_fct_eval {
            return Some(QnStop::FunctionEvaluations);
        }
        None
    }
}
