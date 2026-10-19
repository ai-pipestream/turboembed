//! `NegLogLikelihood`, `ParallelNegLogLikelihood`, and `QNMinimizer.L2RegFunction`.

use turbo_jvm::strict_math;

use crate::index::DataIndex;

use super::{QnParams, inner_product, log_sum_of_exps};

/// `NegLogLikelihood` and `ParallelNegLogLikelihood`, with the L2 penalty
/// of `QNMinimizer.L2RegFunction` applied on top: the objective over the
/// indexed events at a point `x[outcome * features + feature]`.
pub(super) struct NegLogLikelihood<'i> {
    index: &'i DataIndex,
    pub(super) num_outcomes: usize,
    pub(super) num_features: usize,
    pub(super) dimension: usize,
    l2_cost: f64,
    /// The events of each thread's task: `(start, len)`.
    pub(super) chunks: Vec<(usize, usize)>,
}

impl<'i> NegLogLikelihood<'i> {
    pub(super) fn new(index: &'i DataIndex, params: &QnParams) -> NegLogLikelihood<'i> {
        let num_outcomes = index.outcome_labels().len();
        let num_features = index.pred_labels().len();
        let n = index.num_unique_events();
        let threads = params.threads.max(1) as usize;
        let task_size = n / threads;
        let left_over = n % threads;
        let chunks = if threads > 1 && task_size == 0 {
            vec![(0, 0), (0, left_over)]
        } else {
            (0..threads)
                .map(|t| (t * task_size, if t == threads - 1 { task_size + left_over } else { task_size }))
                .collect()
        };
        NegLogLikelihood {
            index,
            num_outcomes,
            num_features,
            dimension: num_outcomes * num_features,
            l2_cost: params.l2_cost,
            chunks,
        }
    }

    #[inline]
    fn index_of(&self, outcome: usize, feature: u32) -> usize {
        outcome * self.num_features + feature as usize
    }

    /// Event `ei`'s sum per outcome under `x`, into `sums`.
    #[inline]
    fn sums(&self, ei: usize, x: &[f64], sums: &mut [f64]) {
        let context = self.index.context(ei);
        let values = self.index.values(ei);
        for (oi, sum) in sums.iter_mut().enumerate() {
            *sum = 0.0;
            for (ai, &p) in context.iter().enumerate() {
                let value = values.map_or(1.0, |v| f64::from(v[ai]));
                *sum += value * x[self.index_of(oi, p)];
            }
        }
    }

    /// `NegLLComputeTask.call`: the negative log likelihood of the events
    /// `start..start + len`.
    fn value_of_chunk(&self, x: &[f64], start: usize, len: usize, sums: &mut [f64]) -> f64 {
        let mut value = 0.0;
        for ei in start..start + len {
            self.sums(ei, x, sums);
            let lse = log_sum_of_exps(sums);
            let outcome = self.index.outcome(ei) as usize;
            value -= (sums[outcome] - lse) * f64::from(self.index.seen(ei));
        }
        value
    }

    /// `L2RegFunction.valueAt`: the chunks' values added in order, then
    /// the L2 term.
    pub(super) fn value_at(&self, x: &[f64], sums: &mut [f64]) -> f64 {
        let mut value = 0.0;
        if self.chunks.len() == 1 {
            value = self.value_of_chunk(x, 0, self.chunks[0].1, sums);
        } else {
            for &(start, len) in &self.chunks {
                value += self.value_of_chunk(x, start, len, sums);
            }
        }
        if self.l2_cost > 0.0 {
            value += self.l2_cost * inner_product(x, x);
        }
        value
    }

    /// `GradientComputeTask.call`: the gradient over the events
    /// `start..start + len` added into `gradient`.
    fn gradient_of_chunk(&self, x: &[f64], start: usize, len: usize, expectation: &mut [f64], gradient: &mut [f64]) {
        for ei in start..start + len {
            self.sums(ei, x, expectation);
            let lse = log_sum_of_exps(expectation);
            for e in expectation.iter_mut() {
                *e = strict_math::exp(*e - lse);
            }
            let context = self.index.context(ei);
            let values = self.index.values(ei);
            let outcome = self.index.outcome(ei) as usize;
            let seen = f64::from(self.index.seen(ei));
            for (oi, &e) in expectation.iter().enumerate() {
                let empirical = if outcome == oi { 1.0 } else { 0.0 };
                for (ai, &p) in context.iter().enumerate() {
                    let value = values.map_or(1.0, |v| f64::from(v[ai]));
                    gradient[self.index_of(oi, p)] += value * (e - empirical) * seen;
                }
            }
        }
    }

    /// `L2RegFunction.gradientAt` into `gradient`: with one thread the
    /// events accumulate into it directly; with more, each chunk into its
    /// own zeroed array (`chunk_grad`, reused) and the arrays are added in
    /// chunk order, `((0 + g0) + g1) + ...` for every element, which is
    /// `ParallelNegLogLikelihood`'s sum over its per thread arrays.
    pub(super) fn gradient_at(&self, x: &[f64], expectation: &mut [f64], gradient: &mut [f64], chunk_grad: &mut [f64]) {
        if self.chunks.len() == 1 {
            gradient.fill(0.0);
            self.gradient_of_chunk(x, 0, self.chunks[0].1, expectation, gradient);
        } else {
            gradient.fill(0.0);
            for &(start, len) in &self.chunks {
                chunk_grad.fill(0.0);
                self.gradient_of_chunk(x, start, len, expectation, chunk_grad);
                for (g, &c) in gradient.iter_mut().zip(chunk_grad.iter()) {
                    *g += c;
                }
            }
        }
        if self.l2_cost > 0.0 {
            for (g, &xi) in gradient.iter_mut().zip(x) {
                *g += 2.0 * self.l2_cost * xi;
            }
        }
    }
}
