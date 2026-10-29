//! Prediction, ported from LIBSVM 3.35's Java `svm.java` (BSD-3-Clause,
//! the version zlibsvm 3.0.0 wraps): `Kernel.k_function`,
//! `svm_predict_values`, `svm_predict_probability`, `sigmoid_predict`,
//! `multiclass_probability` and `predict_one_class_probability`.
//!
//! Every sum is taken in LIBSVM's order with the same operations, and the
//! sigmoid kernel calls `turbo_jvm::strict_math::tanh`, so the linear,
//! polynomial and sigmoid kernels give the JVM's decision values bit for
//! bit. The RBF kernel and the probability sigmoid call `exp`, which the
//! JVM computes with its own intrinsic; Rust's may differ from it in the
//! last place (docs/doccat.md, "Tolerances").

use super::model::{Kernel, LibsvmModel, SvmType};
use crate::java;

/// The lowest pairwise probability, and one minus it the highest.
const MIN_PROB: f64 = 1e-7;

/// Buffers for one prediction, kept between calls.
#[derive(Debug, Default, Clone)]
pub struct PredictScratch {
    kvalue: Vec<f64>,
    vote: Vec<u32>,
    dec_values: Vec<f64>,
    pairwise: Vec<f64>,
    q: Vec<f64>,
    qp: Vec<f64>,
    prob: Vec<f64>,
}

impl PredictScratch {
    pub fn new() -> PredictScratch {
        PredictScratch::default()
    }

    /// The decision values of the last prediction: one per class pair
    /// `(i, j)`, `i < j` in label order, row by row; or one.
    pub fn decision_values(&self) -> &[f64] {
        &self.dec_values
    }

    /// The probability of each class, in label order, after
    /// `predict_probability` on a model that has them.
    pub fn probabilities(&self) -> &[f64] {
        &self.prob
    }
}

/// `Kernel.dot`: the products of the matching indexes, in index order.
#[inline]
fn dot(xi: &[i32], xv: &[f64], yi: &[i32], yv: &[f64]) -> f64 {
    let mut sum = 0.0;
    let (mut i, mut j) = (0, 0);
    while i < xi.len() && j < yi.len() {
        if xi[i] == yi[j] {
            sum += xv[i] * yv[j];
            i += 1;
            j += 1;
        } else if xi[i] > yi[j] {
            j += 1;
        } else {
            i += 1;
        }
    }
    sum
}

/// `Kernel.powi`: `base^times` by repeated squaring; 1 for `times <= 0`.
#[inline]
fn powi(base: f64, times: i32) -> f64 {
    let mut tmp = base;
    let mut ret = 1.0;
    let mut t = times;
    while t > 0 {
        if t % 2 == 1 {
            ret *= tmp;
        }
        tmp *= tmp;
        t /= 2;
    }
    ret
}

/// `Kernel.k_function(x, y, param)`, `x` the input and `y` a support
/// vector.
#[inline]
fn k_function(kernel: Kernel, xi: &[i32], xv: &[f64], yi: &[i32], yv: &[f64]) -> f64 {
    match kernel {
        Kernel::Linear => dot(xi, xv, yi, yv),
        Kernel::Polynomial { degree, gamma, coef0 } => powi(gamma * dot(xi, xv, yi, yv) + coef0, degree),
        Kernel::Rbf { gamma } => {
            let mut sum = 0.0;
            let (mut i, mut j) = (0, 0);
            while i < xi.len() && j < yi.len() {
                if xi[i] == yi[j] {
                    let d = xv[i] - yv[j];
                    sum += d * d;
                    i += 1;
                    j += 1;
                } else if xi[i] > yi[j] {
                    sum += yv[j] * yv[j];
                    j += 1;
                } else {
                    sum += xv[i] * xv[i];
                    i += 1;
                }
            }
            while i < xi.len() {
                sum += xv[i] * xv[i];
                i += 1;
            }
            while j < yi.len() {
                sum += yv[j] * yv[j];
                j += 1;
            }
            (-gamma * sum).exp()
        }
        Kernel::Sigmoid { gamma, coef0 } => turbo_jvm::strict_math::tanh(gamma * dot(xi, xv, yi, yv) + coef0),
    }
}

/// `sigmoid_predict`: Platt's sigmoid, written so `1 - p` keeps its
/// precision.
#[inline]
fn sigmoid_predict(decision_value: f64, a: f64, b: f64) -> f64 {
    let f_apb = decision_value * a + b;
    if f_apb >= 0.0 { (-f_apb).exp() / (1.0 + (-f_apb).exp()) } else { 1.0 / (1.0 + f_apb.exp()) }
}

/// `multiclass_probability`: method 2 of Wu, Lin and Weng (2004), the
/// class probabilities from the pairwise ones `r` (`k` by `k`, row major)
/// into `p`. `q` and `qp` are buffers.
fn multiclass_probability(k: usize, r: &[f64], p: &mut [f64], q: &mut Vec<f64>, qp: &mut Vec<f64>) {
    let max_iter = 100.max(k);
    let eps = 0.005 / k as f64;
    q.clear();
    q.resize(k * k, 0.0);
    qp.clear();
    qp.resize(k, 0.0);
    for t in 0..k {
        p[t] = 1.0 / k as f64;
        q[t * k + t] = 0.0;
        for j in 0..t {
            q[t * k + t] += r[j * k + t] * r[j * k + t];
            q[t * k + j] = q[j * k + t];
        }
        for j in t + 1..k {
            q[t * k + t] += r[j * k + t] * r[j * k + t];
            q[t * k + j] = -r[j * k + t] * r[t * k + j];
        }
    }
    for _ in 0..max_iter {
        // Stopping condition: recalculate Qp and pQp for numerical accuracy.
        let mut pqp = 0.0;
        for t in 0..k {
            qp[t] = 0.0;
            for j in 0..k {
                qp[t] += q[t * k + j] * p[j];
            }
            pqp += p[t] * qp[t];
        }
        let mut max_error = 0.0;
        for &v in qp.iter() {
            let error = (v - pqp).abs();
            if error > max_error {
                max_error = error;
            }
        }
        if max_error < eps {
            break;
        }
        for t in 0..k {
            let diff = (-qp[t] + pqp) / q[t * k + t];
            p[t] += diff;
            pqp = (pqp + diff * (diff * q[t * k + t] + 2.0 * qp[t])) / (1.0 + diff) / (1.0 + diff);
            for j in 0..k {
                qp[j] = (qp[j] + diff * q[t * k + j]) / (1.0 + diff);
                p[j] /= 1.0 + diff;
            }
        }
    }
}

impl LibsvmModel {
    /// `svm_predict_values` on the sparse input `x` (indexes ascending):
    /// writes the decision values into the scratch and returns the
    /// predicted label (classification, by one-against-one votes, ties to
    /// the first class), the sign of the decision value (one-class) or the
    /// value (regression).
    pub fn predict_values(&self, x_index: &[i32], x_value: &[f64], s: &mut PredictScratch) -> f64 {
        let l = self.sv.len();
        let kernel = self.kernel;
        if !self.svm_type.is_classification() {
            let mut sum = 0.0;
            for i in 0..l {
                let (yi, yv) = self.sv.row(i);
                sum += self.sv_coef[i] * k_function(kernel, x_index, x_value, yi, yv);
            }
            sum -= self.rho[0];
            s.dec_values.clear();
            s.dec_values.push(sum);
            return if self.svm_type == SvmType::OneClass { if sum > 0.0 { 1.0 } else { -1.0 } } else { sum };
        }
        let nr_class = self.nr_class;
        s.kvalue.clear();
        for i in 0..l {
            let (yi, yv) = self.sv.row(i);
            s.kvalue.push(k_function(kernel, x_index, x_value, yi, yv));
        }
        s.vote.clear();
        s.vote.resize(nr_class, 0);
        s.dec_values.clear();
        let mut si = 0;
        let mut p = 0;
        for i in 0..nr_class {
            let ci = self.n_sv[i] as usize;
            let mut sj = si + ci;
            for j in i + 1..nr_class {
                let cj = self.n_sv[j] as usize;
                let coef1 = &self.sv_coef[(j - 1) * l..j * l];
                let coef2 = &self.sv_coef[i * l..(i + 1) * l];
                let mut sum = 0.0;
                for k in 0..ci {
                    sum += coef1[si + k] * s.kvalue[si + k];
                }
                for k in 0..cj {
                    sum += coef2[sj + k] * s.kvalue[sj + k];
                }
                sum -= self.rho[p];
                s.dec_values.push(sum);
                if sum > 0.0 {
                    s.vote[i] += 1;
                } else {
                    s.vote[j] += 1;
                }
                p += 1;
                sj += cj;
            }
            si += ci;
        }
        let mut best = 0;
        for i in 1..nr_class {
            if s.vote[i] > s.vote[best] {
                best = i;
            }
        }
        f64::from(self.label[best])
    }

    /// `svm_predict_probability`: for a C-SVC or nu-SVC model with probA
    /// and probB, the probability of each class into the scratch (label
    /// order) and the label of the most probable one (ties to the first);
    /// for a one-class model with density marks, the probabilities of +1
    /// and -1 and the predicted sign. Otherwise `svm_predict`, and the
    /// scratch holds no probabilities.
    pub fn predict_probability(&self, x_index: &[i32], x_value: &[f64], s: &mut PredictScratch) -> f64 {
        s.prob.clear();
        if let (true, Some(sigmoid)) = (self.svm_type.is_classification(), &self.probability) {
            let k = self.nr_class;
            self.predict_values(x_index, x_value, s);
            s.pairwise.clear();
            s.pairwise.resize(k * k, 0.0);
            let mut pair = 0;
            for i in 0..k {
                for j in i + 1..k {
                    let p = sigmoid_predict(s.dec_values[pair], sigmoid.a[pair], sigmoid.b[pair]);
                    let p = java::min(java::max(p, MIN_PROB), 1.0 - MIN_PROB);
                    s.pairwise[i * k + j] = p;
                    s.pairwise[j * k + i] = 1.0 - p;
                    pair += 1;
                }
            }
            s.prob.resize(k, 0.0);
            if k == 2 {
                s.prob[0] = s.pairwise[1];
                s.prob[1] = s.pairwise[2];
            } else {
                multiclass_probability(k, &s.pairwise, &mut s.prob, &mut s.q, &mut s.qp);
            }
            let mut best = 0;
            for i in 1..k {
                if s.prob[i] > s.prob[best] {
                    best = i;
                }
            }
            return f64::from(self.label[best]);
        }
        if let (SvmType::OneClass, Some(marks)) = (self.svm_type, &self.density_marks) {
            let predicted = self.predict_values(x_index, x_value, s);
            let p = one_class_probability(marks, s.dec_values[0]);
            s.prob.push(p);
            s.prob.push(1.0 - p);
            return predicted;
        }
        self.predict_values(x_index, x_value, s)
    }
}

/// `predict_one_class_probability`: the share of training decision values
/// below `dec_value`, read from the ten marks.
fn one_class_probability(marks: &[f64], dec_value: f64) -> f64 {
    let nr_marks = marks.len();
    let mut prob_estimate = 0.0;
    if dec_value < marks[0] {
        prob_estimate = 0.001;
    } else if dec_value > marks[nr_marks - 1] {
        prob_estimate = 0.999;
    } else {
        if let Some((i, _)) = marks.iter().enumerate().skip(1).find(|&(_, &m)| dec_value < m) {
            prob_estimate = i as f64 / nr_marks as f64;
        }
    }
    prob_estimate
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_multiplies_matching_indexes_only() {
        assert_eq!(
            dot(&[1, 3, 5], &[1.0, 2.0, 3.0], &[2, 3, 5, 7], &[10.0, 20.0, 30.0, 40.0]),
            2.0 * 20.0 + 3.0 * 30.0
        );
        assert_eq!(dot(&[], &[], &[1], &[1.0]), 0.0);
    }

    #[test]
    fn powi_squares_repeatedly_and_is_one_below_one() {
        assert_eq!(powi(2.0, 10), 1024.0);
        assert_eq!(powi(3.0, 3), 27.0);
        assert_eq!(powi(5.0, 0), 1.0);
        assert_eq!(powi(5.0, -2), 1.0);
    }

    #[test]
    fn rbf_sums_every_squared_difference() {
        let k = k_function(Kernel::Rbf { gamma: 0.5 }, &[1, 2], &[1.0, 2.0], &[2, 3], &[1.0, 3.0]);
        // (1 - 0)^2 + (2 - 1)^2 + (0 - 3)^2 = 11
        assert_eq!(k, (-0.5f64 * 11.0).exp());
    }

    #[test]
    fn sigmoid_predict_is_one_minus_itself_mirrored() {
        assert_eq!(sigmoid_predict(0.0, 1.0, 0.0), 0.5);
        let p = sigmoid_predict(2.0, -1.0, 0.0);
        let q = sigmoid_predict(-2.0, -1.0, 0.0);
        assert!((p + q - 1.0).abs() < 1e-15);
        assert!(p > 0.5);
    }

    #[test]
    fn multiclass_probability_of_one_class_is_one() {
        let mut p = [0.0];
        multiclass_probability(1, &[0.0], &mut p, &mut Vec::new(), &mut Vec::new());
        assert_eq!(p, [1.0]);
    }

    #[test]
    fn multiclass_probability_sums_to_one_and_ranks_by_the_pairs() {
        // Class 0 beats 1 and 2, class 1 beats 2.
        let r = [0.0, 0.8, 0.9, 0.2, 0.0, 0.7, 0.1, 0.3, 0.0];
        let mut p = [0.0; 3];
        multiclass_probability(3, &r, &mut p, &mut Vec::new(), &mut Vec::new());
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-9, "{p:?}");
        assert!(p[0] > p[1] && p[1] > p[2], "{p:?}");
    }

    #[test]
    fn one_class_probability_reads_the_marks() {
        let marks: Vec<f64> = (0..10).map(f64::from).collect();
        assert_eq!(one_class_probability(&marks, -1.0), 0.001);
        assert_eq!(one_class_probability(&marks, 10.0), 0.999);
        assert_eq!(one_class_probability(&marks, 3.5), 0.4);
    }
}
