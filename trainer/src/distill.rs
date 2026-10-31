//! Teacher-to-student knowledge distillation for analysis heads (#1520).
//!
//! Combines hard-label cross-entropy with temperature-scaled KL-divergence
//! over soft teacher logits:
//! `L = (1 - alpha) * L_CE(y, z_student) + alpha * T^2 * KL(softmax(z_teacher / T) || softmax(z_student / T))`
//! with analytical gradients w.r.t. student logits `z_student`:
//! `dL / dz_student = (1 - alpha) * (p_student(1) - y_one_hot) + alpha * T * (p_student(T) - p_teacher(T))`.

#[path = "distill_gate.rs"]
mod gate;

pub use gate::{
    DistillExample, DistillReport, DistillTask, StudentGate, ThroughputComparison, distill_multi_head, distill_tagger,
    make_distill_examples, measure_throughput, truncate_examples,
};

use turbo::head::hash_embed::{self, SEEDS};
use turbo::status::{Result, invalid};

use crate::adapter::{self, Shape};
use crate::linear;
use crate::logspace::softmax;
use crate::real::Real;
use crate::tagger::Tagger;

/// Hyperparameters and gate threshold for teacher-to-student distillation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DistillOptions {
    pub epochs: u32,
    pub batch_size: usize,
    pub lr: f64,
    pub weight_decay: f64,
    pub clip: f64,
    pub warmup: f64,
    pub seed: u64,
    pub temperature: f32,
    pub alpha: f32,
    pub max_accuracy_drop: f64,
}

impl Default for DistillOptions {
    fn default() -> Self {
        Self {
            epochs: 30,
            batch_size: 16,
            lr: 2e-3,
            weight_decay: 0.01,
            clip: 5.0,
            warmup: 0.1,
            seed: 1,
            temperature: 2.0,
            alpha: 0.5,
            max_accuracy_drop: 0.02,
        }
    }
}

impl DistillOptions {
    /// Validates distillation hyperparameters and gate bounds.
    pub fn validate(&self) -> Result<()> {
        let ok = self.epochs > 0
            && self.batch_size > 0
            && self.lr.is_finite()
            && self.lr > 0.0
            && self.weight_decay.is_finite()
            && self.weight_decay >= 0.0
            && self.clip.is_finite()
            && self.clip >= 0.0
            && self.warmup.is_finite()
            && (0.0..1.0).contains(&self.warmup)
            && self.temperature.is_finite()
            && self.temperature > 0.0
            && self.alpha.is_finite()
            && (0.0..=1.0).contains(&self.alpha)
            && self.max_accuracy_drop.is_finite()
            && self.max_accuracy_drop >= 0.0;
        if !ok {
            return Err(invalid(
                "distill: invalid DistillOptions (temperature > 0, alpha in [0, 1], max_accuracy_drop >= 0)",
            ));
        }
        Ok(())
    }
}

/// Reusable scratch buffers for allocation-free distillation loss and gradients.
#[derive(Debug, Default, Clone)]
pub struct DistillScratch {
    p_s1: Vec<f64>,
    p_st: Vec<f64>,
    p_tt: Vec<f64>,
    delta: Vec<f64>,
}

/// Combined hard-label cross-entropy and temperature-scaled KL distillation loss.
pub fn distill_loss<R: Real>(
    student_scores: &[R],
    teacher_scores: &[R],
    gold: &[u32],
    labels: usize,
    temperature: f64,
    alpha: f64,
    dscores: &mut [R],
) -> Result<f64> {
    distill_loss_with(
        student_scores,
        teacher_scores,
        gold,
        labels,
        temperature,
        alpha,
        dscores,
        &mut DistillScratch::default(),
    )
}

/// As [`distill_loss`], reusing `scratch` across calls so the inner token loop allocates nothing.
#[allow(clippy::too_many_arguments)]
pub fn distill_loss_with<R: Real>(
    student_scores: &[R],
    teacher_scores: &[R],
    gold: &[u32],
    labels: usize,
    temperature: f64,
    alpha: f64,
    dscores: &mut [R],
    scratch: &mut DistillScratch,
) -> Result<f64> {
    let cells = gold.len().checked_mul(labels);
    if labels == 0 || [student_scores.len(), teacher_scores.len(), dscores.len()].iter().any(|&n| Some(n) != cells) {
        return Err(invalid("distill: student_scores, teacher_scores, and dscores must match [gold, labels]"));
    }
    if !(temperature.is_finite() && temperature > 0.0) || !(alpha.is_finite() && (0.0..=1.0).contains(&alpha)) {
        return Err(invalid("distill: temperature > 0 and alpha in [0, 1] are required"));
    }
    if student_scores.iter().chain(teacher_scores).any(|v| !v.to_f64().is_finite())
        || gold.iter().any(|&g| g as usize >= labels)
    {
        return Err(invalid("distill: non-finite score or gold label out of range"));
    }
    scratch.p_s1.resize(labels, 0.0);
    scratch.p_st.resize(labels, 0.0);
    scratch.p_tt.resize(labels, 0.0);
    scratch.delta.clear();
    scratch.delta.resize(student_scores.len(), 0.0);
    let (inv_t, w_hard, w_soft_l, w_soft_g) =
        (1.0 / temperature, 1.0 - alpha, alpha * temperature * temperature, alpha * temperature);
    let mut total = 0.0;
    for (t, &g) in gold.iter().enumerate() {
        let (g, base) = (g as usize, t * labels);
        let (s_row, t_row) = (&student_scores[base..base + labels], &teacher_scores[base..base + labels]);
        for c in 0..labels {
            let (z, u) = (s_row[c].to_f64(), t_row[c].to_f64());
            (scratch.p_s1[c], scratch.p_st[c], scratch.p_tt[c]) = (z, z * inv_t, u * inv_t);
        }
        let z_gold = scratch.p_s1[g];
        let (m1, ln_z1) = softmax(&mut scratch.p_s1);
        let (ms, ln_zs) = softmax(&mut scratch.p_st);
        let (mt, ln_zt) = softmax(&mut scratch.p_tt);
        let mut kl = 0.0;
        let d_row = &mut scratch.delta[base..base + labels];
        for c in 0..labels {
            let log_pt = (t_row[c].to_f64() * inv_t - mt) - ln_zt;
            let log_ps = (s_row[c].to_f64() * inv_t - ms) - ln_zs;
            let (pt, ps) = (scratch.p_tt[c], scratch.p_st[c]);
            kl += pt * (log_pt - log_ps);
            d_row[c] = w_hard * (scratch.p_s1[c] - if c == g { 1.0 } else { 0.0 }) + w_soft_g * (ps - pt);
        }
        total += w_hard * (ln_z1 - (z_gold - m1)) + w_soft_l * kl;
    }
    if !total.is_finite() || !scratch.delta.iter().all(|d| d.is_finite()) {
        return Err(invalid("distill: the loss or a gradient is not finite"));
    }
    dscores.iter_mut().zip(&scratch.delta).for_each(|(d, &x)| *d += R::from_f64(x));
    Ok(total)
}

/// Optional MSE feature alignment loss between linearly projected student hidden states and teacher states.
#[allow(clippy::too_many_arguments)]
pub fn feature_align_loss<R: Real>(
    proj_w: &[R],
    proj_b: &[R],
    student_features: &[R],
    teacher_features: &[R],
    d_student: usize,
    d_teacher: usize,
    weight: f64,
    d_proj_w: &mut [R],
    d_proj_b: &mut [R],
    d_student_features: Option<&mut [R]>,
) -> Result<f64> {
    if d_student == 0 || d_teacher == 0 || !(weight.is_finite() && weight >= 0.0) {
        return Err(invalid("distill: d_student > 0, d_teacher > 0, and finite weight >= 0 are required"));
    }
    let n = student_features.len() / d_student;
    if proj_w.len() != d_teacher * d_student
        || proj_b.len() != d_teacher
        || d_proj_w.len() != proj_w.len()
        || d_proj_b.len() != proj_b.len()
        || !student_features.len().is_multiple_of(d_student)
        || teacher_features.len() != n * d_teacher
        || d_student_features.as_ref().is_some_and(|dx| dx.len() != student_features.len())
    {
        return Err(invalid("distill: feature projection shapes do not match"));
    }
    let (mut projected, mut d_proj) = (vec![R::ZERO; n * d_teacher], vec![R::ZERO; n * d_teacher]);
    linear::forward(proj_w, proj_b, student_features, d_student, d_teacher, &mut projected);
    let (scale, mut loss) = (weight / d_teacher as f64, 0.0);
    for i in 0..projected.len() {
        let diff = projected[i].to_f64() - teacher_features[i].to_f64();
        loss += 0.5 * scale * diff * diff;
        d_proj[i] = R::from_f64(scale * diff);
    }
    if !loss.is_finite() {
        return Err(invalid("distill: feature alignment loss is not finite"));
    }
    linear::backward(proj_w, student_features, &d_proj, d_student, d_teacher, d_proj_w, d_proj_b, d_student_features);
    Ok(loss)
}

/// Working memory for one sequence's forward and backward through a student [`Tagger`].
#[derive(Debug, Default)]
pub struct DistillWork<R> {
    input: Vec<R>,
    rows: Vec<[[u32; SEEDS]; 4]>,
    adapter: adapter::Work<R>,
    scores: Vec<R>,
    dscores: Vec<R>,
    scratch: DistillScratch,
}

/// Computes the distillation loss of one sequence on `student`, adding gradients into `grads`.
#[allow(clippy::too_many_arguments)]
pub fn tagger_distill_loss<R: Real>(
    student: &Tagger<R>,
    features: &[R],
    forms: Option<&[&str]>,
    teacher_scores: &[R],
    gold: &[u32],
    temperature: f64,
    alpha: f64,
    grads: &mut [Vec<R>],
    w: &mut DistillWork<R>,
) -> Result<f64> {
    let (params, spec, hidden, n) = (student.params(), student.spec(), student.hidden(), gold.len());
    if grads.len() != params.len() || grads.iter().zip(params).any(|(g, p)| g.len() != p.len()) {
        return Err(invalid(format!("distill: gradients do not match head {:?}", spec.name)));
    }
    if !features.len().is_multiple_of(hidden) || features.len() / hidden != n {
        return Err(invalid("distill: features do not match [gold, hidden]"));
    }
    let hashed_dim = spec.hash_embed.map_or(0, |h| 4 * h.dim as usize);
    let (layers, h) = spec.adapter.map_or((0, 0), |a| (a.bilstm.layers as usize, a.bilstm.hidden as usize));
    let shape = Shape { hidden: hidden + hashed_dim, layers, h };
    if let Some(he) = spec.hash_embed {
        w.rows.clear();
        w.input.clear();
        if n > 0 {
            let f = forms.filter(|f| f.len() == n).ok_or_else(|| invalid("distill: forms required for hash_embed"))?;
            let (d, width0, t) = (he.dim as usize, hidden + 4 * he.dim as usize, 6 * layers);
            w.rows.extend(f.iter().map(|s| hash_embed::rows(&he, s)));
            w.input.resize(n * width0, R::ZERO);
            for i in 0..n {
                let row = &mut w.input[i * width0..][..width0];
                row[..hidden].copy_from_slice(&features[i * hidden..][..hidden]);
                for (k, seeds) in w.rows[i].iter().enumerate() {
                    let dst = &mut row[hidden + k * d..][..d];
                    for &r in seeds {
                        dst.iter_mut().zip(&params[t + k][r as usize * d..][..d]).for_each(|(a, &b)| *a += b);
                    }
                }
            }
        }
    }
    let x0: &[R] = if spec.hash_embed.is_some() { &w.input } else { features };
    adapter::forward(shape, params, x0, &mut w.adapter);
    let (width, labels, o) =
        (shape.width(), spec.labels.len(), 6 * layers + usize::from(spec.hash_embed.is_some()) * 4);
    let x = adapter::output(x0, &w.adapter);
    w.scores.clear();
    w.scores.resize(n * labels, R::ZERO);
    linear::forward(&params[o], &params[o + 1], x, width, labels, &mut w.scores);
    w.dscores.clear();
    w.dscores.resize(n * labels, R::ZERO);
    let loss =
        distill_loss_with(&w.scores, teacher_scores, gold, labels, temperature, alpha, &mut w.dscores, &mut w.scratch)?;
    let hashed = spec.hash_embed.is_some();
    let mut dx = vec![R::ZERO; if layers == 0 && !hashed { 0 } else { n * width }];
    let [dw, db] = grads.get_disjoint_mut([o, o + 1]).expect("distinct");
    linear::backward(&params[o], x, &w.dscores, width, labels, dw, db, (!dx.is_empty()).then_some(&mut dx[..]));
    let dx0 = adapter::backward(shape, params, x0, dx, grads, &w.adapter, hashed);
    if let Some(he) = spec.hash_embed {
        let (d, t, width0) = (he.dim as usize, 6 * layers, hidden + 4 * he.dim as usize);
        for (i, word) in w.rows.iter().enumerate() {
            for (k, seeds) in word.iter().enumerate() {
                let src = &dx0[i * width0 + hidden + k * d..][..d];
                for &r in seeds {
                    grads[t + k][r as usize * d..][..d].iter_mut().zip(src).for_each(|(a, &b)| *a += b);
                }
            }
        }
    }
    Ok(loss)
}
