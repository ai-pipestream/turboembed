//! AdamW (Loshchilov and Hutter, arXiv 1711.05101) over a list of
//! tensors, with global-norm clipping and a learning rate that warms up
//! linearly and then decays linearly toward zero: every step of a
//! schedule moves the weights, the last by lr / (total - warmup). Weight decay is decoupled
//! from the gradient and applied only to tensors marked for it.

use turbo::status::{Result, invalid};

use crate::real::Real;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Settings {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    /// The largest global L2 norm of the gradients; 0 turns clipping off.
    pub clip: f64,
    pub warmup: u64,
    pub total: u64,
}

impl Settings {
    /// Refuses settings that would not train: a rate or eps that is not
    /// positive and finite, betas outside [0, 1), a negative or nonfinite
    /// decay or clip, or a warmup longer than the schedule.
    pub fn check(&self) -> Result<()> {
        let finite = |v: f64| v.is_finite();
        if !(finite(self.lr) && self.lr > 0.0) {
            return Err(invalid(format!("adamw: the learning rate {} is not positive and finite", self.lr)));
        }
        if !(0.0..1.0).contains(&self.beta1) || !(0.0..1.0).contains(&self.beta2) {
            return Err(invalid("adamw: beta1 and beta2 lie in [0, 1)"));
        }
        if !(finite(self.eps) && self.eps > 0.0) {
            return Err(invalid(format!("adamw: eps {} is not positive and finite", self.eps)));
        }
        if [self.weight_decay, self.clip].iter().any(|&v| !finite(v) || v < 0.0) {
            return Err(invalid("adamw: weight decay and clip are finite and not negative"));
        }
        if self.total > 0 && self.warmup >= self.total {
            return Err(invalid("adamw: the warmup is shorter than the schedule"));
        }
        Ok(())
    }
}

impl Default for Settings {
    fn default() -> Self {
        Settings { lr: 1e-3, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.01, clip: 5.0, warmup: 0, total: 0 }
    }
}

#[derive(Debug, Clone)]
pub struct AdamW<R> {
    /// Checked by `new` and never changed after.
    settings: Settings,
    step: u64,
    m: Vec<Vec<R>>,
    v: Vec<Vec<R>>,
}

impl<R: Real> AdamW<R> {
    pub fn new(settings: Settings, shapes: &[usize]) -> Result<Self> {
        settings.check()?;
        let zeros = || shapes.iter().map(|&n| vec![R::ZERO; n]).collect();
        Ok(AdamW { settings, step: 0, m: zeros(), v: zeros() })
    }

    /// The settings, as checked when the optimizer was made.
    pub fn settings(&self) -> &Settings {
        &self.settings
    }

    /// The learning rate at step `t` (counted from 1): `t / warmup` of
    /// the rate while warming up, then `(total + 1 - t) / (total - warmup)`
    /// of it, so the first step after warmup takes the full rate and step
    /// `total` the smallest nonzero one. Past `total` it is zero.
    pub fn rate(&self, t: u64) -> f64 {
        let s = &self.settings;
        if s.warmup > 0 && t <= s.warmup {
            return s.lr * t as f64 / s.warmup as f64;
        }
        if s.total > s.warmup && t > s.warmup {
            // Steps left counting this one, without the overflow of
            // total + 1 at u64::MAX.
            let left = if t > s.total { 0.0 } else { (s.total - t) as f64 + 1.0 };
            return s.lr * (left / (s.total - s.warmup) as f64).min(1.0);
        }
        s.lr
    }

    /// One step: clips `grads` in place by their global norm, then moves
    /// `params`. `decay[i]` says whether tensor i takes weight decay.
    /// Returns the norm before clipping. Refuses tensors shaped other than
    /// the optimizer was made for, a nonfinite gradient, and an update
    /// whose moments or weights would not be finite once stored as `R`;
    /// a refusal changes no weight, gradient, moment or step count.
    pub fn step(&mut self, params: &mut [Vec<R>], grads: &mut [Vec<R>], decay: &[bool]) -> Result<f64> {
        let shaped = |v: &[Vec<R>]| v.len() == self.m.len() && v.iter().zip(&self.m).all(|(a, b)| a.len() == b.len());
        if !shaped(params) || !shaped(grads) || decay.len() != self.m.len() {
            return Err(invalid("adamw: the weights, gradients or decay flags do not match the optimizer"));
        }
        let s = self.settings;
        let norm = grads.iter().flatten().map(|g| g.to_f64() * g.to_f64()).sum::<f64>().sqrt();
        if !norm.is_finite() {
            return Err(invalid("adamw: a gradient is not finite"));
        }
        let scale = (s.clip > 0.0 && norm > s.clip).then(|| R::from_f64(s.clip / norm));
        let t = self.step + 1;
        let lr = self.rate(t);
        let (c1, c2) = (1.0 - s.beta1.powi(t as i32), 1.0 - s.beta2.powi(t as i32));
        // The new moments and weight of one value, as stored.
        let update = |p: R, g: R, m: R, v: R, decays: bool| {
            let g = scale.map_or(g, |c| g * c).to_f64();
            let mn = s.beta1 * m.to_f64() + (1.0 - s.beta1) * g;
            let vn = s.beta2 * v.to_f64() + (1.0 - s.beta2) * g * g;
            let mut x = p.to_f64();
            if decays {
                x -= lr * s.weight_decay * x;
            }
            x -= lr * (mn / c1) / ((vn / c2).sqrt() + s.eps);
            (R::from_f64(x), R::from_f64(mn), R::from_f64(vn))
        };
        // Checked in full before anything is written.
        for (i, ((p, g), (m, v))) in params.iter().zip(grads.iter()).zip(self.m.iter().zip(&self.v)).enumerate() {
            for (((&p, &g), &m), &v) in p.iter().zip(g).zip(m).zip(v) {
                let (x, mn, vn) = update(p, g, m, v, decay[i]);
                if !(x.is_finite() && mn.is_finite() && vn.is_finite()) {
                    return Err(invalid("adamw: the update or its moments do not fit the weights' type"));
                }
            }
        }
        for (i, ((p, g), (m, v))) in
            params.iter_mut().zip(grads.iter_mut()).zip(self.m.iter_mut().zip(&mut self.v)).enumerate()
        {
            for (((p, g), m), v) in p.iter_mut().zip(g.iter_mut()).zip(m.iter_mut()).zip(v.iter_mut()) {
                (*p, *m, *v) = update(*p, *g, *m, *v, decay[i]);
                if let Some(c) = scale {
                    *g *= c;
                }
            }
        }
        self.step = t;
        Ok(norm)
    }
}
