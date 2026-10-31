//! Dynamic loss scaling (Micikevicius et al. 2018, arXiv 1710.03740):
//! the loss gradient is multiplied by a scale before backward, so the
//! small gradients a low-precision backward would flush to zero survive
//! it, and the gradients are divided by it again before the step. A step
//! whose gradients are not finite is skipped and the scale backed off;
//! after `interval` finite steps in a row it grows. Scaling by a power of
//! two commutes with rounding, so with the defaults a run that never
//! overflows gives the bytes an unscaled one gives. bfloat16 shares the
//! single's exponent, so under it the scale matters little; it is here
//! for the half-precision kernels a device may run, and the loop is the
//! same either way.

use turbo::status::{Result, invalid};

/// The scaler's settings: the first scale, the factor it grows by after
/// `interval` finite steps in a row, and the factor it backs off by on
/// a step whose gradients are not finite.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LossScale {
    pub initial: f32,
    pub growth: f32,
    pub backoff: f32,
    pub interval: u32,
}

impl Default for LossScale {
    fn default() -> Self {
        LossScale { initial: 65536.0, growth: 2.0, backoff: 0.5, interval: 2000 }
    }
}

impl LossScale {
    /// Refuses a scale that is not positive and finite, a growth below
    /// 1, a backoff outside (0, 1), or an interval of 0.
    pub fn check(&self) -> Result<()> {
        let fine = self.initial.is_finite()
            && self.initial > 0.0
            && self.growth.is_finite()
            && self.growth >= 1.0
            && self.backoff > 0.0
            && self.backoff < 1.0
            && self.interval > 0;
        if !fine {
            return Err(invalid(
                "trainer: loss scaling needs a positive finite initial scale, a growth of at least 1, a backoff in \
                 (0, 1) and an interval of at least 1",
            ));
        }
        Ok(())
    }
}

/// The scale through a run.
#[derive(Debug, Clone, PartialEq)]
pub struct Scaler {
    setting: LossScale,
    scale: f32,
    good: u32,
    skipped: usize,
}

impl Scaler {
    pub fn new(setting: LossScale) -> Scaler {
        Scaler { setting, scale: setting.initial, good: 0, skipped: 0 }
    }

    /// The scale the next step's loss gradient takes.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Steps skipped so far.
    pub fn skipped(&self) -> usize {
        self.skipped
    }

    /// Takes whether a step's unscaled gradients were all finite; returns
    /// whether to take the step. A step that is not backs the scale off
    /// and is skipped; `interval` finite steps in a row grow it.
    pub fn update(&mut self, finite: bool) -> bool {
        if !finite {
            self.scale *= self.setting.backoff;
            self.good = 0;
            self.skipped += 1;
            return false;
        }
        self.good += 1;
        if self.good >= self.setting.interval {
            let grown = self.scale * self.setting.growth;
            if grown.is_finite() {
                self.scale = grown;
            }
            self.good = 0;
        }
        true
    }
}
