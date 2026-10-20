use crate::{Result, check_distribution, probability};

/// Temperature acts on log probabilities. Platt acts on binary log odds
/// independently per class, then renormalizes. Isotonic is right-closed
/// piecewise constant, with endpoint clamping, then renormalization.
#[derive(Clone, Debug, PartialEq)]
pub enum Transform {
    Temperature(f64),
    Platt { slope: f64, intercept: f64 },
    Isotonic { upper: Vec<f64>, value: Vec<f64> },
}
impl Transform {
    pub fn check(&self) -> Result<()> {
        let valid = match self {
            Self::Temperature(t) => t.is_finite() && *t >= 1e-6 && *t <= 1e6,
            Self::Platt { slope, intercept } => {
                slope.is_finite() && (0.0..=1e6).contains(slope) && intercept.is_finite() && intercept.abs() <= 1e6
            }
            Self::Isotonic { upper, value } => {
                upper.len() <= u32::MAX as usize
                    && !upper.is_empty()
                    && upper.len() == value.len()
                    && upper.iter().chain(value).all(|&p| probability(p))
                    && upper.windows(2).all(|w| w[0] < w[1])
                    && value.windows(2).all(|w| w[0] <= w[1])
            }
        };
        if valid { Ok(()) } else { Err("invalid calibration transform parameters".into()) }
    }
    pub fn apply(&self, raw: &[f64], out: &mut [f64]) -> Result<()> {
        self.check()?;
        self.apply_checked(raw, out)
    }
    pub(crate) fn apply_checked(&self, raw: &[f64], out: &mut [f64]) -> Result<()> {
        check_distribution(raw)?;
        if out.len() != raw.len() {
            return Err("calibration output dimension mismatch".into());
        }
        match self {
            Self::Temperature(t) => {
                let max = raw.iter().copied().fold(0.0, f64::max).ln();
                for (o, &p) in out.iter_mut().zip(raw) {
                    *o = if p == 0.0 { 0.0 } else { ((p.ln() - max) / t).exp() };
                }
            }
            Self::Platt { slope, intercept } => {
                for (o, &p) in out.iter_mut().zip(raw) {
                    *o = sigmoid(slope * logit(p) + intercept).max(f64::MIN_POSITIVE);
                }
            }
            Self::Isotonic { upper, value } => {
                for (o, &p) in out.iter_mut().zip(raw) {
                    let i = upper.partition_point(|&x| x < p).min(upper.len() - 1);
                    *o = value[i].max(f64::MIN_POSITIVE);
                }
            }
        }
        let sum: f64 = out.iter().sum();
        for v in out {
            *v /= sum;
        }
        Ok(())
    }
}
pub fn logit(p: f64) -> f64 {
    p.clamp(1e-15, 1.0 - 1e-15).ln() - (-p.clamp(1e-15, 1.0 - 1e-15)).ln_1p()
}
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}
