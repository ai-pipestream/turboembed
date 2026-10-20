//! Calibrated decisions alongside raw categories or token tags. Raw
//! producer layers are retained, including when the decision abstains.
use std::{marker::PhantomData, sync::Arc};
use turbo_analysis::{Anchor, Layer, LayerName};
use turbo_calibration::{Identity, Policy, Prediction, Predictor};

#[derive(Debug)]
pub struct Calibrated<L> {
    pub identity: Arc<Identity>,
    pub calibration_id: Arc<str>,
    pub policy: Policy,
    pub cutoff: f64,
    pub rows: Vec<Prediction>,
    pub _of: PhantomData<fn() -> L>,
}
impl<L: Layer> Calibrated<L> {
    pub fn new(predictor: &Predictor) -> Self {
        Self {
            identity: predictor.shared_identity(),
            calibration_id: predictor.shared_id(),
            policy: predictor.policy(),
            cutoff: predictor.cutoff(),
            rows: Vec::new(),
            _of: PhantomData,
        }
    }
}
impl<L: Layer> Layer for Calibrated<L> {
    const ID: &'static str = "turbo:calibrated";
    const OF: Option<&'static LayerName> = Some(L::NAME);
    fn anchor() -> Anchor {
        Anchor::aligned::<L>()
    }
    fn len(&self) -> usize {
        self.rows.len()
    }
    fn check(&self) -> Result<(), String> {
        self.identity.check()?;
        self.policy.check()?;
        if !(0.0..=1.0).contains(&self.cutoff) && self.cutoff != f64::INFINITY {
            return Err("invalid conformal cutoff".into());
        }
        if self.calibration_id.len() != 64
            || !self.calibration_id.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err("invalid calibration identity digest".into());
        }
        for row in &self.rows {
            row.check_policy(self.policy, self.cutoff)?;
            if row.raw.len() != self.identity.labels.len() {
                return Err("calibrated layer label dimension mismatch".into());
            }
        }
        Ok(())
    }
}

impl<L> Clone for Calibrated<L> {
    fn clone(&self) -> Self {
        Self {
            identity: self.identity.clone(),
            calibration_id: self.calibration_id.clone(),
            policy: self.policy,
            cutoff: self.cutoff,
            rows: self.rows.clone(),
            _of: PhantomData,
        }
    }
}
