use crate::{Calibration, Identity, Result, check_distribution, probability};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Abstention {
    EmptySet,
    SetTooLarge,
    BelowThreshold,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Policy {
    pub coverage: f64,
    pub max_set_size: usize,
    pub min_confidence: f64,
}
impl Policy {
    pub fn check(self) -> Result<()> {
        if !self.coverage.is_finite()
            || self.coverage <= 0.0
            || self.coverage >= 1.0
            || self.max_set_size == 0
            || !probability(self.min_confidence)
        {
            Err("invalid coverage, set size or abstention threshold".into())
        } else {
            Ok(())
        }
    }
}
#[derive(Clone, Debug, PartialEq)]
pub struct Prediction {
    pub raw: Vec<f64>,
    pub calibrated: Vec<f64>,
    /// Label indices in model order. Retained even when abstaining.
    pub set: Vec<usize>,
    pub abstention: Option<Abstention>,
}
impl Prediction {
    pub fn check(&self) -> Result<()> {
        check_distribution(&self.raw)?;
        check_distribution(&self.calibrated)?;
        if self.raw.len() != self.calibrated.len()
            || self.set.iter().any(|&i| i >= self.raw.len())
            || self.set.windows(2).any(|w| w[0] >= w[1])
        {
            return Err("invalid calibrated prediction columns".into());
        }
        if (self.set.is_empty()) != (self.abstention == Some(Abstention::EmptySet)) {
            return Err("empty prediction set reason mismatch".into());
        }
        Ok(())
    }
    /// Validate a persisted decision against its recorded policy and quantile.
    pub fn check_policy(&self, policy: Policy, cutoff: f64) -> Result<()> {
        self.check()?;
        policy.check()?;
        if !probability(cutoff) && cutoff != f64::INFINITY {
            return Err("invalid conformal cutoff".into());
        }
        let expected: Vec<_> =
            self.calibrated.iter().enumerate().filter_map(|(i, &p)| (1.0 - p <= cutoff).then_some(i)).collect();
        let reason = if expected.is_empty() {
            Some(Abstention::EmptySet)
        } else if expected.len() > policy.max_set_size {
            Some(Abstention::SetTooLarge)
        } else if self.calibrated.iter().copied().fold(0.0, f64::max) < policy.min_confidence {
            Some(Abstention::BelowThreshold)
        } else {
            None
        };
        if self.set != expected || self.abstention != reason {
            return Err("calibrated decision does not match policy".into());
        }
        Ok(())
    }
}
#[derive(Clone, Debug)]
pub struct Predictor {
    calibration: Calibration,
    policy: Policy,
    cutoff: f64,
    id: Arc<str>,
    identity: Arc<Identity>,
    hash: u64,
}
impl Predictor {
    pub fn new(calibration: Calibration, expected: &Identity, policy: Policy) -> Result<Self> {
        calibration.bind(expected)?;
        policy.check()?;
        let n = calibration.scores().len();
        // The extra rank is +infinity: all labels, never clamp to the last
        // observed score when the requested coverage exceeds n/(n+1).
        let rank = ((n as f64 + 1.0) * policy.coverage).ceil() as usize;
        let cutoff = if rank > n { f64::INFINITY } else { calibration.scores()[rank.max(1) - 1] };
        let id: Arc<str> = calibration.id().into();
        let identity = Arc::new(calibration.identity().clone());
        let mut bytes = calibration.to_bytes();
        bytes.extend(policy.coverage.to_le_bytes());
        bytes.extend((policy.max_set_size as u64).to_le_bytes());
        bytes.extend(policy.min_confidence.to_le_bytes());
        let hash = bytes.iter().fold(0xcbf29ce484222325, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x100000001b3));
        Ok(Self { calibration, policy, cutoff, id, identity, hash })
    }
    pub fn artifact(&self) -> &Calibration {
        &self.calibration
    }
    pub fn shared_identity(&self) -> Arc<Identity> {
        Arc::clone(&self.identity)
    }
    pub fn shared_id(&self) -> Arc<str> {
        Arc::clone(&self.id)
    }
    pub fn calibration_id(&self) -> &str {
        &self.id
    }
    pub fn policy(&self) -> Policy {
        self.policy
    }
    pub fn config_hash(&self) -> u64 {
        self.hash
    }
    pub fn cutoff(&self) -> f64 {
        self.cutoff
    }
    pub fn predict(&self, raw: &[f64]) -> Result<Prediction> {
        if raw.len() != self.calibration.identity().labels.len() {
            return Err("calibration label dimension mismatch".into());
        }
        let mut calibrated = vec![0.0; raw.len()];
        self.calibration.transform().apply_checked(raw, &mut calibrated)?;
        let set: Vec<_> =
            calibrated.iter().enumerate().filter_map(|(i, &p)| (1.0 - p <= self.cutoff).then_some(i)).collect();
        let abstention = if set.is_empty() {
            Some(Abstention::EmptySet)
        } else if set.len() > self.policy.max_set_size {
            Some(Abstention::SetTooLarge)
        } else if calibrated.iter().copied().fold(0.0, f64::max) < self.policy.min_confidence {
            Some(Abstention::BelowThreshold)
        } else {
            None
        };
        Ok(Prediction { raw: raw.to_vec(), calibrated, set, abstention })
    }
}
