//! Immutable, checked identity and decision policy shared by output batches.
use crate::{Identity, Policy, Result, probability};
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq)]
pub struct DecisionMetadata {
    identity: Arc<Identity>,
    id: Arc<str>,
    policy: Policy,
    cutoff: f64,
}
impl DecisionMetadata {
    pub fn new(identity: Arc<Identity>, id: Arc<str>, policy: Policy, cutoff: f64) -> Result<Self> {
        identity.check()?;
        policy.check()?;
        if (!probability(cutoff) && cutoff != f64::INFINITY)
            || id.len() != 64
            || !id.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err("invalid calibration digest or cutoff".into());
        }
        Ok(Self { identity, id, policy, cutoff })
    }
    pub fn identity(&self) -> &Identity {
        &self.identity
    }
    pub fn calibration_id(&self) -> &str {
        &self.id
    }
    pub fn policy(&self) -> Policy {
        self.policy
    }
    pub fn cutoff(&self) -> f64 {
        self.cutoff
    }
}
