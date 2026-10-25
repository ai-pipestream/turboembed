//! Probability calibration and split conformal prediction. Fitting lives
//! in turbo-train; applications bind artifacts to an exact model/task/domain.
mod codec;
mod metadata;
pub mod metrics;
mod prediction;
pub use metadata::DecisionMetadata;
mod transform;
pub use prediction::{Abstention, Policy, Prediction, Predictor};
use sha2::{Digest, Sha256};
pub use transform::{PreparedTransform, Transform};

pub type Result<T> = std::result::Result<T, String>;

pub fn digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    Sha256::digest(bytes)
        .iter()
        .flat_map(|b| [HEX[(b >> 4) as usize] as char, HEX[(b & 15) as usize] as char])
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Task {
    Categorization,
    TokenTagging,
    SpanTagging,
}

/// Label order, domain and inference configuration are part of identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Identity {
    pub model: String,
    pub revision: String,
    pub sha256: String,
    pub task: Task,
    pub domain: String,
    pub labels: Vec<String>,
    pub inference_hash: u64,
}
impl Identity {
    pub fn check(&self) -> Result<()> {
        if [&self.model, &self.revision, &self.sha256, &self.domain]
            .into_iter()
            .chain(self.labels.iter())
            .any(|s| s.len() > u32::MAX as usize)
        {
            return Err("calibration text too large".into());
        }
        if self.model.is_empty() || self.domain.is_empty() || !is_digest(&self.sha256) {
            return Err("calibration requires model, domain and lowercase SHA-256".into());
        }
        if self.labels.len() < 2 || self.labels.len() > 65536 || self.labels.iter().any(|s| s.is_empty()) {
            return Err("calibration requires 2..65536 nonempty labels".into());
        }
        let unique: std::collections::BTreeSet<_> = self.labels.iter().collect();
        if unique.len() != self.labels.len() {
            return Err("duplicate calibration label".into());
        }
        Ok(())
    }
}
fn is_digest(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Caller-named split and SHA-256 of the probability/target records used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DataIdentity {
    pub split: String,
    pub sha256: String,
}
impl DataIdentity {
    fn check(&self) -> Result<()> {
        if self.split.len() > u32::MAX as usize || self.split.is_empty() || !is_digest(&self.sha256) {
            Err("invalid calibration data identity".into())
        } else {
            Ok(())
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Calibration {
    identity: Identity,
    transform: Transform,
    fitted_on: DataIdentity,
    conformal_on: DataIdentity,
    nonconformity: Vec<f64>,
}
impl Calibration {
    pub fn new(
        identity: Identity,
        transform: Transform,
        fitted_on: DataIdentity,
        conformal_on: DataIdentity,
        nonconformity: Vec<f64>,
    ) -> Result<Self> {
        identity.check()?;
        transform.check()?;
        fitted_on.check()?;
        conformal_on.check()?;
        if fitted_on.split == conformal_on.split || fitted_on.sha256 == conformal_on.sha256 {
            return Err("calibration fitting and conformal splits must differ".into());
        }
        if nonconformity.len() > u32::MAX as usize
            || nonconformity.is_empty()
            || nonconformity.iter().any(|&p| !probability(p))
            || nonconformity.windows(2).any(|w| w[0] > w[1])
        {
            return Err("conformal scores must be nonempty, sorted and in [0,1]".into());
        }
        Ok(Self { identity, transform, fitted_on, conformal_on, nonconformity })
    }
    pub fn identity(&self) -> &Identity {
        &self.identity
    }
    pub fn transform(&self) -> &Transform {
        &self.transform
    }
    pub fn fitted_on(&self) -> &DataIdentity {
        &self.fitted_on
    }
    pub fn conformal_on(&self) -> &DataIdentity {
        &self.conformal_on
    }
    pub fn scores(&self) -> &[f64] {
        &self.nonconformity
    }
    pub fn id(&self) -> String {
        digest(&self.to_bytes())
    }
    pub fn bind(&self, expected: &Identity) -> Result<()> {
        expected.check()?;
        if &self.identity != expected {
            return Err(format!(
                "calibration identity mismatch: artifact {} model {} domain {}; expected model {} domain {}",
                self.id(),
                self.identity.model,
                self.identity.domain,
                expected.model,
                expected.domain
            ));
        }
        Ok(())
    }
}
#[inline]
pub fn probability(p: f64) -> bool {
    p.is_finite() && (0.0..=1.0).contains(&p)
}
#[inline]
pub fn check_distribution(p: &[f64]) -> Result<()> {
    if p.len() < 2 || p.iter().any(|&v| !probability(v)) || (p.iter().sum::<f64>() - 1.0).abs() > 1e-6 {
        return Err("expected finite probability distribution summing to one".into());
    }
    Ok(())
}
