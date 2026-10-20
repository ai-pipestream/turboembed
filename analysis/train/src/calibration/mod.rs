//! Fit on held-out probability/target rows; use a disjoint second split
//! for conformal scores. The caller must also keep both out of model training.
mod fitting;
pub use fitting::{Method, fit_transform};
pub use turbo_calibration::{Calibration, DataIdentity, Identity, Policy, Predictor, Task, Transform};
use turbo_calibration::{Result, check_distribution, digest};

pub struct Samples<'a> {
    pub split: &'a str,
    pub probabilities: &'a [Vec<f64>],
    pub gold: &'a [usize],
}
impl Samples<'_> {
    pub fn check(&self, labels: usize) -> Result<()> {
        if self.split.is_empty() || self.probabilities.is_empty() || self.probabilities.len() != self.gold.len() {
            return Err("empty or inconsistent calibration samples".into());
        }
        for (p, &y) in self.probabilities.iter().zip(self.gold) {
            check_distribution(p)?;
            if p.len() != labels || y >= labels {
                return Err("calibration sample label dimension mismatch".into());
            }
        }
        Ok(())
    }
    pub fn data_identity(&self) -> DataIdentity {
        let mut bytes = Vec::new();
        for (p, &y) in self.probabilities.iter().zip(self.gold) {
            bytes.extend((p.len() as u64).to_le_bytes());
            bytes.extend((y as u64).to_le_bytes());
            for v in p {
                bytes.extend(v.to_le_bytes());
            }
        }
        DataIdentity { split: self.split.to_owned(), sha256: digest(&bytes) }
    }
}
/// Fit a probability transform, freeze it, then compute nonconformity on
/// the independent conformal split. Never optimize on conformal rows.
pub fn fit(identity: Identity, method: Method, fitting: Samples<'_>, conformal: Samples<'_>) -> Result<Calibration> {
    identity.check()?;
    fitting.check(identity.labels.len())?;
    conformal.check(identity.labels.len())?;
    let transform = fit_transform(method, &fitting)?;
    let mut scores = Vec::with_capacity(conformal.gold.len());
    let mut out = vec![0.0; identity.labels.len()];
    for (p, &y) in conformal.probabilities.iter().zip(conformal.gold) {
        transform.apply(p, &mut out)?;
        scores.push(1.0 - out[y]);
    }
    scores.sort_by(f64::total_cmp);
    Calibration::new(identity, transform, fitting.data_identity(), conformal.data_identity(), scores)
}
