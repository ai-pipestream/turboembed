use super::{CrfConstraints, CrfError, CrfLattice, CrfPath, CrfScores, Result, finite, invalid};
use crate::namefind::NameCodec;
use std::collections::HashMap;
use turbo_maxent::JavaString;

/// Sparse binary or signed numeric features for one token. Duplicate
/// features are additive, matching the existing feature-generator contract.
pub type FeatureRow = Vec<(JavaString, f64)>;

#[derive(Clone, Debug)]
pub struct CrfSequenceModel {
    pub(crate) labels: Vec<String>,
    pub(crate) features: Vec<JavaString>,
    pub(crate) index: HashMap<JavaString, usize>,
    pub(crate) codec: NameCodec,
    pub(crate) weights: Vec<f64>,
    pub(crate) mask: CrfConstraints,
}
impl CrfSequenceModel {
    /// Feature-major emissions, then previous-label-major transitions,
    /// start weights, and end weights. All parameters must be finite.
    pub fn new(labels: Vec<String>, features: Vec<JavaString>, codec: NameCodec, weights: Vec<f64>) -> Result<Self> {
        if labels.iter().any(|s| s.len() > u32::MAX as usize)
            || features.iter().any(|s| s.as_bytes().len() > u32::MAX as usize)
        {
            return Err(CrfError::Capacity);
        }
        let mask = CrfConstraints::new(&labels, codec)?;
        let size = Self::parameter_count(labels.len(), features.len())?;
        if weights.len() != size {
            return Err(invalid("parameter count"));
        }
        for &x in &weights {
            finite(x)?;
        }
        let index: HashMap<_, _> = features.iter().cloned().enumerate().map(|(i, s)| (s, i)).collect();
        if index.len() != features.len() {
            return Err(invalid("duplicate attribute"));
        }
        Ok(Self { labels, features, index, codec, weights, mask })
    }
    pub fn parameter_count(labels: usize, features: usize) -> Result<usize> {
        if labels == 0 || labels > 4096 || features > 2_000_000 {
            return Err(CrfError::Capacity);
        }
        features
            .checked_add(labels)
            .and_then(|x| x.checked_add(2))
            .and_then(|x| x.checked_mul(labels))
            .filter(|&x| x <= 100_000_000)
            .ok_or(CrfError::Capacity)
    }
    pub fn labels(&self) -> &[String] {
        &self.labels
    }
    pub fn features(&self) -> &[JavaString] {
        &self.features
    }
    pub fn parameters(&self) -> &[f64] {
        &self.weights
    }
    pub fn codec(&self) -> NameCodec {
        self.codec
    }
    pub fn constraints(&self) -> &CrfConstraints {
        &self.mask
    }
    pub fn lattice(&self, rows: &[FeatureRow]) -> Result<CrfLattice> {
        let n = self.labels.len();
        if rows.len().checked_mul(n).is_none_or(|x| x > 16_000_000) {
            return Err(CrfError::Capacity);
        }
        let mut emissions = vec![vec![0.0; n]; rows.len()];
        for (row, out) in rows.iter().zip(&mut emissions) {
            for (feature, value) in row {
                finite(*value)?;
                if let Some(&a) = self.index.get(feature) {
                    for (y, v) in out.iter_mut().enumerate() {
                        *v = finite(*v + value * self.weights[a * n + y])?;
                    }
                }
            }
        }
        let at = self.features.len() * n;
        CrfLattice::new(
            CrfScores {
                emissions: &emissions,
                transitions: &self.weights[at..at + n * n],
                start: &self.weights[at + n * n..at + n * n + n],
                end: &self.weights[at + n * n + n..],
            },
            &self.mask,
        )
    }
    pub fn best_sequences(&self, rows: &[FeatureRow], k: usize) -> Result<Vec<CrfPath>> {
        self.lattice(rows)?.best_paths(k)
    }
    pub fn config_hash(&self) -> u64 {
        turbo_analysis::fnv1a(&self.to_bytes())
    }
}
