//! One label's One-vs-Rest head: a linear scorer (`bias` plus a weight
//! per document feature present) that the sigmoid turns into a score in
//! `0.0..=1.0`.

use std::collections::BTreeMap;

use super::bytes::MAX_UTF_BYTES;
use super::error::MultiLabelError;

/// The logistic sigmoid, into `0.0..=1.0`, stable for either sign.
pub(crate) fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// One label's One-vs-Rest head: a linear scorer (`bias` plus a weight
/// per document feature present) that the sigmoid turns into a score in
/// `0.0..=1.0`. The weights are the nonzero ones, sorted by feature name
/// (`LinearHead::new` enforces the strict order): the bytes a model
/// writes stay deterministic, and scoring binary-searches a feature's
/// weight in them.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearHead {
    pub(crate) bias: f64,
    pub(crate) weights: Vec<(String, f64)>,
}

impl LinearHead {
    /// Builds a head from its `bias` and `(feature, weight)` pairs.
    ///
    /// Returns [`MultiLabelError::Invalid`] when the bias or a weight is
    /// NaN or infinite, a feature name is empty, over `writeUTF`'s 65535
    /// bytes, or the names are not strictly ascending (so unique).
    pub fn new(bias: f64, weights: Vec<(String, f64)>) -> Result<LinearHead, MultiLabelError> {
        if !bias.is_finite() {
            return Err(MultiLabelError::Invalid(format!("a head bias of {bias}")));
        }
        for (i, (name, weight)) in weights.iter().enumerate() {
            if name.is_empty() {
                return Err(MultiLabelError::Invalid("a head with an empty feature name".into()));
            }
            if name.len() > MAX_UTF_BYTES {
                return Err(MultiLabelError::Invalid(format!(
                    "feature {name:?} is {0} bytes, over writeUTF's {MAX_UTF_BYTES}",
                    name.len()
                )));
            }
            if !weight.is_finite() {
                return Err(MultiLabelError::Invalid(format!("weight {weight} of feature {name:?}")));
            }
            if i > 0 && weights[i - 1].0.as_str() >= name.as_str() {
                return Err(MultiLabelError::Invalid(format!(
                    "head features not strictly ascending: {:?} then {name:?}",
                    weights[i - 1].0
                )));
            }
        }
        Ok(LinearHead { bias, weights })
    }

    /// The bias: the log-odds of a document with no weighted feature.
    pub fn bias(&self) -> f64 {
        self.bias
    }

    /// The nonzero weights, sorted by feature name.
    pub fn weights(&self) -> &[(String, f64)] {
        &self.weights
    }

    /// The linear score for a document whose feature counts are
    /// `features`, before the sigmoid: the bias plus every weight times
    /// its feature's count. The counts are the bag-of-features convention
    /// [`train`](crate::multilabel::train) fits the heads with, so a
    /// document scores the same at inference as in training: a repeated
    /// token counts once per occurrence. Each feature looks its weight up
    /// with a binary search over the head's sorted weights, so scoring
    /// costs `O(features * log weights)` per head, not `O(weights)`.
    pub fn score(&self, features: &BTreeMap<&str, f64>) -> f64 {
        let mut sum = self.bias;
        for (name, count) in features {
            if let Ok(i) = self.weights.binary_search_by(|(n, _)| n.as_str().cmp(name)) {
                sum += self.weights[i].1 * count;
            }
        }
        sum
    }

    /// The sigmoid score in `0.0..=1.0`.
    pub fn probability(&self, features: &BTreeMap<&str, f64>) -> f64 {
        sigmoid(self.score(features))
    }
}
