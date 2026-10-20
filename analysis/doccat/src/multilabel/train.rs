//! Training a multi-label model: one binary logistic-regression head per
//! label over the document features, fitted one-vs-rest with full-batch
//! gradient descent, deterministically (a fixed iteration count over the
//! samples in their given order).

use std::collections::{BTreeMap, BTreeSet};

use super::error::MultiLabelError;
use super::model::{DEFAULT_THRESHOLD, LinearHead, MultiLabelDoccatModel, sigmoid};
use crate::DoccatFactory;

/// One training document: its category labels (zero or more) and its
/// tokens. Labels the union of the samples holds become the model's
/// labels, sorted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiLabelSample {
    labels: BTreeSet<String>,
    tokens: Vec<String>,
}

impl MultiLabelSample {
    /// A document of `labels` and `tokens` (either may be empty; a sample
    /// with no labels still trains every head against).
    pub fn new(
        labels: impl IntoIterator<Item = impl Into<String>>,
        tokens: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        MultiLabelSample {
            labels: labels.into_iter().map(Into::into).collect(),
            tokens: tokens.into_iter().map(Into::into).collect(),
        }
    }

    /// The document's labels.
    pub fn labels(&self) -> &BTreeSet<String> {
        &self.labels
    }

    /// The document's tokens.
    pub fn tokens(&self) -> &[String] {
        &self.tokens
    }
}

/// How [`train_multilabel`] fits the heads: full-batch gradient descent
/// over the binary cross-entropy, `iterations` passes, `learning_rate`
/// as the step size, `l2` as the weight decay (the bias is not decayed).
#[derive(Clone, Debug, PartialEq)]
pub struct MultiLabelTrainingParameters {
    pub iterations: usize,
    pub learning_rate: f64,
    pub l2: f64,
}

impl Default for MultiLabelTrainingParameters {
    /// 200 iterations, a 0.5 learning rate, and a small 1e-4 weight decay.
    fn default() -> Self {
        MultiLabelTrainingParameters { iterations: 200, learning_rate: 0.5, l2: 1e-4 }
    }
}

/// Trains a [`MultiLabelDoccatModel`] on `samples` with `factory`'s
/// feature generators: every label gets a [`LinearHead`] fit one-vs-rest
/// (its documents against all the others), every threshold starts at
/// [`DEFAULT_THRESHOLD`] for [`ThresholdCalibrator`](super::ThresholdCalibrator)
/// to move.
///
/// Returns [`MultiLabelError::Train`] when there are no samples or the
/// samples hold no label at all, and [`MultiLabelError::Invalid`] when
/// the parameters are NaN, infinite, or negative.
pub fn train_multilabel(
    language: &str,
    samples: &[MultiLabelSample],
    factory: &DoccatFactory,
    params: &MultiLabelTrainingParameters,
) -> Result<MultiLabelDoccatModel, MultiLabelError> {
    if samples.is_empty() {
        return Err(MultiLabelError::Train("no training samples".into()));
    }
    if !params.learning_rate.is_finite() || params.learning_rate <= 0.0 {
        return Err(MultiLabelError::Invalid(format!(
            "learning_rate {}, must be positive and finite",
            params.learning_rate
        )));
    }
    if !params.l2.is_finite() || params.l2 < 0.0 {
        return Err(MultiLabelError::Invalid(format!("l2 {}, must be non-negative and finite", params.l2)));
    }
    let labels: BTreeSet<&str> = samples.iter().flat_map(|s| s.labels().iter().map(String::as_str)).collect();
    if labels.is_empty() {
        return Err(MultiLabelError::Train("the samples hold no label".into()));
    }

    // The documents' features, in the samples' order, over the shared
    // vocabulary the distinct features sort into (deterministic head
    // bytes and gradient sums).
    let extra = BTreeMap::new();
    let mut scratch = Vec::new();
    let mut docs: Vec<Vec<(u32, f64)>> = Vec::with_capacity(samples.len());
    let mut vocab: BTreeSet<String> = BTreeSet::new();
    let mut per_doc: Vec<Vec<String>> = Vec::with_capacity(samples.len());
    for sample in samples {
        scratch.clear();
        let tokens: Vec<&str> = sample.tokens.iter().map(String::as_str).collect();
        factory.context_generator().extract_features(&tokens, &extra, &mut scratch);
        vocab.extend(scratch.iter().cloned());
        per_doc.push(scratch.clone());
    }
    let vocab: Vec<String> = vocab.into_iter().collect();
    let index: BTreeMap<&str, u32> = vocab.iter().enumerate().map(|(i, f)| (f.as_str(), i as u32)).collect();
    for features in &per_doc {
        let mut counts: BTreeMap<u32, f64> = BTreeMap::new();
        for f in features {
            *counts.entry(index[f.as_str()]).or_default() += 1.0;
        }
        docs.push(counts.into_iter().collect());
    }

    let n = docs.len() as f64;
    let heads = labels
        .iter()
        .map(|&label| {
            let target: Vec<f64> = samples.iter().map(|s| f64::from(u8::from(s.labels.contains(label)))).collect();
            fit_head(&vocab, &docs, &target, n, params)
        })
        .collect::<Vec<LinearHead>>();

    let names: Vec<&str> = labels.into_iter().collect();
    let k = names.len();
    MultiLabelDoccatModel::new(language, &names, heads, vec![DEFAULT_THRESHOLD; k], factory.context_generator().clone())
}

/// Full-batch gradient descent for one label: weights over the shared
/// vocabulary, a bias, and the mean cross-entropy gradient, stepped
/// `params.iterations` times.
fn fit_head(
    vocab: &[String],
    docs: &[Vec<(u32, f64)>],
    target: &[f64],
    n: f64,
    params: &MultiLabelTrainingParameters,
) -> LinearHead {
    let mut w = vec![0.0; vocab.len()];
    let mut bias = 0.0;
    for _ in 0..params.iterations {
        let mut grad = vec![0.0; w.len()];
        let mut grad_b = 0.0;
        for (x, &y) in docs.iter().zip(target) {
            let z: f64 = bias + x.iter().map(|&(j, v)| w[j as usize] * v).sum::<f64>();
            let err = sigmoid(z) - y;
            grad_b += err;
            for &(j, v) in x {
                grad[j as usize] += err * v;
            }
        }
        for (wj, gj) in w.iter_mut().zip(&mut grad) {
            *wj -= params.learning_rate * (*gj / n + params.l2 * *wj);
        }
        bias -= params.learning_rate * grad_b / n;
    }
    let weights = w.into_iter().enumerate().filter(|(_, w)| *w != 0.0).map(|(j, w)| (vocab[j].clone(), w)).collect();
    LinearHead::new(bias, weights).expect("trained weights are finite and the vocabulary is sorted")
}
