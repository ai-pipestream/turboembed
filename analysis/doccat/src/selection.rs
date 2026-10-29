//! Sparse feature selection by mutual information or chi-squared ranking.
//!
//! Operates on native [`FeatureDocument`] records (`outcome`, `context`,
//! optional `values`) without depending on the classical maxent event stream.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

/// A labeled sparse-feature document for feature selection.
#[derive(Clone, Debug, PartialEq)]
pub struct FeatureDocument {
    outcome: String,
    context: Vec<String>,
    values: Option<Vec<f32>>,
}

impl FeatureDocument {
    /// Creates a binary-presence [`FeatureDocument`] from `outcome` and `features`.
    pub fn from_strs(outcome: &str, features: &[&str]) -> Self {
        Self { outcome: outcome.to_owned(), context: features.iter().map(|&f| f.to_owned()).collect(), values: None }
    }

    /// Creates a weighted [`FeatureDocument`] with optional feature `values`.
    pub fn with_values(
        outcome: impl Into<String>,
        context: Vec<String>,
        values: Option<Vec<f32>>,
    ) -> Result<Self, String> {
        if let Some(vs) = &values
            && vs.len() != context.len()
        {
            return Err(format!("values length {} != context length {}", vs.len(), context.len()));
        }
        Ok(Self { outcome: outcome.into(), context, values })
    }

    /// Outcome label.
    pub fn outcome(&self) -> &str {
        &self.outcome
    }

    /// Feature predicates.
    pub fn context(&self) -> &[String] {
        &self.context
    }

    /// Optional feature values aligned with [`Self::context`].
    pub fn values(&self) -> Option<&[f32]> {
        self.values.as_deref()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectionMetric {
    MutualInformation,
    Chi2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FeatureSelection {
    pub metric: SelectionMetric,
    pub top_k: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SelectedFeatures {
    kept: BTreeSet<String>,
    ranked: Vec<(String, f64)>,
}

impl FeatureSelection {
    /// Ranks every feature present in `docs` against the class distribution,
    /// returning the top `top_k` features.
    pub fn fit(&self, docs: &[FeatureDocument]) -> Result<SelectedFeatures, String> {
        if self.top_k == 0 {
            return Err("top_k must be positive".into());
        }
        if docs.is_empty() {
            return Err("feature selection requires at least one document".into());
        }
        let n = docs.len() as f64;
        let mut classes: BTreeMap<String, usize> = BTreeMap::new();
        let mut features: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
        for doc in docs {
            if let Some(vs) = doc.values()
                && vs.iter().any(|v| !v.is_finite() || *v < 0.0)
            {
                return Err("feature selection requires finite non-negative feature values".into());
            }
            *classes.entry(doc.outcome().to_owned()).or_default() += 1;
            let mut seen = BTreeSet::new();
            for (i, pred) in doc.context().iter().enumerate() {
                if doc.values().is_none_or(|vs| vs[i] > 0.0) && seen.insert(pred.clone()) {
                    *features.entry(pred.clone()).or_default().entry(doc.outcome().to_owned()).or_default() += 1;
                }
            }
        }
        if classes.len() < 2 {
            return Err("feature selection requires at least two classes".into());
        }
        if features.is_empty() {
            return Err("feature selection requires at least one active feature".into());
        }
        let mut ranked: Vec<(String, f64)> = features
            .into_iter()
            .map(|(f, by_class)| {
                let n1: usize = by_class.values().sum();
                let n1f = n1 as f64;
                let n0f = n - n1f;
                let mut score = 0.0;
                for (c, &nc) in &classes {
                    let n11 = *by_class.get(c).unwrap_or(&0) as f64;
                    let n10 = n1f - n11;
                    let n01 = nc as f64 - n11;
                    let n00 = n0f - n01;
                    let ncf = nc as f64;
                    let nc0f = n - ncf;
                    match self.metric {
                        SelectionMetric::MutualInformation => {
                            for (nij, ni, nj) in [(n11, n1f, ncf), (n10, n1f, nc0f), (n01, n0f, ncf), (n00, n0f, nc0f)]
                            {
                                if nij > 0.0 {
                                    score += (nij / n) * ((n * nij) / (ni * nj)).ln();
                                }
                            }
                        }
                        SelectionMetric::Chi2 => {
                            let num = n * (n11 * n00 - n10 * n01).powi(2);
                            let den = n1f * n0f * ncf * nc0f;
                            if den > 0.0 {
                                score += num / den;
                            }
                        }
                    }
                }
                if matches!(self.metric, SelectionMetric::MutualInformation | SelectionMetric::Chi2)
                    && classes.len() == 2
                {
                    score *= 0.5;
                }
                (f, score)
            })
            .collect();
        ranked.sort_by(|(fa, sa), (fb, sb)| match sb.total_cmp(sa) {
            Ordering::Equal => fa.encode_utf16().cmp(fb.encode_utf16()),
            ord => ord,
        });
        let kept = ranked.iter().take(self.top_k).map(|(f, _)| f.clone()).collect();
        Ok(SelectedFeatures { kept, ranked })
    }
}

impl SelectedFeatures {
    pub fn kept(&self) -> &BTreeSet<String> {
        &self.kept
    }

    pub fn ranked(&self) -> &[(String, f64)] {
        &self.ranked
    }

    /// Filters `docs` so each document retains only predicates in [`Self::kept`].
    pub fn filter(&self, docs: &[FeatureDocument]) -> Result<Vec<FeatureDocument>, String> {
        docs.iter()
            .map(|doc| {
                let mut context = Vec::new();
                let mut values = doc.values().map(|_| Vec::new());
                for (i, pred) in doc.context().iter().enumerate() {
                    if self.kept.contains(pred) {
                        context.push(pred.clone());
                        if let (Some(out), Some(vs)) = (&mut values, doc.values()) {
                            out.push(vs[i]);
                        }
                    }
                }
                FeatureDocument::with_values(doc.outcome().to_owned(), context, values)
            })
            .collect()
    }
}
