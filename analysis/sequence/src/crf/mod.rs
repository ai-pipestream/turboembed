//! Globally normalized, first-order CRFs with BIO/BILOU constraints.
//! The same start, transition and end masks apply to training and decoding.
mod decode;
mod lattice;
mod model;
mod wire;
use crate::namefind::NameCodec;
pub use lattice::{CrfGradient, CrfLattice, CrfScores};
pub use model::{CrfSequenceModel, FeatureRow};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CrfError {
    Invalid(String),
    NonFinite,
    NoPath,
    Capacity,
}
impl std::fmt::Display for CrfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(s) => write!(f, "invalid CRF: {s}"),
            Self::NonFinite => f.write_str("non-finite CRF arithmetic"),
            Self::NoPath => f.write_str("no valid CRF path"),
            Self::Capacity => f.write_str("CRF size exceeds supported capacity"),
        }
    }
}
impl std::error::Error for CrfError {}
pub type Result<T> = std::result::Result<T, CrfError>;
pub(crate) fn invalid(s: impl Into<String>) -> CrfError {
    CrfError::Invalid(s.into())
}
pub(crate) fn finite(x: f64) -> Result<f64> {
    if x.is_finite() { Ok(x) } else { Err(CrfError::NonFinite) }
}

/// Immutable legality masks, including closed BILOU entities at sentence end.
#[derive(Clone, Debug)]
pub struct CrfConstraints {
    pub(crate) start: Vec<bool>,
    pub(crate) transition: Vec<bool>,
    pub(crate) end: Vec<bool>,
}
impl CrfConstraints {
    pub fn new(labels: &[String], codec: NameCodec) -> Result<Self> {
        if labels.is_empty() || labels.len() > 4096 {
            return Err(CrfError::Capacity);
        }
        let mut parsed = Vec::new();
        for (i, label) in labels.iter().enumerate() {
            if labels[..i].contains(label) {
                return Err(invalid("duplicate outcome"));
            }
            let part = if label == "other" {
                ("", "other")
            } else {
                let kind = crate::codec::name_type(label);
                let suffix = if kind.is_some() { label.rsplit_once('-').unwrap().1 } else { label.as_str() };
                if !matches!(suffix, "start" | "cont" | "last" | "unit")
                    || (codec == NameCodec::Bio && matches!(suffix, "last" | "unit"))
                {
                    return Err(invalid(format!("outcome {label:?} does not match codec")));
                }
                (kind.unwrap_or(""), suffix)
            };
            parsed.push(part);
        }
        let mut transition = Vec::with_capacity(labels.len() * labels.len());
        for &(pk, ps) in &parsed {
            for &(nk, ns) in &parsed {
                let allowed = if codec == NameCodec::Bio {
                    ns != "cont" || (pk == nk && matches!(ps, "start" | "cont"))
                } else if matches!(ps, "start" | "cont") {
                    pk == nk && matches!(ns, "cont" | "last")
                } else {
                    matches!(ns, "other" | "start" | "unit")
                };
                transition.push(allowed);
            }
        }
        Ok(Self {
            start: parsed.iter().map(|(_, s)| matches!(*s, "other" | "start" | "unit")).collect(),
            end: parsed
                .iter()
                .map(|(_, s)| codec == NameCodec::Bio || matches!(*s, "other" | "last" | "unit"))
                .collect(),
            transition,
        })
    }
    pub fn labels(&self) -> usize {
        self.start.len()
    }
    pub fn allows_start(&self, label: usize) -> bool {
        self.start.get(label).copied().unwrap_or(false)
    }
    pub fn allows_end(&self, label: usize) -> bool {
        self.end.get(label).copied().unwrap_or(false)
    }
    pub fn allows_transition(&self, from: usize, to: usize) -> bool {
        from < self.labels() && to < self.labels() && self.transition[from * self.labels() + to]
    }
    pub fn accepts(&self, path: &[usize]) -> bool {
        path.is_empty()
            || (self.allows_start(path[0])
                && self.allows_end(*path.last().unwrap())
                && path.windows(2).all(|w| self.allows_transition(w[0], w[1])))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CrfPath {
    pub labels: Vec<usize>,
    /// Log probability under the complete constrained lattice, not the top-k set.
    pub log_probability: f64,
    pub token_probabilities: Vec<f64>,
}
