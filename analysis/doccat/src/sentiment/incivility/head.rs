//! A learned multi-label head over sentence or chunk vectors: for each of
//! the six [`IncivilityLabel`]s, `sigmoid(w · v + b)`, with an optional
//! context matrix that adds `w_ctx · p` for a context vector `p` (the
//! preceding unit's vector, or the vector of the post a reply answers).
//!
//! The weights come from the caller, trained elsewhere or with
//! [`IncivilityVectorHead::fit`], and travel in a small binary format
//! ([`IncivilityVectorHead::to_bytes`]): the magic `TIVH`, a version, the
//! dimension, a flag for the context matrix, then the biases, the label
//! matrix and the context matrix as little-endian `f32`, rows in
//! [`IncivilityLabel::ALL`] order. Reading checks every part, so bad
//! bytes are an error and never a panic.

use std::fmt;

use turbo_analysis::ModelRef;
use turbo_analysis_layers::{IncivilityLabel, IncivilityScores};

use super::IncivilityVectorScorer;

const MAGIC: &[u8; 4] = b"TIVH";
const VERSION: u32 = 1;
const LABELS: usize = IncivilityLabel::ALL.len();
const HEADER: usize = 16;

/// Why a head could not be made or read.
#[derive(Clone, Debug, PartialEq)]
pub enum IncivilityHeadError {
    /// The bytes are not a head of this format: magic, version, flags or
    /// length.
    Format(String),
    /// A dimension of 0, a matrix of the wrong size, or a weight that is
    /// not finite.
    Value(String),
}

impl fmt::Display for IncivilityHeadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IncivilityHeadError::Format(m) => write!(f, "incivility head bytes: {m}"),
            IncivilityHeadError::Value(m) => write!(f, "incivility head: {m}"),
        }
    }
}

impl std::error::Error for IncivilityHeadError {}

/// See the module comment.
#[derive(Clone, Debug, PartialEq)]
pub struct IncivilityVectorHead {
    dim: usize,
    /// `[6, dim]`, row `l` for `IncivilityLabel::ALL[l]`.
    weights: Vec<f32>,
    bias: [f32; LABELS],
    /// `[6, dim]` when the head reads a context vector.
    context: Option<Vec<f32>>,
    model: Option<ModelRef>,
}

fn finite(what: &str, values: &[f32]) -> Result<(), IncivilityHeadError> {
    match values.iter().position(|v| !v.is_finite()) {
        Some(i) => Err(IncivilityHeadError::Value(format!("{what}[{i}] is {}", values[i]))),
        None => Ok(()),
    }
}

impl IncivilityVectorHead {
    /// A head over vectors of `dim` values: `weights` is `[6, dim]`.
    pub fn new(dim: usize, weights: Vec<f32>, bias: [f32; LABELS]) -> Result<Self, IncivilityHeadError> {
        if dim == 0 || dim > u32::MAX as usize {
            return Err(IncivilityHeadError::Value(format!("a dimension of {dim}")));
        }
        if weights.len() != LABELS * dim {
            return Err(IncivilityHeadError::Value(format!("{} weights, expected 6 x {dim}", weights.len())));
        }
        finite("weights", &weights)?;
        finite("bias", &bias)?;
        Ok(IncivilityVectorHead { dim, weights, bias, context: None, model: None })
    }

    /// The same head, reading a context vector through `context`, `[6, dim]`.
    pub fn with_context(mut self, context: Vec<f32>) -> Result<Self, IncivilityHeadError> {
        if context.len() != LABELS * self.dim {
            return Err(IncivilityHeadError::Value(format!(
                "{} context weights, expected 6 x {}",
                context.len(),
                self.dim
            )));
        }
        finite("context", &context)?;
        self.context = Some(context);
        Ok(self)
    }

    /// The encoder the vectors must come from; the annotator refuses
    /// vectors of any other.
    pub fn with_model(mut self, model: ModelRef) -> Self {
        self.model = Some(model);
        self
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn has_context(&self) -> bool {
        self.context.is_some()
    }

    /// The six probabilities for `vector`, with `context` read through the
    /// context matrix when the head has one; a head without one ignores it.
    pub fn score(&self, vector: &[f32], context: Option<&[f32]>) -> Result<IncivilityScores, String> {
        let check = |what: &str, v: &[f32]| {
            if v.len() != self.dim {
                return Err(format!("a {what} of {} values, the head reads {}", v.len(), self.dim));
            }
            if v.iter().any(|x| !x.is_finite()) {
                return Err(format!("a {what} value that is not finite"));
            }
            Ok(())
        };
        check("vector", vector)?;
        if let Some(p) = context {
            check("context vector", p)?;
        }
        let mut out = IncivilityScores::default();
        for (l, label) in IncivilityLabel::ALL.into_iter().enumerate() {
            let mut logit = f64::from(self.bias[l]) + dot(&self.weights[l * self.dim..(l + 1) * self.dim], vector);
            if let (Some(w), Some(p)) = (&self.context, context) {
                logit += dot(&w[l * self.dim..(l + 1) * self.dim], p);
            }
            out.set(label, sigmoid(logit));
        }
        Ok(out)
    }

    /// The head in the binary format of the module comment.
    pub fn to_bytes(&self) -> Vec<u8> {
        let n = LABELS * self.dim;
        let mut out = Vec::with_capacity(HEADER + 4 * (LABELS + n * (1 + usize::from(self.has_context()))));
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&(self.dim as u32).to_le_bytes());
        out.extend_from_slice(&u32::from(self.has_context()).to_le_bytes());
        let floats = self.bias.iter().chain(&self.weights).chain(self.context.iter().flatten());
        for v in floats {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// A head from [`IncivilityVectorHead::to_bytes`]'s format, every part
    /// checked: magic, version, flags, a dimension above 0, the exact
    /// length and finite values.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, IncivilityHeadError> {
        let bad = |m: String| Err(IncivilityHeadError::Format(m));
        if bytes.len() < HEADER {
            return bad(format!("{} bytes, shorter than the {HEADER}-byte header", bytes.len()));
        }
        if &bytes[..4] != MAGIC {
            return bad("not an incivility head (magic)".into());
        }
        let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"));
        if word(4) != VERSION {
            return bad(format!("version {}, this reads {VERSION}", word(4)));
        }
        let dim = word(8) as usize;
        let flags = word(12);
        if flags > 1 {
            return bad(format!("flags {flags}"));
        }
        if dim == 0 {
            return Err(IncivilityHeadError::Value("a dimension of 0".into()));
        }
        let matrices = 1 + flags as usize;
        let want = (LABELS as u64 + LABELS as u64 * dim as u64 * matrices as u64)
            .checked_mul(4)
            .and_then(|n| n.checked_add(HEADER as u64));
        if want != Some(bytes.len() as u64) {
            return bad(format!("{} bytes, a head of dimension {dim} has {want:?}", bytes.len()));
        }
        let floats: Vec<f32> = bytes[HEADER..].as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
        let n = LABELS * dim;
        let bias: [f32; LABELS] = floats[..LABELS].try_into().expect("six biases");
        let head = IncivilityVectorHead::new(dim, floats[LABELS..LABELS + n].to_vec(), bias)?;
        if flags == 1 { head.with_context(floats[LABELS + n..].to_vec()) } else { Ok(head) }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| f64::from(x) * f64::from(y)).sum()
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

impl IncivilityVectorScorer for IncivilityVectorHead {
    fn score_incivility(&self, vec: &[f32]) -> Result<IncivilityScores, String> {
        self.score(vec, None)
    }

    fn score_incivility_in_context(&self, vec: &[f32], context: Option<&[f32]>) -> Result<IncivilityScores, String> {
        self.score(vec, context)
    }

    fn model(&self) -> Option<ModelRef> {
        self.model.clone()
    }

    fn config_hash(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in self.to_bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
        h
    }
}

#[path = "fit.rs"]
mod fit;
pub use fit::IncivilityFitOptions;
