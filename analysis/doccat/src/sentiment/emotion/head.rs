//! A learned multi-label head over sentence or chunk vectors: for each of
//! the 28 `EmotionDimension`s, `sigmoid(w · v + b)`.
//!
//! The weights come from the caller and travel in a small binary format
//! ([`EmotionVectorHead::to_bytes`]): the magic `TEVH`, the version 1, the
//! dimension and flags (bit 1 an encoder; all `u32` little-endian), the 28
//! biases and the `[28, dim]` weights as little-endian `f32`, rows in
//! `EmotionDimension::ALL` order, then, for a head bound to an encoder,
//! the encoder's id, revision and SHA-256, each a `u32` length and UTF-8
//! bytes. Reading checks every part, so bad bytes are an error and never
//! a panic, and a head read back keeps its encoder.

use std::fmt;

use turbo_analysis::ModelRef;
use turbo_analysis_layers::{EMOTION_DIMENSIONS, EmotionScores};

const MAGIC: &[u8; 4] = b"TEVH";
const VERSION: u32 = 1;
const HEADER: usize = 16;
const ENCODER: u32 = 2;
const MAX_STRING: usize = 4096;
const N: usize = EMOTION_DIMENSIONS;

/// A multi-label emotion head over one vector per unit.
pub trait EmotionVectorScorer: Send + Sync {
    /// The 28 scores of `vec`, each in `0.0..=1.0`.
    fn score_emotions(&self, vec: &[f32]) -> Result<EmotionScores, String>;

    /// The encoder the vectors must come from, when the head names one.
    fn model(&self) -> Option<ModelRef> {
        None
    }

    /// A hash of the head's configuration, for provenance.
    fn config_hash(&self) -> u64 {
        0
    }
}

/// Why a head could not be made or read.
#[derive(Clone, Debug, PartialEq)]
pub enum EmotionHeadError {
    /// Not a head of this format: magic, version, flags or length.
    Format(String),
    /// A dimension of 0, a matrix of the wrong size, or a value that is not
    /// finite.
    Value(String),
}

impl fmt::Display for EmotionHeadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EmotionHeadError::Format(m) => write!(f, "emotion head bytes: {m}"),
            EmotionHeadError::Value(m) => write!(f, "emotion head: {m}"),
        }
    }
}

impl std::error::Error for EmotionHeadError {}

/// See the module comment.
#[derive(Clone, Debug, PartialEq)]
pub struct EmotionVectorHead {
    dim: usize,
    /// `[28, dim]`.
    weights: Vec<f32>,
    bias: [f32; N],
    model: Option<ModelRef>,
}

impl EmotionVectorHead {
    /// A head over vectors of `dim` values: `weights` is `[28, dim]`.
    pub fn new(dim: usize, weights: Vec<f32>, bias: [f32; N]) -> Result<Self, EmotionHeadError> {
        if dim == 0 || dim > u32::MAX as usize {
            return Err(EmotionHeadError::Value(format!("a dimension of {dim}")));
        }
        if weights.len() != N * dim {
            return Err(EmotionHeadError::Value(format!("{} weights, expected 28 x {dim}", weights.len())));
        }
        if let Some(i) = weights.iter().chain(&bias).position(|v| !v.is_finite()) {
            return Err(EmotionHeadError::Value(format!("value {i} is not finite")));
        }
        Ok(EmotionVectorHead { dim, weights, bias, model: None })
    }

    /// The encoder the vectors must come from.
    pub fn with_model(mut self, model: ModelRef) -> Self {
        self.model = Some(model);
        self
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The 28 probabilities of `vector`.
    pub fn score(&self, vector: &[f32]) -> Result<EmotionScores, String> {
        if vector.len() != self.dim {
            return Err(format!("a vector of {} values, the head reads {}", vector.len(), self.dim));
        }
        if vector.iter().any(|x| !x.is_finite()) {
            return Err("a vector value that is not finite".into());
        }
        let mut out = EmotionScores::default();
        for (l, s) in out.0.iter_mut().enumerate() {
            let row = &self.weights[l * self.dim..(l + 1) * self.dim];
            let z = f64::from(self.bias[l])
                + row.iter().zip(vector).map(|(&w, &x)| f64::from(w) * f64::from(x)).sum::<f64>();
            *s = if z >= 0.0 { 1.0 / (1.0 + (-z).exp()) } else { z.exp() / (1.0 + z.exp()) };
        }
        Ok(out)
    }

    /// The head in the format of the module comment.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER + 4 * N * (1 + self.dim));
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&(self.dim as u32).to_le_bytes());
        out.extend_from_slice(&(if self.model.is_some() { ENCODER } else { 0 }).to_le_bytes());
        for v in self.bias.iter().chain(&self.weights) {
            out.extend_from_slice(&v.to_le_bytes());
        }
        if let Some(m) = &self.model {
            for text in [&m.id, &m.revision, &m.sha256] {
                out.extend_from_slice(&(text.len() as u32).to_le_bytes());
                out.extend_from_slice(text.as_bytes());
            }
        }
        out
    }

    /// A head from [`EmotionVectorHead::to_bytes`]'s format, every part
    /// checked.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, EmotionHeadError> {
        let bad = |m: String| EmotionHeadError::Format(m);
        if bytes.len() < HEADER {
            return Err(bad(format!("{} bytes, shorter than the {HEADER}-byte header", bytes.len())));
        }
        if &bytes[..4] != MAGIC {
            return Err(bad("not an emotion head (magic)".into()));
        }
        let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"));
        if word(4) != VERSION {
            return Err(bad(format!("version {}, this reads {VERSION}", word(4))));
        }
        let (dim, flags) = (word(8) as usize, word(12));
        if flags & !ENCODER != 0 {
            return Err(bad(format!("flags {flags}")));
        }
        if dim == 0 {
            return Err(EmotionHeadError::Value("a dimension of 0".into()));
        }
        let end = (N as u64 * (1 + dim as u64))
            .checked_mul(4)
            .and_then(|n| n.checked_add(HEADER as u64))
            .filter(|&n| n <= bytes.len() as u64)
            .ok_or_else(|| bad(format!("{} bytes, too few for a head of dimension {dim}", bytes.len())))?
            as usize;
        let floats: Vec<f32> = bytes[HEADER..end].as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
        let bias: [f32; N] = floats[..N].try_into().expect("28 biases");
        let mut head = EmotionVectorHead::new(dim, floats[N..].to_vec(), bias)?;
        let mut rest = &bytes[end..];
        if flags & ENCODER != 0 {
            let mut text = || -> Result<String, EmotionHeadError> {
                let len = rest.get(..4).map(|b| u32::from_le_bytes(b.try_into().expect("four bytes")) as usize);
                let Some(len) = len.filter(|&l| l <= MAX_STRING && 4 + l <= rest.len()) else {
                    return Err(bad("an encoder string past the end or too long".into()));
                };
                let s = std::str::from_utf8(&rest[4..4 + len])
                    .map_err(|_| bad("an encoder string that is not UTF-8".into()))?;
                rest = &rest[4 + len..];
                Ok(s.to_owned())
            };
            let (id, revision, sha256) = (text()?, text()?, text()?);
            head = head.with_model(ModelRef { id, revision, sha256 });
        }
        if !rest.is_empty() {
            return Err(bad(format!("{} bytes past the end of the head", rest.len())));
        }
        Ok(head)
    }
}

impl EmotionVectorScorer for EmotionVectorHead {
    fn score_emotions(&self, vec: &[f32]) -> Result<EmotionScores, String> {
        self.score(vec)
    }

    fn model(&self) -> Option<ModelRef> {
        self.model.clone()
    }

    fn config_hash(&self) -> u64 {
        turbo_analysis::fnv1a(&self.to_bytes())
    }
}
