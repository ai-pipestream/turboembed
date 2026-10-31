//! Three-class (`positive`, `negative`, `neutral`) dense linear head for
//! sentence and document polarity classification.

use std::hash::{Hash, Hasher};

use turbo_analysis::ModelRef;
use turbo_analysis_layers::{Polarity, PolarityScores};
use turbo_sha256::sha256;

use crate::scorers::PolarityVectorScorer;

/// Number of coarse polarity classes (`positive`, `negative`, `neutral`).
pub const POLARITY_COUNT: usize = 3;

/// Canonical polarity label names in tag order (`Tag(0)`, `Tag(1)`, `Tag(2)`).
pub const POLARITY_NAMES: [&str; POLARITY_COUNT] = ["positive", "negative", "neutral"];

const MAGIC: &[u8; 4] = b"TPVH";
const FORMAT_VERSION: u32 = 1;

/// Maps a [`Polarity`] variant to its canonical row index (`0..3`).
pub const fn polarity_index(polarity: Polarity) -> usize {
    match polarity {
        Polarity::Positive => 0,
        Polarity::Negative => 1,
        Polarity::Neutral => 2,
    }
}

/// Maps a canonical row index (`0..3`) back to [`Polarity`].
pub const fn index_polarity(index: usize) -> Option<Polarity> {
    match index {
        0 => Some(Polarity::Positive),
        1 => Some(Polarity::Negative),
        2 => Some(Polarity::Neutral),
        _ => None,
    }
}

/// Numerically stable three-class softmax over `logits` in `f64`.
pub fn softmax_polarity(logits: [f64; POLARITY_COUNT]) -> Result<PolarityScores, String> {
    if logits.iter().any(|x| !x.is_finite()) {
        return Err("polarity head produced non-finite logit".into());
    }
    let max_logit = logits[0].max(logits[1]).max(logits[2]);
    let e0 = (logits[0] - max_logit).exp();
    let e1 = (logits[1] - max_logit).exp();
    let e2 = (logits[2] - max_logit).exp();
    let sum = e0 + e1 + e2;
    if !sum.is_finite() || sum <= 0.0 {
        return Err("polarity head softmax normalization failed".into());
    }
    Ok(PolarityScores::new(e0 / sum, e1 / sum, e2 / sum))
}

/// Validates and normalizes caller-supplied or scorer-returned [`PolarityScores`].
pub fn validate_polarity_scores(scores: PolarityScores) -> Result<PolarityScores, String> {
    if !scores.is_valid() {
        return Err(format!("invalid polarity scores: {scores:?}"));
    }
    let sum = scores.positive + scores.negative + scores.neutral;
    if !sum.is_finite() || sum <= 0.0 {
        return Err("polarity scores must have a positive finite sum".into());
    }
    if (sum - 1.0).abs() > 1e-3 {
        return Err(format!("polarity scores must sum to 1.0 (got {sum:.6})"));
    }
    Ok(PolarityScores::new(scores.positive / sum, scores.negative / sum, scores.neutral / sum))
}

/// Linear three-class polarity head (`[positive, negative, neutral]`) over
/// `dim`-dimensional dense embeddings, trained with `turbo-trainer`.
#[derive(Clone, Debug)]
pub struct PolarityVectorHead {
    dim: usize,
    weights: Vec<f32>,
    biases: [f32; POLARITY_COUNT],
    encoder: Option<ModelRef>,
    model_ref: ModelRef,
}

impl PolarityVectorHead {
    /// Constructs a validated [`PolarityVectorHead`] from `weights` (`[3, dim]`
    /// row-major in `[positive, negative, neutral]` order) and `biases`.
    pub fn new(
        dim: usize,
        weights: Vec<f32>,
        biases: [f32; POLARITY_COUNT],
        encoder: Option<ModelRef>,
    ) -> Result<Self, String> {
        if dim == 0 || dim > u32::MAX as usize {
            return Err("polarity head dimension must be in 1..=u32::MAX".into());
        }
        let expected = dim.checked_mul(POLARITY_COUNT).ok_or("polarity head weight count overflows")?;
        if weights.len() != expected {
            return Err(format!("expected {expected} weights for 3x{dim} polarity head, got {}", weights.len()));
        }
        if weights.iter().any(|w| !w.is_finite()) || biases.iter().any(|b| !b.is_finite()) {
            return Err("polarity head weights and biases must be finite".into());
        }
        if let Some(enc) = &encoder
            && (enc.id.len() > u32::MAX as usize
                || enc.revision.len() > u32::MAX as usize
                || enc.sha256.len() > u32::MAX as usize)
        {
            return Err("encoder ModelRef field exceeds u32::MAX".into());
        }
        let bytes = encode_bytes(dim, &weights, &biases, encoder.as_ref());
        let digest = sha256(&bytes).to_hex();
        let model_ref = ModelRef { id: "polarity:vector-head".into(), revision: "1".into(), sha256: digest };
        Ok(Self { dim, weights, biases, encoder, model_ref })
    }

    /// Binds this head to a required encoder [`ModelRef`].
    pub fn with_encoder(self, encoder: ModelRef) -> Result<Self, String> {
        Self::new(self.dim, self.weights, self.biases, Some(encoder))
    }

    /// Input embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Row-major weights of shape `[3, dim]`.
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Per-class biases `[positive, negative, neutral]`.
    pub fn biases(&self) -> &[f32; POLARITY_COUNT] {
        &self.biases
    }

    /// Optional encoder [`ModelRef`] required at inference time.
    pub fn encoder(&self) -> Option<&ModelRef> {
        self.encoder.as_ref()
    }

    /// [`ModelRef`] identifying the serialized head weights and configuration.
    pub fn model_ref(&self) -> &ModelRef {
        &self.model_ref
    }

    /// Predicts both the argmax [`Polarity`] and normalized [`PolarityScores`]
    /// for a single `dim`-dimensional embedding `vector`.
    pub fn predict(&self, vector: &[f32]) -> Result<(Polarity, PolarityScores), String> {
        let scores = self.score_polarity(vector)?;
        Ok((scores.label(), scores))
    }

    /// Serializes the head in the canonical `TPVH` v1 binary format.
    pub fn to_bytes(&self) -> Vec<u8> {
        encode_bytes(self.dim, &self.weights, &self.biases, self.encoder.as_ref())
    }

    /// Deserializes a [`PolarityVectorHead`] from `TPVH` v1 bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 16 {
            return Err("polarity head bytes too short".into());
        }
        if &bytes[0..4] != MAGIC {
            return Err("invalid polarity head magic (expected TPVH)".into());
        }
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if version != FORMAT_VERSION {
            return Err(format!("unsupported polarity head format version {version}"));
        }
        let dim = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let classes = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        if classes != POLARITY_COUNT {
            return Err(format!("expected {POLARITY_COUNT} polarity classes, got {classes}"));
        }
        if dim == 0 {
            return Err("polarity head dimension must be > 0".into());
        }
        let weight_count = dim.checked_mul(POLARITY_COUNT).ok_or("polarity head weight count overflow")?;
        let float_bytes = (weight_count + POLARITY_COUNT).checked_mul(4).ok_or("polarity head byte count overflow")?;
        let mut cursor = 16usize;
        if bytes.len() < cursor + float_bytes + 1 {
            return Err("truncated polarity head payload".into());
        }
        let mut weights = Vec::with_capacity(weight_count);
        for _ in 0..weight_count {
            let v = f32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
            weights.push(v);
            cursor += 4;
        }
        let mut biases = [0.0f32; POLARITY_COUNT];
        for b in &mut biases {
            *b = f32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
            cursor += 4;
        }
        let has_encoder = bytes[cursor];
        cursor += 1;
        let encoder = match has_encoder {
            0 => None,
            1 => {
                let id = read_str(bytes, &mut cursor)?;
                let revision = read_str(bytes, &mut cursor)?;
                let sha256 = read_str(bytes, &mut cursor)?;
                Some(ModelRef { id, revision, sha256 })
            }
            other => return Err(format!("invalid encoder flag {other} in polarity head")),
        };
        if cursor != bytes.len() {
            return Err("trailing bytes in polarity head".into());
        }
        Self::new(dim, weights, biases, encoder)
    }
}

impl PolarityVectorScorer for PolarityVectorHead {
    fn score_polarity(&self, vec: &[f32]) -> Result<PolarityScores, String> {
        if vec.len() != self.dim {
            return Err(format!("expected vector of dim {}, got {}", self.dim, vec.len()));
        }
        if vec.iter().any(|x| !x.is_finite()) {
            return Err("input vector contains non-finite values".into());
        }
        let mut logits = [0.0f64; POLARITY_COUNT];
        for (c, logit) in logits.iter_mut().enumerate() {
            let row = &self.weights[c * self.dim..(c + 1) * self.dim];
            let mut dot = self.biases[c] as f64;
            for (&w, &x) in row.iter().zip(vec.iter()) {
                dot += (w as f64) * (x as f64);
            }
            *logit = dot;
        }
        softmax_polarity(logits)
    }

    fn model(&self) -> Option<ModelRef> {
        self.encoder.clone()
    }

    fn config_hash(&self) -> u64 {
        let mut h = crate::hash::FoldHasher::default();
        self.model_ref.sha256.hash(&mut h);
        h.finish()
    }
}

fn encode_bytes(dim: usize, weights: &[f32], biases: &[f32; POLARITY_COUNT], encoder: Option<&ModelRef>) -> Vec<u8> {
    let mut out = Vec::with_capacity(16 + (weights.len() + POLARITY_COUNT) * 4 + 1);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&(dim as u32).to_le_bytes());
    out.extend_from_slice(&(POLARITY_COUNT as u32).to_le_bytes());
    for &w in weights {
        out.extend_from_slice(&w.to_bits().to_le_bytes());
    }
    for &b in biases {
        out.extend_from_slice(&b.to_bits().to_le_bytes());
    }
    match encoder {
        None => out.push(0),
        Some(enc) => {
            out.push(1);
            write_str(&mut out, &enc.id);
            write_str(&mut out, &enc.revision);
            write_str(&mut out, &enc.sha256);
        }
    }
    out
}

fn write_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn read_str(bytes: &[u8], cursor: &mut usize) -> Result<String, String> {
    if *cursor + 4 > bytes.len() {
        return Err("truncated string length in polarity head".into());
    }
    let len = u32::from_le_bytes(bytes[*cursor..*cursor + 4].try_into().unwrap()) as usize;
    *cursor += 4;
    if *cursor + len > bytes.len() {
        return Err("truncated string body in polarity head".into());
    }
    let s = std::str::from_utf8(&bytes[*cursor..*cursor + len])
        .map_err(|_| "invalid UTF-8 in polarity head string")?
        .to_owned();
    *cursor += len;
    Ok(s)
}
