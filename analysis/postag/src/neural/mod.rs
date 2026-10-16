//! Pure-Rust CPU reference neural part-of-speech taggers and trainers, ported
//! from OpenNLP's `opennlp.tools.postag.neural` package:
//!
//! - **BiLSTM-CRF POS tagger (`#249`)**: [`BilstmPosModel`], [`BilstmPosTagger`],
//!   [`BilstmPosTrainer`], [`BilstmSettings`], [`MultiTaskSample`], [`LstmLayer`],
//!   [`CrfScorer`], [`CrfScoreTable`], and [`AdamOptimizer`]. Reads and writes
//!   the `ONLP-BLPT-1` through `ONLP-BLPT-5` binary formats.
//! - **Feedforward neural POS tagger (`#205`)**: [`FeedforwardPosContext`],
//!   [`FeedforwardPosModel`], [`FeedforwardPosTagger`], [`FeedforwardPosTrainer`],
//!   and [`FeedforwardSettings`]. Reads and writes the `ONLP-FFPT-1` and
//!   `ONLP-FFPT-2` binary formats.

#![allow(clippy::needless_range_loop)]

pub mod adam;
pub mod bilstm_model;
pub mod bilstm_tagger;
pub mod bilstm_trainer;
pub mod crf;
pub mod feedforward;
pub mod lstm;

use std::fmt;
use std::io::{self, Read, Write};

use turbo_unicode::modified_utf8::{push_modified_utf8, push_str_modified_utf8};

pub use adam::AdamOptimizer;
pub use bilstm_model::BilstmPosModel;
pub use bilstm_tagger::BilstmPosTagger;
pub use bilstm_trainer::{BilstmPosTrainer, BilstmSettings, MultiTaskSample};
pub use crf::{CrfScoreTable, CrfScorer};
pub use feedforward::{
    FeedforwardPosContext, FeedforwardPosModel, FeedforwardPosTagger, FeedforwardPosTrainer, FeedforwardSettings,
};
pub use lstm::{ForwardCache, LstmGradients, LstmLayer};

/// A tagged sentence sample `(sentence, tags)` used for neural POS training.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PosSample {
    /// The tokens of the sentence.
    pub sentence: Vec<String>,
    /// The gold part-of-speech tag for each token, of the same length as `sentence`.
    pub tags: Vec<String>,
}

impl PosSample {
    /// Creates a new `PosSample` after checking that `sentence` and `tags` have equal length.
    pub fn new(
        sentence: impl IntoIterator<Item = impl Into<String>>,
        tags: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<Self, NeuralError> {
        let sentence: Vec<String> = sentence.into_iter().map(Into::into).collect();
        let tags: Vec<String> = tags.into_iter().map(Into::into).collect();
        if sentence.len() != tags.len() {
            return Err(NeuralError::InvalidArgument(format!(
                "sentence length ({}) != tags length ({})",
                sentence.len(),
                tags.len()
            )));
        }
        Ok(Self { sentence, tags })
    }
}

/// A scored tag sequence returned by `top_k_sequences` on [`BilstmPosTagger`] or [`FeedforwardPosTagger`].
#[derive(Clone, Debug, PartialEq)]
pub struct NeuralSequence {
    outcomes: Vec<String>,
    probs: Vec<f64>,
    score: f64,
}

impl NeuralSequence {
    /// Creates a sequence whose score is the sum of log-probabilities (`ln(p)`), matching OpenNLP's `Sequence(List<String>, double[])`.
    pub fn from_outcomes_and_probs(outcomes: Vec<String>, probs: Vec<f64>) -> Self {
        let mut score = 0.0f64;
        for &p in &probs {
            score += p.ln();
        }
        Self { outcomes, probs, score }
    }

    /// Assigned tag strings for each token in the sentence.
    pub fn outcomes(&self) -> &[String] {
        &self.outcomes
    }

    /// Per-token probabilities in `(0, 1]`.
    pub fn probs(&self) -> &[f64] {
        &self.probs
    }

    /// Total sequence log-probability score.
    pub fn score(&self) -> f64 {
        self.score
    }
}

/// Errors returned by neural POS model loading, serialization, inference, or training.
#[derive(Debug)]
pub enum NeuralError {
    /// An I/O error occurred while reading or writing a model stream or file.
    Io(io::Error),
    /// A serialized model stream violated the binary format contract.
    Format(String),
    /// A caller passed an invalid argument or hyperparameter.
    InvalidArgument(String),
    /// Inference or optimization produced a non-finite (`NaN` or infinite) value.
    NonFiniteScore(String),
}

impl fmt::Display for NeuralError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NeuralError::Io(e) => write!(f, "neural POS I/O error: {e}"),
            NeuralError::Format(msg) => write!(f, "neural POS format error: {msg}"),
            NeuralError::InvalidArgument(msg) => write!(f, "neural POS invalid argument: {msg}"),
            NeuralError::NonFiniteScore(msg) => write!(f, "neural POS numerical error: {msg}"),
        }
    }
}

impl std::error::Error for NeuralError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            NeuralError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for NeuralError {
    fn from(e: io::Error) -> Self {
        NeuralError::Io(e)
    }
}

/// Reproduces `java.util.Random` (48-bit LCG) for deterministic weight initialization, dropout, and shuffling.
#[derive(Clone, Debug)]
pub struct JavaRng {
    seed: u64,
    next_next_gaussian: Option<f64>,
}

impl JavaRng {
    const MULTIPLIER: u64 = 0x5_deec_e66d;
    const ADDEND: u64 = 0xb;
    const MASK: u64 = (1u64 << 48) - 1;
    const DOUBLE_UNIT: f64 = 1.0 / ((1u64 << 53) as f64);

    /// Creates a new `JavaRng` seeded identically to `new java.util.Random(seed)`.
    pub fn new(seed: i64) -> Self {
        Self { seed: ((seed as u64) ^ Self::MULTIPLIER) & Self::MASK, next_next_gaussian: None }
    }

    #[inline]
    fn next(&mut self, bits: u32) -> i32 {
        self.seed = (self.seed.wrapping_mul(Self::MULTIPLIER).wrapping_add(Self::ADDEND)) & Self::MASK;
        (self.seed >> (48 - bits)) as i32
    }

    /// Reproduces `Random.nextInt(bound)`.
    pub fn next_int_bounded(&mut self, bound: i32) -> i32 {
        assert!(bound > 0, "bound must be positive");
        let mut r = self.next(31);
        let m = bound - 1;
        if (bound & m) == 0 {
            r = (((bound as i64) * (r as i64)) >> 31) as i32;
        } else {
            let mut u = r;
            loop {
                r = u % bound;
                if u.wrapping_sub(r).wrapping_add(m) >= 0 {
                    break;
                }
                u = self.next(31);
            }
        }
        r
    }

    /// Reproduces `Random.nextLong()`.
    pub fn next_long(&mut self) -> i64 {
        let high = (self.next(32) as i64) << 32;
        high.wrapping_add(self.next(32) as i64)
    }

    /// Reproduces `Random.nextDouble()`.
    pub fn next_double(&mut self) -> f64 {
        let high = (self.next(26) as i64) << 27;
        ((high + self.next(27) as i64) as f64) * Self::DOUBLE_UNIT
    }

    /// Reproduces `Random.nextGaussian()` via the polar Marsaglia method.
    pub fn next_gaussian(&mut self) -> f64 {
        if let Some(g) = self.next_next_gaussian.take() {
            return g;
        }
        let (mut v1, mut v2, mut s);
        loop {
            v1 = 2.0 * self.next_double() - 1.0;
            v2 = 2.0 * self.next_double() - 1.0;
            s = v1 * v1 + v2 * v2;
            if !(s >= 1.0 || s == 0.0) {
                break;
            }
        }
        let multiplier = (-2.0 * s.ln() / s).sqrt();
        self.next_next_gaussian = Some(v2 * multiplier);
        v1 * multiplier
    }

    /// Reproduces `Collections.shuffle(list, rng)`.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        let size = i32::try_from(items.len()).expect("list length fits in i32");
        for i in (2..=size).rev() {
            let j = self.next_int_bounded(i) as usize;
            items.swap((i - 1) as usize, j);
        }
    }
}

/// Big-endian binary reader matching `java.io.DataInputStream`.
pub(crate) struct DataReader<R> {
    inner: R,
    scratch: Vec<u8>,
}

impl<R: Read> DataReader<R> {
    pub(crate) fn new(inner: R) -> Self {
        Self { inner, scratch: Vec::new() }
    }

    pub(crate) fn read_boolean(&mut self) -> Result<bool, NeuralError> {
        let mut buf = [0u8; 1];
        self.inner.read_exact(&mut buf).map_err(eof_as_format)?;
        Ok(buf[0] != 0)
    }

    pub(crate) fn read_i32(&mut self) -> Result<i32, NeuralError> {
        let mut buf = [0u8; 4];
        self.inner.read_exact(&mut buf).map_err(eof_as_format)?;
        Ok(i32::from_be_bytes(buf))
    }

    pub(crate) fn read_u16(&mut self) -> Result<u16, NeuralError> {
        let mut buf = [0u8; 2];
        self.inner.read_exact(&mut buf).map_err(eof_as_format)?;
        Ok(u16::from_be_bytes(buf))
    }

    pub(crate) fn read_f32(&mut self) -> Result<f32, NeuralError> {
        let mut buf = [0u8; 4];
        self.inner.read_exact(&mut buf).map_err(eof_as_format)?;
        Ok(f32::from_be_bytes(buf))
    }

    pub(crate) fn read_f64(&mut self) -> Result<f64, NeuralError> {
        let mut buf = [0u8; 8];
        self.inner.read_exact(&mut buf).map_err(eof_as_format)?;
        Ok(f64::from_be_bytes(buf))
    }

    /// Reads a `DataInputStream.readUTF()` string as raw modified-UTF-8 bytes after validating its encoding.
    pub(crate) fn read_utf_bytes(&mut self) -> Result<Vec<u8>, NeuralError> {
        let len = self.read_u16()? as usize;
        self.scratch.resize(len, 0);
        self.inner.read_exact(&mut self.scratch).map_err(eof_as_format)?;
        validate_modified_utf8(&self.scratch)?;
        Ok(self.scratch.clone())
    }

    /// Reads a `DataInputStream.readUTF()` string into a Rust `String`.
    pub(crate) fn read_utf(&mut self) -> Result<String, NeuralError> {
        let len = self.read_u16()? as usize;
        self.scratch.resize(len, 0);
        self.inner.read_exact(&mut self.scratch).map_err(eof_as_format)?;
        decode_modified_utf8(&self.scratch)
    }
}

fn eof_as_format(err: io::Error) -> NeuralError {
    if err.kind() == io::ErrorKind::UnexpectedEof {
        NeuralError::Format(format!("truncated model stream: {err}"))
    } else {
        NeuralError::Io(err)
    }
}

/// Big-endian binary writer matching `java.io.DataOutputStream`.
pub(crate) struct DataWriter<W> {
    inner: W,
    scratch: Vec<u8>,
}

impl<W: Write> DataWriter<W> {
    pub(crate) fn new(inner: W) -> Self {
        Self { inner, scratch: Vec::new() }
    }

    pub(crate) fn write_boolean(&mut self, v: bool) -> Result<(), NeuralError> {
        self.inner.write_all(&[u8::from(v)])?;
        Ok(())
    }

    pub(crate) fn write_i32(&mut self, v: i32) -> Result<(), NeuralError> {
        self.inner.write_all(&v.to_be_bytes())?;
        Ok(())
    }

    pub(crate) fn write_f32(&mut self, v: f32) -> Result<(), NeuralError> {
        self.inner.write_all(&v.to_be_bytes())?;
        Ok(())
    }

    pub(crate) fn write_f64(&mut self, v: f64) -> Result<(), NeuralError> {
        self.inner.write_all(&v.to_be_bytes())?;
        Ok(())
    }

    pub(crate) fn write_utf(&mut self, s: &str) -> Result<(), NeuralError> {
        self.scratch.clear();
        push_str_modified_utf8(&mut self.scratch, s);
        let len = u16::try_from(self.scratch.len())
            .map_err(|_| NeuralError::Format(format!("UTF string too long: {} bytes", self.scratch.len())))?;
        self.inner.write_all(&len.to_be_bytes())?;
        self.inner.write_all(&self.scratch)?;
        Ok(())
    }

    pub(crate) fn write_utf_bytes(&mut self, bytes: &[u8]) -> Result<(), NeuralError> {
        let len = u16::try_from(bytes.len())
            .map_err(|_| NeuralError::Format(format!("UTF string too long: {} bytes", bytes.len())))?;
        self.inner.write_all(&len.to_be_bytes())?;
        self.inner.write_all(bytes)?;
        Ok(())
    }

    pub(crate) fn flush(&mut self) -> Result<(), NeuralError> {
        self.inner.flush()?;
        Ok(())
    }
}

/// Encodes a `&str` into Java modified UTF-8 bytes.
pub(crate) fn encode_str_modified_utf8(s: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len());
    push_str_modified_utf8(&mut out, s);
    out
}

/// Encodes a single UTF-16 code unit into Java modified UTF-8 bytes.
pub(crate) fn encode_unit_modified_utf8(unit: u16) -> Vec<u8> {
    let mut out = Vec::with_capacity(3);
    push_modified_utf8(&mut out, u32::from(unit));
    out
}

fn validate_modified_utf8(bytes: &[u8]) -> Result<(), NeuralError> {
    let mut i = 0;
    while i < bytes.len() {
        let b1 = bytes[i];
        match b1 >> 4 {
            0..=7 => {
                i += 1;
            }
            12 | 13 => {
                if i + 1 >= bytes.len() {
                    return Err(NeuralError::Format("truncated 2-byte modified UTF-8".into()));
                }
                if (bytes[i + 1] & 0xC0) != 0x80 {
                    return Err(NeuralError::Format("invalid 2-byte modified UTF-8".into()));
                }
                i += 2;
            }
            14 => {
                if i + 2 >= bytes.len() {
                    return Err(NeuralError::Format("truncated 3-byte modified UTF-8".into()));
                }
                if (bytes[i + 1] & 0xC0) != 0x80 || (bytes[i + 2] & 0xC0) != 0x80 {
                    return Err(NeuralError::Format("invalid 3-byte modified UTF-8".into()));
                }
                i += 3;
            }
            _ => {
                return Err(NeuralError::Format("invalid modified UTF-8 leading byte".into()));
            }
        }
    }
    Ok(())
}

pub(crate) fn decode_modified_utf8(bytes: &[u8]) -> Result<String, NeuralError> {
    let mut units = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let b1 = bytes[i];
        match b1 >> 4 {
            0..=7 => {
                units.push(u16::from(b1));
                i += 1;
            }
            12 | 13 => {
                if i + 1 >= bytes.len() {
                    return Err(NeuralError::Format("truncated 2-byte modified UTF-8".into()));
                }
                let b2 = bytes[i + 1];
                if (b2 & 0xC0) != 0x80 {
                    return Err(NeuralError::Format("invalid 2-byte modified UTF-8".into()));
                }
                units.push(((u16::from(b1 & 0x1F)) << 6) | u16::from(b2 & 0x3F));
                i += 2;
            }
            14 => {
                if i + 2 >= bytes.len() {
                    return Err(NeuralError::Format("truncated 3-byte modified UTF-8".into()));
                }
                let b2 = bytes[i + 1];
                let b3 = bytes[i + 2];
                if (b2 & 0xC0) != 0x80 || (b3 & 0xC0) != 0x80 {
                    return Err(NeuralError::Format("invalid 3-byte modified UTF-8".into()));
                }
                units.push(((u16::from(b1 & 0x0F)) << 12) | ((u16::from(b2 & 0x3F)) << 6) | u16::from(b3 & 0x3F));
                i += 3;
            }
            _ => {
                return Err(NeuralError::Format("invalid modified UTF-8 leading byte".into()));
            }
        }
    }
    let mut out = String::with_capacity(units.len());
    for decoded in char::decode_utf16(units) {
        let ch = decoded.map_err(|_| NeuralError::Format("unpaired UTF-16 surrogate in string".into()))?;
        out.push(ch);
    }
    Ok(out)
}
