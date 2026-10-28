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
mod batch_words;
pub mod bilstm_model;
pub mod bilstm_tagger;
pub mod bilstm_trainer;
pub mod crf;
pub mod feedforward;
pub mod lstm;

use std::fmt;
use std::io::{self, Write};

use turbo_unicode::modified_utf8::{push_modified_utf8, push_str_modified_utf8};

pub use adam::AdamOptimizer;
pub use bilstm_model::BilstmPosModel;
pub use bilstm_tagger::{BilstmPosTagger, Numerics};
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
    /// Creates a sequence whose score is the sum of log-probabilities (`StrictMath.log(p)`), matching OpenNLP's `Sequence(List<String>, double[])`.
    pub fn from_outcomes_and_probs(outcomes: Vec<String>, probs: Vec<f64>) -> Self {
        let mut score = 0.0f64;
        for &p in &probs {
            score += turbo_jvm::strict_math::log(p);
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
    /// OpenNLP model imports are retired under #1511.
    Disabled,
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
            NeuralError::Disabled => write!(f, "OpenNLP model loading is disabled; see #1509 for replacement heads"),
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

/// `java.util.Random`, which the trainers draw weight initialization,
/// dropout masks and shuffles from: `turbo_jvm`'s port, which is checked
/// against values written by a JVM.
pub use turbo_jvm::random::JavaRandom as JavaRng;

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
