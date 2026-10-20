//! Checked span decoding for extractive question answering.
//!
//! Encoder integration supplies joint question/context windows. Context offsets
//! always refer to the original UTF-8 context, never the question or padding.

mod decode;
pub use decode::decode;

use std::ops::Range;

/// Raw token-head outputs for one joint question/context window.
#[derive(Clone, Debug)]
pub struct WindowLogits {
    pub start: Vec<f32>,
    pub end: Vec<f32>,
    /// None for question, special and padding positions.
    pub context: Vec<Option<Range<usize>>>,
    /// Each context token starts candidates in its most-interior window only.
    /// An answer can end later in that same window even if its final token has
    /// another owner. This avoids losing spans at ownership boundaries.
    pub max_context: Vec<bool>,
    pub cls: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct DecodeOptions {
    pub max_answer_tokens: usize,
    pub top_k: usize,
    /// Abstain when best span minus the minimum window CLS score is <= this.
    pub null_threshold: f64,
    pub non_overlapping: bool,
}
impl Default for DecodeOptions {
    fn default() -> Self {
        Self { max_answer_tokens: 30, top_k: 5, null_threshold: 0.0, non_overlapping: true }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SpanAnswer {
    pub bytes: Range<usize>,
    pub score: f64,
    /// Softmax over the returned candidate spans and the null candidate.
    pub probability: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Answer {
    None { null_score: f64 },
    Spans { answers: Vec<SpanAnswer>, null_score: f64, null_probability: f64 },
}
