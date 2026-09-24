//! Deterministic source chunking with model token budgets and byte offsets.
//!
//! The optional `CHUNK` task (PLAN.md section 3). It prepares model-sized
//! text chunks for the embedding path; callers can always skip it and
//! provide their own chunks or prepared tokens. Token counts come from a
//! [`crate::tokenizer::Tokenizer`] (which implements [`TokenCounter`]) so the
//! plan uses the same tokenizer the model runs.
//!
//! Contract:
//!
//! * **Model token budgets.** Budgets are expressed in model tokens.
//!   [`ChunkerConfig::max_tokens`] is the model sequence capacity and
//!   [`ChunkerConfig::reserved_tokens`] reserves space for special tokens and
//!   model prefixes (for example `[CLS]`/`[SEP]`). Token counts come from a
//!   caller-supplied [`TokenCounter`], so the count is produced by the same
//!   tokenizer the model will run — this module does not guess a tokenizer.
//! * **Original source offsets.** Every chunk reports half-open UTF-8 **byte**
//!   offsets into the *original* input. The input is never normalized or
//!   copied; `&text[chunk.byte_range]` is exactly the text the chunk covers.
//! * **Explicit overlap.** [`ChunkerConfig::overlap_tokens`] bounds how many
//!   counted tokens consecutive chunks of the same paragraph may share.
//! * **Deterministic behavior.** The output is a pure function of the input
//!   text, the configuration, and the (required-deterministic) token counter.
//! * **Forward progress.** Long spans split even without punctuation — first
//!   at sentence boundaries (optional), then at whitespace, then at character
//!   boundaries. Each chunk ends strictly later than the previous one. If even
//!   a single character exceeds the budget, chunking fails with an explicit
//!   error instead of looping or silently truncating.
//!
//! Not provided: linguistic analysis. Sentence segmentation is the same
//! deterministic terminator scan the E2E harness uses (`.`, `!`, `?` plus a
//! small abbreviation list) — not a linguistic parser — and is only a split
//! preference, never a semantic claim. Chunks never span paragraph boundaries.
//!
//! This utility is host CPU code. The C ABI exposes it through
//! `turbo_chunk_plan_*`; bindings that need UTF-16 offsets convert from the
//! byte offsets reported here.

use std::fmt;
use std::ops::Range;

/// Counts model tokens for a text span, excluding special tokens and model
/// prefixes (those are covered by [`ChunkerConfig::reserved_tokens`]).
///
/// Implementations must be deterministic: the same input must always produce
/// the same count, or the chunk plan itself stops being deterministic.
/// Any `Fn(&str) -> usize` closure implements this trait, so a model
/// tokenizer's `encode(text).len()` plugs in directly.
pub trait TokenCounter {
    /// Model tokens in `text`, excluding special tokens and prefixes.
    fn count_tokens(&self, text: &str) -> usize;
}

impl<F: Fn(&str) -> usize> TokenCounter for F {
    fn count_tokens(&self, text: &str) -> usize {
        self(text)
    }
}

/// Chunking configuration. Echoed back in the resulting [`ChunkPlan`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkerConfig {
    /// Model sequence capacity in tokens (for example 256 for the MiniLM
    /// bundle's default sequence limit).
    pub max_tokens: usize,
    /// Tokens reserved out of `max_tokens` for special tokens and model
    /// prefixes (for example 2 for `[CLS]` + `[SEP]`). The usable content
    /// budget per chunk is `max_tokens - reserved_tokens`.
    pub reserved_tokens: usize,
    /// Maximum counted tokens that consecutive chunks *of the same paragraph*
    /// may share. `0` disables overlap. Must be smaller than the content
    /// budget so every chunk still makes forward progress.
    pub overlap_tokens: usize,
    /// When `true`, a paragraph over budget is split at sentence boundaries
    /// before falling back to whitespace and character boundaries. When
    /// `false`, sentence segmentation is skipped entirely.
    pub sentence_boundaries: bool,
}

impl ChunkerConfig {
    /// Usable content-token budget per chunk.
    pub fn content_budget(&self) -> usize {
        self.max_tokens.saturating_sub(self.reserved_tokens)
    }
}

/// Chunking failure. No partial plan is returned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkError {
    /// The configuration cannot produce chunks: zero content budget, or an
    /// overlap that would prevent forward progress.
    InvalidConfig(String),
    /// A single character at `byte_offset` exceeds the content budget by the
    /// supplied counter, so no chunk boundary can make forward progress.
    NoProgress {
        /// Byte offset at which no chunk could be advanced.
        byte_offset: usize,
    },
}

impl fmt::Display for ChunkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid chunker config: {msg}"),
            Self::NoProgress { byte_offset } => write!(
                f,
                "no forward progress: a single character at byte {byte_offset} \
                 exceeds the content token budget"
            ),
        }
    }
}

impl std::error::Error for ChunkError {}

/// One chunk of the original input, identified by byte offsets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceChunk {
    /// Position of this chunk in the plan (0-based, in source order).
    pub index: usize,
    /// 0-based index of the source paragraph this chunk belongs to. Chunks
    /// never span paragraph boundaries.
    pub paragraph_index: usize,
    /// Half-open UTF-8 byte offsets into the original input text.
    pub byte_range: Range<usize>,
    /// Content tokens counted for `byte_range` by the supplied counter
    /// (excluding `reserved_tokens`). Always within the content budget.
    pub token_count: usize,
}

impl SourceChunk {
    /// The chunk text as a view of the original input. `input` must be the
    /// same text that produced this chunk.
    pub fn text<'t>(&self, input: &'t str) -> &'t str {
        &input[self.byte_range.clone()]
    }
}

/// The deterministic result of chunking one source text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkPlan {
    /// Caller-supplied source identity (file name, document id, …).
    pub source: String,
    /// The configuration that produced this plan.
    pub config: ChunkerConfig,
    /// Chunks in source order. Empty when the input has no non-whitespace
    /// content.
    pub chunks: Vec<SourceChunk>,
}

/// Chunk `text` into model-sized pieces.
///
/// `source` is an identity label echoed into the plan; it does not affect
/// chunk boundaries. `counter` must be the model tokenizer's content-token
/// count (see [`TokenCounter`]).
///
/// Empty and whitespace-only inputs produce an empty plan. Embedded NUL bytes
/// and non-ASCII text are ordinary content; splits always land on `char`
/// boundaries.
pub fn chunk_source(
    source: &str,
    text: &str,
    config: &ChunkerConfig,
    counter: &dyn TokenCounter,
) -> Result<ChunkPlan, ChunkError> {
    let budget = config.content_budget();
    if budget == 0 {
        return Err(ChunkError::InvalidConfig(format!(
            "max_tokens ({}) must exceed reserved_tokens ({})",
            config.max_tokens, config.reserved_tokens
        )));
    }
    if config.overlap_tokens >= budget {
        return Err(ChunkError::InvalidConfig(format!(
            "overlap_tokens ({}) must be smaller than the content budget ({budget})",
            config.overlap_tokens
        )));
    }

    let mut chunks = Vec::new();
    for (paragraph_index, para_range) in paragraph_spans(text).into_iter().enumerate() {
        let units = paragraph_units(text, para_range, config, budget, counter)?;
        pack_units(text, &units, paragraph_index, budget, config, counter, &mut chunks);
    }
    Ok(ChunkPlan { source: source.to_string(), config: config.clone(), chunks })
}

/// Paragraph spans of `text`: half-open byte ranges over maximal runs of
/// non-blank lines (a line is blank when it contains only whitespace), with
/// leading/trailing whitespace trimmed from each span. Offsets index the
/// original input; the text is not normalized or copied.
pub fn paragraph_spans(text: &str) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let mut para_start: Option<usize> = None;
    let mut para_end = 0usize;
    let mut line_start = 0usize;
    let bytes = text.as_bytes();
    let mut i = 0usize;
    loop {
        let at_eof = i >= bytes.len();
        if at_eof || bytes[i] == b'\n' {
            let line = &text[line_start..i];
            if line.trim().is_empty() {
                if let Some(start) = para_start.take() {
                    if let Some(r) = trim_range(text, start..para_end) {
                        out.push(r);
                    }
                }
            } else {
                if para_start.is_none() {
                    para_start = Some(line_start);
                }
                para_end = i;
            }
            if at_eof {
                break;
            }
            line_start = i + 1;
        }
        i += 1;
    }
    if let Some(start) = para_start {
        if let Some(r) = trim_range(text, start..para_end) {
            out.push(r);
        }
    }
    out
}

/// Sentence spans of `text`: half-open byte ranges into `text`, trimmed of
/// surrounding whitespace. Deterministic terminator scan (`.`, `!`, `?`
/// followed by end or whitespace, with closing quotes/brackets attached and a
/// small abbreviation list) — not a linguistic parser. A block with no
/// terminator produces no sentence spans, so callers can keep treating it as
/// one unit; a trailing fragment after at least one terminator is a span.
pub fn sentence_spans(text: &str) -> Vec<Range<usize>> {
    let Some(trimmed) = trim_range(text, 0..text.len()) else {
        return Vec::new();
    };
    let base = trimmed.start;
    let para = &text[trimmed];
    let bytes = para.as_bytes();
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        let ch = bytes[i];
        if matches!(ch, b'.' | b'!' | b'?') {
            let prev = &para[start..i];
            let token = prev
                .rsplit(|c: char| c.is_whitespace() || matches!(c, ',' | ';' | ':' | '(' | '['))
                .next()
                .unwrap_or("")
                .trim_matches(|c: char| !c.is_ascii_alphabetic() && c != '.')
                .to_ascii_lowercase();
            let is_abbrev = ch == b'.' && ABBREV.iter().any(|a| token == *a);
            let mut j = i + 1;
            while j < bytes.len() && matches!(bytes[j], b'"' | b'\'' | b')' | b']') {
                j += 1;
            }
            let at_end = j >= bytes.len();
            let next_ws = !at_end && bytes[j].is_ascii_whitespace();
            if !is_abbrev && (at_end || next_ws) {
                if let Some(r) = trim_range(para, start..j) {
                    out.push(base + r.start..base + r.end);
                }
                while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                    j += 1;
                }
                start = j;
                i = j;
                continue;
            }
        }
        i += 1;
    }
    // Only emit a trailing fragment when we already split on a terminator
    // (start > 0). A block with no `.?!` yields no sentence spans so the
    // caller keeps a single stable unit for it.
    if start > 0 {
        if let Some(r) = trim_range(para, start..para.len()) {
            out.push(base + r.start..base + r.end);
        }
    }
    out
}

const ABBREV: &[&str] = &["mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "cf"];

/// Trim whitespace from both ends of `range` in `text`. `None` when nothing
/// remains.
fn trim_range(text: &str, range: Range<usize>) -> Option<Range<usize>> {
    let slice = &text[range.clone()];
    let trimmed = slice.trim_start();
    let start = range.start + (slice.len() - trimmed.len());
    let trimmed = trimmed.trim_end();
    if trimmed.is_empty() {
        return None;
    }
    Some(start..start + trimmed.len())
}

/// Split one paragraph into units that each fit the content budget.
fn paragraph_units(
    text: &str,
    para: Range<usize>,
    config: &ChunkerConfig,
    budget: usize,
    counter: &dyn TokenCounter,
) -> Result<Vec<Range<usize>>, ChunkError> {
    if counter.count_tokens(&text[para.clone()]) <= budget {
        return Ok(vec![para]);
    }
    let mut coarse = Vec::new();
    if config.sentence_boundaries {
        let para_text = &text[para.clone()];
        for r in sentence_spans(para_text) {
            coarse.push(para.start + r.start..para.start + r.end);
        }
    }
    if coarse.is_empty() {
        coarse.push(para);
    }
    let mut units = Vec::new();
    for unit in coarse {
        refine_unit(text, unit, budget, counter, &mut units)?;
    }
    Ok(units)
}

/// Refine one span into minimal units that each fit the budget: the span
/// itself when it fits, otherwise single whitespace-separated words, and
/// character pieces for a word over budget. The packer regroups units, so
/// keeping them minimal gives overlap its finest deterministic granularity.
fn refine_unit(
    text: &str,
    unit: Range<usize>,
    budget: usize,
    counter: &dyn TokenCounter,
    out: &mut Vec<Range<usize>>,
) -> Result<(), ChunkError> {
    if counter.count_tokens(&text[unit.clone()]) <= budget {
        out.push(unit);
        return Ok(());
    }
    let words = word_spans(text, unit.clone());
    if words.len() > 1 {
        for word in words {
            if counter.count_tokens(&text[word.clone()]) <= budget {
                out.push(word);
            } else {
                split_chars(text, word, budget, counter, out)?;
            }
        }
        return Ok(());
    }
    split_chars(text, unit, budget, counter, out)
}

/// Whitespace-separated word runs inside `range`, as absolute byte ranges.
fn word_spans(text: &str, range: Range<usize>) -> Vec<Range<usize>> {
    let slice = &text[range.clone()];
    let mut out = Vec::new();
    let mut start: Option<usize> = None;
    for (off, ch) in slice.char_indices() {
        if ch.is_whitespace() {
            if let Some(s) = start.take() {
                out.push(range.start + s..range.start + off);
            }
        } else if start.is_none() {
            start = Some(off);
        }
    }
    if let Some(s) = start {
        out.push(range.start + s..range.end);
    }
    out
}

/// Last-resort split at character boundaries: repeatedly take the longest
/// char prefix that still fits the budget (checked left to right, stopping at
/// the first prefix that exceeds it, which keeps the result deterministic for
/// any counter). Errors when a single character exceeds the budget.
fn split_chars(
    text: &str,
    range: Range<usize>,
    budget: usize,
    counter: &dyn TokenCounter,
    out: &mut Vec<Range<usize>>,
) -> Result<(), ChunkError> {
    let mut start = range.start;
    while start < range.end {
        let mut end = None;
        for (off, ch) in text[start..range.end].char_indices() {
            let candidate = start + off + ch.len_utf8();
            if counter.count_tokens(&text[start..candidate]) <= budget {
                end = Some(candidate);
            } else {
                break;
            }
        }
        let Some(end) = end else {
            return Err(ChunkError::NoProgress { byte_offset: start });
        };
        out.push(start..end);
        start = end;
    }
    Ok(())
}

/// Pack a paragraph's units into chunks, applying overlap between consecutive
/// chunks of the same paragraph. Every unit individually fits the budget.
fn pack_units(
    text: &str,
    units: &[Range<usize>],
    paragraph_index: usize,
    budget: usize,
    config: &ChunkerConfig,
    counter: &dyn TokenCounter,
    chunks: &mut Vec<SourceChunk>,
) {
    let fits = |i: usize, j: usize| -> bool { counter.count_tokens(&text[units[i].start..units[j].end]) <= budget };
    let mut i = 0usize;
    // When overlap steps the next start back into the previous chunk, the new
    // chunk is pre-verified to reach through this unit, guaranteeing its end
    // advances past the previous chunk even for a non-monotone counter.
    let mut forced_through: Option<usize> = None;
    while i < units.len() {
        let mut j = forced_through.take().unwrap_or(i);
        while j + 1 < units.len() && fits(i, j + 1) {
            j += 1;
        }
        let byte_range = units[i].start..units[j].end;
        let token_count = counter.count_tokens(&text[byte_range.clone()]);
        chunks.push(SourceChunk { index: chunks.len(), paragraph_index, byte_range, token_count });
        if j + 1 >= units.len() {
            break;
        }
        let mut next = j + 1;
        if config.overlap_tokens > 0 {
            while next > i + 1 {
                let cand = next - 1;
                let overlap_ok = counter.count_tokens(&text[units[cand].start..units[j].end]) <= config.overlap_tokens;
                if overlap_ok && fits(cand, j + 1) {
                    next = cand;
                    forced_through = Some(j + 1);
                } else {
                    break;
                }
            }
        }
        i = next;
    }
}
