//! Inference options, text normalization, and sliding-window chunking for
//! `DocumentCategorizerDL` (`opennlp.dl.InferenceOptions` and
//! `opennlp.dl.AbstractDL`).

use std::ops::Range;

use super::DlDoccatError;

/// Default `lowerCase` setting for `DocumentCategorizerDL`
/// (`DocumentCategorizerDL.LOWER_CASE_DEFAULT = true`).
pub const DEFAULT_LOWER_CASE: bool = true;

/// Default `documentSplitSize` (`InferenceOptions.DOCUMENT_SPLIT_SIZE_DEFAULT = 250`).
pub const DEFAULT_DOCUMENT_SPLIT_SIZE: usize = 250;

/// Default `splitOverlapSize` (`InferenceOptions.SPLIT_OVERLAP_SIZE_DEFAULT = 50`).
pub const DEFAULT_SPLIT_OVERLAP_SIZE: usize = 50;

/// How long documents or spans are split into sequence-classification windows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DlChunkingMode {
    /// Splits text on Unicode `White_Space` into overlapping windows of
    /// `document_split_size` whitespace-delimited words with
    /// `split_overlap_size` overlap, joined by single spaces (`' '`), matching
    /// `AbstractDL.whitespaceChunks` in OpenNLP's `DocumentCategorizerDL`.
    #[default]
    WhitespaceWords,
    /// Windows directly over pre-tokenized `opennlp:subwords` (`Subwords`)
    /// piece IDs in overlapping windows of `min(document_split_size, max_content_tokens)`
    /// pieces with `split_overlap_size` piece overlap.
    Subwords,
}

/// Inference options for `DocumentCategorizerDL`, ported from
/// `opennlp.dl.InferenceOptions`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DlInferenceOptions {
    /// Whether to lowercase input text before tokenization (`default: true`
    /// for `DocumentCategorizerDL`).
    pub lower_case: bool,
    /// Maximum number of words (or subwords in [`DlChunkingMode::Subwords`])
    /// per sliding window (`default: 250`, must be `> 0`).
    pub document_split_size: usize,
    /// Number of overlapping words (or subwords) between consecutive windows
    /// (`default: 50`, must be `< document_split_size`).
    pub split_overlap_size: usize,
    /// Whether to normalize all Unicode whitespace characters to ASCII `' '`
    /// before tokenization (`default: false`).
    pub normalize_whitespace: bool,
    /// Whether to normalize Unicode dash characters to ASCII `'-'` before
    /// tokenization (`default: false`).
    pub normalize_dashes: bool,
    /// Whether the model takes `token_type_ids` (`default: true`).
    pub include_token_type_ids: bool,
    /// How long inputs are windowed (`default: DlChunkingMode::WhitespaceWords`).
    pub chunking_mode: DlChunkingMode,
}

impl Default for DlInferenceOptions {
    fn default() -> Self {
        Self {
            lower_case: DEFAULT_LOWER_CASE,
            document_split_size: DEFAULT_DOCUMENT_SPLIT_SIZE,
            split_overlap_size: DEFAULT_SPLIT_OVERLAP_SIZE,
            normalize_whitespace: false,
            normalize_dashes: false,
            include_token_type_ids: true,
            chunking_mode: DlChunkingMode::WhitespaceWords,
        }
    }
}

impl DlInferenceOptions {
    /// Validates `document_split_size > 0` and
    /// `split_overlap_size < document_split_size`, matching
    /// `InferenceOptions.setDocumentSplitSize` / `setSplitOverlapSize` and
    /// `AbstractDL.whitespaceChunks`.
    pub fn validate(&self) -> Result<(), DlDoccatError> {
        if self.document_split_size == 0 {
            return Err(DlDoccatError::InvalidOptions("documentSplitSize must be > 0".into()));
        }
        if self.split_overlap_size >= self.document_split_size {
            return Err(DlDoccatError::InvalidOptions("splitOverlapSize must be >= 0 and < documentSplitSize".into()));
        }
        Ok(())
    }

    pub fn with_lower_case(mut self, lower_case: bool) -> Self {
        self.lower_case = lower_case;
        self
    }

    pub fn with_document_split_size(mut self, document_split_size: usize) -> Self {
        self.document_split_size = document_split_size;
        self
    }

    pub fn with_split_overlap_size(mut self, split_overlap_size: usize) -> Self {
        self.split_overlap_size = split_overlap_size;
        self
    }

    pub fn with_normalize_whitespace(mut self, normalize_whitespace: bool) -> Self {
        self.normalize_whitespace = normalize_whitespace;
        self
    }

    pub fn with_normalize_dashes(mut self, normalize_dashes: bool) -> Self {
        self.normalize_dashes = normalize_dashes;
        self
    }

    pub fn with_include_token_type_ids(mut self, include_token_type_ids: bool) -> Self {
        self.include_token_type_ids = include_token_type_ids;
        self
    }

    pub fn with_chunking_mode(mut self, chunking_mode: DlChunkingMode) -> Self {
        self.chunking_mode = chunking_mode;
        self
    }
}

/// Returns `true` if `c` is one of the Unicode dash characters replaced by
/// `AbstractDL.DASHES` (`[\u00AD\u2010\u2011\u2012\u2013\u2014\u2015\u2212\uFE58\uFE63\uFF0D]`).
fn is_dl_dash(c: char) -> bool {
    matches!(c as u32, 0x00AD | 0x2010..=0x2015 | 0x2212 | 0xFE58 | 0xFE63 | 0xFF0D)
}

/// Applies `normalize_whitespace`, `normalize_dashes`, and `lower_case` to
/// `text` in the exact order of `opennlp.dl.AbstractDL.tokenize`.
pub fn preprocess_text(text: &str, options: &DlInferenceOptions) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        let mapped = if options.normalize_whitespace && c.is_whitespace() {
            ' '
        } else if options.normalize_dashes && is_dl_dash(c) {
            '-'
        } else {
            c
        };
        if options.lower_case {
            for lc in mapped.to_lowercase() {
                out.push(lc);
            }
        } else {
            out.push(mapped);
        }
    }
    out
}

/// Splits `text` on Unicode `White_Space` (`char::is_whitespace`) into
/// overlapping chunks of `document_split_size` words with `split_overlap_size`
/// word overlap, joined by single spaces (`' '`), matching
/// `opennlp.dl.AbstractDL.whitespaceChunks`.
pub fn whitespace_chunks(
    text: &str,
    document_split_size: usize,
    split_overlap_size: usize,
) -> Result<Vec<String>, DlDoccatError> {
    if document_split_size == 0 {
        return Err(DlDoccatError::InvalidOptions("documentSplitSize must be > 0".into()));
    }
    if split_overlap_size >= document_split_size {
        return Err(DlDoccatError::InvalidOptions("splitOverlapSize must be >= 0 and < documentSplitSize".into()));
    }
    let tokens: Vec<&str> = text.split(char::is_whitespace).filter(|t| !t.is_empty()).collect();
    let ranges = window_ranges(tokens.len(), document_split_size, split_overlap_size)?;
    Ok(ranges.into_iter().map(|r| tokens[r].join(" ")).collect())
}

/// Computes sliding-window index ranges `[start..end]` over `total` items with
/// window length `window_size` and overlap `overlap_size`.
pub fn window_ranges(
    total: usize,
    window_size: usize,
    overlap_size: usize,
) -> Result<Vec<Range<usize>>, DlDoccatError> {
    if window_size == 0 {
        return Err(DlDoccatError::InvalidOptions("documentSplitSize must be > 0".into()));
    }
    if overlap_size >= window_size {
        return Err(DlDoccatError::InvalidOptions("splitOverlapSize must be >= 0 and < documentSplitSize".into()));
    }
    if total == 0 {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    let mut start = 0usize;
    while start < total {
        let end = start.saturating_add(window_size).min(total);
        out.push(start..end);
        start = if end == total { end } else { end - overlap_size };
    }
    Ok(out)
}
