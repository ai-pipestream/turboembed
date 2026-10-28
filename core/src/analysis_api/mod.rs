//! In-process text analysis, OpenNLP-compatible stages, and binary
//! `DocumentGraph` (`TGRF`) wire export over the `libturbo` C ABI.

use std::sync::Arc;

use turbo_analysis::OffsetTable;

use crate::status::{Error, INVALID_HANDLE, Result};
use crate::turbo_text;

mod engine;
mod ffi;
mod models;

pub use engine::*;
pub use ffi::*;

pub(crate) const ANALYZER_MAGIC: u64 = 0x7475_7262_6f61_6e31; // "turboan1"
pub(crate) const ANALYSIS_RESULT_MAGIC: u64 = 0x7475_7262_6f61_7231; // "turboar1"

/// C ABI configuration descriptor for `turbo_analyzer_create`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct turbo_analyzer_desc {
    pub struct_size: u32,
    pub flags: u32,
    pub language: turbo_text,
    pub sent_model_path: turbo_text,
    pub tok_model_path: turbo_text,
    pub pos_model_path: turbo_text,
    pub lemma_model_path: turbo_text,
    pub ner_model_path: turbo_text,
}

/// C ABI span record with bi-directional UTF-8 byte, UTF-16 code-unit,
/// Unicode scalar (codepoint), and token-index offsets.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct turbo_span {
    pub byte_start: u32,
    pub byte_end: u32,
    pub utf16_start: u32,
    pub utf16_end: u32,
    pub codepoint_start: u32,
    pub codepoint_end: u32,
    pub token_start: u32,
    pub token_end: u32,
    pub label_offset: u32,
    pub label_len: u32,
    pub prob: f64,
}

/// Opaque handle to an in-process text analysis pipeline.
pub struct turbo_analyzer {
    pub(crate) magic: u64,
    pub(crate) inner: Arc<AnalyzerEngine>,
}

/// Opaque handle to an analysis result holding spans, a contiguous UTF-8
/// string pool, and an optional binary `DocumentGraph` (`TGRF`) buffer.
pub struct turbo_analysis_result {
    pub(crate) magic: u64,
    pub(crate) inner: AnalysisResultInner,
}

#[derive(Clone, Debug, Default)]
pub struct AnalysisResultInner {
    pub spans: Vec<turbo_span>,
    pub pool: Vec<u8>,
    pub graph: Vec<u8>,
}

#[derive(Default)]
pub(crate) struct SpanPoolBuilder {
    pub(crate) spans: Vec<turbo_span>,
    pub(crate) pool: Vec<u8>,
}

impl SpanPoolBuilder {
    pub(crate) fn with_capacity(span_cap: usize, pool_cap: usize) -> Self {
        Self { spans: Vec::with_capacity(span_cap), pool: Vec::with_capacity(pool_cap) }
    }

    pub(crate) fn intern_label(&mut self, label: &str) -> (u32, u32) {
        let offset = self.pool.len() as u32;
        let len = label.len() as u32;
        self.pool.extend_from_slice(label.as_bytes());
        (offset, len)
    }

    pub(crate) fn push_text_span(
        &mut self,
        text: &str,
        offsets: &OffsetTable,
        bytes: (u32, u32),
        tokens: (u32, u32),
        label: &str,
        prob: f64,
    ) {
        let (byte_start, byte_end) = bytes;
        let (token_start, token_end) = tokens;
        let utf16_start = offsets.to_utf16(text, byte_start).0;
        let utf16_end = offsets.to_utf16(text, byte_end).0;
        let codepoint_start = offsets.to_code_point(text, byte_start).0;
        let codepoint_end = offsets.to_code_point(text, byte_end).0;
        let (label_offset, label_len) = self.intern_label(label);
        self.spans.push(turbo_span {
            byte_start,
            byte_end,
            utf16_start,
            utf16_end,
            codepoint_start,
            codepoint_end,
            token_start,
            token_end,
            label_offset,
            label_len,
            prob,
        });
    }

    pub(crate) fn push_token_span(&mut self, token_start: u32, token_end: u32, label: &str, prob: f64) {
        let (label_offset, label_len) = self.intern_label(label);
        self.spans.push(turbo_span {
            byte_start: 0,
            byte_end: 0,
            utf16_start: 0,
            utf16_end: 0,
            codepoint_start: 0,
            codepoint_end: 0,
            token_start,
            token_end,
            label_offset,
            label_len,
            prob,
        });
    }

    pub(crate) fn finish(self, graph: Vec<u8>) -> AnalysisResultInner {
        AnalysisResultInner { spans: self.spans, pool: self.pool, graph }
    }
}

pub(crate) unsafe fn analyzer_handle<'a>(a: *mut turbo_analyzer) -> Result<&'a turbo_analyzer> {
    match unsafe { a.as_ref() } {
        Some(r) if r.magic == ANALYZER_MAGIC => Ok(r),
        _ => Err(Error::new(INVALID_HANDLE, "not a turbo_analyzer")),
    }
}

pub(crate) unsafe fn analysis_result_handle<'a>(r: *const turbo_analysis_result) -> Result<&'a turbo_analysis_result> {
    match unsafe { r.as_ref() } {
        Some(res) if res.magic == ANALYSIS_RESULT_MAGIC => Ok(res),
        _ => Err(Error::new(INVALID_HANDLE, "not a turbo_analysis_result")),
    }
}
