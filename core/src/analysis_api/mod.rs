//! In-process text analysis, OpenNLP-compatible stages, and binary
//! `DocumentGraph` (`TGRF`) wire export over the `libturbo` C ABI.

use std::collections::HashMap;
use std::sync::Arc;

use turbo_analysis::OffsetTable;

use crate::status::{CAPACITY, Error, INVALID_HANDLE, Result};
use crate::turbo_text;

mod engine;
mod ffi;
mod graph;
mod heads;

pub use engine::*;
pub use ffi::*;

pub(crate) const ANALYZER_MAGIC: u64 = 0x7475_7262_6f61_6e31; // "turboan1"
pub(crate) const ANALYSIS_RESULT_MAGIC: u64 = 0x7475_7262_6f61_7231; // "turboar1"

pub const TURBO_ANALYSIS_LAYER_SENTENCES: u32 = 1;
pub const TURBO_ANALYSIS_LAYER_TOKENS: u32 = 2;
pub const TURBO_ANALYSIS_LAYER_POS: u32 = 3;
pub const TURBO_ANALYSIS_LAYER_LEMMAS: u32 = 4;
pub const TURBO_ANALYSIS_LAYER_ENTITIES: u32 = 5;
pub const TURBO_ANALYSIS_LAYER_PII: u32 = 6;
pub const TURBO_ANALYSIS_LAYER_DEPENDENCIES: u32 = 7;

pub const TURBO_SPAN_VIEW_RANGE: u32 = 1 << 0;
pub const TURBO_SPAN_SOURCE_RANGE: u32 = 1 << 1;
pub const TURBO_SPAN_TOKEN_RANGE: u32 = 1 << 2;
pub const TURBO_SPAN_LABEL: u32 = 1 << 3;
pub const TURBO_SPAN_VALUE: u32 = 1 << 4;
pub const TURBO_SPAN_PROBABILITY: u32 = 1 << 5;
const TURBO_SPAN_KNOWN_FLAGS: u32 = TURBO_SPAN_VIEW_RANGE
    | TURBO_SPAN_SOURCE_RANGE
    | TURBO_SPAN_TOKEN_RANGE
    | TURBO_SPAN_LABEL
    | TURBO_SPAN_VALUE
    | TURBO_SPAN_PROBABILITY;

pub const TURBO_PII_CONFIDENCE_FORMAT: u32 = 1;
pub const TURBO_PII_CONFIDENCE_CHECKSUM: u32 = 2;
pub const TURBO_PII_CONFIDENCE_LABELLED: u32 = 3;

/// C ABI configuration descriptor for `turbo_analyzer_create`. The five
/// model path fields are retired: they keep their place so a program
/// built against an earlier header still runs, and a non-empty value is
/// refused. Learned components come from the bundle at `bundle_path`.
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
    pub bundle_path: turbo_text,
}

/// The size of turbo_analyzer_desc before bundle_path was appended, which
/// turbo_analyzer_create still accepts.
pub const TURBO_ANALYZER_DESC_SIZE_V1: usize = std::mem::offset_of!(turbo_analyzer_desc, bundle_path);

/// Typed C ABI span row.
///
/// A row belongs to one layer instance and one item within that instance.
/// Discontinuous items emit one row per fragment. The view range is in the
/// coordinate space of `view_id`; the source range is the checked projection
/// to `ViewId::SOURCE`. A range or string field is meaningful only when its
/// flag is set. Absent numeric fields are zero, except `prob`, which is NaN
/// when no probability is present.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct turbo_span {
    pub layer_kind: u32,
    pub flags: u32,
    pub item_index: u32,
    pub fragment_index: u32,
    pub view_id: u32,
    pub classification: u32,
    pub view_byte_start: u32,
    pub view_byte_end: u32,
    pub view_utf16_start: u32,
    pub view_utf16_end: u32,
    pub view_codepoint_start: u32,
    pub view_codepoint_end: u32,
    pub source_byte_start: u32,
    pub source_byte_end: u32,
    pub source_utf16_start: u32,
    pub source_utf16_end: u32,
    pub source_codepoint_start: u32,
    pub source_codepoint_end: u32,
    pub token_start: u32,
    pub token_end: u32,
    pub layer_offset: u32,
    pub layer_len: u32,
    pub instance_offset: u32,
    pub instance_len: u32,
    pub label_offset: u32,
    pub label_len: u32,
    pub value_offset: u32,
    pub value_len: u32,
    pub prob: f64,
}

const _: () = assert!(size_of::<turbo_span>() == 120);
const _: () = assert!(align_of::<turbo_span>() == 8);

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

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<AnalyzerEngine>();
    assert_send_sync::<AnalysisResultInner>();
    assert_send_sync::<turbo_analyzer>();
    assert_send_sync::<turbo_analysis_result>();
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PoolRef {
    pub(crate) offset: u32,
    pub(crate) len: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CoordRange {
    pub(crate) byte_start: u32,
    pub(crate) byte_end: u32,
    pub(crate) utf16_start: u32,
    pub(crate) utf16_end: u32,
    pub(crate) codepoint_start: u32,
    pub(crate) codepoint_end: u32,
}

pub(crate) fn coordinate_range(text: &str, offsets: &OffsetTable, bytes: (u32, u32)) -> Result<CoordRange> {
    let (byte_start, byte_end) = bytes;
    let (start, end) = (byte_start as usize, byte_end as usize);
    if start > end || end > text.len() || !text.is_char_boundary(start) || !text.is_char_boundary(end) {
        return Err(Error::new(CAPACITY, format!("analysis span {byte_start}..{byte_end} is outside its text")));
    }
    Ok(CoordRange {
        byte_start,
        byte_end,
        utf16_start: offsets.to_utf16(text, byte_start).0,
        utf16_end: offsets.to_utf16(text, byte_end).0,
        codepoint_start: offsets.to_code_point(text, byte_start).0,
        codepoint_end: offsets.to_code_point(text, byte_end).0,
    })
}

#[derive(Clone, Copy)]
pub(crate) struct LayerRef {
    pub(crate) kind: u32,
    pub(crate) view_id: u32,
    pub(crate) layer: PoolRef,
    pub(crate) instance: PoolRef,
}

#[derive(Clone, Copy, Default)]
pub(crate) struct SpanDraft {
    pub(crate) item_index: u32,
    pub(crate) fragment_index: u32,
    pub(crate) classification: u32,
    pub(crate) view: Option<CoordRange>,
    pub(crate) source: Option<CoordRange>,
    pub(crate) tokens: Option<(u32, u32)>,
    pub(crate) label: Option<PoolRef>,
    pub(crate) value: Option<PoolRef>,
    pub(crate) prob: Option<f64>,
}

#[derive(Default)]
pub(crate) struct SpanPoolBuilder {
    pub(crate) spans: Vec<turbo_span>,
    pub(crate) pool: Vec<u8>,
    interned: HashMap<Box<str>, PoolRef>,
}

impl SpanPoolBuilder {
    pub(crate) fn with_capacity(span_cap: usize, pool_cap: usize) -> Self {
        Self { spans: Vec::with_capacity(span_cap), pool: Vec::with_capacity(pool_cap), interned: HashMap::new() }
    }

    fn append_pool(&mut self, value: &str) -> Result<PoolRef> {
        let offset =
            u32::try_from(self.pool.len()).map_err(|_| Error::new(CAPACITY, "analysis string pool exceeds u32"))?;
        let len = u32::try_from(value.len()).map_err(|_| Error::new(CAPACITY, "analysis string exceeds u32"))?;
        if offset.checked_add(len).is_none() {
            return Err(Error::new(CAPACITY, "analysis string pool exceeds u32"));
        }
        self.pool.extend_from_slice(value.as_bytes());
        Ok(PoolRef { offset, len })
    }

    pub(crate) fn string_ref(&mut self, value: &str) -> Result<PoolRef> {
        self.intern(value)
    }

    pub(crate) fn intern(&mut self, value: &str) -> Result<PoolRef> {
        if let Some(found) = self.interned.get(value) {
            return Ok(*found);
        }
        let found = self.append_pool(value)?;
        self.interned.insert(Box::from(value), found);
        Ok(found)
    }

    pub(crate) fn layer_ref(&mut self, kind: u32, layer: &str, instance: &str, view_id: u32) -> Result<LayerRef> {
        if !(TURBO_ANALYSIS_LAYER_SENTENCES..=TURBO_ANALYSIS_LAYER_DEPENDENCIES).contains(&kind) {
            return Err(Error::new(CAPACITY, format!("unknown analysis layer kind {kind}")));
        }
        Ok(LayerRef { kind, view_id, layer: self.intern(layer)?, instance: self.intern(instance)? })
    }

    pub(crate) fn push(&mut self, layer: LayerRef, draft: SpanDraft) -> Result<()> {
        if self.spans.len() >= u32::MAX as usize {
            return Err(Error::new(CAPACITY, "analysis span count exceeds u32"));
        }
        if let Some(prob) = draft.prob
            && (!prob.is_finite() || !(0.0..=1.0).contains(&prob))
        {
            return Err(Error::new(CAPACITY, format!("analysis probability {prob} is outside 0..=1")));
        }
        if let Some((start, end)) = draft.tokens
            && start > end
        {
            return Err(Error::new(CAPACITY, "analysis token range is reversed"));
        }
        let mut flags = 0;
        if draft.view.is_some() {
            flags |= TURBO_SPAN_VIEW_RANGE;
        }
        if draft.source.is_some() {
            flags |= TURBO_SPAN_SOURCE_RANGE;
        }
        if draft.tokens.is_some() {
            flags |= TURBO_SPAN_TOKEN_RANGE;
        }
        if draft.label.is_some() {
            flags |= TURBO_SPAN_LABEL;
        }
        if draft.value.is_some() {
            flags |= TURBO_SPAN_VALUE;
        }
        if draft.prob.is_some() {
            flags |= TURBO_SPAN_PROBABILITY;
        }
        debug_assert_eq!(flags & !TURBO_SPAN_KNOWN_FLAGS, 0);
        let view = draft.view.unwrap_or(CoordRange {
            byte_start: 0,
            byte_end: 0,
            utf16_start: 0,
            utf16_end: 0,
            codepoint_start: 0,
            codepoint_end: 0,
        });
        let source = draft.source.unwrap_or(view);
        let source = if draft.source.is_some() {
            source
        } else {
            CoordRange {
                byte_start: 0,
                byte_end: 0,
                utf16_start: 0,
                utf16_end: 0,
                codepoint_start: 0,
                codepoint_end: 0,
            }
        };
        let (token_start, token_end) = draft.tokens.unwrap_or((0, 0));
        let label = draft.label.unwrap_or_default();
        let value = draft.value.unwrap_or_default();
        self.spans.push(turbo_span {
            layer_kind: layer.kind,
            flags,
            item_index: draft.item_index,
            fragment_index: draft.fragment_index,
            view_id: layer.view_id,
            classification: draft.classification,
            view_byte_start: view.byte_start,
            view_byte_end: view.byte_end,
            view_utf16_start: view.utf16_start,
            view_utf16_end: view.utf16_end,
            view_codepoint_start: view.codepoint_start,
            view_codepoint_end: view.codepoint_end,
            source_byte_start: source.byte_start,
            source_byte_end: source.byte_end,
            source_utf16_start: source.utf16_start,
            source_utf16_end: source.utf16_end,
            source_codepoint_start: source.codepoint_start,
            source_codepoint_end: source.codepoint_end,
            token_start,
            token_end,
            layer_offset: layer.layer.offset,
            layer_len: layer.layer.len,
            instance_offset: layer.instance.offset,
            instance_len: layer.instance.len,
            label_offset: label.offset,
            label_len: label.len,
            value_offset: value.offset,
            value_len: value.len,
            prob: draft.prob.unwrap_or(f64::NAN),
        });
        Ok(())
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
