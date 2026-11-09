//! PII constants, C ABI tables (`turbo_pii_span`, `turbo_pii_fact`,
//! `turbo_pii_candidate`), layer extraction, and result borrow entrypoints.

use turbo_analysis::{Ix, Layer, Span};
use turbo_analysis_layers::{
    AllowRule, AllowScope, Pii, PiiCandidates, PiiConfidence, PiiDetectors, PiiDisposition, PiiEvidence, PiiScore,
    RejectReason,
};

use super::{
    CoordRange, PoolRef, SpanDraft, SpanPoolBuilder, TURBO_ANALYSIS_LAYER_PII, analysis_result_handle,
    turbo_analysis_result,
};
use crate::ffi::{call, out_ptr};
use crate::status::{CAPACITY, Error, Result};
use crate::turbo_error;

/// `turbo_analyzer_desc.flags` bits for PII output (`#1904`, `#1940`, `#1941`).
pub const TURBO_ANALYZER_PII_VALUES: u32 = 1 << 0;
pub const TURBO_ANALYZER_PII_KIND_IDS: u32 = 1 << 1;
pub const TURBO_ANALYZER_PII_DETAIL: u32 = 1 << 2;
pub(crate) const TURBO_ANALYZER_KNOWN_FLAGS: u32 =
    TURBO_ANALYZER_PII_VALUES | TURBO_ANALYZER_PII_KIND_IDS | TURBO_ANALYZER_PII_DETAIL;

/// `TURBO_PII_KIND_CUSTOM_BASE + i` is entry `i` of a layer's custom-kind table.
pub const TURBO_PII_KIND_CUSTOM_BASE: u32 = turbo_analysis_layers::CUSTOM_CODE_BASE;

pub const TURBO_PII_CONFIDENCE_FORMAT: u32 = 1;
pub const TURBO_PII_CONFIDENCE_CHECKSUM: u32 = 2;
pub const TURBO_PII_CONFIDENCE_LABELLED_CHECKSUM: u32 = 3;
pub const TURBO_PII_CONFIDENCE_LABELLED: u32 = 4;

pub const TURBO_PII_EVIDENCE_SHAPE: u32 = 1;
pub const TURBO_PII_EVIDENCE_CHECKSUM: u32 = 2;
pub const TURBO_PII_EVIDENCE_LABEL: u32 = 3;
pub const TURBO_PII_EVIDENCE_DENY_HIT: u32 = 4;
pub const TURBO_PII_EVIDENCE_RULE: u32 = 5;

pub const TURBO_PII_CHECKSUM_LUHN: u32 = 1;
pub const TURBO_PII_CHECKSUM_MOD97: u32 = 2;
pub const TURBO_PII_CHECKSUM_MOD11: u32 = 3;
pub const TURBO_PII_CHECKSUM_MOD23: u32 = 4;
pub const TURBO_PII_CHECKSUM_ES_CIF: u32 = 5;
pub const TURBO_PII_CHECKSUM_ISO7064_MOD11_10: u32 = 6;
pub const TURBO_PII_CHECKSUM_ICAO9303: u32 = 7;
pub const TURBO_PII_CHECKSUM_BASE58CHECK: u32 = 8;
pub const TURBO_PII_CHECKSUM_BECH32: u32 = 9;
pub const TURBO_PII_CHECKSUM_EIP55: u32 = 10;
pub const TURBO_PII_CHECKSUM_ABA_WEIGHTS: u32 = 11;
pub const TURBO_PII_CHECKSUM_OTHER: u32 = 12;
pub const TURBO_PII_CHECKSUM_LA_POSTE_MOD5: u32 = 13;
pub const TURBO_PII_CHECKSUM_VIN: u32 = 14;
pub const TURBO_PII_CHECKSUM_UPS_MOD10: u32 = 15;
pub const TURBO_PII_CHECKSUM_FEDEX_MOD11: u32 = 16;
pub const TURBO_PII_CHECKSUM_GS1_MOD10: u32 = 17;

pub const TURBO_PII_SCORE_PROBABILITY: u32 = 1;
pub const TURBO_PII_SCORE_LOGIT: u32 = 2;
pub const TURBO_PII_SCORE_HEURISTIC: u32 = 3;

pub const TURBO_PII_DISPOSITION_KEPT: u32 = 1;
pub const TURBO_PII_DISPOSITION_ALLOWED_BY_RULE: u32 = 2;
pub const TURBO_PII_DISPOSITION_NESTED_IN: u32 = 3;
pub const TURBO_PII_DISPOSITION_REJECTED_BY_RULE: u32 = 4;
pub const TURBO_PII_DISPOSITION_REPEATS: u32 = 5;

pub const TURBO_PII_ALLOW_SCOPE_GLOBAL: u32 = 1;
pub const TURBO_PII_ALLOW_SCOPE_KIND: u32 = 2;

pub const TURBO_PII_ALLOW_RULE_TERM: u32 = 1;
pub const TURBO_PII_ALLOW_RULE_PATTERN: u32 = 2;

pub const TURBO_PII_REJECT_LABEL_REQUIRED: u32 = 1;
pub const TURBO_PII_REJECT_THRESHOLD: u32 = 2;

/// What PII rows and candidate tables carry beyond the default kind name,
/// derived confidence tier, evidence facts, detector identity and score.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PiiRows {
    /// `TURBO_ANALYZER_PII_VALUES`: the normalized value (#1904).
    pub(crate) values: bool,
    /// `TURBO_ANALYZER_PII_KIND_IDS`: the numeric kind (#1940).
    pub(crate) kind_ids: bool,
    /// `TURBO_ANALYZER_PII_DETAIL`: candidate rows (#1941).
    pub(crate) detail: bool,
}

/// One structured evidence fact supporting a PII mention or candidate.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct turbo_pii_fact {
    pub kind: u32,
    pub a: u32,
    pub b: u32,
    pub c: u32,
}

pub const PII_FACT_INTS: usize = 4;
const _: () = assert!(size_of::<turbo_pii_fact>() == PII_FACT_INTS * 4);
const _: () = assert!(align_of::<turbo_pii_fact>() == 4);

/// Per-row PII detector identity, evidence slice and optional score for one
/// `TURBO_ANALYSIS_LAYER_PII` row in `turbo_analysis_result_spans`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct turbo_pii_span {
    pub span_index: u32,
    pub detector_id_offset: u32,
    pub detector_id_len: u32,
    pub detector_version_offset: u32,
    pub detector_version_len: u32,
    pub fact_start: u32,
    pub fact_end: u32,
    pub score_kind: u32,
    pub score_calibrated: u32,
    pub score_model_id_offset: u32,
    pub score_model_id_len: u32,
    pub score_model_version_offset: u32,
    pub score_model_version_len: u32,
    pub score_value: f32,
}

pub const PII_SPAN_INTS: usize = 14;
const _: () = assert!(std::mem::offset_of!(turbo_pii_span, score_value) == (PII_SPAN_INTS - 1) * 4);
const _: () = assert!(size_of::<turbo_pii_span>() == PII_SPAN_INTS * 4);
const _: () = assert!(align_of::<turbo_pii_span>() == 4);

/// One PII candidate row emitted under `TURBO_ANALYZER_PII_DETAIL`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct turbo_pii_candidate {
    pub item_index: u32,
    pub fragment_index: u32,
    pub view_id: u32,
    pub kind_code: u32,
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
    pub instance_offset: u32,
    pub instance_len: u32,
    pub pii_instance_offset: u32,
    pub pii_instance_len: u32,
    pub label_offset: u32,
    pub label_len: u32,
    pub detector_id_offset: u32,
    pub detector_id_len: u32,
    pub detector_version_offset: u32,
    pub detector_version_len: u32,
    pub fact_start: u32,
    pub fact_end: u32,
    pub disposition: u32,
    pub disposition_a: u32,
    pub disposition_b: u32,
    pub disposition_c: u32,
    pub score_kind: u32,
    pub score_calibrated: u32,
    pub score_model_id_offset: u32,
    pub score_model_id_len: u32,
    pub score_model_version_offset: u32,
    pub score_model_version_len: u32,
    pub score_value: f32,
}

pub const PII_CANDIDATE_INTS: usize = 39;
const _: () = assert!(std::mem::offset_of!(turbo_pii_candidate, score_value) == (PII_CANDIDATE_INTS - 1) * 4);
const _: () = assert!(size_of::<turbo_pii_candidate>() == PII_CANDIDATE_INTS * 4);
const _: () = assert!(align_of::<turbo_pii_candidate>() == 4);

#[derive(Clone, Copy)]
struct ItemMeta {
    label: PoolRef,
    detector_id: PoolRef,
    detector_version: PoolRef,
    fact_start: u32,
    fact_end: u32,
    score_kind: u32,
    score_calibrated: u32,
    score_model_id: PoolRef,
    score_model_version: PoolRef,
    score_value: f32,
}

fn index_u32(index: usize, what: &str) -> Result<u32> {
    u32::try_from(index).map_err(|_| Error::new(CAPACITY, format!("{what} exceeds u32")))
}

fn fact_row(fact: PiiEvidence) -> turbo_pii_fact {
    let (kind, a, b, c) = match fact {
        PiiEvidence::Shape => (TURBO_PII_EVIDENCE_SHAPE, 0, 0, 0),
        PiiEvidence::Checksum { algorithm, passed } => {
            (TURBO_PII_EVIDENCE_CHECKSUM, u32::from(algorithm.id()), u32::from(passed), 0)
        }
        PiiEvidence::Label { span, phrase } => (TURBO_PII_EVIDENCE_LABEL, span.start, span.end, phrase),
        PiiEvidence::DenyHit { list, term } => (TURBO_PII_EVIDENCE_DENY_HIT, list, term, 0),
        PiiEvidence::Rule { rule } => (TURBO_PII_EVIDENCE_RULE, rule, 0, 0),
    };
    turbo_pii_fact { kind, a, b, c }
}

fn disposition_fields(disposition: PiiDisposition) -> (u32, u32, u32, u32) {
    match disposition {
        PiiDisposition::Kept { mention } => (TURBO_PII_DISPOSITION_KEPT, mention.get() as u32, 0, 0),
        PiiDisposition::AllowedByRule { scope, rule } => {
            let scope_id = if matches!(scope, AllowScope::Global) {
                TURBO_PII_ALLOW_SCOPE_GLOBAL
            } else {
                TURBO_PII_ALLOW_SCOPE_KIND
            };
            let (rule_kind, rule_index) = match rule {
                AllowRule::Term(i) => (TURBO_PII_ALLOW_RULE_TERM, i),
                AllowRule::Pattern(i) => (TURBO_PII_ALLOW_RULE_PATTERN, i),
            };
            (TURBO_PII_DISPOSITION_ALLOWED_BY_RULE, scope_id, rule_kind, rule_index)
        }
        PiiDisposition::NestedIn { mention } => (TURBO_PII_DISPOSITION_NESTED_IN, mention.get() as u32, 0, 0),
        PiiDisposition::RejectedByRule { reason } => {
            let id = if matches!(reason, RejectReason::LabelRequired) {
                TURBO_PII_REJECT_LABEL_REQUIRED
            } else {
                TURBO_PII_REJECT_THRESHOLD
            };
            (TURBO_PII_DISPOSITION_REJECTED_BY_RULE, id, 0, 0)
        }
        PiiDisposition::Repeats { mention } => (TURBO_PII_DISPOSITION_REPEATS, mention.get() as u32, 0, 0),
    }
}

type ItemInputs<'a> =
    (Option<&'a str>, &'a PiiDetectors, turbo_analysis_layers::PiiDetectorRef, &'a [PiiEvidence], Option<PiiScore>);

impl SpanPoolBuilder {
    fn item_meta(
        &mut self,
        what: &str,
        item: usize,
        inputs: ItemInputs<'_>,
        invalid: &impl Fn(String) -> Error,
    ) -> Result<ItemMeta> {
        let (kind_name, detectors, detector, evidence, score) = inputs;
        let name =
            kind_name.ok_or_else(|| invalid(format!("{what} {item} has a custom kind outside its layer's table")))?;
        let label = self.string_ref(name)?;
        let (det_id, det_ver) = detectors
            .get(detector)
            .ok_or_else(|| invalid(format!("{what} {item} has a detector outside its layer's table")))?;
        let (detector_id, detector_version) = (self.string_ref(det_id)?, self.string_ref(det_ver)?);
        let fact_start = index_u32(self.pii_facts.len(), "PII evidence fact count")?;
        let len = index_u32(evidence.len(), "PII evidence fact count")?;
        let fact_end =
            fact_start.checked_add(len).ok_or_else(|| Error::new(CAPACITY, "PII evidence fact count exceeds u32"))?;
        self.pii_facts.extend(evidence.iter().copied().map(fact_row));
        let (score_kind, score_calibrated, score_model_id, score_model_version, score_value) = match score {
            None => (0, 0, PoolRef::default(), PoolRef::default(), f32::NAN),
            Some(s) => {
                let (id, ver) = detectors.get(s.model).ok_or_else(|| {
                    invalid(format!("PII score model index {} is outside the detector table", s.model.index()))
                })?;
                (u32::from(s.kind.id()), u32::from(s.calibrated), self.string_ref(id)?, self.string_ref(ver)?, s.value)
            }
        };
        Ok(ItemMeta {
            label,
            detector_id,
            detector_version,
            fact_start,
            fact_end,
            score_kind,
            score_calibrated,
            score_model_id,
            score_model_version,
            score_value,
        })
    }
}

pub(crate) fn extract_pii_layer(
    builder: &mut SpanPoolBuilder,
    layer: &Pii,
    instance: &str,
    view_id: u32,
    pii: PiiRows,
    invalid: impl Fn(String) -> Error,
    coords: impl Fn(Span) -> Result<(CoordRange, CoordRange)>,
) -> Result<()> {
    layer.check().map_err(&invalid)?;
    let layer_ref = builder.layer_ref(TURBO_ANALYSIS_LAYER_PII, Pii::ID, instance, view_id)?;
    for item in 0..layer.spans.len() {
        let item_index = index_u32(item, "PII mention count")?;
        let m = layer.get(Ix::new(item_index));
        let meta = builder.item_meta(
            "PII mention",
            item,
            (layer.kind_name(m.kind), &layer.detectors, m.detector, m.evidence, m.score),
            &invalid,
        )?;
        let value = if pii.values { Some(builder.string_ref(m.normalized)?) } else { None };
        let classification = match m.confidence() {
            PiiConfidence::Format => TURBO_PII_CONFIDENCE_FORMAT,
            PiiConfidence::Checksum => TURBO_PII_CONFIDENCE_CHECKSUM,
            PiiConfidence::LabelledChecksum => TURBO_PII_CONFIDENCE_LABELLED_CHECKSUM,
            PiiConfidence::Labelled => TURBO_PII_CONFIDENCE_LABELLED,
        };
        let pii_kind = pii.kind_ids.then(|| m.kind.code());
        for (fragment, span) in layer.spans.fragments(Ix::new(item_index)).enumerate() {
            let (view, source) = coords(span)?;
            let span_index = index_u32(builder.spans.len(), "analysis span count")?;
            builder.push(
                layer_ref,
                SpanDraft {
                    item_index,
                    fragment_index: index_u32(fragment, "PII fragment count")?,
                    classification,
                    view: Some(view),
                    source: Some(source),
                    label: Some(meta.label),
                    value,
                    pii_kind,
                    ..SpanDraft::default()
                },
            )?;
            builder.pii_spans.push(turbo_pii_span {
                span_index,
                detector_id_offset: meta.detector_id.offset,
                detector_id_len: meta.detector_id.len,
                detector_version_offset: meta.detector_version.offset,
                detector_version_len: meta.detector_version.len,
                fact_start: meta.fact_start,
                fact_end: meta.fact_end,
                score_kind: meta.score_kind,
                score_calibrated: meta.score_calibrated,
                score_model_id_offset: meta.score_model_id.offset,
                score_model_id_len: meta.score_model_id.len,
                score_model_version_offset: meta.score_model_version.offset,
                score_model_version_len: meta.score_model_version.len,
                score_value: meta.score_value,
            });
        }
    }
    Ok(())
}

pub(crate) fn extract_pii_candidates_layer(
    builder: &mut SpanPoolBuilder,
    layer: &PiiCandidates,
    (instance, pii_instance): (&str, &str),
    view_id: u32,
    invalid: impl Fn(String) -> Error,
    coords: impl Fn(Span) -> Result<(CoordRange, CoordRange)>,
) -> Result<()> {
    layer.check().map_err(&invalid)?;
    let instance_ref = builder.intern(instance)?;
    let pii_instance_ref = builder.intern(pii_instance)?;
    for item in 0..layer.spans.len() {
        let item_index = index_u32(item, "PII candidate count")?;
        let c = layer.get(Ix::new(item_index));
        let meta = builder.item_meta(
            "PII candidate",
            item,
            (layer.kind_name(c.kind), &layer.detectors, c.detector, c.evidence, c.score),
            &invalid,
        )?;
        let (disposition, disposition_a, disposition_b, disposition_c) = disposition_fields(c.disposition);
        let kind_code = c.kind.code();
        for (fragment, span) in layer.spans.fragments(Ix::new(item_index)).enumerate() {
            if builder.pii_candidates.len() >= u32::MAX as usize {
                return Err(Error::new(CAPACITY, "PII candidate row count exceeds u32"));
            }
            let (view, source) = coords(span)?;
            builder.pii_candidates.push(turbo_pii_candidate {
                item_index,
                fragment_index: index_u32(fragment, "PII candidate fragment count")?,
                view_id,
                kind_code,
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
                instance_offset: instance_ref.offset,
                instance_len: instance_ref.len,
                pii_instance_offset: pii_instance_ref.offset,
                pii_instance_len: pii_instance_ref.len,
                label_offset: meta.label.offset,
                label_len: meta.label.len,
                detector_id_offset: meta.detector_id.offset,
                detector_id_len: meta.detector_id.len,
                detector_version_offset: meta.detector_version.offset,
                detector_version_len: meta.detector_version.len,
                fact_start: meta.fact_start,
                fact_end: meta.fact_end,
                disposition,
                disposition_a,
                disposition_b,
                disposition_c,
                score_kind: meta.score_kind,
                score_calibrated: meta.score_calibrated,
                score_model_id_offset: meta.score_model_id.offset,
                score_model_id_len: meta.score_model_id.len,
                score_model_version_offset: meta.score_model_version.offset,
                score_model_version_len: meta.score_model_version.len,
                score_value: meta.score_value,
            });
        }
    }
    Ok(())
}

unsafe fn borrow_result_slice<T>(
    r: *const turbo_analysis_result,
    out: *mut *const T,
    out_name: &'static str,
    count: *mut u32,
    what: &str,
    err: *mut turbo_error,
    get: impl FnOnce(&turbo_analysis_result) -> &[T],
) -> i32 {
    unsafe {
        call(err, || {
            let out = out_ptr(out, out_name)?;
            *out = std::ptr::null();
            let count = out_ptr(count, "count")?;
            *count = 0;
            let slice = get(analysis_result_handle(r)?);
            *count = index_u32(slice.len(), what)?;
            *out = if slice.is_empty() { std::ptr::null() } else { slice.as_ptr() };
            Ok(())
        })
    }
}

/// # Safety
/// Pointers are NULL or valid for the call, as `turbo.h` says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_analysis_result_pii_spans(
    r: *const turbo_analysis_result,
    pii_spans: *mut *const turbo_pii_span,
    count: *mut u32,
    err: *mut turbo_error,
) -> i32 {
    unsafe { borrow_result_slice(r, pii_spans, "pii_spans", count, "PII span count", err, |r| &r.inner.pii_spans) }
}

/// # Safety
/// Pointers are NULL or valid for the call, as `turbo.h` says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_analysis_result_pii_facts(
    r: *const turbo_analysis_result,
    facts: *mut *const turbo_pii_fact,
    count: *mut u32,
    err: *mut turbo_error,
) -> i32 {
    unsafe { borrow_result_slice(r, facts, "facts", count, "PII evidence fact count", err, |r| &r.inner.pii_facts) }
}

/// # Safety
/// Pointers are NULL or valid for the call, as `turbo.h` says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_analysis_result_pii_candidates(
    r: *const turbo_analysis_result,
    candidates: *mut *const turbo_pii_candidate,
    count: *mut u32,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        borrow_result_slice(r, candidates, "candidates", count, "PII candidate count", err, |r| &r.inner.pii_candidates)
    }
}
