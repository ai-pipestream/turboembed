//! Windowed viewport pagination, sweep-line combined elementary segmentation,
//! and document-level chip extraction (`InspectDocumentWindow`, issue #1220).

use std::collections::BTreeMap;

mod layers;
#[cfg(test)]
mod tests;

use turbo_analysis::{Document, Ix, Layer, Span, Spans, View, ViewId};
use turbo_analysis_layers::{Categories, EmotionDimension, EmotionLayer, Incivility, Language, Regions, Tag};

use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT};
use crate::offsets::Offsetter;
use crate::proto::result::{CombinedSegmentProto, DocumentChipProto, DocumentWindowProto, SegmentAnnotationProto};
use crate::results::{MAX_FRAGMENTS, MAX_WINDOWS, over};

/// Default viewport window size in UTF-8 bytes when `page_size == 0`.
pub const DEFAULT_PAGE_SIZE: u32 = 16_000;

/// Maximum viewport window size in UTF-8 bytes allowed on `InspectDocumentWindow`.
pub const MAX_PAGE_SIZE: u32 = 65_536;

fn bad(message: impl Into<String>) -> Failure {
    Failure::new(TURBO_E_INVALID_ARGUMENT, 0, message)
}

pub(crate) fn snap_down(text: &str, mut at: usize) -> usize {
    at = at.min(text.len());
    while at > 0 && !text.is_char_boundary(at) {
        at -= 1;
    }
    at
}

/// Computes the character-aligned [`DocumentWindowProto`] and internal byte
/// [`Span`] for `page` of `view`.
pub(crate) fn compute_window(
    text: &str,
    view: u32,
    offsetter: &Offsetter,
    page_size: u32,
    page: u32,
) -> Result<(DocumentWindowProto, Span), Failure> {
    if page_size > MAX_PAGE_SIZE {
        return Err(over("max_page_size", "window page_size", page_size as u64));
    }
    let step = if page_size == 0 { DEFAULT_PAGE_SIZE } else { page_size } as usize;
    let total_bytes = text.len();
    let page_count = if total_bytes == 0 { 1 } else { total_bytes.div_ceil(step) as u32 };
    if page_count as u64 > MAX_WINDOWS {
        return Err(over("max_windows", "document pages", page_count as u64));
    }
    if page >= page_count {
        return Err(bad(format!("page {page} is out of range for view {view} with {page_count} page(s)")));
    }
    let raw_start = (page as usize).saturating_mul(step).min(total_bytes);
    let raw_end = raw_start.saturating_add(step);
    let clamped = raw_end > total_bytes;
    let byte_start = snap_down(text, raw_start) as u32;
    let byte_end = if raw_end >= total_bytes { total_bytes as u32 } else { snap_down(text, raw_end) as u32 };
    let span = Span::new(byte_start, byte_end.max(byte_start));
    let (start, end) = offsetter.to_wire(text, span);
    let proto = DocumentWindowProto {
        view,
        byte_start: span.start,
        byte_end: span.end,
        start,
        end,
        page,
        page_count,
        clamped,
        text: text[span.range()].to_owned(),
    };
    Ok((proto, span))
}

#[derive(Clone)]
struct ClippedFrag {
    span: Span,
    ann: SegmentAnnotationProto,
}

fn layer_allowed(filter_layers: &[String], layer_id: &str) -> bool {
    filter_layers.is_empty() || filter_layers.iter().any(|f| f.trim() == layer_id)
}

/// Binary-searches a strictly non-overlapping (`Overlap::None`) [`Spans`]
/// column for the index range of items whose spans may overlap `window`.
fn ordered_range<L>(spans: &Spans<L>, window: Span) -> std::ops::Range<usize> {
    let (starts, ends) = spans.columns();
    let first = ends.partition_point(|&e| e <= window.start);
    let last = starts.partition_point(|&s| s < window.end);
    first..last.max(first)
}

fn push_clipped_item<L>(
    out: &mut Vec<ClippedFrag>,
    spans: &Spans<L>,
    idx: usize,
    window: Span,
    identity: (&str, &str),
    label: String,
    score: f64,
) -> Result<(), Failure> {
    let ix = Ix::<L>::new(idx as u32);
    for frag in spans.fragments(ix) {
        let s = frag.start.max(window.start);
        let e = frag.end.min(window.end);
        if s < e {
            if out.len() as u64 >= MAX_FRAGMENTS {
                return Err(over("max_fragments", "clipped window fragments", out.len() as u64 + 1));
            }
            out.push(ClippedFrag {
                span: Span::new(s, e),
                ann: SegmentAnnotationProto {
                    layer: identity.0.to_owned(),
                    instance: identity.1.to_owned(),
                    item: idx as u32,
                    label: label.clone(),
                    score,
                },
            });
        }
    }
    Ok(())
}

/// Computes the combined elementary segmentation of all requested positional
/// layers on `view` inside `window` via a sweep-line over fragment endpoints.
pub(crate) fn combined_segments(
    doc: &Document,
    view: ViewId,
    text: &str,
    window: Span,
    offsetter: &Offsetter,
    filter_layers: &[String],
) -> Result<Vec<CombinedSegmentProto>, Failure> {
    combine_with_budget(doc, view, text, window, offsetter, filter_layers, MAX_FRAGMENTS)
}

fn combine_with_budget(
    doc: &Document,
    view: ViewId,
    text: &str,
    window: Span,
    offsetter: &Offsetter,
    filter_layers: &[String],
    max_memberships: u64,
) -> Result<Vec<CombinedSegmentProto>, Failure> {
    if window.is_empty() {
        return Ok(Vec::new());
    }
    let clipped = layers::collect_fragments(doc, view, window, filter_layers)?;

    if clipped.len() as u64 > MAX_FRAGMENTS {
        return Err(over("max_fragments", "clipped window fragments", clipped.len() as u64));
    }

    // Sweep-line events: (byte_pos, kind: 0 = close, 1 = open, 2 = window_bound, frag_idx).
    let mut events: Vec<(u32, u8, usize)> = Vec::with_capacity(clipped.len() * 2 + 2);
    events.push((window.start, 2, usize::MAX));
    events.push((window.end, 2, usize::MAX));
    for (idx, c) in clipped.iter().enumerate() {
        events.push((c.span.start, 1, idx));
        events.push((c.span.end, 0, idx));
    }
    events.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));

    let mut active_set: BTreeMap<usize, &SegmentAnnotationProto> = BTreeMap::new();
    let mut raw_segments: Vec<(Span, Vec<SegmentAnnotationProto>)> = Vec::new();
    let mut i = 0usize;
    let mut memberships = 0u64;
    while i < events.len() {
        let pos = events[i].0;
        while i < events.len() && events[i].0 == pos {
            let (_, kind, idx) = events[i];
            match kind {
                0 => {
                    active_set.remove(&idx);
                }
                1 => {
                    active_set.insert(idx, &clipped[idx].ann);
                }
                _ => {}
            }
            i += 1;
        }
        if i < events.len() {
            let next_pos = events[i].0;
            if pos >= window.start && next_pos <= window.end && pos < next_pos {
                if raw_segments.len() as u64 >= MAX_FRAGMENTS {
                    return Err(over("max_fragments", "combined window segments", raw_segments.len() as u64 + 1));
                }
                memberships = memberships.saturating_add(active_set.len() as u64);
                if memberships > max_memberships {
                    return Err(over("max_fragments", "combined segment annotation memberships", memberships));
                }
                let mut active: Vec<SegmentAnnotationProto> = active_set.values().map(|&a| a.clone()).collect();
                active.sort_by(|x, y| {
                    x.layer.cmp(&y.layer).then_with(|| x.instance.cmp(&y.instance)).then(x.item.cmp(&y.item))
                });
                if let Some((last_span, last_active)) = raw_segments.last_mut()
                    && last_span.end == pos
                    && *last_active == active
                {
                    last_span.end = next_pos;
                } else {
                    raw_segments.push((Span::new(pos, next_pos), active));
                }
            }
        }
    }

    let mut out = Vec::with_capacity(raw_segments.len());
    for (seg_span, annotations) in raw_segments {
        let (start, end) = offsetter.to_wire(text, seg_span);
        out.push(CombinedSegmentProto { start, end, text: text[seg_span.range()].to_owned(), annotations });
    }
    Ok(out)
}

/// Extracts document-level classification, language, region, incivility, and
/// emotion chips from `doc`.
pub(crate) fn document_chips(doc: &Document) -> Vec<DocumentChipProto> {
    let mut chips = Vec::new();

    if let Some(cats) = doc.get::<Categories<View>>()
        && !cats.category.is_empty()
    {
        let best = cats.category[0];
        if best != Tag::NONE {
            let label = cats.categories.label(best).to_owned();
            let score = cats.scores.get(best.0 as usize).copied().unwrap_or(0.0);
            chips.push(DocumentChipProto {
                layer: Categories::<View>::ID.to_owned(),
                instance: String::new(),
                label,
                score,
            });
        }
    }

    if let Some(lang) = doc.get::<Language>()
        && let Some((code, score)) = lang.best()
    {
        chips.push(DocumentChipProto {
            layer: Language::ID.to_owned(),
            instance: String::new(),
            label: code.to_owned(),
            score,
        });
    }

    if let Some(regions) = doc.get::<Regions>()
        && !regions.country.is_empty()
    {
        chips.push(DocumentChipProto {
            layer: Regions::ID.to_owned(),
            instance: String::new(),
            label: regions.country[0].as_str().to_owned(),
            score: regions.share.first().copied().unwrap_or(1.0),
        });
    }

    if let Some(inciv) = doc.get::<Incivility>()
        && !inciv.target.is_empty()
    {
        let max_tox = inciv.severe_toxicity.iter().copied().fold(0.0_f64, f64::max);
        chips.push(DocumentChipProto {
            layer: Incivility::ID.to_owned(),
            instance: String::new(),
            label: "severe_toxicity".to_owned(),
            score: max_tox,
        });
    }

    if let Some(emotions) = doc.get::<EmotionLayer>()
        && !emotions.scores.is_empty()
    {
        let mut best_idx = 0usize;
        let mut best_score = 0.0_f64;
        for (idx, &sc) in emotions.scores.iter().take(28).enumerate() {
            if sc > best_score {
                best_score = sc;
                best_idx = idx;
            }
        }
        let label = EmotionDimension::ALL[best_idx].name().to_owned();
        chips.push(DocumentChipProto {
            layer: EmotionLayer::ID.to_owned(),
            instance: String::new(),
            label,
            score: best_score,
        });
    }

    chips
}
