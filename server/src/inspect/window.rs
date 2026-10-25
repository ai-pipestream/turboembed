//! Windowed viewport pagination, sweep-line combined elementary segmentation,
//! and document-level chip extraction (`InspectDocumentWindow`, issue #1220).

use std::collections::BTreeMap;

use turbo_analysis::{Document, Ix, Key, Layer, Span, Spans, View, ViewId};
use turbo_analysis_layers::{
    Abbreviations, Boilerplate, Categories, Chunks, EmotionDimension, EmotionLayer, Entities, Incivility, Language,
    LanguageSpans, Numbers, Pii, PiiConfidence, PosTags, Regions, Sentences, Structure, Subwords, Tag, Tokens,
};

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
    layer: &str,
    label: String,
    score: f64,
) {
    let ix = Ix::<L>::new(idx as u32);
    for frag in spans.fragments(ix) {
        let s = frag.start.max(window.start);
        let e = frag.end.min(window.end);
        if s < e {
            out.push(ClippedFrag {
                span: Span::new(s, e),
                ann: SegmentAnnotationProto {
                    layer: layer.to_owned(),
                    instance: String::new(),
                    item: idx as u32,
                    label: label.clone(),
                    score,
                },
            });
        }
    }
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
    if window.is_empty() {
        return Ok(Vec::new());
    }
    let mut clipped: Vec<ClippedFrag> = Vec::new();

    if layer_allowed(filter_layers, Sentences::ID)
        && let Some(sentences) = doc.get::<Sentences>()
        && sentences.spans.view() == view
    {
        for i in ordered_range(&sentences.spans, window) {
            push_clipped_item(&mut clipped, &sentences.spans, i, window, Sentences::ID, format!("S{i}"), 1.0);
        }
    }

    if layer_allowed(filter_layers, Tokens::ID)
        && let Some(tokens) = doc.get::<Tokens>()
        && tokens.spans.view() == view
    {
        let tok_key = Key::<Tokens>::DEFAULT.id();
        let pos_key = Key::<PosTags>::DEFAULT.id();
        let pos_bound_to_default =
            doc.made_on(pos_key).iter().find(|k| k.ty == tok_key.ty).is_none_or(|&k| k == tok_key);
        let pos = if pos_bound_to_default { doc.get::<PosTags>() } else { None };
        for i in 0..tokens.spans.len() {
            let (label, score) = match pos {
                Some(p) if i < p.tags.len() => {
                    let lbl = p.tagset.label(p.tags[i]).to_owned();
                    let sc = p.prob.as_ref().and_then(|pr| pr.get(i).copied()).unwrap_or(1.0);
                    (lbl, sc)
                }
                _ => ("token".to_owned(), 1.0),
            };
            push_clipped_item(&mut clipped, &tokens.spans, i, window, Tokens::ID, label, score);
        }
    }

    // Subwords uses Overlap::Any, so its ends column is not guaranteed monotonic;
    // scan items directly rather than binary-searching ends.
    if layer_allowed(filter_layers, Subwords::ID)
        && let Some(subwords) = doc.get::<Subwords>()
        && subwords.spans.view() == view
    {
        for i in 0..subwords.spans.len() {
            push_clipped_item(&mut clipped, &subwords.spans, i, window, Subwords::ID, "subword".to_owned(), 1.0);
        }
    }

    if layer_allowed(filter_layers, Structure::ID)
        && let Some(structure) = doc.get::<Structure>()
        && structure.spans.view() == view
    {
        for i in 0..structure.spans.len() {
            let label = format!("{:?}", structure.kind[i]);
            push_clipped_item(&mut clipped, &structure.spans, i, window, Structure::ID, label, 1.0);
        }
    }

    if layer_allowed(filter_layers, Entities::ID)
        && let Some(entities) = doc.get::<Entities>()
        && entities.spans.view() == view
    {
        for i in 0..entities.spans.len() {
            let label = entities.kinds.label(entities.kind[i]).to_owned();
            let score = entities.score.get(i).copied().unwrap_or(1.0);
            push_clipped_item(&mut clipped, &entities.spans, i, window, Entities::ID, label, score);
        }
    }

    if layer_allowed(filter_layers, Chunks::ID)
        && let Some(chunks) = doc.get::<Chunks>()
        && chunks.spans.view() == view
    {
        for i in 0..chunks.spans.len() {
            let label = format!("L{}:{:?}", chunks.level[i], chunks.cut[i]);
            let score = chunks.cost[i] as f64;
            push_clipped_item(&mut clipped, &chunks.spans, i, window, Chunks::ID, label, score);
        }
    }

    if layer_allowed(filter_layers, Pii::ID)
        && let Some(pii) = doc.get::<Pii>()
        && pii.spans.view() == view
    {
        for i in 0..pii.spans.len() {
            let label = pii.kind[i].name().to_owned();
            let score = match pii.confidence[i] {
                PiiConfidence::Format => 0.5,
                PiiConfidence::Checksum => 0.85,
                PiiConfidence::LabelledChecksum => 1.0,
            };
            push_clipped_item(&mut clipped, &pii.spans, i, window, Pii::ID, label, score);
        }
    }

    if layer_allowed(filter_layers, Numbers::ID)
        && let Some(numbers) = doc.get::<Numbers>()
        && numbers.spans.view() == view
    {
        for i in 0..numbers.spans.len() {
            let label = format!("{:?}", numbers.notation[i]);
            push_clipped_item(&mut clipped, &numbers.spans, i, window, Numbers::ID, label, 1.0);
        }
    }

    if layer_allowed(filter_layers, LanguageSpans::ID)
        && let Some(langs) = doc.get::<LanguageSpans>()
        && langs.spans.view() == view
    {
        for i in 0..langs.spans.len() {
            let m = langs.get(Ix::new(i as u32));
            push_clipped_item(
                &mut clipped,
                &langs.spans,
                i,
                window,
                LanguageSpans::ID,
                m.code.to_owned(),
                m.confidence,
            );
        }
    }

    if layer_allowed(filter_layers, Abbreviations::ID)
        && let Some(abbr) = doc.get::<Abbreviations>()
        && abbr.spans.view() == view
    {
        for i in 0..abbr.spans.len() {
            let label = format!("{:?}", abbr.form[i]);
            push_clipped_item(&mut clipped, &abbr.spans, i, window, Abbreviations::ID, label, 1.0);
        }
    }

    if layer_allowed(filter_layers, Boilerplate::<Sentences>::ID)
        && let Some(bp) = doc.get::<Boilerplate<Sentences>>()
        && let Some(sentences) = doc.get::<Sentences>()
        && sentences.spans.view() == view
    {
        for i in 0..bp.kind.len().min(sentences.spans.len()) {
            if let Some(kind) = bp.kind[i] {
                let label = format!("{kind:?}");
                push_clipped_item(
                    &mut clipped,
                    &sentences.spans,
                    i,
                    window,
                    Boilerplate::<Sentences>::ID,
                    label,
                    bp.score[i] as f64,
                );
            }
        }
    }

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
