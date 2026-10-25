//! Multi-lane document heatmaps (`HeatmapLaneProto`, issue #1220) in both
//! exact-segment mode (`heatmap_buckets == 0`) and `N`-bucket minimap mode
//! (`heatmap_buckets > 0`), with all bucket boundaries projected into the
//! caller's [`OffsetMode`].

use std::collections::BTreeMap;

use turbo_analysis::{Document, Ix, Span, ViewId};
use turbo_analysis_layers::{
    Categories, Chunks, Entities, Incivility, IncivilityTarget, Pii, PiiConfidence, Sentences, Tag, Tokens,
};

use super::window::snap_down;
use crate::api::Failure;
use crate::offsets::Offsetter;
use crate::proto::result::{HeatmapBucketProto, HeatmapLaneProto};
use crate::results::{MAX_FRAGMENTS, over};

/// Maximum number of minimap buckets allowed per heatmap lane.
pub const MAX_HEATMAP_BUCKETS: u32 = 4_096;

#[derive(Clone, Copy)]
struct ScoredSpan {
    span: Span,
    score: f32,
    item: u32,
}

/// Builds multi-lane heatmaps for `view` of `doc`, in exact-segment mode when
/// `heatmap_buckets == 0` or binned into `heatmap_buckets` equal-width minimap
/// buckets when `heatmap_buckets > 0`.
pub(crate) fn build_heatmap_lanes(
    doc: &Document,
    view: ViewId,
    text: &str,
    offsetter: &Offsetter,
    heatmap_buckets: u32,
) -> Result<Vec<HeatmapLaneProto>, Failure> {
    if heatmap_buckets > MAX_HEATMAP_BUCKETS {
        return Err(over("max_heatmap_buckets", "heatmap_buckets", heatmap_buckets as u64));
    }
    let mut lanes = Vec::new();

    if let Some(tokens) = doc.get::<Tokens>()
        && tokens.spans.view() == view
        && !tokens.spans.is_empty()
    {
        let items: Vec<ScoredSpan> = (0..tokens.spans.len())
            .map(|i| ScoredSpan { span: tokens.spans.get(Ix::new(i as u32)), score: 1.0, item: i as u32 })
            .collect();
        lanes.push(make_lane("tokens", "TokenDensity", view.0 as u32, text, offsetter, heatmap_buckets, &items, true)?);
    }

    if let Some(entities) = doc.get::<Entities>()
        && entities.spans.view() == view
        && !entities.spans.is_empty()
    {
        let mut items = Vec::new();
        for i in 0..entities.spans.len() {
            let ix = Ix::new(i as u32);
            let sc = entities.score.get(i).copied().unwrap_or(1.0) as f32;
            for frag in entities.spans.fragments(ix) {
                items.push(ScoredSpan { span: frag, score: sc, item: i as u32 });
            }
        }
        lanes.push(make_lane(
            "entities",
            "EntityDensity",
            view.0 as u32,
            text,
            offsetter,
            heatmap_buckets,
            &items,
            false,
        )?);
    }

    if let Some(pii) = doc.get::<Pii>()
        && pii.spans.view() == view
        && !pii.spans.is_empty()
    {
        let mut items = Vec::new();
        for i in 0..pii.spans.len() {
            let ix = Ix::new(i as u32);
            let sc = match pii.confidence[i] {
                PiiConfidence::Format => 0.5,
                PiiConfidence::Checksum => 0.85,
                PiiConfidence::LabelledChecksum => 1.0,
            };
            for frag in pii.spans.fragments(ix) {
                items.push(ScoredSpan { span: frag, score: sc, item: i as u32 });
            }
        }
        lanes.push(make_lane("pii", "PiiDensity", view.0 as u32, text, offsetter, heatmap_buckets, &items, false)?);
    }

    if let Some(sentences) = doc.get::<Sentences>()
        && sentences.spans.view() == view
        && let Some(cats) = doc.get::<Categories<Sentences>>()
        && !cats.category.is_empty()
    {
        let mut items = Vec::new();
        for i in 0..cats.category.len().min(sentences.spans.len()) {
            let tag = cats.category[i];
            if tag == Tag::NONE {
                continue;
            }
            let n_cats = cats.categories.len().max(1);
            let row_scores = &cats.scores[i * n_cats..(i + 1) * n_cats];
            let best_score = row_scores.get(tag.0 as usize).copied().unwrap_or(0.0) as f32;
            let lbl = cats.categories.label(tag).to_ascii_lowercase();
            let signed = if lbl.contains("neg") { -best_score } else { best_score };
            items.push(ScoredSpan { span: sentences.spans.get(Ix::new(i as u32)), score: signed, item: i as u32 });
        }
        if !items.is_empty() {
            lanes.push(make_lane(
                "sentiment",
                "SentimentPolarity",
                view.0 as u32,
                text,
                offsetter,
                heatmap_buckets,
                &items,
                false,
            )?);
        }
    }

    if let Some(inciv) = doc.get::<Incivility>()
        && !inciv.target.is_empty()
    {
        let sentences = doc.get::<Sentences>();
        let chunks = doc.get::<Chunks>();
        let mut items = Vec::new();
        for (i, &t) in inciv.target.iter().enumerate() {
            let span = match t {
                IncivilityTarget::Sentence(ix) => sentences
                    .filter(|s| s.spans.view() == view)
                    .and_then(|s| (ix.get() < s.spans.len()).then(|| s.spans.get(ix))),
                IncivilityTarget::Chunk(ix) => chunks
                    .filter(|c| c.spans.view() == view)
                    .and_then(|c| (ix.get() < c.spans.len()).then(|| c.spans.get(ix))),
                IncivilityTarget::Block(_) => None,
            };
            if let Some(sp) = span {
                let sc = [
                    inciv.insult[i],
                    inciv.profanity[i],
                    inciv.threat[i],
                    inciv.identity_attack[i],
                    inciv.intolerance[i],
                    inciv.severe_toxicity[i],
                ]
                .into_iter()
                .fold(0.0_f64, f64::max) as f32;
                items.push(ScoredSpan { span: sp, score: sc, item: i as u32 });
            }
        }
        if !items.is_empty() {
            lanes.push(make_lane(
                "incivility",
                "IncivilityScore",
                view.0 as u32,
                text,
                offsetter,
                heatmap_buckets,
                &items,
                false,
            )?);
        }
    }

    if let Some(chunks) = doc.get::<Chunks>()
        && chunks.spans.view() == view
        && !chunks.spans.is_empty()
    {
        let mut items = Vec::new();
        for i in 0..chunks.spans.len() {
            let ix = Ix::new(i as u32);
            let sc = chunks.model_tokens[i] as f32;
            for frag in chunks.spans.fragments(ix) {
                items.push(ScoredSpan { span: frag, score: sc, item: i as u32 });
            }
        }
        lanes.push(make_lane("chunks", "ChunkScore", view.0 as u32, text, offsetter, heatmap_buckets, &items, false)?);
    }

    Ok(lanes)
}

#[allow(clippy::too_many_arguments)]
fn make_lane(
    id: &str,
    kind: &str,
    view: u32,
    text: &str,
    offsetter: &Offsetter,
    heatmap_buckets: u32,
    items: &[ScoredSpan],
    sum_density: bool,
) -> Result<HeatmapLaneProto, Failure> {
    let buckets = if heatmap_buckets == 0 {
        exact_segments(text, offsetter, items, sum_density)?
    } else {
        binned_buckets(text, offsetter, heatmap_buckets, items, sum_density)
    };
    let mut min_score = 0.0_f32;
    let mut max_score = 0.0_f32;
    for (idx, b) in buckets.iter().enumerate() {
        if idx == 0 {
            min_score = b.score;
            max_score = b.score;
        } else {
            min_score = min_score.min(b.score);
            max_score = max_score.max(b.score);
        }
    }
    Ok(HeatmapLaneProto { id: id.to_owned(), kind: kind.to_owned(), view, min_score, max_score, buckets })
}

/// Sweep-line exact elementary segmentation: sorts endpoints once and maintains
/// an active map of open [`ScoredSpan`]s so cost is linear in the number of
/// endpoints (`O(M log M)`).
fn exact_segments(
    text: &str,
    offsetter: &Offsetter,
    items: &[ScoredSpan],
    sum_density: bool,
) -> Result<Vec<HeatmapBucketProto>, Failure> {
    // (byte_pos, is_start, idx_in_items)
    let mut events: Vec<(u32, bool, usize)> = Vec::with_capacity(items.len() * 2);
    for (idx, it) in items.iter().enumerate() {
        if !it.span.is_empty() {
            events.push((it.span.start, true, idx));
            events.push((it.span.end, false, idx));
        }
    }
    if events.len() as u64 > MAX_FRAGMENTS * 2 {
        return Err(over("max_fragments", "heatmap span endpoints", events.len() as u64));
    }
    events.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));

    let mut active: BTreeMap<usize, ScoredSpan> = BTreeMap::new();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < events.len() {
        let pos = events[i].0;
        while i < events.len() && events[i].0 == pos {
            let (_, is_start, idx) = events[i];
            if is_start {
                active.insert(idx, items[idx]);
            } else {
                active.remove(&idx);
            }
            i += 1;
        }
        if i < events.len() && !active.is_empty() {
            let next_pos = events[i].0;
            if pos < next_pos {
                let count = active.len() as u32;
                let mut best_score = 0.0_f32;
                let mut primary_item: Option<u32> = None;
                for it in active.values() {
                    if sum_density {
                        best_score += it.score;
                        if primary_item.is_none() {
                            primary_item = Some(it.item);
                        }
                    } else if primary_item.is_none() || it.score.abs() > best_score.abs() {
                        best_score = it.score;
                        primary_item = Some(it.item);
                    }
                }
                let (start, end) = offsetter.to_wire(text, Span::new(pos, next_pos));
                out.push(HeatmapBucketProto { start, end, score: best_score, count, primary_item });
            }
        }
    }
    Ok(out)
}

fn binned_buckets(
    text: &str,
    offsetter: &Offsetter,
    num_buckets: u32,
    items: &[ScoredSpan],
    sum_density: bool,
) -> Vec<HeatmapBucketProto> {
    let len = text.len();
    if len == 0 || num_buckets == 0 {
        return Vec::new();
    }
    let n = num_buckets as usize;
    let mut bounds = Vec::with_capacity(n + 1);
    for b_idx in 0..n {
        bounds.push(snap_down(text, (b_idx * len) / n) as u32);
    }
    bounds.push(len as u32);

    let mut counts = vec![0u32; n];
    let mut scores = vec![0.0_f32; n];
    let mut primaries: Vec<Option<u32>> = vec![None; n];

    for it in items {
        if it.span.is_empty() {
            continue;
        }
        let first_b = bounds[1..].partition_point(|&b_end| b_end <= it.span.start);
        let last_b = bounds[..n].partition_point(|&b_start| b_start < it.span.end);
        for b_idx in first_b..last_b.min(n) {
            let a = bounds[b_idx];
            let b = bounds[b_idx + 1];
            if it.span.start < b && it.span.end > a {
                counts[b_idx] += 1;
                if sum_density {
                    scores[b_idx] += it.score;
                    if primaries[b_idx].is_none() {
                        primaries[b_idx] = Some(it.item);
                    }
                } else if primaries[b_idx].is_none() || it.score.abs() > scores[b_idx].abs() {
                    scores[b_idx] = it.score;
                    primaries[b_idx] = Some(it.item);
                }
            }
        }
    }

    let mut out = Vec::with_capacity(n);
    for b_idx in 0..n {
        let a = bounds[b_idx];
        let b = bounds[b_idx + 1].max(a);
        let (start, end) = offsetter.to_wire(text, Span::new(a, b));
        out.push(HeatmapBucketProto {
            start,
            end,
            score: scores[b_idx],
            count: counts[b_idx],
            primary_item: primaries[b_idx],
        });
    }
    out
}
