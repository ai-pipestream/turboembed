//! Multi-lane document heatmaps (`HeatmapLaneProto`, issue #1220) in both
//! exact-segment mode (`heatmap_buckets == 0`) and `N`-bucket minimap mode
//! (`heatmap_buckets > 0`), with all bucket boundaries projected into the
//! caller's [`OffsetMode`].

use std::any::TypeId;

use turbo_analysis::{Document, Ix, Key, Span, ViewId};
use turbo_analysis_layers::{
    Categories, Chunks, Entities, Incivility, IncivilityTarget, Pii, PiiConfidence, Sentences, Tag, Tokens,
};

mod aggregate;
use aggregate::ActiveScores;

use super::bad;
use super::window::snap_down;
use crate::api::Failure;
use crate::offsets::Offsetter;
use crate::proto::result::{HeatmapBucketProto, HeatmapLaneProto};
use crate::results::{MAX_FRAGMENTS, over};

#[cfg(test)]
std::thread_local! {
    static AGGREGATE_QUERIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
mod tests;

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
    for key in doc.layers() {
        let instance = key.name;

        if key.ty == TypeId::of::<Tokens>()
            && let Some(tokens) = doc.get_key(&Key::<Tokens>::named(instance))
            && tokens.spans.view() == view
            && !tokens.spans.is_empty()
        {
            let items: Vec<ScoredSpan> = (0..tokens.spans.len())
                .map(|i| ScoredSpan { span: tokens.spans.get(Ix::new(i as u32)), score: 1.0, item: i as u32 })
                .collect();
            lanes.push(make_lane(
                &lane_id("tokens", instance),
                "TokenDensity",
                view.0 as u32,
                text,
                offsetter,
                heatmap_buckets,
                &items,
                true,
            )?);
        }

        if key.ty == TypeId::of::<Entities>()
            && let Some(entities) = doc.get_key(&Key::<Entities>::named(instance))
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
                &lane_id("entities", instance),
                "EntityDensity",
                view.0 as u32,
                text,
                offsetter,
                heatmap_buckets,
                &items,
                false,
            )?);
        }

        if key.ty == TypeId::of::<Pii>()
            && let Some(pii) = doc.get_key(&Key::<Pii>::named(instance))
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
            lanes.push(make_lane(
                &lane_id("pii", instance),
                "PiiDensity",
                view.0 as u32,
                text,
                offsetter,
                heatmap_buckets,
                &items,
                false,
            )?);
        }

        if key.ty == TypeId::of::<Categories<Sentences>>()
            && let Some(cats) = doc.get_key(&Key::<Categories<Sentences>>::named(instance))
            && !cats.category.is_empty()
            && let Some(anchor) = doc.made_on(key).iter().find(|k| k.ty == TypeId::of::<Sentences>())
            && let Some(sentences) = doc.get_key(&Key::<Sentences>::named(anchor.name))
            && sentences.spans.view() == view
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
                    &lane_id("sentiment", instance),
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

        if key.ty == TypeId::of::<Incivility>()
            && let Some(inciv) = doc.get_key(&Key::<Incivility>::named(instance))
            && !inciv.target.is_empty()
        {
            let bindings = doc.made_on(key);
            let sentences = bindings
                .iter()
                .find(|k| k.ty == TypeId::of::<Sentences>())
                .and_then(|k| doc.get_key(&Key::<Sentences>::named(k.name)));
            let chunks = bindings
                .iter()
                .find(|k| k.ty == TypeId::of::<Chunks>())
                .and_then(|k| doc.get_key(&Key::<Chunks>::named(k.name)));
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
                    &lane_id("incivility", instance),
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

        if key.ty == TypeId::of::<Chunks>()
            && let Some(chunks) = doc.get_key(&Key::<Chunks>::named(instance))
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
            lanes.push(make_lane(
                &lane_id("chunks", instance),
                "ChunkScore",
                view.0 as u32,
                text,
                offsetter,
                heatmap_buckets,
                &items,
                false,
            )?);
        }
    }
    lanes.sort_by(|a, b| lane_order(&a.kind).cmp(&lane_order(&b.kind)).then_with(|| a.id.cmp(&b.id)));
    Ok(lanes)
}

fn lane_order(kind: &str) -> u8 {
    match kind {
        "TokenDensity" => 0,
        "EntityDensity" => 1,
        "PiiDensity" => 2,
        "SentimentPolarity" => 3,
        "IncivilityScore" => 4,
        "ChunkScore" => 5,
        _ => 6,
    }
}

fn lane_id(kind: &str, instance: &str) -> String {
    if instance.is_empty() { kind.to_owned() } else { format!("{kind}@{instance}") }
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
    if items.len() as u64 > MAX_FRAGMENTS {
        return Err(over("max_fragments", "heatmap input fragments", items.len() as u64));
    }
    if items.iter().any(|item| !item.score.is_finite()) {
        return Err(bad("heatmap contains a non-finite score"));
    }
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

    let mut active = ActiveScores::new(items);
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < events.len() {
        let pos = events[i].0;
        while i < events.len() && events[i].0 == pos {
            let (_, is_start, idx) = events[i];
            if is_start {
                active.insert(idx);
            } else {
                active.remove(idx);
            }
            i += 1;
        }
        if i < events.len() && !active.is_empty() {
            let next_pos = events[i].0;
            if pos < next_pos {
                let (best_score, count, primary_item) = active.summary(sum_density);
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

    // Assign two events per input span, rather than visiting every bucket it
    // overlaps. Bucket boundaries can repeat at UTF-8 character boundaries;
    // the two partition points preserve the existing overlap rule there.
    let mut opens = vec![Vec::new(); n];
    let mut closes = vec![Vec::new(); n + 1];
    for (index, item) in items.iter().enumerate() {
        if item.span.is_empty() {
            continue;
        }
        let first = bounds[1..].partition_point(|&end| end <= item.span.start);
        let last = bounds[..n].partition_point(|&start| start < item.span.end);
        if first < last {
            opens[first].push(index);
            closes[last].push(index);
        }
    }
    let mut active = ActiveScores::new(items);
    let mut out = Vec::with_capacity(n);
    for b in 0..n {
        for &index in &closes[b] {
            active.remove(index);
        }
        for &index in &opens[b] {
            active.insert(index);
        }
        let (score, count, primary_item) = active.summary(sum_density);
        let (start, end) = offsetter.to_wire(text, Span::new(bounds[b], bounds[b + 1]));
        out.push(HeatmapBucketProto { start, end, score, count, primary_item });
    }
    out
}
