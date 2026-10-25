//! Multi-lane document heatmaps (`HeatmapLaneProto`, issue #1220) in both
//! exact-segment mode (`heatmap_buckets == 0`) and `N`-bucket minimap mode
//! (`heatmap_buckets > 0`), with all bucket boundaries projected into the
//! caller's [`OffsetMode`].

use turbo_analysis::{Document, Ix, Span, ViewId};
use turbo_analysis_layers::{
    Categories, Chunks, Entities, Incivility, IncivilityTarget, Pii, PiiConfidence, Sentences, Tag, Tokens,
};

use super::window::{snap_down, snap_up};
use crate::api::Failure;
use crate::offsets::Offsetter;
use crate::proto::result::{HeatmapBucketProto, HeatmapLaneProto};
use crate::results::{MAX_WINDOWS, over};

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
    if heatmap_buckets as u64 > MAX_WINDOWS {
        return Err(over("max_windows", "heatmap_buckets", heatmap_buckets as u64));
    }
    let mut lanes = Vec::new();

    if let Some(tokens) = doc.get::<Tokens>()
        && tokens.spans.view() == view
        && !tokens.spans.is_empty()
    {
        let items: Vec<ScoredSpan> = (0..tokens.spans.len())
            .map(|i| ScoredSpan { span: tokens.spans.get(Ix::new(i as u32)), score: 1.0, item: i as u32 })
            .collect();
        lanes.push(make_lane("tokens", "TokenDensity", view.0 as u32, text, offsetter, heatmap_buckets, &items, true));
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
        ));
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
        lanes.push(make_lane("pii", "PiiDensity", view.0 as u32, text, offsetter, heatmap_buckets, &items, false));
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
            ));
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
            ));
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
        lanes.push(make_lane("chunks", "ChunkScore", view.0 as u32, text, offsetter, heatmap_buckets, &items, false));
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
) -> HeatmapLaneProto {
    let buckets = if heatmap_buckets == 0 {
        exact_segments(text, offsetter, items, sum_density)
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
    HeatmapLaneProto { id: id.to_owned(), kind: kind.to_owned(), view, min_score, max_score, buckets }
}

fn exact_segments(
    text: &str,
    offsetter: &Offsetter,
    items: &[ScoredSpan],
    sum_density: bool,
) -> Vec<HeatmapBucketProto> {
    let mut bounds = Vec::with_capacity(items.len() * 2);
    for it in items {
        if !it.span.is_empty() {
            bounds.push(it.span.start);
            bounds.push(it.span.end);
        }
    }
    bounds.sort_unstable();
    bounds.dedup();

    let mut out = Vec::new();
    for pair in bounds.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        if a >= b {
            continue;
        }
        let mut count = 0u32;
        let mut best_score = 0.0_f32;
        let mut primary_item: Option<u32> = None;
        for it in items {
            if it.span.start <= a && it.span.end >= b {
                count += 1;
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
        }
        if count > 0 {
            let (start, end) = offsetter.to_wire(text, Span::new(a, b));
            out.push(HeatmapBucketProto { start, end, score: best_score, count, primary_item });
        }
    }
    out
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
    let mut out = Vec::with_capacity(n);
    for b_idx in 0..n {
        let raw_a = (b_idx * len) / n;
        let raw_b = ((b_idx + 1) * len) / n;
        let a = snap_down(text, raw_a) as u32;
        let b = snap_up(text, raw_b).max(a as usize) as u32;
        let mut count = 0u32;
        let mut score = 0.0_f32;
        let mut primary_item: Option<u32> = None;
        for it in items {
            if it.span.start < b && it.span.end > a {
                count += 1;
                if sum_density {
                    score += it.score;
                    if primary_item.is_none() {
                        primary_item = Some(it.item);
                    }
                } else if primary_item.is_none() || it.score.abs() > score.abs() {
                    score = it.score;
                    primary_item = Some(it.item);
                }
            }
        }
        let (start, end) = offsetter.to_wire(text, Span::new(a, b));
        out.push(HeatmapBucketProto { start, end, score, count, primary_item });
    }
    out
}
