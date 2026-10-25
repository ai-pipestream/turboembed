//! Wire projection of the shared DocumentHeatmap, including named sources.
use super::{bad, window::snap_down};
mod preflight;
use crate::{
    api::Failure,
    offsets::Offsetter,
    proto::result::{HeatmapBucketProto, HeatmapLaneProto},
    results::over,
};
use std::num::NonZeroU32;
use turbo_analysis::{Document, Span, ViewId};
use turbo_analysis_layers::{DocumentHeatmap, HeatmapLaneKind};

/// Maximum number of minimap buckets allowed per heatmap lane.
pub const MAX_HEATMAP_BUCKETS: u32 = 4_096;

pub(crate) fn build_heatmap_lanes(
    doc: &Document,
    view: ViewId,
    text: &str,
    offsetter: &Offsetter,
    heatmap_buckets: u32,
) -> Result<Vec<HeatmapLaneProto>, Failure> {
    build_with_budget(doc, view, text, offsetter, heatmap_buckets, crate::results::MAX_FRAGMENTS)
}

fn build_with_budget(
    doc: &Document,
    view: ViewId,
    text: &str,
    offsetter: &Offsetter,
    heatmap_buckets: u32,
    fragment_budget: u64,
) -> Result<Vec<HeatmapLaneProto>, Failure> {
    if heatmap_buckets > MAX_HEATMAP_BUCKETS {
        return Err(over("max_heatmap_buckets", "heatmap_buckets", heatmap_buckets as u64));
    }
    preflight::check(doc, view, fragment_budget)?;
    let heatmap = DocumentHeatmap::build_instances(doc, view, NonZeroU32::new(heatmap_buckets))
        .map_err(|e| bad(format!("document heatmap: {e}")))?;
    heatmap
        .lanes
        .into_iter()
        .map(|lane| {
            let (kind, prefix) = wire_kind(lane.kind);
            let suffix = lane.id.strip_prefix(lane.kind.name()).ok_or_else(|| bad("invalid shared heatmap lane ID"))?;
            let buckets = lane
                .segments
                .into_iter()
                .map(|s| {
                    // Scores/coverage describe the native raw byte bin. Only the
                    // displayed coordinates follow the RPC's documented snapping.
                    let display = Span::new(
                        snap_down(text, s.span.start as usize) as u32,
                        snap_down(text, s.span.end as usize) as u32,
                    );
                    let (start, end) = offsetter.to_wire(text, display);
                    HeatmapBucketProto {
                        start,
                        end,
                        score: s.score,
                        count: s.count,
                        primary_item: s.primary_item,
                        byte_start: s.span.start,
                        byte_end: s.span.end,
                    }
                })
                .collect();
            Ok(HeatmapLaneProto {
                id: format!("{prefix}{suffix}"),
                kind: kind.into(),
                view: view.0 as u32,
                min_score: lane.min_score,
                max_score: lane.max_score,
                buckets,
            })
        })
        .collect()
}

fn wire_kind(kind: HeatmapLaneKind) -> (&'static str, &'static str) {
    match kind {
        HeatmapLaneKind::TokenDensity => ("TokenDensity", "tokens"),
        HeatmapLaneKind::EntityDensity => ("EntityDensity", "entities"),
        HeatmapLaneKind::PiiDensity => ("PiiDensity", "pii"),
        HeatmapLaneKind::SentimentPolarity => ("SentimentPolarity", "sentiment"),
        HeatmapLaneKind::IncivilityScore => ("IncivilityScore", "incivility"),
        HeatmapLaneKind::EmotionIntensity => ("EmotionIntensity", "emotion"),
        HeatmapLaneKind::BoundaryDrift => ("BoundaryDrift", "drift"),
        HeatmapLaneKind::ChunkScore => ("ChunkScore", "chunks"),
    }
}
