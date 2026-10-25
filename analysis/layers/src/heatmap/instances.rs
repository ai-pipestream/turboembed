//! Named source selection for a view; scoring stays in the common lanes.
use super::lanes::{chunk_score_lane, emotion_lane, incivility_lane, scored_target, sentiment_lane};
use super::{DocumentHeatmap, HeatmapLane, HeatmapLaneKind, invalid, spanned_lane};
use crate::{Categories, Chunks, EmotionLayer, Entities, Incivility, Pii, Sentences, Tokens};
use std::{any::TypeId, num::NonZeroU32};
use turbo_analysis::{Document, Error, Key, KeyId, Layer, Status, View, ViewId};

impl DocumentHeatmap {
    /// Builds every supported default and named source instance on `view`.
    ///
    /// Unlike [`Self::build`], other-view sources are omitted. Category
    /// instances follow their recorded sentence/chunk anchor. Incivility
    /// and emotion retain only targets on this view, keeping original item
    /// indices; instances with no covered target are omitted. Density and
    /// category sources on this view retain empty lanes. Failed/skipped
    /// source records emit no lane. Selected sources
    /// receive the same validation, scoring, raw byte-bin geometry and
    /// aggregation as `build`; there is no second scoring implementation.
    ///
    /// Lane IDs append `@instance` to the kind's stable name for named
    /// sources. Order is kind order, then instance name. An absent view is
    /// refused, including when the document contains no supported source.
    pub fn build_instances(doc: &Document, view: ViewId, buckets: Option<NonZeroU32>) -> Result<Self, Error> {
        if view.0 as usize >= doc.view_count() {
            return Err(Error::Bounds {
                layer: Key::<View>::DEFAULT.id(),
                view,
                detail: format!("the document has {} views, view {} does not exist", doc.view_count(), view.0),
            });
        }
        let len = doc.text(view).len() as u32;
        let mut lanes = Vec::new();
        for key in doc.layers() {
            if !matches!(doc.status(key), Some(Status::Complete)) {
                continue;
            }
            let lane = if key.ty == TypeId::of::<Tokens>() {
                density::<Tokens>(doc, view, len, buckets, key, HeatmapLaneKind::TokenDensity)?
            } else if key.ty == TypeId::of::<Entities>() {
                density::<Entities>(doc, view, len, buckets, key, HeatmapLaneKind::EntityDensity)?
            } else if key.ty == TypeId::of::<Pii>() {
                density::<Pii>(doc, view, len, buckets, key, HeatmapLaneKind::PiiDensity)?
            } else if key.ty == TypeId::of::<Categories<Sentences>>() {
                if scored_target::<Sentences>(doc, key, None)?.spans.view() != view {
                    continue;
                }
                let cats = doc.get_key(&Key::<Categories<Sentences>>::named(key.name)).ok_or(Error::Missing(key))?;
                Some(sentiment_lane(doc, view, len, buckets, key, cats)?)
            } else if key.ty == TypeId::of::<Categories<Chunks>>() {
                if scored_target::<Chunks>(doc, key, None)?.spans.view() != view {
                    continue;
                }
                let cats = doc.get_key(&Key::<Categories<Chunks>>::named(key.name)).ok_or(Error::Missing(key))?;
                Some(chunk_score_lane(doc, view, len, buckets, key, cats)?)
            } else if key.ty == TypeId::of::<Incivility>() {
                let inc = doc.get_key(&Key::<Incivility>::named(key.name)).ok_or(Error::Missing(key))?;
                covered(incivility_lane(doc, view, len, buckets, key, inc, true)?)
            } else if key.ty == TypeId::of::<EmotionLayer>() {
                let emotion = doc.get_key(&Key::<EmotionLayer>::named(key.name)).ok_or(Error::Missing(key))?;
                covered(emotion_lane(doc, view, len, buckets, key, emotion, true)?)
            } else {
                None
            };
            if let Some(mut lane) = lane {
                if !key.name.is_empty() {
                    lane.id.push('@');
                    lane.id.push_str(key.name);
                }
                lanes.push(lane);
            }
        }
        lanes.sort_by(|a, b| rank(a.kind).cmp(&rank(b.kind)).then_with(|| a.id.cmp(&b.id)));
        Ok(Self { view, lanes })
    }
}

fn covered(lane: HeatmapLane) -> Option<HeatmapLane> {
    lane.segments.iter().any(|s| s.count > 0).then_some(lane)
}

fn density<L: Layer>(
    doc: &Document,
    view: ViewId,
    len: u32,
    buckets: Option<NonZeroU32>,
    key: KeyId,
    kind: HeatmapLaneKind,
) -> Result<Option<HeatmapLane>, Error> {
    let layer = doc.get_key(&Key::<L>::named(key.name)).ok_or(Error::Missing(key))?;
    let spans = layer.text_spans().ok_or_else(|| invalid(key, "density source has no spans".into()))?;
    if spans.view != view {
        return Ok(None);
    }
    layer.check().map_err(|d| invalid(key, d))?;
    Ok(Some(spanned_lane(doc, view, len, buckets, kind, key, spans)?))
}

fn rank(kind: HeatmapLaneKind) -> u8 {
    match kind {
        HeatmapLaneKind::TokenDensity => 0,
        HeatmapLaneKind::EntityDensity => 1,
        HeatmapLaneKind::PiiDensity => 2,
        HeatmapLaneKind::SentimentPolarity => 3,
        HeatmapLaneKind::IncivilityScore => 4,
        HeatmapLaneKind::EmotionIntensity => 5,
        HeatmapLaneKind::BoundaryDrift => 6,
        HeatmapLaneKind::ChunkScore => 7,
    }
}
