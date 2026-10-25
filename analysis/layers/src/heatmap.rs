//! Multi-lane document heatmaps: per-lane segment scores over one view,
//! for minimap and inspector renderers.
//!
//! [`DocumentHeatmap::build`] emits one lane per source layer present,
//! in a fixed order: token, entity, and PII density; sentiment polarity
//! from `Categories<Sentences>` (positive minus negative, else a signed
//! best score); incivility and emotion intensity as each item's largest
//! label score over its target's fragments; and chunk scores from
//! `Categories<Chunks>` best scores. A lane whose source is absent is
//! itself absent. `BoundaryDrift`'s source lives in the chunker crate,
//! which depends on this one, so `build` never emits it; the kind
//! exists so chunker-side lanes share this vocabulary.
//! `build` selects default source instances; `build_instances` selects
//! every complete source instance on its requested view. Category rows use their
//! recorded sentence or chunk anchor instance, including its view and
//! individual fragments.
//!
//! With no bucket count a lane is segmented exactly with the same elementary
//! segmentation as its items' fragments over the whole view (see
//! [`Document::combined_segments`]); with one, the view is cut into
//! that many equal-width bins. A density segment's score is its
//! covering item count; a scored segment's score is its primary item's
//! (largest by absolute value, ties to the lowest item index); an
//! uncovered segment scores 0 with no primary item. Exact lanes tile
//! the view, uncovered runs included.
//! Positive-length fragments never cover empty bins.

use std::num::NonZeroU32;

use turbo_analysis::{Document, Error, Key, KeyId, Layer, Span, TextSpans, ViewId};

use crate::sentiment::{EmotionLayer, Incivility};
use crate::{Categories, Chunks, Entities, Pii, Sentences, Tokens};

mod buckets;
mod instances;
mod lanes;
mod sweep;

use buckets::bucket_segments;

use lanes::{chunk_score_lane, emotion_lane, incivility_lane, sentiment_lane};

/// What one heatmap lane measures.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum HeatmapLaneKind {
    TokenDensity,
    EntityDensity,
    PiiDensity,
    SentimentPolarity,
    IncivilityScore,
    EmotionIntensity,
    BoundaryDrift,
    ChunkScore,
}

impl HeatmapLaneKind {
    /// The lane's stable identifier, also its [`HeatmapLane::id`].
    pub const fn name(self) -> &'static str {
        match self {
            HeatmapLaneKind::TokenDensity => "token-density",
            HeatmapLaneKind::EntityDensity => "entity-density",
            HeatmapLaneKind::PiiDensity => "pii-density",
            HeatmapLaneKind::SentimentPolarity => "sentiment-polarity",
            HeatmapLaneKind::IncivilityScore => "incivility-score",
            HeatmapLaneKind::EmotionIntensity => "emotion-intensity",
            HeatmapLaneKind::BoundaryDrift => "boundary-drift",
            HeatmapLaneKind::ChunkScore => "chunk-score",
        }
    }

    fn is_density(self) -> bool {
        matches!(self, HeatmapLaneKind::TokenDensity | HeatmapLaneKind::EntityDensity | HeatmapLaneKind::PiiDensity)
    }
}

/// One scored run of a lane. `count` is how many source items cover the
/// span (in bucket mode, how many overlap the bin); `primary_item` is
/// the source item the score came from (the lowest-indexed covering
/// item on density lanes), or `None` when nothing covers it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeatmapSegment {
    pub span: Span,
    pub score: f32,
    pub count: u32,
    pub primary_item: Option<u32>,
}

/// One lane: its segments in text order and the score range over them
/// (both ends 0 when the lane has no segments).
#[derive(Clone, Debug, PartialEq)]
pub struct HeatmapLane {
    pub id: String,
    pub kind: HeatmapLaneKind,
    pub view: ViewId,
    pub min_score: f32,
    pub max_score: f32,
    pub segments: Vec<HeatmapSegment>,
}

/// Every heatmap lane a document's layers support over one view.
#[derive(Clone, Debug, PartialEq)]
pub struct DocumentHeatmap {
    pub view: ViewId,
    pub lanes: Vec<HeatmapLane>,
}

impl DocumentHeatmap {
    /// The lane of one kind, when its source layer was present.
    pub fn lane(&self, kind: HeatmapLaneKind) -> Option<&HeatmapLane> {
        self.lanes.iter().find(|l| l.kind == kind)
    }

    /// Builds every lane the document's layers support over `view`:
    /// exact elementary segments when `buckets` is `None`, that many
    /// equal-width bins when it is `Some`.
    ///
    /// Refuses a view the document does not have, a present source
    /// layer that fails its own `check`, source spans anchored to a
    /// different view, a scored layer misaligned with the layer it
    /// scores, a target index outside its target layer, and any
    /// non-finite score a lane would carry, each naming the layer and
    /// item. A scored layer present without the layer it scores is
    /// [`Error::Missing`] for that layer.
    /// Bucket storage with an unrepresentable layout or a failed reservation
    /// returns [`Error::Invalid`] naming the source layer. Output retains the
    /// requested bin count, with memory proportional to that count.
    pub fn build(doc: &Document, view: ViewId, buckets: Option<NonZeroU32>) -> Result<DocumentHeatmap, Error> {
        if view.0 as usize >= doc.view_count() {
            return Err(Error::Bounds {
                layer: Key::<turbo_analysis::View>::DEFAULT.id(),
                view,
                detail: format!("the document has {} views, view {} does not exist", doc.view_count(), view.0),
            });
        }
        let text_len = doc.text(view).len() as u32;
        let mut lanes = Vec::new();
        if let Some(tokens) = doc.get::<Tokens>() {
            tokens.check().map_err(|d| invalid(Key::<Tokens>::DEFAULT.id(), d))?;
            lanes.push(spanned_lane(
                doc,
                view,
                text_len,
                buckets,
                HeatmapLaneKind::TokenDensity,
                Key::<Tokens>::DEFAULT.id(),
                tokens.spans.text_spans(),
            )?);
        }
        if let Some(entities) = doc.get::<Entities>() {
            entities.check().map_err(|d| invalid(Key::<Entities>::DEFAULT.id(), d))?;
            lanes.push(spanned_lane(
                doc,
                view,
                text_len,
                buckets,
                HeatmapLaneKind::EntityDensity,
                Key::<Entities>::DEFAULT.id(),
                entities.spans.text_spans(),
            )?);
        }
        if let Some(pii) = doc.get::<Pii>() {
            pii.check().map_err(|d| invalid(Key::<Pii>::DEFAULT.id(), d))?;
            lanes.push(spanned_lane(
                doc,
                view,
                text_len,
                buckets,
                HeatmapLaneKind::PiiDensity,
                Key::<Pii>::DEFAULT.id(),
                pii.spans.text_spans(),
            )?);
        }
        if let Some(cats) = doc.get::<Categories<Sentences>>() {
            lanes.push(sentiment_lane(doc, view, text_len, buckets, Key::<Categories<Sentences>>::DEFAULT.id(), cats)?);
        }
        if let Some(inc) = doc.get::<Incivility>() {
            lanes.push(incivility_lane(doc, view, text_len, buckets, Key::<Incivility>::DEFAULT.id(), inc, false)?);
        }
        if let Some(emotion) = doc.get::<EmotionLayer>() {
            lanes.push(emotion_lane(doc, view, text_len, buckets, Key::<EmotionLayer>::DEFAULT.id(), emotion, false)?);
        }
        if let Some(cats) = doc.get::<Categories<Chunks>>() {
            lanes.push(chunk_score_lane(doc, view, text_len, buckets, Key::<Categories<Chunks>>::DEFAULT.id(), cats)?);
        }
        Ok(DocumentHeatmap { view, lanes })
    }
}

/// One source item: its index, score, the view its spans are in and
/// its fragments there.
struct Item {
    source: u32,
    score: f32,
    view: ViewId,
    frags: Vec<(u32, u32)>,
}

fn invalid(layer: KeyId, detail: String) -> Error {
    Error::Invalid { layer, detail }
}

fn check_source_view(layer: KeyId, source: ViewId, view: ViewId) -> Result<(), Error> {
    if source != view {
        return Err(Error::Bounds {
            layer,
            view,
            detail: format!("source spans are in view {}, the heatmap is over view {}", source.0, view.0),
        });
    }
    Ok(())
}

fn frags_of(ts: &TextSpans<'_>, item: usize) -> Vec<(u32, u32)> {
    if ts.frag_item_end.is_empty() {
        vec![(ts.start[item], ts.end[item])]
    } else {
        ts.item_frag_range(item).map(|f| (ts.frag_start[f], ts.frag_end[f])).collect()
    }
}

/// A density lane over a positional layer's own span columns.
#[allow(clippy::too_many_arguments)]
fn spanned_lane(
    doc: &Document,
    view: ViewId,
    text_len: u32,
    buckets: Option<NonZeroU32>,
    kind: HeatmapLaneKind,
    key: KeyId,
    ts: TextSpans<'_>,
) -> Result<HeatmapLane, Error> {
    check_source_view(key, ts.view, view)?;
    let items: Vec<Item> = (0..ts.start.len())
        .map(|i| Item { source: i as u32, score: 0.0, view: ts.view, frags: frags_of(&ts, i) })
        .collect();
    finish_lane(doc, view, text_len, buckets, kind, key, &items)
}

/// Segments `items` exactly or by bucket and aggregates each segment.
/// The sweep maintains item summaries instead of materializing active memberships.
#[allow(clippy::too_many_arguments)]
fn finish_lane(
    doc: &Document,
    view: ViewId,
    text_len: u32,
    buckets: Option<NonZeroU32>,
    kind: HeatmapLaneKind,
    key: KeyId,
    items: &[Item],
) -> Result<HeatmapLane, Error> {
    for it in items {
        if !kind.is_density() && !it.score.is_finite() {
            return Err(invalid(key, format!("item {} has a non-finite score {}", it.source, it.score)));
        }
    }
    let segments = match buckets {
        None => exact_segments(doc, view, text_len, kind, key, items)?,
        Some(n) => bucket_segments(doc, view, key, text_len, n.get(), kind, items)?,
    };
    let (mut min_score, mut max_score) = (0.0f32, 0.0f32);
    for (i, s) in segments.iter().enumerate() {
        if i == 0 || s.score < min_score {
            min_score = s.score;
        }
        if i == 0 || s.score > max_score {
            max_score = s.score;
        }
    }
    Ok(HeatmapLane { id: kind.name().to_string(), kind, view, min_score, max_score, segments })
}

/// Refuses an item whose spans are in a view other than `view`, or a
/// fragment that is reversed, past the view's `text_len` bytes or inside
/// a character: both modes lay the offsets on `view`'s text as they are.
fn check_items(doc: &Document, view: ViewId, key: KeyId, text_len: u32, items: &[Item]) -> Result<(), Error> {
    let text = doc.text(view);
    let bounds = |detail: String| Error::Bounds { layer: key, view, detail };
    for it in items {
        if it.view != view {
            let detail =
                format!("item {} spans are in view {}, the heatmap is over view {}", it.source, it.view.0, view.0);
            return Err(bounds(detail));
        }
        for &(s, e) in &it.frags {
            if it.frags.len() > 1 && s == e {
                return Err(bounds(format!("item {} has an empty discontinuous fragment", it.source)));
            }
            if s > e || e > text_len {
                let detail =
                    format!("item {} fragment {s}..{e} is past the end of the {text_len}-byte view", it.source);
                return Err(bounds(detail));
            }
            if !text.is_char_boundary(s as usize) || !text.is_char_boundary(e as usize) {
                return Err(bounds(format!("item {} fragment {s}..{e} is inside a character of the view", it.source)));
            }
        }
        if it.frags.windows(2).any(|pair| pair[0].1 >= pair[1].0) {
            return Err(bounds(format!("item {} fragments are not separated in text order", it.source)));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn exact_segments(
    doc: &Document,
    view: ViewId,
    text_len: u32,
    kind: HeatmapLaneKind,
    key: KeyId,
    items: &[Item],
) -> Result<Vec<HeatmapSegment>, Error> {
    check_items(doc, view, key, text_len, items)?;
    sweep::exact(text_len, key, kind, items)
}

#[cfg(test)]
mod tests;
