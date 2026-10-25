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
//!
//! With no bucket count a lane is segmented exactly, by the combined
//! segmentation of its items' fragments over the whole view (see
//! [`Document::combined_segments`]); with one, the view is cut into
//! that many equal-width bins. A density segment's score is its
//! covering item count; a scored segment's score is its primary item's
//! (largest by absolute value, ties to the lowest item index); an
//! uncovered segment scores 0 with no primary item. Exact lanes tile
//! the view, uncovered runs included.

use std::num::NonZeroU32;

use turbo_analysis::{Document, Error, Key, KeyId, Layer, Span, Tag, TextSpans, ViewId};

use crate::sentiment::{EMOTION_DIMENSIONS, EmotionLayer, EmotionTarget, Incivility, IncivilityTarget};
use crate::{Categories, Chunks, Entities, Pii, Sentences, Structure, Tokens};

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
            lanes.push(sentiment_lane(doc, view, text_len, buckets, cats)?);
        }
        if let Some(inc) = doc.get::<Incivility>() {
            lanes.push(incivility_lane(doc, view, text_len, buckets, inc)?);
        }
        if let Some(emotion) = doc.get::<EmotionLayer>() {
            lanes.push(emotion_lane(doc, view, text_len, buckets, emotion)?);
        }
        if let Some(cats) = doc.get::<Categories<Chunks>>() {
            lanes.push(chunk_score_lane(doc, view, text_len, buckets, cats)?);
        }
        Ok(DocumentHeatmap { view, lanes })
    }
}

struct Item {
    source: u32,
    score: f32,
    frags: Vec<(u32, u32)>,
}

fn invalid(layer: KeyId, detail: String) -> Error {
    Error::Invalid { layer, detail }
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
    let items: Vec<Item> =
        (0..ts.start.len()).map(|i| Item { source: i as u32, score: 0.0, frags: frags_of(&ts, i) }).collect();
    finish_lane(doc, view, text_len, buckets, kind, key, &items, Some(ts))
}

/// Segments `items` exactly or by bucket and aggregates each segment.
/// `columns` reuses an existing layer's span columns in exact mode
/// when the items are exactly that layer's items in order; otherwise
/// columns are synthesized from the item fragments.
#[allow(clippy::too_many_arguments)]
fn finish_lane(
    doc: &Document,
    view: ViewId,
    text_len: u32,
    buckets: Option<NonZeroU32>,
    kind: HeatmapLaneKind,
    key: KeyId,
    items: &[Item],
    columns: Option<TextSpans<'_>>,
) -> Result<HeatmapLane, Error> {
    for it in items {
        if !kind.is_density() && !it.score.is_finite() {
            return Err(invalid(key, format!("item {} has a non-finite score {}", it.source, it.score)));
        }
    }
    let segments = match buckets {
        None => exact_segments(doc, view, text_len, kind, key, items, columns)?,
        Some(n) => bucket_segments(text_len, n.get(), kind, items),
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

fn aggregate(kind: HeatmapLaneKind, items: &[Item], covering: &[usize]) -> (f32, u32, Option<u32>) {
    let count = covering.len() as u32;
    if covering.is_empty() {
        return (0.0, 0, None);
    }
    if kind.is_density() {
        let first = covering.iter().map(|&i| items[i].source).min();
        return (count as f32, count, first);
    }
    let mut best = covering[0];
    for &i in &covering[1..] {
        let (a, b) = (items[best].score.abs(), items[i].score.abs());
        if b > a || (b == a && items[i].source < items[best].source) {
            best = i;
        }
    }
    (items[best].score, count, Some(items[best].source))
}

#[allow(clippy::too_many_arguments)]
fn exact_segments(
    doc: &Document,
    view: ViewId,
    text_len: u32,
    kind: HeatmapLaneKind,
    key: KeyId,
    items: &[Item],
    columns: Option<TextSpans<'_>>,
) -> Result<Vec<HeatmapSegment>, Error> {
    // Synthesized columns when items are not a layer's own, kept alive
    // for the combined call.
    let (mut start, mut end, mut frag_start, mut frag_end, mut frag_item_end) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let ts = match columns {
        Some(ts) => ts,
        None => {
            for it in items {
                start.push(it.frags.first().map_or(0, |f| f.0));
                end.push(it.frags.last().map_or(0, |f| f.1));
                for f in &it.frags {
                    frag_start.push(f.0);
                    frag_end.push(f.1);
                }
                frag_item_end.push(frag_start.len() as u32);
            }
            TextSpans {
                view,
                start: &start,
                end: &end,
                frag_start: &frag_start,
                frag_end: &frag_end,
                frag_item_end: &frag_item_end,
            }
        }
    };
    let window = doc.window(view, 0, text_len.max(1))?;
    let shape = doc.combined_segments(&window, &[(key, ts)])?;
    Ok(shape
        .segments
        .iter()
        .map(|seg| {
            let covering: Vec<usize> = shape.members_of(seg).iter().map(|m| m.item as usize).collect();
            let (score, count, primary_item) = aggregate(kind, items, &covering);
            HeatmapSegment { span: seg.span, score, count, primary_item }
        })
        .collect())
}

fn bucket_segments(text_len: u32, buckets: u32, kind: HeatmapLaneKind, items: &[Item]) -> Vec<HeatmapSegment> {
    let n = buckets as usize;
    let edge = |i: usize| -> u32 { (i as u64 * text_len as u64 / buckets as u64) as u32 };
    let mut covering: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut stamp: Vec<u32> = vec![0; n];
    for (ix, it) in items.iter().enumerate() {
        let mark = ix as u32 + 1;
        for &(s, e) in &it.frags {
            // Bins a fragment touches: positive-length overlap, or the
            // bin holding a zero-width point (the last bin at the end).
            if text_len == 0 && s != e {
                continue;
            }
            let bin = |at: u32| -> usize {
                if text_len == 0 { 0 } else { ((at as u64 * buckets as u64 / text_len as u64) as usize).min(n - 1) }
            };
            let (first, last) = if s == e { (bin(s), bin(s)) } else { (bin(s), bin(e - 1)) };
            for (b, cov) in covering.iter_mut().enumerate().take(last + 1).skip(first) {
                if stamp[b] != mark {
                    stamp[b] = mark;
                    cov.push(ix);
                }
            }
        }
    }
    (0..n)
        .map(|i| {
            let (score, count, primary_item) = aggregate(kind, items, &covering[i]);
            HeatmapSegment { span: Span { start: edge(i), end: edge(i + 1) }, score, count, primary_item }
        })
        .collect()
}

fn sentiment_lane(
    doc: &Document,
    view: ViewId,
    text_len: u32,
    buckets: Option<NonZeroU32>,
    cats: &Categories<Sentences>,
) -> Result<HeatmapLane, Error> {
    let key = Key::<Categories<Sentences>>::DEFAULT.id();
    cats.check().map_err(|d| invalid(key, d))?;
    let sentences = doc.get::<Sentences>().ok_or(Error::Missing(Key::<Sentences>::DEFAULT.id()))?;
    if cats.len() != sentences.spans.len() {
        return Err(invalid(key, format!("{} category rows for {} sentences", cats.len(), sentences.spans.len())));
    }
    let ts = sentences.spans.text_spans();
    let items = cats
        .iter()
        .enumerate()
        .filter_map(|(i, row)| polarity_of(&row).map(|score| Item { source: i as u32, score, frags: frags_of(&ts, i) }))
        .collect::<Vec<_>>();
    // Non-finite category scores would hide inside polarity_of; refuse
    // them up front so a lane never silently drops a scored sentence.
    for (i, row) in cats.iter().enumerate() {
        if row.scores.iter().any(|s| !s.is_finite()) {
            return Err(invalid(key, format!("sentence {i} has a non-finite category score")));
        }
    }
    finish_lane(doc, view, text_len, buckets, HeatmapLaneKind::SentimentPolarity, key, &items, None)
}

fn polarity_of(row: &crate::CategoryRow<'_>) -> Option<f32> {
    let pos = row.score_of("positive").or_else(|| row.score_of("pos"));
    let neg = row.score_of("negative").or_else(|| row.score_of("neg"));
    if let (Some(p), Some(n)) = (pos, neg) {
        return Some((p - n).clamp(-1.0, 1.0) as f32);
    }
    if row.best_tag == Tag::NONE {
        return None;
    }
    let mag = row.best_score().clamp(0.0, 1.0) as f32;
    if row.best.eq_ignore_ascii_case("positive") || row.best.eq_ignore_ascii_case("pos") {
        Some(mag)
    } else if row.best.eq_ignore_ascii_case("negative") || row.best.eq_ignore_ascii_case("neg") {
        Some(-mag)
    } else {
        None
    }
}

fn chunk_score_lane(
    doc: &Document,
    view: ViewId,
    text_len: u32,
    buckets: Option<NonZeroU32>,
    cats: &Categories<Chunks>,
) -> Result<HeatmapLane, Error> {
    let key = Key::<Categories<Chunks>>::DEFAULT.id();
    cats.check().map_err(|d| invalid(key, d))?;
    let chunks = doc.get::<Chunks>().ok_or(Error::Missing(Key::<Chunks>::DEFAULT.id()))?;
    if cats.len() != chunks.spans.len() {
        return Err(invalid(key, format!("{} category rows for {} chunks", cats.len(), chunks.spans.len())));
    }
    let ts = chunks.spans.text_spans();
    let mut items = Vec::new();
    for (i, row) in cats.iter().enumerate() {
        if row.scores.iter().any(|s| !s.is_finite()) {
            return Err(invalid(key, format!("chunk {i} has a non-finite category score")));
        }
        if row.best_tag != Tag::NONE {
            items.push(Item { source: i as u32, score: row.best_score() as f32, frags: frags_of(&ts, i) });
        }
    }
    finish_lane(doc, view, text_len, buckets, HeatmapLaneKind::ChunkScore, key, &items, None)
}

fn target_frags(key: KeyId, spans_of: &str, item: usize, ts: &TextSpans<'_>) -> Result<Vec<(u32, u32)>, Error> {
    if item >= ts.start.len() {
        return Err(invalid(key, format!("target {item} is outside {spans_of} with {} items", ts.start.len())));
    }
    Ok(frags_of(ts, item))
}

fn incivility_lane(
    doc: &Document,
    view: ViewId,
    text_len: u32,
    buckets: Option<NonZeroU32>,
    inc: &Incivility,
) -> Result<HeatmapLane, Error> {
    let key = Key::<Incivility>::DEFAULT.id();
    inc.check().map_err(|d| invalid(key, d))?;
    let mut items = Vec::with_capacity(inc.target.len());
    for (i, target) in inc.target.iter().enumerate() {
        let score = [
            inc.insult[i],
            inc.profanity[i],
            inc.threat[i],
            inc.identity_attack[i],
            inc.intolerance[i],
            inc.severe_toxicity[i],
        ]
        .iter()
        .fold(0.0f64, |m, &s| m.max(s));
        if !score.is_finite() {
            return Err(invalid(key, format!("item {i} has a non-finite incivility score")));
        }
        let frags = match target {
            IncivilityTarget::Sentence(s) => {
                let layer =
                    doc.get_key::<Sentences>(&inc.sentences_key).ok_or(Error::Missing(inc.sentences_key.id()))?;
                target_frags(key, "sentences", s.get(), &layer.spans.text_spans())?
            }
            IncivilityTarget::Chunk(c) => {
                let layer = doc.get_key::<Chunks>(&inc.chunks_key).ok_or(Error::Missing(inc.chunks_key.id()))?;
                target_frags(key, "chunks", c.get(), &layer.spans.text_spans())?
            }
            IncivilityTarget::Block(b) => {
                let layer =
                    doc.get_key::<Structure>(&inc.structure_key).ok_or(Error::Missing(inc.structure_key.id()))?;
                target_frags(key, "structure blocks", b.get(), &layer.spans.text_spans())?
            }
        };
        items.push(Item { source: i as u32, score: score as f32, frags });
    }
    finish_lane(doc, view, text_len, buckets, HeatmapLaneKind::IncivilityScore, key, &items, None)
}

fn emotion_lane(
    doc: &Document,
    view: ViewId,
    text_len: u32,
    buckets: Option<NonZeroU32>,
    emotion: &EmotionLayer,
) -> Result<HeatmapLane, Error> {
    let key = Key::<EmotionLayer>::DEFAULT.id();
    emotion.check().map_err(|d| invalid(key, d))?;
    let mut items = Vec::with_capacity(emotion.target.len());
    for (i, target) in emotion.target.iter().enumerate() {
        let row = &emotion.scores[i * EMOTION_DIMENSIONS..(i + 1) * EMOTION_DIMENSIONS];
        if row.iter().any(|s| !s.is_finite()) {
            return Err(invalid(key, format!("item {i} has a non-finite emotion score")));
        }
        let score = row.iter().fold(0.0f64, |m, &s| m.max(s));
        let frags = match target {
            EmotionTarget::Sentence(s) => {
                let layer = doc
                    .get_key::<Sentences>(&emotion.sentences_key)
                    .ok_or(Error::Missing(emotion.sentences_key.id()))?;
                target_frags(key, "sentences", s.get(), &layer.spans.text_spans())?
            }
            EmotionTarget::Chunk(c) => {
                let layer =
                    doc.get_key::<Chunks>(&emotion.chunks_key).ok_or(Error::Missing(emotion.chunks_key.id()))?;
                target_frags(key, "chunks", c.get(), &layer.spans.text_spans())?
            }
        };
        items.push(Item { source: i as u32, score: score as f32, frags });
    }
    finish_lane(doc, view, text_len, buckets, HeatmapLaneKind::EmotionIntensity, key, &items, None)
}
