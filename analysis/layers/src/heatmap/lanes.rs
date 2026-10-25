//! The scored lanes: sentiment polarity, chunk scores, incivility and
//! emotion intensity, each item carrying the view its spans are in.

use std::any::TypeId;
use std::num::NonZeroU32;

use turbo_analysis::{Document, Error, Key, KeyId, Layer, Tag, TextSpans, ViewId};

use super::{HeatmapLane, HeatmapLaneKind, Item, check_source_view, finish_lane, frags_of, invalid};
use crate::sentiment::{EMOTION_DIMENSIONS, EmotionLayer, EmotionTarget, Incivility, IncivilityTarget};
use crate::{Categories, Chunks, Sentences, Structure};

fn scored_target<L: Layer>(doc: &Document, key: KeyId, view: ViewId) -> Result<&L, Error> {
    let target = doc.made_on(key).iter().find(|k| k.ty == TypeId::of::<L>()).copied().unwrap_or(Key::<L>::DEFAULT.id());
    let layer = doc.get_key(&Key::<L>::named(target.name)).ok_or(Error::Missing(target))?;
    layer.check().map_err(|detail| invalid(target, detail))?;
    if let Some(spans) = layer.text_spans() {
        check_source_view(key, spans.view, view)?;
    }
    Ok(layer)
}

pub(super) fn sentiment_lane(
    doc: &Document,
    view: ViewId,
    text_len: u32,
    buckets: Option<NonZeroU32>,
    cats: &Categories<Sentences>,
) -> Result<HeatmapLane, Error> {
    let key = Key::<Categories<Sentences>>::DEFAULT.id();
    cats.check().map_err(|d| invalid(key, d))?;
    let sentences = scored_target::<Sentences>(doc, key, view)?;
    if cats.len() != sentences.spans.len() {
        return Err(invalid(key, format!("{} category rows for {} sentences", cats.len(), sentences.spans.len())));
    }
    let ts = sentences.spans.text_spans();
    let items = cats
        .iter()
        .enumerate()
        .filter_map(|(i, row)| {
            polarity_of(&row).map(|score| Item { source: i as u32, score, view: ts.view, frags: frags_of(&ts, i) })
        })
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
    if row.best_tag == Tag::NONE {
        return None;
    }
    let pos = row.score_of("positive").or_else(|| row.score_of("pos"));
    let neg = row.score_of("negative").or_else(|| row.score_of("neg"));
    if let (Some(p), Some(n)) = (pos, neg) {
        return Some((p - n).clamp(-1.0, 1.0) as f32);
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

pub(super) fn chunk_score_lane(
    doc: &Document,
    view: ViewId,
    text_len: u32,
    buckets: Option<NonZeroU32>,
    cats: &Categories<Chunks>,
) -> Result<HeatmapLane, Error> {
    let key = Key::<Categories<Chunks>>::DEFAULT.id();
    cats.check().map_err(|d| invalid(key, d))?;
    let chunks = scored_target::<Chunks>(doc, key, view)?;
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
            items.push(Item {
                source: i as u32,
                score: row.best_score() as f32,
                view: ts.view,
                frags: frags_of(&ts, i),
            });
        }
    }
    finish_lane(doc, view, text_len, buckets, HeatmapLaneKind::ChunkScore, key, &items, None)
}

/// The view and fragments of target `item` in `spans_of`'s columns.
type Frags = (ViewId, Vec<(u32, u32)>);

fn target_frags(key: KeyId, spans_of: &str, item: usize, ts: &TextSpans<'_>) -> Result<Frags, Error> {
    if item >= ts.start.len() {
        return Err(invalid(key, format!("target {item} is outside {spans_of} with {} items", ts.start.len())));
    }
    Ok((ts.view, frags_of(ts, item)))
}

pub(super) fn incivility_lane(
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
        let scores = [
            inc.insult[i],
            inc.profanity[i],
            inc.threat[i],
            inc.identity_attack[i],
            inc.intolerance[i],
            inc.severe_toxicity[i],
        ];
        // f64::max drops NaN, so check before folding.
        if scores.iter().any(|s| !s.is_finite()) {
            return Err(invalid(key, format!("item {i} has a non-finite incivility score")));
        }
        let score = scores.iter().fold(0.0f64, |m, &s| m.max(s));
        let (view_of, frags) = match target {
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
        items.push(Item { source: i as u32, score: score as f32, view: view_of, frags });
    }
    finish_lane(doc, view, text_len, buckets, HeatmapLaneKind::IncivilityScore, key, &items, None)
}

pub(super) fn emotion_lane(
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
        let (view_of, frags) = match target {
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
        items.push(Item { source: i as u32, score: score as f32, view: view_of, frags });
    }
    finish_lane(doc, view, text_len, buckets, HeatmapLaneKind::EmotionIntensity, key, &items, None)
}
