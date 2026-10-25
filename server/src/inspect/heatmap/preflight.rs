//! Input limits before the shared builder allocates item/fragment storage.
use super::super::bad;
use crate::{api::Failure, results::over};
use std::any::TypeId;
use turbo_analysis::{Document, Ix, Key, KeyId, Layer, Spans, Status, ViewId};
use turbo_analysis_layers::{
    Categories, Chunks, EmotionLayer, EmotionTarget, Entities, Incivility, IncivilityTarget, Pii, Sentences, Tokens,
};

fn all<L: Layer>(doc: &Document, key: KeyId, view: ViewId) -> u64 {
    doc.get_key(&Key::<L>::named(key.name))
        .and_then(Layer::text_spans)
        .filter(|s| s.view == view)
        .map_or(0, |s| if s.frag_item_end.is_empty() { s.start.len() } else { s.frag_start.len() }) as u64
}
fn anchor<L: Layer>(doc: &Document, key: KeyId) -> KeyId {
    doc.made_on(key).iter().find(|k| k.ty == TypeId::of::<L>()).copied().unwrap_or(Key::<L>::DEFAULT.id())
}
fn one<L>(spans: &Spans<L>, ix: Ix<L>, view: ViewId) -> Result<u64, Failure> {
    if spans.view() != view {
        return Ok(0);
    }
    if ix.get() >= spans.len() {
        return Err(bad("heatmap target outside its anchor"));
    }
    Ok(spans.fragments(ix).len() as u64)
}
fn missing() -> Failure {
    bad("heatmap target anchor missing")
}

pub(super) fn check(doc: &Document, view: ViewId, limit: u64) -> Result<(), Failure> {
    for key in doc.layers() {
        if !matches!(doc.status(key), Some(Status::Complete)) {
            continue;
        }
        let n = if key.ty == TypeId::of::<Tokens>() {
            all::<Tokens>(doc, key, view)
        } else if key.ty == TypeId::of::<Entities>() {
            all::<Entities>(doc, key, view)
        } else if key.ty == TypeId::of::<Pii>() {
            all::<Pii>(doc, key, view)
        } else if key.ty == TypeId::of::<Categories<Sentences>>() {
            all::<Sentences>(doc, anchor::<Sentences>(doc, key), view)
        } else if key.ty == TypeId::of::<Categories<Chunks>>() {
            all::<Chunks>(doc, anchor::<Chunks>(doc, key), view)
        } else if key.ty == TypeId::of::<Incivility>() {
            let layer = doc.get_key(&Key::<Incivility>::named(key.name)).ok_or_else(missing)?;
            let mut n = 0u64;
            for target in &layer.target {
                n = n.saturating_add(match *target {
                    IncivilityTarget::Sentence(i) => {
                        one(&doc.get_key(&layer.sentences_key).ok_or_else(missing)?.spans, i, view)?
                    }
                    IncivilityTarget::Chunk(i) => {
                        one(&doc.get_key(&layer.chunks_key).ok_or_else(missing)?.spans, i, view)?
                    }
                    IncivilityTarget::Block(i) => {
                        one(&doc.get_key(&layer.structure_key).ok_or_else(missing)?.spans, i, view)?
                    }
                });
                if n > limit {
                    break;
                }
            }
            n
        } else if key.ty == TypeId::of::<EmotionLayer>() {
            let layer = doc.get_key(&Key::<EmotionLayer>::named(key.name)).ok_or_else(missing)?;
            let mut n = 0u64;
            for target in &layer.target {
                n = n.saturating_add(match *target {
                    EmotionTarget::Sentence(i) => {
                        one(&doc.get_key(&layer.sentences_key).ok_or_else(missing)?.spans, i, view)?
                    }
                    EmotionTarget::Chunk(i) => {
                        one(&doc.get_key(&layer.chunks_key).ok_or_else(missing)?.spans, i, view)?
                    }
                });
                if n > limit {
                    break;
                }
            }
            n
        } else {
            0
        };
        if n > limit {
            return Err(over("max_fragments", "heatmap input fragments", n));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_analysis::{Provenance, Span};
    #[test]
    fn candidate_fragments_are_refused_before_building_a_lane() {
        let mut d = Document::new("abc");
        let mut spans = Spans::new(ViewId::SOURCE);
        for i in 0..3 {
            spans.push(Span::new(i, i + 1));
        }
        d.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("test")).unwrap();
        assert!(check(&d, ViewId::SOURCE, 3).is_ok());
        let e = check(&d, ViewId::SOURCE, 2).unwrap_err();
        assert_eq!(e.code, crate::api::TURBO_E_CAPACITY);
        assert!(e.message.contains("max_fragments"));
    }
}
