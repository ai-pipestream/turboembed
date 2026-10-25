//! Input limits before the shared builder allocates item/fragment storage.
use super::super::bad;
use crate::{api::Failure, results::over};
use std::any::TypeId;
use turbo_analysis::{Document, Ix, Key, KeyId, Layer, Spans, Status, ViewId};
use turbo_analysis_layers::{
    Categories, Chunks, EmotionLayer, EmotionTarget, Entities, Incivility, IncivilityTarget, Pii, Sentences, Tokens,
};

fn all<L: Layer>(doc: &Document, key: KeyId, view: ViewId) -> Option<u64> {
    doc.get_key(&Key::<L>::named(key.name))
        .and_then(Layer::text_spans)
        .filter(|s| s.view == view)
        .map(|s| if s.frag_item_end.is_empty() { s.start.len() } else { s.frag_start.len() } as u64)
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

pub(super) fn check(doc: &Document, view: ViewId, buckets: u32, limit: u64) -> Result<(), Failure> {
    let mut response_bound = 0u64;
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
            (n > 0).then_some(n)
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
            (n > 0).then_some(n)
        } else {
            None
        };
        let Some(n) = n else {
            continue;
        };
        if n > limit {
            return Err(over("max_fragments", "heatmap input fragments", n));
        }
        // Exact segmentation has at most two boundaries/point contributions
        // per fragment plus one uncovered interval. Binned output is exact N.
        let segments = if buckets > 0 {
            u64::from(buckets)
        } else if doc.text(view).is_empty() {
            u64::from(n > 0)
        } else {
            n.saturating_mul(2).saturating_add(1)
        };
        response_bound = response_bound.saturating_add(segments);
        if response_bound > limit {
            return Err(over("max_fragments", "heatmap prospective segments", response_bound));
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
        assert!(check(&d, ViewId::SOURCE, 1, 3).is_ok());
        let e = check(&d, ViewId::SOURCE, 1, 2).unwrap_err();
        assert_eq!(e.code, crate::api::TURBO_E_CAPACITY);
        assert!(e.message.contains("max_fragments"));
    }
    #[test]
    fn empty_named_sources_cannot_expand_past_the_response_budget() {
        let mut d = Document::new("abc");
        for name in ["a", "b"] {
            d.add(
                &Key::named(name),
                Tokens { spans: Spans::new(ViewId::SOURCE), sentence: None },
                Provenance::by("test"),
            )
            .unwrap();
        }
        let source = d.get::<turbo_analysis::View>().unwrap().clone();
        let mut b = turbo_analysis::ViewBuilder::new(source.text());
        b.keep(source.text().len());
        let (text, align) = b.finish();
        d.add(&Key::named("child"), turbo_analysis::View::derived(&source, text, align), Provenance::by("test"))
            .unwrap();
        d.add(&Key::named("other"), Tokens { spans: Spans::new(ViewId(1)), sentence: None }, Provenance::by("test"))
            .unwrap();
        // Use the real adapter and its production budget path, with a small
        // limit so no oversized graph or allocation is needed.
        let offsetter = crate::offsets::Offsetter::new(d.source(), crate::offsets::OffsetMode::Utf8);
        let build = |limit| super::super::build_with_budget(&d, ViewId::SOURCE, d.source(), &offsetter, 3, limit);
        assert_eq!(build(6).unwrap().iter().map(|l| l.buckets.len()).sum::<usize>(), 6);
        let e = build(5).expect_err("two three-bin lanes exceed a five-segment result budget");
        assert_eq!(e.code, crate::api::TURBO_E_CAPACITY);
        let exact = |limit| super::super::build_with_budget(&d, ViewId::SOURCE, d.source(), &offsetter, 0, limit);
        assert_eq!(exact(2).unwrap().iter().map(|l| l.buckets.len()).sum::<usize>(), 2);
        assert_eq!(exact(1).unwrap_err().code, crate::api::TURBO_E_CAPACITY);
    }
}
