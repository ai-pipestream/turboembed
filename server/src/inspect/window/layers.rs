//! Collection of each named positional layer instance for document inspection.
use super::{ClippedFrag, LayerFilter, bad, ordered_range, push_clipped_item};
use crate::api::Failure;
use std::any::TypeId;
use turbo_analysis::{Document, Ix, Key, Layer, Span, ViewId};
use turbo_analysis_layers::{
    Abbreviations, Boilerplate, Chunks, Entities, LanguageSpans, Numbers, Pii, PiiConfidence, PosTags, Sentences,
    Structure, Subwords, Tokens,
};

pub(super) fn collect_fragments(
    doc: &Document,
    view: ViewId,
    window: Span,
    filter: LayerFilter<'_>,
) -> Result<Vec<ClippedFrag>, Failure> {
    let mut clipped = Vec::new();
    for key in doc.layers() {
        let instance = key.name;

        if key.ty == TypeId::of::<Sentences>()
            && filter.allows(Sentences::ID, instance)
            && let Some(sentences) = doc.get_key(&Key::<Sentences>::named(instance))
            && sentences.spans.view() == view
        {
            for i in ordered_range(&sentences.spans, window) {
                push_clipped_item(
                    &mut clipped,
                    &sentences.spans,
                    i,
                    window,
                    (Sentences::ID, instance),
                    format!("S{i}"),
                    1.0,
                )?;
            }
        }

        if key.ty == TypeId::of::<Tokens>()
            && filter.allows(Tokens::ID, instance)
            && let Some(tokens) = doc.get_key(&Key::<Tokens>::named(instance))
            && tokens.spans.view() == view
        {
            let tok_key = Key::<Tokens>::named(instance).id();
            let pos = [Key::<PosTags>::named(instance), Key::<PosTags>::DEFAULT]
                .into_iter()
                .find_map(|key| doc.get_key(&key).filter(|_| doc.made_on(key.id()).contains(&tok_key)));
            for i in 0..tokens.spans.len() {
                let (label, score) = match pos {
                    Some(p) if i < p.tags.len() => {
                        let lbl = p.tagset.label(p.tags[i]).to_owned();
                        let sc = p.prob.as_ref().and_then(|pr| pr.get(i).copied()).unwrap_or(1.0);
                        (lbl, sc)
                    }
                    _ => ("token".to_owned(), 1.0),
                };
                push_clipped_item(&mut clipped, &tokens.spans, i, window, (Tokens::ID, instance), label, score)?;
            }
        }

        // Subwords uses Overlap::Any, so its ends column is not guaranteed monotonic;
        // scan items directly rather than binary-searching ends.
        if key.ty == TypeId::of::<Subwords>()
            && filter.allows(Subwords::ID, instance)
            && let Some(subwords) = doc.get_key(&Key::<Subwords>::named(instance))
            && subwords.spans.view() == view
        {
            for i in 0..subwords.spans.len() {
                push_clipped_item(
                    &mut clipped,
                    &subwords.spans,
                    i,
                    window,
                    (Subwords::ID, instance),
                    "subword".to_owned(),
                    1.0,
                )?;
            }
        }

        if key.ty == TypeId::of::<Structure>()
            && filter.allows(Structure::ID, instance)
            && let Some(structure) = doc.get_key(&Key::<Structure>::named(instance))
            && structure.spans.view() == view
        {
            for i in 0..structure.spans.len() {
                let label = format!("{:?}", structure.kind[i]);
                push_clipped_item(&mut clipped, &structure.spans, i, window, (Structure::ID, instance), label, 1.0)?;
            }
        }

        if key.ty == TypeId::of::<Entities>()
            && filter.allows(Entities::ID, instance)
            && let Some(entities) = doc.get_key(&Key::<Entities>::named(instance))
            && entities.spans.view() == view
        {
            for i in 0..entities.spans.len() {
                let label = entities.kinds.label(entities.kind[i]).to_owned();
                let score = entities.score.get(i).copied().unwrap_or(1.0);
                push_clipped_item(&mut clipped, &entities.spans, i, window, (Entities::ID, instance), label, score)?;
            }
        }

        if key.ty == TypeId::of::<Chunks>()
            && filter.allows(Chunks::ID, instance)
            && let Some(chunks) = doc.get_key(&Key::<Chunks>::named(instance))
            && chunks.spans.view() == view
        {
            for i in 0..chunks.spans.len() {
                let label = format!("L{}:{:?}", chunks.level[i], chunks.cut[i]);
                let score = chunks.cost[i] as f64;
                push_clipped_item(&mut clipped, &chunks.spans, i, window, (Chunks::ID, instance), label, score)?;
            }
        }

        if key.ty == TypeId::of::<Pii>()
            && filter.allows(Pii::ID, instance)
            && let Some(pii) = doc.get_key(&Key::<Pii>::named(instance))
            && pii.spans.view() == view
        {
            for i in 0..pii.spans.len() {
                let label = pii.kind_name(pii.kind[i]).unwrap_or("").to_owned();
                let score = match pii.get(Ix::new(i as u32)).confidence() {
                    PiiConfidence::Format => 0.5,
                    PiiConfidence::Labelled => 0.75,
                    PiiConfidence::Checksum => 0.85,
                    PiiConfidence::LabelledChecksum => 1.0,
                };
                push_clipped_item(&mut clipped, &pii.spans, i, window, (Pii::ID, instance), label, score)?;
            }
        }

        if key.ty == TypeId::of::<Numbers>()
            && filter.allows(Numbers::ID, instance)
            && let Some(numbers) = doc.get_key(&Key::<Numbers>::named(instance))
            && numbers.spans.view() == view
        {
            for i in 0..numbers.spans.len() {
                let label = format!("{:?}", numbers.notation[i]);
                push_clipped_item(&mut clipped, &numbers.spans, i, window, (Numbers::ID, instance), label, 1.0)?;
            }
        }

        if key.ty == TypeId::of::<LanguageSpans>()
            && filter.allows(LanguageSpans::ID, instance)
            && let Some(langs) = doc.get_key(&Key::<LanguageSpans>::named(instance))
            && langs.spans.view() == view
        {
            for i in 0..langs.spans.len() {
                let m = langs.get(Ix::new(i as u32));
                push_clipped_item(
                    &mut clipped,
                    &langs.spans,
                    i,
                    window,
                    (LanguageSpans::ID, instance),
                    m.code.to_owned(),
                    m.confidence,
                )?;
            }
        }

        if key.ty == TypeId::of::<Abbreviations>()
            && filter.allows(Abbreviations::ID, instance)
            && let Some(abbr) = doc.get_key(&Key::<Abbreviations>::named(instance))
            && abbr.spans.view() == view
        {
            for i in 0..abbr.spans.len() {
                let label = format!("{:?}", abbr.form[i]);
                push_clipped_item(&mut clipped, &abbr.spans, i, window, (Abbreviations::ID, instance), label, 1.0)?;
            }
        }

        if key.ty == TypeId::of::<Boilerplate<Sentences>>()
            && filter.allows(Boilerplate::<Sentences>::ID, instance)
            && let Some(bp) = doc.get_key(&Key::<Boilerplate<Sentences>>::named(instance))
        {
            let anchor = doc
                .made_on(key)
                .iter()
                .find(|k| k.ty == TypeId::of::<Sentences>())
                .ok_or_else(|| bad(format!("{key} has no recorded sentence binding")))?;
            let sentences = doc
                .get_key(&Key::<Sentences>::named(anchor.name))
                .ok_or_else(|| bad(format!("{key} has a missing sentence binding {anchor}")))?;
            if sentences.spans.view() == view {
                for i in 0..bp.kind.len() {
                    if let Some(kind) = bp.kind[i] {
                        push_clipped_item(
                            &mut clipped,
                            &sentences.spans,
                            i,
                            window,
                            (Boilerplate::<Sentences>::ID, instance),
                            format!("{kind:?}"),
                            bp.score[i] as f64,
                        )?;
                    }
                }
            }
        }
    }
    Ok(clipped)
}
