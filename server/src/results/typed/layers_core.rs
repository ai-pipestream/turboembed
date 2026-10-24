//! Core NLP segmentation, morphology, and syntax layer projections
//! (`Sentences`, `Tokens`, `Words`, `Subwords`, `PosTags`, `WordPosTags`,
//! `MorphFeatures`, `Lemmas`, `WordLemmas`, `Stems`, `Terms`, `ChunkTags`,
//! `PhraseChunks`, `Phrases`, and `Dependencies`).

use std::any::TypeId;

use turbo_analysis::{Ix, Key, KeyId, Layer};
use turbo_analysis_layers::{
    ChunkTags, Dependencies, Lemmas, MorphFeatures, PhraseChunks, Phrases, PosTags, Sentences, Stems, Subwords, Terms,
    Tokens, WordLemmas, WordPosTags, Words,
};

use super::{ProjectionCx, build_layer};
use crate::proto::result::typed_annotation::Value;
use crate::proto::result::{
    ChunkTagAnnotation, DependencyAnnotation, LemmaAnnotation, MorphFeaturesAnnotation, PhraseAnnotation,
    PhraseChunkAnnotation, PosTagAnnotation, SentenceAnnotation, SpanFragment, StemAnnotation, SubwordAnnotation,
    TermAnnotation, TokenAnnotation, TypedAnnotation, TypedLayer, WordAnnotation,
};

pub(crate) fn token_spans(cx: &ProjectionCx<'_>, key: KeyId, token_ix: usize) -> Vec<SpanFragment> {
    let tok_key = cx
        .doc
        .made_on(key)
        .iter()
        .find(|k| k.ty == TypeId::of::<Tokens>())
        .map_or(Key::<Tokens>::DEFAULT, |k| Key::<Tokens>::named(k.name));
    match cx.doc.get_key(&tok_key) {
        Some(tokens) if token_ix < tokens.spans.len() => cx.project_spans(&tokens.spans, Ix::new(token_ix as u32)),
        _ => Vec::new(),
    }
}

pub(crate) fn word_spans(cx: &ProjectionCx<'_>, key: KeyId, word_ix: usize) -> Vec<SpanFragment> {
    let words_key = cx
        .doc
        .made_on(key)
        .iter()
        .find(|k| k.ty == TypeId::of::<Words>())
        .map_or(Key::<Words>::DEFAULT, |k| Key::<Words>::named(k.name));
    let Some(words) = cx.doc.get_key(&words_key) else { return Vec::new() };
    let Some(tokens) = cx.doc.get_key(&words.tokens_key) else { return Vec::new() };
    if word_ix >= words.len() {
        return Vec::new();
    }
    let span = words.token_span(Ix::new(word_ix as u32), tokens);
    vec![cx.project_wire_span(tokens.spans.view(), span)]
}

pub(crate) fn project_core_layer(cx: &mut ProjectionCx<'_>, key: KeyId) -> Option<TypedLayer> {
    let ty = key.ty;
    if ty == TypeId::of::<Sentences>() {
        return Some(build_layer::<Sentences>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    let span = l.spans.get(ix);
                    let text = cx.target_slice(l.spans.view(), span).to_owned();
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: None,
                        model_ref: mref.clone(),
                        value: Some(Value::Sentence(SentenceAnnotation { index: i as u32, text })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Tokens>() {
        return Some(build_layer::<Tokens>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    let span = l.spans.get(ix);
                    let surface = cx.target_slice(l.spans.view(), span).to_owned();
                    let sentence_index = l.sentence.as_ref().map(|s| s[i].get() as u32);
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: None,
                        model_ref: mref.clone(),
                        value: Some(Value::Token(TokenAnnotation { sentence_index, surface })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Words>() {
        return Some(build_layer::<Words>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            let tokens = cx.doc.get_key(&l.tokens_key);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    let r = l.tokens[i];
                    let (spans, subspan, form) = match tokens {
                        Some(toks) => {
                            let view = toks.spans.view();
                            let view_text = cx.doc.text(view);
                            let t_span = l.token_span(ix, toks);
                            let sub = l.subspan_of(ix).map(|s| cx.project_wire_span(view, s));
                            let f = l.form(ix, view_text, toks).to_owned();
                            (vec![cx.project_wire_span(view, t_span)], sub, f)
                        }
                        None => (
                            Vec::new(),
                            None,
                            l.form.as_ref().and_then(|f| f[i].as_deref()).unwrap_or_default().to_owned(),
                        ),
                    };
                    TypedAnnotation {
                        spans,
                        probability: None,
                        model_ref: mref.clone(),
                        value: Some(Value::Word(WordAnnotation {
                            token_start: r.start.get() as u32,
                            token_end: r.end.get() as u32,
                            subspan,
                            form,
                        })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Subwords>() {
        return Some(build_layer::<Subwords>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    let span = l.spans.get(ix);
                    let surface = cx.target_slice(l.spans.view(), span).to_owned();
                    let token_index = l.token.as_ref().map(|t| t[i].get() as u32);
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: None,
                        model_ref: mref.clone(),
                        value: Some(Value::Subword(SubwordAnnotation { vocab_id: l.ids[i], token_index, surface })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<PosTags>() {
        return Some(build_layer::<PosTags>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| TypedAnnotation {
                    spans: token_spans(cx, key, i),
                    probability: l.prob.as_ref().map(|p| p[i]),
                    model_ref: mref.clone(),
                    value: Some(Value::PosTag(PosTagAnnotation {
                        tagset: l.tagset.name.clone(),
                        tag: l.tagset.label(l.tags[i]).to_owned(),
                        target_index: i as u32,
                    })),
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<WordPosTags>() {
        return Some(build_layer::<WordPosTags>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| TypedAnnotation {
                    spans: word_spans(cx, key, i),
                    probability: l.prob.as_ref().map(|p| p[i]),
                    model_ref: mref.clone(),
                    value: Some(Value::PosTag(PosTagAnnotation {
                        tagset: l.tagset.name.clone(),
                        tag: l.tagset.label(l.tags[i]).to_owned(),
                        target_index: i as u32,
                    })),
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<MorphFeatures<Tokens>>() {
        return Some(build_layer::<MorphFeatures<Tokens>>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| TypedAnnotation {
                    spans: token_spans(cx, key, i),
                    probability: None,
                    model_ref: mref.clone(),
                    value: Some(Value::MorphFeatures(MorphFeaturesAnnotation {
                        target_index: i as u32,
                        features: l.feats.get(i).to_owned(),
                    })),
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<MorphFeatures<Words>>() {
        return Some(build_layer::<MorphFeatures<Words>>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| TypedAnnotation {
                    spans: word_spans(cx, key, i),
                    probability: None,
                    model_ref: mref.clone(),
                    value: Some(Value::MorphFeatures(MorphFeaturesAnnotation {
                        target_index: i as u32,
                        features: l.feats.get(i).to_owned(),
                    })),
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Lemmas>() {
        return Some(build_layer::<Lemmas>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| TypedAnnotation {
                    spans: token_spans(cx, key, i),
                    probability: None,
                    model_ref: mref.clone(),
                    value: Some(Value::Lemma(LemmaAnnotation {
                        target_index: i as u32,
                        lemma: l.lemma.get(i).to_owned(),
                    })),
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<WordLemmas>() {
        return Some(build_layer::<WordLemmas>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| TypedAnnotation {
                    spans: word_spans(cx, key, i),
                    probability: None,
                    model_ref: mref.clone(),
                    value: Some(Value::Lemma(LemmaAnnotation {
                        target_index: i as u32,
                        lemma: l.lemma.get(i).to_owned(),
                    })),
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Stems>() {
        return Some(build_layer::<Stems>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| TypedAnnotation {
                    spans: token_spans(cx, key, i),
                    probability: None,
                    model_ref: mref.clone(),
                    value: Some(Value::Stem(StemAnnotation { token_index: i as u32, stem: l.stem.get(i).to_owned() })),
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Terms>() {
        return Some(build_layer::<Terms>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| {
                    let slot = l.get(Ix::new(i as u32));
                    TypedAnnotation {
                        spans: token_spans(cx, key, i),
                        probability: None,
                        model_ref: mref.clone(),
                        value: Some(Value::Term(TermAnnotation {
                            token_index: i as u32,
                            surface: slot.surface.to_owned(),
                            folded: slot.folded.to_owned(),
                            stem: slot.stem.map(ToOwned::to_owned),
                            lemma: slot.lemma.map(ToOwned::to_owned),
                        })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<ChunkTags>() {
        return Some(build_layer::<ChunkTags>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| TypedAnnotation {
                    spans: token_spans(cx, key, i),
                    probability: l.prob.as_ref().map(|p| p[i]),
                    model_ref: mref.clone(),
                    value: Some(Value::ChunkTag(ChunkTagAnnotation {
                        tagset: l.tagset.name.clone(),
                        tag: l.tagset.label(l.tags[i]).to_owned(),
                        token_index: i as u32,
                    })),
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<PhraseChunks>() {
        return Some(build_layer::<PhraseChunks>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    let r = l.tokens[i];
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: None,
                        model_ref: mref.clone(),
                        value: Some(Value::PhraseChunk(PhraseChunkAnnotation {
                            tagset: l.kinds.name.clone(),
                            kind: l.kinds.label(l.kind[i]).to_owned(),
                            token_start: r.start.get() as u32,
                            token_end: r.end.get() as u32,
                        })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Phrases>() {
        return Some(build_layer::<Phrases>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    let r = l.tokens[i];
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: None,
                        model_ref: mref.clone(),
                        value: Some(Value::Phrase(PhraseAnnotation {
                            tagset: l.kinds.name.clone(),
                            kind: l.kinds.label(l.kind[i]).to_owned(),
                            token_start: r.start.get() as u32,
                            token_end: r.end.get() as u32,
                            head_token: l.head[i].get() as u32,
                            parent_index: l.parent[i].map(|p| p.get() as u32),
                        })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Dependencies<Tokens>>() {
        return Some(build_layer::<Dependencies<Tokens>>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| TypedAnnotation {
                    spans: token_spans(cx, key, i),
                    probability: None,
                    model_ref: mref.clone(),
                    value: Some(Value::Dependency(DependencyAnnotation {
                        tagset: l.relations.name.clone(),
                        target_index: i as u32,
                        head_index: l.head[i].map(|h| h.get() as u32),
                        relation: l.relations.label(l.relation[i]).to_owned(),
                    })),
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Dependencies<Words>>() {
        return Some(build_layer::<Dependencies<Words>>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| TypedAnnotation {
                    spans: word_spans(cx, key, i),
                    probability: None,
                    model_ref: mref.clone(),
                    value: Some(Value::Dependency(DependencyAnnotation {
                        tagset: l.relations.name.clone(),
                        target_index: i as u32,
                        head_index: l.head[i].map(|h| h.get() as u32),
                        relation: l.relations.label(l.relation[i]).to_owned(),
                    })),
                })
                .collect()
        }));
    }
    None
}
