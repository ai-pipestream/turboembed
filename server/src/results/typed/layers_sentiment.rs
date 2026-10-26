//! Language, category, and sentiment layer projections
//! (`Language`, `LanguageSpans`, `Categories`, `Aspects`, `Opinions`,
//! `EmotionLayer`, `Incivility`, and `Pragmatics`).

use std::any::TypeId;

use turbo_analysis::{Ix, Key, KeyId, Layer, Span, Tag, View};
use turbo_analysis_layers::{
    Aspects, Categories, CategoryScope, Chunks, EmotionLayer, EmotionTarget, Incivility, IncivilityTarget, Language,
    LanguageSpans, Opinions, Polarity, Pragmatics, Sentences, Structure, Tokens,
};

use super::{ProjectionCx, build_layer};
use crate::proto::result::typed_annotation::Value;
use crate::proto::result::{
    AspectAnnotation, CategoryAnnotation, CategoryScoreProto, EmotionAnnotation, IncivilityAnnotation,
    LabelTokenSpanProto, LanguageAnnotation, LanguageSpanAnnotation, OpinionAnnotation, PragmaticsAnnotation,
    SpanFragment, TypedAnnotation, TypedLayer,
};

fn project_categories<S: CategoryScope>(
    cx: &mut ProjectionCx<'_>,
    key: KeyId,
    item_spans: impl Fn(&ProjectionCx<'_>, KeyId, usize) -> Vec<SpanFragment>,
) -> TypedLayer {
    build_layer::<Categories<S>>(cx, key, |cx, l, mref| {
        let mref = mref.map(ToOwned::to_owned);
        (0..l.len())
            .map(|i| {
                let row = l.get(i).expect("category row in bounds");
                let abstained = row.best_tag == Tag::NONE;
                let probability = if abstained {
                    None
                } else {
                    let s = row.best_score();
                    (s.is_finite() && (0.0..=1.0).contains(&s)).then_some(s)
                };
                let selected = row.selected.iter().map(|&t| row.categories.label(t).to_owned()).collect();
                let scores = row
                    .iter_scores()
                    .map(|(label, score)| CategoryScoreProto { label: label.to_owned(), score })
                    .collect();
                TypedAnnotation {
                    spans: item_spans(cx, key, i),
                    probability,
                    model_ref: mref.clone(),
                    value: Some(Value::Category(CategoryAnnotation {
                        tagset: l.categories.name.clone(),
                        best_label: row.best.to_owned(),
                        abstained,
                        selected,
                        scores,
                    })),
                }
            })
            .collect()
    })
}

fn anchor_spans<L: Layer>(
    cx: &ProjectionCx<'_>,
    key: KeyId,
    index: usize,
    get: impl FnOnce(&ProjectionCx<'_>, &L, usize) -> Vec<SpanFragment>,
) -> Vec<SpanFragment> {
    let target_key = cx
        .doc
        .made_on(key)
        .iter()
        .find(|k| k.ty == TypeId::of::<L>())
        .map_or(Key::<L>::DEFAULT, |k| Key::<L>::named(k.name));
    match cx.doc.get_key(&target_key) {
        Some(layer) if index < layer.len() => get(cx, layer, index),
        _ => Vec::new(),
    }
}

pub(crate) fn project_sentiment_layer(cx: &mut ProjectionCx<'_>, key: KeyId) -> Option<TypedLayer> {
    let ty = key.ty;
    if ty == TypeId::of::<Language>() {
        return Some(build_layer::<Language>(cx, key, |_, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            l.iter()
                .map(|(code, score)| TypedAnnotation {
                    spans: Vec::new(),
                    probability: (score.is_finite() && (0.0..=1.0).contains(&score)).then_some(score),
                    model_ref: mref.clone(),
                    value: Some(Value::Language(LanguageAnnotation { code: code.to_owned(), score })),
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<LanguageSpans>() {
        return Some(build_layer::<LanguageSpans>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    let ls = l.get(ix);
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: Some(ls.confidence),
                        model_ref: mref.clone(),
                        value: Some(Value::LanguageSpan(LanguageSpanAnnotation {
                            code: ls.code.to_owned(),
                            script: format!("{:?}", ls.script),
                            confidence: ls.confidence,
                        })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Categories<View>>() {
        return Some(project_categories::<View>(cx, key, |_, _, _| Vec::new()));
    }
    if ty == TypeId::of::<Categories<Sentences>>() {
        return Some(project_categories::<Sentences>(cx, key, |cx, key, i| {
            anchor_spans::<Sentences>(cx, key, i, |cx, s, idx| cx.project_spans(&s.spans, Ix::new(idx as u32)))
        }));
    }
    if ty == TypeId::of::<Categories<Structure>>() {
        return Some(project_categories::<Structure>(cx, key, |cx, key, i| {
            anchor_spans::<Structure>(cx, key, i, |cx, s, idx| cx.project_spans(&s.spans, Ix::new(idx as u32)))
        }));
    }
    if ty == TypeId::of::<Categories<Chunks>>() {
        return Some(project_categories::<Chunks>(cx, key, |cx, key, i| {
            anchor_spans::<Chunks>(cx, key, i, |cx, s, idx| cx.project_spans(&s.spans, Ix::new(idx as u32)))
        }));
    }
    if ty == TypeId::of::<Categories<Opinions>>() {
        return Some(project_categories::<Opinions>(cx, key, |cx, key, i| {
            anchor_spans::<Opinions>(cx, key, i, |cx, ops, idx| {
                cx.doc
                    .get_key(&ops.tokens_key)
                    .map(|toks| vec![cx.project_wire_span(toks.spans.view(), ops.span(Ix::new(idx as u32), toks))])
                    .unwrap_or_default()
            })
        }));
    }
    if ty == TypeId::of::<Categories<Aspects>>() {
        return Some(project_categories::<Aspects>(cx, key, |cx, key, i| {
            anchor_spans::<Aspects>(cx, key, i, |cx, asp, idx| {
                let toks = cx.doc.get_key(&asp.tokens_key);
                let sents = cx.doc.get_key(&asp.sentences_key);
                toks.and_then(|t| {
                    asp.span(Ix::new(idx as u32), t, sents).map(|sp| vec![cx.project_wire_span(t.spans.view(), sp)])
                })
                .unwrap_or_default()
            })
        }));
    }
    if ty == TypeId::of::<Aspects>() {
        return Some(build_layer::<Aspects>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            let toks = cx.doc.get_key(&l.tokens_key);
            let sents = cx.doc.get_key(&l.sentences_key);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    let a = l.get(ix);
                    let spans = match (toks, sents) {
                        (Some(t), s) => l.span(ix, t, s).map(|sp| vec![cx.project_wire_span(t.spans.view(), sp)]),
                        (None, Some(s)) => a
                            .target
                            .sentence()
                            .filter(|&si| si.get() < s.len())
                            .map(|si| cx.project_spans(&s.spans, si)),
                        _ => None,
                    }
                    .unwrap_or_default();
                    let (token_start, token_end) = match a.target.tokens() {
                        Some(r) => (Some(r.start.get() as u32), Some(r.end.get() as u32)),
                        None => (None, None),
                    };
                    let dom = a.polarity.positive.max(a.polarity.negative).max(a.polarity.neutral);
                    TypedAnnotation {
                        spans,
                        probability: (dom.is_finite() && (0.0..=1.0).contains(&dom)).then_some(dom),
                        model_ref: mref.clone(),
                        value: Some(Value::Aspect(AspectAnnotation {
                            token_start,
                            token_end,
                            sentence_index: a.target.sentence().map(|s| s.get() as u32),
                            category: l.category_label(ix).map(ToOwned::to_owned),
                            polarity: a.polarity.label().name().to_owned(),
                            positive: a.polarity.positive,
                            negative: a.polarity.negative,
                            neutral: a.polarity.neutral,
                            opinion_indices: a.opinions.iter().map(|o| o.get() as u32).collect(),
                        })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Opinions>() {
        return Some(build_layer::<Opinions>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            let toks = cx.doc.get_key(&l.tokens_key);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    let op = l.get(ix);
                    let spans = toks
                        .filter(|t| op.tokens.end.get() <= t.len())
                        .map(|t| vec![cx.project_wire_span(t.spans.view(), l.span(ix, t))])
                        .unwrap_or_default();
                    TypedAnnotation {
                        spans,
                        probability: None,
                        model_ref: mref.clone(),
                        value: Some(Value::Opinion(OpinionAnnotation {
                            token_start: op.tokens.start.get() as u32,
                            token_end: op.tokens.end.get() as u32,
                            polarity: op.polarity,
                            intensity: op.intensity,
                            negated: op.negated,
                        })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<EmotionLayer>() {
        return Some(build_layer::<EmotionLayer>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            let sents = cx.doc.get_key(&l.sentences_key);
            let chunks = cx.doc.get_key(&l.chunks_key);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    let (target_kind, target_index, spans) = match l.target[i] {
                        EmotionTarget::Sentence(s) => (
                            "sentence",
                            s.get() as u32,
                            sents
                                .filter(|sl| s.get() < sl.len())
                                .map(|sl| cx.project_spans(&sl.spans, s))
                                .unwrap_or_default(),
                        ),
                        EmotionTarget::Chunk(c) => (
                            "chunk",
                            c.get() as u32,
                            chunks
                                .filter(|cl| c.get() < cl.len())
                                .map(|cl| cx.project_spans(&cl.spans, c))
                                .unwrap_or_default(),
                        ),
                    };
                    let scores = l.scores_of(ix).0.to_vec();
                    let evidence = l
                        .evidence_of(ix)
                        .map(|(dim, r)| LabelTokenSpanProto {
                            label: dim.name().to_owned(),
                            token_start: r.start.get() as u32,
                            token_end: r.end.get() as u32,
                        })
                        .collect();
                    TypedAnnotation {
                        spans,
                        probability: None,
                        model_ref: mref.clone(),
                        value: Some(Value::Emotion(EmotionAnnotation {
                            target_kind: target_kind.to_owned(),
                            target_index,
                            scores,
                            evidence,
                        })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Incivility>() {
        return Some(build_layer::<Incivility>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            let sents = cx.doc.get_key(&l.sentences_key);
            let chunks = cx.doc.get_key(&l.chunks_key);
            let structure = cx.doc.get_key(&l.structure_key);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    let target = l.target[i];
                    let spans = match target {
                        IncivilityTarget::Sentence(s) => {
                            sents.filter(|sl| s.get() < sl.len()).map(|sl| cx.project_spans(&sl.spans, s))
                        }
                        IncivilityTarget::Chunk(c) => {
                            chunks.filter(|cl| c.get() < cl.len()).map(|cl| cx.project_spans(&cl.spans, c))
                        }
                        IncivilityTarget::Block(b) => {
                            structure.filter(|st| b.get() < st.len()).map(|st| cx.project_spans(&st.spans, b))
                        }
                    }
                    .unwrap_or_default();
                    let sc = l.scores(ix);
                    let evidence = l
                        .evidence_of(ix)
                        .map(|(lbl, r)| LabelTokenSpanProto {
                            label: lbl.name().to_owned(),
                            token_start: r.start.get() as u32,
                            token_end: r.end.get() as u32,
                        })
                        .collect();
                    TypedAnnotation {
                        spans,
                        probability: None,
                        model_ref: mref.clone(),
                        value: Some(Value::Incivility(IncivilityAnnotation {
                            target_kind: target.kind_name().to_owned(),
                            target_index: target.raw_index(),
                            insult: sc.insult,
                            profanity: sc.profanity,
                            threat: sc.threat,
                            identity_attack: sc.identity_attack,
                            intolerance: sc.intolerance,
                            severe_toxicity: sc.severe_toxicity,
                            evidence,
                        })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Pragmatics>() {
        return Some(build_layer::<Pragmatics>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| {
                    let item = l.get(Ix::new(i as u32));
                    let spans = anchor_spans::<Sentences>(cx, key, i, |cx, s, idx| {
                        cx.project_spans(&s.spans, Ix::new(idx as u32))
                    });
                    let (pos_s, pos_e, neg_s, neg_e) = match item.evidence {
                        Some(inc) => (
                            Some(inc.positive.start.get() as u32),
                            Some(inc.positive.end.get() as u32),
                            Some(inc.negative.start.get() as u32),
                            Some(inc.negative.end.get() as u32),
                        ),
                        None => (None, None, None, None),
                    };
                    TypedAnnotation {
                        spans,
                        probability: Some(item.score),
                        model_ref: mref.clone(),
                        value: Some(Value::Pragmatics(PragmaticsAnnotation {
                            sentence_index: i as u32,
                            score: item.score,
                            positive_token_start: pos_s,
                            positive_token_end: pos_e,
                            negative_token_start: neg_s,
                            negative_token_end: neg_e,
                        })),
                    }
                })
                .collect()
        }));
    }
    let _ = (Span::new(0, 0), Polarity::Neutral, Tokens::ID);
    None
}
