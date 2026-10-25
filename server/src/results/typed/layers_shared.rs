//! Shared rhetoric, modality, event, consolidated entity, sparse term,
//! lattice/n-best alternative, WordNet, table, and schema-record layer
//! projections (`#1219`).

use std::any::TypeId;

use turbo_analysis::{Alternatives, Edges, Ix, Key, KeyId, Layer, NBest, Spans, View, ViewId};
use turbo_analysis_layers::{
    AnchoredLayerKind, ArgumentRole, ChunkTags, DeonticKind, Discourse, DocumentEntities, Edus, Entities,
    EpistemicStatus, Events, Levels, Modality, ModalityPolarity, PosTags, Sentences, SparseTerms, SyntacticRole,
    Tokens,
};
use turbo_geo::Locations;

use super::layers_core::token_spans;
use super::{ProjectionCx, build_layer, coalesce_wire_spans};
use crate::proto::result::typed_annotation::Value;
use crate::proto::result::{
    AlternativeAnnotation, ConsolidatedMentionProto, DocumentEntityAnnotation, EdgeAnnotation, EduAnnotation,
    EntityRelationProto, EventAnnotation, EventArgumentProto, ModalityAnnotation, NBestAnnotation, NBestCandidateProto,
    SpanFragment, SparseTermEntryProto, SparseTermsAnnotation, TypedAnnotation, TypedLayer,
};

fn modality_polarity_name(p: ModalityPolarity) -> &'static str {
    match p {
        ModalityPolarity::Affirmed => "affirmed",
        ModalityPolarity::Negated => "negated",
    }
}

fn epistemic_name(e: EpistemicStatus) -> &'static str {
    match e {
        EpistemicStatus::Asserted => "asserted",
        EpistemicStatus::Hedged => "hedged",
        EpistemicStatus::Reported => "reported",
        EpistemicStatus::Conditional => "conditional",
        EpistemicStatus::Presupposed => "presupposed",
    }
}

fn deontic_name(d: DeonticKind) -> &'static str {
    match d {
        DeonticKind::None => "none",
        DeonticKind::Obligation => "obligation",
        DeonticKind::Permission => "permission",
        DeonticKind::Prohibition => "prohibition",
    }
}

fn role_name(r: SyntacticRole) -> &'static str {
    match r {
        SyntacticRole::Subject => "subject",
        SyntacticRole::Object => "object",
        SyntacticRole::Modifier => "modifier",
        SyntacticRole::Other => "other",
    }
}

fn arg_role_name(r: ArgumentRole) -> &'static str {
    match r {
        ArgumentRole::Agent => "agent",
        ArgumentRole::Patient => "patient",
        ArgumentRole::Recipient => "recipient",
        ArgumentRole::Time => "time",
        ArgumentRole::Location => "location",
        ArgumentRole::Measure => "measure",
        ArgumentRole::Instrument => "instrument",
        ArgumentRole::Cause => "cause",
        ArgumentRole::Purpose => "purpose",
        ArgumentRole::Attribute => "attribute",
    }
}

fn anchored_layer_name(k: AnchoredLayerKind) -> &'static str {
    match k {
        AnchoredLayerKind::None => "none",
        AnchoredLayerKind::Temporals => "turbo:temporal",
        AnchoredLayerKind::Money => "turbo:money",
        AnchoredLayerKind::Quantities => "turbo:quantities",
        AnchoredLayerKind::Names => "opennlp:entities",
    }
}

fn project_alternatives<L: Layer>(cx: &mut ProjectionCx<'_>, key: KeyId) -> TypedLayer {
    build_layer::<Alternatives<L>>(cx, key, |cx, l, mref| {
        let mref = mref.map(ToOwned::to_owned);
        (0..l.len())
            .map(|i| {
                let ix = Ix::new(i as u32);
                let score = l.score.as_ref().map(|s| s[i]);
                let prob = score.filter(|s| s.is_finite() && (0.0..=1.0).contains(s));
                let form = l.form.as_ref().and_then(|f| f[i].as_deref()).map(ToOwned::to_owned);
                TypedAnnotation {
                    spans: cx.project_spans(&l.spans, ix),
                    probability: prob,
                    model_ref: mref.clone(),
                    value: Some(Value::Alternative(AlternativeAnnotation {
                        position: l.position[i],
                        position_length: l.position_length[i],
                        mode_bits: u32::from(l.mode[i].0),
                        score,
                        form,
                    })),
                }
            })
            .collect()
    })
}

fn project_nbest<L: Layer>(
    cx: &mut ProjectionCx<'_>,
    key: KeyId,
    item_spans: impl Fn(&ProjectionCx<'_>, KeyId, usize) -> Vec<SpanFragment>,
) -> TypedLayer {
    build_layer::<NBest<L>>(cx, key, |cx, l, mref| {
        let mref = mref.map(ToOwned::to_owned);
        (0..l.len())
            .map(|i| {
                let ix = Ix::<L>::new(i as u32);
                let candidates: Vec<NBestCandidateProto> = (0..l.candidates())
                    .map(|rank| {
                        let (tag, prob) = l.at(rank, ix);
                        NBestCandidateProto {
                            rank: rank as u32,
                            sequence_score: l.score_of(rank),
                            tag: l.tagset.label(tag).to_owned(),
                            probability: prob,
                        }
                    })
                    .collect();
                let best_prob = candidates.first().and_then(|c| c.probability);
                TypedAnnotation {
                    spans: item_spans(cx, key, i),
                    probability: best_prob,
                    model_ref: mref.clone(),
                    value: Some(Value::Nbest(NBestAnnotation {
                        target_index: i as u32,
                        tagset: l.tagset.name.clone(),
                        candidates,
                    })),
                }
            })
            .collect()
    })
}

fn aligned_anchor_spans<L: Layer>(
    cx: &ProjectionCx<'_>,
    key: KeyId,
    index: usize,
    spans_of: fn(&L) -> &Spans<L>,
) -> Vec<SpanFragment> {
    let target_key = cx
        .doc
        .made_on(key)
        .iter()
        .find(|k| k.ty == TypeId::of::<L>())
        .map_or(Key::<L>::DEFAULT, |k| Key::<L>::named(k.name));
    match cx.doc.get_key(&target_key) {
        Some(layer) => {
            let s = spans_of(layer);
            if index < s.len() { cx.project_spans(s, Ix::new(index as u32)) } else { Vec::new() }
        }
        None => Vec::new(),
    }
}

fn project_sparse<L: Layer>(
    cx: &mut ProjectionCx<'_>,
    key: KeyId,
    unit_view_and_spans: impl Fn(&ProjectionCx<'_>, KeyId, usize) -> (ViewId, Vec<SpanFragment>),
) -> TypedLayer {
    build_layer::<SparseTerms<L>>(cx, key, |cx, l, mref| {
        let mref = mref.map(ToOwned::to_owned);
        (0..l.len())
            .map(|i| {
                let (view, unit_spans) = unit_view_and_spans(cx, key, i);
                let r = l.row[i] as usize..l.row[i + 1] as usize;
                let entries = r
                    .map(|e| {
                        let span = l.span.get(e).copied().flatten().map(|s| cx.project_wire_span(view, s));
                        let occurrences = l.occurrences(e).iter().map(|&s| cx.project_wire_span(view, s)).collect();
                        SparseTermEntryProto { term_id: l.term[e], weight: l.weight[e], span, occurrences }
                    })
                    .collect();
                TypedAnnotation {
                    spans: unit_spans,
                    probability: None,
                    model_ref: mref.clone(),
                    value: Some(Value::SparseTerms(SparseTermsAnnotation { target_index: i as u32, entries })),
                }
            })
            .collect()
    })
}

pub(crate) fn project_shared_layer(cx: &mut ProjectionCx<'_>, key: KeyId) -> Option<TypedLayer> {
    let ty = key.ty;
    if ty == TypeId::of::<Modality>() {
        return Some(build_layer::<Modality>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            let view = l.spans.view();
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    let tag = l.tag(i);
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: Some(f64::from(tag.confidence)),
                        model_ref: mref.clone(),
                        value: Some(Value::Modality(ModalityAnnotation {
                            polarity: modality_polarity_name(tag.polarity).to_owned(),
                            epistemic: epistemic_name(tag.epistemic).to_owned(),
                            deontic: deontic_name(tag.deontic).to_owned(),
                            cue: tag.cue.map(|s| cx.project_wire_span(view, s)),
                            confidence: tag.confidence,
                        })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<DocumentEntities>() {
        return Some(build_layer::<DocumentEntities>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            l.entities
                .iter()
                .enumerate()
                .map(|(i, e)| {
                    let spans =
                        coalesce_wire_spans(e.mentions.iter().map(|m| cx.project_wire_span(l.view, m.span)).collect());
                    let mentions = e
                        .mentions
                        .iter()
                        .map(|m| ConsolidatedMentionProto {
                            span: Some(cx.project_wire_span(l.view, m.span)),
                            entity_index: m.entity.map(|ix| ix.get() as u32),
                            is_representative: m.is_representative,
                            role: m.role.map(|r| role_name(r).to_owned()),
                        })
                        .collect();
                    let relations = l
                        .relations
                        .iter()
                        .filter(|r| r.subject == i as u32)
                        .map(|r| EntityRelationProto {
                            subject: r.subject,
                            object: r.object,
                            kind: r.kind.to_string(),
                            score: r.score,
                            support: u32::from(r.support),
                        })
                        .collect();
                    let prob = (e.confidence.is_finite() && (0.0..=1.0).contains(&e.confidence))
                        .then_some(f64::from(e.confidence));
                    TypedAnnotation {
                        spans,
                        probability: prob,
                        model_ref: mref.clone(),
                        value: Some(Value::DocumentEntity(DocumentEntityAnnotation {
                            canonical_name: e.canonical_name.to_string(),
                            kind: e.kind.to_string(),
                            kb_id: e.kb_id.as_deref().map(ToOwned::to_owned),
                            latitude: e.geo_coord.map(|(lat, _)| lat),
                            longitude: e.geo_coord.map(|(_, lon)| lon),
                            country_code: e.country_code.map(|cc| String::from_utf8_lossy(&cc).into_owned()),
                            mentions,
                            subject_count: u32::from(e.role_counts.subject_count),
                            object_count: u32::from(e.role_counts.object_count),
                            modifier_count: u32::from(e.role_counts.modifier_count),
                            other_count: u32::from(e.role_counts.other_count),
                            mean_sentiment: e.mean_sentiment,
                            sentiment_mentions: u32::from(e.sentiment_mentions),
                            confidence: e.confidence,
                            relations,
                        })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Events>() {
        return Some(build_layer::<Events>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            let view = l.spans.view();
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    let f = l.get(i);
                    let arguments = f
                        .arguments
                        .iter()
                        .map(|a| EventArgumentProto {
                            span: Some(cx.project_wire_span(view, a.span)),
                            role: arg_role_name(a.role).to_owned(),
                            anchored_layer: anchored_layer_name(a.anchored_layer).to_owned(),
                            anchored_index: a.anchored_index,
                        })
                        .collect();
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: Some(f64::from(f.confidence)),
                        model_ref: mref.clone(),
                        value: Some(Value::Event(EventAnnotation {
                            trigger_lemma: f.trigger_lemma.map(ToOwned::to_owned),
                            sentence_index: f.sentence_ix,
                            confidence: f.confidence,
                            arguments,
                        })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Edus>() {
        return Some(build_layer::<Edus>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    let text = cx.target_slice(l.spans.view(), l.spans.get(ix)).to_owned();
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: None,
                        model_ref: mref.clone(),
                        value: Some(Value::Edu(EduAnnotation {
                            index: i as u32,
                            sentence_index: l.sentence[i].get() as u32,
                            text,
                        })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Discourse>() {
        return Some(build_layer::<Discourse>(cx, key, |_, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            l.iter()
                .map(|(_, edge)| TypedAnnotation {
                    spans: Vec::new(),
                    probability: (edge.score.is_finite() && (0.0..=1.0).contains(&edge.score)).then_some(edge.score),
                    model_ref: mref.clone(),
                    value: Some(Value::Edge(EdgeAnnotation {
                        from_index: edge.from.get() as u32,
                        to_index: edge.to.get() as u32,
                        kind: l.kinds.label(edge.kind).to_owned(),
                        score: edge.score,
                    })),
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Edges<Entities, Locations>>() {
        return Some(build_layer::<Edges<Entities, Locations>>(cx, key, |_, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            l.iter()
                .map(|(_, edge)| TypedAnnotation {
                    spans: Vec::new(),
                    probability: (edge.score.is_finite() && (0.0..=1.0).contains(&edge.score)).then_some(edge.score),
                    model_ref: mref.clone(),
                    value: Some(Value::Edge(EdgeAnnotation {
                        from_index: edge.from.get() as u32,
                        to_index: edge.to.get() as u32,
                        kind: l.kinds.label(edge.kind).to_owned(),
                        score: edge.score,
                    })),
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<SparseTerms<View>>() {
        return Some(project_sparse::<View>(cx, key, |cx, k, _| {
            let view_id = cx
                .doc
                .made_on(k)
                .iter()
                .find(|m| m.ty == TypeId::of::<View>())
                .and_then(|m| cx.doc.get_key(&Key::<View>::named(m.name)).map(View::id))
                .unwrap_or(ViewId::SOURCE);
            (view_id, Vec::new())
        }));
    }
    if ty == TypeId::of::<SparseTerms<Levels>>() {
        return Some(project_sparse::<Levels>(cx, key, |cx, k, i| {
            let lvl_key = cx
                .doc
                .made_on(k)
                .iter()
                .find(|m| m.ty == TypeId::of::<Levels>())
                .map_or(Key::<Levels>::DEFAULT, |m| Key::<Levels>::named(m.name));
            match cx.doc.get_key(&lvl_key) {
                Some(lvls) => {
                    let view = lvls.spans.view();
                    let spans = if i < lvls.spans.len() {
                        cx.project_spans(&lvls.spans, Ix::new(i as u32))
                    } else {
                        Vec::new()
                    };
                    (view, spans)
                }
                None => (ViewId::SOURCE, Vec::new()),
            }
        }));
    }
    if ty == TypeId::of::<Alternatives<Tokens>>() {
        return Some(project_alternatives::<Tokens>(cx, key));
    }
    if ty == TypeId::of::<Alternatives<PosTags>>() {
        return Some(project_alternatives::<PosTags>(cx, key));
    }
    if ty == TypeId::of::<Alternatives<Sentences>>() {
        return Some(project_alternatives::<Sentences>(cx, key));
    }
    if ty == TypeId::of::<Alternatives<Entities>>() {
        return Some(project_alternatives::<Entities>(cx, key));
    }
    if ty == TypeId::of::<NBest<PosTags>>() {
        return Some(project_nbest::<PosTags>(cx, key, token_spans));
    }
    if ty == TypeId::of::<NBest<ChunkTags>>() {
        return Some(project_nbest::<ChunkTags>(cx, key, token_spans));
    }
    if ty == TypeId::of::<NBest<Tokens>>() {
        return Some(project_nbest::<Tokens>(cx, key, token_spans));
    }
    if ty == TypeId::of::<NBest<Sentences>>() {
        return Some(project_nbest::<Sentences>(cx, key, |cx, k, i| {
            aligned_anchor_spans::<Sentences>(cx, k, i, |s| &s.spans)
        }));
    }
    if ty == TypeId::of::<NBest<Entities>>() {
        return Some(project_nbest::<Entities>(cx, key, |cx, k, i| {
            aligned_anchor_spans::<Entities>(cx, k, i, |e| &e.spans)
        }));
    }
    None
}
