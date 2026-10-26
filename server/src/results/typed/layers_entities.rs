//! Entities, relations, coreference, and lexical layer projections
//! (`Entities`, `LinkedEntities`, `Relations`, `Mentions`, `Groups<Mentions>`,
//! `Speakers`, `GlossaryTerms`, `Keywords`, `Stopwords`, `WordTypes`, `Readings`,
//! `Abbreviations`, and `Matches`).

use std::any::TypeId;

use turbo_analysis::{Evidence, Groups, Ix, KeyId, Layer, ViewId};
use turbo_analysis_layers::{AbbreviationForm, Abbreviations, Entities, Matches};
use turbo_coref::{MentionKind, Mentions, Speakers};
use turbo_entity_linker::LinkedEntities;
use turbo_glossary_layers::GlossaryTerms;
use turbo_keywords::Keywords;
use turbo_relation::Relations;
use turbo_stopword_layers::Stopwords;
use turbo_tokenize_layers::{Readings, WordTypes};

use super::layers_core::token_spans;
use super::{ProjectionCx, build_layer, coalesce_wire_spans, find_aligned_to};
use crate::proto::result::typed_annotation::Value;
use crate::proto::result::{
    AbbreviationAnnotation, CorefChainAnnotation, CorefMentionAnnotation, EntityAnnotation, EntityVote,
    GlossaryTermAnnotation, KeywordAnnotation, LinkedEntityAnnotation, MatchAnnotation, ReadingAnnotation,
    RelationAnnotation, SpeakerAnnotation, StopwordFlagAnnotation, TypedAnnotation, TypedLayer, WordTypeAnnotation,
};

fn mention_kind_name(k: MentionKind) -> &'static str {
    match k {
        MentionKind::Proper => "proper",
        MentionKind::Nominal => "nominal",
        MentionKind::Pronominal => "pronominal",
        MentionKind::Demonstrative => "demonstrative",
    }
}

pub(crate) fn project_entity_layer(cx: &mut ProjectionCx<'_>, key: KeyId) -> Option<TypedLayer> {
    let ty = key.ty;
    if ty == TypeId::of::<Entities>() {
        return Some(project_entities(cx, key));
    }
    if ty == TypeId::of::<LinkedEntities>() {
        return Some(build_layer::<LinkedEntities>(cx, key, |_, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| {
                    let row = l.get(Ix::new(i as u32));
                    TypedAnnotation {
                        spans: Vec::new(),
                        probability: Some(row.score),
                        model_ref: mref.clone(),
                        value: Some(Value::LinkedEntity(LinkedEntityAnnotation {
                            entity_index: i as u32,
                            kb_id: row.kb_id.map(ToOwned::to_owned),
                            provider_id: row.provider_id.to_owned(),
                            canonical_name: row.canonical_name.to_owned(),
                            matched_alias: row.matched_alias.to_owned(),
                            score: row.score,
                            nil_reason: row.nil_reason.map(|r| r.as_str().to_owned()),
                            nil_cluster: row.nil_cluster,
                        })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Relations>() {
        return Some(build_layer::<Relations>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: Some(l.score[i]),
                        model_ref: mref.clone(),
                        value: Some(Value::Relation(RelationAnnotation {
                            tagset: l.kinds.name.clone(),
                            kind: l.kinds.label(l.kind[i]).to_owned(),
                            subject_entity: l.subject[i].get() as u32,
                            object_entity: l.object[i].get() as u32,
                        })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Mentions>() {
        return Some(build_layer::<Mentions>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    let r = l.tokens[i];
                    let agr = l.agreement[i];
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: None,
                        model_ref: mref.clone(),
                        value: Some(Value::CorefMention(CorefMentionAnnotation {
                            sentence_index: l.sentence[i].get() as u32,
                            token_start: r.start.get() as u32,
                            token_end: r.end.get() as u32,
                            head_token: l.head[i].get() as u32,
                            entity_index: l.entity_of(ix).map(|e| e.get() as u32),
                            kind: mention_kind_name(l.kind[i]).to_owned(),
                            origin: l.origin[i].as_str().to_owned(),
                            number: agr.number().java_name().to_owned(),
                            gender: agr.gender().java_name().to_owned(),
                            animacy: agr.animacy().java_name().to_owned(),
                            person: agr.person().java_name().to_owned(),
                        })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Groups<Mentions>>() {
        return Some(build_layer::<Groups<Mentions>>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            let mentions = cx.doc.get_key(&l.of_key);
            l.iter()
                .map(|(_, group)| {
                    let mut spans = Vec::new();
                    let mut representative = String::new();
                    if let Some(m) = mentions {
                        for (pos, &mix) in group.iter().enumerate() {
                            if pos == 0 {
                                representative = cx.target_slice(m.spans.view(), m.spans.get(mix)).to_owned();
                            }
                            spans.extend(cx.project_spans(&m.spans, mix));
                        }
                    }
                    TypedAnnotation {
                        spans: coalesce_wire_spans(spans),
                        probability: None,
                        model_ref: mref.clone(),
                        value: Some(Value::CorefChain(CorefChainAnnotation {
                            representative,
                            mention_indices: group.iter().map(|ix| ix.get() as u32).collect(),
                        })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Speakers>() {
        return Some(build_layer::<Speakers>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: None,
                        model_ref: mref.clone(),
                        value: Some(Value::Speaker(SpeakerAnnotation {
                            speaker: l.speakers.label(l.speaker[i]).to_owned(),
                        })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<GlossaryTerms>() {
        return Some(build_layer::<GlossaryTerms>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    let entry = l.entry_of(i);
                    let r = l.tokens[i];
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: None,
                        model_ref: mref.clone(),
                        value: Some(Value::GlossaryTerm(GlossaryTermAnnotation {
                            canonical_id: entry.id.to_owned(),
                            preferred_label: entry.term.to_owned(),
                            entry_index: l.entry[i].0,
                            token_start: r.start.get() as u32,
                            token_end: r.end.get() as u32,
                        })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Keywords>() {
        return Some(build_layer::<Keywords>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    let kw = l.get(i);
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: Some(kw.score),
                        model_ref: mref.clone(),
                        value: Some(Value::Keyword(KeywordAnnotation {
                            phrase: kw.phrase.to_owned(),
                            score: kw.score,
                            algorithm: kw.algorithm.to_owned(),
                            terms: u32::from(kw.terms),
                        })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Stopwords>() {
        return Some(build_layer::<Stopwords>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            let lang = l.language.iso639_1().to_owned();
            (0..l.len())
                .map(|i| TypedAnnotation {
                    spans: token_spans(cx, key, i),
                    probability: None,
                    model_ref: mref.clone(),
                    value: Some(Value::StopwordFlag(StopwordFlagAnnotation {
                        token_index: i as u32,
                        language: lang.clone(),
                        is_stopword: l.stop[i],
                    })),
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<WordTypes>() {
        return Some(build_layer::<WordTypes>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| TypedAnnotation {
                    spans: token_spans(cx, key, i),
                    probability: None,
                    model_ref: mref.clone(),
                    value: Some(Value::WordType(WordTypeAnnotation {
                        token_index: i as u32,
                        kind: l.kind[i].java_name().to_owned(),
                    })),
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Readings>() {
        return Some(build_layer::<Readings>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| TypedAnnotation {
                    spans: token_spans(cx, key, i),
                    probability: None,
                    model_ref: mref.clone(),
                    value: Some(Value::Reading(ReadingAnnotation {
                        token_index: i as u32,
                        reading: l.reading.as_ref().map(|s| s.get(i).to_owned()),
                        pronunciation: l.pronunciation.as_ref().map(|s| s.get(i).to_owned()),
                        lemma: l.lemma.as_ref().map(|s| s.get(i).to_owned()),
                    })),
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Abbreviations>() {
        return Some(build_layer::<Abbreviations>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    let form = match l.form[i] {
                        AbbreviationForm::Short => "short",
                        AbbreviationForm::Long => "long",
                        AbbreviationForm::Mention => "mention",
                    };
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: None,
                        model_ref: mref.clone(),
                        value: Some(Value::Abbreviation(AbbreviationAnnotation { form: form.to_owned() })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Matches>() {
        return Some(build_layer::<Matches>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    let r = l.tokens[i];
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: None,
                        model_ref: mref.clone(),
                        value: Some(Value::RuleMatch(MatchAnnotation {
                            token_start: r.start.get() as u32,
                            token_end: r.end.get() as u32,
                            label: l.label_of(ix).to_owned(),
                            rule_id: l.rule[i],
                        })),
                    }
                })
                .collect()
        }));
    }
    None
}

fn project_entities(cx: &mut ProjectionCx<'_>, key: KeyId) -> TypedLayer {
    let has_model = cx.doc.provenance(key).and_then(|p| p.model.as_ref()).is_some();
    let evidence = find_aligned_to::<Evidence<Entities>>(cx.doc, key);
    let linked = find_aligned_to::<LinkedEntities>(cx.doc, key);

    build_layer::<Entities>(cx, key, |cx, l, mref| {
        let mref = mref.map(ToOwned::to_owned);
        (0..l.len())
            .map(|i| {
                let ix = Ix::new(i as u32);
                let r = l.tokens[i];
                let score = l.score[i];
                let mut probability =
                    if (has_model || score > 0.0) && (0.0..=1.0).contains(&score) { Some(score) } else { None };

                let mut votes = Vec::new();
                if let Some(ev) = evidence
                    && i < ev.records()
                {
                    let prod = ev.producer_of(i);
                    let vote_mref = cx.register_producer(&prod.annotator, prod.model.as_ref());
                    // Evidence is stored in source coordinates. A target view
                    // can delete or collapse its fragments just like the
                    // entity's own spans; nested votes obey the same contract.
                    let vote_spans = coalesce_wire_spans(
                        ev.spans_of(i).iter().map(|&s| cx.project_wire_span(ViewId::SOURCE, s)).collect(),
                    );
                    let raw = ev.raw_of(i).filter(|&v| v.is_finite() && (0.0..=1.0).contains(&v));
                    let cal = ev.calibrated_of(i);
                    if probability.is_none() {
                        probability = cal.or(raw);
                    }
                    votes.push(EntityVote {
                        annotator: prod.annotator.clone(),
                        model_ref: Some(vote_mref),
                        spans: vote_spans,
                        probability: raw,
                        calibrated_probability: cal,
                        abstention: ev.abstention_of(i).map(ToOwned::to_owned),
                    });
                }

                let (canonical, kb_id) = match linked {
                    Some(le) if i < le.len() => {
                        let row = le.get(ix);
                        let can = if row.canonical_name.is_empty() {
                            cx.target_slice(l.spans.view(), l.spans.get(ix)).to_owned()
                        } else {
                            row.canonical_name.to_owned()
                        };
                        (can, row.kb_id.map(ToOwned::to_owned))
                    }
                    _ => (cx.target_slice(l.spans.view(), l.spans.get(ix)).to_owned(), None),
                };

                TypedAnnotation {
                    spans: cx.project_spans(&l.spans, ix),
                    probability,
                    model_ref: mref.clone(),
                    value: Some(Value::Entity(EntityAnnotation {
                        tagset: l.kinds.name.clone(),
                        kind: l.kinds.label(l.kind[i]).to_owned(),
                        token_start: r.start.get() as u32,
                        token_end: r.end.get() as u32,
                        canonical,
                        kb_id,
                        votes,
                    })),
                }
            })
            .collect()
    })
}
