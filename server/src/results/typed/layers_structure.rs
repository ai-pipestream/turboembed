//! Structure, chunks, levels, boundaries, vectors, calibration, boilerplate,
//! confidence, QA, and graph (`Edges`/`Groups`) layer projections.

use std::any::TypeId;

use turbo_analysis::{Edges, Groups, Ix, Key, KeyId, Layer, Spans, View};
use turbo_analysis_embed::Answers;
use turbo_analysis_layers::{
    Abbreviations, Block, Boilerplate, BoilerplateKind, Boundaries, Calibrated, Categories, Chunks, Confidence, Cut,
    Entities, Levels, Sentences, Structure, Subwords, TokenVectorWindows, Tokens, Unit, Utf16Spans, Vectors,
};
use turbo_coref::Mentions;

use super::{ProjectionCx, build_layer, coalesce_wire_spans};
use crate::proto::result::typed_annotation::Value;
use crate::proto::result::{
    AnswerAnnotation, BoilerplateAnnotation, BoundaryAnnotation, CalibratedAnnotation, ChunkAnnotation,
    ConfidenceAnnotation, EdgeAnnotation, GroupAnnotation, LevelAnnotation, SpanFragment, StructureAnnotation,
    TokenVectorWindowAnnotation, TypedAnnotation, TypedLayer, Utf16SpanAnnotation, VectorAnnotation,
};

fn block_name(b: Block) -> &'static str {
    match b {
        Block::Section => "section",
        Block::Heading => "heading",
        Block::Paragraph => "paragraph",
        Block::List => "list",
        Block::ListItem => "list_item",
        Block::Table => "table",
        Block::TableRow => "table_row",
        Block::TableCell => "table_cell",
        Block::Code => "code",
        Block::Quote => "quote",
        Block::Caption => "caption",
        Block::Footnote => "footnote",
        Block::PageHeader => "page_header",
        Block::PageFooter => "page_footer",
        Block::LooseList => "loose_list",
        Block::Html => "html",
    }
}

fn cut_name(c: Cut) -> String {
    match c {
        Cut::DocumentStart => "document_start".to_owned(),
        Cut::Sentence => "sentence".to_owned(),
        Cut::Section => "section".to_owned(),
        Cut::Heading => "heading".to_owned(),
        Cut::Paragraph => "paragraph".to_owned(),
        Cut::TopicShift => "topic_shift".to_owned(),
        Cut::Clause => "clause".to_owned(),
        Cut::Item => "item".to_owned(),
        Cut::Inside(b) => format!("inside:{}", block_name(b)),
        Cut::Overlap(b) => format!("overlap:{}", block_name(b)),
    }
}

fn boilerplate_kind_name(k: BoilerplateKind) -> &'static str {
    match k {
        BoilerplateKind::Header => "header",
        BoilerplateKind::Footer => "footer",
        BoilerplateKind::Navigation => "navigation",
        BoilerplateKind::Disclaimer => "disclaimer",
        BoilerplateKind::Template => "template",
    }
}

fn aligned_spans<L: Layer>(
    cx: &ProjectionCx<'_>,
    key: KeyId,
    index: usize,
    spans_of: impl FnOnce(&L) -> &Spans<L>,
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

fn project_vectors<L: Layer>(
    cx: &mut ProjectionCx<'_>,
    key: KeyId,
    item_spans: impl Fn(&ProjectionCx<'_>, KeyId, usize) -> Vec<SpanFragment>,
) -> TypedLayer {
    build_layer::<Vectors<L>>(cx, key, |cx, l, mref| {
        let mref = mref.map(ToOwned::to_owned);
        (0..l.len())
            .map(|i| {
                let ix = Ix::new(i as u32);
                TypedAnnotation {
                    spans: item_spans(cx, key, i),
                    probability: None,
                    model_ref: mref.clone(),
                    value: Some(Value::Vector(VectorAnnotation {
                        target_index: i as u32,
                        dim: l.dim,
                        empty: l.is_empty(ix),
                        values: l.row(ix).to_vec(),
                    })),
                }
            })
            .collect()
    })
}

fn project_boilerplate<L: Layer>(cx: &mut ProjectionCx<'_>, key: KeyId, spans_of: fn(&L) -> &Spans<L>) -> TypedLayer {
    build_layer::<Boilerplate<L>>(cx, key, |cx, l, mref| {
        let mref = mref.map(ToOwned::to_owned);
        (0..l.len())
            .map(|i| {
                let sc = l.score[i];
                TypedAnnotation {
                    spans: aligned_spans::<L>(cx, key, i, spans_of),
                    probability: Some(f64::from(sc)),
                    model_ref: mref.clone(),
                    value: Some(Value::Boilerplate(BoilerplateAnnotation {
                        target_index: i as u32,
                        kind: l.kind[i].map(|k| boilerplate_kind_name(k).to_owned()),
                        score: sc,
                    })),
                }
            })
            .collect()
    })
}

fn project_confidence<L: Layer>(cx: &mut ProjectionCx<'_>, key: KeyId, spans_of: fn(&L) -> &Spans<L>) -> TypedLayer {
    build_layer::<Confidence<L>>(cx, key, |cx, l, mref| {
        let mref = mref.map(ToOwned::to_owned);
        (0..l.len())
            .map(|i| {
                let conf = l.value[i];
                TypedAnnotation {
                    spans: aligned_spans::<L>(cx, key, i, spans_of),
                    probability: Some(conf),
                    model_ref: mref.clone(),
                    value: Some(Value::Confidence(ConfidenceAnnotation { target_index: i as u32, confidence: conf })),
                }
            })
            .collect()
    })
}

fn project_utf16<L: Layer>(cx: &mut ProjectionCx<'_>, key: KeyId, spans_of: fn(&L) -> &Spans<L>) -> TypedLayer {
    build_layer::<Utf16Spans<L>>(cx, key, |cx, l, mref| {
        let mref = mref.map(ToOwned::to_owned);
        (0..l.len())
            .map(|i| TypedAnnotation {
                spans: aligned_spans::<L>(cx, key, i, spans_of),
                probability: None,
                model_ref: mref.clone(),
                value: Some(Value::Utf16Span(Utf16SpanAnnotation {
                    target_index: i as u32,
                    start16: l.start16[i],
                    end16: l.end16[i],
                    cut: l.cut[i],
                })),
            })
            .collect()
    })
}

fn project_calibrated<L: Layer>(
    cx: &mut ProjectionCx<'_>,
    key: KeyId,
    item_spans: impl Fn(&ProjectionCx<'_>, KeyId, usize) -> Vec<SpanFragment>,
) -> TypedLayer {
    build_layer::<Calibrated<L>>(cx, key, |cx, l, mref| {
        let mref = mref.map(ToOwned::to_owned);
        (0..l.len())
            .map(|i| {
                let row = l.row(i);
                let best = row
                    .calibrated
                    .iter()
                    .copied()
                    .fold(None::<f64>, |acc, v| Some(acc.map_or(v, |a| a.max(v))))
                    .filter(|v| v.is_finite() && (0.0..=1.0).contains(v));
                TypedAnnotation {
                    spans: item_spans(cx, key, i),
                    probability: best,
                    model_ref: mref.clone(),
                    value: Some(Value::Calibrated(CalibratedAnnotation {
                        target_index: i as u32,
                        raw: row.raw.to_vec(),
                        calibrated: row.calibrated.to_vec(),
                        prediction_set: row.set.iter().map(|&idx| idx as u32).collect(),
                        abstention: row.abstention.map(|a| format!("{a:?}")),
                    })),
                }
            })
            .collect()
    })
}

fn project_edges<From: Layer, To: Layer>(cx: &mut ProjectionCx<'_>, key: KeyId) -> TypedLayer {
    build_layer::<Edges<From, To>>(cx, key, |_, l, mref| {
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
    })
}

fn project_groups<L: Layer>(cx: &mut ProjectionCx<'_>, key: KeyId, spans_of: fn(&L) -> &Spans<L>) -> TypedLayer {
    build_layer::<Groups<L>>(cx, key, |cx, l, mref| {
        let mref = mref.map(ToOwned::to_owned);
        let target = cx.doc.get_key(&l.of_key);
        l.iter()
            .map(|(_, group)| {
                let mut spans = Vec::new();
                if let Some(t) = target {
                    let s = spans_of(t);
                    for &mix in group {
                        if mix.get() < s.len() {
                            spans.extend(cx.project_spans(s, mix));
                        }
                    }
                }
                TypedAnnotation {
                    spans: coalesce_wire_spans(spans),
                    probability: None,
                    model_ref: mref.clone(),
                    value: Some(Value::Group(GroupAnnotation {
                        members: group.iter().map(|ix| ix.get() as u32).collect(),
                    })),
                }
            })
            .collect()
    })
}

pub(crate) fn project_structure_layer(cx: &mut ProjectionCx<'_>, key: KeyId) -> Option<TypedLayer> {
    let ty = key.ty;
    if ty == TypeId::of::<Structure>() {
        return Some(build_layer::<Structure>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: None,
                        model_ref: mref.clone(),
                        value: Some(Value::Structure(StructureAnnotation {
                            kind: block_name(l.kind[i]).to_owned(),
                            parent_index: l.parent[i].map(|p| p.get() as u32),
                            depth: u32::from(l.depth[i]),
                        })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Chunks>() {
        return Some(build_layer::<Chunks>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    let r = l.sentences[i];
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: None,
                        model_ref: mref.clone(),
                        value: Some(Value::Chunk(ChunkAnnotation {
                            level: u32::from(l.level[i]),
                            parent_index: l.parent[i].map(|p| p.get() as u32),
                            sentence_start: r.start.get() as u32,
                            sentence_end: r.end.get() as u32,
                            cut_reason: cut_name(l.cut[i]),
                            cost: l.cost[i],
                            model_tokens: l.model_tokens[i],
                            over_budget: l.over_budget[i],
                        })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Levels>() {
        return Some(build_layer::<Levels>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    let (kind, unit_index) = match l.unit[i] {
                        Unit::Document => ("document", 0),
                        Unit::Section(s) => ("section", s.get() as u32),
                        Unit::Chunk(c) => ("chunk", c.get() as u32),
                    };
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: None,
                        model_ref: mref.clone(),
                        value: Some(Value::LevelUnit(LevelAnnotation {
                            kind: kind.to_owned(),
                            unit_index,
                            parent_index: l.parent[i].map(|p| p.get() as u32),
                        })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Boundaries>() {
        return Some(build_layer::<Boundaries>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| TypedAnnotation {
                    spans: aligned_spans::<Sentences>(cx, key, i, |s| &s.spans),
                    probability: None,
                    model_ref: mref.clone(),
                    value: Some(Value::Boundary(BoundaryAnnotation {
                        sentence_index: i as u32,
                        cost: l.cost[i],
                        reason: cut_name(l.reason[i]),
                    })),
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Vectors<View>>() {
        return Some(project_vectors::<View>(cx, key, |_, _, _| Vec::new()));
    }
    if ty == TypeId::of::<Vectors<Sentences>>() {
        return Some(project_vectors::<Sentences>(cx, key, |cx, k, i| {
            aligned_spans::<Sentences>(cx, k, i, |s| &s.spans)
        }));
    }
    if ty == TypeId::of::<Vectors<Tokens>>() {
        return Some(project_vectors::<Tokens>(cx, key, |cx, k, i| aligned_spans::<Tokens>(cx, k, i, |s| &s.spans)));
    }
    if ty == TypeId::of::<Vectors<Subwords>>() {
        return Some(project_vectors::<Subwords>(cx, key, |cx, k, i| {
            aligned_spans::<Subwords>(cx, k, i, |s| &s.spans)
        }));
    }
    if ty == TypeId::of::<Vectors<Chunks>>() {
        return Some(project_vectors::<Chunks>(cx, key, |cx, k, i| aligned_spans::<Chunks>(cx, k, i, |s| &s.spans)));
    }
    if ty == TypeId::of::<Vectors<Structure>>() {
        return Some(project_vectors::<Structure>(cx, key, |cx, k, i| {
            aligned_spans::<Structure>(cx, k, i, |s| &s.spans)
        }));
    }
    if ty == TypeId::of::<Vectors<Levels>>() {
        return Some(project_vectors::<Levels>(cx, key, |cx, k, i| aligned_spans::<Levels>(cx, k, i, |s| &s.spans)));
    }
    if ty == TypeId::of::<Vectors<Entities>>() {
        return Some(project_vectors::<Entities>(cx, key, |cx, k, i| {
            aligned_spans::<Entities>(cx, k, i, |e| &e.spans)
        }));
    }
    if ty == TypeId::of::<TokenVectorWindows>() {
        return Some(build_layer::<TokenVectorWindows>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: None,
                        model_ref: mref.clone(),
                        value: Some(Value::TokenVectorWindow(TokenVectorWindowAnnotation { window_index: i as u32 })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Utf16Spans<Tokens>>() {
        return Some(project_utf16::<Tokens>(cx, key, |t| &t.spans));
    }
    if ty == TypeId::of::<Utf16Spans<Sentences>>() {
        return Some(project_utf16::<Sentences>(cx, key, |s| &s.spans));
    }
    if ty == TypeId::of::<Confidence<Tokens>>() {
        return Some(project_confidence::<Tokens>(cx, key, |t| &t.spans));
    }
    if ty == TypeId::of::<Confidence<Sentences>>() {
        return Some(project_confidence::<Sentences>(cx, key, |s| &s.spans));
    }
    if ty == TypeId::of::<Confidence<Entities>>() {
        return Some(project_confidence::<Entities>(cx, key, |e| &e.spans));
    }
    if ty == TypeId::of::<Boilerplate<Structure>>() {
        return Some(project_boilerplate::<Structure>(cx, key, |s| &s.spans));
    }
    if ty == TypeId::of::<Boilerplate<Sentences>>() {
        return Some(project_boilerplate::<Sentences>(cx, key, |s| &s.spans));
    }
    if ty == TypeId::of::<Boilerplate<Chunks>>() {
        return Some(project_boilerplate::<Chunks>(cx, key, |c| &c.spans));
    }
    if ty == TypeId::of::<Calibrated<Categories>>() {
        return Some(project_calibrated::<Categories>(cx, key, |_, _, _| Vec::new()));
    }
    if ty == TypeId::of::<Calibrated<Tokens>>() {
        return Some(project_calibrated::<Tokens>(cx, key, |cx, k, i| aligned_spans::<Tokens>(cx, k, i, |t| &t.spans)));
    }
    if ty == TypeId::of::<Calibrated<Entities>>() {
        return Some(project_calibrated::<Entities>(cx, key, |cx, k, i| {
            aligned_spans::<Entities>(cx, k, i, |e| &e.spans)
        }));
    }
    if ty == TypeId::of::<Answers>() {
        return Some(build_layer::<Answers>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            let view = l.spans.view();
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    let prob = l.probabilities[i];
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: Some(prob),
                        model_ref: mref.clone(),
                        value: Some(Value::Answer(AnswerAnnotation {
                            score: l.scores[i],
                            probability: prob,
                            null_score: l.null_score,
                            null_probability: l.null_probability,
                            text: cx.target_slice(view, l.spans.get(ix)).to_owned(),
                        })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Edges<Abbreviations, Abbreviations>>() {
        return Some(project_edges::<Abbreviations, Abbreviations>(cx, key));
    }
    if ty == TypeId::of::<Edges<Structure, Structure>>() {
        return Some(project_edges::<Structure, Structure>(cx, key));
    }
    if ty == TypeId::of::<Edges<Sentences, Sentences>>() {
        return Some(project_edges::<Sentences, Sentences>(cx, key));
    }
    if ty == TypeId::of::<Edges<Chunks, Chunks>>() {
        return Some(project_edges::<Chunks, Chunks>(cx, key));
    }
    if ty == TypeId::of::<Edges<Mentions, Mentions>>() {
        return Some(project_edges::<Mentions, Mentions>(cx, key));
    }
    if ty == TypeId::of::<Edges<Entities, Entities>>() {
        return Some(project_edges::<Entities, Entities>(cx, key));
    }
    if ty == TypeId::of::<Groups<Structure>>() {
        return Some(project_groups::<Structure>(cx, key, |s| &s.spans));
    }
    if ty == TypeId::of::<Groups<Sentences>>() {
        return Some(project_groups::<Sentences>(cx, key, |s| &s.spans));
    }
    if ty == TypeId::of::<Groups<Chunks>>() {
        return Some(project_groups::<Chunks>(cx, key, |c| &c.spans));
    }
    if ty == TypeId::of::<Groups<Abbreviations>>() {
        return Some(project_groups::<Abbreviations>(cx, key, |a| &a.spans));
    }
    if ty == TypeId::of::<Groups<Entities>>() {
        return Some(project_groups::<Entities>(cx, key, |e| &e.spans));
    }
    None
}
