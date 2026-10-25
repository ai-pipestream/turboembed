//! Text quality, privacy, and encoding layer projections
//! (`TextQuality`, `SpanQuality`, privacy, artifacts, and encoding).

use std::any::TypeId;

use turbo_analysis::{Ix, KeyId, Layer};
use turbo_analysis_layers::{
    ColumnAlign, KeyValuePairs, LexicalExpansions, Pii, QualityMetrics, SpanQuality, Tables, TextQuality, WordNetSenses,
};
use turbo_encoding_layers::{EncodingEvidence, SourceEncoding};
use turbo_normalize_layers::{
    Artifacts, AssetEncoding, Assets, Corrections, Emoji, EmojiKind, Noise, PlaceholderKind, Placeholders,
    ProtectedSpans,
};
use turbo_schema_extract::Records;

use super::{ProjectionCx, build_layer};
use crate::proto::result::typed_annotation::Value;
use crate::proto::result::{
    EmbeddedAsset, EmojiAnnotation, KeyValuePairAnnotation, LexicalExpansionAnnotation, NoiseSpan, PiiMention,
    PlaceholderAnnotation, ProtectedSpanAnnotation, QualityMetricsAnnotation, RecordValueAnnotation,
    SourceEncodingAnnotation, SpellingCorrection, TableCellAnnotation, TextArtifact, TypedAnnotation, TypedLayer,
    WordNetSenseAnnotation,
};

fn placeholder_kind_name(k: PlaceholderKind) -> &'static str {
    match k {
        PlaceholderKind::Url => "url",
        PlaceholderKind::Email => "email",
        PlaceholderKind::Hashtag => "hashtag",
        PlaceholderKind::Mention => "mention",
        PlaceholderKind::RetweetMarker => "retweet_marker",
        PlaceholderKind::Emoticon => "emoticon",
        PlaceholderKind::Number => "number",
        PlaceholderKind::Emoji => "emoji",
        PlaceholderKind::Asset => "asset",
    }
}

fn emoji_kind_name(k: EmojiKind) -> &'static str {
    match k {
        EmojiKind::Basic => "basic",
        EmojiKind::Keycap => "keycap",
        EmojiKind::Flag => "flag",
        EmojiKind::Modifier => "modifier",
        EmojiKind::Tag => "tag",
        EmojiKind::Zwj => "zwj",
    }
}

fn asset_encoding_name(e: AssetEncoding) -> &'static str {
    match e {
        AssetEncoding::Base64 => "base64",
        AssetEncoding::Base64Url => "base64url",
        AssetEncoding::Hex => "hex",
    }
}

fn encoding_evidence_name(e: EncodingEvidence) -> &'static str {
    match e {
        EncodingEvidence::Bom => "bom",
        EncodingEvidence::Declared => "declared",
        EncodingEvidence::ValidUtf8 => "valid_utf8",
        EncodingEvidence::Ascii => "ascii",
        EncodingEvidence::Detected => "detected",
    }
}

fn quality_metrics(m: QualityMetrics) -> QualityMetricsAnnotation {
    QualityMetricsAnnotation {
        composite_score: Some(m.composite_score),
        mtld_lexical_diversity: Some(m.mtld_lexical_diversity),
        flesch_kincaid_grade: Some(m.flesch_kincaid_grade),
        automated_readability_index: Some(m.automated_readability_index),
        coleman_liau_index: Some(m.coleman_liau_index),
        information_density: Some(m.information_density),
        alpha_ratio: Some(m.alpha_ratio),
        digit_ratio: Some(m.digit_ratio),
        symbol_ratio: Some(m.symbol_ratio),
        whitespace_ratio: Some(m.whitespace_ratio),
        replacement_ratio: Some(m.replacement_ratio),
        flags: u32::from(m.flags.0),
    }
}

pub(crate) fn project_quality_layer(cx: &mut ProjectionCx<'_>, key: KeyId) -> Option<TypedLayer> {
    let ty = key.ty;
    if ty == TypeId::of::<TextQuality>() {
        return Some(build_layer::<TextQuality>(cx, key, |cx, l, mref| {
            vec![TypedAnnotation {
                spans: cx.project_spans(&l.spans, Ix::new(0)),
                probability: None,
                model_ref: mref.map(ToOwned::to_owned),
                value: Some(Value::TextQuality(quality_metrics(l.metrics))),
            }]
        }));
    }
    if ty == TypeId::of::<SpanQuality>() {
        return Some(build_layer::<SpanQuality>(cx, key, |cx, l, mref| {
            l.metrics
                .iter()
                .enumerate()
                .map(|(i, &metrics)| TypedAnnotation {
                    spans: cx.project_spans(&l.spans, Ix::new(i as u32)),
                    probability: None,
                    model_ref: mref.map(ToOwned::to_owned),
                    value: Some(Value::SpanQuality(quality_metrics(metrics))),
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Pii>() {
        return Some(build_layer::<Pii>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    let m = l.get(ix);
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: None,
                        model_ref: mref.clone(),
                        value: Some(Value::Pii(PiiMention {
                            kind: m.kind.name().to_owned(),
                            confidence_tier: m.confidence.name().to_owned(),
                            normalized: m.normalized.to_owned(),
                        })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Noise>() {
        return Some(build_layer::<Noise>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    let sc = l.score[i];
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: Some(sc),
                        model_ref: mref.clone(),
                        value: Some(Value::Noise(NoiseSpan { severity: l.severity[i].name().to_owned(), score: sc })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Artifacts>() {
        return Some(build_layer::<Artifacts>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: None,
                        model_ref: mref.clone(),
                        value: Some(Value::Artifact(TextArtifact { kind: l.kind[i].name().to_owned() })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Assets>() {
        return Some(build_layer::<Assets>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            let view = l.spans.view();
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: None,
                        model_ref: mref.clone(),
                        value: Some(Value::Asset(EmbeddedAsset {
                            format: l.format_name(i).to_owned(),
                            media_type: l.media_type(i).to_owned(),
                            declared_type: l.declared_type.get(i).to_owned(),
                            encoding: asset_encoding_name(l.encoding[i]).to_owned(),
                            payload_span: Some(cx.project_wire_span(view, l.payload[i])),
                            decoded_bytes: l.decoded_len[i],
                            width: l.width[i],
                            height: l.height[i],
                        })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Corrections>() {
        return Some(build_layer::<Corrections>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: None,
                        model_ref: mref.clone(),
                        value: Some(Value::Correction(SpellingCorrection {
                            suggestion: l.suggestion.get(i).to_owned(),
                            edit_distance: u32::from(l.edit_distance[i]),
                        })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Placeholders>() {
        return Some(build_layer::<Placeholders>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: None,
                        model_ref: mref.clone(),
                        value: Some(Value::Placeholder(PlaceholderAnnotation {
                            kind: placeholder_kind_name(l.kind[i]).to_owned(),
                        })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Emoji>() {
        return Some(build_layer::<Emoji>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: None,
                        model_ref: mref.clone(),
                        value: Some(Value::Emoji(EmojiAnnotation {
                            kind: emoji_kind_name(l.kind[i]).to_owned(),
                            name: l.name.get(i).to_owned(),
                            region: l.region.get(i).to_owned(),
                        })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<ProtectedSpans>() {
        return Some(build_layer::<ProtectedSpans>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: None,
                        model_ref: mref.clone(),
                        value: Some(Value::ProtectedSpan(ProtectedSpanAnnotation { index: i as u32 })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<SourceEncoding>() {
        return Some(build_layer::<SourceEncoding>(cx, key, |_, l, mref| {
            vec![TypedAnnotation {
                spans: Vec::new(),
                probability: Some(f64::from(l.confidence)),
                model_ref: mref.map(ToOwned::to_owned),
                value: Some(Value::SourceEncoding(SourceEncodingAnnotation {
                    encoding: l.encoding.name().to_owned(),
                    confidence: l.confidence,
                    evidence: encoding_evidence_name(l.evidence).to_owned(),
                })),
            }]
        }));
    }
    if ty == TypeId::of::<WordNetSenses>() {
        return Some(build_layer::<WordNetSenses>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: None,
                        model_ref: mref.clone(),
                        value: Some(Value::WordnetSense(WordNetSenseAnnotation {
                            token_index: l.token[i].get() as u32,
                            synset_offset: 0,
                            pos: char::from(l.pos[i]).to_string(),
                            lexname: String::new(),
                            gloss: l.gloss.get(i).to_owned(),
                            lemmas: l.members(i).map(ToOwned::to_owned).collect(),
                            hypernym_offsets: Vec::new(),
                            synset_id: l.synset_id.get(i).to_owned(),
                            sense_id: l.sense_id.get(i).to_owned(),
                            lemma: l.lemma.get(i).to_owned(),
                            sense_rank: l.sense_rank[i],
                            lexicon_id: l.lexicon_id.to_string(),
                        })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<LexicalExpansions>() {
        return Some(build_layer::<LexicalExpansions>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    let w = l.weight[i];
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: Some(w),
                        model_ref: mref.clone(),
                        value: Some(Value::LexicalExpansion(LexicalExpansionAnnotation {
                            token_index: l.token[i].get() as u32,
                            synonyms: Vec::new(),
                            hypernyms: Vec::new(),
                            derivations: Vec::new(),
                            morph_roots: Vec::new(),
                            term: l.term.get(i).to_owned(),
                            relation: u32::from(l.relation[i]),
                            depth: u32::from(l.depth[i]),
                            sense_rank: l.sense_rank[i],
                            weight: w,
                            synset_id: l.synset_id.get(i).to_owned(),
                            lexicon_id: l.lexicon_id.to_string(),
                        })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Tables>() {
        return Some(build_layer::<Tables>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    let c = &l.cells[i];
                    let align = match c.align {
                        ColumnAlign::None => "none",
                        ColumnAlign::Left => "left",
                        ColumnAlign::Center => "center",
                        ColumnAlign::Right => "right",
                    };
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: None,
                        model_ref: mref.clone(),
                        value: Some(Value::TableCell(TableCellAnnotation {
                            table_id: c.table_id,
                            row_index: u32::from(c.row_ix),
                            col_index: u32::from(c.col_ix),
                            row_span: u32::from(c.row_span),
                            col_span: u32::from(c.col_span),
                            is_header: c.is_header,
                            align: align.to_owned(),
                        })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<KeyValuePairs>() {
        return Some(build_layer::<KeyValuePairs>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            let view = l.spans.view();
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    let p = l.pairs[i];
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: None,
                        model_ref: mref.clone(),
                        value: Some(Value::KeyValuePair(KeyValuePairAnnotation {
                            key_span: Some(cx.project_wire_span(view, p.key)),
                            separator_span: Some(cx.project_wire_span(view, p.separator)),
                            value_span: Some(cx.project_wire_span(view, p.value)),
                            key_text: cx.target_slice(view, p.key).to_owned(),
                            value_text: cx.target_slice(view, p.value).to_owned(),
                        })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Records>() {
        return Some(build_layer::<Records>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    let r_ix = l.record[i] as usize;
                    let sc = l.score[i];
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: Some(sc),
                        model_ref: mref.clone(),
                        value: Some(Value::RecordValue(RecordValueAnnotation {
                            record_index: l.record[i],
                            record_type: l.types.label(l.record_type[r_ix]).to_owned(),
                            instance: l.instance[r_ix],
                            path: l.paths.label(l.path[i]).to_owned(),
                            score: sc,
                            logit: l.logit[i],
                        })),
                    }
                })
                .collect()
        }));
    }
    None
}
