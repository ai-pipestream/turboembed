//! Text quality, privacy, and encoding layer projections
//! (`Pii`, `Noise`, `Artifacts`, `Assets`, `Corrections`, `Placeholders`,
//! `Emoji`, `ProtectedSpans`, and `SourceEncoding`).

use std::any::TypeId;

use turbo_analysis::{Ix, KeyId, Layer};
use turbo_analysis_layers::Pii;
use turbo_encoding_layers::{EncodingEvidence, SourceEncoding};
use turbo_normalize_layers::{
    Artifacts, AssetEncoding, Assets, Corrections, Emoji, EmojiKind, Noise, PlaceholderKind, Placeholders,
    ProtectedSpans,
};

use super::{ProjectionCx, build_layer};
use crate::proto::result::typed_annotation::Value;
use crate::proto::result::{
    EmbeddedAsset, EmojiAnnotation, NoiseSpan, PiiMention, PlaceholderAnnotation, ProtectedSpanAnnotation,
    SourceEncodingAnnotation, SpellingCorrection, TextArtifact, TypedAnnotation, TypedLayer,
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

pub(crate) fn project_quality_layer(cx: &mut ProjectionCx<'_>, key: KeyId) -> Option<TypedLayer> {
    let ty = key.ty;
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
    None
}
