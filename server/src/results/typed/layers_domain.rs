//! Numeric, temporal, and geographic layer projections
//! (`Numbers`, `Ordinals`, `Money`, `Quantities`, `Temporals`, `Times`,
//! `DocumentDate`, `Locations`, `Containment`, and `Regions`).

use std::any::TypeId;

use turbo_analysis::{Ix, KeyId, Layer};
use turbo_analysis_layers::{
    Containment, DocumentDate, Money, Multiplier, NumberNotation, Numbers, Ordinals, Quantities, Regions, ScaleMarker,
    Temporals, TimeGrain, Times,
};
use turbo_geo::Locations;

use super::{ProjectionCx, build_layer};
use crate::proto::result::typed_annotation::Value;
use crate::proto::result::{
    ContainmentChain, DocumentDateAnnotation, MoneyAnnotation, NumberAnnotation, OrdinalAnnotation, PlaceAncestorProto,
    PlaceResolution, QuantityAnnotation, RegionVote, TemporalAnnotation, TimeAnnotation, TypedAnnotation, TypedLayer,
};

fn multiplier_name(m: Multiplier) -> &'static str {
    match m {
        Multiplier::One => "one",
        Multiplier::Thousand => "thousand",
        Multiplier::Million => "million",
        Multiplier::Billion => "billion",
        Multiplier::Trillion => "trillion",
    }
}

fn scale_marker_name(m: ScaleMarker) -> &'static str {
    match m {
        ScaleMarker::None => "none",
        ScaleMarker::Suffix => "suffix",
        ScaleMarker::Word => "word",
    }
}

fn notation_name(n: NumberNotation) -> &'static str {
    match n {
        NumberNotation::LatinUs => "latin-us",
        NumberNotation::LatinEu => "latin-eu",
    }
}

fn time_grain_name(g: TimeGrain) -> &'static str {
    match g {
        TimeGrain::Second => "second",
        TimeGrain::Minute => "minute",
        TimeGrain::Hour => "hour",
        TimeGrain::PartOfDay => "part_of_day",
        TimeGrain::Day => "day",
        TimeGrain::Week => "week",
        TimeGrain::Weekend => "weekend",
        TimeGrain::Month => "month",
        TimeGrain::Quarter => "quarter",
        TimeGrain::Half => "half",
        TimeGrain::Season => "season",
        TimeGrain::Year => "year",
        TimeGrain::Decade => "decade",
        TimeGrain::Century => "century",
    }
}

pub(crate) fn project_domain_layer(cx: &mut ProjectionCx<'_>, key: KeyId) -> Option<TypedLayer> {
    let ty = key.ty;
    if ty == TypeId::of::<Numbers>() {
        return Some(build_layer::<Numbers>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            let view = l.spans.view();
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: None,
                        model_ref: mref.clone(),
                        value: Some(Value::Number(NumberAnnotation {
                            numeral_span: Some(cx.project_wire_span(view, l.numeral(ix))),
                            negative: l.negative[i],
                            coefficient: l.coefficient.get(i).to_owned(),
                            scale: l.scale[i],
                            multiplier: multiplier_name(l.multiplier[i]).to_owned(),
                            marker: scale_marker_name(l.marker[i]).to_owned(),
                            notation: notation_name(l.notation[i]).to_owned(),
                            plain_value: l.value(ix).to_plain_string(),
                        })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Ordinals>() {
        return Some(build_layer::<Ordinals>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: None,
                        model_ref: mref.clone(),
                        value: Some(Value::Ordinal(OrdinalAnnotation {
                            base: l.base[i].name().to_owned(),
                            offset: l.offset[i],
                        })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Money>() {
        return Some(build_layer::<Money>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: None,
                        model_ref: mref.clone(),
                        value: Some(Value::Money(MoneyAnnotation {
                            currency: l.currency[i].as_str().to_owned(),
                            negative: l.negative[i],
                            coefficient: l.coefficient.get(i).to_owned(),
                            scale: l.scale[i],
                            plain_amount: l.amount(ix).to_plain_string(),
                        })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Quantities>() {
        return Some(build_layer::<Quantities>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: None,
                        model_ref: mref.clone(),
                        value: Some(Value::Quantity(QuantityAnnotation {
                            unit: l.unit(ix).to_owned(),
                            negative: l.negative[i],
                            coefficient: l.coefficient.get(i).to_owned(),
                            scale: l.scale[i],
                            plain_value: l.value(ix).to_plain_string(),
                            is_percent: l.is_percent(ix),
                        })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Temporals>() {
        return Some(build_layer::<Temporals>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    let val = l.value(ix);
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: None,
                        model_ref: mref.clone(),
                        value: Some(Value::Temporal(TemporalAnnotation {
                            granularity: l.granularity[i].name().to_owned(),
                            origin: l.origin[i].name().to_owned(),
                            iso: val.to_string(),
                            year: l.year[i],
                            period: u32::from(l.period[i]),
                            day: u32::from(l.day[i]),
                        })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Times>() {
        return Some(build_layer::<Times>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    let m = l.get(ix);
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: None,
                        model_ref: mref.clone(),
                        value: Some(Value::Time(TimeAnnotation {
                            kind: m.kind.name().to_owned(),
                            grain: time_grain_name(m.grain).to_owned(),
                            modifier: m.modifier.to_string(),
                            start: m.start,
                            end: m.end,
                            duration_ms: m.duration_ms,
                            ambiguous: m.ambiguous,
                            timex: m.timex.to_owned(),
                        })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<DocumentDate>() {
        return Some(build_layer::<DocumentDate>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            let Some(date) = l.date else { return Vec::new() };
            let spans = l
                .mention
                .and_then(|mix| {
                    cx.doc.get::<Temporals>().filter(|t| mix.get() < t.len()).map(|t| cx.project_spans(&t.spans, mix))
                })
                .unwrap_or_default();
            vec![TypedAnnotation {
                spans,
                probability: None,
                model_ref: mref,
                value: Some(Value::DocumentDate(DocumentDateAnnotation {
                    year: date.year(),
                    month: u32::from(date.month()),
                    day: u32::from(date.day()),
                    iso: date.to_string(),
                })),
            }]
        }));
    }
    if ty == TypeId::of::<Locations>() {
        return Some(build_layer::<Locations>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    let (_, entry, conf) = l.resolution(ix);
                    let pt = entry.location();
                    let chain = entry.containment();
                    let admin1_code = chain.first().cloned().unwrap_or_default();
                    let admin2_code = chain.get(1).cloned().unwrap_or_default();
                    let timezone = entry.attribute("timezone").map(|a| a.value().to_owned()).unwrap_or_default();
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: Some(conf),
                        model_ref: mref.clone(),
                        value: Some(Value::Place(PlaceResolution {
                            source: entry.source().to_owned(),
                            record_id: entry.record_id().to_owned(),
                            canonical_name: entry.name().to_owned(),
                            feature_class: entry.feature_class().unwrap_or_default().to_owned(),
                            country_code: entry.country_code().unwrap_or_default().to_owned(),
                            admin1_code,
                            admin2_code,
                            latitude: pt.latitude(),
                            longitude: pt.longitude(),
                            population: entry.population().max(0) as u64,
                            timezone,
                        })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Containment>() {
        return Some(build_layer::<Containment>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    let ancestors = l
                        .ancestors(ix)
                        .map(|a| PlaceAncestorProto {
                            id: a.id.to_owned(),
                            name: a.name.to_owned(),
                            kind: a.kind.to_owned(),
                        })
                        .collect();
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: None,
                        model_ref: mref.clone(),
                        value: Some(Value::Containment(ContainmentChain { ancestors })),
                    }
                })
                .collect()
        }));
    }
    if ty == TypeId::of::<Regions>() {
        return Some(build_layer::<Regions>(cx, key, |_, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            l.iter()
                .map(|v| TypedAnnotation {
                    spans: Vec::new(),
                    probability: Some(v.share),
                    model_ref: mref.clone(),
                    value: Some(Value::RegionVote(RegionVote {
                        country_code: v.country_code().to_owned(),
                        share: v.share,
                    })),
                })
                .collect()
        }));
    }
    None
}
