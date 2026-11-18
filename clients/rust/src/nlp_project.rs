//! Projection helpers mapping `Document` NLP layers back onto `ViewId::SOURCE` (`#2338`, `#2345`).

use turbo_analysis::{Document, Ix, Key, Layer, OffsetTable, Span, Tag, View, ViewId};
use turbo_analysis_layers::{
    CONVERTED_MONEY, Categories, Containment, DAY_SECONDS, Date, DocumentDate, Entities, EpistemicStatus, Modality,
    ModalityPolarity, Money, Quantities, TemporalValue, Temporals, TextQuality,
};
use turbo_entity_linker::LinkedEntities;
use turbo_geo::Locations;
use turbo_group::{Scratch as SketchScratch, Sketches};
use turbo_normalize_layers::{Artifacts, Noise};

use crate::error::Result;
use crate::lexical::ProjectedSpanSet;
use crate::nlp::{
    DocumentNlpProjection, DocumentSketch, EntityPolarityHit, GeoHit, MoneyHit, QualitySummary, QuantityHit,
    SentimentSummary, TemporalHit, UtcDateSummary,
};

fn date_epoch_seconds(d: Date) -> i64 {
    d.epoch_day().saturating_mul(DAY_SECONDS)
}

fn temporal_bounds(val: TemporalValue) -> (i64, i64) {
    match val {
        TemporalValue::Day(d) => (date_epoch_seconds(d), d.plus_days(1).map_or(i64::MAX, date_epoch_seconds)),
        TemporalValue::Week { year, week } => {
            let Some(jan4) = Date::new(year, 1, 4) else { return (0, 0) };
            let wd = (jan4.epoch_day() + 3).rem_euclid(7);
            let Some(mon) = jan4.plus_days(-wd).and_then(|w1| w1.plus_weeks(i64::from(week.saturating_sub(1)))) else {
                return (0, 0);
            };
            (date_epoch_seconds(mon), mon.plus_days(7).map_or(i64::MAX, date_epoch_seconds))
        }
        TemporalValue::Month { year, month } => {
            let Some(f) = Date::new(year, month, 1) else { return (0, 0) };
            (date_epoch_seconds(f), f.plus_months(1).map_or(i64::MAX, date_epoch_seconds))
        }
        TemporalValue::Quarter { year, quarter } => {
            let m = quarter.saturating_sub(1).saturating_mul(3).saturating_add(1);
            let Some(f) = Date::new(year, m, 1) else { return (0, 0) };
            (date_epoch_seconds(f), f.plus_months(3).map_or(i64::MAX, date_epoch_seconds))
        }
        TemporalValue::Year(year) => {
            let Some(f) = Date::new(year, 1, 1) else { return (0, 0) };
            (date_epoch_seconds(f), f.plus_years(1).map_or(i64::MAX, date_epoch_seconds))
        }
    }
}

fn union_noise_chars(text: &str, spans: &[Span]) -> i64 {
    let mut iv: Vec<(u32, u32)> = spans.iter().filter(|s| s.start < s.end).map(|s| (s.start, s.end)).collect();
    if iv.is_empty() {
        return 0;
    }
    iv.sort_unstable();
    let (mut cs, mut ce, mut total) = (iv[0].0, iv[0].1, 0i64);
    for &(s, e) in &iv[1..] {
        if s <= ce {
            ce = ce.max(e);
        } else {
            total += text.get(cs as usize..ce as usize).map_or(0, |sl| sl.chars().count() as i64);
            cs = s;
            ce = e;
        }
    }
    total + text.get(cs as usize..ce as usize).map_or(0, |sl| sl.chars().count() as i64)
}

pub(crate) fn project_document(source: &str, doc: &Document, shingle_width: u8) -> Result<DocumentNlpProjection> {
    let table = OffsetTable::build(source);
    let map_span = |view: ViewId, sp: Span| ProjectedSpanSet::from_source_span(source, &table, doc.to_source(view, sp));

    let mut money = Vec::new();
    for (layer_opt, ecb_flag) in [(doc.get_key(&Key::<Money>::DEFAULT), false), (doc.get_key(&CONVERTED_MONEY), true)] {
        if let Some(m) = layer_opt {
            let v = m.spans.view();
            for i in 0..m.len() {
                let ix = Ix::new(i as u32);
                let plain = m.amount(ix).to_plain_string();
                money.push(MoneyHit {
                    spans: map_span(v, m.spans.get(ix)),
                    currency: m.currency[i].as_str().to_owned(),
                    amount_f64: plain.parse::<f64>().unwrap_or(0.0),
                    plain_amount: plain,
                    per: m.per(ix).unwrap_or_default().to_owned(),
                    ecb_converted: ecb_flag,
                });
            }
        }
    }

    let mut quantities = Vec::new();
    if let Some(q) = doc.get::<Quantities>() {
        let v = q.spans.view();
        for i in 0..q.len() {
            let ix = Ix::new(i as u32);
            let base = q.base_unit.get(i);
            let (u, dec) = if base.is_empty() { (q.unit(ix), q.value(ix)) } else { (base, q.normalized_value(ix)) };
            let plain = dec.to_plain_string();
            quantities.push(QuantityHit {
                spans: map_span(v, q.spans.get(ix)),
                unit: u.to_owned(),
                value_f64: plain.parse::<f64>().unwrap_or(0.0),
                plain_value: plain,
                is_percent: q.is_percent(ix),
            });
        }
    }

    let document_date = doc.get::<DocumentDate>().and_then(|dd| dd.date).map(|d| UtcDateSummary {
        iso: d.to_string(),
        epoch_seconds: date_epoch_seconds(d),
        year: d.year(),
        month: u32::from(d.month()),
        day: u32::from(d.day()),
    });

    let mut temporals = Vec::new();
    if let Some(t) = doc.get::<Temporals>() {
        let v = t.spans.view();
        for i in 0..t.len() {
            let ix = Ix::new(i as u32);
            let val = t.value(ix);
            let (s, e) = temporal_bounds(val);
            temporals.push(TemporalHit {
                spans: map_span(v, t.spans.get(ix)),
                granularity: t.granularity[i].name().to_owned(),
                origin: t.origin[i].name().to_owned(),
                iso: val.to_string(),
                start_epoch_seconds: s,
                end_epoch_seconds: e,
            });
        }
    }

    let mut geo = Vec::new();
    if let Some(locs) = doc.get::<Locations>() {
        let cont = doc.get::<Containment>();
        let v = locs.spans.view();
        for i in 0..locs.len() {
            let ix = Ix::new(i as u32);
            let (_, entry, conf) = locs.resolution(ix);
            let spine_path = if let Some(cl) = cont.filter(|c| i < cl_len(c)) {
                let anc: Vec<&str> = cl.ancestors(Ix::new(i as u32)).map(|a| a.name).collect();
                let mut parts: Vec<&str> = anc.into_iter().rev().collect();
                parts.push(entry.name());
                parts.join("/")
            } else {
                let mut parts: Vec<&str> = entry.country_code().into_iter().collect();
                parts.extend(entry.containment().iter().map(String::as_str));
                parts.push(entry.name());
                parts.join("/")
            };
            geo.push(GeoHit {
                spans: map_span(v, locs.spans.get(ix)),
                record_id: entry.record_id().to_owned(),
                canonical_name: entry.name().to_owned(),
                country_code: entry.country_code().unwrap_or_default().to_owned(),
                latitude: entry.location().latitude(),
                longitude: entry.location().longitude(),
                spine_path,
                confidence: conf,
            });
        }
    }

    let sentiment = extract_sentiment(doc);
    let quality = extract_quality(doc);
    let (affirmed_entities, negated_entities) = extract_entities(source, &table, doc);

    let mut scratch = SketchScratch::default();
    let mut sk = Sketches::<View>::empty(shingle_width);
    sk.push([source], &mut scratch);
    let sketch = DocumentSketch {
        word_count: sk.words.first().copied().unwrap_or(0),
        exact: sk.exact.first().copied().unwrap_or(0),
        shape: sk.shape.first().copied().unwrap_or(0),
        simhash: sk.simhash.first().copied().unwrap_or(0),
        minhash: if sk.words.is_empty() { Vec::new() } else { sk.row(Ix::new(0)).to_vec() },
    };

    Ok(DocumentNlpProjection {
        money,
        quantities,
        document_date,
        temporals,
        geo,
        sentiment,
        quality,
        affirmed_entities,
        negated_entities,
        sketch,
    })
}

fn cl_len(c: &Containment) -> usize {
    c.len()
}

fn epistemic_str(e: EpistemicStatus) -> &'static str {
    match e {
        EpistemicStatus::Asserted => "asserted",
        EpistemicStatus::Hedged => "hedged",
        EpistemicStatus::Reported => "reported",
        EpistemicStatus::Conditional => "conditional",
        EpistemicStatus::Presupposed => "presupposed",
    }
}

fn extract_sentiment(doc: &Document) -> SentimentSummary {
    for k in doc.layers().filter(|k| k.ty == std::any::TypeId::of::<Categories<View>>()) {
        let Some(cat) = doc.get_key(&Key::<Categories<View>>::named(k.name)) else { continue };
        let Some(row) = cat.get(0) else { continue };
        if row.best_tag != Tag::NONE {
            let sc = row.best_score();
            return SentimentSummary {
                label: Some(row.best.to_owned()),
                probability: (sc.is_finite() && (0.0..=1.0).contains(&sc)).then_some(sc),
            };
        }
    }
    SentimentSummary { label: None, probability: None }
}

fn extract_quality(doc: &Document) -> QualitySummary {
    let noise = doc.get::<Noise>();
    let arts = doc.get::<Artifacts>();
    let tq = doc.get::<TextQuality>();
    let (worst_noise_score, union_chars) = match noise {
        Some(n) => {
            let mut worst = 0.0f64;
            let mut spans = Vec::with_capacity(n.len());
            for i in 0..n.len() {
                if n.score[i].is_finite() && n.score[i] > worst {
                    worst = n.score[i].min(1.0);
                }
                spans.push(n.spans.get(Ix::new(i as u32)));
            }
            (worst, union_noise_chars(doc.text(n.spans.view()), &spans))
        }
        None => (0.0, 0),
    };
    QualitySummary {
        assessed: noise.is_some() && arts.is_some(),
        worst_noise_score,
        union_noise_chars: union_chars,
        artifact_count: arts.map_or(0, |a| a.len() as i64),
        composite_score: tq.map(|q| f64::from(q.metrics.composite_score)),
    }
}

fn extract_entities(
    source: &str,
    table: &OffsetTable,
    doc: &Document,
) -> (Vec<EntityPolarityHit>, Vec<EntityPolarityHit>) {
    let Some(ents) = doc.get::<Entities>() else { return (Vec::new(), Vec::new()) };
    let linked = doc.get::<LinkedEntities>();
    let modality = doc.get::<Modality>();
    let v = ents.spans.view();
    let (mut affirmed, mut negated) = (Vec::new(), Vec::new());
    for i in 0..ents.len() {
        let sp = ents.spans.get(Ix::new(i as u32));
        let src_sp = doc.to_source(v, sp);
        let tag = modality.and_then(|m| {
            let msp = if v == m.spans.view() { sp } else { doc.map_span(v, m.spans.view(), sp).ok()? };
            m.tag_for_span(msp)
        });
        let is_neg = tag.as_ref().is_some_and(|t| t.polarity == ModalityPolarity::Negated);
        let hit = EntityPolarityHit {
            spans: ProjectedSpanSet::from_source_span(source, table, src_sp),
            surface: source.get(src_sp.range()).unwrap_or_default().to_owned(),
            kind: ents.kinds.label(ents.kind[i]).to_owned(),
            kb_id: linked.and_then(|l| l.kb_id.get(i).cloned().flatten()),
            canonical_name: linked.and_then(|l| l.canonical_name.get(i).cloned()).filter(|s| !s.is_empty()),
            score: linked.and_then(|l| l.score.get(i).copied()).unwrap_or(ents.score[i]),
            epistemic: tag.as_ref().map_or("asserted", |t| epistemic_str(t.epistemic)).to_owned(),
            cue: tag.and_then(|t| {
                t.cue.and_then(|csp| {
                    let mv = modality.expect("modality present").spans.view();
                    doc.text(mv).get(csp.range()).map(str::to_owned)
                })
            }),
        };
        if is_neg {
            negated.push(hit);
        } else {
            affirmed.push(hit);
        }
    }
    (affirmed, negated)
}
