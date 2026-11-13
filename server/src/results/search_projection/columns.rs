//! Numeric, temporal, geospatial, category/sentiment, and monotone text
//! quality column projections for `SearchDocumentProjection` (`#2345`).

use std::any::TypeId;

use turbo_analysis::{Document, Ix, Key, KeyId, Layer, Span, Status, Tag, View};
use turbo_analysis_layers::{
    Aspects, CONVERTED_MONEY, Categories, CategoryScope, Chunks, Containment, DAY_SECONDS, Date, DocumentDate, Money,
    QualityMetrics, Quantities, Sentences, TemporalValue, Temporals, TextQuality, TimeGrain, Times,
};
use turbo_geo::Locations;
use turbo_normalize_layers::{Artifacts, Noise};

use super::{Failure, SearchProjCx};
use crate::proto::result::{
    CategoryScoreProto, PlaceAncestorProto, QualityMetricsAnnotation, SearchAspectSummary, SearchAssessmentState,
    SearchCategoryRow, SearchCategorySentiment, SearchGeoBbox, SearchGeoPoint, SearchMoneyValue,
    SearchNumericTemporalColumns, SearchQualitySummary, SearchQuantityValue, SearchTemporalRange, SearchTimeRange,
    SearchUtcDate,
};
use crate::results::receipt::error::layer_status;

fn time_grain_str(g: TimeGrain) -> &'static str {
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

fn date_epoch_seconds(d: Date) -> i64 {
    d.epoch_day().saturating_mul(DAY_SECONDS)
}

fn temporal_epoch_bounds(val: TemporalValue) -> (i64, i64) {
    match val {
        TemporalValue::Day(d) => {
            let start = date_epoch_seconds(d);
            let end = d.plus_days(1).map_or(i64::MAX, date_epoch_seconds);
            (start, end)
        }
        TemporalValue::Week { year, week } => {
            let Some(jan4) = Date::new(year, 1, 4) else { return (0, 0) };
            let weekday = (jan4.epoch_day() + 3).rem_euclid(7);
            let Some(w1_mon) = jan4.plus_days(-weekday) else { return (0, 0) };
            let Some(mon) = w1_mon.plus_weeks(i64::from(week.saturating_sub(1))) else { return (0, 0) };
            let start = date_epoch_seconds(mon);
            let end = mon.plus_days(7).map_or(i64::MAX, date_epoch_seconds);
            (start, end)
        }
        TemporalValue::Month { year, month } => {
            let Some(first) = Date::new(year, month, 1) else { return (0, 0) };
            let start = date_epoch_seconds(first);
            let end = first.plus_months(1).map_or(i64::MAX, date_epoch_seconds);
            (start, end)
        }
        TemporalValue::Quarter { year, quarter } => {
            let month = quarter.saturating_sub(1).saturating_mul(3).saturating_add(1);
            let Some(first) = Date::new(year, month, 1) else { return (0, 0) };
            let start = date_epoch_seconds(first);
            let end = first.plus_months(3).map_or(i64::MAX, date_epoch_seconds);
            (start, end)
        }
        TemporalValue::Year(year) => {
            let Some(first) = Date::new(year, 1, 1) else { return (0, 0) };
            let start = date_epoch_seconds(first);
            let end = first.plus_years(1).map_or(i64::MAX, date_epoch_seconds);
            (start, end)
        }
    }
}

fn push_money_layer(
    cx: &SearchProjCx<'_>,
    layer: &Money,
    ecb_converted: bool,
    out: &mut Vec<SearchMoneyValue>,
) -> Result<(), Failure> {
    let view = layer.spans.view();
    for i in 0..layer.len() {
        let ix = Ix::new(i as u32);
        let (span_start, span_end) = cx.map_to_wire(view, layer.spans.get(ix))?;
        let plain_amount = layer.amount(ix).to_plain_string();
        let amount_f64 = plain_amount.parse::<f64>().unwrap_or(0.0);
        out.push(SearchMoneyValue {
            span_start,
            span_end,
            currency: layer.currency[i].as_str().to_owned(),
            amount_f64,
            plain_amount,
            per: layer.per(ix).unwrap_or_default().to_owned(),
            ecb_converted,
        });
    }
    Ok(())
}

pub(crate) fn project_numeric_temporal(cx: &SearchProjCx<'_>) -> Result<SearchNumericTemporalColumns, Failure> {
    let mut money = Vec::new();
    if let Some(raw_m) = cx.doc.get_key(&Key::<Money>::DEFAULT) {
        push_money_layer(cx, raw_m, false, &mut money)?;
    }
    if let Some(ecb_m) = cx.doc.get_key(&CONVERTED_MONEY) {
        push_money_layer(cx, ecb_m, true, &mut money)?;
    }

    let mut quantities = Vec::new();
    if let Some(q) = cx.doc.get::<Quantities>() {
        let view = q.spans.view();
        for i in 0..q.len() {
            let ix = Ix::new(i as u32);
            let (span_start, span_end) = cx.map_to_wire(view, q.spans.get(ix))?;
            let base = q.base_unit.get(i);
            let (unit_str, dec_ref) =
                if base.is_empty() { (q.unit(ix), q.value(ix)) } else { (base, q.normalized_value(ix)) };
            let plain_value = dec_ref.to_plain_string();
            let value_f64 = plain_value.parse::<f64>().unwrap_or(0.0);
            quantities.push(SearchQuantityValue {
                span_start,
                span_end,
                unit: unit_str.to_owned(),
                value_f64,
                plain_value,
                is_percent: q.is_percent(ix),
            });
        }
    }

    let document_date = cx.doc.get::<DocumentDate>().and_then(|dd| dd.date).map(|d| SearchUtcDate {
        iso: d.to_string(),
        epoch_seconds: date_epoch_seconds(d),
        year: d.year(),
        month: u32::from(d.month()),
        day: u32::from(d.day()),
    });

    let mut temporals = Vec::new();
    if let Some(t) = cx.doc.get::<Temporals>() {
        let view = t.spans.view();
        for i in 0..t.len() {
            let ix = Ix::new(i as u32);
            let (span_start, span_end) = cx.map_to_wire(view, t.spans.get(ix))?;
            let val = t.value(ix);
            let (start_epoch_seconds, end_epoch_seconds) = temporal_epoch_bounds(val);
            temporals.push(SearchTemporalRange {
                span_start,
                span_end,
                granularity: t.granularity[i].name().to_owned(),
                origin: t.origin[i].name().to_owned(),
                iso: val.to_string(),
                start_epoch_seconds,
                end_epoch_seconds,
            });
        }
    }

    let mut times = Vec::new();
    if let Some(tm) = cx.doc.get::<Times>() {
        let view = tm.spans.view();
        for i in 0..tm.len() {
            let ix = Ix::new(i as u32);
            let (span_start, span_end) = cx.map_to_wire(view, tm.spans.get(ix))?;
            let m = tm.get(ix);
            times.push(SearchTimeRange {
                span_start,
                span_end,
                kind: m.kind.name().to_owned(),
                grain: time_grain_str(m.grain).to_owned(),
                timex: m.timex.to_owned(),
                start_epoch_ms: m.start,
                end_epoch_ms: m.end,
                duration_ms: m.duration_ms,
                ambiguous: m.ambiguous,
            });
        }
    }

    Ok(SearchNumericTemporalColumns { money, quantities, document_date, temporals, times })
}

pub(crate) fn project_geo(cx: &SearchProjCx<'_>) -> Result<Vec<SearchGeoPoint>, Failure> {
    let Some(locs) = cx.doc.get::<Locations>() else {
        return Ok(Vec::new());
    };
    let cont_layer = cx.doc.get::<Containment>();
    let view = locs.spans.view();
    let mut out = Vec::with_capacity(locs.len());
    for i in 0..locs.len() {
        let ix = Ix::new(i as u32);
        let (span_start, span_end) = cx.map_to_wire(view, locs.spans.get(ix))?;
        let (_, entry, confidence) = locs.resolution(ix);
        let pt = entry.location();
        let bbox = entry.bounding_box().map(|b| SearchGeoBbox {
            west: b.west(),
            south: b.south(),
            east: b.east(),
            north: b.north(),
        });
        let mut containment = Vec::new();
        if let Some(cl) = cont_layer.filter(|c| i < c.len()) {
            for a in cl.ancestors(Ix::new(i as u32)) {
                containment.push(PlaceAncestorProto {
                    id: a.id.to_owned(),
                    name: a.name.to_owned(),
                    kind: a.kind.to_owned(),
                });
            }
        }
        let spine_path = if !containment.is_empty() {
            let mut parts: Vec<&str> = containment.iter().rev().map(|a| a.name.as_str()).collect();
            parts.push(entry.name());
            parts.join("/")
        } else {
            let mut parts = Vec::new();
            if let Some(cc) = entry.country_code() {
                parts.push(cc);
            }
            for level in entry.containment() {
                parts.push(level.as_str());
            }
            parts.push(entry.name());
            parts.join("/")
        };
        out.push(SearchGeoPoint {
            span_start,
            span_end,
            source: entry.source().to_owned(),
            record_id: entry.record_id().to_owned(),
            canonical_name: entry.name().to_owned(),
            feature_class: entry.feature_class().unwrap_or_default().to_owned(),
            country_code: entry.country_code().unwrap_or_default().to_owned(),
            latitude: pt.latitude(),
            longitude: pt.longitude(),
            bbox,
            containment,
            spine_path,
            confidence,
        });
    }
    Ok(out)
}

fn collect_categories<S: CategoryScope>(doc: &Document, scope_name: &str, out: &mut Vec<SearchCategoryRow>) {
    for k in doc.layers().filter(|k| k.ty == TypeId::of::<Categories<S>>()) {
        let Some(cat) = doc.get_key(&Key::<Categories<S>>::named(k.name)) else { continue };
        for i in 0..cat.len() {
            let Some(row) = cat.get(i) else { continue };
            let abstained = row.best_tag == Tag::NONE;
            let probability = if abstained {
                None
            } else {
                let s = row.best_score();
                (s.is_finite() && (0.0..=1.0).contains(&s)).then_some(s)
            };
            let selected = row.selected.iter().map(|&t| row.categories.label(t).to_owned()).collect();
            let scores =
                row.iter_scores().map(|(label, score)| CategoryScoreProto { label: label.to_owned(), score }).collect();
            out.push(SearchCategoryRow {
                scope: scope_name.to_owned(),
                tagset: cat.categories.name.clone(),
                best_label: row.best.to_owned(),
                abstained,
                probability,
                selected,
                scores,
            });
        }
    }
}

pub(crate) fn project_categories_sentiment(cx: &SearchProjCx<'_>) -> Result<SearchCategorySentiment, Failure> {
    let mut categories = Vec::new();
    collect_categories::<View>(cx.doc, "document", &mut categories);
    collect_categories::<Chunks>(cx.doc, "chunk", &mut categories);
    collect_categories::<Sentences>(cx.doc, "sentence", &mut categories);

    let mut sentiment_label = None;
    let mut sentiment_score = None;
    for row in &categories {
        let ts = row.tagset.to_ascii_lowercase();
        let bl = row.best_label.to_ascii_lowercase();
        if !row.abstained && (ts.contains("sentiment") || matches!(bl.as_str(), "positive" | "negative" | "neutral")) {
            sentiment_label = Some(row.best_label.clone());
            sentiment_score = row.probability;
            break;
        }
    }

    let mut aspects = Vec::new();
    if let Some(asp) = cx.doc.get::<Aspects>() {
        let toks = cx.doc.get_key(&asp.tokens_key);
        let sents = cx.doc.get_key(&asp.sentences_key);
        for i in 0..asp.len() {
            let ix = Ix::new(i as u32);
            let a = asp.get(ix);
            let (span_start, span_end) = match (toks, sents) {
                (Some(t), s) => match asp.span(ix, t, s) {
                    Some(sp) => cx.map_to_wire(t.spans.view(), sp)?,
                    None => (0, 0),
                },
                (None, Some(s)) => match a.target.sentence().filter(|&si| si.get() < s.len()) {
                    Some(si) => cx.map_to_wire(s.spans.view(), s.spans.get(si))?,
                    None => (0, 0),
                },
                _ => (0, 0),
            };
            aspects.push(SearchAspectSummary {
                span_start,
                span_end,
                category: asp.category_label(ix).map(ToOwned::to_owned),
                polarity: a.polarity.label().name().to_owned(),
                positive: a.polarity.positive,
                negative: a.polarity.negative,
                neutral: a.polarity.neutral,
            });
        }
    }

    Ok(SearchCategorySentiment { categories, sentiment_label, sentiment_score, aspects })
}

fn quality_metrics_proto(m: QualityMetrics) -> QualityMetricsAnnotation {
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

fn assessment_state_of(
    doc: &Document,
    key: KeyId,
) -> (SearchAssessmentState, Option<crate::proto::result::LayerError>) {
    match doc.status(key) {
        None => (SearchAssessmentState::Unassessed, None),
        Some(Status::Complete) => (SearchAssessmentState::Assessed, None),
        Some(_) => {
            let (_, err) = layer_status(doc, key);
            (SearchAssessmentState::Failed, err)
        }
    }
}

/// Computes `union_noise_chars` by sort-and-sweep over `spans` on `view_text`
/// so overlapping or out-of-order findings count each damaged character once.
pub fn union_span_chars(view_text: &str, spans: &[Span]) -> i64 {
    let mut intervals: Vec<(u32, u32)> = spans.iter().filter(|s| s.start < s.end).map(|s| (s.start, s.end)).collect();
    if intervals.is_empty() {
        return 0;
    }
    intervals.sort_unstable();
    let mut chars = 0i64;
    let (mut cur_s, mut cur_e) = intervals[0];
    for &(s, e) in &intervals[1..] {
        if s <= cur_e {
            cur_e = cur_e.max(e);
        } else {
            if let Some(slice) = view_text.get(cur_s as usize..cur_e as usize) {
                chars += slice.chars().count() as i64;
            }
            cur_s = s;
            cur_e = e;
        }
    }
    if let Some(slice) = view_text.get(cur_s as usize..cur_e as usize) {
        chars += slice.chars().count() as i64;
    }
    chars
}

pub(crate) fn project_quality(doc: &Document) -> Result<SearchQualitySummary, Failure> {
    let noise_key = Key::<Noise>::DEFAULT.id();
    let art_key = Key::<Artifacts>::DEFAULT.id();
    let tq_key = Key::<TextQuality>::DEFAULT.id();

    let (noise_state, noise_error) = assessment_state_of(doc, noise_key);
    let (artifacts_state, artifacts_error) = assessment_state_of(doc, art_key);
    let (text_quality_state, _) = assessment_state_of(doc, tq_key);

    let (worst_noise_score, union_noise_chars) = match doc.get::<Noise>() {
        Some(n) => {
            let mut worst = 0.0f64;
            let mut spans = Vec::with_capacity(n.len());
            for i in 0..n.len() {
                let sc = n.score[i];
                if sc.is_finite() && sc > worst {
                    worst = sc.min(1.0);
                }
                spans.push(n.spans.get(Ix::new(i as u32)));
            }
            let view_text = doc.text(n.spans.view());
            (worst, union_span_chars(view_text, &spans))
        }
        None => (0.0, 0),
    };

    let artifact_count = doc.get::<Artifacts>().map_or(0, |a| a.len() as i64);
    let metrics = doc.get::<TextQuality>().map(|tq| quality_metrics_proto(tq.metrics));

    Ok(SearchQualitySummary {
        noise_state: noise_state as i32,
        artifacts_state: artifacts_state as i32,
        text_quality_state: text_quality_state as i32,
        worst_noise_score,
        union_noise_chars,
        artifact_count,
        metrics,
        noise_error,
        artifacts_error,
    })
}
