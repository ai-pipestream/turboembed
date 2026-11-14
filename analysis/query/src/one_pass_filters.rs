//! Structured filter enrichment (ECB money, SI quantities, geo bounding boxes
//! and containment spines, KB entity linking, and relational cue widening) for
//! [`crate::OnePassQueryAnalyzer`] (`#2339`).

use std::collections::HashMap;
use std::sync::Arc;

use turbo_analysis::{Annotator, Document, Error, Ix, Key, Layer, Span, View};
use turbo_analysis_layers::{Currency, Date, DecimalRef, Entities, Quantities, Relation, Sentences, Tokens};
use turbo_entity_linker::EntityLinker;
use turbo_geo::{GeoBoundingBox, Locations};
use turbo_grammar::GRAMMAR_PERCENTS;
use turbo_numeric::money::{Decimal, EcbRates, FxRates};

use crate::extract::{DICT_ENTITIES, REGEX_ENTITIES};
use crate::model::{FilterKind, PlaceFilter, QuantityFilter, QueryAnalysis, TemporalFilter};
use crate::one_pass_types::{ConvertedMoneyValue, QueryComparisonOp, StructuredFilterPayload, StructuredFilterPlan};

pub(crate) struct FilterEnricherConfig<'a> {
    pub(crate) ecb: Option<(&'a EcbRates, Currency, Date)>,
    pub(crate) entity_linker: Option<&'a Arc<EntityLinker>>,
}

pub(crate) fn build_structured_filters(
    doc: &Document,
    base: &QueryAnalysis,
    cfg: &FilterEnricherConfig<'_>,
) -> Result<Vec<StructuredFilterPlan>, Error> {
    let src = doc.source();
    let src_tokens = source_word_spans(doc);
    let linked_by_span = link_entity_spans(doc, cfg.entity_linker)?;
    let mut out = Vec::with_capacity(base.filters.len());

    for f in &base.filters {
        let value_span = f.span;
        match &f.kind {
            FilterKind::Temporal(t) => {
                let (span, comparison, adjusted) = enrich_temporal(src, &src_tokens, value_span, t.clone());
                out.push(StructuredFilterPlan {
                    span,
                    value_span,
                    surface: src[span.range()].to_owned(),
                    comparison,
                    evidence: f.evidence.clone(),
                    payload: StructuredFilterPayload::Temporal(adjusted),
                });
            }
            FilterKind::Money(m) => {
                let (span, comparison) = widen_numeric_cue(src, &src_tokens, value_span);
                let amount_f64 = m.decimal_text.parse::<f64>().unwrap_or(0.0);
                let ecb_converted = convert_money_ecb(m, amount_f64, cfg.ecb);
                out.push(StructuredFilterPlan {
                    span,
                    value_span,
                    surface: src[span.range()].to_owned(),
                    comparison,
                    evidence: f.evidence.clone(),
                    payload: StructuredFilterPayload::Money { raw: m.clone(), amount_f64, ecb_converted },
                });
            }
            FilterKind::Quantity(q) => {
                let (span, comparison) = widen_numeric_cue(src, &src_tokens, value_span);
                let (si_base_unit, si_plain_value, si_value_f64) = lookup_si_quantity(doc, value_span, q);
                out.push(StructuredFilterPlan {
                    span,
                    value_span,
                    surface: src[span.range()].to_owned(),
                    comparison,
                    evidence: f.evidence.clone(),
                    payload: StructuredFilterPayload::Quantity {
                        raw: q.clone(),
                        si_base_unit,
                        si_plain_value,
                        si_value_f64,
                    },
                });
            }
            FilterKind::Place(p) => {
                let span = widen_place_preposition(src, &src_tokens, value_span);
                let (bbox, spine_path) = lookup_place_geo(doc, value_span, p);
                out.push(StructuredFilterPlan {
                    span,
                    value_span,
                    surface: src[span.range()].to_owned(),
                    comparison: QueryComparisonOp::Eq,
                    evidence: f.evidence.clone(),
                    payload: StructuredFilterPayload::Place { raw: p.clone(), bbox, spine_path },
                });
            }
            FilterKind::Glossary(g) => {
                out.push(StructuredFilterPlan {
                    span: value_span,
                    value_span,
                    surface: f.surface.clone(),
                    comparison: QueryComparisonOp::Eq,
                    evidence: f.evidence.clone(),
                    payload: StructuredFilterPayload::Glossary(g.clone()),
                });
            }
            FilterKind::Entity(e) => {
                let (kb_id, provider_id, canonical_name) =
                    linked_by_span.get(&(value_span.start, value_span.end)).cloned().unwrap_or((None, None, None));
                out.push(StructuredFilterPlan {
                    span: value_span,
                    value_span,
                    surface: f.surface.clone(),
                    comparison: QueryComparisonOp::Eq,
                    evidence: f.evidence.clone(),
                    payload: StructuredFilterPayload::Entity { raw: e.clone(), kb_id, provider_id, canonical_name },
                });
            }
        }
    }
    out.sort_by_key(|f| (f.span.start, f.span.end));
    Ok(out)
}

fn source_word_spans(doc: &Document) -> Vec<Span> {
    let Some(tokens) = doc.get::<Tokens>() else { return Vec::new() };
    let view = tokens.spans.view();
    tokens.spans.iter().map(|(_, s)| doc.to_source(view, s)).collect()
}

type PrevToken<'a> = Option<(Span, &'a str)>;

fn prev_tokens<'a>(src: &'a str, toks: &[Span], before_start: u32) -> (PrevToken<'a>, PrevToken<'a>) {
    let idx = toks.partition_point(|s| s.end <= before_start);
    let p1 = idx.checked_sub(1).and_then(|i| {
        let sp = toks[i];
        (before_start.saturating_sub(sp.end) <= 4
            && src[sp.end as usize..before_start as usize].chars().all(char::is_whitespace))
        .then_some((sp, &src[sp.range()]))
    });
    let p2 = match (p1, idx.checked_sub(2)) {
        (Some((sp1, _)), Some(i2)) => {
            let sp2 = toks[i2];
            (sp1.start.saturating_sub(sp2.end) <= 4
                && src[sp2.end as usize..sp1.start as usize].chars().all(char::is_whitespace))
            .then_some((sp2, &src[sp2.range()]))
        }
        _ => None,
    };
    (p1, p2)
}

fn widen_numeric_cue(src: &str, toks: &[Span], val_span: Span) -> (Span, QueryComparisonOp) {
    let (p1, p2) = prev_tokens(src, toks, val_span.start);
    if let (Some((_, w1)), Some((sp2, w2))) = (p1, p2) {
        let l1 = w1.to_ascii_lowercase();
        let l2 = w2.to_ascii_lowercase();
        let two_word = match (l2.as_str(), l1.as_str()) {
            ("more" | "greater" | "higher", "than") => Some(QueryComparisonOp::Gt),
            ("less" | "fewer" | "lower", "than") => Some(QueryComparisonOp::Lt),
            ("at", "least") | ("no", "less") => Some(QueryComparisonOp::Gte),
            ("at", "most") | ("up", "to") => Some(QueryComparisonOp::Lte),
            _ => None,
        };
        if let Some(op) = two_word {
            return (Span::new(sp2.start, val_span.end), op);
        }
    }
    if let Some((sp1, w1)) = p1 {
        let l1 = w1.to_ascii_lowercase();
        let one_word = match l1.as_str() {
            "over" | "above" | "exceeding" | ">" => Some(QueryComparisonOp::Gt),
            "min" | "minimum" | ">=" => Some(QueryComparisonOp::Gte),
            "under" | "below" | "<" => Some(QueryComparisonOp::Lt),
            "max" | "maximum" | "<=" => Some(QueryComparisonOp::Lte),
            _ => None,
        };
        if let Some(op) = one_word {
            return (Span::new(sp1.start, val_span.end), op);
        }
    }
    (val_span, QueryComparisonOp::Eq)
}

fn enrich_temporal(
    src: &str,
    toks: &[Span],
    val_span: Span,
    mut t: TemporalFilter,
) -> (Span, QueryComparisonOp, TemporalFilter) {
    let existing_op = match (t.start_epoch_seconds, t.end_epoch_seconds, t.modifier.relation) {
        (_, _, Relation::After | Relation::More) => Some(QueryComparisonOp::Gt),
        (_, _, Relation::Since) => Some(QueryComparisonOp::Gte),
        (_, _, Relation::Before | Relation::Less) => Some(QueryComparisonOp::Lt),
        (_, _, Relation::Until) => Some(QueryComparisonOp::Lte),
        (Some(_), None, _) => {
            let low = src[val_span.range()].to_ascii_lowercase();
            Some(if low.starts_with("after ") { QueryComparisonOp::Gt } else { QueryComparisonOp::Gte })
        }
        (None, Some(_), _) => Some(QueryComparisonOp::Lt),
        _ => None,
    };
    if let Some(op) = existing_op {
        return (val_span, op, t);
    }
    if let (Some((sp1, w1)), _) = prev_tokens(src, toks, val_span.start) {
        let l1 = w1.to_ascii_lowercase();
        match l1.as_str() {
            "since" | "from" => {
                t.end_epoch_seconds = None;
                t.duration_ms = None;
                return (Span::new(sp1.start, val_span.end), QueryComparisonOp::Gte, t);
            }
            "after" => {
                t.start_epoch_seconds = t.end_epoch_seconds.or(t.start_epoch_seconds);
                t.end_epoch_seconds = None;
                t.duration_ms = None;
                return (Span::new(sp1.start, val_span.end), QueryComparisonOp::Gt, t);
            }
            "before" | "until" => {
                t.end_epoch_seconds = t.start_epoch_seconds.or(t.end_epoch_seconds);
                t.start_epoch_seconds = None;
                t.duration_ms = None;
                return (Span::new(sp1.start, val_span.end), QueryComparisonOp::Lt, t);
            }
            "in" | "during" | "on" => {
                let cmp = if t.start_epoch_seconds.is_some() && t.end_epoch_seconds.is_some() {
                    QueryComparisonOp::Range
                } else {
                    QueryComparisonOp::Eq
                };
                return (Span::new(sp1.start, val_span.end), cmp, t);
            }
            _ => {}
        }
    }
    let cmp = if t.start_epoch_seconds.is_some() && t.end_epoch_seconds.is_some() {
        QueryComparisonOp::Range
    } else {
        QueryComparisonOp::Eq
    };
    (val_span, cmp, t)
}

fn widen_place_preposition(src: &str, toks: &[Span], val_span: Span) -> Span {
    if let (Some((sp1, w1)), _) = prev_tokens(src, toks, val_span.start)
        && matches!(w1.to_ascii_lowercase().as_str(), "in" | "at" | "near" | "around" | "within")
    {
        return Span::new(sp1.start, val_span.end);
    }
    val_span
}

fn convert_money_ecb(
    m: &crate::model::MoneyFilter,
    raw_f64: f64,
    ecb: Option<(&EcbRates, Currency, Date)>,
) -> Option<ConvertedMoneyValue> {
    let (rates, target, as_of) = ecb?;
    if m.currency == target {
        return Some(ConvertedMoneyValue {
            currency: target,
            decimal_text: m.decimal_text.clone(),
            amount_f64: raw_f64,
        });
    }
    let dec_ref = DecimalRef { negative: m.negative, digits: &m.coefficient, scale: i64::from(m.scale) };
    let mut out = Decimal::default();
    let ok = FxRates::convert(rates, dec_ref, m.currency, target, as_of, &mut out).ok()?;
    if !ok {
        return None;
    }
    let decimal_text = out.as_ref().to_plain_string();
    let amount_f64 = decimal_text.parse::<f64>().unwrap_or(0.0);
    Some(ConvertedMoneyValue { currency: target, decimal_text, amount_f64 })
}

fn lookup_si_quantity(doc: &Document, val_span: Span, q: &QuantityFilter) -> (String, String, f64) {
    for key in [Key::<Quantities>::DEFAULT, GRAMMAR_PERCENTS] {
        let Some(layer) = doc.get_key(&key) else { continue };
        let view = layer.spans.view();
        for (ix, raw) in layer.spans.iter() {
            if doc.to_source(view, raw) == val_span {
                let i = ix.get();
                let base = layer.base_unit.get(i);
                let (u, dref) = if base.is_empty() {
                    (layer.unit(ix), layer.value(ix))
                } else {
                    (base, layer.normalized_value(ix))
                };
                let plain = dref.to_plain_string();
                let f = plain.parse::<f64>().unwrap_or(0.0);
                return (u.to_owned(), plain, f);
            }
        }
    }
    let f = q.decimal_text.parse::<f64>().unwrap_or(0.0);
    (q.unit.clone(), q.decimal_text.clone(), f)
}

fn lookup_place_geo(doc: &Document, val_span: Span, p: &PlaceFilter) -> (Option<GeoBoundingBox>, String) {
    let mut bbox = None;
    if let Some(locs) = doc.get::<Locations>() {
        let view = locs.spans.view();
        for i in 0..locs.len() {
            let ix = Ix::new(i as u32);
            if doc.to_source(view, locs.spans.get(ix)) == val_span {
                let (_, entry, _) = locs.resolution(ix);
                bbox = entry.bounding_box();
                break;
            }
        }
    }
    let mut parts = Vec::new();
    if let Some(cc) = p.country_code {
        parts.push(cc.as_str().to_owned());
    }
    parts.extend(p.containment.iter().cloned());
    parts.push(p.canonical_name.clone());
    (bbox, parts.join("/"))
}

type LinkedSpanMap = HashMap<(u32, u32), (Option<String>, Option<String>, Option<String>)>;

fn link_entity_spans(doc: &Document, linker: Option<&Arc<EntityLinker>>) -> Result<LinkedSpanMap, Error> {
    let mut map = HashMap::new();
    let Some(linker) = linker else { return Ok(map) };
    let view = doc.require(&Key::<View>::DEFAULT)?;
    let sentences = doc.get::<Sentences>();
    let tokens = doc.get::<Tokens>();
    for key in [Key::<Entities>::named(DICT_ENTITIES), Key::<Entities>::named(REGEX_ENTITIES), Key::<Entities>::DEFAULT]
    {
        let Some(entities) = doc.get_key(&key) else { continue };
        if entities.spans.is_empty() {
            continue;
        }
        let (linked,) = linker.annotate((view, entities, sentences, tokens, None, None, None, None))?;
        let eview = entities.spans.view();
        for (ix, raw) in entities.spans.iter() {
            let sp = doc.to_source(eview, raw);
            let row = linked.get(Ix::new(ix.get() as u32));
            if let Some(kb_id) = row.kb_id {
                let provider = (!row.provider_id.is_empty()).then(|| row.provider_id.to_owned());
                let canonical = (!row.canonical_name.is_empty()).then(|| row.canonical_name.to_owned());
                map.entry((sp.start, sp.end)).or_insert((Some(kb_id.to_owned()), provider, canonical));
            }
        }
    }
    Ok(map)
}
