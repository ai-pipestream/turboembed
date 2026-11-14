//! Unified one-pass `AnalyzeQuery` execution and wire conversion (`#2339`).

mod encode;
mod validate;

use std::sync::Arc;

use turbo_analysis_layers::{Date, TimeGrain};
use turbo_matcher::PhraseSynonymExpander;
use turbo_query::{
    FieldQueryPlan, FieldQuerySpec, FieldTokenizerKind, OnePassQueryAnalyzer, OrdinalQueryTerm, QuerySynonymMatch,
    StructuredFilterPayload, StructuredFilterPlan, TermForm,
};

use crate::api::{Failure, Result, TURBO_E_INTERNAL, TURBO_E_INVALID_ARGUMENT};
use crate::offsets::{OffsetMode, Offsetter};
use crate::proto::result::{
    AnalyzeQueryRequest, AnalyzeQueryResponse, QueryEntityFilterProto, QueryFieldPlanProto, QueryFieldProfileProto,
    QueryFieldTokenizerProto, QueryGlossaryFilterProto, QueryMoneyFilterProto, QueryOrdinalTermProto,
    QueryPlaceFilterProto, QueryQuantityFilterProto, QueryStructuredFilterProto, QuerySynonymMatchProto,
    QuerySynonymRuleProto, QueryTemporalFilterProto, QueryTermFormProto, SearchGeoBbox,
    query_structured_filter_proto::Payload as FilterPayloadProto,
};
use crate::state::State;

fn bad(msg: impl Into<String>) -> Failure {
    Failure::new(TURBO_E_INVALID_ARGUMENT, 0, msg)
}

fn parse_iso_date(s: &str) -> Result<Date> {
    let bytes = s.as_bytes();
    if bytes.len() != 10 || bytes[4] != b'-' || bytes[7] != b'-' {
        return Err(bad(format!("reference_date_iso {s:?} must be YYYY-MM-DD")));
    }
    let y: i32 = s[0..4].parse().map_err(|_| bad(format!("invalid year in reference_date_iso {s:?}")))?;
    let m: u8 = s[5..7].parse().map_err(|_| bad(format!("invalid month in reference_date_iso {s:?}")))?;
    let d: u8 = s[8..10].parse().map_err(|_| bad(format!("invalid day in reference_date_iso {s:?}")))?;
    Date::new(y, m, d).ok_or_else(|| bad(format!("out-of-range calendar date in reference_date_iso {s:?}")))
}

fn compile_synonyms(rules: &[QuerySynonymRuleProto]) -> Result<Option<Arc<PhraseSynonymExpander>>> {
    if rules.is_empty() {
        return Ok(None);
    }
    let mut exp = PhraseSynonymExpander::lower();
    for (idx, r) in rules.iter().enumerate() {
        if r.trigger.is_empty() || r.expansion.is_empty() {
            return Err(bad(format!("synonym rule {idx} must have non-empty trigger and expansion tokens")));
        }
        let trig: Vec<&str> = r.trigger.iter().map(String::as_str).collect();
        let targ: Vec<&str> = r.expansion.iter().map(String::as_str).collect();
        let res =
            if r.bidirectional { exp.add_equivalent_group(&[&trig, &targ]) } else { exp.add_synonym(&trig, &targ) };
        res.map_err(|e| bad(format!("invalid synonym rule {idx}: {e}")))?;
    }
    Ok(Some(Arc::new(exp)))
}

fn compile_field_spec(p: &QueryFieldProfileProto) -> Result<FieldQuerySpec> {
    if p.field.trim().is_empty() {
        return Err(bad("field_profiles[].field must be non-empty"));
    }
    let tokenizer = match QueryFieldTokenizerProto::try_from(p.tokenizer) {
        Ok(QueryFieldTokenizerProto::QueryFieldTokenizerWord) => FieldTokenizerKind::Word,
        Ok(QueryFieldTokenizerProto::QueryFieldTokenizerWhitespace) => FieldTokenizerKind::Whitespace,
        Err(_) => return Err(bad(format!("unknown QueryFieldTokenizerProto {}", p.tokenizer))),
    };
    let term_form = match QueryTermFormProto::try_from(p.term_form) {
        Ok(QueryTermFormProto::QueryTermFormStem) => TermForm::Stem,
        Ok(QueryTermFormProto::QueryTermFormFolded) => TermForm::Folded,
        Ok(QueryTermFormProto::QueryTermFormSurface) => TermForm::Surface,
        Ok(QueryTermFormProto::QueryTermFormLemma) => TermForm::Lemma,
        Err(_) => return Err(bad(format!("unknown QueryTermFormProto {}", p.term_form))),
    };
    let mut spec = FieldQuerySpec::new(p.field.trim()).with_tokenizer(tokenizer).with_term_form(term_form);
    spec.strip_invisible = !p.keep_invisible;
    spec.filter_stopwords = !p.include_stopwords_in_residual;
    if !p.language.trim().is_empty() {
        spec.language = Some(p.language.trim().to_owned());
    }
    spec.synonyms = compile_synonyms(&p.synonyms)?;
    Ok(spec)
}

fn grain_str(g: TimeGrain) -> &'static str {
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

fn map_span(offsetter: &Offsetter, text: &str, start: u32, end: u32) -> (u32, u32) {
    let w = offsetter.column_to_wire(text, &[start, end]);
    (w[0], w[1])
}

fn term_to_proto(offsetter: &Offsetter, text: &str, t: &OrdinalQueryTerm) -> QueryOrdinalTermProto {
    let (span_start, span_end) = map_span(offsetter, text, t.span.start, t.span.end);
    QueryOrdinalTermProto {
        term: t.term.clone(),
        ordinal: t.ordinal,
        pos_len: t.pos_len,
        span_start,
        span_end,
        surface: t.surface.clone(),
        is_stopword: t.is_stopword,
        covered_by_filter: t.covered_by_filter,
    }
}

fn syn_to_proto(offsetter: &Offsetter, text: &str, s: &QuerySynonymMatch) -> QuerySynonymMatchProto {
    let (span_start, span_end) = map_span(offsetter, text, s.span.start, s.span.end);
    QuerySynonymMatchProto {
        start_ordinal: s.start_ordinal,
        end_ordinal: s.end_ordinal,
        pos_len: s.pos_len,
        span_start,
        span_end,
        surface: s.surface.clone(),
        terms: s.terms.clone(),
    }
}

fn field_to_proto(offsetter: &Offsetter, text: &str, fp: &FieldQueryPlan) -> QueryFieldPlanProto {
    QueryFieldPlanProto {
        field: fp.field.clone(),
        total_token_ordinals: fp.total_token_ordinals,
        empty_normalized_ordinals: fp.empty_normalized_ordinals.clone(),
        bm25_terms: fp.bm25_terms.iter().map(|t| term_to_proto(offsetter, text, t)).collect(),
        residual_terms: fp.residual_terms.iter().map(|t| term_to_proto(offsetter, text, t)).collect(),
        synonyms: fp.synonyms.iter().map(|s| syn_to_proto(offsetter, text, s)).collect(),
    }
}

fn filter_to_proto(offsetter: &Offsetter, text: &str, f: &StructuredFilterPlan) -> QueryStructuredFilterProto {
    let (span_start, span_end) = map_span(offsetter, text, f.span.start, f.span.end);
    let (value_span_start, value_span_end) = map_span(offsetter, text, f.value_span.start, f.value_span.end);
    let payload = match &f.payload {
        StructuredFilterPayload::Temporal(t) => FilterPayloadProto::Temporal(QueryTemporalFilterProto {
            start_epoch_seconds: t.start_epoch_seconds,
            end_epoch_seconds: t.end_epoch_seconds,
            duration_ms: t.duration_ms,
            value_text: t.value_text.clone(),
            kind: t.kind.name().to_owned(),
            grain: grain_str(t.grain).to_owned(),
            origin: t.origin.name().to_owned(),
            ambiguous: t.ambiguous,
        }),
        StructuredFilterPayload::Money { raw, amount_f64, ecb_converted } => {
            FilterPayloadProto::Money(QueryMoneyFilterProto {
                currency: raw.currency.as_str().to_owned(),
                decimal_text: raw.decimal_text.clone(),
                amount_f64: *amount_f64,
                per: raw.per.clone().unwrap_or_default(),
                ecb_currency: ecb_converted.as_ref().map(|c| c.currency.as_str().to_owned()),
                ecb_decimal_text: ecb_converted.as_ref().map(|c| c.decimal_text.clone()),
                ecb_amount_f64: ecb_converted.as_ref().map(|c| c.amount_f64),
            })
        }
        StructuredFilterPayload::Quantity { raw, si_base_unit, si_plain_value, si_value_f64 } => {
            FilterPayloadProto::Quantity(QueryQuantityFilterProto {
                written_unit: raw.unit.clone(),
                written_decimal_text: raw.decimal_text.clone(),
                si_base_unit: si_base_unit.clone(),
                si_plain_value: si_plain_value.clone(),
                si_value_f64: *si_value_f64,
                is_percent: raw.is_percent,
            })
        }
        StructuredFilterPayload::Place { raw, bbox, spine_path } => FilterPayloadProto::Place(QueryPlaceFilterProto {
            source: raw.source.clone(),
            record_id: raw.record_id.clone(),
            canonical_name: raw.canonical_name.clone(),
            country_code: raw.country_code.map_or_else(String::new, |c| c.as_str().to_owned()),
            containment: raw.containment.clone(),
            spine_path: spine_path.clone(),
            feature_class: raw.feature_class.clone().unwrap_or_default(),
            latitude: raw.location.latitude(),
            longitude: raw.location.longitude(),
            bbox: bbox.map(|b| SearchGeoBbox { west: b.west(), south: b.south(), east: b.east(), north: b.north() }),
            population: raw.population,
            confidence: raw.confidence,
        }),
        StructuredFilterPayload::Glossary(g) => FilterPayloadProto::Glossary(QueryGlossaryFilterProto {
            id: g.id.clone(),
            term: g.term.clone(),
            entry_index: g.entry_index,
        }),
        StructuredFilterPayload::Entity { raw, kb_id, provider_id, canonical_name } => {
            FilterPayloadProto::Entity(QueryEntityFilterProto {
                entity_type: raw.entity_type.clone(),
                score: raw.score,
                kb_id: kb_id.clone(),
                provider_id: provider_id.clone(),
                canonical_name: canonical_name.clone(),
            })
        }
    };
    QueryStructuredFilterProto {
        span_start,
        span_end,
        value_span_start,
        value_span_end,
        surface: f.surface.clone(),
        comparison: f.comparison.as_str().to_owned(),
        layer_id: f.evidence.layer_id.to_owned(),
        annotator: f.evidence.annotator.clone(),
        payload: Some(payload),
    }
}

/// Executes one-pass `AnalyzeQuery` against `state` (`#2339`).
pub(crate) fn analyze_query(
    state: &State,
    req: &AnalyzeQueryRequest,
    priority: crate::priority::Priority,
    deadline: &crate::deadline::Deadline,
) -> Result<AnalyzeQueryResponse> {
    crate::priority::check_before_exec(deadline, &state.metrics)?;
    let offsets = OffsetMode::from_proto(req.offsets)?;
    let mut builder = OnePassQueryAnalyzer::builder();
    if !req.default_language.trim().is_empty() {
        builder = builder.with_default_language(req.default_language.trim());
    }
    if !req.reference_date_iso.trim().is_empty() {
        builder = builder.with_reference_date(parse_iso_date(req.reference_date_iso.trim())?);
    }
    if let Some(syns) = compile_synonyms(&req.global_synonyms)? {
        builder = builder.with_synonyms(syns);
    }
    for fp in &req.field_profiles {
        builder = builder.add_field(compile_field_spec(fp)?);
    }
    let analyzer = builder.build().map_err(|e| Failure::new(TURBO_E_INTERNAL, 0, e.to_string()))?;
    let plan = analyzer.analyze(&req.query).map_err(|e| Failure::new(TURBO_E_INTERNAL, 0, e.to_string()))?;

    let dense_results = encode::encode_dense_targets(state, &req.dense_targets, &plan, priority, deadline)?;
    let splade_results = encode::encode_splade_targets(state, &req.splade_targets, &plan, offsets, priority, deadline)?;

    let offsetter = Offsetter::new(&plan.query, offsets);
    let (detected_language, language_score) = match &plan.language {
        Some(l) => (l.best_code.clone(), l.best_score),
        None => (String::new(), 0.0),
    };

    Ok(AnalyzeQueryResponse {
        query: plan.query.clone(),
        offsets: offsets.to_proto(),
        detected_language,
        language_score,
        fields: plan.fields.iter().map(|fp| field_to_proto(&offsetter, &plan.query, fp)).collect(),
        filters: plan.filters.iter().map(|f| filter_to_proto(&offsetter, &plan.query, f)).collect(),
        residual_terms: plan.residual_terms.iter().map(|t| term_to_proto(&offsetter, &plan.query, t)).collect(),
        free_text: plan.free_text,
        dense_results,
        splade_results,
    })
}
