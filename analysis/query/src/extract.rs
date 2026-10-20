//! Extracts a [`QueryAnalysis`] with source-byte spans and [`FilterEvidence`]
//! from an analyzed [`Document`].

use turbo_analysis::{Document, Error, Ix, Key, Layer, Span, ViewId};
use turbo_analysis_layers::{
    Entities, Language, LanguageSpans, Money, Origin, Quantities, Relation, Temporals, Terms as LinguisticTerms,
    TimeKind, Times, Tokens,
};
use turbo_chunker::terms::{IndexedTerms, Vocabulary};
use turbo_geo::Locations;
use turbo_glossary_layers::GlossaryTerms;
use turbo_grammar::{GRAMMAR_MONEY, GRAMMAR_PERCENTS};
use turbo_stopword::Stopwords;

use crate::model::{
    EntityFilter, FilterEvidence, FilterKind, GlossaryFilter, LanguageEvidence, LanguageSpanEvidence, MoneyFilter,
    PlaceFilter, QuantityFilter, QueryAnalysis, QueryFilter, QueryTerm, TemporalFilter, point_end_from_grain,
    resolve_temporal_value_range,
};

pub(crate) const REGEX_ENTITIES: &str = "regex";
pub(crate) const DICT_ENTITIES: &str = "dict";
pub(crate) const ES_TIMES: &str = "es";

fn evidence_for<L: Layer>(doc: &Document, key: &Key<L>) -> FilterEvidence {
    let key_id = key.id();
    let annotator = doc.provenance(key_id).map(|p| p.annotator.to_owned()).unwrap_or_default();
    let instance = (!key.name.is_empty()).then(|| key.name.to_owned());
    FilterEvidence { layer_id: L::ID, instance, annotator }
}

fn source_span_and_surface(doc: &Document, view: ViewId, span: Span) -> (Span, String) {
    let src_span = doc.to_source(view, span);
    let surface = doc.source()[src_span.range()].to_owned();
    (src_span, surface)
}

pub(crate) fn extract_query_analysis(doc: &Document) -> Result<QueryAnalysis, Error> {
    let language = extract_language(doc);
    let mut filters = Vec::new();
    let temporal_filters = extract_temporals(doc);
    let money_filters = extract_money(doc, &temporal_filters);
    let quantity_filters = extract_quantities(doc, &temporal_filters, &money_filters);
    let place_filters = extract_places(doc, &temporal_filters, &money_filters, &quantity_filters);
    let glossary_filters = extract_glossary(doc);

    filters.extend(temporal_filters);
    filters.extend(money_filters);
    filters.extend(quantity_filters);
    filters.extend(place_filters);
    filters.extend(glossary_filters);

    let entity_filters = extract_entities(doc, &filters);
    filters.extend(entity_filters);

    filters.sort_by_key(|f| (f.span.start, f.span.end));

    let tokens = doc.get::<Tokens>();
    let ling_terms = doc.get::<LinguisticTerms>();
    let vocab = doc.get::<Vocabulary>();
    let indexed = doc.get::<IndexedTerms>();
    let stopwords = doc.get::<Stopwords>();

    let mut terms = Vec::new();
    let mut bm25_terms = Vec::new();
    let mut residual_terms = Vec::new();
    let mut free_text_spans = Vec::new();
    let mut free_text_parts = Vec::new();

    if let Some(tokens) = tokens {
        let view = tokens.spans.view();
        for (tok_ix, raw_span) in tokens.spans.iter() {
            let i = tok_ix.get();
            let (span, surface) = source_span_and_surface(doc, view, raw_span);
            let slot = ling_terms.map(|lt| lt.get(tok_ix));
            let folded = slot.map(|s| s.folded.to_owned()).unwrap_or_else(|| surface.to_lowercase());
            let stem = slot.and_then(|s| s.stem.map(str::to_owned));
            let indexed_term = indexed
                .and_then(|ix_layer| ix_layer.term.get(i).copied().flatten())
                .and_then(|v_ix| vocab.map(|v| v.get(v_ix).to_owned()));
            let is_stopword = stopwords.and_then(|sw| sw.stop.get(i).copied()).unwrap_or(false);
            let covered_by_filter = filters.iter().any(|f| f.span.overlaps(span));

            if let Some(ref term_str) = indexed_term {
                bm25_terms.push(term_str.clone());
                if !covered_by_filter && !is_stopword {
                    residual_terms.push(term_str.clone());
                }
            }
            if !covered_by_filter && !is_stopword && indexed_term.is_some() {
                free_text_spans.push(span);
                free_text_parts.push(surface.clone());
            }

            terms.push(QueryTerm {
                token_index: i,
                span,
                surface,
                folded,
                stem,
                indexed_term,
                is_stopword,
                covered_by_filter,
            });
        }
    }

    Ok(QueryAnalysis {
        query: doc.source().to_owned(),
        language,
        terms,
        bm25_terms,
        residual_terms,
        free_text_spans,
        free_text: free_text_parts.join(" "),
        filters,
    })
}

fn extract_language(doc: &Document) -> Option<LanguageEvidence> {
    let lang_key = Key::<Language>::DEFAULT;
    let lang = doc.get_key(&lang_key)?;
    let (best_code, best_score) = lang.best()?;
    let rankings = lang.iter().map(|(c, s)| (c.to_owned(), s)).collect();
    let evidence = evidence_for(doc, &lang_key);

    let mut spans = Vec::new();
    let span_key = Key::<LanguageSpans>::DEFAULT;
    if let Some(ls) = doc.get_key(&span_key) {
        let span_evidence = evidence_for(doc, &span_key);
        let view = ls.spans.view();
        for item in ls.iter() {
            let (src_span, surface) = source_span_and_surface(doc, view, item.span);
            spans.push(LanguageSpanEvidence {
                span: src_span,
                surface,
                code: item.code.to_owned(),
                script: item.script,
                confidence: item.confidence,
                evidence: span_evidence.clone(),
            });
        }
    }

    Some(LanguageEvidence { best_code: best_code.to_owned(), best_score, rankings, spans, evidence })
}

fn extract_temporals(doc: &Document) -> Vec<QueryFilter> {
    let mut candidates: Vec<(u8, QueryFilter)> = Vec::new();

    let temp_key = Key::<Temporals>::DEFAULT;
    if let Some(temporals) = doc.get_key(&temp_key) {
        let ev = evidence_for(doc, &temp_key);
        let view = temporals.spans.view();
        for (ix, raw_span) in temporals.spans.iter() {
            let mention = temporals.get(ix);
            let (span, surface) = source_span_and_surface(doc, view, raw_span);
            let (start_epoch_seconds, end_epoch_seconds, grain) = resolve_temporal_value_range(mention.value);
            let duration_ms =
                start_epoch_seconds.zip(end_epoch_seconds).map(|(s, e)| e.saturating_sub(s).saturating_mul(1_000));
            candidates.push((
                0,
                QueryFilter {
                    span,
                    surface,
                    evidence: ev.clone(),
                    kind: FilterKind::Temporal(TemporalFilter {
                        start_epoch_seconds,
                        end_epoch_seconds,
                        duration_ms,
                        value_text: mention.value.to_string(),
                        kind: TimeKind::DateRange,
                        grain,
                        modifier: Default::default(),
                        origin: mention.origin,
                        ambiguous: false,
                    }),
                },
            ));
        }
    }

    for time_key in [Key::<Times>::DEFAULT, Key::<Times>::named(ES_TIMES)] {
        if let Some(times) = doc.get_key(&time_key) {
            let ev = evidence_for(doc, &time_key);
            let view = times.spans.view();
            for (ix, raw_span) in times.spans.iter() {
                let m = times.get(ix);
                let (span, surface) = source_span_and_surface(doc, view, raw_span);
                let start_epoch_seconds = m.start;
                let end_epoch_seconds = match (m.start, m.end) {
                    (_, Some(e)) => Some(e),
                    (Some(s), None)
                        if !matches!(m.kind, TimeKind::Duration | TimeKind::Set)
                            && m.modifier.relation == Relation::None =>
                    {
                        Some(point_end_from_grain(s, m.grain))
                    }
                    _ => None,
                };
                let duration_ms = m.duration_ms.or_else(|| {
                    start_epoch_seconds.zip(end_epoch_seconds).map(|(s, e)| e.saturating_sub(s).saturating_mul(1_000))
                });
                let origin = if is_relative_time_surface(&surface) { Origin::Relative } else { Origin::Absolute };
                candidates.push((
                    1,
                    QueryFilter {
                        span,
                        surface,
                        evidence: ev.clone(),
                        kind: FilterKind::Temporal(TemporalFilter {
                            start_epoch_seconds,
                            end_epoch_seconds,
                            duration_ms,
                            value_text: m.timex.to_owned(),
                            kind: m.kind,
                            grain: m.grain,
                            modifier: m.modifier,
                            origin,
                            ambiguous: m.ambiguous,
                        }),
                    },
                ));
            }
        }
    }

    select_non_overlapping(candidates)
}

fn is_relative_time_surface(surface: &str) -> bool {
    let lower = surface.to_lowercase();
    #[rustfmt::skip]
    const RELATIVE_MARKERS: &[&str] = &[
        "ago", "ayer", "coming", "esta", "este", "hoy", "last", "mañana", "next", "now", "pasada", "pasado", "past",
        "próxima", "próximo", "this", "today", "tomorrow", "tonight", "yesterday",
    ];
    lower.split_whitespace().any(|w| RELATIVE_MARKERS.contains(&w))
}

fn extract_money(doc: &Document, existing: &[QueryFilter]) -> Vec<QueryFilter> {
    let mut candidates: Vec<(u8, QueryFilter)> = Vec::new();
    for (priority, key) in [(0u8, Key::<Money>::DEFAULT), (1u8, GRAMMAR_MONEY)] {
        if let Some(money) = doc.get_key(&key) {
            let ev = evidence_for(doc, &key);
            let view = money.spans.view();
            for (ix, raw_span) in money.spans.iter() {
                let (span, surface) = source_span_and_surface(doc, view, raw_span);
                if existing.iter().any(|f| f.span.overlaps(span)) {
                    continue;
                }
                let m = money.get(ix);
                let decimal_text = money.amount(ix).to_plain_string();
                candidates.push((
                    priority,
                    QueryFilter {
                        span,
                        surface,
                        evidence: ev.clone(),
                        kind: FilterKind::Money(MoneyFilter {
                            decimal_text,
                            negative: m.negative,
                            coefficient: m.coefficient.to_owned(),
                            scale: m.scale,
                            currency: m.currency,
                        }),
                    },
                ));
            }
        }
    }
    select_non_overlapping(candidates)
}

fn extract_quantities(doc: &Document, temporals: &[QueryFilter], money: &[QueryFilter]) -> Vec<QueryFilter> {
    let mut candidates: Vec<(u8, QueryFilter)> = Vec::new();
    for (priority, key) in [(0u8, Key::<Quantities>::DEFAULT), (1u8, GRAMMAR_PERCENTS)] {
        if let Some(quantities) = doc.get_key(&key) {
            let ev = evidence_for(doc, &key);
            let view = quantities.spans.view();
            for (ix, raw_span) in quantities.spans.iter() {
                let (span, surface) = source_span_and_surface(doc, view, raw_span);
                if temporals.iter().chain(money.iter()).any(|f| f.span.overlaps(span)) {
                    continue;
                }
                let q = quantities.get(ix);
                let decimal_text = quantities.value(ix).to_plain_string();
                let is_percent = quantities.is_percent(ix);
                candidates.push((
                    priority,
                    QueryFilter {
                        span,
                        surface,
                        evidence: ev.clone(),
                        kind: FilterKind::Quantity(QuantityFilter {
                            decimal_text,
                            negative: q.negative,
                            coefficient: q.coefficient.to_owned(),
                            scale: q.scale,
                            unit: q.unit.to_owned(),
                            is_percent,
                        }),
                    },
                ));
            }
        }
    }
    select_non_overlapping(candidates)
}

fn extract_places(
    doc: &Document,
    temporals: &[QueryFilter],
    money: &[QueryFilter],
    quantities: &[QueryFilter],
) -> Vec<QueryFilter> {
    let loc_key = Key::<Locations>::DEFAULT;
    let Some(locations) = doc.get_key(&loc_key) else { return Vec::new() };
    let ev = evidence_for(doc, &loc_key);
    let view = locations.spans.view();
    let mut candidates: Vec<(u8, QueryFilter)> = Vec::new();

    for i in 0..locations.len() {
        let ix = Ix::new(i as u32);
        let (raw_span, entry, confidence) = locations.resolution(ix);
        let (span, surface) = source_span_and_surface(doc, view, raw_span);
        if temporals.iter().chain(money.iter()).chain(quantities.iter()).any(|f| f.span.overlaps(span)) {
            continue;
        }
        candidates.push((
            0,
            QueryFilter {
                span,
                surface,
                evidence: ev.clone(),
                kind: FilterKind::Place(PlaceFilter {
                    record_id: entry.record_id().to_owned(),
                    source: entry.source().to_owned(),
                    canonical_name: entry.name().to_owned(),
                    country_code: entry.region_code(),
                    containment: entry.containment().to_vec(),
                    feature_class: entry.feature_class().map(str::to_owned),
                    location: entry.location(),
                    population: entry.population(),
                    confidence,
                }),
            },
        ));
    }
    select_non_overlapping(candidates)
}

fn extract_glossary(doc: &Document) -> Vec<QueryFilter> {
    let key = Key::<GlossaryTerms>::DEFAULT;
    let Some(gt) = doc.get_key(&key) else { return Vec::new() };
    let ev = evidence_for(doc, &key);
    let view = gt.spans.view();
    let mut out = Vec::with_capacity(gt.len());
    for (ix, raw_span) in gt.spans.iter() {
        let i = ix.get();
        let entry_ix = gt.entry[i];
        let entry = gt.glossary.get(entry_ix);
        let (span, surface) = source_span_and_surface(doc, view, raw_span);
        out.push(QueryFilter {
            span,
            surface,
            evidence: ev.clone(),
            kind: FilterKind::Glossary(GlossaryFilter {
                id: entry.id.to_owned(),
                term: entry.term.to_owned(),
                entry_index: entry_ix.0,
            }),
        });
    }
    out
}

fn extract_entities(doc: &Document, existing: &[QueryFilter]) -> Vec<QueryFilter> {
    let mut candidates: Vec<(u8, QueryFilter)> = Vec::new();
    for (priority, key) in [
        (0u8, Key::<Entities>::named(DICT_ENTITIES)),
        (1u8, Key::<Entities>::named(REGEX_ENTITIES)),
        (2u8, Key::<Entities>::DEFAULT),
    ] {
        if let Some(entities) = doc.get_key(&key) {
            let ev = evidence_for(doc, &key);
            let view = entities.spans.view();
            for (ix, raw_span) in entities.spans.iter() {
                let i = ix.get();
                let entity_type = entities.kinds.label(entities.kind[i]);
                let (span, surface) = source_span_and_surface(doc, view, raw_span);
                if entity_type.eq_ignore_ascii_case("location")
                    && existing.iter().any(|f| matches!(f.kind, FilterKind::Place(_)) && f.span.overlaps(span))
                {
                    continue;
                }
                if existing.iter().any(|f| !matches!(f.kind, FilterKind::Glossary(_)) && f.span.overlaps(span)) {
                    continue;
                }
                candidates.push((
                    priority,
                    QueryFilter {
                        span,
                        surface,
                        evidence: ev.clone(),
                        kind: FilterKind::Entity(EntityFilter {
                            entity_type: entity_type.to_owned(),
                            score: entities.score[i],
                        }),
                    },
                ));
            }
        }
    }
    select_non_overlapping(candidates)
}

/// Selects non-overlapping filters by preferring longer byte spans first,
/// then lower priority rank (`0` before `1`), then earlier start offset.
fn select_non_overlapping(mut candidates: Vec<(u8, QueryFilter)>) -> Vec<QueryFilter> {
    candidates.sort_by(|(pa, a), (pb, b)| {
        let len_a = a.span.end.saturating_sub(a.span.start);
        let len_b = b.span.end.saturating_sub(b.span.start);
        len_b.cmp(&len_a).then_with(|| pa.cmp(pb)).then_with(|| a.span.start.cmp(&b.span.start))
    });
    let mut kept: Vec<QueryFilter> = Vec::with_capacity(candidates.len());
    for (_, cand) in candidates {
        if kept.iter().any(|k| k.span.overlaps(cand.span)) {
            continue;
        }
        kept.push(cand);
    }
    kept.sort_by_key(|f| (f.span.start, f.span.end));
    kept
}
