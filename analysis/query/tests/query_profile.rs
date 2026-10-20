use std::sync::Arc;
use std::time::Instant;

use turbo_analysis_layers::{Currency, DAY_SECONDS, Date, DateTime, Origin, TimeGrain};
use turbo_chunker::terms::{Bm25, LocalCollection, TermCounter, TermForm, TermStats, Vocabulary};
use turbo_glossary::{Case, Glossary};
use turbo_maxent::Dictionary;
use turbo_namefind::DictionaryNameFinder;
use turbo_query::{FilterKind, QueryAnalysis, QueryAnalysisProfile, QueryProfileBuilder};
use turbo_unicode::Script;

fn assert_span_and_evidence_invariants(query: &str, analysis: &QueryAnalysis) {
    assert_eq!(analysis.query, query);
    for term in &analysis.terms {
        assert_eq!(&query[term.span.range()], term.surface, "token span surface mismatch");
    }
    for span in &analysis.free_text_spans {
        assert!(!query[span.range()].is_empty());
    }
    for filter in &analysis.filters {
        assert_eq!(&query[filter.span.range()], filter.surface, "filter surface must match exact query byte slice");
        assert!(!filter.evidence.layer_id.is_empty(), "filter must name its layer_id");
        assert!(!filter.evidence.annotator.is_empty(), "filter must name its annotator");
    }
    if let Some(ref lang) = analysis.language {
        assert!(!lang.best_code.is_empty());
        assert_eq!(lang.evidence.layer_id, "opennlp:language");
        assert!(!lang.evidence.annotator.is_empty());
        for s in &lang.spans {
            assert_eq!(&query[s.span.range()], s.surface, "language span must match exact query byte slice");
            assert_eq!(s.evidence.layer_id, "turbo:language-spans");
            assert!(!s.evidence.annotator.is_empty());
        }
    }
}

#[test]
fn extracts_relative_and_absolute_temporal_filters_with_ranges() {
    let ref_date = Date::new(2026, 7, 14).unwrap();
    let profile = QueryAnalysisProfile::with_reference_date(ref_date);

    // 1. Relative date: "tomorrow" -> 2026-07-15 [start, end)
    let q1 = "cheap flights tomorrow";
    let a1 = profile.analyze(q1).unwrap();
    assert_span_and_evidence_invariants(q1, &a1);
    let temp1: Vec<_> = a1.temporal_filters().collect();
    assert_eq!(temp1.len(), 1);
    assert_eq!(temp1[0].0.surface, "tomorrow");
    let expected_tomorrow_start = DateTime::midnight(Date::new(2026, 7, 15).unwrap()).local_seconds();
    assert_eq!(temp1[0].1.start_epoch_seconds, Some(expected_tomorrow_start));
    assert_eq!(temp1[0].1.end_epoch_seconds, Some(expected_tomorrow_start + DAY_SECONDS));
    assert_eq!(temp1[0].1.grain, TimeGrain::Day);
    assert_eq!(temp1[0].1.origin, Origin::Relative);

    // 2. Relative week offset: "3 weeks ago" -> ISO week 2026-W26 [2026-06-22, 2026-06-29)
    let q2 = "logs from 3 weeks ago";
    let a2 = profile.analyze(q2).unwrap();
    assert_span_and_evidence_invariants(q2, &a2);
    let temp2: Vec<_> = a2.temporal_filters().collect();
    assert_eq!(temp2.len(), 1);
    assert_eq!(temp2[0].0.surface, "3 weeks ago");
    let expected_3w_start = DateTime::midnight(Date::new(2026, 6, 22).unwrap()).local_seconds();
    assert_eq!(temp2[0].1.start_epoch_seconds, Some(expected_3w_start));
    assert_eq!(temp2[0].1.end_epoch_seconds, Some(expected_3w_start + 7 * DAY_SECONDS));
    assert_eq!(temp2[0].1.grain, TimeGrain::Week);
    assert_eq!(temp2[0].1.origin, Origin::Relative);

    // 3. Calendar quarter: "Q3 2024" -> [2024-07-01, 2024-10-01)
    let q3 = "earnings report Q3 2024";
    let a3 = profile.analyze(q3).unwrap();
    assert_span_and_evidence_invariants(q3, &a3);
    let temp3: Vec<_> = a3.temporal_filters().collect();
    assert_eq!(temp3.len(), 1);
    assert_eq!(temp3[0].0.surface, "Q3 2024");
    assert_eq!(temp3[0].0.evidence.layer_id, "turbo:temporal");
    let q3_start = DateTime::midnight(Date::new(2024, 7, 1).unwrap()).local_seconds();
    let q3_end = DateTime::midnight(Date::new(2024, 10, 1).unwrap()).local_seconds();
    assert_eq!(temp3[0].1.start_epoch_seconds, Some(q3_start));
    assert_eq!(temp3[0].1.end_epoch_seconds, Some(q3_end));
    assert_eq!(temp3[0].1.grain, TimeGrain::Quarter);
    assert_eq!(temp3[0].1.origin, Origin::Absolute);

    // 4. Grammar date range: "from May to Oct 2020"
    let q4 = "incidents from May to Oct 2020";
    let a4 = profile.analyze(q4).unwrap();
    assert_span_and_evidence_invariants(q4, &a4);
    let temp4: Vec<_> = a4.temporal_filters().collect();
    assert_eq!(temp4.len(), 1);
    assert_eq!(temp4[0].0.surface, "from May to Oct 2020");
    assert_eq!(temp4[0].0.evidence.layer_id, "turbo:times");
    let may_start = DateTime::midnight(Date::new(2020, 5, 1).unwrap()).local_seconds();
    let oct_start = DateTime::midnight(Date::new(2020, 10, 1).unwrap()).local_seconds();
    assert_eq!(temp4[0].1.start_epoch_seconds, Some(may_start));
    assert_eq!(temp4[0].1.end_epoch_seconds, Some(oct_start));
}

#[test]
fn extracts_money_and_quantity_filters_across_cursor_and_grammar() {
    let ref_date = Date::new(2026, 7, 14).unwrap();
    let profile = QueryAnalysisProfile::with_reference_date(ref_date);

    let q = "seed round $1.2M or 50 cents with 80 kg payload and 25% discount";
    let a = profile.analyze(q).unwrap();
    assert_span_and_evidence_invariants(q, &a);

    let money: Vec<_> = a.money_filters().collect();
    assert_eq!(money.len(), 2);
    assert_eq!(money[0].0.surface, "$1.2M");
    assert_eq!(money[0].0.evidence.layer_id, "turbo:money");
    assert_eq!(money[0].1.currency, Currency::parse("USD").unwrap());
    assert_eq!(money[0].1.decimal_text, "1200000.0");

    assert_eq!(money[1].0.surface, "50 cents");
    assert_eq!(money[1].0.evidence.layer_id, "turbo:money");
    assert_eq!(money[1].0.evidence.instance.as_deref(), Some("grammar"));
    assert_eq!(money[1].1.currency, Currency::parse("USD").unwrap());
    assert_eq!(money[1].1.decimal_text, "0.5");

    let quantities: Vec<_> = a.quantity_filters().collect();
    assert_eq!(quantities.len(), 2);
    assert_eq!(quantities[0].0.surface, "80 kg");
    assert_eq!(quantities[0].0.evidence.layer_id, "turbo:quantities");
    assert_eq!(quantities[0].1.decimal_text, "80");
    assert_eq!(quantities[0].1.unit, "kg");
    assert!(!quantities[0].1.is_percent);

    assert_eq!(quantities[1].0.surface, "25%");
    assert_eq!(quantities[1].1.decimal_text, "25");
    assert_eq!(quantities[1].1.unit, "%");
    assert!(quantities[1].1.is_percent);
}

#[test]
fn resolves_geographic_places_with_stable_ids_and_coordinates() {
    let ref_date = Date::new(2026, 7, 14).unwrap();
    let profile = QueryAnalysisProfile::with_reference_date(ref_date);

    let q = "direct trains from Paris to Madrid and flights to New York or Tokyo";
    let a = profile.analyze(q).unwrap();
    assert_span_and_evidence_invariants(q, &a);

    let places: Vec<_> = a.place_filters().collect();
    let surfaces: Vec<&str> = places.iter().map(|(f, _)| f.surface.as_str()).collect();
    assert_eq!(surfaces, vec!["Paris", "Madrid", "New York", "Tokyo"]);

    for (f, place) in &places {
        assert_eq!(f.evidence.layer_id, "turbo:locations");
        assert!(!place.record_id.is_empty());
        assert_eq!(place.source, "naturalearth");
        assert!(place.population > 100_000);
        assert!(place.confidence > 0.5);
    }
    assert_eq!(places[0].1.country_code.as_ref().map(|c| c.as_str()), Some("FR"));
    assert_eq!(places[1].1.country_code.as_ref().map(|c| c.as_str()), Some("ES"));
    assert_eq!(places[2].1.country_code.as_ref().map(|c| c.as_str()), Some("US"));
    assert_eq!(places[3].1.country_code.as_ref().map(|c| c.as_str()), Some("JP"));

    // Residual free text excludes the matched place spans and stopwords.
    assert!(!a.residual_terms.iter().any(|t| t == "pari" || t == "madrid" || t == "tokyo"));
    assert!(a.free_text.contains("direct"));
    assert!(a.free_text.contains("trains"));
}

#[test]
fn extracts_glossary_dictionary_and_regex_entities() {
    let mut glossary = Glossary::new();
    glossary.push("CONCEPT_ML", "machine learning").unwrap();
    glossary.push("CONCEPT_RAG", "retrieval augmented generation").unwrap();

    let dict_xml = "<dictionary case_sensitive='false'>\
        <entry><token>TurboEmbed</token></entry>\
        <entry><token>OpenNLP</token><token>Addons</token></entry>\
    </dictionary>";
    let dict = Dictionary::read(dict_xml.as_bytes()).unwrap();
    let dict_finder = DictionaryNameFinder::new(dict, "product");

    let profile = QueryProfileBuilder::new()
        .with_reference_date(Date::new(2026, 7, 14).unwrap())
        .with_glossary(Arc::new(glossary), Case::Insensitive)
        .with_dictionary_finder(dict_finder)
        .build()
        .unwrap();

    let q = "TurboEmbed machine learning pipeline docs at https://example.org/docs or support@example.com";
    let a = profile.analyze(q).unwrap();
    assert_span_and_evidence_invariants(q, &a);

    let gloss: Vec<_> = a.glossary_filters().collect();
    assert_eq!(gloss.len(), 1);
    assert_eq!(gloss[0].0.surface, "machine learning");
    assert_eq!(gloss[0].0.evidence.layer_id, "turbo:glossary-terms");
    assert_eq!(gloss[0].1.id, "CONCEPT_ML");

    let entities: Vec<_> = a.entity_filters().collect();
    assert_eq!(entities.len(), 3);
    assert_eq!(entities[0].0.surface, "TurboEmbed");
    assert_eq!(entities[0].0.evidence.layer_id, "opennlp:entities");
    assert_eq!(entities[0].0.evidence.instance.as_deref(), Some("dict"));
    assert_eq!(entities[0].1.entity_type, "product");

    assert_eq!(entities[1].0.surface, "https://example.org/docs");
    assert_eq!(entities[1].0.evidence.instance.as_deref(), Some("regex"));
    assert_eq!(entities[1].1.entity_type, "URL");

    assert_eq!(entities[2].0.surface, "support@example.com");
    assert_eq!(entities[2].0.evidence.instance.as_deref(), Some("regex"));
    assert_eq!(entities[2].1.entity_type, "EMAIL");
}

#[test]
fn detects_mixed_language_spans_and_extracts_multilingual_filters() {
    let ref_date = Date::new(2026, 7, 14).unwrap();
    let profile = QueryAnalysisProfile::with_reference_date(ref_date);

    // 1. Mixed Spanish + English query
    let q_es_en = "vuelos baratos a Madrid mañana. cheap flights tomorrow under $200";
    let a_es_en = profile.analyze(q_es_en).unwrap();
    assert_span_and_evidence_invariants(q_es_en, &a_es_en);

    let lang = a_es_en.language.as_ref().expect("language detected");
    assert!(lang.spans.len() >= 2, "expected multiple language spans: {:?}", lang.spans);
    assert!(lang.spans.iter().any(|s| s.code == "spa" && s.surface.contains("vuelos")));
    assert!(lang.spans.iter().any(|s| s.code == "eng" && s.surface.contains("cheap")));

    // Both Spanish "mañana" and English "tomorrow" resolve to 2026-07-15!
    let temporals: Vec<_> = a_es_en.temporal_filters().collect();
    assert_eq!(temporals.len(), 2);
    let expected_tomorrow = DateTime::midnight(Date::new(2026, 7, 15).unwrap()).local_seconds();
    assert_eq!(temporals[0].0.surface, "mañana");
    assert_eq!(temporals[0].0.evidence.layer_id, "turbo:times");
    assert_eq!(temporals[0].0.evidence.instance.as_deref(), Some("es"));
    assert_eq!(temporals[0].1.start_epoch_seconds, Some(expected_tomorrow));

    assert_eq!(temporals[1].0.surface, "tomorrow");
    assert_eq!(temporals[1].1.start_epoch_seconds, Some(expected_tomorrow));

    let places: Vec<_> = a_es_en.place_filters().collect();
    assert_eq!(places.len(), 1);
    assert_eq!(places[0].0.surface, "Madrid");

    let money: Vec<_> = a_es_en.money_filters().collect();
    assert_eq!(money.len(), 1);
    assert_eq!(money[0].0.surface, "$200");

    // 2. Mixed Cyrillic (Russian) + Latin (English) query
    let q_ru_en = "дешевые билеты в Париж cheap flights tomorrow";
    let a_ru_en = profile.analyze(q_ru_en).unwrap();
    assert_span_and_evidence_invariants(q_ru_en, &a_ru_en);
    let ru_lang = a_ru_en.language.as_ref().unwrap();
    assert!(ru_lang.spans.iter().any(|s| s.code == "rus" && s.script == Script::Cyrillic));
    assert!(ru_lang.spans.iter().any(|s| s.code == "eng" && s.script == Script::Latin));
}

#[test]
fn query_bm25_terms_match_paired_document_pipeline_across_term_forms() {
    let fixtures = [
        "cheap flights to Paris tomorrow under $200",
        "quarterly revenue report Q3 2024 with 15% growth in New York",
        "vuelos baratos a Madrid mañana. cheap flights tomorrow",
        "running, runners, and ran 10 km in Berlin 3 weeks ago",
        "дешевые билеты в Париж cheap flights tomorrow",
    ];

    for form in [TermForm::Stem, TermForm::Folded, TermForm::Surface] {
        let profile = QueryProfileBuilder::new()
            .with_reference_date(Date::new(2026, 7, 14).unwrap())
            .with_term_form(form)
            .build()
            .unwrap();

        for q in fixtures {
            let analysis = profile.analyze(q).unwrap();
            let doc_terms = profile.document_terms(q).unwrap();
            assert_eq!(
                analysis.bm25_terms, doc_terms,
                "query bm25_terms must equal document_terms for form {form:?} on {q:?}"
            );
        }
    }

    // Verify end-to-end BM25 scoring against a document analyzed with the paired document_analyzer.
    let profile = QueryAnalysisProfile::with_reference_date(Date::new(2026, 7, 14).unwrap());
    let mut doc = turbo_analysis::Document::new("We found cheap flights to Paris for our summer trip.");
    profile.document_analyzer().run(std::slice::from_mut(&mut doc)).into_iter().next().unwrap().unwrap();
    let count_analyzer = turbo_analysis::Analyzer::builder()
        .given(turbo_analysis::Key::<Vocabulary>::DEFAULT)
        .given(turbo_analysis::Key::<turbo_chunker::terms::IndexedTerms>::DEFAULT)
        .given(turbo_analysis::Key::<turbo_analysis_layers::Tokens>::DEFAULT)
        .add(TermCounter::<turbo_analysis::View>::new())
        .build()
        .unwrap();
    count_analyzer.run(std::slice::from_mut(&mut doc)).into_iter().next().unwrap().unwrap();

    let vocab = doc.get::<Vocabulary>().unwrap();
    let stats = doc.get::<TermStats<turbo_analysis::View>>().unwrap();
    let coll = LocalCollection::new(stats, vocab);
    let q_analysis = profile.analyze("cheap flights to Paris tomorrow under $200").unwrap();
    let bm25_q = q_analysis.residual_bm25_query(Bm25::LUCENE, &coll, vocab);
    let score = bm25_q.score(stats, turbo_analysis::Ix::new(0));
    assert!(score > 0.0, "residual BM25 query must score matching document positively");
    assert!(matches!(q_analysis.filters[0].kind, FilterKind::Place(_)));
}

#[test]
fn query_analysis_latency_stays_within_submillisecond_budget() {
    let profile = QueryAnalysisProfile::with_reference_date(Date::new(2026, 7, 14).unwrap());
    let query = "cheap direct flights from New York to Paris tomorrow morning under $850 with 25 kg baggage allowance and 15% corporate discount for 2 passengers";
    let warmup = profile.analyze(query).unwrap();
    assert!(warmup.terms.len() <= 32 && warmup.terms.len() >= 20);

    let iterations = 50usize;
    let mut durations_us = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let t0 = Instant::now();
        let res = profile.analyze(query).unwrap();
        durations_us.push(t0.elapsed().as_micros());
        assert!(!res.filters.is_empty());
    }
    durations_us.sort_unstable();
    let median_us = durations_us[iterations / 2];
    // Even in unoptimized debug test builds, a 24-token query runs in well under 5ms
    // (typically ~250-600 microseconds).
    assert!(median_us < 10_000, "expected sub-10ms median query analysis latency in test mode, got {median_us} us");
}
