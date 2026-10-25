use super::*;
use turbo_analysis_layers::{QualityFlags as F, QualityMetrics, SpanQuality};

#[test]
fn quality_floor_precedes_candidate_budget_on_the_real_lexical_pipeline() {
    let doc = document();
    let levels = doc.get::<Levels>().unwrap();
    let collection = LocalCollection::new(doc.get::<TermStats<Levels>>().unwrap(), doc.get::<Vocabulary>().unwrap());
    let mut cascade = Cascade::new(
        levels,
        CascadeConfig { candidates: 1, top_k: 10, ..Default::default() },
        None,
        None,
        Some(lexical(&doc, &collection)),
    )
    .unwrap();
    let terms = ["Cats", "Cats", "Cats", "Dogs"];
    let baseline = cascade.search(&terms);
    assert_eq!(baseline.hits.len(), 1);
    assert!(baseline.hits[0].spans.iter().any(|s| TEXT[s.range()].contains("Cats")));
    let mut quality = SpanQuality::new(ViewId::SOURCE);
    quality.push(
        Span::new(0, TEXT.find("# Dogs").unwrap() as u32),
        QualityMetrics { composite_score: 0.2, flags: F::NAV_BOILERPLATE, ..Default::default() },
    );
    cascade.retain_by_quality(&quality, 0.5, F::CLEAN).unwrap();
    let result = cascade.search(&terms);
    assert_eq!(result.pool, 1, "the next eligible unit fills the budget");
    assert_eq!(result.hits.len(), 1);
    assert!(result.hits[0].spans.iter().all(|s| !TEXT[s.range()].contains("Cats")));
    assert!(result.hits[0].spans.iter().any(|s| TEXT[s.range()].contains("Dogs")));
    assert_eq!(result.contextual_scored, 0);
    // An uncovered region is allowed, but cannot restore previously removed units.
    cascade.retain_by_quality(&SpanQuality::new(ViewId::SOURCE), 0.0, F::CLEAN).unwrap();
    assert_eq!(cascade.search(&terms), result);
    assert!(cascade.retain_by_quality(&SpanQuality::new(ViewId(999)), 0.0, F::CLEAN).is_err());
    assert_eq!(cascade.search(&terms), result, "failed configuration is atomic");
}

#[test]
fn rejected_units_never_reach_the_contextual_vector_stage() {
    let doc = document();
    let levels = doc.get::<Levels>().unwrap();
    // These deterministic vectors test cascade control flow, with no claim
    // about trained model accuracy or inference latency.
    let static_rows = rows(&doc, &["Cats", "Dogs"], "static");
    let contextual_rows = rows(&doc, &["Cats", "Dogs", "mice"], "contextual");
    let (sm, cm) = (model("static"), model("contextual"));
    let mut cascade = Cascade::new(
        levels,
        CascadeConfig::default(),
        Some(VectorSpace { rows: &static_rows, query: QueryVector { model: &sm, vector: &[1.0, 0.0] } }),
        Some(VectorSpace { rows: &contextual_rows, query: QueryVector { model: &cm, vector: &[0.0, 0.0, 1.0] } }),
        None,
    )
    .unwrap();
    assert!(cascade.search(&[]).contextual_scored > 0);
    let mut quality = SpanQuality::new(ViewId::SOURCE);
    quality.push(Span::new(0, TEXT.len() as u32), QualityMetrics { composite_score: 0.1, ..Default::default() });
    cascade.retain_by_quality(&quality, 0.5, F::CLEAN).unwrap();
    let out = cascade.search(&[]);
    assert_eq!(out.pool, 0);
    assert_eq!(out.contextual_scored, 0);
    assert!(out.hits.is_empty());
}

#[test]
fn good_coverage_retains_candidates_and_forbidden_flags_remove_them_even_at_zero_floor() {
    let doc = document();
    let levels = doc.get::<Levels>().unwrap();
    let collection = LocalCollection::new(doc.get::<TermStats<Levels>>().unwrap(), doc.get::<Vocabulary>().unwrap());
    let mut cascade =
        Cascade::new(levels, CascadeConfig::default(), None, None, Some(lexical(&doc, &collection))).unwrap();
    let baseline = cascade.search(&["Cats", "Dogs"]);
    assert!(!baseline.hits.is_empty());
    let mut quality = SpanQuality::new(ViewId::SOURCE);
    quality.push(Span::new(0, TEXT.len() as u32), QualityMetrics { composite_score: 1.0, ..Default::default() });
    cascade.retain_by_quality(&quality, 1.0, F::MOJIBAKE_SUSPECT).unwrap();
    assert_eq!(cascade.search(&["Cats", "Dogs"]), baseline);
    quality.metrics[0].flags = F::MOJIBAKE_SUSPECT;
    cascade.retain_by_quality(&quality, 0.0, F::MOJIBAKE_SUSPECT).unwrap();
    let result = cascade.search(&["Cats", "Dogs"]);
    assert!(result.hits.is_empty());
    assert_eq!(result.pool, 0);
}
