//! Stage-3 filters built from `ChunkFacets` (#857): incivility
//! thresholds drop chunks, aspect/polarity filters require and
//! boost, and mismatched facet layers are refused.

use super::*;
use turbo_analysis_layers::{Chunks, IncivilityLabel, IncivilityScores, Polarity, Strs};
use turbo_chunker::ChunkFacets;

/// Facets for `document()`: Chunks row 1 (the first level chunk;
/// row 0 is the heading) is civil and positive about "care";
/// every other row is maximally uncivil.
fn facets(doc: &Document) -> ChunkFacets {
    let chunks = doc.get::<Chunks>().unwrap();
    let n = chunks.len();
    assert!(n >= 2, "fixture holds several chunks: {n}");
    let uncivil = IncivilityScores { insult: 0.9, ..Default::default() };
    let aspect_row: Vec<u32> = (0..=n).map(|i| u32::from(i > 1)).collect();
    ChunkFacets {
        polarity: vec![Polarity::Positive; n],
        polarity_score: vec![0.5; n],
        aspect_row,
        aspect: vec![Ix::new(0)],
        aspect_label: {
            let mut labels = Strs::default();
            labels.push("care");
            labels
        },
        aspect_polarity: vec![Polarity::Positive],
        incivility_row: vec![0; n + 1],
        incivility_label: Vec::new(),
        incivility_max: (0..n).map(|i| if i == 1 { IncivilityScores::default() } else { uncivil }).collect(),
    }
}

fn run(doc: &Document, filters: Vec<UnitFilter>) -> turbo_chunker::CascadeResult {
    let levels = doc.get::<Levels>().unwrap();
    let stats = doc.get::<TermStats<Levels>>().unwrap();
    let vocab = doc.get::<Vocabulary>().unwrap();
    let collection = LocalCollection::new(stats, vocab);
    let static_rows = rows(doc, &["Cats", "Dogs"], "static");
    let contextual_rows = rows(doc, &["Cats", "Dogs", "mice"], "contextual");
    let (sm, cm) = (model("static"), model("contextual"));
    Cascade::new(
        levels,
        CascadeConfig { candidates: 64, top_k: 10, filters, ..CascadeConfig::default() },
        Some(VectorSpace { rows: &static_rows, query: QueryVector { model: &sm, vector: &[1.0, 0.0] } }),
        Some(VectorSpace { rows: &contextual_rows, query: QueryVector { model: &cm, vector: &[0.0, 0.0, 1.0] } }),
        Some(lexical(doc, &collection)),
    )
    .unwrap()
    .search(&["cats"])
}

#[test]
fn incivility_threshold_drops_uncivil_chunks() {
    let doc = document();
    let levels = doc.get::<Levels>().unwrap();
    let facets = facets(&doc);
    facets.check().unwrap();
    let civil = UnitFilter::incivility_at_most("civil", levels, &facets, 0.5).unwrap();
    assert_eq!(civil.units.len(), 1, "only the civil chunk unit is named");
    let out = run(&doc, vec![civil.clone()]);
    assert!(!out.hits.is_empty() && out.hits.iter().all(|h| civil.units.contains(&h.unit)));
    assert!(UnitFilter::incivility_at_most("bad", levels, &facets, 1.5).is_err());
    assert!(UnitFilter::incivility_at_most("bad", levels, &facets, f64::NAN).is_err());
}

#[test]
fn aspect_filter_requires_and_boosts_matching_chunks() {
    let doc = document();
    let levels = doc.get::<Levels>().unwrap();
    let facets = facets(&doc);
    let want =
        UnitFilter::aspect_is("care+", FilterMode::Require, levels, &facets, "care", Some(Polarity::Positive)).unwrap();
    let miss =
        UnitFilter::aspect_is("care-", FilterMode::Require, levels, &facets, "care", Some(Polarity::Negative)).unwrap();
    assert_eq!(want.units.len(), 1);
    assert!(miss.units.is_empty());
    let out = run(&doc, vec![want]);
    assert!(!out.hits.is_empty());
    let boost = UnitFilter::aspect_is("care+", FilterMode::Boost(4.0), levels, &facets, "care", None).unwrap();
    let out = run(&doc, vec![boost]);
    assert_eq!(out.hits[0].boosted_by, ["care+"]);
}

#[test]
fn mismatched_facets_are_refused_without_panicking() {
    let doc = document();
    let levels = doc.get::<Levels>().unwrap();
    let mut short = facets(&doc);
    short.polarity.truncate(1);
    short.polarity_score.truncate(1);
    short.incivility_max.truncate(1);
    short.aspect_row = vec![0, 1];
    short.incivility_row = vec![0, 0];
    let err = UnitFilter::incivility_at_most("civil", levels, &short, 0.5).expect_err("too few rows");
    assert!(err.contains("rows"), "{err}");
    let _ = IncivilityLabel::Insult;
}

#[test]
fn per_dimension_and_sarcasm_filters_from_layers() {
    use turbo_analysis_layers::Pragmatics;
    let doc = document();
    let levels = doc.get::<Levels>().unwrap();
    let chunks = doc.get::<Chunks>().unwrap();
    let facets = facets(&doc);
    // Only the insult dimension is high in the fixture, so a
    // threat-dimension cap keeps every chunk unit.
    let threat =
        UnitFilter::incivility_label_at_most("no-threats", levels, &facets, IncivilityLabel::Threat, 0.5).unwrap();
    let insult =
        UnitFilter::incivility_label_at_most("no-insults", levels, &facets, IncivilityLabel::Insult, 0.5).unwrap();
    assert!(threat.units.len() > insult.units.len());
    assert_eq!(insult.units.len(), 1);

    let sentences = doc.get::<Sentences>().unwrap();
    let kept = chunks.sentences[1];
    let pragmatics = Pragmatics {
        score: (0..sentences.len())
            .map(|s| if (kept.start.get()..kept.end.get()).contains(&s) { 0.0 } else { 0.95 })
            .collect(),
        evidence: vec![None; sentences.len()],
        tokens_key: Key::<Tokens>::DEFAULT,
    };
    let sincere =
        UnitFilter::sarcasm_at_most("sincere", FilterMode::Require, levels, chunks, &pragmatics, 0.5, true).unwrap();
    assert_eq!(sincere.units.len(), 1);
    let out = run(&doc, vec![sincere.clone()]);
    assert!(!out.hits.is_empty() && out.hits.iter().all(|h| sincere.units.contains(&h.unit)));
    let short = Pragmatics { score: vec![0.0], evidence: vec![None], tokens_key: Key::<Tokens>::DEFAULT };
    assert!(UnitFilter::sarcasm_at_most("sincere", FilterMode::Require, levels, chunks, &short, 0.5, true).is_err());
    assert!(
        UnitFilter::sarcasm_at_most("sincere", FilterMode::Boost(0.5), levels, chunks, &pragmatics, 2.0, false)
            .is_err()
    );
}
