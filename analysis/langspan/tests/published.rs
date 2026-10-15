//! The span detector over OpenNLP's `langdetect-183.bin` on the mixed
//! language fixtures, with the scores it gets recorded here, so a change
//! to the detector shows what it does to them. The model is not checked
//! in: without it these tests say so and pass, unless
//! `TURBO_TEST_REQUIRE_OPENNLP_MODELS=1`, which CI sets.

mod common;

use std::slice;

use turbo_analysis::{Analyzer, Document};
use turbo_analysis_layers::{Language, LanguageSpans};
use turbo_langspan::gold::{GoldDocument, Scores, parse};
use turbo_langspan::scripts::written_in;
use turbo_langspan::{Run, Settings, SpanDetector, SpanScratch};

use common::published_detector;

fn fixtures(name: &str) -> Vec<GoldDocument> {
    let path = format!("{}/tests/data/{name}", env!("CARGO_MANIFEST_DIR"));
    parse(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn score(detector: &SpanDetector, docs: &[GoldDocument]) -> Scores {
    let mut scratch = SpanScratch::new();
    let mut runs: Vec<Run> = Vec::new();
    let mut scores = Scores::default();
    for doc in docs {
        detector.detect(&doc.text, None, None, &mut scratch, &mut runs);
        let predicted: Vec<_> = runs.iter().map(|r| (r.span, detector.code(r))).collect();
        scores.add(doc, &predicted);
    }
    scores
}

#[test]
fn the_fixtures_are_well_formed_and_cover_many_languages() {
    let dev = fixtures("mixed-dev.txt");
    let test = fixtures("mixed-test.txt");
    assert_eq!((dev.len(), test.len()), (30, 40));
    let mut codes: Vec<&str> = dev.iter().chain(&test).flat_map(|d| d.spans.iter().map(|(_, c)| c.as_str())).collect();
    codes.sort_unstable();
    codes.dedup();
    assert_eq!(codes.len(), 28, "{codes:?}");
    // No document is in both sets.
    for d in &dev {
        assert!(test.iter().all(|t| t.text != d.text && t.id != d.id), "{}", d.id);
    }
}

#[test]
fn every_language_of_the_model_has_its_scripts_in_the_table() {
    let Some(detector) = published_detector() else { return };
    let codes: Vec<&str> = detector.model().supported_languages().collect();
    assert_eq!(codes.len(), 103);
    let missing: Vec<&str> = codes.iter().copied().filter(|c| written_in(c).is_none()).collect();
    assert!(missing.is_empty(), "{missing:?}");
}

/// The development set, which the settings were chosen on.
#[test]
fn the_development_set_scores_as_recorded() {
    let Some(detector) = published_detector() else { return };
    let scores = score(&SpanDetector::new(detector, Settings::default()), &fixtures("mixed-dev.txt"));
    let expected = Scores {
        documents: 30,
        exact_documents: 30,
        gold_spans: 67,
        correct_spans: 67,
        gold_boundaries: 37,
        predicted_boundaries: 37,
        matched_boundaries: 37,
        letters: 3142,
        correct_letters: 3142,
        correct_document_languages: 30,
    };
    assert_eq!(scores, expected, "\n{scores}");
}

/// The held-out set, scored once the settings were fixed. The misses: a
/// short English stretch next to Japanese or Hindi and a three-word
/// French clause read as Waray, a one-word quotation (`Schadenfreude`)
/// and two short quotations stay with the surrounding language, and
/// Indonesian reads as Sundanese.
#[test]
fn the_held_out_set_scores_as_recorded() {
    let Some(detector) = published_detector() else { return };
    let scores = score(&SpanDetector::new(detector, Settings::default()), &fixtures("mixed-test.txt"));
    let expected = Scores {
        documents: 40,
        exact_documents: 33,
        gold_spans: 87,
        correct_spans: 81,
        gold_boundaries: 47,
        predicted_boundaries: 42,
        matched_boundaries: 41,
        letters: 4020,
        correct_letters: 3845,
        correct_document_languages: 40,
    };
    assert_eq!(scores, expected, "\n{scores}");
}

#[test]
fn in_a_pipeline_the_spans_and_the_summary_agree() {
    let Some(detector) = published_detector() else { return };
    let analyzer = Analyzer::builder().add(SpanDetector::new(detector, Settings::default())).build().unwrap();
    let text = "Gestern hat mir mein Kollege geschrieben, und am Ende stand nur: \"Let me know if you need anything \
                else from our side.\" Ich habe ihm sofort geantwortet und mich bedankt.";
    let mut doc = Document::new(text);
    analyzer.run(slice::from_mut(&mut doc)).remove(0).unwrap();
    let spans = doc.get::<LanguageSpans>().unwrap();
    assert_eq!(spans.iter().map(|s| s.code).collect::<Vec<_>>(), ["deu", "eng", "deu"]);
    assert_eq!(spans.iter().next().unwrap().span.start, 0);
    assert_eq!(spans.iter().last().unwrap().span.end as usize, text.len());
    let language = doc.get::<Language>().unwrap();
    assert_eq!(language.iter().map(|l| l.0).collect::<Vec<_>>(), ["deu", "eng"]);
    assert_eq!(language.best().unwrap().0, "deu");
}
