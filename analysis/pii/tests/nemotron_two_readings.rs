//! #2398: a name the PII head (#1888) reads two ways comes as two
//! candidates on one span, `city` and a person's name, each with its
//! own probability. Both keep a row in the candidates layer with their
//! score, the threshold decides which become mentions, and the less
//! likely reading never takes the place of the likelier one.

use turbo_analysis::{Layer, Span, ViewId};
use turbo_analysis_layers::{PiiDisposition, PiiKind, RejectReason};
use turbo_pii::nemotron::{ModelCandidate, ModelProvenance, ModelResolution, resolve_model_with_rules};
use turbo_pii::{Pack, PiiDetector, nemotron_custom_kinds};

const PROVENANCE: ModelProvenance<'static> = ModelProvenance {
    detector_id: "turbo-tagger:pii",
    detector_version: "0.1.0",
    model_id: "turbo-en-nemotron-pii-modernbert-base-1",
    model_version: "1",
    calibrated: true,
};

const UNDER: PiiDisposition = PiiDisposition::RejectedByRule { reason: RejectReason::Threshold };

/// The mentions as (kind name, score) and the scored rows as (kind name,
/// score, disposition) of `text` with a `city` and a `first-name`
/// candidate of the head on `Paris`.
#[allow(clippy::type_complexity)]
fn read(
    rules: Option<&PiiDetector>,
    text: &str,
    (city, person): (f32, f32),
    min_score: Option<f32>,
) -> (Vec<(String, Option<f32>)>, Vec<(String, f32, PiiDisposition)>) {
    let kinds = nemotron_custom_kinds();
    let opts = ModelResolution { rules, model_kinds: &kinds, provenance: PROVENANCE, min_score, extra_allow: None };
    let at = text.find("Paris").unwrap() as u32;
    let span = Span::new(at, at + 5);
    let first_name = kinds.find("first-name").unwrap();
    let mut model = vec![
        ModelCandidate { span, kind: PiiKind::City, probability: city },
        ModelCandidate { span, kind: first_name, probability: person },
    ];
    // The tagger gives the likelier reading first.
    if person > city {
        model.reverse();
    }
    let (pii, candidates) = resolve_model_with_rules(text, ViewId::SOURCE, &opts, &model).unwrap();
    pii.check().unwrap();
    candidates.check().unwrap();
    candidates.check_against(&pii).unwrap();
    let mentions = pii
        .mentions()
        .filter(|m| m.span == span)
        .map(|m| (pii.kind_name(m.kind).unwrap().to_owned(), m.score.map(|s| s.value)))
        .collect();
    let rows = candidates
        .candidates()
        .filter(|c| c.span == span)
        .filter_map(|c| Some((candidates.kind_name(c.kind).unwrap().to_owned(), c.score?.value, c.disposition)))
        .collect();
    (mentions, rows)
}

const TEXT: &str = "Names alone: Paris won the race on Saturday.";

#[test]
fn both_readings_keep_a_row_with_their_own_score() {
    let rules = PiiDetector::pack(Pack::Places);
    for rules in [None, Some(&rules)] {
        let (mentions, rows) = read(rules, TEXT, (0.72, 0.12), Some(0.5));
        assert_eq!(mentions.len(), 1, "{mentions:?}");
        assert_eq!(mentions[0].0, "city");
        assert_eq!(rows.len(), 2, "{rows:?}");
        assert_eq!((rows[0].0.as_str(), rows[0].1), ("city", 0.72));
        assert!(matches!(rows[0].2, PiiDisposition::Kept { .. } | PiiDisposition::Repeats { .. }), "{rows:?}");
        assert_eq!((rows[1].0.as_str(), rows[1].1, rows[1].2), ("first-name", 0.12, UNDER));
    }
}

#[test]
fn a_person_with_a_city_as_second_reading_keeps_both_rows() {
    let rules = PiiDetector::pack(Pack::Places);
    for rules in [None, Some(&rules)] {
        let (mentions, rows) = read(rules, TEXT, (0.2, 0.7), Some(0.5));
        assert_eq!(mentions, [("first-name".to_owned(), Some(0.7))]);
        // Rows come in the candidates layer's order, the city's kind first.
        let scored: Vec<(&str, f32)> = rows.iter().map(|r| (r.0.as_str(), r.1)).collect();
        assert_eq!(scored, [("city", 0.2), ("first-name", 0.7)]);
        assert_eq!(rows[0].2, UNDER);
    }
}

/// With no threshold the head's readings both stand, and the second
/// reading of a city does not take the rules' bare city away as a person
/// span of the head would (#2312).
#[test]
fn the_second_reading_is_no_person_span_that_drops_the_city() {
    let rules = PiiDetector::pack(Pack::Places);
    let (mentions, rows) = read(Some(&rules), TEXT, (0.72, 0.12), None);
    assert_eq!(mentions, [("city".to_owned(), None)], "the rules' city stands");
    assert_eq!(rows.len(), 2, "{rows:?}");
    let (mentions, _) = read(None, TEXT, (0.72, 0.12), None);
    let kinds: Vec<&str> = mentions.iter().map(|m| m.0.as_str()).collect();
    assert_eq!(kinds.len(), 2, "both readings are mentions with no threshold: {kinds:?}");
    assert!(kinds.contains(&"city") && kinds.contains(&"first-name"));
    // A threshold under both gives the same.
    let (mentions, _) = read(None, TEXT, (0.72, 0.12), Some(0.1));
    assert_eq!(mentions.len(), 2);
}
