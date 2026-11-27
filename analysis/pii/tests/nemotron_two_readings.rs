//! #2398: a name the PII head (#1888) reads two ways comes as two
//! candidates on one span, `city` and a person's name, each with its
//! own probability. Both keep a row in the candidates layer with their
//! score, the threshold decides which become mentions, and the less
//! likely reading never takes the place of the likelier one.

use turbo_analysis::{Ix, Layer, Span, ViewId};
use turbo_analysis_layers::{PiiDisposition, PiiEvidence, PiiKind, PiiScoreKind, RejectReason};
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

#[test]
fn suppressed_places_hit_and_both_scored_readings_remain_auditable() {
    let rules = PiiDetector::pack(Pack::Places);
    let kinds = nemotron_custom_kinds();
    let first = kinds.find("first-name").unwrap();
    let start = TEXT.find("Paris").unwrap() as u32;
    let span = Span::new(start, start + 5);
    let model = [
        ModelCandidate { span, kind: first, probability: 0.70 },
        ModelCandidate { span, kind: PiiKind::City, probability: 0.20 },
    ];
    let opts = ModelResolution {
        rules: Some(&rules),
        model_kinds: &kinds,
        provenance: PROVENANCE,
        min_score: Some(0.5),
        extra_allow: None,
    };
    let (pii, audit) = resolve_model_with_rules(TEXT, ViewId::SOURCE, &opts, &model).unwrap();
    audit.check_against(&pii).unwrap();
    let mentions: Vec<_> = pii.mentions().map(|m| (m.span, pii.kind_name(m.kind).unwrap())).collect();
    assert_eq!(mentions, [(span, "first-name")], "only the same person is admitted");

    let rows: Vec<_> = audit.candidates().collect();
    // Independent Places already emitted the unscored city. The merged
    // result should report why that row lost instead of erasing it.
    let (_, independent) = rules.layer_with_candidates(TEXT, ViewId::SOURCE);
    let city = independent.candidates().find(|c| c.span == span && c.kind == PiiKind::City).unwrap();
    let source = independent.detectors.get(city.detector);
    assert_eq!(source.map(|(id, _)| id), Some("turbo-pii:city"));
    let rejected_city = rows.iter().find(|r| audit.detectors.get(r.detector) == source).expect("missing Places row");
    assert_eq!(
        (rejected_city.span, rejected_city.kind, rejected_city.evidence, rejected_city.score),
        (span, PiiKind::City, city.evidence, None)
    );
    assert_eq!(rejected_city.disposition, PiiDisposition::RejectedByRule { reason: RejectReason::RulePrecedence });
    assert_eq!(rows.len(), 3);

    for (kind, probability, disposition) in
        [(first, 0.70f32, PiiDisposition::Kept { mention: Ix::new(0) }), (PiiKind::City, 0.20f32, UNDER)]
    {
        let row = rows.iter().find(|r| r.kind == kind && r.score.is_some()).unwrap();
        assert_eq!(row.span, span);
        assert_eq!(row.evidence, &[PiiEvidence::Shape]);
        assert_eq!(row.disposition, disposition);
        assert_eq!(audit.detectors.get(row.detector), Some((PROVENANCE.detector_id, PROVENANCE.detector_version)));
        let score = row.score.unwrap();
        assert_eq!(score.value.to_bits(), probability.to_bits());
        assert_eq!(score.kind, PiiScoreKind::Probability);
        assert_eq!(score.calibrated, PROVENANCE.calibrated);
        assert_eq!(audit.detectors.get(score.model), Some((PROVENANCE.model_id, PROVENANCE.model_version)));
    }
}
