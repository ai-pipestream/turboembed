//! #2312: a bare city name inside a person span of the PII head (#1888)
//! is the person's (`Paris won the race`). A city before a state or after
//! a city label stays, as does one inside a span of another kind, under
//! the score threshold, or found by another rule as well.

use turbo_analysis::{Layer, Span, ViewId};
use turbo_analysis_layers::{CustomKinds, PiiDisposition, PiiEvidence, PiiKind, RejectReason};
use turbo_pii::nemotron::{ModelCandidate, ModelProvenance, ModelResolution, resolve_model_with_rules};
use turbo_pii::{Pack, PiiDetector, PiiSink, Scanner, nemotron_custom_kinds};

const PROVENANCE: ModelProvenance<'static> = ModelProvenance {
    detector_id: "turbo-tagger:pii",
    detector_version: "0.1.0",
    model_id: "turbo-en-nemotron-pii-modernbert-base-1",
    model_version: "1",
    calibrated: true,
};

fn span_of(text: &str, needle: &str) -> Span {
    let s = text.find(needle).unwrap();
    Span::new(s as u32, (s + needle.len()) as u32)
}

/// The mentions of `text` under the Places pack and one model candidate
/// of `kind` over `name` at `probability`, as (covered, kind name), and
/// the candidates' (covered, kind name, disposition).
#[allow(clippy::type_complexity)]
fn resolve(
    rules: &PiiDetector,
    text: &str,
    name: &str,
    kind: &str,
    probability: f32,
) -> (Vec<(String, String)>, Vec<(String, String, PiiDisposition)>) {
    let kinds: CustomKinds = nemotron_custom_kinds();
    let kind = kinds.resolve(kind).unwrap();
    let opts = ModelResolution {
        rules: Some(rules),
        model_kinds: &kinds,
        provenance: PROVENANCE,
        min_score: Some(0.5),
        extra_allow: None,
    };
    let model = [ModelCandidate { span: span_of(text, name), kind, probability }];
    let (pii, candidates) = resolve_model_with_rules(text, ViewId::SOURCE, &opts, &model).unwrap();
    pii.check().unwrap();
    candidates.check().unwrap();
    candidates.check_against(&pii).unwrap();
    let mentions =
        pii.mentions().map(|m| (text[m.span.range()].to_owned(), pii.kind_name(m.kind).unwrap().to_owned())).collect();
    let rows = candidates
        .candidates()
        .map(|c| (text[c.span.range()].to_owned(), candidates.kind_name(c.kind).unwrap().to_owned(), c.disposition))
        .collect();
    (mentions, rows)
}

fn owned(rows: &[(&str, &str)]) -> Vec<(String, String)> {
    rows.iter().map(|&(a, b)| (a.to_owned(), b.to_owned())).collect()
}

#[test]
fn the_rules_alone_read_a_bare_subject_as_the_city() {
    let rules = PiiDetector::pack(Pack::Places);
    let text = "Paris won the race on Saturday.";
    let found: Vec<PiiKind> = rules.layer(text, ViewId::SOURCE).mentions().map(|m| m.kind).collect();
    assert_eq!(found, [PiiKind::City]);
}

#[test]
fn a_bare_city_inside_a_person_span_is_the_person() {
    let rules = PiiDetector::pack(Pack::Places);
    for (text, name, kind) in [
        ("Paris won the race on Saturday.", "Paris", "first-name"),
        ("After lunch we saw Madison at the desk.", "Madison", "first-name"),
        ("They called Salem twice before noon.", "Salem", "last-name"),
    ] {
        let (mentions, rows) = resolve(&rules, text, name, kind, 0.9);
        assert_eq!(mentions, owned(&[(name, kind)]), "{text:?}");
        // The city candidate is taken out: only the person's row is left.
        assert_eq!(rows.len(), 1, "{text:?}: {rows:?}");
        assert!(matches!(rows[0].2, PiiDisposition::Kept { .. }), "{text:?}: {rows:?}");
    }
}

#[test]
fn a_city_before_a_state_or_after_a_label_stays_a_city() {
    let rules = PiiDetector::pack(Pack::Places);
    for text in [
        "Columbus, Ohio won the bid for the fair.",
        "City: Columbus",
        "Place of birth:   Columbus",
        "She works for the city of Columbus now.",
    ] {
        let (mentions, rows) = resolve(&rules, text, "Columbus", "first-name", 0.9);
        assert!(mentions.contains(&("Columbus".to_owned(), "city".to_owned())), "{text:?}: {mentions:?}");
        assert!(!mentions.iter().any(|(_, k)| k == "first-name"), "{text:?}: {mentions:?}");
        let person = rows.iter().find(|r| r.1 == "first-name").unwrap();
        assert_eq!(person.2, PiiDisposition::RejectedByRule { reason: RejectReason::RulePrecedence }, "{text:?}");
    }
}

#[test]
fn a_span_under_the_threshold_or_of_another_kind_leaves_the_city() {
    let rules = PiiDetector::pack(Pack::Places);
    let text = "Paris won the race on Saturday.";
    let (mentions, _) = resolve(&rules, text, "Paris", "first-name", 0.3);
    assert_eq!(mentions, owned(&[("Paris", "city")]));
    let (mentions, _) = resolve(&rules, text, "Paris", "company-name", 0.9);
    assert_eq!(mentions, owned(&[("Paris", "city")]));
}

/// A caller's scanner that reads every `Paris` as a city.
fn every_paris(text: &str, sink: &mut PiiSink<'_>) {
    for (at, _) in text.match_indices("Paris") {
        sink.add_text(Span::new(at as u32, (at + 5) as u32), PiiKind::City, &[PiiEvidence::Shape]);
    }
}

#[test]
fn a_city_another_rule_finds_on_the_same_span_stays() {
    let mut scanners = Pack::Places.scanners();
    scanners.push(Scanner::custom_fn("VenueExtractor", PiiKind::City, every_paris));
    let rules = PiiDetector::new(scanners);
    let text = "Paris won the race on Saturday.";
    let (mentions, _) = resolve(&rules, text, "Paris", "first-name", 0.9);
    assert_eq!(mentions, owned(&[("Paris", "city")]));
}

#[test]
fn without_the_city_scanner_nothing_changes() {
    let rules = PiiDetector::pack(Pack::AllStructured);
    let text = "Paris won the race on Saturday.";
    let (mentions, rows) = resolve(&rules, text, "Paris", "first-name", 0.9);
    assert_eq!(mentions, owned(&[("Paris", "first-name")]));
    assert_eq!(rows.len(), 1);
}

#[test]
fn only_a_person_span_around_the_whole_name_overrules_the_city() {
    let rules = PiiDetector::pack(Pack::Places);
    let text = "We heard that Salt Lake City won the vote.";
    let found: Vec<PiiKind> = rules.layer(text, ViewId::SOURCE).mentions().map(|m| m.kind).collect();
    assert_eq!(found, [PiiKind::City], "the rules alone read the bare name");
    // A span on part of the name, or one that starts on one side of it and
    // ends inside it, leaves the city standing.
    for part in ["Salt", "Lake", "that Salt", "City won"] {
        let (mentions, _) = resolve(&rules, text, part, "first-name", 0.9);
        assert!(mentions.contains(&("Salt Lake City".to_owned(), "city".to_owned())), "{part:?}: {mentions:?}");
    }
    // A span around the whole name, exactly or wider, overrules it.
    for whole in ["Salt Lake City", "that Salt Lake City won"] {
        let (mentions, _) = resolve(&rules, text, whole, "first-name", 0.9);
        assert_eq!(mentions, owned(&[(whole, "first-name")]), "{whole:?}");
    }
}
