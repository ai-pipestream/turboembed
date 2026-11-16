//! #2388: a `city` candidate of the PII head (#1888) on a name the person
//! rules (#2295) read as a person's (`we met Madison`, `Salem, who`) is
//! turned down, as the bare reading of the rules already was. A head city
//! with no person word around it, after a place word, before a state or
//! after a city label stays, and so does every other kind of the head.

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

const REFUSED: PiiDisposition = PiiDisposition::RejectedByRule { reason: RejectReason::RulePrecedence };

/// The mentions of `text` under `rules` and one head candidate of `kind`
/// over `name` at 0.9, as (covered, kind name), and the candidates'
/// (covered, kind name, disposition).
#[allow(clippy::type_complexity)]
fn resolve(
    rules: Option<&PiiDetector>,
    text: &str,
    name: &str,
    kind: &str,
) -> (Vec<(String, String)>, Vec<(String, String, PiiDisposition)>) {
    let kinds = nemotron_custom_kinds();
    let kind = PiiKind::parse(kind).or_else(|| kinds.find(kind)).unwrap();
    let opts =
        ModelResolution { rules, model_kinds: &kinds, provenance: PROVENANCE, min_score: Some(0.5), extra_allow: None };
    let at = text.find(name).unwrap();
    let span = Span::new(at as u32, (at + name.len()) as u32);
    let model = [ModelCandidate { span, kind, probability: 0.9 }];
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

/// A person who shares a city's name, with the person word the rules see.
const PEOPLE: [(&str, &str); 12] = [
    ("We met Madison at the front desk on Monday.", "Madison"),
    ("Later we asked Columbus to sign the visitor log.", "Columbus"),
    ("Then we thanked Dayton for the flowers.", "Dayton"),
    ("Salem, who runs the Tuesday class, moved the session.", "Salem"),
    ("Paris's mother called to say the bus was late.", "Paris"),
    ("Paris\u{2019}s mother called to say the bus was late.", "Paris"),
    ("Madison herself drove the van.", "Madison"),
    ("Dayton knew she would be late.", "Dayton"),
    ("So Columbus thought he could cover the shift.", "Columbus"),
    ("Madison said the bus was late.", "Madison"),
    ("Ask Dayton about the key.", "Dayton"),
    ("Tomorrow.\nRemind Paris about the forms.", "Paris"),
];

#[test]
fn a_head_city_on_a_name_the_person_rules_refuse_is_no_city() {
    let rules = PiiDetector::pack(Pack::Places);
    for (text, name) in PEOPLE {
        let alone: Vec<PiiKind> = rules.layer(text, ViewId::SOURCE).mentions().map(|m| m.kind).collect();
        assert_eq!(alone, [], "{text:?}: the rules alone read no city");
        let (mentions, rows) = resolve(Some(&rules), text, name, "city");
        assert_eq!(mentions, [], "{text:?}");
        // The head's candidate keeps its row, with the rules' decision.
        assert_eq!(rows, [(name.to_owned(), "city".to_owned(), REFUSED)], "{text:?}");
    }
}

#[test]
fn a_head_city_with_no_person_word_around_it_stays() {
    let rules = PiiDetector::pack(Pack::Places);
    for (text, name) in [
        // A place word before the name outweighs the words after it.
        ("A donor in Columbus who funds the program wrote.", "Columbus"),
        ("We heard from people in Paris who wrote letters.", "Paris"),
        ("A source in Boston said so.", "Boston"),
        ("People in\nParis who wrote letters came too.", "Paris"),
        // Verbs that take a place as readily, and other words.
        ("We visited Madison last spring.", "Madison"),
        ("They left Dayton then she moved north.", "Dayton"),
        ("The youth trip went to Paris in May.", "Paris"),
        // Names the bare reading of the rules does not hold at all.
        ("We drove through Xenia on the way.", "Xenia"),
        ("The office in Yellow Springs who hired her closed.", "Yellow Springs"),
        ("Paris's museums were full.", "Paris"),
    ] {
        let (mentions, rows) = resolve(Some(&rules), text, name, "city");
        assert_eq!(mentions, [(name.to_owned(), "city".to_owned())], "{text:?}");
        assert!(rows.iter().all(|r| r.2 != REFUSED), "{text:?}: {rows:?}");
    }
}

#[test]
fn a_head_city_before_a_state_or_after_a_label_stays() {
    let rules = PiiDetector::pack(Pack::Places);
    for text in [
        "Later we asked Columbus, Ohio to send the forms.",
        "We met the city of Columbus who signed it.",
        "City: Columbus, who knew",
    ] {
        let (mentions, _) = resolve(Some(&rules), text, "Columbus", "city");
        assert!(mentions.contains(&("Columbus".to_owned(), "city".to_owned())), "{text:?}: {mentions:?}");
    }
}

#[test]
fn only_a_city_of_the_head_is_turned_down() {
    let rules = PiiDetector::pack(Pack::Places);
    for kind in ["first-name", "last-name", "company-name", "county"] {
        let (mentions, _) = resolve(Some(&rules), "We met Madison at the front desk.", "Madison", kind);
        assert_eq!(mentions, [("Madison".to_owned(), kind.to_owned())]);
    }
}

#[test]
fn without_the_city_scanner_the_head_city_stands() {
    let text = "We met Madison at the front desk.";
    let structured = PiiDetector::pack(Pack::AllStructured);
    for rules in [None, Some(&structured)] {
        let (mentions, _) = resolve(rules, text, "Madison", "city");
        assert_eq!(mentions, [("Madison".to_owned(), "city".to_owned())]);
    }
}

#[test]
fn a_head_city_under_the_threshold_keeps_the_threshold_decision() {
    let rules = PiiDetector::pack(Pack::Places);
    let kinds = nemotron_custom_kinds();
    let opts = ModelResolution {
        rules: Some(&rules),
        model_kinds: &kinds,
        provenance: PROVENANCE,
        min_score: Some(0.5),
        extra_allow: None,
    };
    let text = "We met Madison at the front desk.";
    let model = [ModelCandidate { span: Span::new(7, 14), kind: PiiKind::City, probability: 0.3 }];
    let (pii, candidates) = resolve_model_with_rules(text, ViewId::SOURCE, &opts, &model).unwrap();
    assert_eq!(pii.mentions().count(), 0);
    let rows: Vec<PiiDisposition> = candidates.candidates().map(|c| c.disposition).collect();
    assert_eq!(rows, [PiiDisposition::RejectedByRule { reason: RejectReason::Threshold }]);
}
