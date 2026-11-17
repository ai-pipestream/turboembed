//! #2398: a name the text uses for a person at one place (`Paris's
//! mother`, `we asked Columbus`) is a person's where it stands alone in
//! the same text (`Paris won the race`), unless the PII head (#1888) is
//! sure of its `city` there or a place word stands before the name. The
//! bare reading of the rules goes with the head's city. A name with no
//! person use in the text, a city the head is sure of, one after a place
//! word and one before a state all stay.

use turbo_analysis::{Layer, Span, ViewId};
use turbo_analysis_layers::{PiiDisposition, PiiKind, RejectReason};
use turbo_pii::nemotron::{
    CITY_OF_PERSON_NAME_MIN, ModelCandidate, ModelProvenance, ModelResolution, resolve_model_with_rules,
};
use turbo_pii::{Pack, PiiDetector, nemotron_custom_kinds};

const PROVENANCE: ModelProvenance<'static> = ModelProvenance {
    detector_id: "turbo-tagger:pii",
    detector_version: "0.1.0",
    model_id: "turbo-en-nemotron-pii-modernbert-base-1",
    model_version: "1",
    calibrated: true,
};

const REFUSED: PiiDisposition = PiiDisposition::RejectedByRule { reason: RejectReason::RulePrecedence };

/// The city mentions of `text` under `rules` with a head `city` at
/// `probability` on the last use of `name`, and the disposition of that
/// candidate's row.
fn last_use(rules: Option<&PiiDetector>, text: &str, name: &str, probability: f32) -> (Vec<String>, PiiDisposition) {
    let kinds = nemotron_custom_kinds();
    let opts =
        ModelResolution { rules, model_kinds: &kinds, provenance: PROVENANCE, min_score: Some(0.5), extra_allow: None };
    let at = text.rfind(name).unwrap();
    let span = Span::new(at as u32, (at + name.len()) as u32);
    let model = [ModelCandidate { span, kind: PiiKind::City, probability }];
    let (pii, candidates) = resolve_model_with_rules(text, ViewId::SOURCE, &opts, &model).unwrap();
    pii.check().unwrap();
    candidates.check().unwrap();
    candidates.check_against(&pii).unwrap();
    let cities = pii.mentions().filter(|m| m.kind == PiiKind::City).map(|m| text[m.span.range()].to_owned()).collect();
    let row = candidates.candidates().find(|c| c.span == span && c.score.is_some()).unwrap();
    (cities, row.disposition)
}

/// A person use the rules see, then the name standing alone.
const ALONE: [(&str, &str); 4] = [
    ("Paris's mother called about the bus. Names alone: Paris won the race on Saturday.", "Paris"),
    ("Later we asked Columbus to sign the log. We lost, and Columbus drove home.", "Columbus"),
    ("We will invite Salem to the meeting. They called Salem twice.", "Salem"),
    ("Ask Dayton about the key. They rang twice before Dayton answered.", "Dayton"),
];

#[test]
fn a_name_the_text_uses_for_a_person_is_no_city_standing_alone() {
    let rules = PiiDetector::pack(Pack::Places);
    for (text, name) in ALONE {
        // The rules alone read the name standing alone as a city.
        let alone: Vec<&str> = rules.layer(text, ViewId::SOURCE).mentions().map(|m| &text[m.span.range()]).collect();
        assert!(alone.contains(&name), "{text:?}: {alone:?}");
        for probability in [0.5, 0.72, 0.81, CITY_OF_PERSON_NAME_MIN - 0.001] {
            let (cities, row) = last_use(Some(&rules), text, name, probability);
            assert_eq!(cities, Vec::<String>::new(), "{text:?} at {probability}");
            assert_eq!(row, REFUSED, "{text:?} at {probability}");
        }
    }
}

/// Only the use the head is unsure of gives way: a use with no head
/// candidate stays the rules' city, and the person use may come after.
#[test]
fn the_person_use_may_follow_and_other_uses_are_left_alone() {
    let rules = PiiDetector::pack(Pack::Places);
    let text = "Paris won the race on Saturday. Later we thanked Paris for the flowers.\nParis drove home.";
    let (cities, row) = last_use(Some(&rules), text, "Paris", 0.72);
    assert_eq!(cities, ["Paris"]);
    assert_eq!(row, REFUSED);
    let first = rules.layer(text, ViewId::SOURCE).mentions().map(|m| m.span.start).collect::<Vec<_>>();
    assert_eq!(first, [0, text.rfind("Paris").unwrap() as u32], "the rules alone read both");
}

#[test]
fn a_city_the_head_is_sure_of_stays() {
    let rules = PiiDetector::pack(Pack::Places);
    for (text, name) in &ALONE[..4] {
        for probability in [CITY_OF_PERSON_NAME_MIN, 0.9, 0.99] {
            let (cities, row) = last_use(Some(&rules), text, name, probability);
            assert_eq!(cities, [*name], "{text:?} at {probability}");
            assert_ne!(row, REFUSED, "{text:?} at {probability}");
        }
    }
}

#[test]
fn a_city_after_a_place_word_or_before_a_state_stays() {
    let rules = PiiDetector::pack(Pack::Places);
    for (text, name, cities) in [
        ("Paris's mother called about the bus. The youth trip went to Paris in May.", "Paris", vec!["Paris"]),
        ("We asked Columbus to sign. Two staff flew in from\nColumbus for it.", "Columbus", vec!["Columbus"]),
        ("Ask Dayton about the key. The Dayton office still holds the files.", "Dayton", vec!["Dayton"]),
        ("We will invite Salem to the meeting. The old office was in Salem, OR.", "Salem", vec!["Salem"]),
    ] {
        let (got, row) = last_use(Some(&rules), text, name, 0.72);
        assert_eq!(got, cities, "{text:?}");
        assert_ne!(row, REFUSED, "{text:?}");
    }
}

#[test]
fn only_a_person_use_of_the_same_name_counts() {
    let rules = PiiDetector::pack(Pack::Places);
    for (text, name) in [
        // No person use at all.
        ("The relay was close. Paris won the race on Saturday.", "Paris"),
        // Another name's person use.
        ("We met Madison at the front desk. Paris won the race on Saturday.", "Paris"),
        // A longer word that starts with the name, and one that ends with it.
        ("We met Parisa at the front desk. Paris won the race on Saturday.", "Paris"),
        ("We met NewParis at the front desk. Paris won the race on Saturday.", "Paris"),
        // The other use is a place: a place word before it, or a state after it.
        ("People in Paris who wrote letters came. Paris won the bid.", "Paris"),
        ("We met in Salem, OR last year. Salem voted on the measure.", "Salem"),
    ] {
        let (cities, row) = last_use(Some(&rules), text, name, 0.72);
        assert!(cities.iter().any(|c| c == name), "{text:?}: {cities:?}");
        assert_ne!(row, REFUSED, "{text:?}");
    }
}

#[test]
fn without_the_city_scanner_the_head_city_stands() {
    let (text, name) = ALONE[0];
    for rules in [None, Some(PiiDetector::pack(Pack::AllStructured))] {
        let (cities, row) = last_use(rules.as_ref(), text, name, 0.72);
        assert_eq!(cities, [name]);
        assert_ne!(row, REFUSED);
    }
}

#[test]
fn another_kind_of_the_head_on_the_name_is_left_alone() {
    let rules = PiiDetector::pack(Pack::Places);
    let kinds = nemotron_custom_kinds();
    let (text, name) = ALONE[0];
    let opts = ModelResolution {
        rules: Some(&rules),
        model_kinds: &kinds,
        provenance: PROVENANCE,
        min_score: Some(0.5),
        extra_allow: None,
    };
    let at = text.rfind(name).unwrap();
    let span = Span::new(at as u32, (at + name.len()) as u32);
    let model = [ModelCandidate { span, kind: kinds.find("company-name").unwrap(), probability: 0.72 }];
    let (pii, _) = resolve_model_with_rules(text, ViewId::SOURCE, &opts, &model).unwrap();
    let cities: Vec<&str> = pii.mentions().filter(|m| m.kind == PiiKind::City).map(|m| &text[m.span.range()]).collect();
    assert_eq!(cities, [name], "the rules' bare reading stands under another kind of the head");
}

/// Each unsure name costs a pass over the text, so only the first 32 of
/// a text are read; a name past them keeps its city.
#[test]
fn a_text_of_many_unsure_names_keeps_the_cities_past_the_cap() {
    let rules = PiiDetector::pack(Pack::Places);
    let kinds = nemotron_custom_kinds();
    let opts = ModelResolution {
        rules: Some(&rules),
        model_kinds: &kinds,
        provenance: PROVENANCE,
        min_score: Some(0.5),
        extra_allow: None,
    };
    for (made_up, paris_is_city) in [(31, false), (32, true)] {
        let mut text = String::new();
        let mut model = Vec::new();
        for i in 0..made_up {
            let at = text.len() as u32;
            let word = format!("zq{i:02}x");
            text.push_str(&word);
            text.push_str(" came. ");
            model.push(ModelCandidate {
                span: Span::new(at, at + word.len() as u32),
                kind: PiiKind::City,
                probability: 0.6,
            });
        }
        text.push_str(ALONE[0].0);
        let at = text.rfind("Paris").unwrap() as u32;
        model.push(ModelCandidate { span: Span::new(at, at + 5), kind: PiiKind::City, probability: 0.72 });
        let (pii, _) = resolve_model_with_rules(&text, ViewId::SOURCE, &opts, &model).unwrap();
        let paris = pii.mentions().any(|m| m.kind == PiiKind::City && &text[m.span.range()] == "Paris");
        assert_eq!(paris, paris_is_city, "{made_up} made-up names before it");
    }
}
