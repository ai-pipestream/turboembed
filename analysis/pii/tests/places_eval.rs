//! #1886: precision and recall of the Places pack (`state`, `county`,
//! `city`) on the human-written evaluation set and a project-authored memo
//! of places and near misses (`places-memos.txt`), gold in
//! `labels-places.tsv`. The test prints a table (`--nocapture`) and holds
//! each kind to its measured counts.

mod support;

use support::{HUMAN_DOCS, Labelled, layer, load_labelled, measure_with, pack, ratio_text};
use turbo_analysis::{Layer as _, Span, ViewId};
use turbo_pii::nemotron::{ModelCandidate, ModelProvenance, ModelResolution, resolve_model_with_rules};
use turbo_pii::{Pack, PiiKind, nemotron_custom_kinds};

#[test]
fn places_precision_and_recall_on_human_written_text() {
    let mut docs = HUMAN_DOCS.to_vec();
    docs.push((
        "places-memos",
        false,
        "places-memos.txt",
        "5b6d7e9f28069ea39d9fd63fe1f8dcb27e2d7d873fe755211a456a7cd823fb8b",
    ));
    let set = load_labelled(&docs, "labels-places.tsv", false);
    let d = pack(Pack::Places);
    for l in &set {
        layer(&d, &l.text).check().expect("layer is ordered");
    }
    let by_kind = measure_with(&d, &set);
    eprintln!("\nplaces on the human-written set:");
    eprintln!("{:<8} {:>5} {:>5} {:>5} {:>9} {:>7}", "kind", "tp", "fp", "fn", "precision", "recall");
    for (name, c) in &by_kind {
        let (p, r) = (ratio_text(c.tp, c.tp + c.fp), ratio_text(c.tp, c.tp + c.fn_));
        eprintln!("{name:<8} {:>5} {:>5} {:>5} {p:>9} {r:>7}", c.tp, c.fp, c.fn_);
    }
    let got: Vec<(&str, usize, usize, usize)> = by_kind.iter().map(|(k, c)| (*k, c.tp, c.fp, c.fn_)).collect();
    assert_eq!(got, [("city", 35, 0, 1), ("county", 9, 0, 0), ("state", 40, 0, 0)]);
}

/// #2295: a project-authored memo of people who share a city's name
/// (`we met Madison`, `Salem, who`, `Paris's mother`) and city uses of the
/// same names (`from Madison to Columbus`, `people in Paris who`), gold in
/// `labels-places-person.tsv`. Every person use stays no city; the counts
/// are pinned.
/// The person-name memo and its gold.
fn person_memo() -> Vec<Labelled> {
    let docs = [(
        "places-person-names",
        false,
        "places-person-names.txt",
        "1bb27bed8ad4f40e981f2273f13f9d41136c6dec5bc021d48421125b605f983d",
    )];
    load_labelled(&docs, "labels-places-person.tsv", false)
}

/// Without the PII head, the names standing alone (`Paris won the race`,
/// `we saw Madison`, #2312) still read as cities: five false cities, pinned.
#[test]
fn person_names_that_share_a_city_name_are_no_city() {
    let set = person_memo();
    let by_kind = measure_with(&pack(Pack::Places), &set);
    eprintln!("\nplaces on the person-name memo:");
    for (name, c) in &by_kind {
        eprintln!("{name:<8} tp {} fp {} fn {}", c.tp, c.fp, c.fn_);
    }
    let got: Vec<(&str, usize, usize, usize)> = by_kind.iter().map(|(k, c)| (*k, c.tp, c.fp, c.fn_)).collect();
    assert_eq!(got, [("city", 13, 5, 0)]);
}

/// The names of people in the memo: every use of them in paragraphs 1
/// to 4 and 7, where the notes speak of people.
const PEOPLE: [&str; 5] = ["Columbus", "Dayton", "Madison", "Paris", "Salem"];

/// The `first-name` spans a head that reads the memo right gives: each
/// name of [`PEOPLE`] in the person paragraphs.
fn person_spans(text: &str, first_name: PiiKind) -> Vec<ModelCandidate> {
    let at = |p: &str| text.find(p).unwrap();
    let people = [0..at("\n5. "), at("\n7. ")..at("\n8. ")];
    let mut spans = Vec::new();
    for name in PEOPLE {
        for (start, _) in text.match_indices(name).filter(|(start, _)| people.iter().any(|p| p.contains(start))) {
            let span = Span::new(start as u32, (start + name.len()) as u32);
            spans.push(ModelCandidate { span, kind: first_name, probability: 0.9 });
        }
    }
    spans
}

/// #2312: with the person spans of the PII head (#1888), every bare
/// person use gives way and every city use stays: the memo scores no
/// false city and misses none.
#[test]
fn person_spans_of_the_head_settle_the_names_standing_alone() {
    let kinds = nemotron_custom_kinds();
    let first_name = kinds.find("first-name").unwrap();
    let rules = pack(Pack::Places);
    let provenance = ModelProvenance {
        detector_id: "turbo-tagger:pii",
        detector_version: "0.1.0",
        model_id: "turbo-en-nemotron-pii-modernbert-base-1",
        model_version: "1",
        calibrated: true,
    };
    let opts = ModelResolution {
        rules: Some(&rules),
        model_kinds: &kinds,
        provenance,
        min_score: Some(0.5),
        extra_allow: None,
    };
    let (mut tp, mut fp, mut fn_) = (0, 0, 0);
    for l in &person_memo() {
        let spans = person_spans(&l.text, first_name);
        assert_eq!(spans.len(), 18, "every person use of the memo");
        let (pii, _) = resolve_model_with_rules(&l.text, ViewId::SOURCE, &opts, &spans).unwrap();
        let cities: Vec<(u32, u32, &str)> =
            pii.mentions().filter(|m| m.kind == PiiKind::City).map(|m| (m.span.start, m.span.end, "city")).collect();
        let people = pii.mentions().filter(|m| m.kind == first_name).count();
        assert_eq!(people, 18, "no person span yields to a bare city");
        tp += cities.iter().filter(|c| l.labels.contains(c)).count();
        fp += cities.iter().filter(|c| !l.labels.contains(c)).count();
        fn_ += l.labels.iter().filter(|g| !cities.contains(g)).count();
    }
    eprintln!("\nplaces with person spans on the person-name memo: tp {tp} fp {fp} fn {fn_}");
    assert_eq!((tp, fp, fn_), (13, 0, 0));
}

/// #2388: what the trained head gave on the memo (the gated run of
/// `pii_places_person.rs`): a `first-name` span on `Madison` alone, and
/// its own `city` span on every other use of the names. The person rules
/// turn down the head's cities they turned down as bare readings, so
/// only the four names standing alone that no person span covers are
/// left as false cities (`Paris won`, `Columbus drove`, `called Salem`,
/// `Dayton answered`), one fewer than with the rules alone.
#[test]
fn head_cities_on_names_the_person_rules_refuse_are_no_city() {
    let kinds = nemotron_custom_kinds();
    let first_name = kinds.find("first-name").unwrap();
    let rules = pack(Pack::Places);
    let provenance = ModelProvenance {
        detector_id: "turbo-tagger:pii",
        detector_version: "0.1.0",
        model_id: "turbo-en-nemotron-pii-modernbert-base-1",
        model_version: "1",
        calibrated: true,
    };
    let opts = ModelResolution {
        rules: Some(&rules),
        model_kinds: &kinds,
        provenance,
        min_score: Some(0.5),
        extra_allow: None,
    };
    for l in &person_memo() {
        let mut spans: Vec<ModelCandidate> =
            person_spans(&l.text, first_name).into_iter().filter(|c| &l.text[c.span.range()] == "Madison").collect();
        assert_eq!(spans.len(), 4, "every person use of Madison");
        for name in PEOPLE.into_iter().filter(|&n| n != "Madison") {
            for (start, _) in l.text.match_indices(name) {
                let span = Span::new(start as u32, (start + name.len()) as u32);
                spans.push(ModelCandidate { span, kind: PiiKind::City, probability: 0.9 });
            }
        }
        let (pii, candidates) = resolve_model_with_rules(&l.text, ViewId::SOURCE, &opts, &spans).unwrap();
        candidates.check_against(&pii).unwrap();
        let cities: Vec<(u32, u32, &str)> =
            pii.mentions().filter(|m| m.kind == PiiKind::City).map(|m| (m.span.start, m.span.end, "city")).collect();
        let mut false_cities: Vec<&str> =
            cities.iter().filter(|c| !l.labels.contains(c)).map(|&(s, e, _)| &l.text[s as usize..e as usize]).collect();
        false_cities.sort_unstable();
        let missed = l.labels.iter().filter(|g| !cities.contains(g)).count();
        eprintln!(
            "\nplaces with the head's own cities on the person-name memo: false {false_cities:?}, missed {missed}"
        );
        assert_eq!(false_cities, ["Columbus", "Dayton", "Paris", "Salem"]);
        assert_eq!(missed, 0);
        assert_eq!(pii.mentions().filter(|m| m.kind == first_name).count(), 4);
    }
}

/// What the trained head gave on the memo at temperature 1.0938 (the
/// gated run of `pii_places_person.rs` on 2026-11-17): start, kind and
/// probability of each candidate on one of the names.
const HEAD_ON_MEMO: [(u32, &str, f32); 29] = [
    (106, "first-name", 0.717),
    (158, "city", 0.943),
    (207, "city", 0.760),
    (235, "city", 0.983),
    (289, "city", 0.532),
    (336, "first-name", 0.745),
    (371, "city", 0.731),
    (405, "city", 0.953),
    (463, "city", 0.911),
    (504, "company-name", 0.403),
    (532, "first-name", 0.796),
    (567, "city", 0.926),
    (654, "city", 0.810),
    (692, "city", 0.873),
    (703, "city", 0.999),
    (732, "city", 0.899),
    (840, "city", 0.990),
    (882, "city", 0.983),
    (928, "city", 0.994),
    (945, "city", 0.990),
    (969, "city", 0.915),
    (994, "city", 0.724),
    (1030, "city", 0.954),
    (1074, "first-name", 0.877),
    (1108, "city", 0.758),
    (1127, "city", 0.807),
    (1190, "city", 0.849),
    (1216, "city", 0.989),
    (1264, "city", 0.851),
];

/// #2398: with the head's own spans and probabilities, the names the
/// memo uses for a person elsewhere give way where they stand alone and
/// the head is unsure of its city (`Paris won` at 0.72, `called Salem`
/// at 0.76, `Dayton answered` at 0.81). `Columbus drove home` stays a
/// false city, as the head gives it 0.95, above the city uses with no
/// place word before them (`Salem is where` at 0.90). Every city use
/// stays, those under the bar too, as a place word stands before each
/// (`went to Paris` at 0.81).
#[test]
fn names_the_memo_uses_for_people_are_no_city_standing_alone() {
    let kinds = nemotron_custom_kinds();
    let rules = pack(Pack::Places);
    let provenance = ModelProvenance {
        detector_id: "turbo-tagger:pii",
        detector_version: "0.1.0",
        model_id: "turbo-en-nemotron-pii-modernbert-base-1",
        model_version: "1",
        calibrated: true,
    };
    let opts = ModelResolution {
        rules: Some(&rules),
        model_kinds: &kinds,
        provenance,
        min_score: Some(0.5),
        extra_allow: None,
    };
    for l in &person_memo() {
        let spans: Vec<ModelCandidate> = HEAD_ON_MEMO
            .iter()
            .map(|&(start, kind, probability)| {
                let rest = &l.text[start as usize..];
                let name = PEOPLE.into_iter().find(|n| rest.starts_with(n)).expect("a name of the memo");
                let kind = PiiKind::parse(kind).or_else(|| kinds.find(kind)).unwrap();
                ModelCandidate { span: Span::new(start, start + name.len() as u32), kind, probability }
            })
            .collect();
        let (pii, candidates) = resolve_model_with_rules(&l.text, ViewId::SOURCE, &opts, &spans).unwrap();
        candidates.check_against(&pii).unwrap();
        let cities: Vec<(u32, u32, &str)> =
            pii.mentions().filter(|m| m.kind == PiiKind::City).map(|m| (m.span.start, m.span.end, "city")).collect();
        let false_cities: Vec<&str> =
            cities.iter().filter(|c| !l.labels.contains(c)).map(|&(s, e, _)| &l.text[s as usize..e as usize]).collect();
        let missed = l.labels.iter().filter(|g| !cities.contains(g)).count();
        eprintln!(
            "\nplaces with the head's spans and scores on the person-name memo: false {false_cities:?}, missed {missed}"
        );
        assert_eq!(false_cities, ["Columbus"]);
        assert_eq!(missed, 0);
    }
}

/// The held-out second memo (#2398), written before the PII head was run
/// on it, and its gold.
fn second_person_memo() -> Vec<Labelled> {
    let docs = [(
        "places-person-names-2",
        false,
        "places-person-names-2.txt",
        "89695bbd13868aaea4170178cb94dc73f38a4dd530ef1d4d2f9ccb4e7c4eec68",
    )];
    load_labelled(&docs, "labels-places-person-2.tsv", false)
}

/// The rules alone on the held-out memo: every city use, and the names
/// standing alone as false cities, pinned.
#[test]
fn the_held_out_person_memo_with_the_rules_alone() {
    let by_kind = measure_with(&pack(Pack::Places), &second_person_memo());
    let got: Vec<(&str, usize, usize, usize)> = by_kind.iter().map(|(k, c)| (*k, c.tp, c.fp, c.fn_)).collect();
    eprintln!("\nplaces on the held-out person-name memo: {got:?}");
    assert_eq!(got, [("city", 8, 5, 0)]);
}
