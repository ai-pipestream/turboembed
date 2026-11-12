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
