//! #2398 and #2410 with the trained head: single sentences on names a
//! person and a city share, the name once a person and once a city.
//! Gate the catalog head and the unreleased refit at their measured quality,
//! then compare the eight city subjects from #2410 and both person memos.
//! Set TURBO_NEMOTRON_PII_BUNDLE (catalog) and TURBO_NEMOTRON_PII_REFIT_BUNDLE
//! (refit). Both must name the exact bundles measured in the training record.

#[path = "common/pii_memo.rs"]
mod pii_memo;

use std::path::Path;

use turbo_analysis::{Analyzer, Document, Layer};
use turbo_analysis_embed::EncoderOptions;
use turbo_analysis_layers::{Pii, PiiCandidates};
use turbo_pii::{Pack, PiiDetector};
use turbo_tagger::{NEMOTRON_PII_TEMPERATURE, PiiTagger};
use turbo_tokenize::SimpleTokenizer;

/// Sentences that mean a person, and the name.
const PEOPLE: [(&str, &str); 20] = [
    ("Paris won the race on Saturday.", "Paris"),
    ("Columbus drove home after the shift.", "Columbus"),
    ("They called Salem twice before lunch.", "Salem"),
    ("Austin finished the report early.", "Austin"),
    ("Florence baked a cake for the office.", "Florence"),
    ("Sofia missed the bus this morning.", "Sofia"),
    ("Charlotte forgot her keys again.", "Charlotte"),
    ("Victoria was promoted in March.", "Victoria"),
    ("Sydney plays the cello.", "Sydney"),
    ("Orlando is on leave until Friday.", "Orlando"),
    ("I had lunch with Austin yesterday.", "Austin"),
    ("Dallas and Houston both passed the exam.", "Dallas"),
    ("Denver is late again.", "Denver"),
    ("Phoenix called in sick.", "Phoenix"),
    ("Savannah wants the day off.", "Savannah"),
    ("Please send the form to Madison by noon.", "Madison"),
    ("Dayton answered the phone.", "Dayton"),
    ("Milan lost his badge.", "Milan"),
    ("Valencia will cover the late shift.", "Valencia"),
    ("My manager is Santiago.", "Santiago"),
];

/// Sentences that mean a city, and the name.
const REPORTED: [(&str, &str); 8] = [
    ("Paris is lovely in the spring.", "Paris"),
    ("Madison passed a new zoning rule.", "Madison"),
    ("Salem is where the old office was.", "Salem"),
    ("Milan has two football clubs.", "Milan"),
    ("Victoria lies on the southern tip of the island.", "Victoria"),
    ("Austin has grown quickly over the past decade.", "Austin"),
    ("Sydney hosted the games in 2000.", "Sydney"),
    ("Orlando draws millions of tourists.", "Orlando"),
];

/// The other city cases in the original 22-sentence evaluation.
const OTHER: [(&str, &str); 14] = [
    ("Columbus hosts the fair every August.", "Columbus"),
    ("Salem voted on the measure last week.", "Salem"),
    ("Charlotte is a banking hub.", "Charlotte"),
    ("Sofia was founded thousands of years ago.", "Sofia"),
    ("We visited Florence last summer.", "Florence"),
    ("Dallas and Houston are four hours apart.", "Houston"),
    ("Denver sits a mile above sea level.", "Denver"),
    ("Dayton built the first airplanes.", "Dayton"),
    ("They left Salem at dawn.", "Salem"),
    ("Phoenix reached 115 degrees.", "Phoenix"),
    ("We saw Columbus from the plane.", "Columbus"),
    ("The Dayton office still holds the files.", "Dayton"),
    ("Valencia floods most winters.", "Valencia"),
    ("The flight to Santiago was delayed.", "Santiago"),
];

/// How many of `sentences` get a `wanted` reading on their name: from
/// the head alone as its likeliest candidate, and as a mention of the
/// head with the Places pack.
fn count(tagger: &Analyzer, with_rules: &Analyzer, sentences: &[(&str, &str)], wanted: &[&str]) -> (usize, usize) {
    let (mut head, mut both) = (0, 0);
    for &(text, name) in sentences {
        let at = text.find(name).unwrap() as u32;
        let on_name = |start: u32, end: u32| start == at && end == at + name.len() as u32;
        let mut docs = vec![Document::new(text)];
        for r in tagger.run(&mut docs) {
            r.unwrap();
        }
        let candidates = docs[0].get::<PiiCandidates>().unwrap();
        let mut scored: Vec<(f32, &str)> = candidates
            .candidates()
            .filter(|c| on_name(c.span.start, c.span.end))
            .filter_map(|c| Some((c.score?.value, candidates.kind_name(c.kind)?)))
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut docs = vec![Document::new(text)];
        for r in with_rules.run(&mut docs) {
            r.unwrap();
        }
        let pii = docs[0].get::<Pii>().unwrap();
        pii.check().unwrap();
        let mentions: Vec<&str> = pii
            .mentions()
            .filter(|m| on_name(m.span.start, m.span.end))
            .filter_map(|m| pii.kind_name(m.kind))
            .collect();
        let right_head = scored.first().is_some_and(|&(p, k)| p >= 0.5 && wanted.contains(&k));
        let right_both = mentions.len() == 1 && wanted.contains(&mentions[0]);
        head += usize::from(right_head);
        both += usize::from(right_both);
        println!(
            "  {} {} {text:<52} head {scored:?} with rules {mentions:?}",
            u8::from(right_head),
            u8::from(right_both)
        );
    }
    (head, both)
}

/// Counts from the user-facing head plus Places, alongside the head in isolation.
struct Scores {
    people: (usize, usize),
    cities: (usize, usize),
    reported: (usize, usize),
    memo_errors: [(usize, usize); 2], // false cities and missed cities, per memo
}

fn measure(var: &str, expected: &str) -> Scores {
    let path = std::env::var_os(var).unwrap_or_else(|| panic!("{var} must name the measured PII bundle"));
    let bundle_dir = turbo_pins::require(Path::new(&path));
    let (_, digest) = turbo_pins::digest(&bundle_dir).unwrap();
    assert_eq!(digest, expected, "{var} contains the wrong pinned bundle");
    let open = || {
        PiiTagger::open(&bundle_dir, None, &EncoderOptions::default())
            .unwrap()
            .with_temperature(NEMOTRON_PII_TEMPERATURE)
            .unwrap()
            .with_score_threshold(0.50)
            .unwrap()
    };
    let head = Analyzer::builder().add(SimpleTokenizer::default()).add(open()).build().unwrap();
    let with_rules = Analyzer::builder()
        .add(SimpleTokenizer::default())
        .add(open().with_rules(PiiDetector::pack(Pack::Places)))
        .build()
        .unwrap();
    let people = count(&head, &with_rules, &PEOPLE, &["first-name", "last-name"]);
    let reported = count(&head, &with_rules, &REPORTED, &["city"]);
    let other = count(&head, &with_rules, &OTHER, &["city"]);
    let cities = (reported.0 + other.0, reported.1 + other.1);
    let memo_errors = [
        ("places-person-names.txt", "labels-places-person.tsv", 13),
        ("places-person-names-2.txt", "labels-places-person-2.tsv", 8),
    ]
    .map(|(file, labels, expected_gold)| {
        let memo = pii_memo::read(&with_rules, file, labels);
        assert_eq!(memo.gold_count, expected_gold, "{file}: the held-out gold changed");
        assert!(memo.unread.is_empty(), "{file}: lost city readings {:?}", memo.unread);
        (memo.false_cities.len(), memo.missed.len())
    });
    println!("{var}: people {people:?}/20, cities {cities:?}/22, reported {reported:?}/8, memo errors {memo_errors:?}");
    Scores { people, cities, reported, memo_errors }
}

fn at_least(actual: (usize, usize), floor: (usize, usize), which: &str) {
    assert!(actual.0 >= floor.0 && actual.1 >= floor.1, "{which}: (head, with Places) {actual:?} below {floor:?}");
}

#[test]
#[ignore = "requires the pinned catalog and refitted Nemotron PII bundles"]
fn single_sentences_on_names_a_person_and_a_city_share() {
    let baseline =
        measure("TURBO_NEMOTRON_PII_BUNDLE", "19d3fb87daac08f0361390809b747af638bc2dbb93e447695c3f5609fc735b6d");
    let refit =
        measure("TURBO_NEMOTRON_PII_REFIT_BUNDLE", "dd73330bc8bef5bc0f1b9c7b6c004cfbf6713fa1dadcc03a6238a159b6ddc3b7");
    at_least(baseline.people, (10, 10), "catalog people");
    at_least(baseline.cities, (4, 7), "catalog cities");
    at_least(refit.people, (9, 9), "refit people");
    at_least(refit.cities, (19, 20), "refit cities");
    at_least(refit.reported, (6, 7), "refit reported subjects");
    assert!(refit.reported.1 > baseline.reported.1, "the refit must retain more reported city subjects with Places");
    for (i, (old, new)) in baseline.memo_errors.into_iter().zip(refit.memo_errors).enumerate() {
        assert!(new.0 <= old.0 && new.1 <= old.1, "memo {} regressed: (false, missed) {old:?} -> {new:?}", i + 1);
        assert!(new.0 <= [4, 3][i] && new.1 == 0, "refit memo {} exceeded its measured errors: {new:?}", i + 1);
    }
}
