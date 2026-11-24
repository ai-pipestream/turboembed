//! #2398 and #2410 with the trained head: single sentences on names a
//! person and a city share, the name once a person and once a city.
//! Prints what the head reads on each name, with its second reading, and
//! what the head with the Places pack makes of it, and counts how many
//! names get the reading the sentence means. Ignored: run with
//! `-- --ignored` where `TURBO_NEMOTRON_PII_BUNDLE` names the assembled
//! bundle, and without it the test fails. The floors
//! are those of `turbo-en-nemotron-pii-modernbert-base-1`.

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
const CITIES: [(&str, &str); 22] = [
    ("Paris is lovely in the spring.", "Paris"),
    ("Columbus hosts the fair every August.", "Columbus"),
    ("Salem voted on the measure last week.", "Salem"),
    ("Austin has grown quickly over the past decade.", "Austin"),
    ("Charlotte is a banking hub.", "Charlotte"),
    ("Sofia was founded thousands of years ago.", "Sofia"),
    ("We visited Florence last summer.", "Florence"),
    ("Victoria lies on the southern tip of the island.", "Victoria"),
    ("Sydney hosted the games in 2000.", "Sydney"),
    ("Orlando draws millions of tourists.", "Orlando"),
    ("Dallas and Houston are four hours apart.", "Houston"),
    ("Denver sits a mile above sea level.", "Denver"),
    ("Dayton built the first airplanes.", "Dayton"),
    ("They left Salem at dawn.", "Salem"),
    ("Madison passed a new zoning rule.", "Madison"),
    ("Phoenix reached 115 degrees.", "Phoenix"),
    ("Salem is where the old office was.", "Salem"),
    ("We saw Columbus from the plane.", "Columbus"),
    ("The Dayton office still holds the files.", "Dayton"),
    ("Milan has two football clubs.", "Milan"),
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

#[test]
#[ignore = "needs the assembled Nemotron PII bundle in TURBO_NEMOTRON_PII_BUNDLE"]
fn single_sentences_on_names_a_person_and_a_city_share() {
    let bundle_path = std::env::var_os("TURBO_NEMOTRON_PII_BUNDLE")
        .expect("TURBO_NEMOTRON_PII_BUNDLE names the assembled Nemotron PII bundle");
    let bundle_dir = turbo_pins::require(Path::new(&bundle_path));
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
    println!("people (head right, head with rules right)");
    let people = count(&head, &with_rules, &PEOPLE, &["first-name", "last-name"]);
    println!("cities (head right, head with rules right)");
    let cities = count(&head, &with_rules, &CITIES, &["city"]);
    println!("people: head {} of {}, with rules {}", people.0, PEOPLE.len(), people.1);
    println!("cities: head {} of {}, with rules {}", cities.0, CITIES.len(), cities.1);
}
