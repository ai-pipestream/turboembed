//! #2312 and #2388 with the trained head: the Places pack and the
//! Nemotron-PII head (#1888) over the person-name memo of `turbo-pii`
//! (`places-person-names.txt`, gold `labels-places-person.tsv`). The
//! head gives a person span to few of the names (`Madison` alone at
//! 0.50) and its own `city` span to the rest, so the person rules of the
//! city scanner turn those down wherever a person word stands around the
//! name (`we asked Columbus`, `Salem, who`). What is left is the names
//! standing alone in paragraph 7 that the head gives no person span
//! (`Paris won the race`): no more false cities than the rules alone
//! give there, and every city use is still found. Runs only where
//! `TURBO_NEMOTRON_PII_BUNDLE` names the assembled bundle, as the head is
//! not in the repository.

use std::path::Path;

use turbo_analysis::{Analyzer, Document, Layer};
use turbo_analysis_embed::EncoderOptions;
use turbo_analysis_layers::{Pii, PiiKind};
use turbo_pii::{Pack, PiiDetector};
use turbo_tagger::{NEMOTRON_PII_TEMPERATURE, PiiTagger};
use turbo_tokenize::SimpleTokenizer;

/// The memo's text and its gold city spans.
fn memo() -> (String, Vec<(u32, u32)>) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../pii/tests/data/human-eval");
    let text = std::fs::read_to_string(dir.join("places-person-names.txt")).unwrap();
    let gold = std::fs::read_to_string(dir.join("labels-places-person.tsv"))
        .unwrap()
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            assert_eq!(f[3], "city", "{l}");
            (f[1].parse().unwrap(), f[2].parse().unwrap())
        })
        .collect();
    (text, gold)
}

#[test]
fn the_person_rules_turn_down_the_heads_cities_on_people() {
    let Some(bundle_path) = std::env::var_os("TURBO_NEMOTRON_PII_BUNDLE") else {
        return;
    };
    let bundle_dir = turbo_pins::require(Path::new(&bundle_path));
    let tagger = PiiTagger::open(&bundle_dir, None, &EncoderOptions::default())
        .unwrap()
        .with_temperature(NEMOTRON_PII_TEMPERATURE)
        .unwrap()
        .with_score_threshold(0.50)
        .unwrap()
        .with_rules(PiiDetector::pack(Pack::Places));
    let analyzer = Analyzer::builder().add(SimpleTokenizer::default()).add(tagger).build().unwrap();
    let (text, gold) = memo();
    let mut docs = vec![Document::new(text.as_str())];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    let pii = docs[0].get::<Pii>().unwrap();
    pii.check().unwrap();
    let cities: Vec<(u32, u32)> =
        pii.mentions().filter(|m| m.kind == PiiKind::City).map(|m| (m.span.start, m.span.end)).collect();
    let people: Vec<&str> = pii
        .mentions()
        .filter(|m| pii.kind_name(m.kind).is_some_and(|k| k == "first-name" || k == "last-name"))
        .map(|m| &text[m.span.range()])
        .collect();
    let false_cities: Vec<&str> =
        cities.iter().filter(|c| !gold.contains(c)).map(|&(s, e)| &text[s as usize..e as usize]).collect();
    let missed: Vec<&str> =
        gold.iter().filter(|g| !cities.contains(g)).map(|&(s, e)| &text[s as usize..e as usize]).collect();
    println!("head person spans: {people:?}");
    println!("city tp {} fp {} fn {}", cities.len() - false_cities.len(), false_cities.len(), missed.len());
    println!("false cities: {false_cities:?}");
    assert_eq!(missed, Vec::<&str>::new(), "city uses the head took away");
    // Paragraphs 1 to 4 put a person word next to each name, so the rules
    // decide them whatever the head tags; paragraph 7 has none, and only a
    // person span of the head settles a name there.
    let alone = text.find("\n7. ").unwrap()..text.find("\n8. ").unwrap();
    let beside_person_words: Vec<&str> = cities
        .iter()
        .filter(|c| !gold.contains(c) && !alone.contains(&(c.0 as usize)))
        .map(|&(s, e)| &text[s as usize..e as usize])
        .collect();
    assert_eq!(beside_person_words, Vec::<&str>::new(), "false cities next to a person word");
    assert!(false_cities.len() < 5, "more false cities than the rules alone give: {false_cities:?}");
}
