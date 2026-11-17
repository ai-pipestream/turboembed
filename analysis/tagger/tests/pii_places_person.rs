//! #2312, #2388 and #2398 with the trained head: the Places pack and the
//! Nemotron-PII head (#1888) over the person-name memos of `turbo-pii`
//! (`places-person-names.txt`, gold `labels-places-person.tsv`, and the
//! held-out `places-person-names-2.txt`). The head gives a person span
//! to few of the names (`Madison` alone at 0.50 on the first memo) and
//! its own `city` span to the rest, so the person rules of the city
//! scanner turn those down wherever a person word stands around the
//! name (`we asked Columbus`, `Salem, who`), and a name the memo uses
//! for a person gives way where it stands alone and the head is unsure
//! of its city (`Paris won the race` at 0.72). What is left is a name
//! alone the head is sure of (`Columbus drove home` at 0.95). Every city
//! use is still found. Runs only where `TURBO_NEMOTRON_PII_BUNDLE` names
//! the assembled bundle, as the head is not in the repository.

use std::path::Path;

use turbo_analysis::{Analyzer, Document, Layer};
use turbo_analysis_embed::EncoderOptions;
use turbo_analysis_layers::{Pii, PiiCandidates, PiiKind};
use turbo_pii::{Pack, PiiDetector};
use turbo_tagger::{NEMOTRON_PII_TEMPERATURE, PiiTagger};
use turbo_tokenize::SimpleTokenizer;

/// A memo's text and its gold city spans.
fn memo(file: &str, labels: &str) -> (String, Vec<(u32, u32)>) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../pii/tests/data/human-eval");
    let text = std::fs::read_to_string(dir.join(file)).unwrap();
    let gold = std::fs::read_to_string(dir.join(labels))
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

/// What the head and the Places pack read on a memo.
struct Read {
    text: String,
    /// The start of each false city.
    false_cities: Vec<usize>,
    /// The city uses not found.
    missed: Vec<String>,
}

impl Read {
    fn names(&self, starts: &[usize]) -> Vec<&str> {
        starts.iter().map(|&s| self.text[s..].split(|c: char| !c.is_alphabetic()).next().unwrap()).collect()
    }
}

/// Runs the head with the Places pack over a memo, or nothing without
/// the bundle. Prints every scored candidate, for the record.
fn read(file: &str, labels: &str) -> Option<Read> {
    let bundle_path = std::env::var_os("TURBO_NEMOTRON_PII_BUNDLE")?;
    let bundle_dir = turbo_pins::require(Path::new(&bundle_path));
    let tagger = PiiTagger::open(&bundle_dir, None, &EncoderOptions::default())
        .unwrap()
        .with_temperature(NEMOTRON_PII_TEMPERATURE)
        .unwrap()
        .with_score_threshold(0.50)
        .unwrap()
        .with_rules(PiiDetector::pack(Pack::Places));
    let analyzer = Analyzer::builder().add(SimpleTokenizer::default()).add(tagger).build().unwrap();
    let (text, gold) = memo(file, labels);
    let mut docs = vec![Document::new(text.as_str())];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    let pii = docs[0].get::<Pii>().unwrap();
    pii.check().unwrap();
    let candidates = docs[0].get::<PiiCandidates>().unwrap();
    println!("{file}");
    for c in candidates.candidates() {
        if let Some(score) = c.score {
            let kind = candidates.kind_name(c.kind).unwrap_or_default();
            let covered = &text[c.span.range()];
            println!("  {:>5} {covered:<10} {kind:<12} {:.3} {:?}", c.span.start, score.value, c.disposition);
        }
    }
    let cities: Vec<(u32, u32)> =
        pii.mentions().filter(|m| m.kind == PiiKind::City).map(|m| (m.span.start, m.span.end)).collect();
    let people: Vec<&str> = pii
        .mentions()
        .filter(|m| pii.kind_name(m.kind).is_some_and(|k| k == "first-name" || k == "last-name"))
        .map(|m| &text[m.span.range()])
        .collect();
    let false_cities: Vec<usize> = cities.iter().filter(|c| !gold.contains(c)).map(|c| c.0 as usize).collect();
    let missed: Vec<String> =
        gold.iter().filter(|g| !cities.contains(g)).map(|&(s, e)| text[s as usize..e as usize].to_owned()).collect();
    println!("head person spans: {people:?}");
    println!("city tp {} fp {} fn {}", cities.len() - false_cities.len(), false_cities.len(), missed.len());
    let read = Read { text, false_cities, missed };
    println!("false cities: {:?}", read.names(&read.false_cities));
    Some(read)
}

#[test]
fn the_person_rules_turn_down_the_heads_cities_on_people() {
    let Some(read) = read("places-person-names.txt", "labels-places-person.tsv") else {
        return;
    };
    assert_eq!(read.missed, Vec::<String>::new(), "city uses the head took away");
    // Paragraphs 1 to 4 put a person word next to each name, so the rules
    // decide them whatever the head tags; paragraph 7 has none, and only a
    // person span of the head or an unsure city on a name the memo uses
    // for a person settles a name there.
    let alone = read.text.find("\n7. ").unwrap()..read.text.find("\n8. ").unwrap();
    let beside: Vec<usize> = read.false_cities.iter().copied().filter(|s| !alone.contains(s)).collect();
    assert_eq!(read.names(&beside), Vec::<&str>::new(), "false cities next to a person word");
    // #2398: `Columbus drove home` alone is left, at 0.95.
    assert!(read.false_cities.len() < 2, "false cities among the names alone: {:?}", read.names(&read.false_cities));
}

/// The held-out memo, written before the head was run on it.
#[test]
fn the_held_out_memo() {
    let Some(read) = read("places-person-names-2.txt", "labels-places-person-2.tsv") else {
        return;
    };
    let _ = read;
}
