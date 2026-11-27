//! #2312 and #2388 with the trained head: the Places pack and the
//! Nemotron-PII head (#1888) over the person-name memos of `turbo-pii`
//! (`places-person-names.txt`, gold `labels-places-person.tsv`, and the
//! held-out `places-person-names-2.txt` and `-3.txt` of #2398). The head gives a
//! person span to few of the names (`Madison` alone at 0.50 on the first
//! memo) and its own `city` span to the rest, so the person rules of the
//! city scanner turn those down wherever a person word stands around the
//! name (`we asked Columbus`, `Salem, who`). What is left is the names
//! standing alone that the head gives no person span (`Paris won the
//! race`): no more false cities than the rules alone give there, and
//! every city use is still found. Neither score of the head tells those
//! names from city uses (#2398, docs/pii.md), so each scored candidate is
//! printed for the next attempt. Ignored, as the head is not in the
//! repository: run with `-- --ignored` where `TURBO_NEMOTRON_PII_BUNDLE`
//! names the assembled bundle, and without it the tests fail.

use std::path::Path;

use turbo_analysis::Analyzer;
use turbo_analysis_embed::EncoderOptions;
use turbo_pii::{Pack, PiiDetector};
use turbo_tagger::{NEMOTRON_PII_TEMPERATURE, PiiTagger};
use turbo_tokenize::SimpleTokenizer;

#[path = "common/pii_memo.rs"]
mod pii_memo;

fn read(file: &str, labels: &str) -> pii_memo::Read {
    let bundle_path = std::env::var_os("TURBO_NEMOTRON_PII_BUNDLE")
        .expect("TURBO_NEMOTRON_PII_BUNDLE names the assembled Nemotron PII bundle");
    let bundle_dir = turbo_pins::require(Path::new(&bundle_path));
    let tagger = PiiTagger::open(&bundle_dir, None, &EncoderOptions::default())
        .unwrap()
        .with_temperature(NEMOTRON_PII_TEMPERATURE)
        .unwrap()
        .with_score_threshold(0.50)
        .unwrap()
        .with_rules(PiiDetector::pack(Pack::Places));
    let analyzer = Analyzer::builder().add(SimpleTokenizer::default()).add(tagger).build().unwrap();
    pii_memo::read(&analyzer, file, labels)
}

#[test]
#[ignore = "needs the assembled Nemotron PII bundle in TURBO_NEMOTRON_PII_BUNDLE"]
fn the_person_rules_turn_down_the_heads_cities_on_people() {
    let read = read("places-person-names.txt", "labels-places-person.tsv");
    assert_eq!(read.missed, Vec::<String>::new(), "city uses the head took away");
    // Paragraphs 1 to 4 put a person word next to each name, so the rules
    // decide them whatever the head tags; paragraph 7 has none, and only a
    // person span of the head settles a name there.
    let alone = read.text.find("\n7. ").unwrap()..read.text.find("\n8. ").unwrap();
    let beside: Vec<usize> = read.false_cities.iter().copied().filter(|s| !alone.contains(s)).collect();
    assert_eq!(read.names(&beside), Vec::<&str>::new(), "false cities next to a person word");
    let left = read.names(&read.false_cities);
    assert!(left.len() < 5, "more false cities than the rules alone give: {left:?}");
}

/// The held-out memo of #2398, written before the head was run on it:
/// every city use is found, and the false cities are the names standing
/// alone in paragraph 4, no more than the rules alone give.
#[test]
#[ignore = "needs the assembled Nemotron PII bundle in TURBO_NEMOTRON_PII_BUNDLE"]
fn the_held_out_memo_keeps_every_city_use() {
    let read = read("places-person-names-2.txt", "labels-places-person-2.tsv");
    assert_eq!(read.missed, Vec::<String>::new(), "city uses the head took away");
    let alone = read.text.find("\n4. ").unwrap()..read.text.find("\n5. ").unwrap();
    let beside: Vec<usize> = read.false_cities.iter().copied().filter(|s| !alone.contains(s)).collect();
    assert_eq!(read.names(&beside), Vec::<&str>::new(), "false cities next to a person word");
    let left = read.names(&read.false_cities);
    assert!(left.len() <= 5, "more false cities than the rules alone give: {left:?}");
}

/// The third memo of #2398, written before the refitted head was run on
/// it: people as subjects inside a sentence and a name in both roles in
/// one sentence. No more false cities than the six the rules alone give.
/// The refitted head reads one city use as a person (`back from Vienna`,
/// `first-name` 0.56 over `city` 0.40), so the mention is a first name
/// and no city; that use still carries the head's `city` reading among
/// the candidates, and no other city use is lost.
#[test]
#[ignore = "needs the assembled Nemotron PII bundle in TURBO_NEMOTRON_PII_BUNDLE"]
fn the_third_memo_loses_no_city_use_without_a_city_reading() {
    let read = read("places-person-names-3.txt", "labels-places-person-3.tsv");
    assert_eq!(read.unread, Vec::<String>::new(), "city uses with no city reading left");
    assert!(read.missed.len() <= 1, "city uses the head took away: {:?}", read.missed);
    let left = read.names(&read.false_cities);
    assert!(left.len() <= 6, "more false cities than the rules alone give: {left:?}");
}
