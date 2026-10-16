//! The name recipes' reader on documents written in MASC's layout: the
//! names it keeps, those it leaves out and counts, the entities in no
//! sentence, where a sample clears the adaptive data, and how its names
//! differ from those OpenNLP's MASC stream reads.

mod common;

use common::{SENTENCES, temp_dir, write_document};
use turbo_formats::masc::{MascCorpus, MascNameReader};
use turbo_formats::{Field, NameSample};
use turbo_train_recipes::card::NamesCard;
use turbo_train_recipes::namefind::{DATE, LOCATION, NameRecipe, ORGANIZATION, PERSON, read_samples};
use turbo_train_recipes::{Error, LeftOut};

/// The samples `recipe` reads from `corpus`, and what it counted.
fn read<'c>(corpus: &'c MascCorpus, recipe: &NameRecipe) -> (Vec<NameSample<'c>>, NamesCard, usize) {
    let mut samples = Vec::new();
    let (counts, skipped) = read_samples(corpus, &recipe.types, |s| {
        samples.push(s);
        Ok(())
    })
    .unwrap();
    (samples, counts, skipped)
}

/// A sample's names as `start end type`, and its tokens.
fn names(sample: &NameSample<'_>) -> Vec<String> {
    sample.names().iter().map(|n| format!("{} {} {}", n.start, n.end, n.kind.unwrap_or("null"))).collect()
}

fn words<'s>(sample: &'s NameSample<'_>) -> Vec<&'s str> {
    sample.tokens().iter().map(Field::as_str).collect()
}

const ODD: &[&str] = &[
    // A person over tokens 0 and 2 with a gap, two persons that overlap
    // it, a location, and a person after them.
    "A/DT B/NN C/NN D/NN ~./. | person 0 2 | person 1 3 | person 1 2 | location 0 | person 3",
    // No Penn token: skipped.
    "^Not ^tokenized ~^.",
    // MASC's label for an organization.
    "Acme/NNP won/VBD ~./. | org 0",
    // The same tokens labelled twice.
    "Bo/NNP ran/VBD ~./. | person 0 | person 0",
];

#[test]
fn gapped_names_are_kept_and_overlaps_and_other_types_are_left_out_and_counted() {
    let root = temp_dir("names-odd");
    write_document(&root.join("written/news"), "a", ODD, &[0, 2, 3]);
    write_document(&root.join("written/news"), "b", ODD, &[1, 3, 0]);
    let corpus = MascCorpus::read(&root).unwrap();

    let (samples, counts, skipped) = read(&corpus, &PERSON);
    // Document b's first sentence has no token: skipped, and the next
    // sample, its first, clears the adaptive data.
    assert_eq!(skipped, 1);
    let clear: Vec<bool> = samples.iter().map(NameSample::clear_adaptive_data).collect();
    assert_eq!(clear, [true, false, false, true, false]);
    assert_eq!(words(&samples[0]), ["A", "B", "C", "D", "."]);
    // The gapped person is its first token to one past its last; the two
    // that overlap it are left out, and the person after it is kept. The
    // repeat of `Bo` is left out as an overlap.
    assert_eq!(names(&samples[0]), ["0 3 person", "3 4 person"]);
    assert_eq!(names(&samples[1]), Vec::<String>::new());
    assert_eq!(names(&samples[2]), ["0 1 person"]);
    let card = |names, gapped, overlapping, other_types| NamesCard {
        documents: 2,
        names,
        gapped,
        overlapping,
        other_types,
        outside_sentences: 0,
        in_dropped_sentences: 0,
    };
    assert_eq!(counts, card(6, 2, 6, 3));

    // The organization recipe renames MASC's `org`.
    let (samples, counts, _) = read(&corpus, &ORGANIZATION);
    assert_eq!(names(&samples[1]), ["0 1 organization"]);
    assert_eq!(counts, card(1, 0, 0, 14));
    let (samples, counts, _) = read(&corpus, &LOCATION);
    assert_eq!(names(&samples[0]), ["0 1 location"]);
    assert_eq!((counts.names, counts.other_types), (2, 13));
    let (_, counts, _) = read(&corpus, &DATE);
    assert_eq!((counts.names, counts.other_types), (0, 15));
    std::fs::remove_dir_all(&root).unwrap();
}

/// OpenNLP's `MascNamedEntitySampleStream` on the test corpus, whose
/// entity files list each entity's tokens first to last: `that week`
/// loses `week`, a name of one token at a sentence's start is empty, a
/// sentence of two or more names here keeps only its first (the removal
/// of overlapping names compares a name with itself), and every sentence
/// clears the adaptive data; the recipes read the entities as the file
/// links them.
#[test]
fn the_recipes_read_whole_names_where_opennlp_s_stream_drops_their_last_token() {
    let root = temp_dir("names-opennlp");
    write_document(&root.join("written/news"), "a", SENTENCES, &[2, 6, 7]);
    let corpus = MascCorpus::read(&root).unwrap();
    let opennlp: Vec<NameSample<'_>> = MascNameReader::new(&corpus).unwrap().map(|s| s.unwrap().sample).collect();
    // `that week`, two tokens listed first to last, has lost `week`, and
    // then the overlap removal drops it.
    assert_eq!(names(&opennlp[0]), ["5 6 location"]);
    assert_eq!(names(&opennlp[1]), ["0 0 person"]);
    assert_eq!(names(&opennlp[2]), ["1 2 person"]);
    assert!(opennlp.iter().all(NameSample::clear_adaptive_data));

    let (dates, _, _) = read(&corpus, &DATE);
    assert_eq!(words(&dates[0])[6..8], ["that", "week"]);
    assert_eq!(names(&dates[0]), ["6 8 date"]);
    assert_eq!(names(&dates[2]), ["6 8 date", "9 10 date"]);
    let (persons, _, _) = read(&corpus, &PERSON);
    assert_eq!(names(&persons[1]), ["0 1 person", "6 7 person"]);
    let clear: Vec<bool> = persons.iter().map(NameSample::clear_adaptive_data).collect();
    assert_eq!(clear, [true, false, false]);
    std::fs::remove_dir_all(&root).unwrap();
}

/// Entities of one name each, listed in different orders.
const ORDERS: &[&str] = &[
    // MASC's order, last to first.
    "He/PRP met/VBD Ann/NNP Lee/NNP ~./. | person 3 2",
    // First to last: two tokens, then three.
    "He/PRP met/VBD Ann/NNP Lee/NNP ~./. | person 2 3",
    "He/PRP met/VBD Ann/NNP Mary/NNP Lee/NNP ~./. | person 2 3 4",
    // At the sentence's start: first to last, two tokens then three, then
    // last to first.
    "Ann/NNP Lee/NNP left/VBD ~./. | person 0 1",
    "Ann/NNP Mary/NNP Lee/NNP left/VBD ~./. | person 0 1 2",
    "Ann/NNP Mary/NNP Lee/NNP left/VBD ~./. | person 2 1 0",
];

/// What OpenNLP's stream keeps of a name depends on the order the entity
/// file lists its tokens (`MascSentence.java:217`): listed last to first,
/// as MASC lists 8,770 of its 9,068 entities of several tokens, a name is
/// whole; listed first to last, a name of an even number of tokens loses
/// its last token and one of an odd number is whole, but at a sentence's
/// start the parity flips. The recipes read every name whole.
#[test]
fn opennlp_s_stream_keeps_a_name_whole_in_masc_s_order_and_by_parity_in_the_other() {
    let root = temp_dir("names-orders");
    write_document(&root.join("written/news"), "a", ORDERS, &[0, 1, 2, 3, 4, 5]);
    let corpus = MascCorpus::read(&root).unwrap();
    let opennlp: Vec<Vec<String>> = MascNameReader::new(&corpus).unwrap().map(|s| names(&s.unwrap().sample)).collect();
    let one = |s: &str| vec![s.to_owned()];
    assert_eq!(
        opennlp,
        [
            one("2 4 person"),
            one("2 3 person"),
            one("2 5 person"),
            one("0 2 person"),
            one("0 2 person"),
            one("0 3 person")
        ]
    );
    let (persons, _, _) = read(&corpus, &PERSON);
    let ours: Vec<Vec<String>> = persons.iter().map(names).collect();
    assert_eq!(
        ours,
        [
            one("2 4 person"),
            one("2 4 person"),
            one("2 5 person"),
            one("0 2 person"),
            one("0 3 person"),
            one("0 3 person")
        ]
    );
    std::fs::remove_dir_all(&root).unwrap();
}

/// Replaces `from` with `to` in the file at `path`, which must hold it.
fn edit(path: &std::path::Path, from: &str, to: &str) {
    let text = std::fs::read_to_string(path).unwrap();
    assert!(text.contains(from), "{}: {from}", path.display());
    std::fs::write(path, text.replacen(from, to, 1)).unwrap();
}

/// The anchors of sentence region `n` of a sentence file written by
/// `write_document`.
fn anchors(s: &str, n: usize) -> (String, String) {
    let at = s.find(&format!("xml:id=\"s-r{n}\" anchors=\"")).unwrap();
    let rest = &s[at..];
    let open = rest.find("anchors=\"").unwrap() + "anchors=\"".len();
    let close = open + rest[open..].find('"').unwrap();
    let (start, end) = rest[open..close].split_once(' ').unwrap();
    (start.to_owned(), end.to_owned())
}

/// An entity outside every region of the sentence file, or in a region
/// the reader drops (one that does not end before the next starts), is
/// in no sample, and the recipe of its type counts it on the card.
#[test]
fn entities_in_no_sentence_are_counted_by_why() {
    let root = temp_dir("names-unplaced");
    let dir = root.join("written/news");
    write_document(&dir, "a", SENTENCES, &[0, 1, 2, 7]);
    let s_file = dir.join("a-s.xml");
    let s = std::fs::read_to_string(&s_file).unwrap();
    // The first sentence's region goes, so `Dr. Smith` is in no region;
    // the second's is made to end where the third starts, so the reader
    // drops it and `Jones` with it.
    let (start, end) = anchors(&s, 0);
    edit(
        &s_file,
        &format!(
            "  <region xml:id=\"s-r0\" anchors=\"{start} {end}\"/>\n  <node xml:id=\"s-n0\"><link targets=\"s-r0\"/></node>\n"
        ),
        "",
    );
    let (start, end) = anchors(&s, 1);
    let (next, _) = anchors(&s, 2);
    edit(&s_file, &format!("anchors=\"{start} {end}\""), &format!("anchors=\"{start} {next}\""));
    let corpus = MascCorpus::read(&root).unwrap();

    let (samples, counts, skipped) = read(&corpus, &PERSON);
    assert_eq!((samples.len(), skipped), (2, 0));
    assert_eq!(words(&samples[0])[..3], ["It", "was", "cold"]);
    assert_eq!((counts.names, counts.outside_sentences, counts.in_dropped_sentences), (2, 1, 1));
    let (_, counts, _) = read(&corpus, &DATE);
    assert_eq!((counts.names, counts.outside_sentences, counts.in_dropped_sentences), (3, 0, 0));
    std::fs::remove_dir_all(&root).unwrap();
}

/// A name recipe needs a name of its type on both sides, and skips at
/// most one sentence in a thousand, as the other recipes do.
#[test]
fn a_side_without_a_name_of_the_type_or_with_too_many_skips_is_refused() {
    let root = temp_dir("names-empty");
    write_document(&root.join("written/news"), "a", SENTENCES, &[3, 3, 0]);
    let corpus = MascCorpus::read(&root).unwrap();
    assert!(matches!(DATE.held_out(&corpus), Err(Error::EmptySplit { what: "held-out name" })));
    assert!(matches!(DATE.check_training(&corpus), Err(Error::EmptySplit { what: "training name" })));
    assert!(matches!(DATE.train(&corpus), Err(Error::EmptySplit { what: "training name" })));
    let held_out = PERSON.held_out(&corpus).unwrap();
    assert_eq!((held_out.len(), held_out.skipped(), held_out.counts().names), (3, 0, 1));

    write_document(&root.join("written/news"), "b", ODD, &[1]);
    let corpus = MascCorpus::read(&root).unwrap();
    match PERSON.held_out(&corpus) {
        Err(Error::TooManyLeftOut { what: LeftOut::HeldOutSentences, count: 1, of: 4, .. }) => {}
        other => panic!("{other:?}"),
    }
    match PERSON.check_training(&corpus) {
        Err(Error::TooManyLeftOut { what: LeftOut::TrainingSentences, count: 1, of: 4, .. }) => {}
        other => panic!("{other:?}"),
    }
    std::fs::remove_dir_all(&root).unwrap();
}
