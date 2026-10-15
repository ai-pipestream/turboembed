//! The name recipes' reader on documents written in MASC's layout: the
//! names it keeps, those it leaves out and counts, where a sample clears
//! the adaptive data, and how its names differ from those OpenNLP's MASC
//! stream reads.

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
    assert_eq!(counts, NamesCard { documents: 2, names: 6, gapped: 2, overlapping: 6, other_types: 3 });

    // The organization recipe renames MASC's `org`.
    let (samples, counts, _) = read(&corpus, &ORGANIZATION);
    assert_eq!(names(&samples[1]), ["0 1 organization"]);
    assert_eq!(counts, NamesCard { documents: 2, names: 1, gapped: 0, overlapping: 0, other_types: 14 });
    let (samples, counts, _) = read(&corpus, &LOCATION);
    assert_eq!(names(&samples[0]), ["0 1 location"]);
    assert_eq!((counts.names, counts.other_types), (2, 13));
    let (_, counts, _) = read(&corpus, &DATE);
    assert_eq!((counts.names, counts.other_types), (0, 15));
    std::fs::remove_dir_all(&root).unwrap();
}

/// OpenNLP's `MascNamedEntitySampleStream` reads a name of two or more
/// tokens without its last token and a name of one token at a
/// sentence's start as empty, keeps only the first name of a sentence of
/// two or more (its removal of overlapping names compares a name with
/// itself), and marks every sentence to clear the adaptive data; the
/// recipes read the entities as the file links them.
#[test]
fn the_recipes_read_whole_names_where_opennlp_s_stream_drops_their_last_token() {
    let root = temp_dir("names-opennlp");
    write_document(&root.join("written/news"), "a", SENTENCES, &[2, 6, 7]);
    let corpus = MascCorpus::read(&root).unwrap();
    let opennlp: Vec<NameSample<'_>> = MascNameReader::new(&corpus).unwrap().map(|s| s.unwrap().sample).collect();
    // `that week` has lost `week`, and then the overlap removal drops it.
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
