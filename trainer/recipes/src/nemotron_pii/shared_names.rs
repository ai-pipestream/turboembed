//! Project-written sentences on names a person and a city share (#2398),
//! added to the `Nemotron-PII` training mix as `nemotron-pii-names`.
//!
//! The head trained on `Nemotron-PII` alone reads a capitalized name at
//! a sentence's start by the name, not by the sentence: `Paris won the
//! race` is a city and `Paris is lovely in the spring` a first name. The
//! sentences here put the same names in both roles, as a subject and as
//! an object, with and without a preposition before them, so the words
//! around a name decide. Each document is a few sentences drawn by a
//! fixed generator from the lists below, some joined into one so a name
//! also stands inside a sentence, and it draws most names from three
//! shared ones, so the same name is a person and a city in one document
//! as in the memos; a person is labelled
//! `first_name` and a city `city`, and nothing else in a sentence is of
//! any `Nemotron-PII` type (no dates, times, jobs or companies).
//!
//! The names of the three person-name memos that score the head
//! (`places-person-names.txt` and its `-2` and `-3` of `turbo-pii`) are
//! left out, and no sentence of the memos is used, so
//! the memos stay held out.

use crate::Result;
use crate::corpus::Sentence;

use super::spans;

/// Documents added to the training split.
pub const TRAIN_DOCUMENTS: usize = 6_000;
/// Documents added to the development split.
pub const DEV_DOCUMENTS: usize = 600;

/// Names a person and a city share. None is a name of the memos.
pub const SHARED: [&str; 48] = [
    "Abilene",
    "Alexandria",
    "Augusta",
    "Aurora",
    "Austin",
    "Bethany",
    "Beverly",
    "Bristol",
    "Brooklyn",
    "Camden",
    "Carmel",
    "Carson",
    "Charlotte",
    "Chelsea",
    "Clayton",
    "Cleveland",
    "Dallas",
    "Denver",
    "Devon",
    "Elizabeth",
    "Eugene",
    "Florence",
    "Geneva",
    "Helena",
    "Houston",
    "Irving",
    "Jackson",
    "Kimberley",
    "Lancaster",
    "Lincoln",
    "Lourdes",
    "Marion",
    "Memphis",
    "Odessa",
    "Olympia",
    "Orlando",
    "Phoenix",
    "Preston",
    "Regina",
    "Santiago",
    "Savannah",
    "Selma",
    "Shelby",
    "Sofia",
    "Sydney",
    "Trenton",
    "Tyler",
    "Valencia",
];

/// Given names that are no well-known city.
pub const PEOPLE: [&str; 30] = [
    "Anika", "Beatriz", "Caleb", "Dmitri", "Elena", "Emma", "Felix", "Greta", "Hannah", "Hugo", "Imani", "Ingrid",
    "Kenji", "Leila", "Liam", "Marcus", "Mateo", "Naomi", "Nikhil", "Noah", "Olivia", "Omar", "Priya", "Rashid",
    "Rosa", "Soren", "Tobias", "Tomas", "Wanjiru", "Yara",
];

/// Cities whose name is no usual given name.
pub const CITIES: [&str; 30] = [
    "Accra", "Bergen", "Bogota", "Boston", "Calgary", "Chicago", "Dublin", "Fresno", "Hanoi", "Jakarta", "Lagos",
    "Leeds", "Lisbon", "Lyon", "Munich", "Nairobi", "Omaha", "Osaka", "Ottawa", "Perth", "Prague", "Pune", "Quito",
    "Seattle", "Tampa", "Toronto", "Tucson", "Tulsa", "Turin", "Zagreb",
];

/// Sentences about a person; `{}` is the name, `{2}` a second person.
pub const PERSON_SENTENCES: [&str; 56] = [
    "{} wrote the summary before lunch.",
    "{} walked the dog around the block.",
    "{} mislaid the umbrella again.",
    "{} is stuck in traffic.",
    "{} was pleased with the outcome.",
    "{} will join us after all.",
    "{} has a cold and stayed in bed.",
    "{} wants to swap desks.",
    "{} brought cake for everyone.",
    "{} laughed and shook it off.",
    "{} needs a ride to the station.",
    "{} cooked dinner for the whole team.",
    "{} plays the violin on weekends.",
    "{} signed the card and passed it on.",
    "{} sat down next to me.",
    "{} borrowed my charger.",
    "{} can explain it better than I can.",
    "{} looked tired after the trip.",
    "{} waved from across the street.",
    "{} paid for the coffee.",
    "After that {} went quiet.",
    "Then {} opened the window.",
    "Luckily {} had a spare.",
    "Meanwhile {} set the table.",
    "We waited for {} outside.",
    "They picked {} for the team.",
    "I gave {} the spare ticket.",
    "Nobody told {} about the change.",
    "She helped {} carry the boxes.",
    "He taught {} how to swim.",
    "We surprised {} with a party.",
    "I sat beside {} at dinner.",
    "The dog followed {} everywhere.",
    "Everyone likes {}.",
    "I trust {} completely.",
    "It was a gift from {}.",
    "I shared a cab with {}.",
    "Hand the folder to {} when it is ready.",
    "We owe {} an apology.",
    "{} and {2} share an office.",
    "Both {} and {2} came by bike.",
    "It was {} who noticed the leak.",
    "My neighbor {} keeps bees.",
    "The idea came from {} and {2}.",
    "We asked around until {} turned up.",
    "I waited until {} was ready.",
    "Nobody left before {} spoke.",
    "She smiled when {} walked in.",
    "He said that {} would handle it.",
    "I think {} already knows.",
    "We clapped when {} sat down.",
    "They emailed {} yesterday.",
    "I ran into {} at the bakery.",
    "Someone should tell {} soon.",
    "Ask whether {} can stay late.",
    "By then {} had left for the day.",
];

/// Sentences about a city; `{}` is the name, `{2}` a second city.
pub const CITY_SENTENCES: [&str; 52] = [
    "{} gets crowded in summer.",
    "{} has three bridges over the river.",
    "{} is two hours north of here.",
    "{} began as a trading post.",
    "{} lies in a wide valley.",
    "{} has doubled in size since the railway came.",
    "{} banned cars from the old town.",
    "{} is known for its street food.",
    "{} faces the open sea.",
    "{} elected a new mayor.",
    "{} has about half a million people.",
    "{} is cheaper than the capital.",
    "{} will build a second airport.",
    "{} flooded twice that decade.",
    "{} was the place my grandparents settled.",
    "{} gets very little rain.",
    "Downtown {} empties out at night.",
    "Then {} came into view.",
    "{} has the better museums.",
    "{} raised its bus fares.",
    "We toured {} on foot.",
    "They explored {} by tram.",
    "The storm missed {} entirely.",
    "The river divides {} in two.",
    "I love {} in the autumn.",
    "The new line connects {} with the coast.",
    "We reached {} by nightfall.",
    "She moved to {} for work.",
    "The train from {} was full.",
    "They live near {}.",
    "He grew up in {}.",
    "The {} branch closes early.",
    "A {} court heard the case.",
    "The road between {} and {2} is closed.",
    "{} and {2} are sister cities.",
    "Tickets to {} sold out.",
    "It rained all week in {}.",
    "We passed through {} on the way.",
    "The mayor of {} resigned.",
    "Rents in {} keep climbing.",
    "The bus from {} to {2} takes all day.",
    "They drove to {} overnight.",
    "He travels to {} every month.",
    "Our flight to {} was delayed.",
    "We arrived in {} before dawn.",
    "The ferry leaves {} at noon.",
    "She trained in {} for a year.",
    "They are moving from {} to {2}.",
    "A letter came from {} this morning.",
    "The parcel was sent to {} by mistake.",
    "We stopped in {} for fuel.",
    "It is a long way from {} to the border.",
];

/// The words that join two sentences into one, so a name also stands
/// inside a sentence (`..., and Austin paid for the coffee`).
pub const JOINS: [&str; 6] = [", and ", ", but ", ", so ", " while ", " before ", " after "];

/// Sentences that name nobody and no place.
pub const PLAIN_SENTENCES: [&str; 12] = [
    "The printer is out of paper again.",
    "Nobody had read the notes.",
    "It took longer than we planned.",
    "The room was too warm.",
    "We agreed to try again later.",
    "The coffee machine is broken.",
    "That was not the plan.",
    "Nothing else needs doing.",
    "The rest can wait.",
    "It was a quiet week.",
    "The list keeps growing.",
    "Everything fit in one box.",
];

/// A fixed linear congruential generator, so the documents are the same
/// on every run.
struct Lcg(u64);

impl Lcg {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) as usize) % n
    }

    /// A name for a slot: three times in five one of the document's shared
    /// names, so a document has the same name as a person and as a city,
    /// else one of `own`.
    fn name(&mut self, pool: &[&'static str; 3], own: &[&'static str]) -> &'static str {
        if self.below(5) < 3 { pool[self.below(3)] } else { own[self.below(own.len())] }
    }
}

/// A sentence template, the names of its own kind and its label.
type Template = (&'static str, &'static [&'static str], &'static str);

fn template(rng: &mut Lcg) -> Template {
    match rng.below(9) {
        0 => (PLAIN_SENTENCES[rng.below(PLAIN_SENTENCES.len())], &[], ""),
        1..=4 => (PERSON_SENTENCES[rng.below(PERSON_SENTENCES.len())], &PEOPLE, "first_name"),
        _ => (CITY_SENTENCES[rng.below(CITY_SENTENCES.len())], &CITIES, "city"),
    }
}

/// One document's text and the spans of its names as (start, end, label).
fn document(rng: &mut Lcg) -> (String, Vec<(usize, usize, &'static str)>) {
    let mut text = String::new();
    let mut names = Vec::new();
    let pool = [(); 3].map(|()| SHARED[rng.below(SHARED.len())]);
    // Whether the sentence before was left open for a joining word.
    let mut open = false;
    for _ in 0..3 + rng.below(5) {
        let (template, own, label) = template(rng);
        let mut rest = template;
        if open {
            text.push_str(JOINS[rng.below(JOINS.len())]);
            // A joined sentence starts small, but for a name and `I`.
            if !rest.starts_with('{') && !rest.starts_with("I ") {
                text.extend(rest[..1].chars().flat_map(char::to_lowercase));
                rest = &rest[1..];
            }
        } else if !text.is_empty() {
            text.push(' ');
        }
        let mut first = "";
        while let Some(at) = rest.find('{') {
            text.push_str(&rest[..at]);
            let close = at + rest[at..].find('}').expect("a closed slot");
            // The second name of a sentence is another than the first.
            let mut name = rng.name(&pool, own);
            while name == first {
                name = rng.name(&pool, own);
            }
            first = name;
            names.push((text.len(), text.len() + name.len(), label));
            text.push_str(name);
            rest = &rest[close + 1..];
        }
        // One sentence in four is joined to the next.
        open = rng.below(4) == 0;
        text.push_str(if open { rest.trim_end_matches('.') } else { rest });
    }
    if open {
        text.push('.');
    }
    (text, names)
}

/// The documents added to the training and the development split, each
/// tokenized and labelled as the `Nemotron-PII` rows are.
///
/// # Errors
///
/// A span that does not fall on word boundaries, which no document here
/// has.
pub fn documents() -> Result<(Vec<Sentence>, Vec<Sentence>)> {
    let mut rng = Lcg(2398);
    let mut make = |count: usize, file: &str| -> Result<Vec<Sentence>> {
        let mut out = Vec::with_capacity(count);
        for row in 0..count {
            let (text, names) = document(&mut rng);
            let repr: Vec<String> = names
                .iter()
                .map(|(s, e, label)| format!("{{'start': {s}, 'end': {e}, 'text': 'x', 'label': '{label}'}}"))
                .collect();
            let sentence = spans::to_sentence(file, row, &text, &format!("[{}]", repr.join(", ")))?;
            out.push(sentence.expect("a document has words"));
        }
        Ok(out)
    };
    Ok((make(TRAIN_DOCUMENTS, "shared-names-train")?, make(DEV_DOCUMENTS, "shared-names-dev")?))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The names of the memos that score the head, which stay held out.
    const MEMO_NAMES: [&str; 15] = [
        "Paris", "Madison", "Columbus", "Salem", "Dayton", "Adelaide", "Raleigh", "Verona", "Kingston", "Milan",
        "Vienna", "Kobe", "Denton", "Windsor", "Irvine",
    ];

    #[test]
    fn the_lists_are_sorted_distinct_and_hold_no_name_of_the_memos() {
        for list in [&SHARED[..], &PEOPLE[..], &CITIES[..]] {
            assert!(list.windows(2).all(|w| w[0] < w[1]), "sorted and distinct: {list:?}");
            assert!(list.iter().all(|n| !MEMO_NAMES.contains(n)));
        }
        assert!(SHARED.iter().all(|n| !PEOPLE.contains(n) && !CITIES.contains(n)));
        assert!(PEOPLE.iter().all(|n| !CITIES.contains(n)));
    }

    #[test]
    fn every_name_is_labelled_and_nothing_else_is() {
        let (train, dev) = documents().unwrap();
        assert_eq!((train.len(), dev.len()), (TRAIN_DOCUMENTS, DEV_DOCUMENTS));
        let known = |w: &str| SHARED.contains(&w) || PEOPLE.contains(&w) || CITIES.contains(&w);
        let (mut person, mut city) = (0usize, 0usize);
        let mut as_person = std::collections::BTreeSet::new();
        let mut as_city = std::collections::BTreeSet::new();
        for s in train.iter().chain(&dev) {
            assert_eq!(s.words.len(), s.labels.len());
            for (w, l) in s.words.iter().zip(&s.labels) {
                match l.as_str() {
                    "B-first_name" => {
                        person += 1;
                        as_person.insert(w.as_str());
                    }
                    "B-city" => {
                        city += 1;
                        as_city.insert(w.as_str());
                    }
                    "O" => assert!(!known(w), "an unlabelled name {w} in {:?}", s.words),
                    other => panic!("label {other}"),
                }
                assert_eq!(l != "O", known(w), "{w} in {:?}", s.words);
            }
        }
        // Every shared name stands in both roles, and the roles are of like size.
        assert!(SHARED.iter().all(|n| as_person.contains(n) && as_city.contains(n)));
        assert!(person.abs_diff(city) * 5 < person + city, "{person} people, {city} cities");
    }

    #[test]
    fn the_documents_are_the_same_on_every_run_and_differ_between_splits() {
        let (train, dev) = documents().unwrap();
        let (again, _) = documents().unwrap();
        assert_eq!(train[0].words, again[0].words);
        assert_eq!(train[TRAIN_DOCUMENTS - 1].words, again[TRAIN_DOCUMENTS - 1].words);
        assert_ne!(train[0].words, dev[0].words);
    }

    #[test]
    fn no_sentence_of_the_memos_is_a_template() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../analysis/pii/tests/data/human-eval");
        for file in ["places-person-names.txt", "places-person-names-2.txt", "places-person-names-3.txt"] {
            let memo = std::fs::read_to_string(dir.join(file)).unwrap().replace('\n', " ");
            for t in PERSON_SENTENCES.iter().chain(&CITY_SENTENCES) {
                // The words after the name, as far as the next slot.
                let tail = t.rsplit('}').next().unwrap().trim_end_matches('.');
                assert!(tail.len() < 8 || !memo.contains(tail), "{file} holds {tail:?}");
            }
        }
    }
}
