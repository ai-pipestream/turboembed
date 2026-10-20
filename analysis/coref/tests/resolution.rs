//! Port of OpenNLP's `CorefResolutionTest`: the decisions of the
//! entity-centric sieves on agreement, nominal heads, names, speakers
//! and mention detection.

mod common;

use common::{Fixture, chains, forms, v};
use turbo_analysis::{Ix, Span};
use turbo_coref::{CorefAnnotator, MentionKind, Origin};

#[test]
fn gendered_pronoun_skips_nearer_name_of_other_gender() {
    let f = Fixture::new()
        .sentence(&["John/NNP", "greeted/VBD", "Mary/NNP", "./."])
        .sentence(&["He/PRP", "smiled/VBD", "./."])
        .entity("John", "person")
        .entity("Mary", "person");
    assert_eq!(chains(&f), v(&[&["John", "He"]]));
}

#[test]
fn feminine_pronoun_skips_nearer_male_name() {
    let f = Fixture::new()
        .sentence(&["Mary/NNP", "greeted/VBD", "John/NNP", "./."])
        .sentence(&["She/PRP", "smiled/VBD", "./."])
        .entity("Mary", "person")
        .entity("John", "person");
    assert_eq!(chains(&f), v(&[&["Mary", "She"]]));
}

#[test]
fn title_sets_gender_when_first_name_is_absent() {
    let f = Fixture::new()
        .sentence(&["Mr./NNP", "Kowalczyk/NNP", "met/VBD", "Mrs./NNP", "Nowak/NNP", "./."])
        .sentence(&["She/PRP", "spoke/VBD", "./."])
        .entity("Mr. Kowalczyk", "person")
        .entity("Mrs. Nowak", "person");
    assert_eq!(chains(&f), v(&[&["Mrs. Nowak", "She"]]));
}

#[test]
fn first_name_sets_gender_when_name_finder_misses_both_people() {
    let f = Fixture::new()
        .sentence(&["Alice/NNP", "met/VBD", "John/NNP", "./."])
        .sentence(&["She/PRP", "smiled/VBD", "./."])
        .chunk("Alice", "NP")
        .chunk("John", "NP")
        .chunk("She", "NP");
    assert_eq!(chains(&f), v(&[&["Alice", "She"]]));
}

#[test]
fn title_sets_gender_when_name_finder_misses_both_people() {
    let f = Fixture::new()
        .sentence(&["Mrs./NNP", "Nowak/NNP", "met/VBD", "Mr./NNP", "Kowalczyk/NNP", "./."])
        .sentence(&["She/PRP", "spoke/VBD", "./."])
        .chunk("Mrs. Nowak", "NP")
        .chunk("Mr. Kowalczyk", "NP")
        .chunk("She", "NP");
    assert_eq!(chains(&f), v(&[&["Mrs. Nowak", "She"]]));
}

#[test]
fn unknown_proper_noun_phrase_rejects_neutral_pronoun_and_accepts_animate() {
    let it = Fixture::new()
        .sentence(&["Kowalczyk/NNP", "arrived/VBD", "./."])
        .sentence(&["It/PRP", "expanded/VBD", "./."])
        .chunk("Kowalczyk", "NP")
        .chunk("It", "NP");
    assert!(chains(&it).is_empty());
    let she = Fixture::new()
        .sentence(&["Kowalczyk/NNP", "arrived/VBD", "./."])
        .sentence(&["She/PRP", "spoke/VBD", "./."])
        .chunk("Kowalczyk", "NP")
        .chunk("She", "NP");
    assert_eq!(chains(&she), v(&[&["Kowalczyk", "She"]]));
}

#[test]
fn unknown_gender_stays_compatible_with_both_pronouns() {
    let f = Fixture::new()
        .sentence(&["Kowalczyk/NNP", "arrived/VBD", "./."])
        .sentence(&["She/PRP", "spoke/VBD", "./."])
        .entity("Kowalczyk", "person");
    assert_eq!(chains(&f), v(&[&["Kowalczyk", "She"]]));
}

#[test]
fn plural_pronoun_skips_singular_person() {
    let f = Fixture::new()
        .sentence(&["The/DT", "senators/NNS", "met/VBD", "John/NNP", "./."])
        .sentence(&["They/PRP", "voted/VBD", "./."])
        .entity("John", "person")
        .chunk("The senators", "NP")
        .chunk("John", "NP")
        .chunk("They", "NP");
    assert_eq!(chains(&f), v(&[&["The senators", "They"]]));
}

#[test]
fn compound_place_name_is_not_the_bare_name() {
    let f = Fixture::new()
        .sentence(&["Kansas/NNP", "City/NNP", "grew/VBD", "./."])
        .sentence(&["Kansas/NNP", "voted/VBD", "./."])
        .sentence(&["Kansas/NNP", "City/NNP", "won/VBD", "./."])
        .entity("Kansas City", "location")
        .entity("Kansas", "location")
        .entity("Kansas City", "location");
    assert_eq!(chains(&f), v(&[&["Kansas City", "Kansas City"]]));
}

#[test]
fn surname_finds_full_person_name() {
    let f = Fixture::new()
        .sentence(&["Mary/NNP", "Jones/NNP", "spoke/VBD", "./."])
        .sentence(&["Jones/NNP", "left/VBD", "./."])
        .sentence(&["Mary/NNP", "returned/VBD", "./."])
        .entity("Mary Jones", "person")
        .entity("Jones", "person")
        .entity("Mary", "person");
    assert_eq!(chains(&f), v(&[&["Mary Jones", "Jones", "Mary"]]));
}

#[test]
fn nominal_mention_links_by_head_to_entity_cluster() {
    let f = Fixture::new()
        .sentence(&["Aberdeen/NNP", "Grammar/NNP", "School/NNP", "opened/VBD", "./."])
        .sentence(&["The/DT", "school/NN", "grew/VBD", "./."])
        .entity("Aberdeen Grammar School", "organization")
        .chunk("Aberdeen Grammar School", "NP")
        .chunk("The school", "NP");
    assert_eq!(chains(&f), v(&[&["Aberdeen Grammar School", "The school"]]));
}

#[test]
fn chunk_headed_by_entity_widens_the_mention() {
    let f = Fixture::new()
        .sentence(&["The/DT", "poet/NN", "Byron/NNP", "wrote/VBD", "./."])
        .sentence(&["He/PRP", "travelled/VBD", "./."])
        .entity("Byron", "person")
        .chunk("The poet Byron", "NP")
        .chunk("He", "NP");
    let built = f.build();
    let (mentions, _, _) = CorefAnnotator::default().resolve(&built.input()).unwrap();
    assert_eq!(mentions.spans.len(), 2);
    assert_eq!(mentions.spans.get(Ix::new(0)), Span::new(0, 14));
    assert_eq!(mentions.origin[0], Origin::Entity);
    assert_eq!(mentions.kind[0], MentionKind::Proper);
    assert_eq!(mentions.entity_of(Ix::new(0)), Some(Ix::new(0)));
    assert_eq!(chains(&f), v(&[&["The poet Byron", "He"]]));
}

#[test]
fn standalone_demonstrative_is_a_pronoun_mention_the_rules_leave_alone() {
    let f = Fixture::new()
        .sentence(&["The/DT", "plan/NN", "failed/VBD", "./."])
        .sentence(&["That/DT", "was/VBD", "costly/JJ", "./."])
        .sentence(&["That/DT", "plan/NN", "hurt/VBD", "./."])
        .chunk("The plan", "NP")
        .chunk("That", "NP")
        .chunk("That plan", "NP");
    let built = f.build();
    let (mentions, _, _) = CorefAnnotator::default().resolve(&built.input()).unwrap();
    assert_eq!(forms(&built.text, &mentions), ["The plan", "That", "That plan"]);
    assert_eq!(mentions.origin[1], Origin::Pronoun);
    assert_eq!(mentions.kind[1], MentionKind::Demonstrative);
    assert_eq!(chains(&f), v(&[&["The plan", "That plan"]]));
}

#[test]
fn noun_phrase_with_of_phrase_is_a_candidate_beside_its_parts() {
    let f = Fixture::new()
        .sentence(&["The/DT", "law/NN", "of/IN", "negligence/NN", "applies/VBZ", "./."])
        .sentence(&["The/DT", "law/NN", "of/IN", "negligence/NN", "is/VBZ", "old/JJ", "./."])
        .chunk("The law", "NP")
        .chunk("of", "PP")
        .chunk("negligence", "NP")
        .chunk("The law", "NP")
        .chunk("of", "PP")
        .chunk("negligence", "NP");
    let chains = chains(&f);
    assert!(
        chains.contains(&vec!["The law of negligence".to_owned(), "The law of negligence".to_owned()]),
        "{chains:?}"
    );
    assert!(chains.contains(&vec!["The law".to_owned(), "The law".to_owned()]), "{chains:?}");
}

#[test]
fn parser_and_chunk_noun_phrases_are_combined_without_duplicates() {
    let f = Fixture::new()
        .sentence(&["The/DT", "cat/NN", "saw/VBD", "Alice/NNP", "./."])
        .chunk("The cat", "NP")
        .chunk("Alice", "NP")
        .phrase("NP", 0, "Alice", "Alice");
    let built = f.build();
    let (mentions, _, _) = CorefAnnotator::default().resolve(&built.input()).unwrap();
    assert_eq!(forms(&built.text, &mentions), ["The cat", "Alice"]);
}

#[test]
fn parser_appositive_keeps_distinct_chunk_noun_phrases() {
    let f = Fixture::new()
        .sentence(&[
            "Joshua/NNP",
            "Norton/NNP",
            ",/,",
            "known/VBN",
            "as/IN",
            "Emperor/NNP",
            "Norton/NNP",
            "spoke/VBD",
            "./.",
        ])
        .chunk("Joshua Norton", "NP")
        .chunk("Emperor Norton", "NP")
        .phrase("NP", 0, "Joshua Norton , known as Emperor Norton", "Norton")
        .phrase("NP", 2, "Emperor Norton", "Norton");
    let built = f.build();
    let (mentions, _, _) = CorefAnnotator::default().resolve(&built.input()).unwrap();
    assert_eq!(
        forms(&built.text, &mentions),
        ["Joshua Norton , known as Emperor Norton", "Joshua Norton", "Emperor Norton"]
    );
}

#[test]
fn coordinated_noun_phrases_form_a_plural_mention() {
    let f = Fixture::new()
        .sentence(&["Kim/NNP", "and/CC", "Lee/NNP", "arrived/VBD", "./."])
        .sentence(&["They/PRP", "left/VBD", "./."])
        .entity("Kim", "person")
        .entity("Lee", "person")
        .chunk("Kim", "NP")
        .chunk("Lee", "NP")
        .chunk("They", "NP");
    assert_eq!(chains(&f), v(&[&["Kim and Lee", "They"]]));
}

#[test]
fn indefinite_noun_phrase_is_never_anaphoric() {
    let f = Fixture::new()
        .sentence(&["A/DT", "student/NN", "arrived/VBD", "./."])
        .sentence(&["A/DT", "student/NN", "left/VBD", "./."])
        .sentence(&["The/DT", "student/NN", "waved/VBD", "./."])
        .chunk("A student", "NP")
        .chunk("A student", "NP")
        .chunk("The student", "NP");
    assert_eq!(chains(&f), v(&[&["A student", "The student"]]));
}

#[test]
fn pleonastic_it_is_no_mention() {
    let f = Fixture::new()
        .sentence(&["Acme/NNP", "grew/VBD", "./."])
        .sentence(&["It/PRP", "is/VBZ", "clear/JJ", "that/IN", "it/PRP", "won/VBD", "./."])
        .entity("Acme", "organization");
    let built = f.build();
    let (mentions, _, _) = CorefAnnotator::default().resolve(&built.input()).unwrap();
    assert_eq!(forms(&built.text, &mentions), ["Acme", "it"]);
    assert_eq!(chains(&f), v(&[&["Acme", "it"]]));
}

#[test]
fn pronoun_never_links_to_phrase_containing_it() {
    let f = Fixture::new()
        .sentence(&["Byron/NNP", "wrote/VBD", "./."])
        .sentence(&["His/PRP$", "mother/NN", "read/VBD", "./."])
        .entity("Byron", "person")
        .chunk("Byron", "NP")
        .chunk("His mother", "NP");
    assert_eq!(chains(&f), v(&[&["Byron", "His"]]));
}

#[test]
fn reflexive_stays_in_its_sentence() {
    let f = Fixture::new()
        .sentence(&["Mary/NNP", "left/VBD", "./."])
        .sentence(&["John/NNP", "hurt/VBD", "himself/PRP", "./."])
        .entity("Mary", "person")
        .entity("John", "person");
    assert_eq!(chains(&f), v(&[&["John", "himself"]]));
}

#[test]
fn acronym_finds_its_expansion() {
    let f = Fixture::new()
        .sentence(&["The/DT", "World/NNP", "Health/NNP", "Organization/NNP", "met/VBD", "./."])
        .sentence(&["WHO/NNP", "agreed/VBD", "./."])
        .entity("World Health Organization", "organization")
        .entity("WHO", "organization");
    assert_eq!(chains(&f), v(&[&["World Health Organization", "WHO"]]));
}

#[test]
fn acronym_supports_supplementary_uppercase_initials() {
    let (a, b) = ("\u{10400}", "\u{10401}");
    let (first, second) = (format!("{a}lpha"), format!("{b}eta"));
    let full = format!("{first} {second}");
    let acronym = format!("{a}{b}");
    let f = Fixture::new()
        .sentence(&[&format!("{first}/NNP"), &format!("{second}/NNP"), "met/VBD", "./."])
        .sentence(&[&format!("{acronym}/NNP"), "agreed/VBD", "./."])
        .entity(&full, "organization")
        .entity(&acronym, "organization");
    assert_eq!(chains(&f), vec![vec![full, acronym]]);
}

#[test]
fn shared_institution_head_with_different_names_stays_apart() {
    let f = Fixture::new()
        .sentence(&["Harvard/NNP", "University/NNP", "grew/VBD", "./."])
        .sentence(&["Stanford/NNP", "University/NNP", "grew/VBD", "./."])
        .sentence(&["President/NNP", "Obama/NNP", "spoke/VBD", "./."])
        .sentence(&["Barack/NNP", "Obama/NNP", "left/VBD", "./."])
        .entity("Harvard University", "organization")
        .entity("Stanford University", "organization")
        .entity("President Obama", "person")
        .entity("Barack Obama", "person");
    assert_eq!(chains(&f), v(&[&["President Obama", "Barack Obama"]]));
}

#[test]
fn pronoun_reaches_three_sentences_back() {
    let f = Fixture::new()
        .sentence(&["Acme/NNP", "won/VBD", "./."])
        .sentence(&["Time/NN", "passed/VBD", "./."])
        .sentence(&["Markets/NNS", "moved/VBD", "./."])
        .sentence(&["It/PRP", "changed/VBD", "./."])
        .entity("Acme", "organization");
    assert_eq!(chains(&f), v(&[&["Acme", "It"]]));
}

#[test]
fn parse_phrases_give_maximal_mentions_with_parser_heads() {
    let f = Fixture::new()
        .sentence(&["The/DT", "school/NN", "of/IN", "Dr./NNP", "Glennie/NNP", "opened/VBD", "./."])
        .sentence(&["The/DT", "school/NN", "grew/VBD", "./."])
        .sentence(&["Glennie/NNP", "left/VBD", "./."])
        .entity("Dr. Glennie", "person")
        .entity("Glennie", "person")
        .phrase("NP", 0, "The school of Dr. Glennie", "school")
        .phrase("NP", 0, "The school", "school")
        .phrase("NP", 0, "Dr. Glennie", "Glennie")
        .phrase("NP", 26, "The school", "school")
        .phrase("NP", 38, "Glennie", "Glennie");
    let built = f.build();
    let (mentions, _, _) = CorefAnnotator::default().resolve(&built.input()).unwrap();
    assert_eq!(forms(&built.text, &mentions), ["The school of Dr. Glennie", "Dr. Glennie", "The school", "Glennie"]);
    assert_eq!(chains(&f), v(&[&["The school of Dr. Glennie", "The school"], &["Dr. Glennie", "Glennie"]]));
}

#[test]
fn first_person_chains_per_quoted_speaker() {
    let f = Fixture::new()
        .sentence(&["\"/``", "I/PRP", "am/VBP", "tired/JJ", ",/,", "\"/''", "said/VBD", "Mary/NNP", "./."])
        .sentence(&["\"/``", "I/PRP", "agree/VBP", "./.", "\"/''"])
        .sentence(&["I/PRP", "left/VBD", "./."])
        .entity("Mary", "person");
    assert_eq!(chains(&f), v(&[&["I", "Mary"]]));
}

#[test]
fn speakers_layer_chains_first_and_second_person_per_speaker() {
    let f = Fixture::new()
        .sentence(&["I/PRP", "think/VBP", "you/PRP", "know/VBP", "./."])
        .sentence(&["I/PRP", "do/VBP", "./."])
        .sentence(&["We/PRP", "agree/VBP", "./."])
        .sentence(&["I/PRP", "hope/VBP", "so/RB", "./."])
        .speaker(0, "A")
        .speaker(1, "B")
        .speaker(2, "A")
        .speaker(3, "A");
    assert_eq!(chains(&f), v(&[&["I", "I"]]));
    let built = f.build();
    let (mentions, groups, _) = CorefAnnotator::default().resolve(&built.input()).unwrap();
    assert_eq!(mentions.spans.len(), 5);
    let chain_of = |m: u32| groups.iter().position(|(_, g)| g.contains(&Ix::new(m))).unwrap();
    assert_eq!(chain_of(0), chain_of(4));
    assert_ne!(chain_of(0), chain_of(2));
    assert_ne!(chain_of(0), chain_of(3));
}

#[test]
fn narrator_first_person_chains_without_speakers() {
    let f = Fixture::new()
        .sentence(&["I/PRP", "wrote/VBD", "./."])
        .sentence(&["My/PRP$", "friend/NN", "read/VBD", "it/PRP", "./."])
        .sentence(&["I/PRP", "smiled/VBD", "./."]);
    assert_eq!(chains(&f), v(&[&["I", "My", "I"]]));
}

#[test]
fn neutral_pronoun_skips_people_and_animate_nouns() {
    let f = Fixture::new()
        .sentence(&["The/DT", "company/NN", "hired/VBD", "the/DT", "teacher/NN", "./."])
        .sentence(&["It/PRP", "grew/VBD", "./."])
        .chunk("The company", "NP")
        .chunk("the teacher", "NP")
        .chunk("It", "NP");
    assert_eq!(chains(&f), v(&[&["The company", "It"]]));
}
