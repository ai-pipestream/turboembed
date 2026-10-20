//! Port of OpenNLP's `CorefAnnotatorTest`: sieve decisions on names,
//! types and pronouns, untyped entities, and input validation.

mod common;

use common::{Fixture, chain_ids, refusal};
use turbo_analysis::Span;
use turbo_coref::{CorefAnnotator, MentionKind, Origin, UNTYPED};

fn rules() -> CorefAnnotator {
    CorefAnnotator::default()
}

fn story() -> Fixture {
    Fixture::new()
        .sentence(&["Mary/PROPN", "Jones/PROPN", "leads/VERB", "Acme/PROPN", "Corp/PROPN", "./PUNCT"])
        .sentence(&["She/PRON", "joined/VERB", "Acme/PROPN", "in/ADP", "2020/NUM", "./PUNCT"])
        .sentence(&["It/PRON", "thrived/VERB", "./PUNCT"])
        .entity("Mary Jones", "person")
        .entity("Acme Corp", "organization")
        .entity("Acme", "organization")
        .entity("2020", "date")
}

#[test]
fn sieves_link_names_and_pronouns() {
    let f = story();
    assert_eq!(chain_ids(&f, &rules()), [0, 1, 0, 1, 2, 1]);
    let built = f.build();
    let (mentions, _, edges) = rules().resolve(&built.input()).unwrap();
    assert_eq!(mentions.origin[0], Origin::Entity);
    assert_eq!(mentions.origin[2], Origin::Pronoun);
    assert_eq!(mentions.kind[2], MentionKind::Pronominal);
    assert_eq!(mentions.entity.as_ref().unwrap()[0].map(|e| e.get()), Some(0));
    assert_eq!(mentions.entity.as_ref().unwrap()[2], None);
    let forms = common::forms(&built.text, &mentions);
    assert_eq!((forms[2].as_str(), forms[5].as_str()), ("She", "It"));
    // Six mentions in three chains: three links.
    assert_eq!(edges.from.len(), 3);
}

#[test]
fn same_text_different_type_stays_apart() {
    let f = Fixture::new()
        .sentence(&["Paris/PROPN", "met/VERB", "Paris/PROPN", "./PUNCT"])
        .entity("Paris", "person")
        .entity("Paris", "location");
    assert_eq!(chain_ids(&f, &rules()), [0, 1]);
}

#[test]
fn supplementary_digit_modifiers_keep_names_apart() {
    let f = Fixture::new()
        .sentence(&["\u{1D7D9}/CD", "Project/NNP", "launched/VBD", "./."])
        .sentence(&["\u{1D7DA}/CD", "Project/NNP", "failed/VBD", "./."])
        .entity("\u{1D7D9} Project", "organization")
        .entity("\u{1D7DA} Project", "organization");
    assert_eq!(chain_ids(&f, &rules()), [0, 1]);
}

#[test]
fn pronoun_window_limits_resolution() {
    let f = Fixture::new()
        .sentence(&["Acme/PROPN", "Corp/PROPN", "won/VERB", "./PUNCT"])
        .sentence(&["Time/NOUN", "passed/VERB", "./PUNCT"])
        .sentence(&["Markets/NOUN", "moved/VERB", "./PUNCT"])
        .sentence(&["Prices/NOUN", "rose/VERB", "./PUNCT"])
        .sentence(&["It/PRON", "changed/VERB", "./PUNCT"])
        .entity("Acme Corp", "organization");
    assert_eq!(chain_ids(&f, &rules()), [0, 1]);
}

#[test]
fn gendered_pronoun_skips_non_person_entities() {
    let f = Fixture::new()
        .sentence(&["Acme/NNP", "Corp/NNP", "hired/VBD", "John/NNP", "./."])
        .sentence(&["He/PRP", "started/VBD", "./."])
        .entity("Acme Corp", "organization")
        .entity("John", "person");
    assert_eq!(chain_ids(&f, &rules()), [0, 1, 1]);
}

#[test]
fn pronoun_at_document_start_stays_singleton() {
    let f = Fixture::new()
        .sentence(&["He/PRP", "waved/VBD", "./."])
        .sentence(&["John/NNP", "smiled/VBD", "./."])
        .entity("John", "person");
    assert_eq!(chain_ids(&f, &rules()), [0, 1]);
    let built = f.build();
    let (mentions, _, _) = rules().resolve(&built.input()).unwrap();
    assert_eq!(mentions.spans.get(turbo_analysis::Ix::new(0)), Span::new(0, 2));
    assert_eq!(mentions.origin[..], [Origin::Pronoun, Origin::Entity]);
}

#[test]
fn exact_name_match_links_across_distant_sentences() {
    let f = Fixture::new()
        .sentence(&["Alice/NNP", "arrived/VBD", "./."])
        .sentence(&["Rain/NNP", "fell/VBD", "./."])
        .sentence(&["Wind/NNP", "blew/VBD", "./."])
        .sentence(&["Alice/NNP", "left/VBD", "./."])
        .entity("Alice", "person")
        .entity("Alice", "person");
    assert_eq!(chain_ids(&f, &rules()), [0, 0]);
}

#[test]
fn nested_entity_mentions_keep_the_compound_name_apart() {
    let f = Fixture::new()
        .sentence(&["New/NNP", "York/NNP", "City/NNP", "honored/VBD", "New/NNP", "York/NNP", "./."])
        .entity("New York City", "location")
        .entity("New York", "location");
    let mut f = f;
    // The inner "New York" of the city, second in entity order.
    f.entities.insert(1, (Span::new(0, 8), "location".into()));
    let built = f.build();
    let (mentions, _, _) = rules().resolve(&built.input()).unwrap();
    assert_eq!(common::forms(&built.text, &mentions), ["New York City", "New York", "New York"]);
    assert_eq!(chain_ids(&f, &rules()), [0, 1, 1]);
}

#[test]
fn strict_head_match_links_reordered_name_and_accumulates_chain_words() {
    let two = Fixture::new()
        .sentence(&["The/DT", "Florida/NNP", "Supreme/NNP", "Court/NNP", "ruled/VBD", "./."])
        .sentence(&["The/DT", "Supreme/NNP", "Court/NNP", "of/IN", "Florida/NNP", "agreed/VBD", "./."])
        .entity("Florida Supreme Court", "organization")
        .entity("Supreme Court of Florida", "organization");
    assert_eq!(chain_ids(&two, &rules()), [0, 0]);
    let three = two
        .sentence(&["The/DT", "Florida/NNP", "Court/NNP", "closed/VBD", "./."])
        .entity("Florida Court", "organization");
    assert_eq!(chain_ids(&three, &rules()), [0, 0, 0]);
}

#[test]
fn strict_head_match_rejects_foreign_words_and_honors_the_type_guard() {
    let foreign = Fixture::new()
        .sentence(&["Stanford/NNP", "University/NNP", "welcomed/VBD", "guests/NNS", "./."])
        .sentence(&["Harvard/NNP", "University/NNP", "declined/VBD", "./."])
        .entity("Stanford University", "organization")
        .entity("Harvard University", "organization");
    assert_eq!(chain_ids(&foreign, &rules()), [0, 1]);
    let typed = Fixture::new()
        .sentence(&["Acme/NNP", "Court/NNP", "expanded/VBD", "./."])
        .sentence(&["Court/NNP", "is/VBZ", "a/DT", "town/NN", "./."])
        .entity("Acme Court", "organization")
        .entity("Court", "location");
    assert_eq!(chain_ids(&typed, &rules()), [0, 1]);
}

#[test]
fn pronoun_classes_follow_entity_types() {
    let neutral = Fixture::new()
        .sentence(&["John/NNP", "slept/VBD", "./."])
        .sentence(&["It/PRP", "rained/VBD", "./."])
        .entity("John", "person");
    assert_eq!(chain_ids(&neutral, &rules()), [0, 1]);
    let org = Fixture::new()
        .sentence(&["Acme/NNP", "grew/VBD", "./."])
        .sentence(&["They/PRP", "hired/VBD", "./."])
        .entity("Acme", "organization");
    assert_eq!(chain_ids(&org, &rules()), [0, 0]);
    let people = Fixture::new()
        .sentence(&["Smiths/NNPS", "won/VBD", "./."])
        .sentence(&["They/PRP", "cheered/VBD", "./."])
        .entity("Smiths", "person");
    assert_eq!(chain_ids(&people, &rules()), [0, 0]);
}

#[test]
fn first_and_second_person_pronouns_chain_per_speaker_not_to_entities() {
    let f = Fixture::new()
        .sentence(&["John/NNP", "spoke/VBD", "./."])
        .sentence(&["I/PRP", "listened/VBD", "./."])
        .sentence(&["You/PRP", "nodded/VBD", "./."])
        .entity("John", "person");
    assert_eq!(chain_ids(&f, &rules()), [0, 1, 2]);
}

#[test]
fn pronoun_token_inside_entity_is_not_a_mention() {
    let f = Fixture::new()
        .sentence(&["It/PRP", "Corp/NNP", "grew/VBD", "./."])
        .sentence(&["It/PRP", "hired/VBD", "./."])
        .entity("It Corp", "organization");
    assert_eq!(chain_ids(&f, &rules()), [0, 0]);
}

#[test]
fn type_labels_match_case_insensitively() {
    let upper = Fixture::new()
        .sentence(&["Mary/NNP", "spoke/VBD", "./."])
        .sentence(&["She/PRP", "left/VBD", "./."])
        .entity("Mary", "PERSON");
    assert_eq!(chain_ids(&upper, &rules()), [0, 0]);
    let mut custom = upper.clone();
    custom.entities[0].1 = "Per".into();
    let annotator = CorefAnnotator::rules(&["PER"], &["ORG"]).unwrap();
    assert_eq!(chain_ids(&custom, &annotator), [0, 0]);
}

#[test]
fn untyped_entities_resolve_without_bridging_known_types() {
    let mixed = Fixture::new()
        .sentence(&["John/NNP", "Smith/NNP", "joined/VBD", "Acme/NNP", "./."])
        .sentence(&["Smith/NNP", "praised/VBD", "it/PRP", "./."])
        .sentence(&["He/PRP", "stayed/VBD", "./."])
        .entity("John Smith", "person")
        .entity("Acme", UNTYPED)
        .entity("Smith", "person");
    assert_eq!(chain_ids(&mixed, &rules()), [0, 1, 0, 2, 0]);
    for (a, b) in [("person", UNTYPED), (UNTYPED, "person")] {
        let f = Fixture::new()
            .sentence(&["Paris/PROPN", "met/VERB", "Paris/PROPN", "./PUNCT"])
            .entity("Paris", a)
            .entity("Paris", b);
        assert_eq!(chain_ids(&f, &rules()), [0, 0], "{a} then {b}");
    }
    for (a, c) in [("person", "location"), ("location", "person")] {
        let f = Fixture::new()
            .sentence(&["Georgia/PROPN", "met/VERB", "Georgia/PROPN", "in/ADP", "Georgia/PROPN", "./PUNCT"])
            .entity("Georgia", a)
            .entity("Georgia", UNTYPED)
            .entity("Georgia", c);
        assert_eq!(chain_ids(&f, &rules()), [0, 0, 1], "{a} then {c}");
    }
}

#[test]
fn containment_links_typed_and_untyped_name_forms() {
    let suffix = Fixture::new()
        .sentence(&["Barack/PROPN", "Obama/PROPN", "spoke/VERB", "./PUNCT"])
        .sentence(&["Obama/PROPN", "left/VERB", "./PUNCT"])
        .entity("Barack Obama", "person")
        .entity("Obama", UNTYPED);
    assert_eq!(chain_ids(&suffix, &rules()), [0, 0]);
    let prefix = Fixture::new()
        .sentence(&["Obama/PROPN", "arrived/VERB", "./PUNCT"])
        .sentence(&["Barack/PROPN", "Obama/PROPN", "spoke/VERB", "./PUNCT"])
        .entity("Obama", UNTYPED)
        .entity("Barack Obama", "person");
    assert_eq!(chain_ids(&prefix, &rules()), [0, 0]);
}

#[test]
fn no_mentions_and_no_tokens_yield_empty_layers() {
    let f = Fixture::new().sentence(&["Dogs/NNS", "bark/VBP", "./."]);
    let built = f.build();
    let (mentions, groups, edges) = rules().resolve(&built.input()).unwrap();
    assert_eq!((mentions.spans.len(), groups.end.len(), edges.from.len()), (0, 0, 0));
    let blank = Fixture::new().build();
    let (mentions, groups, _) = rules().resolve(&blank.input()).unwrap();
    assert_eq!((mentions.spans.len(), groups.end.len()), (0, 0));
}

#[test]
fn invalid_type_sets_and_thresholds_are_rejected() {
    assert!(CorefAnnotator::rules(&[], &["organization"]).is_err());
    assert!(CorefAnnotator::rules(&[" "], &["organization"]).is_err());
    assert!(CorefAnnotator::rules(&["person"], &[]).is_err());
    assert!(CorefAnnotator::rules(&["person"], &[" "]).is_err());
    assert!(CorefAnnotator::new(&["person"], &["org"], None, 1.5, None, None).is_err());
    assert!(CorefAnnotator::new(&["person"], &["org"], None, f64::NAN, None, None).is_err());
}

fn two_sentences() -> Fixture {
    Fixture::new().sentence(&["Alice/NNP", "left/VBD", "./."]).sentence(&["She/PRP", "stayed/VBD", "./."])
}

#[test]
fn rejects_blank_entity_spans_and_types() {
    let mut blank_type = Fixture::new().sentence(&["Ada/NNP", "./."]).entity("Ada", "person");
    blank_type.entities[0].1 = " ".into();
    assert!(refusal(&blank_type.build()).contains("non-blank type"));
    let mut empty = Fixture::new().sentence(&["Ada/NNP", "./."]).entity("Ada", "person");
    empty.entities[0].0 = Span::new(0, 0);
    assert!(refusal(&empty.build()).contains("non-blank text"));
    let mut outside = two_sentences().entity("Alice", "person");
    outside.entities[0].0 = Span::new(0, 15);
    assert!(refusal(&outside.build()).contains("outside the sentence layer"));
}

#[test]
fn rejects_malformed_chunk_phrase_and_speaker_layers() {
    let mut crossing = two_sentences().chunk("Alice", "NP");
    crossing.chunks[0].0 = Span::new(0, 16);
    assert!(refusal(&crossing.build()).contains("crosses a sentence boundary"));
    let mut unaligned = two_sentences().chunk("Alice", "NP");
    unaligned.chunks[0].0 = Span::new(1, 5);
    assert!(refusal(&unaligned.build()).contains("token boundaries"));
    let mut phrase = two_sentences().phrase("NP", 0, "Alice left", "Alice");
    phrase.phrases[0].0 = Span::new(0, 16);
    assert!(refusal(&phrase.build()).contains("crosses a sentence boundary"));
    let blank_speaker = two_sentences().speaker(0, " ");
    assert!(refusal(&blank_speaker.build()).contains("non-blank label"));
}

#[test]
fn rejects_misaligned_tags_and_layers_in_another_view() {
    let f = two_sentences();
    let mut built = f.build();
    built.tags.tags.pop();
    assert!(refusal(&built).contains("aligned"));
    let built = f.build();
    let mut input = built.input();
    input.view = turbo_analysis::ViewId(3);
    assert!(rules().resolve(&input).unwrap_err().contains("view"));
}
