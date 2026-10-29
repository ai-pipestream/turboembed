//! Noun and verb chunks read off UD parses written here by hand, and the
//! annotator over a document.

use turbo_phrase_chunker::{ChunkKind, derive};

/// `(word, tag, head counted from 1 with 0 the root, relation)` rows.
fn chunks(rows: &[(&str, &str, usize, &str)]) -> Vec<(String, &'static str)> {
    let words: Vec<&str> = rows.iter().map(|r| r.0).collect();
    let tags: Vec<&str> = rows.iter().map(|r| r.1).collect();
    let heads: Vec<Option<usize>> = rows.iter().map(|r| r.2.checked_sub(1)).collect();
    let rels: Vec<&str> = rows.iter().map(|r| r.3).collect();
    derive(&tags, &heads, &rels, &words)
        .into_iter()
        .map(|c| (words[c.start..c.end].join(" "), c.kind.label()))
        .collect()
}

#[test]
fn base_noun_phrases_and_verb_groups() {
    // The very large dog has not eaten John 's two red apples .
    let rows = [
        ("The", "DET", 3, "det"),
        ("very", "ADV", 3, "advmod"),
        ("large", "ADJ", 4, "amod"),
        ("dog", "NOUN", 7, "nsubj"),
        ("has", "AUX", 7, "aux"),
        ("not", "PART", 7, "advmod"),
        ("eaten", "VERB", 0, "root"),
        ("John", "PROPN", 12, "nmod:poss"),
        ("'s", "PART", 8, "case"),
        ("two", "NUM", 12, "nummod"),
        ("red", "ADJ", 12, "amod"),
        ("apples", "NOUN", 7, "obj"),
        (".", "PUNCT", 7, "punct"),
    ];
    assert_eq!(
        chunks(&rows),
        [("The very large dog".into(), "NP"), ("has not eaten".into(), "VP"), ("John 's two red apples".into(), "NP")]
    );
}

#[test]
fn an_adverb_of_an_adjective_and_a_determiner_are_inside() {
    let rows = [
        ("The", "DT", 4, "det"),
        ("very", "RB", 3, "advmod"),
        ("large", "JJ", 4, "amod"),
        ("dog", "NN", 5, "nsubj"),
        ("barked", "VBD", 0, "root"),
    ];
    assert_eq!(chunks(&rows), [("The very large dog".into(), "NP"), ("barked".into(), "VP")]);
}

#[test]
fn prepositions_conjunctions_and_clauses_stay_outside() {
    // We met in the old house and the garden .
    let rows = [
        ("We", "PRON", 2, "nsubj"),
        ("met", "VERB", 0, "root"),
        ("in", "ADP", 5, "case"),
        ("the", "DET", 5, "det"),
        ("old", "ADJ", 6, "amod"),
        ("house", "NOUN", 2, "obl"),
        ("and", "CCONJ", 9, "cc"),
        ("the", "DET", 9, "det"),
        ("garden", "NOUN", 6, "conj"),
        (".", "PUNCT", 2, "punct"),
    ];
    let rows: Vec<_> = rows.iter().map(|r| if r.0 == "the" && r.2 == 5 { (r.0, r.1, 6, r.3) } else { *r }).collect();
    assert_eq!(
        chunks(&rows),
        [("We".into(), "NP"), ("met".into(), "VP"), ("the old house".into(), "NP"), ("the garden".into(), "NP")]
    );
}

#[test]
fn a_participle_inside_a_noun_chunk_is_not_a_verb_group() {
    // the running water flows
    let rows = [
        ("the", "DT", 3, "det"),
        ("running", "VBG", 3, "amod"),
        ("water", "NN", 4, "nsubj"),
        ("flows", "VBZ", 0, "root"),
    ];
    assert_eq!(chunks(&rows), [("the running water".into(), "NP"), ("flows".into(), "VP")]);
}

#[test]
fn a_dependent_across_a_gap_is_left_out() {
    // the , dog : the det of dog, but a comma between that is not in it.
    let rows = [("the", "DT", 3, "det"), (",", ",", 3, "punct"), ("dog", "NN", 0, "root")];
    assert_eq!(chunks(&rows), [("dog".into(), "NP")]);
    assert_eq!(ChunkKind::Verb.label(), "VP");
    assert!(chunks(&[]).is_empty());
}
