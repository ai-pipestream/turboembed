//! The tree-insertion parser's right frontier and chunk labels, which need no model. The event
//! stream's features against OpenNLP's and the parser's derivations with a trained model are
//! tested in `turbo-train-components`' `tests/parser_treeinsert.rs`.

use std::str::FromStr;

use turbo_parser::{EnglishHeadRules, Parse, treeinsert};

fn english_head_rules() -> EnglishHeadRules {
    let path = turbo_goldens::data_dir().join("fixtures/en-head-rules.txt");
    EnglishHeadRules::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// The `INC` node of a bracketed `(INC ...)` parse, with heads.
fn inc_parse(bracketed: &str) -> Parse {
    let mut p = Parse::parse(bracketed).unwrap().children()[0].clone();
    p.update_heads(&english_head_rules());
    p
}

#[test]
fn right_frontier_starts_at_first_non_punctuation_child_of_top() {
    let rules = english_head_rules();
    let mut p =
        Parse::parse("(TOP (`` ``) (S (NP (DT The) (NN dog)) (VP (VBD barked) (NP (NNS cats)))) ('' ''))").unwrap();
    p.update_heads(&rules);
    let rf = treeinsert::Parser::get_right_frontier(&p, turbo_parser::HeadRules::punctuation_tags(&rules));
    let kinds: Vec<&str> = rf.iter().map(|n| n.kind()).collect();
    // Deepest first, ending at the S, part-of-speech nodes left out.
    assert_eq!(kinds, ["NP", "VP", "S"]);
}

#[test]
fn chunks_are_marked_complete_after_chunking() {
    let mut parses = vec![inc_parse("(INC (NP (DT The) (NN dog)) (VBD barked))")];
    treeinsert::Parser::mark_chunks_complete(&mut parses);
    let labels: Vec<Option<&str>> = parses[0].children().iter().map(Parse::label).collect();
    assert_eq!(labels, [Some("c"), Some("c")]);
}
