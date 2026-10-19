//! Boundary regressions for the parser context generators (#675): inputs where OpenNLP's loops
//! run zero times or its arrays are read out of bounds must give OpenNLP's features or a typed
//! error, never a panic.

use std::collections::HashSet;
use std::str::FromStr;

use turbo_parser::parse::{Parse, ParseError};
use turbo_parser::treeinsert::TreeInsertError;
use turbo_parser::{EnglishHeadRules, chunking, treeinsert};

fn english_head_rules() -> EnglishHeadRules {
    let path = turbo_goldens::data_dir().join("fixtures/en-head-rules.txt");
    EnglishHeadRules::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn chunking_check_context_with_start_after_end_runs_no_pair_loop() {
    let mut top = Parse::parse("(TOP (NP (DT The) (NN dog)) (VP (VBD barked)) (. .))").unwrap();
    top.update_heads(&english_head_rules());
    let children = top.children();
    // OpenNLP's `for (int pi = start; pi < end; pi++)` loops run zero times when start > end.
    let features = chunking::CheckContextGenerator::new().get_context(children, "S", 1, 0);
    assert_eq!(
        features[..9],
        [
            "default",
            "fl=null",
            "cbegin=VP|barked|S",
            "cbegin*=VP|S",
            "clast=NP|dog|S",
            "clast*=NP|S",
            "p=S->NP",
            "pp=S->NP",
            "s-1=dog|S|NP"
        ]
    );
    assert!(!features.iter().any(|f| f.starts_with("cil=")), "{features:?}");
}

#[test]
fn treeinsert_check_context_on_a_parent_of_only_punctuation_is_an_error() {
    let top = Parse::parse("(TOP (S (, ,) (. .)))").unwrap();
    let parent = &top.children()[0];
    let punct: HashSet<String> = [",", "."].map(str::to_owned).into();
    let generator = treeinsert::CheckContextGenerator::new(punct);
    // OpenNLP reads `children[0]` of the collapsed children, which is empty.
    assert_eq!(
        generator.get_context(parent, &[parent], 0, false),
        Err(TreeInsertError::Parse(ParseError::TokenIndex { index: 0, len: 0 }))
    );
}
