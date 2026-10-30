//! The token split head: split labels, the decode that takes the best
//! label that fits, and the words a split gives.

use super::boundary_tests::spec;
use super::edit_tree::EditTree;
use super::split::{TokenSplitter, parse};
use crate::manifest::{Decode, HeadKind};

fn label(form: &str, words: &[&str]) -> String {
    words.iter().map(|w| EditTree::mine(form, w).label()).collect::<Vec<_>>().join(" ")
}

#[test]
fn a_split_label_is_its_trees_separated_by_one_space() {
    let del = label("del", &["de", "el"]);
    assert_eq!(del, "(m 0 1 () (s 'l' '')) (m 1 0 (s 'd' '') ())");
    let trees = parse(&del).unwrap();
    assert_eq!(trees.len(), 2);
    assert_eq!(trees.iter().map(|t| t.apply("del").unwrap()).collect::<Vec<_>>(), ["de", "el"]);
    // Quotes may hold spaces and parentheses.
    let odd = label("a b)", &["a", " b)"]);
    assert_eq!(parse(&odd).unwrap()[1].apply("a b)").unwrap(), " b)");
}

#[test]
fn a_label_that_is_not_two_to_eight_trees_is_refused() {
    let one = EditTree::mine("del", "de").label();
    let del = label("del", &["de", "el"]);
    for bad in [
        one.clone(),
        format!("{one}  {one}"),
        format!("{one}{one}"),
        format!("{one} "),
        format!(" {del}"),
        format!("{del} x"),
        format!("{one} (m 0 1 ()"),
        "_".into(),
        [one.as_str(); 9].join(" "),
    ] {
        assert!(parse(&bad).is_err(), "{bad:?}");
    }
    assert_eq!(parse(&[one.as_str(); 8].join(" ")).unwrap().len(), 8);
}

#[test]
fn the_best_label_that_fits_each_token_is_taken() {
    let del = label("del", &["de", "el"]);
    let della = label("della", &["di", "la"]);
    let s = spec(HeadKind::TokenSplit, &["_", &del, &della], Decode::EditTree);
    s.validate("h").unwrap();
    let get = |k: &str| Some(vec![0.0; if k == "out.bias" { 3 } else { 9 }]);
    let h = TokenSplitter::from_tensors(&s, 3, &get).unwrap();
    let forms = ["del", "casa", "della"];
    // The best label of each: del's split, del's split (which does not fit
    // "casa"), della's split.
    let scores = [0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 3.0];
    let mut labels = Vec::new();
    h.decode(&scores, &forms, &mut labels).unwrap();
    // "casa": del's split strips 'l' and 'd', which casa does not have;
    // della's needs 'della'-like ends; `_` is left.
    assert_eq!(labels, [1, 0, 2]);
    let words = h.words(&labels, &forms).unwrap();
    assert_eq!(words, [vec!["de", "el"], vec!["casa"], vec!["di", "la"]]);
    assert!(h.words(&[1], &["casa"]).is_err());
    assert!(h.decode(&scores[..8], &forms, &mut labels).is_err());
    assert!(h.decode(&[f32::NAN; 9], &forms, &mut labels).is_err());
}

#[test]
fn a_split_head_needs_keep_and_valid_splits() {
    let del = label("del", &["de", "el"]);
    assert!(spec(HeadKind::TokenSplit, &[&del], Decode::EditTree).validate("h").is_err());
    assert!(spec(HeadKind::TokenSplit, &["_", "(m 0 1 ()"], Decode::EditTree).validate("h").is_err());
    assert!(spec(HeadKind::TokenSplit, &["_", &del], Decode::Argmax).validate("h").is_err());
}
