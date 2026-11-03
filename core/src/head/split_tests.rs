//! The token split head: split labels, the decode that takes the best
//! label that fits, and the words a split gives.

use super::boundary_tests::spec;
use super::edit_tree::EditTree;
use super::split::{TokenSplitter, parse};
use crate::manifest::{Decode, HeadKind, HeadSpec, SeenLookup, SeenToken};

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

/// A split head spec whose seen lookup holds the given (token, label).
fn seen_spec(tokens: &[(&str, &str)]) -> HeadSpec {
    let del = label("del", &["de", "el"]);
    let della = label("della", &["di", "la"]);
    let mut s = spec(HeadKind::TokenSplit, &["_", &del, &della], Decode::EditTree);
    s.seen = Some(SeenLookup {
        min_count: 2,
        tokens: tokens.iter().map(|(t, l)| SeenToken { token: (*t).to_owned(), label: (*l).to_owned() }).collect(),
    });
    s
}

/// A 3-wide split head of zero weights (every label scores 0).
fn zero_head(s: &HeadSpec) -> TokenSplitter {
    let k = s.labels.len();
    let get = |suffix: &str| Some(vec![0.0; if suffix == "out.bias" { k } else { 3 * k }]);
    TokenSplitter::from_tensors(s, 3, &get).unwrap()
}

#[test]
fn a_seen_token_takes_its_label_and_the_head_decides_the_rest() {
    let del = label("del", &["de", "el"]);
    let della = label("della", &["di", "la"]);
    // `del` is in the lookup (unambiguous in training); `della` is not (it
    // split two ways, say), and `casa` training never saw.
    let s = seen_spec(&[("del", &del)]);
    s.validate("h").unwrap();
    let h = zero_head(&s);
    let forms = ["del", "della", "casa"];
    // Scores that keep every token whole, but `della`'s row scores its own
    // split highest: the lookup splits `del`, the scorer splits `della`,
    // and the unseen `casa` stays whole.
    let scores = [5.0, 1.0, 1.0, 1.0, 0.0, 3.0, 5.0, 1.0, 1.0];
    let mut labels = Vec::new();
    h.decode(&scores, &forms, &mut labels).unwrap();
    assert_eq!(labels, [1, 2, 0]);
    let words = h.words(&labels, &forms).unwrap();
    assert_eq!(words, [vec!["de", "el"], vec!["di", "la"], vec!["casa"]]);
    // Without the lookup the same scores keep `del` whole.
    let plain = spec(HeadKind::TokenSplit, &["_", &del, &della], Decode::EditTree);
    let plain = zero_head(&plain);
    let mut without = Vec::new();
    plain.decode(&scores, &forms, &mut without).unwrap();
    assert_eq!(without, [0, 2, 0]);
}

#[test]
fn a_looked_up_tokens_scores_are_not_read() {
    let del = label("del", &["de", "el"]);
    let s = seen_spec(&[("del", &del)]);
    let h = zero_head(&s);
    let forms = ["del", "casa"];
    let scores = [f32::NAN, f32::NAN, f32::NAN, 5.0, 1.0, 1.0];
    let mut labels = Vec::new();
    h.decode(&scores, &forms, &mut labels).unwrap();
    assert_eq!(labels, [1, 0]);
    // A non-finite row the lookup does not own is still refused.
    let scores = [0.0, 1.0, 1.0, f32::NAN, 1.0, 1.0];
    assert!(h.decode(&scores, &forms, &mut labels).is_err());
}

#[test]
fn a_seen_lookup_is_checked_against_the_head() {
    let del = label("del", &["de", "el"]);
    let della = label("della", &["di", "la"]);
    // A token whose label does not fit it is refused at load.
    let bad = seen_spec(&[("casa", &della)]);
    let get = |suffix: &str| Some(vec![0.0; if suffix == "out.bias" { 3 } else { 9 }]);
    let e = TokenSplitter::from_tensors(&bad, 3, &get).unwrap_err();
    assert!(e.message.contains("does not fit"), "{e:?}");
    // A label the head does not carry, and a token listed twice.
    let unknown = seen_spec(&[("del", "no-such-label")]);
    let e = TokenSplitter::from_tensors(&unknown, 3, &get).unwrap_err();
    assert!(e.message.contains("not one of"), "{e:?}");
    let twice = seen_spec(&[("del", &del), ("del", &del)]);
    let e = TokenSplitter::from_tensors(&twice, 3, &get).unwrap_err();
    assert!(e.message.contains("twice"), "{e:?}");
}

#[test]
fn a_seen_lookup_is_a_split_heads_alone_and_well_formed() {
    let del = label("del", &["de", "el"]);
    let mut s = spec(HeadKind::TokenSplit, &["_", &del], Decode::EditTree);
    s.seen = Some(SeenLookup { min_count: 2, tokens: vec![SeenToken { token: "del".into(), label: del.clone() }] });
    s.validate("h").unwrap();
    // A min_count of 0, a label the head lacks, a token twice, and the
    // lookup on another kind are all refused at validate.
    let mut zero = s.clone();
    zero.seen.as_mut().unwrap().min_count = 0;
    assert!(zero.validate("h").unwrap_err().message.contains("min_count"));
    let mut nope = s.clone();
    nope.seen.as_mut().unwrap().tokens[0].label = "nope".into();
    assert!(nope.validate("h").is_err());
    let mut twice = s.clone();
    let again = twice.seen.as_ref().unwrap().tokens[0].clone();
    twice.seen.as_mut().unwrap().tokens.push(again);
    assert!(twice.validate("h").unwrap_err().message.contains("twice"));
    let mut tagger = spec(HeadKind::TokenTagger, &["NOUN"], Decode::Argmax);
    tagger.seen = s.seen.clone();
    assert!(tagger.validate("h").unwrap_err().message.contains("seen"));
}
