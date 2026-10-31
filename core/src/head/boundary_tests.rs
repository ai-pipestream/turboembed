//! The sentence boundary head: a token tagger's scores, a decode that
//! always ends the last word's sentence, and the spec rules for it.

use super::boundary::{LAST, SentenceBoundary, sentences, tensors};
use super::tagger::NO_WORD;
use super::words::Work;
use crate::manifest::{Decode, HeadKind, HeadSpec, Scheme, WordPooling};

pub(super) fn spec(kind: HeadKind, labels: &[&str], decode: Decode) -> HeadSpec {
    HeadSpec {
        name: "s".into(),
        layer: "turboembed:sentences".into(),
        kind,
        labels: labels.iter().map(|s| s.to_string()).collect(),
        scheme: Scheme::Tags,
        word_pooling: WordPooling::First,
        adapter: None,
        decode,
        biaffine: None,
        pos_input: None,
        hash_embed: None,
        span: None,
        relation: None,
        coref: None,
        license_class: None,
        artifact: "weights-f32".into(),
        tensor_prefix: "heads.s".into(),
    }
}

/// A head over a 2-wide encoder that scores `last` by the first state
/// value and `inside` by the second, with no bias.
fn head() -> SentenceBoundary {
    let s = spec(HeadKind::SentenceBoundary, &["inside", "last"], Decode::Argmax);
    let get = |suffix: &str| match suffix {
        "out.weight" => Some(vec![0.0, 1.0, 1.0, 0.0]),
        "out.bias" => Some(vec![0.0, 0.0]),
        _ => None,
    };
    SentenceBoundary::from_tensors(&s, 2, &get).unwrap()
}

#[test]
fn a_word_ends_its_sentence_when_last_outscores_inside() {
    let h = head();
    // [CLS], four words of one row each, [SEP]: the second word scores
    // last, the others inside.
    let states = [0.0, 0.0, 0.1, 0.9, 0.8, 0.2, 0.0, 1.0, 0.3, 0.7, 0.0, 0.0];
    let words = [NO_WORD, 0, 1, 2, 3, NO_WORD];
    let (mut scores, mut work, mut ends) = (Vec::new(), Work::default(), Vec::new());
    assert_eq!(h.scores(&states, &words, None, &mut scores, &mut work).unwrap(), 4);
    assert_eq!(scores, [0.9, 0.1, 0.2, 0.8, 1.0, 0.0, 0.7, 0.3]);
    h.decode(&scores, 4, &mut ends).unwrap();
    // The last word ends a sentence though it scores inside.
    assert_eq!(ends, [false, true, false, true]);
    assert_eq!(sentences(&ends).unwrap(), [0..2, 2..4]);
    assert_eq!(LAST, 1);
}

#[test]
fn no_words_are_no_sentences() {
    let h = head();
    let (mut ends, mut scores, mut work) = (vec![true], Vec::new(), Work::default());
    assert_eq!(h.scores(&[0.0; 4], &[NO_WORD, NO_WORD], None, &mut scores, &mut work).unwrap(), 0);
    h.decode(&[], 0, &mut ends).unwrap();
    assert!(ends.is_empty());
    assert!(sentences(&ends).unwrap().is_empty());
    assert!(sentences(&[true, false]).is_err());
}

#[test]
fn the_labels_are_inside_and_last_in_that_order() {
    assert!(spec(HeadKind::SentenceBoundary, &["inside", "last"], Decode::Crf).validate("h").is_ok());
    for labels in [&["last", "inside"][..], &["inside"], &["inside", "last", "other"], &["O", "last"]] {
        assert!(spec(HeadKind::SentenceBoundary, labels, Decode::Argmax).validate("h").is_err(), "{labels:?}");
    }
    for decode in [Decode::EditTree, Decode::Mst, Decode::SpansFlat] {
        assert!(spec(HeadKind::SentenceBoundary, &["inside", "last"], decode).validate("h").is_err());
    }
}

#[test]
fn its_tensors_are_a_token_tagger_s_and_its_kind_is_checked() {
    let s = spec(HeadKind::SentenceBoundary, &["inside", "last"], Decode::Crf);
    let names: Vec<String> = tensors(&s, 4).into_iter().map(|(n, _)| n).collect();
    assert_eq!(names, ["out.weight", "out.bias", "crf.transitions", "crf.start", "crf.end"]);
    let tagger = spec(HeadKind::TokenTagger, &["inside", "last"], Decode::Argmax);
    assert!(SentenceBoundary::from_tensors(&tagger, 2, &|_| Some(vec![0.0; 4])).is_err());
    assert!(super::TokenTagger::from_tensors(&s, 2, &|_| Some(vec![0.0; 4])).is_err());
}
