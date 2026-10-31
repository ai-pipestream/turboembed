//! The edit-tree lemmatizer head on the host: tree scores from word
//! vectors and tag embeddings, the best fitting tree with ties to the
//! lower index, the form when no tree fits, and the spec rules.

use super::edit_tree::EditTree;
use super::lemmatizer::{EditTreeLemmatizer, Work, tensors};
use super::words::NO_WORD;
use crate::manifest::{Decode, HeadKind, HeadSpec, PosInput, Scheme, WordPooling};

const HIDDEN: usize = 6;

fn trees() -> Vec<String> {
    [("walked", "walk"), ("running", "run"), ("cats", "cats"), ("was", "be")]
        .map(|(f, l)| EditTree::mine(f, l).label())
        .to_vec()
}

fn spec(pos: Option<PosInput>) -> HeadSpec {
    HeadSpec {
        name: "lemma".into(),
        layer: "turboembed:lemmas".into(),
        kind: HeadKind::EditTreeLemmatizer,
        labels: trees(),
        scheme: Scheme::Tags,
        word_pooling: WordPooling::First,
        adapter: None,
        decode: Decode::EditTree,
        biaffine: None,
        pos_input: pos,
        hash_embed: None,
        span: None,
        relation: None,
        license_class: None,
        artifact: "weights-f32".into(),
        tensor_prefix: "heads.lemma".into(),
    }
}

fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            ((s.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}

fn head(spec: &HeadSpec) -> EditTreeLemmatizer {
    build(spec, None).unwrap()
}

/// The head of seeded weights, `poison` the first value of its tag
/// embedding when Some.
fn build(spec: &HeadSpec, poison: Option<f32>) -> crate::status::Result<EditTreeLemmatizer> {
    let mut w: Vec<(String, Vec<f32>)> = tensors(spec, HIDDEN)
        .into_iter()
        .enumerate()
        .map(|(i, (s, shape))| (s, values(i as u64 + 3, shape.iter().product::<u64>() as usize)))
        .collect();
    if let (Some(v), Some((_, t))) = (poison, w.iter_mut().find(|(k, _)| k == "pos.embedding")) {
        t[0] = v;
    }
    let g = |s: &str| w.iter().find(|(k, _)| k == s).map(|(_, v)| v.clone());
    EditTreeLemmatizer::from_tensors(spec, HIDDEN, &g)
}

#[test]
fn the_best_fitting_tree_wins_ties_go_low_and_a_form_with_no_tree_is_its_own_lemma() {
    let h = head(&spec(None));
    let forms = ["jumped", "jumped", "gunning", "xyz", "is"];
    // Trees: 0 walked, 1 running, 2 identity, 3 was->be.
    let scores = [
        [5.0, 1.0, 0.0, 0.0],  // the best tree fits
        [1.0, 5.0, 0.0, 0.0],  // the best ("ning") does not fit; walked does
        [0.0, 2.0, 2.0, 0.0],  // a tie between two that fit: the lower index
        [0.0, 0.0, -1.0, 9.0], // only the identity fits
        [0.0, 0.0, 0.0, 9.0],  // "was" does not fit "is"; identity does
    ];
    let flat: Vec<f32> = scores.iter().flatten().copied().collect();
    let mut out = Vec::new();
    h.decode(&flat, &forms, &mut out).unwrap();
    assert_eq!(out, [Some(0), Some(0), Some(1), Some(2), Some(2)]);
    assert_eq!(h.lemmas(&out, &forms).unwrap(), ["jump", "jump", "gun", "xyz", "is"]);

    // Without the identity tree, a form nothing fits stays as it is.
    let mut s = spec(None);
    s.labels.remove(2);
    let h = head(&s);
    h.decode(&[0.0, 0.0, 1.0], &["xyz"], &mut out).unwrap();
    assert_eq!(out, [None]);
    assert_eq!(h.lemmas(&out, &["xyz"]).unwrap(), ["xyz"]);
    assert!(h.lemmas(&[Some(0)], &["xyz"]).is_err(), "a tree that does not fit");
    // The walked tree would leave `ed` with no lemma, so it does not fit
    // and, with nothing else fitting, the word is its own lemma.
    h.decode(&[9.0, 0.0, 0.0], &["ed"], &mut out).unwrap();
    assert_eq!(out, [None]);
    assert_eq!(h.lemmas(&out, &["ed"]).unwrap(), ["ed"]);
    assert!(h.decode(&[0.0, f32::NAN, 0.0], &["a"], &mut out).is_err());
    assert!(h.decode(&[0.0, 0.0], &["a"], &mut out).is_err());
}

#[test]
fn scores_read_the_words_and_with_a_pos_input_their_tags() {
    let states = values(9, 5 * HIDDEN);
    let words = [NO_WORD, 0, 1, 1, NO_WORD];
    let plain = head(&spec(None));
    let (mut out, mut w) = (Vec::new(), Work::default());
    assert_eq!(plain.scores(&states, &words, None, &mut out, &mut w).unwrap(), 2);
    assert_eq!(out.len(), 2 * 4);
    assert!(plain.scores(&states, &words, Some(&[0, 1]), &mut out, &mut w).is_err());

    let tagged = head(&spec(Some(PosInput { head: "pos".into(), labels: 3, dim: 2 })));
    assert!(tagged.scores(&states, &words, None, &mut out, &mut w).is_err(), "tags are required");
    assert!(tagged.scores(&states, &words, Some(&[0]), &mut out, &mut w).is_err(), "a tag a word");
    assert!(tagged.scores(&states, &words, Some(&[0, 3]), &mut out, &mut w).is_err(), "a tag past the table");
    tagged.scores(&states, &words, Some(&[0, 1]), &mut out, &mut w).unwrap();
    let a = out.clone();
    tagged.scores(&states, &words, Some(&[0, 2]), &mut out, &mut w).unwrap();
    assert_eq!(a[..4], out[..4], "the first word's tag is the same");
    assert_ne!(a[4..], out[4..], "the second word's tag moved its scores");
}

#[test]
fn the_spec_refuses_what_a_lemmatizer_cannot_be() {
    assert!(spec(None).validate("head").is_ok());
    let mut bad = spec(None);
    bad.labels[0] = "NOUN".into();
    assert!(bad.validate("head").unwrap_err().message.contains("edit tree"));
    let mut argmax = spec(None);
    argmax.decode = Decode::Argmax;
    assert!(argmax.validate("head").is_err());
    let mut own = spec(Some(PosInput { head: "lemma".into(), labels: 3, dim: 2 }));
    assert!(own.validate("head").is_err(), "its own tags");
    own.pos_input = Some(PosInput { head: "pos".into(), labels: 3, dim: 0 });
    assert!(own.validate("head").is_err(), "zero width");
    let mut tagger = spec(Some(PosInput { head: "pos".into(), labels: 3, dim: 2 }));
    tagger.kind = HeadKind::TokenTagger;
    tagger.decode = Decode::Argmax;
    assert!(tagger.validate("head").is_err(), "a tagger with a pos_input");
}

#[test]
fn a_tie_is_numeric_so_negative_and_positive_zero_go_to_the_lower_index() {
    // The best tree does not fit `cats`, so the fallback order decides
    // between tree 1 at -0 and tree 2 at 0: equal, so the lower index.
    let mut s = spec(None);
    s.labels = vec![
        EditTree::mine("x", "y").label(),
        EditTree::mine("cats", "cats").label(),
        EditTree::mine("cats", "cat").label(),
    ];
    let h = head(&s);
    let mut out = Vec::new();
    h.decode(&[1.0, -0.0, 0.0], &["cats"], &mut out).unwrap();
    assert_eq!(out, [Some(1)]);
    assert_eq!(h.lemmas(&out, &["cats"]).unwrap(), ["cats"]);
    h.decode(&[1.0, 0.0, -0.0], &["cats"], &mut out).unwrap();
    assert_eq!(out, [Some(1)]);
}

#[test]
fn a_tag_embedding_that_is_not_finite_is_refused() {
    let s = spec(Some(PosInput { head: "pos".into(), labels: 3, dim: 2 }));
    assert!(build(&s, Some(0.25)).is_ok());
    for v in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let e = build(&s, Some(v)).unwrap_err();
        assert!(e.message.contains("pos.embedding holds a value that is not finite"), "{v}: {e:?}");
    }
}
