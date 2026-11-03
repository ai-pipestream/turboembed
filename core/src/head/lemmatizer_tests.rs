//! The edit-tree lemmatizer head on the host: tree scores from word
//! vectors and tag embeddings, the best fitting tree with ties to the
//! lower index, the form when no tree fits, and the spec rules.

use super::edit_tree::EditTree;
use super::lemmatizer::{EditTreeLemmatizer, Work, tensors};
use super::words::NO_WORD;
use crate::manifest::{
    Decode, HeadKind, HeadSpec, MAX_SEEN_FORM_BYTES, PosInput, Scheme, SeenLemma, SeenLookup, WordPooling,
};

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
        seen_lookup: None,
        hash_embed: None,
        layer_mix: None,
        span: None,
        relation: None,
        coref: None,
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

#[test]
fn seen_lookup_overrides_the_classifier_for_unambiguous_forms_and_tags() {
    let mut s = spec(Some(PosInput { head: "pos".into(), labels: 3, dim: 2 }));
    s.seen_lookup = Some(SeenLookup {
        min_freq: 2,
        entries: vec![
            // "jumped" is in the lookup table, overriding tree 2 (identity).
            SeenLemma { form: "jumped".into(), tag: None, lemma: "jump".into() },
            // "saw" is ambiguous overall, unambiguous under tag 0 (NOUN) and tag 1 (VERB),
            // and its verb lemma "see" has no edit tree in `s.labels`.
            SeenLemma { form: "saw".into(), tag: Some(0), lemma: "saw".into() },
            SeenLemma { form: "saw".into(), tag: Some(1), lemma: "see".into() },
            // "went" -> "go" has no edit tree in `s.labels` at all.
            SeenLemma { form: "went".into(), tag: None, lemma: "go".into() },
        ],
    });
    let h = head(&s);
    // Trees: 0 walked, 1 running, 2 identity, 3 was->be.
    // Force every word's highest score onto tree 2 (identity).
    let forms = ["jumped", "went", "saw", "saw", "saw", "walked"];
    let scores: Vec<f32> = (0..forms.len()).flat_map(|_| [0.0, 0.0, 9.0, 0.0]).collect();
    let mut out = Vec::new();
    h.decode(&scores, &forms, &mut out).unwrap();
    assert_eq!(out, [Some(2); 6]);
    // Without tags, tag-less entries apply and tag-specific ones fall back to the decoded tree.
    assert_eq!(h.lemmas(&out, &forms).unwrap(), ["jump", "go", "saw", "saw", "saw", "walked"]);
    // With tags, tag 1 maps "saw" to "see", tag 0 to "saw", and tag 2 falls back to the tree.
    let tags = [1, 1, 0, 1, 2, 1];
    assert_eq!(h.lemmas_tagged(&out, &forms, Some(&tags)).unwrap(), ["jump", "go", "saw", "see", "saw", "walked"]);
    assert!(h.lemmas_tagged(&out, &forms, Some(&[0])).is_err(), "a tag a word");
    assert!(h.lemmas_tagged(&out, &forms, Some(&[0, 0, 0, 0, 0, 3])).is_err(), "tag out of range");
}

#[test]
fn the_spec_refuses_malformed_seen_lookup_tables() {
    let entry = |form: &str, tag: Option<u32>, lemma: &str| SeenLemma { form: form.into(), tag, lemma: lemma.into() };
    let mut s = spec(None);
    s.seen_lookup = Some(SeenLookup { min_freq: 0, entries: Vec::new() });
    assert!(s.validate("head").is_err(), "zero min_freq");

    let mut on_tagger = spec(None);
    on_tagger.kind = HeadKind::TokenTagger;
    on_tagger.decode = Decode::Argmax;
    on_tagger.labels = vec!["NOUN".into()];
    on_tagger.seen_lookup = Some(SeenLookup { min_freq: 1, entries: vec![entry("dogs", None, "dog")] });
    assert!(on_tagger.validate("head").is_err(), "seen_lookup on a tagger");

    for bad in [
        vec![entry("", None, "a")],
        vec![entry("a", None, "")],
        vec![entry(&"a".repeat(MAX_SEEN_FORM_BYTES + 1), None, "a")],
        vec![entry("a", Some(0), "a")],
        vec![entry("b", None, "b"), entry("a", None, "a")],
        vec![entry("a", None, "a"), entry("a", None, "b")],
    ] {
        let mut s = spec(None);
        s.seen_lookup = Some(SeenLookup { min_freq: 1, entries: bad });
        assert!(s.validate("head").is_err());
    }

    let mut tagged = spec(Some(PosInput { head: "pos".into(), labels: 2, dim: 2 }));
    tagged.seen_lookup =
        Some(SeenLookup { min_freq: 1, entries: vec![entry("saw", Some(0), "saw"), entry("saw", Some(1), "see")] });
    assert!(tagged.validate("head").is_ok());
    tagged.seen_lookup = Some(SeenLookup { min_freq: 1, entries: vec![entry("saw", Some(2), "see")] });
    assert!(tagged.validate("head").is_err(), "tag past pos_input.labels");
}
