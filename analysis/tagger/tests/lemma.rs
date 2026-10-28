//! The lemma annotator end to end on the small BERT in testdata, with an
//! edit-tree lemmatizer head of seeded weights reading the part-of-speech
//! head's tags: a lemma per token, the core heads' decode, each sentence
//! on its own, a dictionary over the head, and refusals.

mod common;

use std::sync::{Arc, Mutex};

use common::*;
use turbo::head::{lemmatizer, tagger};
use turbo::manifest::WordPooling;
use turbo_analysis::{Annotator, Document, Ix, Key, View};
use turbo_analysis_embed::EncoderOptions;
use turbo_tagger::{DependencyParser, Lemmatizer, PosTagger, WordEncoder};

fn lemmas_of(l: &turbo_analysis_layers::Lemmas) -> Vec<String> {
    l.lemma.iter().map(str::to_owned).collect()
}

#[test]
fn the_lemmatizer_writes_the_core_heads_lemma_for_every_token() {
    let dir = bundle("lemma", |_| {});
    let doc = Document::new(TEXT);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let toks = tokens(TEXT);
    let o = EncoderOptions::default();
    let lem = Lemmatizer::open(&dir, None, &o).unwrap();
    let pos = lem.pos().expect("the head reads the pos head's tags");
    assert_eq!(pos.spec().name, "pos");
    let (layer,) = lem.annotate((view, &toks, None, None)).unwrap();
    let got = lemmas_of(&layer);
    assert_eq!(got.len(), toks.spans.len());

    // The tags come from the pos head on its own pooling (mean), the
    // trees from the lemma head on its own (first).
    let words: Vec<&str> = TEXT.split(' ').collect();
    let mut enc = WordEncoder::open(&dir, &o).unwrap();
    let mean = enc.features(std::slice::from_ref(&words), WordPooling::Mean).unwrap().remove(0);
    let first = enc.features(std::slice::from_ref(&words), WordPooling::First).unwrap().remove(0);
    let ids: Vec<u32> = (0..words.len() as u32).collect();
    let (mut scores, mut tags, mut trees) = (Vec::new(), Vec::new(), Vec::new());
    pos.scores(&mean, &ids, &mut scores, &mut tagger::Work::default()).unwrap();
    pos.decode(&scores, words.len(), &mut tags).unwrap();
    let head = lem.head();
    head.scores(&first, &ids, Some(&tags), &mut scores, &mut lemmatizer::Work::default()).unwrap();
    head.decode(&scores, &words, &mut trees).unwrap();
    let want = head.lemmas(&trees, &words).unwrap();
    assert_eq!(got, want);
    assert_eq!(lem.lemmatize(&[words]).unwrap(), [want]);
    assert!(trees.iter().any(|t| *t != trees[0]), "the seeded head picks more than one tree: {trees:?}");

    // The same document again gives the same layer.
    let (again,) = lem.annotate((view, &toks, None, None)).unwrap();
    assert_eq!(lemmas_of(&again), got);
}

#[test]
fn the_tags_match_the_pos_annotator() {
    let dir = bundle("lemma-tags", |_| {});
    let doc = Document::new(TEXT);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let toks = tokens(TEXT);
    let o = EncoderOptions::default();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = seen.clone();
    let lem = Lemmatizer::open(&dir, None, &o).unwrap().with_dictionary("log", move |_: &str, tag: Option<&str>| {
        log.lock().unwrap().push(tag.map(str::to_owned));
        None
    });
    let (layer,) = lem.annotate((view, &toks, None, None)).unwrap();
    let pos = PosTagger::open(&dir, None, &o).unwrap();
    let (tags,) = pos.annotate((view, &toks, None, None)).unwrap();
    let names: Vec<Option<String>> = tags.tags.iter().map(|t| Some(pos.tagset().label(*t).to_owned())).collect();
    assert_eq!(*seen.lock().unwrap(), names, "the dictionary is asked under each token's tag");
    let plain = Lemmatizer::open(&dir, None, &o).unwrap();
    assert_eq!(lemmas_of(&plain.annotate((view, &toks, None, None)).unwrap().0), lemmas_of(&layer));
}

#[test]
fn a_dictionary_lemma_stands_over_the_heads() {
    let dir = bundle("lemma-dictionary", |_| {});
    let doc = Document::new(TEXT);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let toks = tokens(TEXT);
    let o = EncoderOptions::default();
    let plain = Lemmatizer::open(&dir, None, &o).unwrap();
    let before = lemmas_of(&plain.annotate((view, &toks, None, None)).unwrap().0);
    let lookup = |form: &str, _: Option<&str>| (form == "dog" || form == "met").then(|| format!("<{form}>"));
    let with = Lemmatizer::open(&dir, None, &o).unwrap().with_dictionary("small", lookup);
    let after = lemmas_of(&with.annotate((view, &toks, None, None)).unwrap().0);
    for (i, w) in TEXT.split(' ').enumerate() {
        match w {
            "dog" | "met" => assert_eq!(after[i], format!("<{w}>")),
            _ => assert_eq!(after[i], before[i], "{w}"),
        }
    }
    // The dictionary and its name are part of the configuration.
    let other = Lemmatizer::open(&dir, None, &o).unwrap().with_dictionary("large", lookup);
    let hashes = [plain.config_hash(), with.config_hash(), other.config_hash()];
    assert!(hashes[0] != hashes[1] && hashes[1] != hashes[2] && hashes[0] != hashes[2], "{hashes:?}");
}

#[test]
fn a_head_without_tags_reads_the_words_alone() {
    let dir = bundle("lemma-untagged", |m| {
        m["heads"][3].as_object_mut().unwrap().remove("pos_input");
    });
    let doc = Document::new(TEXT);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let toks = tokens(TEXT);
    let o = EncoderOptions::default();
    let tagless = Arc::new(Mutex::new(true));
    let flag = tagless.clone();
    let lem =
        Lemmatizer::open(&dir, Some("lemma"), &o).unwrap().with_dictionary("t", move |_: &str, tag: Option<&str>| {
            *flag.lock().unwrap() &= tag.is_none();
            None
        });
    assert!(lem.pos().is_none());
    let (layer,) = lem.annotate((view, &toks, None, None)).unwrap();
    assert!(*tagless.lock().unwrap(), "no tag without a pos_input");
    let words: Vec<&str> = TEXT.split(' ').collect();
    let first = WordEncoder::open(&dir, &o).unwrap().features(std::slice::from_ref(&words), WordPooling::First);
    let ids: Vec<u32> = (0..words.len() as u32).collect();
    let (mut scores, mut trees) = (Vec::new(), Vec::new());
    lem.head().scores(&first.unwrap()[0], &ids, None, &mut scores, &mut lemmatizer::Work::default()).unwrap();
    lem.head().decode(&scores, &words, &mut trees).unwrap();
    assert_eq!(lemmas_of(&layer), lem.head().lemmas(&trees, &words).unwrap());
}

#[test]
fn each_sentence_is_lemmatized_on_its_own() {
    let text = "Ann met Bob . Bob slept .";
    let doc = Document::new(text);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let mut toks = tokens(text);
    toks.sentence = Some([0, 0, 0, 0, 1, 1, 1].iter().map(|&i| Ix::new(i)).collect());
    let dir = bundle("lemma-sentences", |_| {});
    let lem = Lemmatizer::open(&dir, None, &EncoderOptions::default()).unwrap();
    let both = lemmas_of(&lem.annotate((view, &toks, None, None)).unwrap().0);
    let alone = lem.lemmatize(&[vec!["Ann", "met", "Bob", "."], vec!["Bob", "slept", "."]]).unwrap();
    assert_eq!(both, alone.concat());
}

#[test]
fn heads_that_are_not_lemmatizers_are_refused_both_ways() {
    let dir = bundle("lemma-refusals", |_| {});
    let o = EncoderOptions::default();
    assert!(Lemmatizer::open(&dir, Some("pos"), &o).is_err());
    assert!(Lemmatizer::open(&dir, Some("missing"), &o).is_err());
    assert!(DependencyParser::open(&dir, Some("lemma"), &o).is_err());
    let none = bundle("lemma-none", |m| {
        m["heads"].as_array_mut().unwrap().retain(|h| h["name"] != "lemma");
    });
    assert!(Lemmatizer::open(&none, None, &o).is_err());
}
