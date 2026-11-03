//! The lexicon lookups over the lemmatizer head: a Morfologik dictionary
//! and WordNet's Morphy answer under the head's tag, through a tag map
//! when they have one, and stand over the head's lemma where they list
//! one, through the annotator end to end.

mod common;

use std::path::Path;
use std::sync::Arc;

use common::*;
use turbo_analysis::{Annotator, Document, Key, View};
use turbo_analysis_embed::EncoderOptions;
use turbo_morfologik::{MorfologikDictionary, MorfologikDictionaryBuilder, MorfologikLemmatizer};
use turbo_tagger::{LemmaDictionary, Lemmatizer, MorfologikLemmas, MorphyLemmas};
use turbo_wordnet::wndb::WndbReader;
use turbo_wordnet::{MorphyExceptions, MorphyLemmatizer};

const INFO: &[u8] = b"fsa.dict.separator=,\nfsa.dict.encoding=UTF-8\nfsa.dict.encoder=prefix\n";

/// `lemma,form,tag` rows compiled to a dictionary.
fn morfologik(rows: &str) -> MorfologikLemmatizer {
    let bytes = MorfologikDictionaryBuilder::compile_bytes(rows.as_bytes(), INFO).unwrap();
    MorfologikLemmatizer::new(Arc::new(MorfologikDictionary::from_bytes(&bytes, INFO).unwrap()))
}

fn morphy() -> MorphyLemmas<turbo_wordnet::InMemoryWordNetLexicon> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../wordnet/tests/fixtures/mini-wndb");
    let lexicon = WndbReader::read(&dir).unwrap();
    MorphyLemmas::new(MorphyLemmatizer::new(lexicon, MorphyExceptions::load(&dir).unwrap()))
}

#[test]
fn a_morfologik_dictionary_answers_the_smallest_lemma_under_the_tag() {
    let d = MorfologikLemmas::new(morfologik("ser,foi,V\nir,foi,V\ncasa,casa,NOUN\ncasar,casa,V\nCasa,Casa,PROP\n"));
    assert_eq!(d.lemma("foi", Some("V")).as_deref(), Some("ir"), "ir before ser in byte order");
    assert_eq!(d.lemma("casa", Some("NOUN")).as_deref(), Some("casa"));
    assert_eq!(d.lemma("casa", Some("V")).as_deref(), Some("casar"));
    assert_eq!(d.lemma("Casa", Some("V")).as_deref(), Some("casar"), "the word is looked up lowercased");
    assert_eq!(d.lemma("casa", Some("PROP")), None, "the dictionary keys PROP by the capital");
    assert_eq!(d.lemma("casa", Some("ADJ")), None);
    assert_eq!(d.lemma("foi", None), None, "no tag, no lookup");
    assert_eq!(d.lemma("carro", Some("NOUN")), None);
    assert_eq!(d.lemmatizer().lemmas_for("foi", "V").len(), 2);

    // A tag map takes the head's tags to the dictionary's.
    let m = MorfologikLemmas::new(morfologik("ser,foi,V\ncasa,casa,NOUN\n"))
        .with_tag_map(|t| (t == "VERB").then(|| "V".to_owned()));
    assert_eq!(m.lemma("foi", Some("VERB")).as_deref(), Some("ser"));
    assert_eq!(m.lemma("casa", Some("NOUN")), None, "the map answers NOUN with None");
    assert_eq!(m.lemma("foi", Some("V")), None);
}

#[test]
fn morphy_answers_its_first_candidate_under_the_tags_part_of_speech() {
    let d = morphy();
    assert_eq!(d.lemma("mice", Some("NOUN")).as_deref(), Some("mouse"), "an exception");
    assert_eq!(d.lemma("Mice", Some("NNS")).as_deref(), Some("mouse"), "folded, under a Penn tag");
    assert_eq!(d.lemma("went", Some("VERB")).as_deref(), Some("go"));
    assert_eq!(d.lemma("went", Some("NOUN")), None, "no noun went");
    assert_eq!(d.lemma("mice", Some("DET")), None, "no part of speech for DET");
    assert_eq!(d.lemma("mice", None), None);
    assert_eq!(d.lemma("qwzx", Some("NOUN")), None, "a word the lexicon does not know");
    let mapped = morphy().with_tag_map(|t| (t == "V").then(|| "VERB".to_owned()));
    assert_eq!(mapped.lemma("ran", Some("V")).as_deref(), Some("run"));
    assert_eq!(mapped.lemma("ran", Some("VERB")), None);
    assert_eq!(d.morphy().exceptions().lookup("oxen", turbo_wordnet::WordNetPos::Noun), ["ox"]);
}

#[test]
fn the_lexicons_stand_over_the_head_through_the_annotator() {
    let dir = bundle("lemma-lexicons", |_| {});
    let o = EncoderOptions::default();
    let doc = Document::new(TEXT);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let toks = tokens(TEXT);
    let plain = Lemmatizer::open(&dir, None, &o).unwrap();
    let (base,) = plain.annotate((view, &toks, None, None)).unwrap();
    let base: Vec<&str> = base.lemma.iter().collect();

    // Every word of the text (keyed lowercase, as the lookup lowercases
    // the word) under every tag of the pos head, through a map to the
    // dictionary's one tag: the dictionary decides each lemma.
    // The punctuation is not in the dictionary (the comma is its
    // separator), so the head's lemma stands there.
    let alpha = |w: &str| w.chars().all(|c| c.is_ascii_alphabetic());
    let rows: String =
        TEXT.split(' ').filter(|w| alpha(w)).map(|w| format!("{l}-x,{l},X\n", l = w.to_ascii_lowercase())).collect();
    let all = MorfologikLemmas::new(morfologik(&rows)).with_tag_map(|_| Some("X".to_owned()));
    let lem = Lemmatizer::open(&dir, None, &o).unwrap().with_dictionary("all", all);
    let (got,) = lem.annotate((view, &toks, None, None)).unwrap();
    let want: Vec<String> = TEXT
        .split(' ')
        .zip(&base)
        .map(|(w, b)| if alpha(w) { format!("{}-x", w.to_ascii_lowercase()) } else { (*b).to_owned() })
        .collect();
    assert!(got.lemma.iter().eq(want.iter().map(String::as_str)), "{:?}", got.lemma.iter().collect::<Vec<_>>());

    // Morphy knows only its exceptions here, so with the words of the
    // text absent from the lexicon it changes nothing, and the lookups
    // name a different configuration all the same.
    let lem = Lemmatizer::open(&dir, None, &o).unwrap().with_dictionary("mini-wndb", morphy());
    let (same,) = lem.annotate((view, &toks, None, None)).unwrap();
    assert!(same.lemma.iter().eq(base.iter().copied()));
    assert_ne!(lem.config_hash(), plain.config_hash());
}
