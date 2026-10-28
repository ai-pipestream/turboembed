//! What the tagger annotators take from a document: each token in exactly
//! one sequence, sentences on the view's character boundaries, the
//! document's subwords in place of tokenizing again when they are this
//! model's, and rows with room for one piece.

mod common;

use common::*;
use turbo_analysis::{Annotator, Document, Key, Span, Spans, View, ViewId};
use turbo_analysis_embed::{EncoderOptions, SubwordAnnotator};
use turbo_analysis_layers::{Sentences, Tokens};
use turbo_tagger::{EntityTagger, HeadRunner, PosTagger, WordEncoder};

fn sentences(spans: &[(u32, u32)]) -> Sentences {
    let mut s = Spans::new(ViewId::SOURCE);
    for &(a, b) in spans {
        s.push(Span::new(a, b));
    }
    Sentences { spans: s }
}

#[test]
fn a_token_across_sentences_or_outside_them_is_refused_by_both_annotators() {
    let dir = bundle("ownership", |_| {});
    let o = EncoderOptions::default();
    let (pos, ner) = (PosTagger::open(&dir, None, &o).unwrap(), EntityTagger::open(&dir, None, &o).unwrap());
    let doc = Document::new("Paris");
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let toks = tokens("Paris");
    for (name, s) in
        [("split", sentences(&[(0, 2), (2, 5)])), ("none", sentences(&[])), ("short", sentences(&[(0, 3)]))]
    {
        assert!(HeadRunner::sequences(view, &toks, Some(&s)).is_err(), "{name}");
        assert!(pos.annotate((view, &toks, None, Some(&s))).is_err(), "{name}: pos");
        assert!(ner.annotate((view, &toks, None, Some(&s))).is_err(), "{name}: entities");
    }
    let whole = sentences(&[(0, 5)]);
    assert_eq!(HeadRunner::sequences(view, &toks, Some(&whole)).unwrap(), [vec![0]]);
}

#[test]
fn a_sentence_off_a_character_boundary_is_refused() {
    let text = "\u{1F600} ok";
    let doc = Document::new(text);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let toks = tokens(text);
    assert!(HeadRunner::sequences(view, &toks, Some(&sentences(&[(0, 2), (2, 7)]))).is_err());
    assert_eq!(HeadRunner::sequences(view, &toks, Some(&sentences(&[(0, 4), (5, 7)]))).unwrap(), [[0], [1]]);
}

fn subwords(dir: &std::path::Path, view: &View, toks: &Tokens) -> turbo_analysis_layers::Subwords {
    SubwordAnnotator::open(dir).unwrap().annotate((view, Some(toks))).unwrap().0
}

#[test]
fn this_models_subwords_are_read_without_tokenizing_again() {
    let dir = bundle("subwords", |_| {});
    let doc = Document::new(TEXT);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let toks = tokens(TEXT);
    let sub = subwords(&dir, view, &toks);
    let seqs = HeadRunner::sequences(view, &toks, None).unwrap();
    let mut enc = WordEncoder::open(&dir, &EncoderOptions::default()).unwrap();
    assert!(enc.reads(&sub, &toks).unwrap());
    for pooling in [turbo::manifest::WordPooling::First, turbo::manifest::WordPooling::Mean] {
        let before = enc.tokenized();
        let reused = enc.token_features(view, &toks, &seqs, Some(&sub), pooling).unwrap();
        assert_eq!(enc.tokenized(), before, "no word tokenized again");
        let fresh = enc.token_features(view, &toks, &seqs, None, pooling).unwrap();
        assert_eq!(enc.tokenized(), before + toks.spans.len() as u64);
        assert_eq!(reused, fresh, "{pooling:?}");
    }
    let o = EncoderOptions::default();
    let pos = PosTagger::open(&dir, None, &o).unwrap();
    let with = pos.annotate((view, &toks, Some(&sub), None)).unwrap().0;
    let without = pos.annotate((view, &toks, None, None)).unwrap().0;
    assert_eq!(with.tags, without.tags);

    // Subwords of another model are refused; ones made without the
    // tokens are passed over and the words tokenized.
    let mut other = subwords(&dir, view, &toks);
    other.model.sha256 = "0".repeat(64);
    assert!(enc.reads(&other, &toks).is_err());
    assert!(pos.annotate((view, &toks, Some(&other), None)).is_err());
    let mut loose = subwords(&dir, view, &toks);
    loose.token = None;
    assert!(!enc.reads(&loose, &toks).unwrap());
    let before = enc.tokenized();
    enc.token_features(view, &toks, &seqs, Some(&loose), turbo::manifest::WordPooling::First).unwrap();
    assert_eq!(enc.tokenized(), before + toks.spans.len() as u64);
}

#[test]
fn rows_with_room_for_one_piece_are_read_in_one_piece_windows() {
    let dir = bundle("one-piece", |_| {});
    // Two special tokens and one piece.
    let mut enc = WordEncoder::open(&dir, &EncoderOptions { max_seq: 3, ..EncoderOptions::default() }).unwrap();
    assert_eq!(enc.width(), 1);
    let words: Vec<&str> = TEXT.split(' ').collect();
    let f = enc.features(std::slice::from_ref(&words), turbo::manifest::WordPooling::Mean).unwrap().remove(0);
    assert_eq!(f.len(), words.len() * enc.hidden());
    assert!(f.iter().all(|v| v.is_finite()));
    assert!(WordEncoder::open(&dir, &EncoderOptions { max_seq: 2, ..EncoderOptions::default() }).is_err());
}
