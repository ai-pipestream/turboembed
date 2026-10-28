//! The dependency annotator end to end on the small BERT in testdata,
//! with a biaffine parser head of seeded weights: one tree per sentence
//! on the view's tokens, the core head's parse, determinism, and
//! refusals.

mod common;

use common::*;
use serde_json::json;
use turbo::head::parser::Work;
use turbo::manifest::WordPooling;
use turbo_analysis::{Annotator, Document, Ix, Key, View};
use turbo_analysis_embed::EncoderOptions;
use turbo_tagger::{DependencyParser, PosTagger, WordEncoder};

#[test]
fn the_parser_writes_one_tree_on_the_tokens() {
    let dir = bundle("parse", |_| {});
    let doc = Document::new(TEXT);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let toks = tokens(TEXT);
    let deps = DependencyParser::open(&dir, None, &EncoderOptions::default()).unwrap();
    assert_eq!(deps.relations().len(), 6);
    let (layer,) = deps.annotate((view, &toks, None, None)).unwrap();
    layer.check_forest().unwrap();
    assert_eq!(layer.head.len(), toks.spans.len());
    assert_eq!(layer.head.iter().filter(|h| h.is_none()).count(), 1, "MST gives one root a sentence");
    assert!(layer.relation.iter().any(|r| *r != layer.relation[0]), "the seeded head gives more than one relation");

    // The same document again gives the same layer.
    let (again,) = deps.annotate((view, &toks, None, None)).unwrap();
    assert_eq!(again.head, layer.head);
    assert_eq!(again.relation, layer.relation);

    // The heads are the core parser's decode of the word features.
    let words: Vec<&str> = TEXT.split(' ').collect();
    let mut enc = WordEncoder::open(&dir, &EncoderOptions::default()).unwrap();
    let f = enc.features(std::slice::from_ref(&words), WordPooling::First).unwrap().remove(0);
    let ids: Vec<u32> = (0..words.len() as u32).collect();
    let (mut arcs, mut heads, mut work) = (Vec::new(), Vec::new(), Work::default());
    let n = deps.parser().arcs(&f, &ids, &mut arcs, &mut work).unwrap();
    deps.parser().decode(&arcs, n, &mut heads).unwrap();
    let got: Vec<u32> = layer.head.iter().map(|h| h.map_or(0, |h| h.get() as u32 + 1)).collect();
    assert_eq!(got, heads);
    assert_eq!(deps.parse(&[words]).unwrap()[0].heads, heads);
}

#[test]
fn each_sentence_is_parsed_on_its_own_with_its_own_root() {
    let text = "Ann met Bob . Bob slept .";
    let doc = Document::new(text);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let mut toks = tokens(text);
    toks.sentence = Some([0, 0, 0, 0, 1, 1, 1].iter().map(|&i| Ix::new(i)).collect());
    let dir = bundle("parse-sentences", |_| {});
    let deps = DependencyParser::open(&dir, Some("deps"), &EncoderOptions::default()).unwrap();
    let (both,) = deps.annotate((view, &toks, None, None)).unwrap();
    both.check_forest().unwrap();
    let roots: Vec<usize> = (0..7).filter(|&i| both.head[i].is_none()).collect();
    assert_eq!(roots.len(), 2);
    assert!(roots[0] < 4 && roots[1] >= 4, "{roots:?}");
    for (i, h) in both.head.iter().enumerate() {
        if let Some(h) = h {
            assert_eq!(i < 4, h.get() < 4, "token {i} hangs off {} across sentences", h.get());
        }
    }
    let alone_text = "Bob slept .";
    let alone_doc = Document::new(alone_text);
    let alone_view = alone_doc.require(&Key::<View>::DEFAULT).unwrap();
    let (alone,) = deps.annotate((alone_view, &tokens(alone_text), None, None)).unwrap();
    let shifted: Vec<Option<usize>> = both.head[4..].iter().map(|h| h.map(|h| h.get() - 4)).collect();
    assert_eq!(shifted, alone.head.iter().map(|h| h.map(|h| h.get())).collect::<Vec<_>>());
    assert_eq!(both.relation[4..], alone.relation[..]);
}

#[test]
fn eisner_gives_a_projective_tree() {
    let dir = bundle("parse-eisner", |m| m["heads"][2]["decode"] = json!("DECODE_EISNER"));
    let doc = Document::new(TEXT);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let toks = tokens(TEXT);
    let deps = DependencyParser::open(&dir, None, &EncoderOptions::default()).unwrap();
    let (layer,) = deps.annotate((view, &toks, None, None)).unwrap();
    layer.check_forest().unwrap();
    let head: Vec<usize> = layer.head.iter().map(|h| h.map_or(0, |h| h.get() + 1)).collect();
    // No two arcs cross, the root's arc running from position 0.
    let arcs: Vec<(usize, usize)> = head.iter().enumerate().map(|(d, &h)| (h.min(d + 1), h.max(d + 1))).collect();
    for &(a, b) in &arcs {
        for &(c, e) in &arcs {
            assert!(!(a < c && c < b && b < e), "arcs {a}-{b} and {c}-{e} cross");
        }
    }
}

#[test]
fn heads_that_are_not_parsers_are_refused_both_ways() {
    let dir = bundle("parse-refusals", |_| {});
    let o = EncoderOptions::default();
    assert!(DependencyParser::open(&dir, Some("pos"), &o).is_err());
    assert!(DependencyParser::open(&dir, Some("missing"), &o).is_err());
    assert!(PosTagger::open(&dir, Some("deps"), &o).is_err());
    let none = bundle("parse-none", |m| {
        m["heads"].as_array_mut().unwrap().retain(|h| h["name"] != "deps");
    });
    assert!(DependencyParser::open(&none, None, &o).is_err());
}
