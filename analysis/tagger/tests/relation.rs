//! The relation annotator end to end on the small BERT in testdata, with
//! a relation classifier head of seeded weights over hand-made entities:
//! the core head's decode on the view's tokens, exact offsets, the
//! entities that are no mention, determinism, and refusals.

mod common;

use common::*;
use serde_json::{Value, json};
use turbo::head::relation::{Mention, RelationWork};
use turbo::manifest::WordPooling;
use turbo_analysis::{Annotator, Document, Ix, IxRange, Key, Layer, Span, Spans, Tag, TagSet, View, ViewId};
use turbo_analysis_embed::EncoderOptions;
use turbo_analysis_layers::{Entities, Sentences, Tokens};
use turbo_tagger::{RelationTagger, SpanTagger, WordEncoder};

fn relation_head(types: &[&str]) -> Value {
    let dim = if types.is_empty() { 0 } else { 3 };
    json!({
        "name": "rels", "layer": "turboembed:relations", "kind": "HEAD_RELATION_CLASSIFIER",
        "labels": ["none", "met", "located_in"], "scheme": "SCHEME_TAGS", "word_pooling": "WORD_POOLING_FIRST",
        "decode": "DECODE_RELATIONS", "relation": {"mlp": 6, "entity_labels": types, "type_dim": dim},
        "artifact": "weights-f32", "tensor_prefix": "heads.rels"
    })
}

/// Entities over the tokens of TEXT, each tokens `first..end` of a kind.
fn entities(toks: &Tokens, list: &[(u32, u32, u16)]) -> Entities {
    let kinds = TagSet::new("test", &["PER", "LOC", "MISC"]);
    let mut e =
        Entities { spans: Spans::new(ViewId::SOURCE), tokens: Vec::new(), kind: Vec::new(), kinds, score: Vec::new() };
    for &(first, end, kind) in list {
        let (a, b) = (toks.spans.get(Ix::new(first)), toks.spans.get(Ix::new(end - 1)));
        e.spans.push(Span::new(a.start, b.end));
        e.tokens.push(IxRange::new(first, end));
        e.kind.push(Tag(kind));
        e.score.push(1.0);
    }
    e.check().unwrap();
    e
}

/// Ann, Bob, Paris, the river (PER, PER, LOC, LOC) and dog (MISC).
const FOUND: [(u32, u32, u16); 5] = [(0, 1, 0), (2, 3, 0), (4, 5, 1), (13, 15, 1), (10, 11, 2)];

fn features(dir: &std::path::Path, words: &[&str]) -> Vec<f32> {
    let mut enc = WordEncoder::open(dir, &EncoderOptions::default()).unwrap();
    enc.features(&[words.to_vec()], WordPooling::First).unwrap().remove(0)
}

#[test]
fn the_relation_tagger_writes_the_heads_relations_between_entities() {
    let dir = bundle("relations-typed", |m| m["heads"] = json!([relation_head(&["PER", "LOC"])]));
    let doc = Document::new(TEXT);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let toks = tokens(TEXT);
    let ents = entities(&toks, &FOUND);
    let tagger = RelationTagger::open(&dir, None, &EncoderOptions::default()).unwrap();
    assert_eq!(tagger.kinds().len(), 2);
    let (layer, edges) = tagger.annotate((view, &toks, &ents, None, None)).unwrap();
    layer.check().unwrap();

    // The MISC entity is no mention: the head lists PER and LOC only.
    let words: Vec<&str> = TEXT.split(' ').collect();
    let f = features(&dir, &words);
    let ids: Vec<u32> = (0..words.len() as u32).collect();
    let mentions: Vec<Mention> =
        (0..).zip(&FOUND[..4]).map(|(i, &(a, b, _))| Mention { first: a, last: b - 1, entity: i }).collect();
    let types: Vec<u32> = FOUND[..4].iter().map(|f| u32::from(f.2)).collect();
    let (mut scores, mut kept, mut w) = (Vec::new(), Vec::new(), RelationWork::default());
    tagger.head().scores(&f, &ids, &mentions, &types, &mut scores, &mut w).unwrap();
    tagger.head().decode(&scores, mentions.len(), &mut kept).unwrap();
    assert!(!kept.is_empty(), "the seeded head keeps some relation");
    let want: Vec<(usize, usize, u16)> =
        kept.iter().map(|r| (r.head as usize, r.tail as usize, r.label as u16 - 1)).collect();
    let got: Vec<(usize, usize, u16)> =
        (0..layer.len()).map(|i| (layer.subject[i].get(), layer.object[i].get(), layer.kind[i].0)).collect();
    assert_eq!(got, want);
    for (i, (_, s)) in layer.spans.iter().enumerate() {
        let (a, b) = (ents.spans.get(layer.subject[i]), ents.spans.get(layer.object[i]));
        assert_eq!((s.start, s.end), (a.start.min(b.start), a.end.max(b.end)));
    }
    assert!(layer.score.iter().all(|p| *p > 0.5 && *p <= 1.0), "a kept label beats none");
    assert_eq!(edges.len(), layer.len());
    let (again, _) = tagger.annotate((view, &toks, &ents, None, None)).unwrap();
    assert_eq!((again.subject, again.object, again.score), (layer.subject, layer.object, layer.score));
}

#[test]
fn an_entity_across_two_sentences_is_no_mention() {
    let dir = bundle("relations-untyped", |m| m["heads"] = json!([relation_head(&[])]));
    let doc = Document::new(TEXT);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let toks = tokens(TEXT);
    // "Ann met Bob in Paris ," and the rest; ", and" crosses them.
    let mut spans = Spans::new(ViewId::SOURCE);
    let comma = toks.spans.get(Ix::new(5));
    spans.push(Span::new(0, comma.end));
    spans.push(Span::new(toks.spans.get(Ix::new(6)).start, TEXT.len() as u32));
    let sents = Sentences { spans };
    let ents = entities(&toks, &[(0, 1, 0), (5, 7, 2), (2, 3, 0), (13, 15, 1), (10, 11, 2)]);
    let tagger = RelationTagger::open(&dir, None, &EncoderOptions::default()).unwrap();
    let (layer, _) = tagger.annotate((view, &toks, &ents, None, Some(&sents))).unwrap();
    // Without types every entity in one sentence is a mention, but the
    // one across the boundary never is, and no pair spans two sentences.
    let sentence = |e: Ix<Entities>| ents.tokens[e.get()].start.get() >= 6;
    for i in 0..layer.len() {
        assert!(layer.subject[i].get() != 1 && layer.object[i].get() != 1, "row {i}");
        assert_eq!(sentence(layer.subject[i]), sentence(layer.object[i]), "row {i}");
    }
    // The second sentence alone, scored by the core, gives its rows.
    let words: Vec<&str> = TEXT.split(' ').skip(6).collect();
    let f = features(&dir, &words);
    let ids: Vec<u32> = (0..words.len() as u32).collect();
    let mentions = [Mention { first: 7, last: 8, entity: 0 }, Mention { first: 4, last: 4, entity: 1 }];
    let (mut scores, mut kept, mut w) = (Vec::new(), Vec::new(), RelationWork::default());
    tagger.head().scores(&f, &ids, &mentions, &[0, 0], &mut scores, &mut w).unwrap();
    tagger.head().decode(&scores, 2, &mut kept).unwrap();
    let entity = [3usize, 4];
    let want: Vec<(usize, usize)> = kept.iter().map(|r| (entity[r.head as usize], entity[r.tail as usize])).collect();
    let got: Vec<(usize, usize)> = (0..layer.len())
        .filter(|&i| sentence(layer.subject[i]))
        .map(|i| (layer.subject[i].get(), layer.object[i].get()))
        .collect();
    assert!(!want.is_empty(), "the seeded head keeps a relation in the second sentence");
    assert_eq!(got, want);
}

#[test]
fn fewer_than_two_mentions_write_nothing_and_other_kinds_are_refused() {
    let span = json!({
        "name": "mentions", "layer": "turboembed:entities", "kind": "HEAD_SPAN_CLASSIFIER",
        "labels": ["O", "PER"], "scheme": "SCHEME_TAGS", "word_pooling": "WORD_POOLING_FIRST",
        "decode": "DECODE_SPANS_FLAT", "span": {"max_words": 2, "width_dim": 2},
        "artifact": "weights-f32", "tensor_prefix": "heads.mentions"
    });
    let dir = bundle("relations-kinds", |m| m["heads"] = json!([relation_head(&["PER"]), span]));
    let o = EncoderOptions::default();
    assert!(RelationTagger::open(&dir, Some("mentions"), &o).is_err());
    assert!(SpanTagger::open(&dir, Some("rels"), &o).is_err());
    let tagger = RelationTagger::open(&dir, None, &o).unwrap();
    let doc = Document::new(TEXT);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let toks = tokens(TEXT);
    // One PER and two kinds the head does not read.
    let ents = entities(&toks, &[(0, 1, 0), (4, 5, 1), (10, 11, 2)]);
    let (layer, edges) = tagger.annotate((view, &toks, &ents, None, None)).unwrap();
    assert_eq!((layer.len(), edges.len()), (0, 0));
}
