//! The coreference annotator end to end on the small BERT in testdata,
//! with a coreference ranker head of seeded weights: chains and links
//! over an entity layer as the core head decodes them, entities over one
//! range sharing a chain, a mention past `span.max_words` standing alone,
//! determinism, and refusals.

mod common;

use common::*;
use serde_json::{Value, json};
use turbo::head::coref::chains;
use turbo::head::tagger::Work;
use turbo::manifest::WordPooling;
use turbo_analysis::{Annotator, Document, Ix, IxRange, Key, Layer, Span, Spans, TagSet, View};
use turbo_analysis_embed::EncoderOptions;
use turbo_analysis_layers::{Entities, Tag};
use turbo_tagger::{CorefTagger, WordEncoder};

fn coref_head(antecedents: u32) -> Value {
    json!({
        "name": "coref", "layer": "turboembed:coref", "kind": "HEAD_COREF_RANKER",
        "labels": ["coref"], "scheme": "SCHEME_TAGS", "word_pooling": "WORD_POOLING_FIRST",
        "decode": "DECODE_ANTECEDENT", "span": {"max_words": 2, "width_dim": 3},
        "coref": {"dim": 6, "antecedents": antecedents},
        "artifact": "weights-f32", "tensor_prefix": "heads.coref"
    })
}

/// Entities over these token ranges (end exclusive) of TEXT, in this
/// order: Ann, Bob, Paris, "the quiet old dog" (four tokens, past
/// max_words), "dog", Ann again, and "the river".
const RANGES: [(u32, u32); 7] = [(0, 1), (2, 3), (4, 5), (7, 11), (10, 11), (0, 1), (14, 16)];

fn entities(toks: &turbo_analysis_layers::Tokens) -> Entities {
    let mut e = Entities {
        spans: Spans::new(toks.spans.view()),
        tokens: Vec::new(),
        kind: Vec::new(),
        kinds: TagSet::new("mentions", &["X"]),
        score: Vec::new(),
    };
    for (a, b) in RANGES {
        e.spans.push(Span::new(toks.spans.get(Ix::new(a)).start, toks.spans.get(Ix::new(b - 1)).end));
        e.tokens.push(IxRange::new(a, b));
        e.kind.push(Tag(0));
        e.score.push(1.0);
    }
    e
}

#[test]
fn the_chains_are_the_core_heads_links_over_the_entities() {
    let dir = bundle("coref", |m| m["heads"] = json!([coref_head(4)]));
    let doc = Document::new(TEXT);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let toks = tokens(TEXT);
    let ents = entities(&toks);
    let tagger = CorefTagger::open(&dir, None, &EncoderOptions::default()).unwrap();
    let (groups, edges) = tagger.annotate((view, &toks, &ents, None)).unwrap();
    groups.check().unwrap();
    edges.check().unwrap();

    // The core head on the distinct ranges of at most two words.
    let scored: Vec<(u32, u32)> = vec![(0, 0), (2, 2), (4, 4), (10, 10), (14, 15)];
    let words: Vec<&str> = TEXT.split(' ').collect();
    let mut enc = WordEncoder::open(&dir, &EncoderOptions::default()).unwrap();
    let f = enc.features(std::slice::from_ref(&words), WordPooling::First).unwrap().remove(0);
    let ids: Vec<u32> = (0..words.len() as u32).collect();
    let (mut g, mut scores, mut best, mut w) = (Vec::new(), Vec::new(), Vec::new(), Work::default());
    tagger.head().mentions(&f, &ids, &scored, &mut g, &mut w).unwrap();
    tagger.head().scores(&g, &mut scores).unwrap();
    tagger.head().decode(&scores, &mut best).unwrap();
    let chain = chains(&best).unwrap();

    // Entity e's scored mention, or None for the long one (entity 3).
    let mention = |e: usize| scored.iter().position(|&(i, j)| (i, j + 1) == RANGES[e]);
    let group_of = |e: usize| groups.groups_containing(Ix::new(e as u32)).map(|(g, _)| g.get()).collect::<Vec<_>>();
    for e in 0..RANGES.len() {
        assert_eq!(group_of(e).len(), 1, "entity {e} is in exactly one chain");
    }
    for (a, ra) in RANGES.iter().enumerate() {
        for (b, rb) in RANGES.iter().enumerate() {
            let same = match (mention(a), mention(b)) {
                (Some(x), Some(y)) => chain[x] == chain[y],
                _ => a == b || ra == rb,
            };
            assert_eq!(group_of(a) == group_of(b), same, "entities {a} and {b}");
        }
    }
    // The two Ann entities share a chain, and the long mention is alone.
    assert_eq!(group_of(0), group_of(5));
    assert_eq!(groups.get(Ix::new(group_of(3)[0] as u32)).len(), 1);
    // One edge from each entity of a linked mention, to an earlier one.
    let want: usize = (0..RANGES.len()).filter(|&e| mention(e).is_some_and(|m| best[m].is_some())).count();
    assert_eq!(edges.from.len(), want);
    for ((from, to), score) in edges.from.iter().zip(&edges.to).zip(&edges.score) {
        let (from, to) = (from.get(), to.get());
        assert!(RANGES[to] < RANGES[from], "{to} before {from}");
        assert!(*score > 0.0 && *score <= 1.0);
    }

    let (again, _) = tagger.annotate((view, &toks, &ents, None)).unwrap();
    let members = |gr: &turbo_analysis::Groups<Entities>| gr.iter().map(|(_, m)| m.to_vec()).collect::<Vec<_>>();
    assert_eq!(members(&again), members(&groups));
}

#[test]
fn bad_entities_and_wrong_heads_are_refused() {
    let dir = bundle("coref-refuse", |m| m["heads"] = json!([coref_head(2), heads()[0].clone()]));
    let o = EncoderOptions::default();
    assert!(CorefTagger::open(&dir, Some("pos"), &o).is_err());
    let tagger = CorefTagger::open(&dir, None, &o).unwrap();
    let doc = Document::new(TEXT);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let toks = tokens(TEXT);
    let mut ents = entities(&toks);
    ents.tokens[1] = IxRange::new(2, 99);
    assert!(tagger.annotate((view, &toks, &ents, None)).is_err());
}
