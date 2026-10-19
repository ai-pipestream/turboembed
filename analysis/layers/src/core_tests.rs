use std::sync::Arc;

use turbo_analysis::{
    Anchor, FilterRule, Ix, IxRange, Key, Layer, ModeSet, ModelRef, Overlap, Span, Spans, SplitMode, Tag, TagSet,
    ViewId,
};

use crate::{
    Block, Chunks, Cut, Entities, Language, Phrases, PosTags, Sentences, Strs, Structure, Tokens, Utf16Spans, Vectors,
};

#[test]
fn utf16_spans_are_named_after_their_layer_and_aligned_to_it() {
    assert_eq!(Utf16Spans::<Tokens>::NAME.to_string(), "turbo:utf16<opennlp:tokens>");
    assert_eq!(Utf16Spans::<Sentences>::NAME.to_string(), "turbo:utf16<opennlp:sentences>");
    assert_eq!(Utf16Spans::<Tokens>::anchor(), Anchor::aligned::<Tokens>());
}

#[test]
fn utf16_spans_check_their_columns() {
    // Two halves of one character outside the BMP, then a whole word.
    let ok = Utf16Spans::<Tokens>::new(vec![0, 1, 3], vec![1, 2, 6], vec![true, true, false]);
    assert_eq!(ok.len(), 3);
    assert_eq!(ok.check(), Ok(()));
    let empty = Utf16Spans::<Tokens>::new(Vec::new(), Vec::new(), Vec::new());
    assert_eq!(empty.check(), Ok(()));
    let short = Utf16Spans::<Tokens>::new(vec![0, 1], vec![1, 2], vec![true]);
    assert!(short.check().unwrap_err().contains("a cut flag for every"));
    let short = Utf16Spans::<Tokens>::new(vec![0, 1], vec![1], vec![true, true]);
    assert!(short.check().is_err());
    let backwards = Utf16Spans::<Tokens>::new(vec![0, 4], vec![1, 3], vec![false, false]);
    assert_eq!(backwards.check().unwrap_err(), "UTF-16 span 1 ends before it starts: 4..3");
}

/// Blocks from (start, end, parent).
fn blocks(list: &[(u32, u32, Option<u32>)]) -> Structure {
    let mut spans = Spans::new(ViewId::SOURCE);
    for &(a, b, _) in list {
        spans.push(turbo_analysis::Span::new(a, b));
    }
    let n = list.len();
    let parent = list.iter().map(|&(_, _, p)| p.map(Ix::new)).collect();
    Structure { spans, kind: vec![Block::Paragraph; n], parent, depth: vec![0; n] }
}

#[test]
fn structure_is_a_tree_in_text_order() {
    // A section holding a heading and a list of two items.
    let ok = [(0, 20, None), (0, 5, Some(0)), (7, 20, Some(0)), (7, 12, Some(2)), (13, 20, Some(2)), (21, 30, None)];
    assert_eq!(blocks(&ok).check(), Ok(()));
    assert_eq!(blocks(&[]).check(), Ok(()));
    // Empty blocks may touch their neighbours.
    assert_eq!(blocks(&[(0, 4, None), (1, 1, Some(0)), (1, 3, Some(0))]).check(), Ok(()));
    let err = |list: &[(u32, u32, Option<u32>)]| blocks(list).check().unwrap_err();
    assert!(err(&[(5, 9, None), (0, 4, None)]).contains("starts before"));
    assert!(err(&[(0, 9, Some(0))]).contains("parent at or after"));
    assert!(err(&[(0, 9, None), (2, 10, Some(0))]).contains("outside its parent"));
    assert!(err(&[(0, 9, None), (2, 6, Some(0)), (4, 8, Some(0))]).contains("overlaps"));
    assert!(err(&[(0, 5, None), (3, 8, None)]).contains("overlaps"));
    let mut short = blocks(&ok);
    short.depth.pop();
    assert!(short.check().unwrap_err().contains("a depth for every block"));
}

#[test]
fn language_holds_the_ranking_best_first() {
    let mut code = Strs::default();
    code.push("deu");
    code.push("nld");
    let language = Language { code, score: vec![0.75, 0.25] };
    assert_eq!(language.len(), 2);
    assert_eq!(language.best(), Some(("deu", 0.75)));
    assert_eq!(language.iter().collect::<Vec<_>>(), [("deu", 0.75), ("nld", 0.25)]);
    assert_eq!(language.check(), Ok(()));
    assert_eq!(Language::anchor(), Anchor::Document);
    assert_eq!(Language::SCHEMA, 2);

    let empty = Language { code: Strs::default(), score: Vec::new() };
    assert_eq!(empty.best(), None);
    assert!(empty.is_empty());

    let mut code = Strs::default();
    code.push("deu");
    let short = Language { code, score: vec![0.5, 0.5] };
    assert_eq!(short.check().unwrap_err(), "1 codes and 2 scores");
}

fn toy() -> ModelRef {
    ModelRef { id: "toy".into(), revision: "1".into(), sha256: "0".into() }
}

/// Three sentences of two values, the middle one with no vector.
fn three(empty: Vec<u32>) -> Vectors<Sentences> {
    let data = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    Vectors::new(2, data, empty.into_iter().map(Ix::new).collect(), toy())
}

#[test]
fn vectors_name_the_items_that_have_none() {
    let v = three(vec![1]);
    assert_eq!(v.check(), Ok(()));
    assert_eq!(v.len(), 3);
    let flags: Vec<bool> = (0..3).map(|i| v.is_empty(Ix::new(i))).collect();
    assert_eq!(flags, [false, true, false]);
    assert_eq!(three(Vec::new()).check(), Ok(()));
    assert!(!three(Vec::new()).is_empty(Ix::new(1)));
}

#[test]
fn vectors_check_the_items_with_no_vector() {
    assert!(three(vec![1, 1]).check().unwrap_err().contains("each once"));
    let mut v = three(vec![1]);
    v.data.extend([0.0, 0.0]);
    v.empty = vec![Ix::new(3), Ix::new(1)];
    assert!(v.check().unwrap_err().contains("in order"));
    assert!(three(vec![1, 3]).check().unwrap_err().contains("among the rows"));
    assert!(three(vec![0]).check().unwrap_err().contains("zeros for every item"));
    let v = Vectors::<Sentences>::new(2, vec![0.0; 3], Vec::new(), toy());
    assert!(v.check().unwrap_err().contains("whole number"));
}

#[test]
fn tokens_from_alternatives_materializes_split_modes() {
    let s = |a: u32, b: u32| turbo_analysis::Span::new(a, b);
    let mut alt = turbo_analysis::Alternatives::<Tokens>::new(ViewId::SOURCE);
    alt.push(s(0, 6), 0, 1, SplitMode::A | SplitMode::B);
    alt.push(s(0, 18), 0, 3, SplitMode::C | SplitMode::Best);
    alt.push(s(6, 12), 1, 1, ModeSet::A);
    alt.push(s(6, 18), 1, 2, ModeSet::B);
    alt.push(s(12, 18), 2, 1, ModeSet::A);
    assert_eq!(alt.check(), Ok(()));

    let tok_a = Tokens::from_alternatives(&alt, SplitMode::A).unwrap();
    assert_eq!(tok_a.spans.len(), 3);
    let tok_b = Tokens::from_alternatives(&alt, SplitMode::B).unwrap();
    assert_eq!(tok_b.spans.len(), 2);
    let tok_c = Tokens::from_alternatives(&alt, SplitMode::C).unwrap();
    assert_eq!(tok_c.spans.len(), 1);
}

#[test]
fn pos_tags_and_chunks_check_probabilities_and_sentence_ranges() {
    let tagset = TagSet::new("upos", &["NOUN", "VERB"]);
    let ok_pos = PosTags { tagset: Arc::clone(&tagset), tags: vec![Tag(0), Tag(1)], prob: Some(vec![0.9, 0.8]) };
    assert_eq!(ok_pos.check(), Ok(()));

    let bad_pos = PosTags { tagset, tags: vec![Tag(0), Tag(1)], prob: Some(vec![0.9]) };
    assert_eq!(bad_pos.check().unwrap_err(), "a probability for every tag");

    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 10));
    let mut chunks = Chunks {
        spans,
        level: vec![0],
        parent: vec![None],
        sentences: vec![IxRange { start: Ix::new(3), end: Ix::new(1) }],
        cut: vec![Cut::DocumentStart],
        cost: vec![0],
        model_tokens: vec![5],
        over_budget: vec![false],
    };
    assert_eq!(chunks.check().unwrap_err(), "no sentence range that ends before it starts");
    chunks.sentences[0] = IxRange::new(1, 3);
    assert_eq!(chunks.check(), Ok(()));
}

#[test]
fn entities_allow_nesting_crossing_and_discontinuous_fragments() {
    use turbo_analysis::{Document, Provenance};

    let kinds = TagSet::new("ner", &["org", "loc", "verb"]);
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 16)); // "Bank of New York"
    spans.push(Span::new(8, 16)); // "New York"
    spans.push_fragments(&[Span::new(17, 21), Span::new(27, 30)]); // "took ... off"
    let entities = Entities {
        spans,
        tokens: vec![IxRange::new(0, 4), IxRange::new(2, 4), IxRange::new(4, 7)],
        kind: vec![Tag(0), Tag(1), Tag(2)],
        kinds: Arc::clone(&kinds),
        score: vec![0.6, 0.9, 0.8],
    };
    assert_eq!(Entities::OVERLAP, Overlap::Any);
    assert_eq!(entities.check(), Ok(()));
    assert!(!entities.text_spans().unwrap().frag_item_end.is_empty());

    let longest = entities.filter(FilterRule::LongestFirst);
    assert_eq!(longest.len(), 2);
    assert_eq!(longest.spans.get(Ix::new(0)), Span::new(0, 16));
    assert_eq!(longest.spans.get(Ix::new(1)), Span::new(17, 30));
    assert!(longest.spans.is_discontinuous(Ix::new(1)));

    let highest = entities.filter(FilterRule::HighestScore);
    assert_eq!(highest.len(), 2);
    assert_eq!(highest.spans.get(Ix::new(0)), Span::new(8, 16));
    assert_eq!(highest.spans.get(Ix::new(1)), Span::new(17, 30));

    // Partially crossing entities (0..10 and 6..16) are allowed on Entities
    // (Overlap::Any) so unmerged finders or crossing patterns can coexist,
    // and filter resolves them to a disjoint subset.
    let mut crossing_spans = Spans::new(ViewId::SOURCE);
    crossing_spans.push(Span::new(0, 10));
    crossing_spans.push(Span::new(6, 16));
    let crossing = Entities {
        spans: crossing_spans,
        tokens: vec![IxRange::new(0, 2), IxRange::new(1, 4)],
        kind: vec![Tag(0), Tag(1)],
        kinds: Arc::clone(&kinds),
        score: vec![0.5, 0.9],
    };
    assert_eq!(crossing.check(), Ok(()));
    let resolved = crossing.filter(FilterRule::HighestScore);
    assert_eq!(resolved.len(), 1);
    assert_eq!(resolved.spans.get(Ix::new(0)), Span::new(6, 16));

    // NaN scores are refused so HighestScore has a total order.
    let mut nan_score = crossing;
    nan_score.score[0] = f64::NAN;
    assert!(nan_score.check().unwrap_err().contains("NaN"));

    // Document::add checks fragment boundaries against the view's text:
    // in "café ok" ("é" is bytes 3..5), a fragment ending at byte 4
    // cuts the character even though the envelope 0..7 does not.
    let mut doc = Document::new("café ok");
    let mut t_spans = Spans::new(ViewId::SOURCE);
    t_spans.push(Span::new(0, 5));
    t_spans.push(Span::new(6, 8));
    doc.add(&Key::DEFAULT, Tokens { spans: t_spans, sentence: None }, Provenance::by("test")).unwrap();
    let mut bad_spans = Spans::new(ViewId::SOURCE);
    bad_spans.push_fragments(&[Span::new(0, 4), Span::new(6, 8)]);
    let bad =
        Entities { spans: bad_spans, tokens: vec![IxRange::new(0, 2)], kind: vec![Tag(0)], kinds, score: vec![1.0] };
    let err = doc.add(&Key::DEFAULT, bad, Provenance::by("test")).unwrap_err();
    assert!(err.to_string().contains("cuts a character"), "{err}");
}

#[test]
fn phrases_check_columns_heads_and_parents() {
    let kinds = TagSet::new("phrase", &["S", "NP", "VP"]);
    let mut t_spans = Spans::new(ViewId::SOURCE);
    t_spans.push(Span::new(0, 3));
    t_spans.push(Span::new(4, 7));
    t_spans.push(Span::new(8, 11));
    let tokens = Tokens { spans: t_spans, sentence: None };

    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 11)); // S: 0..3
    spans.push(Span::new(0, 7)); // NP: 0..2
    spans.push(Span::new(8, 11)); // VP: 2..3
    let phrases = Phrases {
        spans,
        tokens: vec![IxRange::new(0, 3), IxRange::new(0, 2), IxRange::new(2, 3)],
        kind: vec![Tag(0), Tag(1), Tag(2)],
        kinds: Arc::clone(&kinds),
        head: vec![Ix::new(2), Ix::new(1), Ix::new(2)],
        parent: vec![None, Some(Ix::new(0)), Some(Ix::new(0))],
    };
    assert_eq!(Phrases::ID, "opennlp:phrases");
    assert_eq!(phrases.check(), Ok(()));
    assert_eq!(phrases.head_span(Ix::new(1), &tokens), Span::new(4, 7));

    let mut bad_head = phrases.clone();
    bad_head.head[1] = Ix::new(2);
    assert!(bad_head.check().unwrap_err().contains("outside its tokens"));

    let mut bad_parent = phrases.clone();
    bad_parent.parent[0] = Some(Ix::new(0));
    assert!(bad_parent.check().unwrap_err().contains("parent at or after it"));

    let mut outside_parent = phrases;
    outside_parent.parent[2] = Some(Ix::new(1));
    assert!(outside_parent.check().unwrap_err().contains("outside its parent"));
}
