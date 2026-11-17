//! The phrases a built [`TokenGraph`] accepts (`#2340`): every path from
//! the first position to the last reads the base tokens or one whole
//! alternative, never the head of one alternative and the tail of another.

use std::collections::BTreeSet;

use turbo_analysis::{Ix, Layer, Span, ViewId};
use turbo_analysis_layers::{TokenArcKind, TokenGraph, TokenGraphBuilder};

/// Every term sequence read along a path from position 0 to the end.
fn phrases(graph: &TokenGraph) -> BTreeSet<String> {
    fn walk(graph: &TokenGraph, at: u32, so_far: &mut Vec<String>, out: &mut BTreeSet<String>) {
        if at == graph.num_positions {
            out.insert(so_far.join(" "));
            return;
        }
        for i in 0..graph.len() {
            let arc = graph.get(Ix::new(i as u32));
            if arc.position == at {
                so_far.push(arc.term.to_owned());
                walk(graph, at + arc.position_length, so_far, out);
                so_far.pop();
            }
        }
    }
    let mut out = BTreeSet::new();
    walk(graph, 0, &mut Vec::new(), &mut out);
    out
}

fn builder(words: &[&str]) -> (TokenGraphBuilder, Vec<Span>) {
    let mut spans = Vec::new();
    let mut at = 0u32;
    for w in words {
        spans.push(Span::new(at, at + w.len() as u32));
        at += w.len() as u32 + 1;
    }
    let terms = words.iter().map(|w| (*w).to_owned()).collect();
    (TokenGraphBuilder::from_base_tokens(ViewId::SOURCE, spans.clone(), terms), spans)
}

fn set(items: &[&str]) -> BTreeSet<String> {
    items.iter().map(|s| (*s).to_owned()).collect()
}

#[test]
fn two_expansions_of_one_token_do_not_cross() {
    let (mut b, _) = builder(&["fly", "x", "today"]);
    b.expand_token_sequence(1, ["new", "york"], TokenArcKind::Synonym);
    b.expand_token_sequence(1, ["los", "angeles"], TokenArcKind::Synonym);
    let graph = b.finish().unwrap();
    graph.check().unwrap();
    assert_eq!(phrases(&graph), set(&["fly x today", "fly new york today", "fly los angeles today"]));
}

#[test]
fn expansions_of_unequal_length_do_not_cross() {
    let (mut b, _) = builder(&["x"]);
    b.expand_token_sequence(0, ["new", "york", "city"], TokenArcKind::Synonym);
    b.expand_token_sequence(0, ["big", "apple"], TokenArcKind::Synonym);
    b.expand_token_sequence(0, ["nyc"], TokenArcKind::Synonym);
    let graph = b.finish().unwrap();
    assert_eq!(phrases(&graph), set(&["x", "nyc", "new york city", "big apple"]));
}

#[test]
fn a_split_and_an_expansion_of_one_token_do_not_cross() {
    let (mut b, spans) = builder(&["database"]);
    let s = spans[0];
    b.split_token(
        0,
        [
            (Span::new(s.start, s.start + 4), "data".to_owned(), TokenArcKind::CompoundPart),
            (Span::new(s.start + 4, s.end), "base".to_owned(), TokenArcKind::CompoundPart),
        ],
        TokenArcKind::CompoundWhole,
    );
    b.expand_token_sequence(0, ["record", "store"], TokenArcKind::Synonym);
    let graph = b.finish().unwrap();
    assert_eq!(phrases(&graph), set(&["database", "data base", "record store"]));
}

#[test]
fn a_phrase_synonym_does_not_cross_the_tokens_it_spans() {
    let (mut b, spans) = builder(&["in", "new", "york", "city", "now"]);
    let span = Span::new(spans[1].start, spans[3].end);
    b.add_span_sequence(1..4, span, ["big", "apple"], TokenArcKind::Synonym);
    b.add_span_arc(1..4, span, "nyc", TokenArcKind::Synonym);
    let graph = b.finish().unwrap();
    assert_eq!(phrases(&graph), set(&["in new york city now", "in big apple now", "in nyc now"]));
}

#[test]
fn overlapping_phrase_synonyms_keep_to_their_own_paths() {
    let (mut b, spans) = builder(&["a", "b", "c"]);
    b.add_span_sequence(0..2, Span::new(spans[0].start, spans[1].end), ["p", "q", "r"], TokenArcKind::Synonym);
    b.add_span_sequence(1..3, Span::new(spans[1].start, spans[2].end), ["s", "t"], TokenArcKind::Synonym);
    let graph = b.finish().unwrap();
    assert_eq!(phrases(&graph), set(&["a b c", "p q r c", "a s t"]));
}

#[test]
fn a_repeated_expansion_is_one_path() {
    let (mut b, _) = builder(&["x"]);
    b.expand_token_sequence(0, ["new", "york"], TokenArcKind::Synonym);
    b.expand_token_sequence(0, ["new", "york"], TokenArcKind::Synonym);
    let graph = b.finish().unwrap();
    assert_eq!(phrases(&graph), set(&["x", "new york"]));
}
