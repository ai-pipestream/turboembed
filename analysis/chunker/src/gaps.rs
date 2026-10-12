//! What the document's structure says about each gap between two
//! sentences: the strongest break there, whether a heading ends there,
//! and the smallest unit no cut may fall inside that holds sentences on
//! both sides. The boundary costs and the chunker's hard limits are both
//! read from this, so the two never disagree about where a unit starts.

use std::ops::Range;

use turbo_analysis::Spans;
use turbo_analysis_layers::{Block, Cut, Sentences, Structure};

/// A unit no cut may fall inside unless it alone is over the budget: a
/// table row, a code block or a quote.
pub fn is_unit(kind: Block) -> bool {
    matches!(kind, Block::TableRow | Block::Code | Block::Quote)
}

/// A table row, code block or quote that holds sentences on both sides of
/// a gap.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unit {
    pub kind: Block,
    /// The sentences it holds a byte of.
    pub sentences: Range<usize>,
}

/// The gap after one sentence. The gap after the last sentence is the end
/// of the text: nothing is inside it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gap {
    /// The strongest break at the gap; `Cut::Sentence` when no block
    /// starts or ends there.
    pub reason: Cut,
    /// A heading ends at the gap, so a cut there would part it from what
    /// it heads.
    pub after_heading: bool,
    /// The smallest unit that holds sentences on both sides, if any.
    pub inside: Option<Unit>,
}

/// How strong a break is: a section break beats a heading, which beats
/// the end of a paragraph, which beats the end of a list item or table
/// row, which beats a sentence boundary.
fn rank(c: Cut) -> u8 {
    match c {
        Cut::Section => 4,
        Cut::Heading => 3,
        Cut::Paragraph => 2,
        Cut::Item => 1,
        _ => 0,
    }
}

/// The break a block makes where it starts or ends.
fn edge(kind: Block, at_start: bool) -> Cut {
    match kind {
        Block::Section => Cut::Section,
        Block::Heading if at_start => Cut::Heading,
        Block::ListItem | Block::TableRow => Cut::Item,
        _ => Cut::Paragraph,
    }
}

/// One gap per sentence, the last one the end of the text. A block holds
/// the sentences it has a byte of, found with two binary searches; it
/// starts at the gap before the first of them and ends at the gap after
/// the last. A block that holds no sentence's byte (whitespace between
/// two of them, or an empty block) makes no break. Time is the number of
/// blocks times a binary search, plus the gaps inside each unit.
///
/// Only the sentence detector says where sentences are: a block edge that
/// falls inside a sentence moves to the gap before or after it.
pub fn gaps(sentences: &Spans<Sentences>, structure: Option<&Structure>) -> Vec<Gap> {
    let n = sentences.len();
    let mut out = vec![Gap { reason: Cut::Sentence, after_heading: false, inside: None }; n];
    let Some(st) = structure else { return out };
    let (starts, ends) = sentences.columns();
    let mut set = |g: usize, c: Cut| {
        if rank(c) > rank(out[g].reason) {
            out[g].reason = c;
        }
    };
    // The byte length of the unit inside each gap, to prefer the smaller
    // of two that hold the same sentences.
    let mut inner_len = vec![u32::MAX; n];
    let mut units = Vec::new();
    for (i, s) in st.spans.iter() {
        let kind = st.kind[i.get()];
        let first = ends.partition_point(|&e| e <= s.start);
        let last = starts.partition_point(|&x| x < s.end);
        if first >= last {
            continue;
        }
        if first > 0 {
            set(first - 1, edge(kind, true));
        }
        set(last - 1, edge(kind, false));
        if kind == Block::Heading {
            units.push((last - 1, None));
        }
        if is_unit(kind) && last - first > 1 {
            units.push((first, Some((kind, first..last, s.len()))));
        }
    }
    for (g, unit) in units {
        let Some((kind, r, len)) = unit else {
            out[g].after_heading = true;
            continue;
        };
        for gap in r.start..r.end - 1 {
            let smaller = match &out[gap].inside {
                None => true,
                Some(u) => (r.len(), len) < (u.sentences.len(), inner_len[gap]),
            };
            if smaller {
                out[gap].inside = Some(Unit { kind, sentences: r.clone() });
                inner_len[gap] = len;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_analysis::{Ix, Span, ViewId};

    fn sentences(spans: &[(u32, u32)]) -> Spans<Sentences> {
        let mut s = Spans::new(ViewId::SOURCE);
        for &(a, b) in spans {
            s.push(Span::new(a, b));
        }
        s
    }

    fn structure(blocks: &[(u32, u32, Block, Option<u32>)]) -> Structure {
        let mut spans = Spans::new(ViewId::SOURCE);
        for &(a, b, ..) in blocks {
            spans.push(Span::new(a, b));
        }
        Structure {
            spans,
            kind: blocks.iter().map(|b| b.2).collect(),
            parent: blocks.iter().map(|b| b.3.map(Ix::new)).collect(),
            depth: vec![0; blocks.len()],
        }
    }

    // Six sentences of nine bytes and a byte of space after each.
    const SIX: [(u32, u32); 6] = [(0, 9), (10, 19), (20, 29), (30, 39), (40, 49), (50, 59)];

    #[test]
    fn without_structure_every_gap_is_a_sentence_boundary() {
        let g = gaps(&sentences(&SIX), None);
        assert!(g.iter().all(|g| g.reason == Cut::Sentence && !g.after_heading && g.inside.is_none()));
        assert_eq!(g.len(), 6);
    }

    #[test]
    fn each_gap_takes_its_strongest_break() {
        // A section, its heading (sentence 0), a paragraph (1 and 2) and a
        // list of two items (3, then 4 and 5).
        let st = structure(&[
            (0, 59, Block::Section, None),
            (0, 9, Block::Heading, Some(0)),
            (10, 29, Block::Paragraph, Some(0)),
            (30, 59, Block::List, Some(0)),
            (30, 39, Block::ListItem, Some(3)),
            (40, 59, Block::ListItem, Some(3)),
        ]);
        let g = gaps(&sentences(&SIX), Some(&st));
        let reasons: Vec<Cut> = g.iter().map(|g| g.reason).collect();
        assert_eq!(reasons, [Cut::Paragraph, Cut::Sentence, Cut::Paragraph, Cut::Item, Cut::Sentence, Cut::Section]);
        let after: Vec<bool> = g.iter().map(|g| g.after_heading).collect();
        assert_eq!(after, [true, false, false, false, false, false]);
        assert!(g.iter().all(|g| g.inside.is_none()));
    }

    #[test]
    fn a_heading_starts_where_it_starts() {
        let st = structure(&[(0, 19, Block::Paragraph, None), (20, 29, Block::Heading, None)]);
        let g = gaps(&sentences(&SIX), Some(&st));
        assert_eq!(g[1].reason, Cut::Heading);
        assert!(g[2].after_heading);
    }

    #[test]
    fn the_gaps_inside_a_unit_name_it() {
        // A code block over sentences 1 to 3, a table row over 4 and 5.
        let st = structure(&[(10, 39, Block::Code, None), (40, 59, Block::TableRow, None)]);
        let g = gaps(&sentences(&SIX), Some(&st));
        let code = Some(Unit { kind: Block::Code, sentences: 1..4 });
        let row = Some(Unit { kind: Block::TableRow, sentences: 4..6 });
        let inside: Vec<Option<Unit>> = g.iter().map(|g| g.inside.clone()).collect();
        assert_eq!(inside, [None, code.clone(), code, None, row, None]);
        // The code block's end outranks the row's start.
        assert_eq!((g[0].reason, g[3].reason, g[5].reason), (Cut::Paragraph, Cut::Paragraph, Cut::Item));
    }

    #[test]
    fn the_smallest_unit_around_a_gap_is_the_one_named() {
        // A quote over every sentence, holding a code block over 2 to 4.
        let st = structure(&[(0, 59, Block::Quote, None), (20, 49, Block::Code, Some(0))]);
        let g = gaps(&sentences(&SIX), Some(&st));
        let kinds: Vec<Option<Block>> = g.iter().map(|g| g.inside.as_ref().map(|u| u.kind)).collect();
        let (q, c) = (Some(Block::Quote), Some(Block::Code));
        assert_eq!(kinds, [q, q, c, c, q, None]);
        assert_eq!(g[0].inside.as_ref().unwrap().sentences, 0..6);
    }

    #[test]
    fn a_unit_over_the_same_sentences_as_another_is_the_shorter() {
        // A quote and the code block in it hold the same two sentences;
        // the quote has a byte of markup more.
        let st = structure(&[(9, 29, Block::Quote, None), (10, 29, Block::Code, Some(0))]);
        let g = gaps(&sentences(&SIX), Some(&st));
        assert_eq!(g[1].inside, Some(Unit { kind: Block::Code, sentences: 1..3 }));
        // In the other order too.
        let st = structure(&[(10, 29, Block::Code, None), (9, 29, Block::Quote, None)]);
        assert_eq!(gaps(&sentences(&SIX), Some(&st))[1].inside.as_ref().unwrap().kind, Block::Code);
    }

    #[test]
    fn a_block_edge_inside_a_sentence_moves_to_its_gaps() {
        // A code block from the middle of sentence 1 to the middle of 2:
        // it holds bytes of both, so the gap between them is inside it.
        let st = structure(&[(14, 25, Block::Code, None)]);
        let g = gaps(&sentences(&SIX), Some(&st));
        assert_eq!(g[0].reason, Cut::Paragraph);
        assert_eq!(g[1].inside, Some(Unit { kind: Block::Code, sentences: 1..3 }));
        assert_eq!(g[2].reason, Cut::Paragraph);
    }

    #[test]
    fn a_block_with_no_sentence_bytes_makes_no_break() {
        // The space between sentences 1 and 2, and an empty block.
        let st = structure(&[(19, 20, Block::Paragraph, None), (30, 30, Block::Code, None)]);
        let g = gaps(&sentences(&SIX), Some(&st));
        assert!(g.iter().all(|g| g.reason == Cut::Sentence && g.inside.is_none()));
    }

    #[test]
    fn a_unit_of_one_sentence_holds_no_gap() {
        let st = structure(&[(10, 19, Block::Quote, None)]);
        let g = gaps(&sentences(&SIX), Some(&st));
        assert!(g.iter().all(|g| g.inside.is_none()));
        assert_eq!((g[0].reason, g[1].reason), (Cut::Paragraph, Cut::Paragraph));
    }
}
