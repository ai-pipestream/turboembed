//! Candidates made from the subword vectors of a text's levels: a unit's
//! tokens are the pieces inside its span, less empty items and pruned
//! pieces, and a match names the piece and its bytes.

mod common;

use common::model;
use turbo_analysis::{Ix, ModelRef, Span, Spans, ViewId};
use turbo_analysis_layers::{Levels, Subwords, Unit, Vectors};
use turbo_vector_index::{IndexError, Normalization, Prune, Source, Space, TokenIndexBuilder};

const TEXT: &str = "Hello, world. Fine day!";

/// The pieces of TEXT, in order.
const PIECES: [(u32, u32); 7] = [(0, 5), (5, 6), (7, 12), (12, 13), (14, 18), (19, 22), (22, 23)];

fn subwords(m: ModelRef) -> Subwords {
    let mut spans = Spans::new(ViewId::SOURCE);
    for (a, b) in PIECES {
        spans.push(Span::new(a, b));
    }
    Subwords { spans, ids: (0..PIECES.len() as i32).collect(), model: m, token: None }
}

/// Piece `i` is the unit vector on axis `i % 4`, scaled by `i + 1`, except
/// piece 4 ("Fine"), which is an empty item.
fn vectors(m: ModelRef) -> Vectors<Subwords> {
    let mut data = vec![0.0f32; PIECES.len() * 4];
    for i in 0..PIECES.len() {
        if i != 4 {
            data[i * 4 + i % 4] = (i + 1) as f32;
        }
    }
    Vectors::new(4, data, vec![Ix::new(4)], m)
}

/// The document; a section of two fragments, "Hello," and "Fine"; chunks
/// "Hello, world." and "Fine day!"; and an empty chunk at the end.
fn levels() -> Levels {
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 23));
    spans.push_fragments(&[Span::new(0, 6), Span::new(14, 18)]);
    spans.push(Span::new(0, 13));
    spans.push(Span::new(14, 23));
    spans.push(Span::new(23, 23));
    let unit = vec![
        Unit::Document,
        Unit::Section(Ix::new(0)),
        Unit::Chunk(Ix::new(0)),
        Unit::Chunk(Ix::new(1)),
        Unit::Chunk(Ix::new(2)),
    ];
    let parent = vec![None, Some(Ix::new(0)), Some(Ix::new(0)), Some(Ix::new(0)), Some(Ix::new(0))];
    Levels { spans, unit, parent }
}

fn pieces(sources: &[Source]) -> Vec<u32> {
    sources.iter().map(|s| s.piece).collect()
}

#[test]
fn a_unit_holds_the_pieces_inside_its_span() {
    let m = model("colbert-small");
    let sp = Space::new(m.clone(), 4, Normalization::None);
    let (sub, vec, lev) = (subwords(m.clone()), vectors(m), levels());
    let mut b = TokenIndexBuilder::new(sp.clone()).unwrap();
    let units = (0..5).map(Ix::new);
    assert_eq!(b.push_levels(&vec, &sub, &lev, units, None, |u| u.get()), Ok(4));
    let index = b.build();
    // The empty chunk has no piece and is not added; "Fine" is an empty item.
    assert_eq!((0..4).map(|c| *index.meta(c)).collect::<Vec<_>>(), [0, 1, 2, 3]);
    assert_eq!(pieces(index.sources(0)), [0, 1, 2, 3, 5, 6]);
    assert_eq!(pieces(index.sources(1)), [0, 1]);
    assert_eq!(pieces(index.sources(2)), [0, 1, 2, 3]);
    assert_eq!(pieces(index.sources(3)), [5, 6]);
    assert_eq!(index.sources(3)[1].span, Some(Span::new(22, 23)));
    assert!(index.pruning(0).is_none());
    // A query on axis 1 is won by "day" (piece 5, 6 on axis 1) in the
    // document and by "," (piece 1, 2 on axis 1) in the first chunk.
    let q = [0.0, 1.0, 0.0, 0.0];
    let (score, m0) = index.explain(&sp, &q, 0).unwrap();
    assert_eq!((score, m0[0].source.piece, m0[0].source.span), (6.0, 5, Some(Span::new(19, 22))));
    let (_, m2) = index.explain(&sp, &q, 2).unwrap();
    assert_eq!((m2[0].token, m2[0].source.piece, m2[0].score), (1, 1, 2.0));
    let hits = index.top_k(&sp, &q, 4).unwrap();
    assert_eq!(hits.iter().map(|h| h.candidate).collect::<Vec<_>>(), [0, 3, 1, 2]);
}

#[test]
fn pruned_pieces_are_dropped_and_recorded() {
    let m = model("colbert-small");
    let sp = Space::new(m.clone(), 4, Normalization::None);
    let (sub, vec, lev) = (subwords(m.clone()), vectors(m), levels());
    let punctuation = |ix: Ix<Subwords>| TEXT[sub.spans.get(ix).range()].chars().all(|c| c.is_ascii_punctuation());
    let prune = Prune { rule: "punctuation", drop: &punctuation };
    let mut b = TokenIndexBuilder::new(sp.clone()).unwrap();
    assert_eq!(b.push_levels(&vec, &sub, &lev, [Ix::new(0), Ix::new(3)], Some(&prune), |u| u.get()), Ok(2));
    let index = b.build();
    assert_eq!(pieces(index.sources(0)), [0, 2, 5]);
    assert_eq!(index.pruning(0), Some(("punctuation", 3)));
    assert_eq!(pieces(index.sources(1)), [5]);
    assert_eq!(index.pruning(1), Some(("punctuation", 1)));
}

#[test]
fn layers_of_another_model_or_view_or_out_of_step_add_nothing() {
    let m = model("colbert-small");
    let sp = Space::new(m.clone(), 4, Normalization::None);
    let (sub, vec, lev) = (subwords(m.clone()), vectors(m.clone()), levels());
    let mut b = TokenIndexBuilder::new(sp.clone()).unwrap();
    b.push(&[1.0, 0.0, 0.0, 0.0], usize::MAX).unwrap();
    let all = || (0..5).map(Ix::new);
    let other = model("bge-small");
    assert!(matches!(
        b.push_levels(&vectors(other.clone()), &sub, &lev, all(), None, |u| u.get()),
        Err(IndexError::Model { .. })
    ));
    assert!(matches!(
        b.push_levels(&vec, &subwords(other), &lev, all(), None, |u| u.get()),
        Err(IndexError::Model { .. })
    ));
    let short = Vectors::new(4, vec.data[..24].to_vec(), vec![Ix::new(4)], m.clone());
    assert!(matches!(b.push_levels(&short, &sub, &lev, all(), None, |u| u.get()), Err(IndexError::Layer(_))));
    let mut moved = levels();
    moved.spans = {
        let mut s = Spans::new(ViewId(1));
        for (_, span) in lev.spans.iter() {
            s.push(span);
        }
        s
    };
    assert!(matches!(b.push_levels(&vec, &sub, &moved, [Ix::new(0)], None, |u| u.get()), Err(IndexError::Layer(_))));
    // A unit past the layer refuses the call, and the units before it are
    // taken back out.
    assert!(matches!(
        b.push_levels(&vec, &sub, &lev, [Ix::new(0), Ix::new(9)], None, |u| u.get()),
        Err(IndexError::Layer(_))
    ));
    assert_eq!(b.len(), 1);
    assert_eq!(b.push(&[0.0, 1.0, 0.0, 0.0], 5), Ok(1));
    let index = b.build();
    assert_eq!((index.total_tokens(), *index.meta(1)), (2, 5));
}
