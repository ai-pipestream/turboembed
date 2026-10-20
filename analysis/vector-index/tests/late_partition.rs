//! Chunks that split a text at piece boundaries hold every piece once, and
//! the document holds all of them, with empty pieces anywhere: inside a
//! chunk, on a boundary, in runs, at the start and at the end.

mod common;

use common::model;
use turbo_analysis::{Ix, Span, Spans, ViewId};
use turbo_analysis_layers::{Levels, Subwords, Unit, Vectors};
use turbo_vector_index::{Normalization, Space, TokenIndexBuilder};

struct Lcg(u64);

impl Lcg {
    fn below(&mut self, n: u64) -> u64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) % n
    }
}

#[test]
fn chunks_that_split_a_text_hold_each_piece_once() {
    for seed in 0..200u64 {
        let mut r = Lcg(seed * 0x9e37_79b9 + 1);
        // Pieces in text order: a run of empty ones before each non-empty
        // piece and at the end; non-empty pieces 1 to 3 bytes, a gap of 0
        // or 1 byte between them.
        let (mut pieces, mut at, mut starts) = (Vec::new(), 0u32, vec![0u32]);
        for _ in 0..(1 + r.below(12)) {
            for _ in 0..r.below(3) {
                pieces.push((at, at));
            }
            let len = 1 + r.below(3) as u32;
            pieces.push((at, at + len));
            at += len;
            starts.push(at);
            at += r.below(2) as u32;
        }
        for _ in 0..r.below(3) {
            pieces.push((at, at));
        }
        let end = at;
        // Cut the text into chunks at some of the piece boundaries.
        let mut cuts: Vec<u32> = starts.into_iter().filter(|&s| s > 0 && s < end && r.below(2) == 0).collect();
        cuts.dedup();
        let mut bounds = vec![0];
        bounds.extend(cuts);
        bounds.push(end);
        let m = model("colbert-small");
        let mut spans = Spans::new(ViewId::SOURCE);
        for &(a, b) in &pieces {
            spans.push(Span::new(a, b));
        }
        let n = pieces.len();
        let sub = Subwords { spans, ids: (0..n as i32).collect(), model: m.clone(), token: None };
        let vec = Vectors::new(1, vec![1.0; n], Vec::new(), m.clone());
        let mut spans = Spans::new(ViewId::SOURCE);
        spans.push(Span::new(0, end));
        let mut unit = vec![Unit::Document];
        for (c, w) in bounds.windows(2).enumerate() {
            spans.push(Span::new(w[0], w[1]));
            unit.push(Unit::Chunk(Ix::new(c as u32)));
        }
        let units = unit.len();
        let mut parent = vec![None];
        parent.resize(units, Some(Ix::new(0)));
        let lev = Levels { spans, unit, parent };
        let mut b = TokenIndexBuilder::new(Space::new(m, 1, Normalization::None)).unwrap();
        b.push_levels(&vec, &sub, &lev, (0..units as u32).map(Ix::new), None, |u| u.get()).unwrap();
        let index = b.build();
        let mut seen = vec![0; n];
        for c in 0..index.len() as u32 {
            for s in index.sources(c) {
                seen[s.piece as usize] += if *index.meta(c) == 0 { 100 } else { 1 };
            }
        }
        assert!(seen.iter().all(|&s| s == 101), "seed {seed}: pieces {pieces:?}, chunks {bounds:?}, seen {seen:?}");
    }
}
