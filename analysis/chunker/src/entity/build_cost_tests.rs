//! Cost bounds of the header build and its match with the ee8d4c0e
//! build on random documents.

use std::time::Instant;

use turbo_analysis::{Ix, IxRange, Span, Spans, ViewId};
use turbo_analysis_layers::{Chunks, Cut};

use super::oracle;
use super::{carry_ops, chunk_entities, ops};
use crate::entity::{EntityKeys, HeaderPolicy, KeyMention};

/// Byte fragments as `(start, end)`.
type Frags = Vec<(u32, u32)>;
/// A mention as `(key, sentence, fragments)`.
type Mention = (u32, u32, Frags);
/// A chunk row as `(fragments, level, parent)`.
type Row = (Frags, u8, Option<u32>);

/// Keys over mentions, put in start order.
pub(super) fn keys_of(nk: usize, mut ms: Vec<Mention>) -> EntityKeys {
    ms.sort_by_key(|m| (m.2[0].0, m.2.last().map_or(0, |f| f.1), m.0));
    ms.dedup_by(|a, b| a.0 == b.0 && a.2[0].0 == b.2[0].0 && a.2.last() == b.2.last());
    let mut k = EntityKeys { first: (0..nk as u32).map(Ix::new).collect(), ..EntityKeys::default() };
    k.occurrences = vec![Vec::new(); nk];
    for (key, sentence, frags) in ms {
        let from = k.fragments.len() as u32;
        k.fragments.extend(frags.iter().map(|&(a, b)| Span::new(a, b)));
        let span = Span::new(frags[0].0, frags.last().map_or(0, |f| f.1));
        k.mentions.push(KeyMention { key, sentence, span, fragments: (from, k.fragments.len() as u32) });
    }
    k
}

/// Chunks over rows.
pub(super) fn chunks_of(rows: &[Row]) -> Chunks {
    let mut spans = Spans::new(ViewId::SOURCE);
    for (frags, _, _) in rows {
        let frags: Vec<Span> = frags.iter().map(|&(a, b)| Span::new(a, b)).collect();
        spans.push_fragments(&frags);
    }
    let n = rows.len();
    Chunks {
        spans,
        level: rows.iter().map(|r| r.1).collect(),
        parent: rows.iter().map(|r| r.2.map(Ix::new)).collect(),
        sentences: vec![IxRange::new(0, 0); n],
        cut: vec![Cut::Sentence; n],
        cost: vec![0; n],
        model_tokens: vec![1; n],
        over_budget: vec![false; n],
        prompt_overhead: 0,
    }
}

/// `k` entities in sentence 0, then `k` sentence chunks naming none.
fn crowded_then_empty(k: u32) -> (EntityKeys, Chunks) {
    let keys = keys_of(k as usize, (0..k).map(|i| (i, 0, vec![(2 * i, 2 * i + 1)])).collect());
    let mut rows = vec![(vec![(0, 2 * k)], 0, None)];
    rows.extend((0..k).map(|i| (vec![(2 * k + 2 * i, 2 * k + 2 * i + 2)], 0, None)));
    (keys, chunks_of(&rows))
}

#[test]
fn a_crowded_sentence_is_not_rescanned_by_every_later_chunk() {
    let (keys, chunks) = crowded_then_empty(4_000);
    let before = carry_ops();
    let h = chunk_entities(&chunks, &keys, HeaderPolicy::default());
    let steps = carry_ops() - before;
    let last: Vec<(u32, u32)> = h.entries(Ix::new(4_000)).collect();
    assert_eq!(last, [(0, 0), (1, 0), (2, 0), (3, 0)]);
    assert_eq!(h.entries(Ix::new(0)).count(), 4_000);
    println!("crowded: {steps} carried steps for 4001 chunks");
    assert!(steps <= 4_001 * 8, "{steps} carried steps for 4001 chunks");
    let (keys, chunks) = crowded_then_empty(300);
    assert_eq!(h_of(&chunks, &keys, 4), oracle::chunk_entities(&chunks, &keys, HeaderPolicy::default()));
}

/// One mention in fragments at both ends of `n` ten-byte sentence
/// chunks, each of which names its own entity too.
fn split_mention(n: u32) -> (EntityKeys, Chunks) {
    let end = 10 * n;
    let mut ms = vec![(0, 0, vec![(0, 5), (end - 10, end)])];
    ms.extend((1..n).map(|i| (i, i, vec![(10 * i, 10 * i + 3)])));
    let rows: Vec<_> = (0..n).map(|i| (vec![(10 * i, 10 * i + 10)], 0, None)).collect();
    (keys_of(n as usize, ms), chunks_of(&rows))
}

#[test]
fn a_split_mention_does_not_widen_every_scan() {
    let n = 10_000;
    let (keys, chunks) = split_mention(n);
    let before = ops();
    let h = chunk_entities(&chunks, &keys, HeaderPolicy::default());
    let steps = ops() - before;
    let last: Vec<(u32, u32)> = h.entries(Ix::new(n - 1)).collect();
    assert_eq!(last[..2], [(0, 1), (n - 1, 1)]);
    println!("split: {steps} steps for {n} chunks");
    assert!(steps <= 16 * u64::from(n), "{steps} steps for {n} chunks");
    let (keys, chunks) = split_mention(300);
    assert_eq!(h_of(&chunks, &keys, 4), oracle::chunk_entities(&chunks, &keys, HeaderPolicy::default()));
}

fn h_of(chunks: &Chunks, keys: &EntityKeys, max_carried: u32) -> crate::entity::ChunkEntities {
    chunk_entities(chunks, keys, HeaderPolicy { max_carried })
}

struct Rng(u64);

impl Rng {
    fn below(&mut self, n: u32) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % u64::from(n.max(1))) as u32
    }
}

/// Ordered, separated fragments inside `a..b`: one, or up to three.
fn fragments(r: &mut Rng, a: u32, b: u32, empty: bool) -> Vec<(u32, u32)> {
    let mut cuts: Vec<u32> = (0..2 * (1 + r.below(3))).map(|_| a + r.below(b - a + 1)).collect();
    cuts.sort_unstable();
    cuts.dedup();
    let frags: Vec<(u32, u32)> = cuts.as_chunks::<2>().0.iter().map(|c| (c[0], c[1])).collect();
    if frags.len() > 1 && frags.windows(2).all(|w| w[0].1 < w[1].0) {
        frags
    } else {
        let s = a + r.below(b - a);
        vec![(s, (s + u32::from(!empty) + r.below(6)).min(b))]
    }
}

/// Rows partitioning `a..b` at random, each sometimes split around a gap.
fn partition(r: &mut Rng, a: u32, b: u32, pieces: u32) -> Vec<Vec<(u32, u32)>> {
    let mut cuts: Vec<u32> = (0..pieces).map(|_| a + 1 + r.below(b - a - 1)).collect();
    cuts.extend([a, b]);
    cuts.sort_unstable();
    cuts.dedup();
    cuts.windows(2)
        .map(|w| match w[1] - w[0] >= 6 && r.below(4) == 0 {
            true => vec![(w[0], w[0] + 2), (w[1] - 2, w[1])],
            false => vec![(w[0], w[1])],
        })
        .collect()
}

fn random_case(r: &mut Rng) -> (EntityKeys, Chunks, u32) {
    let len = 40 + r.below(400);
    let sentence_len = 5 + r.below(40);
    let nk = 1 + r.below(12) as usize;
    let ms = (0..r.below(60))
        .map(|_| {
            let empty = r.below(8) == 0;
            let f = fragments(r, 0, len, empty);
            (r.below(nk as u32), f[0].0 / sentence_len, f)
        })
        .collect();
    let mut rows = Vec::new();
    let pieces = r.below(6);
    for top in partition(r, 0, len, pieces) {
        let (a, b) = (top[0].0, top.last().unwrap().1);
        let parent = rows.len() as u32;
        rows.push((top, r.below(3) as u8, None));
        if b - a < 4 || r.below(3) == 0 {
            continue;
        }
        let pieces = r.below(5);
        for child in partition(r, a, b, pieces) {
            let orphan = r.below(10) == 0;
            rows.push((child, 0, (!orphan).then_some(parent)));
        }
    }
    // Rows in any order: renumber parents to follow.
    let mut order: Vec<usize> = (0..rows.len()).collect();
    for i in (1..order.len()).rev() {
        order.swap(i, r.below(i as u32 + 1) as usize);
    }
    let mut at = vec![0u32; rows.len()];
    for (new, &old) in order.iter().enumerate() {
        at[old] = new as u32;
    }
    let rows: Vec<_> =
        order.iter().map(|&o| (rows[o].0.clone(), rows[o].1, rows[o].2.map(|p| at[p as usize]))).collect();
    (keys_of(nk, ms), chunks_of(&rows), r.below(7))
}

#[test]
fn matches_the_previous_build_on_random_documents() {
    let mut r = Rng(0x9e37_79b9_7f4a_7c15);
    for case in 0..3_000 {
        let (keys, chunks, max_carried) = random_case(&mut r);
        let policy = HeaderPolicy { max_carried };
        assert_eq!(
            chunk_entities(&chunks, &keys, policy),
            oracle::chunk_entities(&chunks, &keys, policy),
            "case {case}"
        );
    }
}

/// `cargo test --release -p turbo-chunker -- --ignored --nocapture timing`
#[test]
#[ignore]
fn timing_crowded_sentence() {
    for k in [2_000, 4_000, 8_000] {
        let (keys, chunks) = crowded_then_empty(k);
        let t = Instant::now();
        let new = chunk_entities(&chunks, &keys, HeaderPolicy::default());
        let new_ms = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let old = oracle::chunk_entities(&chunks, &keys, HeaderPolicy::default());
        let old_ms = t.elapsed().as_secs_f64() * 1e3;
        assert_eq!(new, old);
        println!("crowded k={k}: ee8d4c0e {old_ms:.2} ms, now {new_ms:.2} ms");
    }
    for n in [10_000, 40_000] {
        let (keys, chunks) = split_mention(n);
        let t = Instant::now();
        let new = chunk_entities(&chunks, &keys, HeaderPolicy::default());
        let new_ms = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let old = oracle::chunk_entities(&chunks, &keys, HeaderPolicy::default());
        let old_ms = t.elapsed().as_secs_f64() * 1e3;
        assert_eq!(new, old);
        println!("split n={n}: ee8d4c0e {old_ms:.2} ms, now {new_ms:.2} ms");
    }
}
