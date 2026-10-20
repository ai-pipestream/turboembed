use super::*;
use crate::gaps::sections;
use turbo_analysis::ViewId;
use turbo_analysis_layers::Block;

/// Fills each chunk to the budget, then cuts: what most libraries do.
fn greedy(tokens: &[u32], b: Budget) -> Vec<usize> {
    let (mut ends, mut size) = (Vec::new(), b.overhead);
    for (i, &t) in tokens.iter().enumerate() {
        if size + t > b.max && size > b.overhead {
            ends.push(i);
            size = b.overhead;
        }
        size += t;
    }
    ends.push(tokens.len());
    ends
}

const B: Budget = Budget { max: 400, target: 300, size_weight: 1000, overhead: 0 };

/// Sentences of these lengths with nothing between them.
fn of(tokens: &[u32]) -> impl Fn(usize, usize) -> u32 {
    let l = Lengths::of(tokens).unwrap();
    move |i, j| l.run(i, j)
}

#[test]
fn boundaries_are_chosen_together_not_one_at_a_time() {
    // Six sentences of 100 tokens; the topic shifts after the third, so
    // a cut there is cheap and a cut anywhere else is not.
    let tokens = [100; 6];
    let cut = [1000, 1000, 100, 1000, 1000, 1000];
    let ends: Vec<usize> = segment(&cut, of(&tokens), B, no_term).iter().map(|s| s.end).collect();
    assert_eq!(ends, [3, 6]);
    // Filling to the budget cuts after the fourth sentence, mid-topic.
    assert_eq!(greedy(&tokens, B), [4, 6]);
}

#[test]
fn a_sentence_over_the_budget_is_a_chunk_of_its_own() {
    let segments = segment(&[1000; 3], of(&[50, 900, 50]), B, no_term);
    let spans: Vec<(usize, usize)> = segments.iter().map(|s| (s.start, s.end)).collect();
    assert_eq!(spans, [(0, 1), (1, 2), (2, 3)]);
    assert_eq!(segments[1].tokens, 900);
}

#[test]
fn a_subword_a_chunk_edge_cuts_counts_on_both_sides() {
    // "See below.) Then." with a sentence ending inside the piece ".)":
    // each chunk's own tokenization makes a token of its part.
    let (mut sentences, mut subwords) = (Spans::new(ViewId::SOURCE), Spans::new(ViewId::SOURCE));
    for (a, b) in [(0, 10), (10, 17)] {
        sentences.push(Span::new(a, b));
    }
    for (a, b) in [(0, 3), (4, 9), (9, 11), (12, 16), (16, 17)] {
        subwords.push(Span::new(a, b));
    }
    let l = Lengths::new(&sentences, &subwords);
    assert_eq!((l.run(0, 1), l.run(1, 2), l.run(0, 2)), (3, 3, 5));
}

#[test]
fn the_overhead_counts_against_the_budget() {
    // Four sentences of 100 fit 400 alone, not with 2 special tokens.
    let b = Budget { overhead: 2, ..B };
    assert!(segment(&[1000; 4], of(&[100; 4]), b, no_term).iter().all(|s| s.tokens <= 400));
    assert_eq!(segment(&[1000; 4], of(&[100; 4]), B, no_term).len(), 1);
}

#[test]
fn the_size_penalty_is_an_integer_rounded_down() {
    // A whole target away costs the weight; a third of one, a ninth.
    assert_eq!(penalty(300, B), 0);
    assert_eq!(penalty(600, B), 1000);
    assert_eq!(penalty(0, B), 1000);
    assert_eq!(penalty(400, B), 111);
    assert_eq!(penalty(200, B), 111);
    // One token off is under a unit, and rounds to nothing.
    assert_eq!(penalty(301, B), 0);
    // A target of zero is taken as one, not divided by.
    assert_eq!(penalty(3, Budget { target: 0, ..B }), 4000);
    // The largest penalty is exact, with no cap and no overflow.
    let heavy = Budget { size_weight: u32::MAX, target: 1, ..B };
    let d = i128::from(u32::MAX) * 2 - 1;
    assert_eq!(penalty(u64::from(u32::MAX) * 2, heavy), i128::from(u32::MAX) * d * d);
}

#[test]
fn huge_penalties_still_tell_cuts_apart() {
    // One chunk of 80,000 tokens costs about 2^64.6 units, four of
    // 20,000 about 2^62.6: under a cap both would be equal and the
    // earliest end, one chunk, would win.
    let b = Budget { max: 200_000, target: 1, size_weight: u32::MAX, overhead: 0 };
    let segments = segment(&[0; 4], of(&[20_000; 4]), b, no_term);
    let ends: Vec<usize> = segments.iter().map(|s| s.end).collect();
    assert_eq!(ends, [1, 2, 3, 4]);
}

#[test]
fn lengths_past_u32_are_refused_not_wrapped() {
    let l = Lengths::of(&[u32::MAX - 10, 10]).unwrap();
    assert_eq!((l.run(0, 2), l.end[1]), (u32::MAX, u32::MAX));
    assert_eq!(Lengths::of(&[u32::MAX - 10, 11]), None);
    assert_eq!(Lengths::of(&[u32::MAX, 0, 1]), None);
}

#[test]
fn of_equal_cuts_the_earliest_wins() {
    // Three sentences of 100 and a target of 150: a chunk of 100 then
    // 200, or of 200 then 100, cost exactly the same.
    let b = Budget { max: 200, target: 150, size_weight: 900, overhead: 0 };
    let segments = segment(&[1000; 3], of(&[100; 3]), b, no_term);
    assert_eq!(segments[0].cost + segments[1].cost, 1000 + 100 + 100);
    let ends: Vec<usize> = segments.iter().map(|s| s.end).collect();
    assert_eq!(ends, [1, 3]);
    // Five equal sentences into three chunks: the cut that ends every
    // chunk soonest, chunk by chunk.
    let ends: Vec<usize> = segment(&[1000; 5], of(&[100; 5]), b, no_term).iter().map(|s| s.end).collect();
    assert_eq!(ends, [1, 3, 5]);
}

#[test]
fn the_same_inputs_give_the_same_cuts_and_costs() {
    let tokens: Vec<u32> = (0..200).map(|i| 37 + (i * 7919 % 113)).collect();
    let cut: Vec<i32> = (0..200).map(|i| (i * 104_729 % 2000) - 300).collect();
    let first = segment(&cut, of(&tokens), B, no_term);
    assert_eq!(first, segment(&cut, of(&tokens), B, no_term));
    assert!(first.windows(2).all(|w| w[0].end == w[1].start));
    assert!(first.iter().all(|s| s.tokens <= u64::from(B.max)));
}

#[test]
fn an_empty_run_has_no_chunks() {
    assert!(segment(&[], of(&[]), B, no_term).is_empty());
}

#[test]
fn a_chunk_term_is_added_to_each_chunk() {
    // Four sentences of 100; a term that favours sentences 1 and 2 as
    // one chunk wins over the size target.
    let favour = |i: usize, j: usize| if (i, j) == (1, 3) { -10_000 } else { 0 };
    let ends: Vec<usize> = segment(&[1000; 4], of(&[100; 4]), B, favour).iter().map(|s| s.end).collect();
    assert_eq!(ends, [1, 3, 4]);
    // A term past the cap counts as the cap.
    let huge = |_: usize, _: usize| i64::MAX;
    assert!(segment(&[1000; 4], of(&[100; 4]), B, huge).iter().all(|s| s.cost <= i128::from(TERM_MAX) + 1000));
}

#[test]
fn a_huge_penalty_never_lets_a_chunk_past_the_maximum() {
    // Ten sentences of 10, one of four million, two of 10, with a
    // weight and a target that make every penalty enormous: the costs
    // never saturate, so the long sentence still stands alone and no
    // chunk of more than one sentence is over the maximum.
    let b = Budget { max: 100, target: 0, size_weight: 1_000_000, overhead: 0 };
    let mut tokens = vec![10; 10];
    tokens.extend([4_000_000, 10, 10]);
    let segments = segment(&[0; 13], of(&tokens), b, no_term);
    assert!(segments.iter().all(|s| s.end - s.start == 1 || s.tokens <= 100));
    assert!(segments.iter().any(|s| (s.start, s.end, s.tokens) == (10, 11, 4_000_000)));
}

#[test]
fn negative_costs_alone_cut_wherever_the_size_allows() {
    // Every boundary argues for a cut and every sentence is the target.
    let segments = segment(&[-1000; 6], of(&[5; 6]), Budget { max: 20, target: 5, ..B }, no_term);
    assert_eq!(segments.len(), 6);
    assert!(segments.iter().all(|s| s.cost <= 0));
}

/// Every way to cut the units, tried one by one: the least total cost
/// and, of equal totals, the cut whose boundaries come earliest.
pub(crate) fn brute(cut: &[i32], tokens: &[u32], b: Budget, term: &dyn Fn(usize, usize) -> i64) -> (i128, Vec<usize>) {
    let n = tokens.len();
    let l = Lengths::of(tokens).unwrap();
    let mut best: Option<(i128, Vec<usize>)> = None;
    'masks: for mask in 0u32..1 << (n - 1) {
        let mut ends: Vec<usize> = (0..n - 1).filter(|g| mask >> g & 1 == 1).map(|g| g + 1).collect();
        ends.push(n);
        let (mut total, mut i) = (0i128, 0);
        for &j in &ends {
            let size = u64::from(l.run(i, j)) + u64::from(b.overhead);
            if j - i > 1 && size > u64::from(b.max) {
                continue 'masks;
            }
            let boundary = if j < n { cut[j - 1] } else { 0 };
            total += i128::from(boundary) + penalty(size, b) + i128::from(term(i, j).clamp(-TERM_MAX, TERM_MAX));
            i = j;
        }
        if best.as_ref().is_none_or(|(t, e)| (total, &ends) < (*t, e)) {
            best = Some((total, ends));
        }
    }
    best.unwrap()
}

#[test]
fn the_optimizer_agrees_with_trying_every_cut() {
    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    let mut rand = |m: u64| {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) % m
    };
    let weights = [0, 1, 1000, 1_000_000, u32::MAX];
    let banded = |i: usize, j: usize| ((i * 7 + j * 13) % 5) as i64 * 400 - 800;
    for case in 0..3000 {
        let n = 1 + rand(11) as usize;
        let tokens: Vec<u32> = (0..n).map(|_| rand(60) as u32).collect();
        let cut: Vec<i32> = (0..n).map(|_| if rand(3) == 0 { 0 } else { rand(6001) as i32 - 3000 }).collect();
        let b = Budget {
            max: 1 + rand(150) as u32,
            target: if rand(4) == 0 { 0 } else { rand(120) as u32 },
            size_weight: weights[rand(5) as usize],
            overhead: rand(5) as u32,
        };
        let term: &dyn Fn(usize, usize) -> i64 = if case % 2 == 0 { &no_term } else { &banded };
        let segments = segment(&cut, of(&tokens), b, term);
        let total: i128 = segments.iter().map(|s| s.cost).sum();
        let ends: Vec<usize> = segments.iter().map(|s| s.end).collect();
        assert_eq!((total, ends), brute(&cut, &tokens, b, term), "case {case}: {tokens:?} {cut:?} {b:?}");
    }
}

#[test]
fn sections_out_of_text_order_stay_apart() {
    // Three sections listed last first, as an unchecked structure
    // could hold them; each is still a run of its own.
    let mut spans = Spans::new(ViewId::SOURCE);
    for (a, b) in [(0, 9), (10, 19), (20, 29)] {
        spans.push(Span::new(a, b));
    }
    let sentences = Sentences { spans };
    let mut blocks = Spans::new(ViewId::SOURCE);
    for (a, b) in [(20, 29), (10, 19), (0, 9)] {
        blocks.push(Span::new(a, b));
    }
    let st = Structure { spans: blocks, kind: vec![Block::Section; 3], parent: vec![None; 3], depth: vec![1; 3] };
    assert_eq!(sections(&sentences, &st), [0..1, 1..2, 2..3]);
}
