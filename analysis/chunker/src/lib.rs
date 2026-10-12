//! The chunker's optimizer. Every boundary of a section is chosen together,
//! the way TeX breaks a paragraph into lines, instead of filling each chunk
//! to the budget and cutting wherever it lands. What a boundary costs comes
//! from the signal layers (structure, topic shifts, coreference, discourse
//! markers) weighed and summed into `Boundaries`; the optimizer only adds
//! up costs. Every cost is an integer, so the same layers give the same
//! cuts on every machine.
//!
//! `boundaries` writes the structural costs, `gaps` reads what the
//! structure says about each gap between sentences, and `limits` turns
//! that into the places a chunk may not end.

mod boundaries;
mod cohesion;
mod drift;
mod gaps;
mod limits;
mod providers;
mod sums;
mod weigh;

pub use boundaries::{Costs, StructureBoundaries};
pub use cohesion::{Cohesion, CohesiveChunker};
pub use drift::{Drift, DriftBoundaries, Score, drift};
pub use providers::{DriftProvider, StructureBoundariesProvider, WeighProvider};
pub use sums::{Rows, Sums};
pub use weigh::{DRIFT, STRUCTURE, WeighBoundaries, Weight, weigh};

pub mod terms;

pub mod compare;

use std::ops::Range;

use turbo_analysis::{Annotator, Error, Ix, IxRange, Refs, Span, Spans};
use turbo_analysis_layers::{Block, Boundaries, Chunks, Cut, Sentences, Structure, Subwords};

/// The token budget of a chunk, in the embedder's own tokens.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Budget {
    /// Never more than this, unless one sentence alone is longer: it then
    /// becomes a chunk of its own, marked `Chunks::over_budget`, for a clause
    /// splitter to cut further.
    pub max: u32,
    /// The size retrieval works best at; a chunk pays for its distance
    /// from it.
    pub target: u32,
    /// What a chunk pays, in `Boundaries::UNIT`s, for being a whole target
    /// away from the target: twice the target's size or empty. The penalty
    /// grows with the square of the distance.
    pub size_weight: u32,
    /// Tokens every chunk carries besides its text: the model's special
    /// tokens and the prompt prefix.
    pub overhead: u32,
}

/// Units `[start, end)` as one chunk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Segment {
    pub start: usize,
    pub end: usize,
    /// Its length in the embedder's tokens, overhead included.
    pub tokens: u64,
    /// What its closing boundary, its size and the chunk term cost, in
    /// `Boundaries::UNIT`s, exactly.
    pub cost: i128,
}

/// The most a chunk term, which the caller supplies, can be worth either
/// way: 2^60 units. A term past it counts as it.
pub const TERM_MAX: i64 = 1 << 60;

/// What a chunk of `size` tokens pays for its distance from the target:
/// `size_weight` times the squared distance over the squared target,
/// rounded down, so it is an integer like the boundary costs. It is
/// exact, never capped: a size is at most 2^33 tokens (a u32 count and a
/// u32 overhead) and the weight under 2^32, so the penalty is under 2^98
/// and fits an i128. A target of zero is taken as one.
pub fn penalty(size: u64, b: Budget) -> i128 {
    let t = i128::from(b.target.max(1));
    let d = i128::from(size) - t;
    i128::from(b.size_weight) * d * d / (t * t)
}

/// What one chunk costs the optimizer: the boundary that closes it (none
/// at the end of its run), its size penalty and its chunk term, clamped to
/// `TERM_MAX`. `segment` minimizes the sum of these, and `compare` prices
/// two runs' cuts with it.
pub(crate) fn chunk_cost(closing: i32, size: u64, b: Budget, term: i64) -> i128 {
    i128::from(closing) + penalty(size, b) + i128::from(term.clamp(-TERM_MAX, TERM_MAX))
}

/// How many of the embedder's tokens each run of sentences holds: the
/// subwords that overlap the run's whole span. What lies between two
/// sentences of one chunk is in the chunk too (a paragraph break, which
/// byte-level BPE makes a token of), and a sum over the sentences alone
/// would miss it. A subword a chunk's edge cuts counts in both chunks, as
/// each chunk's own tokenization makes a token of its part. Sentences
/// `[i, j)` hold `end[j - 1] - start[i]` tokens.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Lengths {
    /// The subwords that end by the time sentence i starts.
    pub start: Vec<u32>,
    /// The subwords that start before sentence i ends.
    pub end: Vec<u32>,
}

impl Lengths {
    /// Two binary searches per sentence over the subwords, which are in
    /// text order.
    pub fn new(sentences: &Spans<Sentences>, subwords: &Spans<Subwords>) -> Lengths {
        let (starts, ends) = subwords.columns();
        Lengths {
            start: sentences.iter().map(|(_, s)| ends.partition_point(|&x| x <= s.start) as u32).collect(),
            end: sentences.iter().map(|(_, s)| starts.partition_point(|&x| x < s.end) as u32).collect(),
        }
    }

    /// Sentences of these lengths with nothing between them, or `None`
    /// when they hold more than `u32::MAX` tokens in all.
    pub fn of(tokens: &[u32]) -> Option<Lengths> {
        let mut l = Lengths::default();
        let mut at = 0u32;
        for &t in tokens {
            l.start.push(at);
            at = at.checked_add(t)?;
            l.end.push(at);
        }
        Some(l)
    }

    /// The tokens of sentences `[i, j)`.
    pub fn run(&self, i: usize, j: usize) -> u32 {
        self.end[j - 1].saturating_sub(self.start[i])
    }
}

/// The least-cost way to cut a run of units (sentences, or atoms no chunk
/// splits) into chunks. `tokens(i, j)` is how many tokens units `[i, j)`
/// hold; `cut[i]` is the cost of a boundary after unit i (the last one is
/// unused: the end of the run is free). No chunk of more than one unit is
/// over the budget's maximum. Time is the number of units times the most
/// that fit in one chunk.
///
/// A chunk costs the boundary that closes it, its size penalty and
/// `term(i, j)`, a cost of its own that a signal over whole chunks (how
/// tightly its sentences cluster) adds; `no_term` adds nothing. The term
/// is clamped to `TERM_MAX` either way; the boundary and the penalty are
/// exact.
///
/// Costs are integers, summed in i128, so two cuts tie only when they cost
/// exactly the same. A chunk costs under 2^99, so a total of fewer than
/// 2^28 chunks cannot overflow; past that the total saturates, which takes
/// a quarter of a billion chunks each billions of tokens from the target.
/// Of equal cuts, the one whose first boundary comes earliest wins, then
/// its second, and so on. The table runs from the end of the run back, so
/// each entry keeps the earliest end of equal least cost.
pub fn segment(
    cut: &[i32],
    tokens: impl Fn(usize, usize) -> u32,
    b: Budget,
    term: impl Fn(usize, usize) -> i64,
) -> Vec<Segment> {
    let n = cut.len();
    let size = |i: usize, j: usize| u64::from(tokens(i, j)) + u64::from(b.overhead);
    let cost = |i: usize, j: usize| {
        let boundary = if j < n { cut[j - 1] } else { 0 };
        chunk_cost(boundary, size(i, j), b, term(i, j))
    };
    // best[i]: the least cost of cutting units [i, n); next[i]: where the
    // first chunk of that cut ends. The first end tried, one unit, is
    // always allowed, so every entry has one.
    let mut best = vec![0i128; n + 1];
    let mut next = vec![n; n + 1];
    for i in (0..n).rev() {
        for j in i + 1..=n {
            if j - i > 1 && size(i, j) > u64::from(b.max) {
                break;
            }
            let c = cost(i, j).saturating_add(best[j]);
            if j == i + 1 || c < best[i] {
                best[i] = c;
                next[i] = j;
            }
        }
    }
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let j = next[i];
        out.push(Segment { start: i, end: j, tokens: size(i, j), cost: cost(i, j) });
        i = j;
    }
    out
}

/// The chunk term that adds nothing.
pub fn no_term(_: usize, _: usize) -> i64 {
    0
}

/// Chunks a document's sentences. Reads the sentences, the embedder's
/// subwords (so every length is exact, over each chunk's whole span) and
/// the boundary costs, and the document's structure when it has one;
/// writes the chunks, each saying why it starts where it does.
///
/// With structure, no chunk crosses a top-level section: each section is
/// its own run for the optimizer (so runs are independent, and an edit
/// re-chunks only its own section), and each becomes a chunk one level up,
/// the parent of the chunks cut from it. Parents come before their
/// children, so the chunks stay in text order. A section's edge is always
/// a cut, even where a sentence that straddles it puts a unit across it.
///
/// The structure also sets the hard limits. A table row, code block or
/// quote that fits the budget is never cut: its sentences are one unit to
/// the optimizer. One that is over the budget is cut between its
/// sentences, and each chunk that starts inside it says so with
/// `Cut::Inside`; units that fit but share sentences, chained into a run
/// over the budget, are cut too, with `Cut::Overlap`. A sentence is never
/// cut; one over the budget alone is a chunk of its own, marked in
/// `Chunks::over_budget`.
pub struct Chunker {
    pub budget: Budget,
}

impl Annotator for Chunker {
    const ID: &'static str = "turbo:chunker";
    type Reads = (Sentences, Subwords, Boundaries, Option<Structure>);
    type Writes = (Chunks,);

    fn annotate(&self, (sentences, subwords, bounds, structure): Refs<'_, Self>) -> Result<(Chunks,), Error> {
        Ok((self.chunk(Self::ID, sentences, subwords, bounds, structure, |_| no_term)?,))
    }

    fn config_hash(&self) -> u64 {
        let b = self.budget;
        [u64::from(b.max), u64::from(b.target), u64::from(b.size_weight), u64::from(b.overhead)]
            .iter()
            .fold(0xcbf2_9ce4_8422_2325, |h, &x| (h ^ x).wrapping_mul(0x0100_0000_01b3))
    }
}

impl Chunker {
    /// The chunks. `term(run)` makes the chunk term of each run the
    /// optimizer cuts on its own (a top-level section), so what it needs
    /// is built for one run at a time; it is called with the sentences
    /// `[i, j)` of a chunk. `id` names the step in errors.
    pub(crate) fn chunk<T: Fn(usize, usize) -> i64>(
        &self,
        id: &'static str,
        sentences: &Sentences,
        subwords: &Subwords,
        bounds: &Boundaries,
        structure: Option<&Structure>,
        term: impl Fn(Range<usize>) -> T,
    ) -> Result<Chunks, Error> {
        let view = sentences.spans.view();
        if subwords.spans.view() != view || structure.is_some_and(|s| s.spans.view() != view) {
            return Err(Error::Annotator { annotator: id, detail: "its inputs are in different views".into() });
        }
        let n = sentences.spans.len();
        let lengths = Lengths::new(&sentences.spans, &subwords.spans);
        let b = self.budget;
        let size = |r: Range<usize>| u64::from(lengths.run(r.start, r.end)) + u64::from(b.overhead);
        let gaps = gaps::gaps(&sentences.spans, structure);
        let limits = limits::limits(&gaps.units, n, |u| size(u.clone()) <= u64::from(b.max));
        let runs = match structure {
            Some(st) => sections(sentences, st),
            None if n == 0 => Vec::new(),
            None => std::iter::once(0..n).collect(),
        };
        let mut chunks = Chunks {
            spans: Spans::new(view),
            level: Vec::new(),
            parent: Vec::new(),
            sentences: Vec::new(),
            cut: Vec::new(),
            cost: Vec::new(),
            model_tokens: Vec::new(),
            over_budget: Vec::new(),
        };
        let span = |r: &Range<usize>| {
            let (first, last) =
                (sentences.spans.get(Ix::new(r.start as u32)), sentences.spans.get(Ix::new(r.end as u32 - 1)));
            Span::new(first.start, last.end)
        };
        let tokens = |t: u64| {
            u32::try_from(t).map_err(|_| Error::Annotator {
                annotator: id,
                detail: format!("a chunk of {t} tokens, more than a u32 holds"),
            })
        };
        for run in runs {
            let opening = if run.start == 0 { Cut::DocumentStart } else { Cut::Section };
            let parent = match structure {
                Some(_) => {
                    let t = size(run.clone());
                    let row =
                        Row { span: span(&run), level: 1, parent: None, cut: opening, cost: 0, tokens: tokens(t)? };
                    Some(row.push(&mut chunks, run.clone(), t > u64::from(b.max)))
                }
                None => None,
            };
            // The optimizer's units are the atoms: sentence `at[k].0` up
            // to `at[k + 1].0`.
            let at = limits::atoms(&limits, run.clone(), |r| size(r) > u64::from(b.max));
            let term = term(run.clone());
            let cut: Vec<i32> = at[1..].iter().map(|&(e, _)| bounds.cost[e - 1]).collect();
            for s in segment(&cut, |i, j| lengths.run(at[i].0, at[j].0), b, |i, j| term(at[i].0, at[j].0)) {
                let r = at[s.start].0..at[s.end].0;
                let cut = match at[s.start] {
                    (i, _) if i == run.start => opening,
                    (_, Some(forced)) => forced,
                    (i, None) => bounds.reason[i - 1],
                };
                let row =
                    Row { span: span(&r), level: 0, parent, cut, cost: saturate(s.cost), tokens: tokens(s.tokens)? };
                row.push(&mut chunks, r, s.tokens > u64::from(b.max));
            }
        }
        Ok(chunks)
    }
}

/// One row of the chunks layer, before it is pushed.
struct Row {
    span: Span,
    level: u8,
    parent: Option<Ix<Chunks>>,
    cut: Cut,
    cost: i64,
    tokens: u32,
}

impl Row {
    fn push(self, c: &mut Chunks, sentences: Range<usize>, over_budget: bool) -> Ix<Chunks> {
        c.level.push(self.level);
        c.parent.push(self.parent);
        c.sentences.push(IxRange::new(sentences.start as u32, sentences.end as u32));
        c.cut.push(self.cut);
        c.cost.push(self.cost);
        c.model_tokens.push(self.tokens);
        c.over_budget.push(over_budget);
        c.spans.push(self.span)
    }
}

/// A chunk's exact cost as the layer stores it: saturated to the i64
/// range, which only a size penalty over 2^63 units leaves.
fn saturate(cost: i128) -> i64 {
    cost.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64
}

/// Runs of sentences in one top-level section. A section holds the
/// sentences that start in it (`gaps::in_section`, the rule the boundary costs
/// use too); sentences outside every section (a preamble) make runs of
/// their own. The sections are sorted by where they start, so a structure
/// out of text order cannot merge two of them. Two binary searches per
/// section.
fn sections(sentences: &Sentences, structure: &Structure) -> Vec<Range<usize>> {
    let mut tops: Vec<Span> = structure
        .spans
        .iter()
        .filter(|&(i, _)| structure.kind[i.get()] == Block::Section && structure.parent[i.get()].is_none())
        .map(|(_, s)| s)
        .collect();
    tops.sort_unstable();
    let (starts, _) = sentences.spans.columns();
    let mut runs = Vec::new();
    let mut at = 0;
    for &t in &tops {
        let r = gaps::in_section(starts, t);
        let r = r.start.max(at)..r.end;
        if r.is_empty() {
            continue;
        }
        if at < r.start {
            runs.push(at..r.start);
        }
        at = r.end;
        runs.push(r);
    }
    if at < starts.len() {
        runs.push(at..starts.len());
    }
    runs
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_analysis::ViewId;

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
    pub(crate) fn brute(
        cut: &[i32],
        tokens: &[u32],
        b: Budget,
        term: &dyn Fn(usize, usize) -> i64,
    ) -> (i128, Vec<usize>) {
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
}
