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
mod gaps;
mod limits;

pub use boundaries::{Costs, StructureBoundaries};

use std::ops::Range;

use turbo_analysis::{Annotator, Error, Ix, IxRange, Refs, Span, Spans};
use turbo_analysis_layers::{Block, Boundaries, Chunks, Cut, Sentences, Structure, Subwords};

/// The token budget of a chunk, in the embedder's own tokens.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Budget {
    /// Never more than this, unless one sentence alone is longer: it then
    /// becomes a chunk of its own, marked `Cut::Budget`, for a clause
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
    pub tokens: u32,
    /// What its closing boundary and its size cost, in `Boundaries::UNIT`s.
    pub cost: i64,
}

/// What a chunk of `size` tokens pays for its distance from the target:
/// `size_weight` times the squared distance over the squared target,
/// rounded down, so it is an integer like the boundary costs. A target of
/// zero is taken as one.
pub fn penalty(size: u64, b: Budget) -> i64 {
    let t = i128::from(b.target.max(1));
    let d = i128::from(size) - t;
    let p = i128::from(b.size_weight) * d * d / (t * t);
    p.min(i128::from(i64::MAX)) as i64
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

    /// Sentences of these lengths with nothing between them.
    pub fn of(tokens: &[u32]) -> Lengths {
        let mut l = Lengths::default();
        let mut at = 0;
        for &t in tokens {
            l.start.push(at);
            at += t;
            l.end.push(at);
        }
        l
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
/// Costs are integers, so two cuts tie only when they cost exactly the
/// same; of those, the one whose first boundary comes earliest wins, then
/// its second, and so on. The table runs from the end of the run back, so
/// each entry keeps the earliest end of equal least cost.
pub fn segment(cut: &[i32], tokens: impl Fn(usize, usize) -> u32, b: Budget) -> Vec<Segment> {
    let n = cut.len();
    let size = |i: usize, j: usize| u64::from(tokens(i, j)) + u64::from(b.overhead);
    let cost = |i: usize, j: usize| {
        let boundary = if j < n { i64::from(cut[j - 1]) } else { 0 };
        boundary.saturating_add(penalty(size(i, j), b))
    };
    // best[i]: the least cost of cutting units [i, n); next[i]: where the
    // first chunk of that cut ends.
    let mut best = vec![i64::MAX; n + 1];
    let mut next = vec![n; n + 1];
    best[n] = 0;
    for i in (0..n).rev() {
        for j in i + 1..=n {
            if j - i > 1 && size(i, j) > u64::from(b.max) {
                break;
            }
            let c = cost(i, j).saturating_add(best[j]);
            if c < best[i] {
                best[i] = c;
                next[i] = j;
            }
        }
    }
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let j = next[i];
        out.push(Segment { start: i, end: j, tokens: size(i, j) as u32, cost: cost(i, j) });
        i = j;
    }
    out
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
/// children, so the chunks stay in text order.
///
/// The structure also sets the hard limits. A table row, code block or
/// quote that fits the budget is never cut: its sentences are one unit to
/// the optimizer. One that is over the budget is cut between its
/// sentences, and each chunk that starts inside it says so with
/// `Cut::Inside`. A sentence is never cut; one over the budget alone is a
/// chunk of its own, marked `Cut::Budget`.
pub struct Chunker {
    pub budget: Budget,
}

impl Annotator for Chunker {
    const ID: &'static str = "turbo:chunker";
    type Reads = (Sentences, Subwords, Boundaries, Option<Structure>);
    type Writes = (Chunks,);

    fn annotate(&self, (sentences, subwords, bounds, structure): Refs<'_, Self>) -> Result<(Chunks,), Error> {
        let view = sentences.spans.view();
        if subwords.spans.view() != view || structure.is_some_and(|s| s.spans.view() != view) {
            return Err(Error::Annotator { annotator: Self::ID, detail: "its inputs are in different views".into() });
        }
        let lengths = Lengths::new(&sentences.spans, &subwords.spans);
        let gaps = gaps::gaps(&sentences.spans, structure);
        let runs = match structure {
            Some(st) => sections(sentences, st),
            None if sentences.spans.is_empty() => Vec::new(),
            None => std::iter::once(0..sentences.spans.len()).collect(),
        };
        let mut chunks = Chunks {
            spans: Spans::new(view),
            level: Vec::new(),
            parent: Vec::new(),
            sentences: Vec::new(),
            cut: Vec::new(),
            cost: Vec::new(),
            model_tokens: Vec::new(),
        };
        let span = |r: &Range<usize>| {
            let (first, last) =
                (sentences.spans.get(Ix::new(r.start as u32)), sentences.spans.get(Ix::new(r.end as u32 - 1)));
            Span::new(first.start, last.end)
        };
        let b = self.budget;
        let fits =
            |u: &Range<usize>| u64::from(lengths.run(u.start, u.end)) + u64::from(b.overhead) <= u64::from(b.max);
        for run in runs {
            let opening = if run.start == 0 { Cut::DocumentStart } else { Cut::Section };
            let parent = structure.map(|_| {
                let size = lengths.run(run.start, run.end) + b.overhead;
                push(&mut chunks, span(&run), 1, None, run.clone(), opening, 0, size)
            });
            // The optimizer's units are the atoms: sentence `at[k]` up to
            // `at[k + 1]`.
            let at = limits::atoms(&gaps, run.clone(), fits);
            let cut: Vec<i32> = at[1..].iter().map(|&e| bounds.cost[e - 1]).collect();
            let tokens = |i: usize, j: usize| lengths.run(at[i], at[j]);
            for s in segment(&cut, tokens, b) {
                let r = at[s.start]..at[s.end];
                let cut = match r.start {
                    _ if s.tokens > b.max => Cut::Budget,
                    i if i == run.start => opening,
                    i => gaps[i - 1].inside.as_ref().map_or(bounds.reason[i - 1], |u| Cut::Inside(u.kind)),
                };
                push(&mut chunks, span(&r), 0, parent, r, cut, s.cost, s.tokens);
            }
        }
        Ok((chunks,))
    }

    fn config_hash(&self) -> u64 {
        let b = self.budget;
        [u64::from(b.max), u64::from(b.target), u64::from(b.size_weight), u64::from(b.overhead)]
            .iter()
            .fold(0xcbf2_9ce4_8422_2325, |h, &x| (h ^ x).wrapping_mul(0x0100_0000_01b3))
    }
}

#[allow(clippy::too_many_arguments)]
fn push(
    c: &mut Chunks,
    span: Span,
    level: u8,
    parent: Option<Ix<Chunks>>,
    r: Range<usize>,
    cut: Cut,
    cost: i64,
    tokens: u32,
) -> Ix<Chunks> {
    c.level.push(level);
    c.parent.push(parent);
    c.sentences.push(IxRange::new(r.start as u32, r.end as u32));
    c.cut.push(cut);
    c.cost.push(cost);
    c.model_tokens.push(tokens);
    c.spans.push(span)
}

/// Runs of sentences in one top-level section. A sentence belongs to the
/// section its start is in; sentences outside every section (a preamble)
/// make runs of their own. Two binary searches per sentence, no scan.
fn sections(sentences: &Sentences, structure: &Structure) -> Vec<Range<usize>> {
    let tops: Vec<Span> = structure
        .spans
        .iter()
        .filter(|&(i, _)| structure.kind[i.get()] == Block::Section && structure.parent[i.get()].is_none())
        .map(|(_, s)| s)
        .collect();
    let of = |s: Span| {
        let k = tops.partition_point(|t| t.end <= s.start);
        tops.get(k).filter(|t| t.start <= s.start).map(|_| k)
    };
    let label: Vec<Option<usize>> = sentences.spans.iter().map(|(_, s)| of(s)).collect();
    let mut runs = Vec::new();
    let mut start = 0;
    for i in 1..=label.len() {
        if i == label.len() || label[i] != label[start] {
            runs.push(start..i);
            start = i;
        }
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
        let l = Lengths::of(tokens);
        move |i, j| l.run(i, j)
    }

    #[test]
    fn boundaries_are_chosen_together_not_one_at_a_time() {
        // Six sentences of 100 tokens; the topic shifts after the third, so
        // a cut there is cheap and a cut anywhere else is not.
        let tokens = [100; 6];
        let cut = [1000, 1000, 100, 1000, 1000, 1000];
        let ends: Vec<usize> = segment(&cut, of(&tokens), B).iter().map(|s| s.end).collect();
        assert_eq!(ends, [3, 6]);
        // Filling to the budget cuts after the fourth sentence, mid-topic.
        assert_eq!(greedy(&tokens, B), [4, 6]);
    }

    #[test]
    fn a_sentence_over_the_budget_is_a_chunk_of_its_own() {
        let segments = segment(&[1000; 3], of(&[50, 900, 50]), B);
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
        assert!(segment(&[1000; 4], of(&[100; 4]), b).iter().all(|s| s.tokens <= 400));
        assert_eq!(segment(&[1000; 4], of(&[100; 4]), B).len(), 1);
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
        // A huge chunk saturates instead of overflowing.
        let heavy = Budget { size_weight: u32::MAX, target: 1, ..B };
        assert_eq!(penalty(u64::from(u32::MAX) * 2, heavy), i64::MAX);
    }

    #[test]
    fn of_equal_cuts_the_earliest_wins() {
        // Three sentences of 100 and a target of 150: a chunk of 100 then
        // 200, or of 200 then 100, cost exactly the same.
        let b = Budget { max: 200, target: 150, size_weight: 900, overhead: 0 };
        let segments = segment(&[1000; 3], of(&[100; 3]), b);
        assert_eq!(segments[0].cost + segments[1].cost, 1000 + 100 + 100);
        let ends: Vec<usize> = segments.iter().map(|s| s.end).collect();
        assert_eq!(ends, [1, 3]);
        // Five equal sentences into three chunks: the cut that ends every
        // chunk soonest, chunk by chunk.
        let ends: Vec<usize> = segment(&[1000; 5], of(&[100; 5]), b).iter().map(|s| s.end).collect();
        assert_eq!(ends, [1, 3, 5]);
    }

    #[test]
    fn the_same_inputs_give_the_same_cuts_and_costs() {
        let tokens: Vec<u32> = (0..200).map(|i| 37 + (i * 7919 % 113)).collect();
        let cut: Vec<i32> = (0..200).map(|i| (i * 104_729 % 2000) - 300).collect();
        let first = segment(&cut, of(&tokens), B);
        assert_eq!(first, segment(&cut, of(&tokens), B));
        assert!(first.windows(2).all(|w| w[0].end == w[1].start));
        assert!(first.iter().all(|s| s.tokens <= B.max));
    }

    #[test]
    fn an_empty_run_has_no_chunks() {
        assert!(segment(&[], of(&[]), B).is_empty());
    }
}
