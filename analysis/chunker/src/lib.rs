//! The chunker's optimizer. Every boundary of a section is chosen together,
//! the way TeX breaks a paragraph into lines, instead of filling each chunk
//! to the budget and cutting wherever it lands. What a boundary costs comes
//! from the signal layers (structure, topic shifts, coreference, discourse
//! markers) summed into `Boundaries`; the optimizer only adds up costs.

use std::ops::Range;

use turbo_analysis::{Annotator, Document, Error, Ix, IxRange, Refs, Span, Spans};
use turbo_analysis_layers::{Block, Boundaries, Chunks, Cut, Sentences, Structure, Subwords};

/// The token budget of a chunk, in the embedder's own tokens.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Budget {
    /// Never more than this, unless one sentence alone is longer: it then
    /// becomes a chunk of its own, marked `Cut::Budget`, for a clause
    /// splitter to cut further.
    pub max: u32,
    /// The size retrieval works best at; a chunk pays for its distance
    /// from it.
    pub target: u32,
    /// How much that distance weighs against boundary costs.
    pub size_weight: f32,
    /// Tokens every chunk carries besides its text: the model's special
    /// tokens and the prompt prefix.
    pub overhead: u32,
}

/// Sentences `[start, end)` as one chunk.
#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    pub start: usize,
    pub end: usize,
    /// Its length in the embedder's tokens, overhead included.
    pub tokens: u32,
    /// What its closing boundary and its size cost.
    pub cost: f32,
}

/// The least-cost way to cut a run of sentences into chunks. `tokens[i]` is
/// sentence i's length; `cut[i]` is the cost of a boundary after sentence
/// i (the last one is unused: the end of the run is free). Time is the
/// number of sentences times the most that fit in one chunk.
pub fn segment(tokens: &[u32], cut: &[f32], b: Budget) -> Vec<Segment> {
    assert_eq!(tokens.len(), cut.len(), "a boundary cost per sentence");
    let n = tokens.len();
    let mut prefix = vec![0u64; n + 1];
    for (i, &t) in tokens.iter().enumerate() {
        prefix[i + 1] = prefix[i] + u64::from(t);
    }
    let size = |i: usize, j: usize| prefix[j] - prefix[i] + u64::from(b.overhead);
    let cost = |i: usize, j: usize| {
        let boundary = if j < n { cut[j - 1] } else { 0.0 };
        let d = (size(i, j) as f32 - b.target as f32) / b.target as f32;
        boundary + b.size_weight * d * d
    };
    // best[j]: the least cost of cutting sentences [0, j); from[j]: where
    // the last chunk of that cut starts.
    let mut best = vec![f32::INFINITY; n + 1];
    let mut from = vec![0; n + 1];
    best[0] = 0.0;
    for j in 1..=n {
        for i in (0..j).rev() {
            if size(i, j) > u64::from(b.max) && j - i > 1 {
                break;
            }
            let c = best[i] + cost(i, j);
            if c < best[j] {
                best[j] = c;
                from[j] = i;
            }
        }
    }
    let mut out = Vec::new();
    let mut j = n;
    while j > 0 {
        let i = from[j];
        out.push(Segment { start: i, end: j, tokens: size(i, j) as u32, cost: cost(i, j) });
        j = i;
    }
    out.reverse();
    out
}

/// Chunks a document's sentences. Reads the sentences, the embedder's
/// subwords (so every length is exact) and the boundary costs, and the
/// document's structure when it has one; writes the chunks, each saying
/// why it starts where it does.
///
/// With structure, no chunk crosses a top-level section: each section is
/// its own run for the optimizer (so runs are independent, and an edit
/// re-chunks only its own section), and each becomes a chunk one level up,
/// the parent of the chunks cut from it. Parents come before their
/// children, so the chunks stay in text order.
pub struct Chunker {
    pub budget: Budget,
}

impl Annotator for Chunker {
    const ID: &'static str = "turbo:chunker";
    type Reads = (Sentences, Subwords, Boundaries, Option<Structure>);
    type Writes = (Chunks,);

    fn annotate(
        &self,
        _: &Document,
        (sentences, subwords, bounds, structure): Refs<'_, Self>,
    ) -> Result<(Chunks,), Error> {
        let view = sentences.spans.view();
        if subwords.spans.view() != view || structure.is_some_and(|s| s.spans.view() != view) {
            return Err(Error::Annotator { annotator: Self::ID, detail: "its inputs are in different views".into() });
        }
        let tokens: Vec<u32> = sentences.spans.iter().map(|(_, s)| subwords.spans.within(s).len() as u32).collect();
        let runs = match structure {
            Some(st) => sections(sentences, st),
            None if tokens.is_empty() => Vec::new(),
            None => std::iter::once(0..tokens.len()).collect(),
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
        for run in runs {
            let opening = if run.start == 0 { Cut::DocumentStart } else { Cut::Section };
            let parent = structure.map(|_| {
                let size = tokens[run.clone()].iter().sum::<u32>() + self.budget.overhead;
                push(&mut chunks, span(&run), 1, None, run.clone(), opening, 0.0, size)
            });
            for s in segment(&tokens[run.clone()], &bounds.cost[run.clone()], self.budget) {
                let r = run.start + s.start..run.start + s.end;
                let cut = match r.start {
                    _ if s.tokens > self.budget.max => Cut::Budget,
                    i if i == run.start => opening,
                    i => bounds.reason[i - 1],
                };
                push(&mut chunks, span(&r), 0, parent, r, cut, s.cost, s.tokens);
            }
        }
        Ok((chunks,))
    }

    fn config_hash(&self) -> u64 {
        let b = self.budget;
        [u64::from(b.max), u64::from(b.target), u64::from(b.size_weight.to_bits()), u64::from(b.overhead)]
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
    cost: f32,
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

    const B: Budget = Budget { max: 400, target: 300, size_weight: 1.0, overhead: 0 };

    #[test]
    fn boundaries_are_chosen_together_not_one_at_a_time() {
        // Six sentences of 100 tokens; the topic shifts after the third, so
        // a cut there is cheap and a cut anywhere else is not.
        let tokens = [100; 6];
        let cut = [1.0, 1.0, 0.1, 1.0, 1.0, 1.0];
        let ends: Vec<usize> = segment(&tokens, &cut, B).iter().map(|s| s.end).collect();
        assert_eq!(ends, [3, 6]);
        // Filling to the budget cuts after the fourth sentence, mid-topic.
        assert_eq!(greedy(&tokens, B), [4, 6]);
    }

    #[test]
    fn a_sentence_over_the_budget_is_a_chunk_of_its_own() {
        let segments = segment(&[50, 900, 50], &[1.0, 1.0, 1.0], B);
        let spans: Vec<(usize, usize)> = segments.iter().map(|s| (s.start, s.end)).collect();
        assert_eq!(spans, [(0, 1), (1, 2), (2, 3)]);
        assert_eq!(segments[1].tokens, 900);
    }

    #[test]
    fn the_overhead_counts_against_the_budget() {
        // Four sentences of 100 fit 400 alone, not with 2 special tokens.
        let b = Budget { overhead: 2, ..B };
        assert!(segment(&[100; 4], &[1.0; 4], b).iter().all(|s| s.tokens <= 400));
        assert_eq!(segment(&[100; 4], &[1.0; 4], B).len(), 1);
    }
}
