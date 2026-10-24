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
//! that into the places a chunk may not end. At query time, `route`
//! ranks the chunk, section and document units of `Levels` together,
//! and `hashes` gives every unit a content-addressed ID. `eval` scores a
//! chunking against a gold one and sweeps the chunker's settings.

mod boundaries;
pub mod cascade;
pub mod clauses;
mod cohesion;
pub mod coref;
mod discourse;
mod drift;
pub mod entity;
pub mod expand;
mod gaps;
mod hashes;
pub mod late;
mod lexical;
mod limits;
pub mod multirep;
mod optimizer;
mod providers;
mod run;
mod text_signals;
mod trace;
pub use optimizer::{DecisionMargin, SegmentTrace, segment_with_trace};
pub use trace::ChunkTrace;
pub mod rechunk;
pub mod route;
pub mod scorer;
mod sentiment;
mod sums;
mod weigh;

pub use boundaries::{Costs, StructureBoundaries};
pub use cascade::{
    Cascade, CascadeConfig, CascadeHit, CascadeResult, FilterMode, LexicalSpace, QueryVector, Stage, UnitFilter,
    VectorSpace, recall_at_k,
};
pub use clauses::{ClauseBoundaries, ClauseBreak, ClauseCosts, ClauseCuts, SentenceClauseCuts, clause_cuts};
pub use cohesion::{Cohesion, CohesiveChunker};
pub use coref::{
    Agreement, Animacy, AntecedentLink, ChunkAntecedentBuilder, ChunkAntecedents, CorefBoundaries, CorefChains,
    CorefCohesion, CorefLink, Gender, MentionKind, MentionOrigin, Mentions, Number, Person, ResolvedMention,
    chunk_antecedents, coref_boundaries, coref_boundaries_from_chains, extract_chains,
};
pub use discourse::{DISCOURSE, DiscourseBoundaries, DiscourseLanguage};
pub use drift::{Drift, DriftBoundaries, Score, drift};
pub use entity::{
    ChunkEntities, ChunkEntityHeaders, EntityBoundaries, EntityContinuity, EntityKeys, HeaderPolicy, Keying,
    chunk_entities, entity_boundaries,
};
pub use expand::{ExpandedHit, ExpansionPolicy, ParentExpander, ScoreAggregation, unit_tokens};
pub use hashes::{ChunkId, ContentHashBuilder, ContentHashes, LevelHashBuilder};
pub use late::{LateChunkAnnotator, Pooling, WindowPolicy};
pub use lexical::{LEXICAL, LexicalBoundaries, LexicalForm};
pub use multirep::{ChunkRepresentation, MultiRepChunkIndex, RepHit, RepresentationKind};
pub use providers::{
    ClauseCutProvider, CorefBoundariesProvider, DistilledBoundariesProvider, DriftProvider, EntityBoundariesProvider,
    SentimentBoundariesProvider, StructureBoundariesProvider, WeighProvider,
};
pub use route::{Hit, LevelRouter, LevelWeights, ParentContext, QueryKind, QueryProfile, Routed};
pub use scorer::{
    BoundaryFeatures, DistilledBoundaryScorer, DistilledWeights, distilled_boundaries, distilled_with_runs,
};
pub use sentiment::{
    ChunkFacetBuilder, ChunkFacets, SentimentBoundaries, SentimentShift, boundary_similarity, chunk_boundary_gaps,
    sentiment_shift,
};
pub use sums::{Rows, Sums};
pub use text_signals::WeighTextBoundaries;
pub use turbo_analysis_layers::TokenVectorWindows;
pub use weigh::{
    COREF, DRIFT, ENTITY, SENTIMENT, STRUCTURE, WeighAllBoundaries, WeighBoundaries, Weight, weigh, weigh_with_coref,
    weigh_with_sentiment,
};

pub mod export;
pub mod levels;
pub mod terms;

pub mod compare;
pub mod eval;

use std::ops::Range;

use turbo_analysis::{Annotator, Error, Fnv1a, Ix, IxRange, Refs, Span, Spans};
use turbo_analysis_layers::{Boundaries, Chunks, Cut, Sentences, Structure, Subwords};

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
    optimizer::solve::<false>(cut, tokens, b, term).segments
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
/// over the budget, are cut too, with `Cut::Overlap`. When `ClauseCuts` is
/// present, a single sentence that exceeds `budget.max` is split at syntactic
/// clause boundaries (`Cut::Clause`) or token windows; otherwise it becomes a
/// chunk of its own, marked in `Chunks::over_budget`.
pub struct Chunker {
    pub budget: Budget,
}

impl Annotator for Chunker {
    const ID: &'static str = "turbo:chunker";
    type Reads = (Sentences, Subwords, Boundaries, Option<Structure>, Option<ClauseCuts>);
    type Writes = (Chunks,);

    fn annotate(&self, (sentences, subwords, bounds, structure, clauses): Refs<'_, Self>) -> Result<(Chunks,), Error> {
        Ok((self.chunk(Self::ID, sentences, subwords, bounds, structure, clauses, |_| no_term)?,))
    }

    fn config_hash(&self) -> u64 {
        let b = self.budget;
        [u64::from(b.max), u64::from(b.target), u64::from(b.size_weight), u64::from(b.overhead)]
            .iter()
            .fold(Fnv1a::new(), |mut h, &x| {
                h.u64(x);
                h
            })
            .finish()
    }
}

impl Chunker {
    /// The chunks. `term(run)` makes the chunk term of each run the
    /// optimizer cuts on its own (a top-level section), so what it needs
    /// is built for one run at a time; it is called with the sentences
    /// `[i, j)` of a chunk. `id` names the step in errors.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn chunk<T: Fn(usize, usize) -> i64>(
        &self,
        id: &'static str,
        sentences: &Sentences,
        subwords: &Subwords,
        bounds: &Boundaries,
        structure: Option<&Structure>,
        clauses: Option<&ClauseCuts>,
        term: impl Fn(Range<usize>) -> T,
    ) -> Result<Chunks, Error> {
        self.chunk_record(id, sentences, subwords, bounds, structure, clauses, term, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn chunk_record<T: Fn(usize, usize) -> i64>(
        &self,
        id: &'static str,
        sentences: &Sentences,
        subwords: &Subwords,
        bounds: &Boundaries,
        structure: Option<&Structure>,
        clauses: Option<&ClauseCuts>,
        term: impl Fn(Range<usize>) -> T,
        mut trace: Option<&mut Vec<Option<DecisionMargin>>>,
    ) -> Result<Chunks, Error> {
        let view = sentences.spans.view();
        if subwords.spans.view() != view || structure.is_some_and(|s| s.spans.view() != view) {
            return Err(Error::Annotator { annotator: id, detail: "its inputs are in different views".into() });
        }
        let n = sentences.spans.len();
        if bounds.cost.len() != n || bounds.reason.len() != n {
            return Err(Error::Annotator {
                annotator: id,
                detail: format!(
                    "{} boundary costs and {} reasons for {n} sentences",
                    bounds.cost.len(),
                    bounds.reason.len()
                ),
            });
        }
        if let Some(cl) = clauses
            && cl.sentences.len() != n
        {
            return Err(Error::Annotator {
                annotator: id,
                detail: format!("{} clause-cut entries for {n} sentences", cl.sentences.len()),
            });
        }
        let lengths = Lengths::new(&sentences.spans, &subwords.spans);
        let b = self.budget;
        let size = |r: Range<usize>| u64::from(lengths.run(r.start, r.end)) + u64::from(b.overhead);
        let gaps = gaps::gaps(&sentences.spans, structure);
        let limits = limits::limits(&gaps.units, n, |u| size(u.clone()) <= u64::from(b.max));
        let runs = match structure {
            Some(st) => gaps::sections(sentences, st),
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
            prompt_overhead: b.overhead,
        };
        let cx = RunCx { sentences, subwords, lengths: &lengths, bounds, limits: &limits, clauses };
        for run in runs {
            let term = term(run.clone());
            self.push_run_with_trace(id, &cx, run, structure.is_some(), term, &mut chunks, trace.as_deref_mut())?;
        }
        Ok(chunks)
    }
}

/// What cutting one run reads, besides the budget: the sentences,
/// their token lengths, the boundary costs and the hard limits at
/// every gap.
pub(crate) struct RunCx<'a> {
    pub sentences: &'a Sentences,
    pub subwords: &'a Subwords,
    pub lengths: &'a Lengths,
    pub bounds: &'a Boundaries,
    pub limits: &'a [limits::Limit],
    pub clauses: Option<&'a ClauseCuts>,
}

/// A run's atoms under the hard limits, as `push_run` cuts them.
pub(crate) fn run_atoms(
    limits: &[limits::Limit],
    run: Range<usize>,
    size: &impl Fn(Range<usize>) -> u64,
    b: Budget,
) -> Vec<(usize, Option<Cut>)> {
    limits::atoms(limits, run, |r| size(r) > u64::from(b.max))
}

/// One row of the chunks layer, before it is pushed.
pub(crate) struct Row {
    pub span: Span,
    pub level: u8,
    pub parent: Option<Ix<Chunks>>,
    pub cut: Cut,
    pub cost: i64,
    pub tokens: u32,
}

impl Row {
    pub(crate) fn push(self, c: &mut Chunks, sentences: Range<usize>, over_budget: bool) -> Ix<Chunks> {
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
pub(crate) fn saturate(cost: i128) -> i64 {
    cost.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64
}

#[cfg(test)]
mod tests;
