//! A matrix of chunking strategies run over one document: what each
//! strategy cut, how fully it used its token budget, and where it
//! disagrees with a reference strategy.
//!
//! Two baselines pack whole sentences greedily (never splitting one),
//! against `budget.target` for the fixed token window and against
//! `budget.max` for the greedy packer. Two strategies run the DP
//! optimizer through the regular [`Chunker`] machinery, hard limits
//! and section parents included: structure-only over freshly computed
//! [`StructureBoundaries`] with default [`Costs`], and semantic over
//! the document's own default `Boundaries` (the weighed sum a pipeline
//! writes), which the caller must have produced first. Baselines price
//! every cut they close as a plain sentence boundary
//! (`Boundaries::UNIT`), so their recorded chunk costs stay comparable
//! with the optimizer's.
//!
//! Summary statistics read level-0 chunks only (section parents are
//! not retrieval units): token counts are `model_tokens` less the
//! recorded prompt overhead, `p95_tokens` is the nearest-rank 95th
//! percentile, and `budget_utilization` is the mean over `budget.max`.
//! Pairwise disagreement reports are not stored (a [`compare::Report`]
//! borrows the runs it compares); [`StrategyMatrix::report`] computes
//! one on demand against the reference strategy, pricing each side
//! with the boundaries that side was cut with.

use turbo_analysis::{Document, Error, Ix, IxRange, Key, Layer, Span, Spans};
use turbo_analysis_layers::{Boundaries, Chunks, Cut, Sentences, Structure, Subwords};

use crate::boundaries::Costs;
use crate::clauses::ClauseCuts;
use crate::compare::{self, Pricing, Run};
use crate::gaps::gaps;
use crate::{Budget, Chunker, Lengths, chunk_cost, no_term};

/// One chunking strategy of the matrix.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ChunkStrategy {
    /// Whole sentences packed until the next would pass `budget.target`.
    FixedTokenWindow { budget: Budget },
    /// Whole sentences packed until the next would pass `budget.max`.
    GreedySentence { budget: Budget },
    /// The DP optimizer over structure boundaries alone.
    StructureOnly { budget: Budget },
    /// The DP optimizer over the document's default weighed
    /// `Boundaries`.
    SemanticDp { budget: Budget },
}

impl ChunkStrategy {
    pub fn budget(self) -> Budget {
        match self {
            ChunkStrategy::FixedTokenWindow { budget }
            | ChunkStrategy::GreedySentence { budget }
            | ChunkStrategy::StructureOnly { budget }
            | ChunkStrategy::SemanticDp { budget } => budget,
        }
    }

    /// The strategy's stable name, the base of its summary `id`.
    pub const fn name(self) -> &'static str {
        match self {
            ChunkStrategy::FixedTokenWindow { .. } => "fixed-token-window",
            ChunkStrategy::GreedySentence { .. } => "greedy-sentence",
            ChunkStrategy::StructureOnly { .. } => "structure-only",
            ChunkStrategy::SemanticDp { .. } => "semantic-dp",
        }
    }
}

/// The cut kinds [`StrategySummary::cut_counts`] is indexed by, in
/// order. `Inside` and `Overlap` count every block kind together.
pub const CUT_KINDS: [&str; 10] = [
    "document-start",
    "sentence",
    "section",
    "heading",
    "paragraph",
    "topic-shift",
    "clause",
    "item",
    "inside",
    "overlap",
];

fn cut_kind(cut: Cut) -> usize {
    match cut {
        Cut::DocumentStart => 0,
        Cut::Sentence => 1,
        Cut::Section => 2,
        Cut::Heading => 3,
        Cut::Paragraph => 4,
        Cut::TopicShift => 5,
        Cut::Clause => 6,
        Cut::Item => 7,
        Cut::Inside(_) => 8,
        Cut::Overlap(_) => 9,
    }
}

/// One strategy's chunks and their budget statistics. `boundaries` is
/// what the strategy was cut with (plain sentence costs for the
/// baselines), kept so [`StrategyMatrix::report`] can price
/// disagreements.
pub struct StrategySummary {
    pub id: String,
    pub chunks: Chunks,
    pub boundaries: Boundaries,
    pub budget: Budget,
    pub chunk_count: u32,
    pub mean_tokens: f32,
    pub min_tokens: u32,
    pub max_tokens: u32,
    pub p95_tokens: u32,
    pub budget_utilization: f32,
    pub over_budget_count: u32,
    pub cut_counts: [u32; 10],
    pub total_cost: i128,
}

impl StrategySummary {
    /// How many level-0 chunks start for this kind of reason.
    pub fn cut_count(&self, cut: Cut) -> u32 {
        self.cut_counts[cut_kind(cut)]
    }
}

/// Every strategy's summary over one document, with the index of the
/// reference strategy the others are compared against.
pub struct StrategyMatrix {
    pub summaries: Vec<StrategySummary>,
    pub reference: usize,
}

const ID: &str = "turbo:strategy-matrix";

fn refused(detail: String) -> Error {
    Error::Annotator { annotator: ID, detail }
}

impl StrategyMatrix {
    /// Runs every strategy over `doc`. The document must carry
    /// `Sentences` and `Subwords`; `Structure` and `ClauseCuts` are
    /// read when present, and `SemanticDp` additionally requires the
    /// default `Boundaries`. Refuses an empty strategy list and a
    /// reference index outside it.
    pub fn run(doc: &Document, strategies: &[ChunkStrategy], reference: usize) -> Result<StrategyMatrix, Error> {
        if strategies.is_empty() {
            return Err(refused("no strategies to run".into()));
        }
        if reference >= strategies.len() {
            return Err(refused(format!("reference strategy {reference} of {}", strategies.len())));
        }
        let sentences = doc.get::<Sentences>().ok_or(Error::Missing(Key::<Sentences>::DEFAULT.id()))?;
        let subwords = doc.get::<Subwords>().ok_or(Error::Missing(Key::<Subwords>::DEFAULT.id()))?;
        let structure = doc.get::<Structure>();
        let clauses = doc.get::<ClauseCuts>();
        let lengths = Lengths::new(&sentences.spans, &subwords.spans);
        let mut summaries = Vec::with_capacity(strategies.len());
        for (i, &strategy) in strategies.iter().enumerate() {
            let budget = strategy.budget();
            let (chunks, boundaries) = match strategy {
                ChunkStrategy::FixedTokenWindow { .. } => {
                    (pack(sentences, &lengths, budget, budget.target), plain_boundaries(sentences))
                }
                ChunkStrategy::GreedySentence { .. } => {
                    (pack(sentences, &lengths, budget, budget.max), plain_boundaries(sentences))
                }
                ChunkStrategy::StructureOnly { .. } => {
                    let b = structure_boundaries(sentences, structure);
                    (Chunker { budget }.chunk(ID, sentences, subwords, &b, structure, clauses, |_| no_term)?, b)
                }
                ChunkStrategy::SemanticDp { .. } => {
                    let b = doc.get::<Boundaries>().ok_or(Error::Missing(Key::<Boundaries>::DEFAULT.id()))?;
                    (
                        Chunker { budget }.chunk(ID, sentences, subwords, b, structure, clauses, |_| no_term)?,
                        Boundaries { cost: b.cost.clone(), reason: b.reason.clone() },
                    )
                }
            };
            chunks.check().map_err(|d| refused(format!("strategy {}: {d}", strategy.name())))?;
            let mut id = strategy.name().to_string();
            if strategies[..i].iter().any(|s| s.name() == strategy.name()) {
                id = format!("{}-{i}", strategy.name());
            }
            summaries.push(summarize(id, chunks, boundaries, budget));
        }
        Ok(StrategyMatrix { summaries, reference })
    }

    /// The disagreement report of strategy `i` against the reference
    /// strategy, priced with each side's own boundaries and budget.
    /// Refuses an index outside the matrix.
    /// `None` when `i` is outside the matrix.
    pub fn report<'a>(&'a self, doc: &'a Document, i: usize) -> Option<Result<compare::Report<'a>, compare::Refused>> {
        let b = self.summaries.get(i)?;
        let sentences = doc.get::<Sentences>().expect("the document the matrix was run on");
        let run = |s: &'a StrategySummary| Run {
            chunks: &s.chunks,
            execution: None,
            pricing: Some(Pricing { sentences, boundaries: &s.boundaries, budget: s.budget }),
        };
        Some(compare::compare(run(&self.summaries[self.reference]), run(b)))
    }
}

fn plain_boundaries(sentences: &Sentences) -> Boundaries {
    let n = sentences.spans.len();
    Boundaries { cost: vec![Boundaries::UNIT; n], reason: vec![Cut::Sentence; n] }
}

fn structure_boundaries(sentences: &Sentences, structure: Option<&Structure>) -> Boundaries {
    let costs = Costs::default();
    let gaps = gaps(&sentences.spans, structure).gap;
    Boundaries {
        cost: gaps
            .iter()
            .map(|g| costs.of(g.reason).saturating_add(if g.after_heading { costs.after_heading } else { 0 }))
            .collect(),
        reason: gaps.iter().map(|g| g.reason).collect(),
    }
}

/// Whole sentences packed left to right: a chunk closes when the next
/// sentence would take it over `limit` tokens. A single sentence over
/// the limit is a chunk of its own.
fn pack(sentences: &Sentences, lengths: &Lengths, budget: Budget, limit: u32) -> Chunks {
    let n = sentences.spans.len();
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && lengths.run(i, j + 1) <= limit {
            j += 1;
        }
        ranges.push((i, j));
        i = j;
    }
    let mut spans = Spans::<Chunks>::new(sentences.spans.view());
    let mut out = Chunks {
        spans: Spans::new(sentences.spans.view()),
        level: Vec::new(),
        parent: Vec::new(),
        sentences: Vec::new(),
        cut: Vec::new(),
        cost: Vec::new(),
        model_tokens: Vec::new(),
        over_budget: Vec::new(),
        prompt_overhead: budget.overhead,
    };
    for (k, &(a, b)) in ranges.iter().enumerate() {
        let tokens = lengths.run(a, b);
        spans.push(Span {
            start: sentences.spans.get(Ix::new(a as u32)).start,
            end: sentences.spans.get(Ix::new(b as u32 - 1)).end,
        });
        out.level.push(0);
        out.parent.push(None);
        out.sentences.push(IxRange::new(a as u32, b as u32));
        out.cut.push(if k == 0 { Cut::DocumentStart } else { Cut::Sentence });
        let closing = if k + 1 == ranges.len() { 0 } else { Boundaries::UNIT };
        out.cost
            .push(chunk_cost(closing, u64::from(tokens), budget, 0).clamp(i128::from(i64::MIN), i128::from(i64::MAX))
                as i64);
        out.model_tokens.push(tokens.saturating_add(budget.overhead));
        out.over_budget.push(tokens > budget.max);
    }
    out.spans = spans;
    out
}

fn summarize(id: String, chunks: Chunks, boundaries: Boundaries, budget: Budget) -> StrategySummary {
    let mut tokens: Vec<u32> = Vec::new();
    let mut cut_counts = [0u32; 10];
    let mut total_cost: i128 = 0;
    let mut over_budget_count = 0u32;
    for i in 0..chunks.spans.len() {
        if chunks.level[i] != 0 {
            continue;
        }
        tokens.push(chunks.model_tokens[i].saturating_sub(chunks.prompt_overhead));
        cut_counts[cut_kind(chunks.cut[i])] += 1;
        total_cost += i128::from(chunks.cost[i]);
        over_budget_count += u32::from(chunks.over_budget[i]);
    }
    let chunk_count = tokens.len() as u32;
    let (mean_tokens, min_tokens, max_tokens, p95_tokens) = if tokens.is_empty() {
        (0.0, 0, 0, 0)
    } else {
        let sum: u64 = tokens.iter().map(|&t| u64::from(t)).sum();
        let mut sorted = tokens.clone();
        sorted.sort_unstable();
        let p95 = sorted[((95 * sorted.len()).div_ceil(100)).max(1) - 1];
        (sum as f32 / tokens.len() as f32, sorted[0], sorted[sorted.len() - 1], p95)
    };
    StrategySummary {
        id,
        chunks,
        boundaries,
        budget,
        chunk_count,
        mean_tokens,
        min_tokens,
        max_tokens,
        p95_tokens,
        budget_utilization: if budget.max == 0 { 0.0 } else { mean_tokens / budget.max as f32 },
        over_budget_count,
        cut_counts,
        total_cost,
    }
}
