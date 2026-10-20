//! Parent expansion ("small-to-big" retrieval): ranked hits
//! on leaf chunks or on chunk representations are promoted
//! to the parent unit that should actually enter the context
//! window, aggregated per parent, with overlapping
//! ancestor/descendant results deduplicated and a total
//! token budget enforced.

use std::collections::{BTreeMap, BTreeSet};

use turbo_analysis::{Ix, Layer, Span};
use turbo_analysis_layers::{Chunks, Level, Levels, Unit};

use crate::hashes::ChunkId;
use crate::multirep::{MultiRepChunkIndex, RepHit};
use crate::route::Hit;

/// How the scores of hits resolving to one parent combine.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum ScoreAggregation {
    /// The best contributing hit's score.
    #[default]
    Max,
    /// The sum of contributing scores.
    Sum,
    /// The mean of contributing scores.
    Mean,
    /// Reciprocal rank fusion over the input ranking:
    /// each hit contributes `1 / (k + rank)` with 1-based
    /// ranks, summed per parent.
    ReciprocalRankFusion { k: u32 },
}

/// The promotion and budget rules of one expansion.
#[derive(Clone, Debug, PartialEq)]
pub struct ExpansionPolicy {
    /// A hit is promoted to a parent only while the parent's
    /// token count fits this budget. `u32::MAX` promotes as
    /// far as `target_level` allows; `0` keeps every hit at
    /// its own unit (a parent holding no tokens at all still
    /// fits).
    pub max_parent_tokens: u32,
    /// The total token budget of the emitted context, when set.
    pub max_total_tokens: Option<u32>,
    pub aggregation: ScoreAggregation,
    /// The level to promote toward (`Section` or `Document`).
    /// `None` promotes as far as the token budget allows, up
    /// to the document. A hit already at or above the target
    /// level stays where it is.
    pub target_level: Option<Level>,
}

impl Default for ExpansionPolicy {
    fn default() -> Self {
        ExpansionPolicy {
            max_parent_tokens: u32::MAX,
            max_total_tokens: None,
            aggregation: ScoreAggregation::Max,
            target_level: Some(Level::Section),
        }
    }
}

/// One expanded result: the unit to put in the context, the
/// aggregated score, and the hits that produced it.
#[derive(Clone, Debug, PartialEq)]
pub struct ExpandedHit {
    pub unit: Ix<Levels>,
    pub level: Level,
    pub span: Span,
    pub score: f64,
    /// The emitted unit's token count.
    pub tokens: u32,
    /// The distinct hit units that resolved here, ascending.
    pub children: Vec<Ix<Levels>>,
    /// The representation entries that resolved here, when
    /// the hits came from a multi-representation index.
    pub reps: Vec<usize>,
    /// Content IDs of the contributing representations, in
    /// `reps` order, when the index carries them.
    pub chunk_ids: Vec<Option<ChunkId>>,
}

/// Promotes ranked hits to their parents under an
/// [`ExpansionPolicy`].
#[derive(Clone, Debug, Default)]
pub struct ParentExpander {
    pub policy: ExpansionPolicy,
}

struct Candidate {
    unit: Ix<Levels>,
    scores: Vec<f64>,
    ranks: Vec<usize>,
    children: BTreeSet<Ix<Levels>>,
    reps: Vec<usize>,
    chunk_ids: Vec<Option<ChunkId>>,
}

impl ParentExpander {
    pub fn new(policy: ExpansionPolicy) -> ParentExpander {
        ParentExpander { policy }
    }

    /// Expands ranked `(unit, score)` hits. `tokens` is the
    /// token count of every `Levels` unit (see
    /// [`unit_tokens`]). Input order is the ranking: earlier
    /// hits rank better, which is what reciprocal rank fusion
    /// reads.
    pub fn expand(
        &self,
        levels: &Levels,
        tokens: &[u32],
        hits: &[(Ix<Levels>, f64)],
    ) -> Result<Vec<ExpandedHit>, String> {
        let inputs: Vec<Input> =
            hits.iter().map(|&(unit, score)| Input { unit, score, rep: None, chunk_id: None }).collect();
        self.run(levels, tokens, &inputs)
    }

    /// Expands ranked representation hits from a
    /// [`MultiRepChunkIndex`], keeping representation
    /// provenance on the results. Every hit must name an
    /// entry of `index` and resolve to the chunk it claims.
    pub fn expand_rep_hits(
        &self,
        index: &MultiRepChunkIndex,
        levels: &Levels,
        tokens: &[u32],
        hits: &[RepHit],
    ) -> Result<Vec<ExpandedHit>, String> {
        let mut inputs = Vec::with_capacity(hits.len());
        for hit in hits {
            let Some((chunk, chunk_id)) = index.resolve(hit.rep) else {
                return Err(format!("representation {} is not in the index", hit.rep));
            };
            if chunk != hit.chunk || chunk_id != hit.chunk_id {
                return Err(format!(
                    "representation {} resolves to chunk {}, not the hit's chunk {}",
                    hit.rep,
                    chunk.get(),
                    hit.chunk.get()
                ));
            }
            inputs.push(Input { unit: hit.chunk, score: hit.score, rep: Some(hit.rep), chunk_id });
        }
        self.run(levels, tokens, &inputs)
    }

    /// Expands [`Hit`]s from the level router.
    pub fn expand_hits(&self, levels: &Levels, tokens: &[u32], hits: &[Hit]) -> Result<Vec<ExpandedHit>, String> {
        let inputs: Vec<Input> =
            hits.iter().map(|h| Input { unit: h.unit, score: h.score, rep: None, chunk_id: None }).collect();
        self.run(levels, tokens, &inputs)
    }

    fn run(&self, levels: &Levels, tokens: &[u32], inputs: &[Input]) -> Result<Vec<ExpandedHit>, String> {
        levels.check()?;
        if tokens.len() != levels.len() {
            return Err(format!("token counts for {} units over {} levels", tokens.len(), levels.len()));
        }
        let mut by_unit: BTreeMap<Ix<Levels>, Candidate> = BTreeMap::new();
        for (at, input) in inputs.iter().enumerate() {
            if input.unit.get() >= levels.len() {
                return Err(format!("hit unit {} is outside {} levels", input.unit.get(), levels.len()));
            }
            if !input.score.is_finite() {
                return Err(format!("hit {} has a non-finite score", at));
            }
            let resolved = self.promote(levels, tokens, input.unit);
            let candidate = by_unit.entry(resolved).or_insert_with(|| Candidate {
                unit: resolved,
                scores: Vec::new(),
                ranks: Vec::new(),
                children: BTreeSet::new(),
                reps: Vec::new(),
                chunk_ids: Vec::new(),
            });
            candidate.scores.push(input.score);
            candidate.ranks.push(at + 1);
            candidate.children.insert(input.unit);
            if let Some(rep) = input.rep {
                candidate.reps.push(rep);
                candidate.chunk_ids.push(input.chunk_id);
            }
        }

        let mut candidates: Vec<ExpandedHit> = by_unit
            .into_values()
            .map(|c| {
                let score = match self.policy.aggregation {
                    ScoreAggregation::Max => c.scores.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                    ScoreAggregation::Sum => c.scores.iter().sum(),
                    ScoreAggregation::Mean => c.scores.iter().sum::<f64>() / c.scores.len() as f64,
                    ScoreAggregation::ReciprocalRankFusion { k } => {
                        c.ranks.iter().map(|&r| 1.0 / (f64::from(k) + r as f64)).sum()
                    }
                };
                ExpandedHit {
                    unit: c.unit,
                    level: levels.level(c.unit),
                    span: levels.spans.get(c.unit),
                    score,
                    tokens: tokens[c.unit.get()],
                    children: c.children.into_iter().collect(),
                    reps: c.reps,
                    chunk_ids: c.chunk_ids,
                }
            })
            .collect();
        for hit in &candidates {
            if !hit.score.is_finite() {
                return Err("aggregation produced a non-finite score".into());
            }
        }
        // Deterministic order: score descending, then the
        // earlier chunk first.
        candidates.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.unit.cmp(&b.unit)));

        // Greedy acceptance in that order: an accepted unit
        // blocks its ancestors and its whole subtree, so a
        // parent and its own child never both reach the
        // context, and the total budget caps what is emitted
        // (a candidate that does not fit is skipped, a smaller
        // later one may still fit).
        let children = children_of(levels);
        let mut blocked = vec![false; levels.len()];
        let mut used = 0u64;
        let mut out = Vec::new();
        for hit in candidates {
            if blocked[hit.unit.get()] {
                continue;
            }
            if let Some(cap) = self.policy.max_total_tokens
                && used + u64::from(hit.tokens) > u64::from(cap)
            {
                continue;
            }
            used += u64::from(hit.tokens);
            block_subtree(&children, hit.unit, &mut blocked);
            let mut ancestor = levels.parent[hit.unit.get()];
            while let Some(a) = ancestor {
                blocked[a.get()] = true;
                ancestor = levels.parent[a.get()];
            }
            blocked[hit.unit.get()] = true;
            out.push(hit);
        }
        Ok(out)
    }

    /// The ancestor a hit on `unit` resolves to: climb while
    /// the next parent fits the token budget and the target
    /// level (when set) is not yet reached.
    fn promote(&self, levels: &Levels, tokens: &[u32], unit: Ix<Levels>) -> Ix<Levels> {
        let mut at = unit;
        loop {
            if let Some(target) = self.policy.target_level
                && levels.level(at) >= target
            {
                return at;
            }
            let Some(parent) = levels.parent[at.get()] else { return at };
            // Never step past the target level: a chunk with no
            // enclosing section has the document as its parent,
            // and promoting it there would overshoot a section
            // target entirely.
            if let Some(target) = self.policy.target_level
                && levels.level(parent) > target
            {
                return at;
            }
            if tokens[parent.get()] > self.policy.max_parent_tokens {
                return at;
            }
            at = parent;
        }
    }
}

struct Input {
    unit: Ix<Levels>,
    score: f64,
    rep: Option<usize>,
    chunk_id: Option<ChunkId>,
}

fn children_of(levels: &Levels) -> Vec<Vec<Ix<Levels>>> {
    let mut children = vec![Vec::new(); levels.len()];
    for (i, parent) in levels.parent.iter().enumerate() {
        if let Some(parent) = parent {
            children[parent.get()].push(Ix::new(i as u32));
        }
    }
    children
}

fn block_subtree(children: &[Vec<Ix<Levels>>], at: Ix<Levels>, blocked: &mut [bool]) {
    let mut stack = vec![at];
    while let Some(u) = stack.pop() {
        blocked[u.get()] = true;
        stack.extend(children[u.get()].iter().copied());
    }
}

/// The token count of every `Levels` unit, derived from the
/// chunk rows: a chunk unit takes its chunk's `model_tokens`,
/// a section or the document the sum over its descendant
/// chunks. Units naming a chunk past the layer are refused.
pub fn unit_tokens(levels: &Levels, chunks: &Chunks) -> Result<Vec<u32>, String> {
    levels.check()?;
    chunks.check()?;
    let mut tokens = vec![0u64; levels.len()];
    for (i, unit) in levels.unit.iter().enumerate() {
        if let Unit::Chunk(c) = unit {
            let count = chunks
                .model_tokens
                .get(c.get())
                .ok_or_else(|| format!("unit {i} names chunk {} of {}", c.get(), chunks.len()))?;
            tokens[i] = u64::from(*count);
        }
    }
    // Parents precede children, so one reverse pass rolls
    // every count up to the root.
    for i in (1..levels.len()).rev() {
        if let Some(parent) = levels.parent[i] {
            tokens[parent.get()] += tokens[i];
        }
    }
    tokens
        .into_iter()
        .map(|n| u32::try_from(n).map_err(|_| "a unit holds more tokens than a u32 counts".to_string()))
        .collect()
}
