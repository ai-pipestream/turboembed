//! Budget compliance and cohesion preservation.
//!
//! Budget: how the leaf chunks sit in the token budget. A chunk's tokens
//! are `Chunks::model_tokens`, overhead included, the same count the
//! optimizer holds to `Budget::max`; its utilization is its tokens over
//! the target (a target of zero is taken as one). The 95th percentile is
//! the nearest rank, `ceil(0.95 n)`. A chunk is empty when it holds only
//! the overhead, and undersized when it is not empty and holds under a
//! quarter of the target.
//!
//! Cohesion: the spans no cut should part, as sentence ranges: a code
//! block, table or quote of the structure, a coreference chain from its
//! first mention's sentence to its last's, a pair of clauses held
//! together. One is intact when no leaf chunk starts strictly inside it.

use std::ops::Range;

use turbo_analysis::IxRange;
use turbo_analysis_layers::{Block, Boundaries, Chunks, Cut, Sentences, Structure};

use super::Segmentation;
use crate::Budget;

/// How the leaf chunks sit in the budget. Shares are of all chunks, and 0
/// when there are none.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BudgetReport {
    pub chunks: usize,
    /// Chunks of at most `Budget::max` tokens.
    pub within_max: f64,
    pub mean_utilization: f64,
    pub p95_utilization: f64,
    pub max_utilization: f64,
    /// Chunks the optimizer marked `Chunks::over_budget`: one sentence
    /// alone over the maximum.
    pub over_budget: f64,
    pub empty: f64,
    pub undersized: f64,
}

/// The leaf rows of a chunks layer as `(tokens, over_budget)`, the input
/// `BudgetReport::of` pools across documents.
pub fn leaves(chunks: &Chunks) -> impl Iterator<Item = (u32, bool)> + '_ {
    (0..chunks.level.len()).filter(|&i| chunks.level[i] == 0).map(|i| (chunks.model_tokens[i], chunks.over_budget[i]))
}

impl BudgetReport {
    /// The report over these chunks, each `(tokens, over_budget)`.
    pub fn of(rows: impl IntoIterator<Item = (u32, bool)>, b: Budget) -> BudgetReport {
        let target = f64::from(b.target.max(1));
        let (mut within, mut over, mut empty, mut under) = (0usize, 0usize, 0usize, 0usize);
        let mut tokens = Vec::new();
        for (t, o) in rows {
            within += usize::from(t <= b.max);
            over += usize::from(o);
            empty += usize::from(t <= b.overhead);
            under += usize::from(t > b.overhead && u64::from(t) * 4 < u64::from(b.target));
            tokens.push(t);
        }
        let n = tokens.len();
        if n == 0 {
            return BudgetReport {
                chunks: 0,
                within_max: 0.0,
                mean_utilization: 0.0,
                p95_utilization: 0.0,
                max_utilization: 0.0,
                over_budget: 0.0,
                empty: 0.0,
                undersized: 0.0,
            };
        }
        tokens.sort_unstable();
        let sum: u64 = tokens.iter().map(|&t| u64::from(t)).sum();
        let share = |c: usize| c as f64 / n as f64;
        BudgetReport {
            chunks: n,
            within_max: share(within),
            mean_utilization: sum as f64 / n as f64 / target,
            p95_utilization: f64::from(tokens[(95 * n).div_ceil(100) - 1]) / target,
            max_utilization: f64::from(tokens[n - 1]) / target,
            over_budget: share(over),
            empty: share(empty),
            undersized: share(under),
        }
    }
}

/// What a protected span holds together.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Protect {
    /// A code block, table or quote.
    Structure,
    /// A coreference chain.
    Coref,
    /// A pair of clauses held together.
    Clause,
}

/// Sentences `[start, end)` no cut should part.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Protected {
    pub kind: Protect,
    pub sentences: Range<usize>,
}

/// Protected spans of one kind, and how many no cut parted.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Kept {
    pub total: usize,
    pub intact: usize,
}

impl Kept {
    /// The share intact; 1 when there are none.
    pub fn rate(self) -> f64 {
        if self.total == 0 { 1.0 } else { self.intact as f64 / self.total as f64 }
    }

    fn add(&mut self, other: Kept) {
        self.total += other.total;
        self.intact += other.intact;
    }
}

/// How many protected spans of each kind a segmentation keeps intact.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Preservation {
    pub structure: Kept,
    pub coref: Kept,
    pub clause: Kept,
}

impl Preservation {
    /// The spans `cuts` (a segmentation of the sentences) keeps intact:
    /// those with no boundary b where `start < b < end`.
    pub fn of(protected: &[Protected], cuts: &Segmentation) -> Preservation {
        let mut p = Preservation::default();
        for s in protected {
            let r = &s.sentences;
            let parted = r.end > r.start + 1 && cuts.between(r.start, r.end - 1) > 0;
            let kept = Kept { total: 1, intact: usize::from(!parted) };
            match s.kind {
                Protect::Structure => p.structure.add(kept),
                Protect::Coref => p.coref.add(kept),
                Protect::Clause => p.clause.add(kept),
            }
        }
        p
    }

    /// Counts summed, for pooling documents.
    pub fn add(&mut self, other: Preservation) {
        self.structure.add(other.structure);
        self.coref.add(other.coref);
        self.clause.add(other.clause);
    }

    /// The share intact over every kind; 1 when there are none.
    pub fn rate(self) -> f64 {
        let mut all = self.structure;
        all.add(self.coref);
        all.add(self.clause);
        all.rate()
    }
}

/// The code blocks, tables and quotes of a structure as protected sentence
/// ranges: the sentences that have a byte inside the block. A block of
/// fewer than two sentences has no gap to part and is left out. Two binary
/// searches per block.
pub fn structure_protected(sentences: &Sentences, structure: &Structure) -> Vec<Protected> {
    let (starts, ends) = sentences.spans.columns();
    structure
        .spans
        .iter()
        .filter(|&(i, _)| matches!(structure.kind[i.get()], Block::Code | Block::Table | Block::Quote))
        .map(|(_, b)| ends.partition_point(|&x| x <= b.start)..starts.partition_point(|&x| x < b.end))
        .filter(|r| r.end > r.start + 1)
        .map(|sentences| Protected { kind: Protect::Structure, sentences })
        .collect()
}

/// A coreference boundary signal for `n` sentences: the gap after
/// sentence g costs `per_chain` for each chain that spans it (a chain of
/// sentences `[s, e)` spans the gaps `s..e - 1`), saturating at the i32
/// range. Every reason is `Cut::Sentence`, so the signal never relabels
/// a cut. Time is the sentences plus the chains, by a difference array.
pub fn coref_costs(chains: &[IxRange<Sentences>], n: usize, per_chain: i32) -> Boundaries {
    let mut delta = vec![0i64; n + 1];
    for c in chains {
        let (s, e) = (c.start.get(), c.end.get().min(n));
        if e > s + 1 {
            delta[s] += 1;
            delta[e - 1] -= 1;
        }
    }
    let mut open = 0i64;
    let cost = delta[..n]
        .iter()
        .map(|&d| {
            open += d;
            (open * i64::from(per_chain)).clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
        })
        .collect();
    Boundaries { cost, reason: vec![Cut::Sentence; n] }
}
