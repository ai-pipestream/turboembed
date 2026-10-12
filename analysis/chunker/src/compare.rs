//! Whether two runs of the chunker over one document cut it in the same
//! places: two backends, or two precision tiers on one backend. The cuts
//! are compared by exact byte offset, level by level and under the same
//! parent, and each place the runs disagree is reported with what each
//! run's own costs say about the other's choice.

use std::fmt;

use turbo_analysis::{Execution, Ix, Layer, ViewId};
use turbo_analysis_layers::{Boundaries, Chunks, Cut, Sentences};

use crate::Budget;

mod margin;

/// One run's chunks and what is known about how they were made.
#[derive(Copy, Clone)]
pub struct Run<'a> {
    pub chunks: &'a Chunks,
    /// Where the model behind the run's boundary costs ran: backend,
    /// device and precision tier. The chunker runs no model, so the
    /// chunks' own provenance has none; the caller takes it from the
    /// layer the costs were computed from, such as the vectors.
    pub execution: Option<&'a Execution>,
    /// The costs the run was cut with. They price the other run's cuts;
    /// without them a disagreement has no margin.
    pub pricing: Option<Pricing<'a>>,
}

/// What the optimizer read to cut a run: the boundary cost after each
/// sentence and the budget that prices a chunk's size.
#[derive(Copy, Clone)]
pub struct Pricing<'a> {
    pub sentences: &'a Sentences,
    pub boundaries: &'a Boundaries,
    pub budget: Budget,
}

/// What two runs cut differently. Empty when they cut alike.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Report<'a> {
    /// Where run A's model ran, as the run was given.
    pub a: Option<&'a Execution>,
    pub b: Option<&'a Execution>,
    /// Cuts both runs made: same level, same parent start, same offset.
    pub matched: usize,
    /// In order of level, parent start and offset.
    pub disagreements: Vec<Disagreement>,
}

impl Report<'_> {
    pub fn is_identical(&self) -> bool {
        self.disagreements.is_empty()
    }
}

/// A stretch of one level, under one parent, where the runs cut
/// differently, between two cuts they share (or the parent's ends). A
/// chunk's cost depends only on its own sentences and the boundary that
/// closes it, so the optimizer's choices on either side of a shared cut
/// are independent: each disagreement is a choice of its own, and a moved
/// cut is one with a cut on each side.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Disagreement {
    pub level: u8,
    /// The start of the parent chunk, the same in both runs; `None` for
    /// chunks with no parent.
    pub parent: Option<u32>,
    /// The shared cut it follows; `None` from the first chunk on.
    pub after: Option<u32>,
    /// The shared cut it ends at; `None` to the last chunk.
    pub before: Option<u32>,
    /// The cuts only run A made here, in text order.
    pub a: Vec<CutAt>,
    pub b: Vec<CutAt>,
    /// How much more run A's own costs charge for cutting this stretch as
    /// B did than as A did, in `Boundaries::UNIT`s, exactly as the
    /// optimizer adds them: zero is a tie, and a margin smaller than what
    /// the precision tier moves a cost by explains the difference. It is
    /// what A would pay to switch to B's choice, so it is at least the gap
    /// between A's best and second best cut. `None` without A's costs,
    /// above level 0 (the levels the optimizer does not cut), from the
    /// first chunk on (where no shared cut fixes the opening chunk), or
    /// where B cuts at a place that is no sentence start of A's.
    pub margin_a: Option<i128>,
    pub margin_b: Option<i128>,
}

/// One run's cut that the other run did not make.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CutAt {
    /// The byte offset the chunk starts at, in the chunks' view.
    pub start: u32,
    pub reason: Cut,
    /// What the chunk starting here cost the run's optimizer, from the
    /// layer: the boundary that closes it plus its size penalty.
    pub cost: i64,
    /// Whether the chunk is over the budget's maximum: one sentence alone.
    pub over_budget: bool,
    /// What each run's boundary costs charge for a cut at this offset,
    /// when the run has costs and the offset is one of its sentence starts:
    /// the number a precision tier moves.
    pub boundary_a: Option<i32>,
    pub boundary_b: Option<i32>,
}

/// Which of the two runs.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Side {
    A,
    B,
}

/// Why two runs cannot be compared.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refused {
    /// The chunks are in different text views, so their offsets name
    /// different bytes.
    Views { a: ViewId, b: ViewId },
    /// A run's layers break their own invariants or do not fit together.
    Invalid { run: Side, detail: String },
}

impl fmt::Display for Refused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refused::Views { a, b } => write!(f, "run A's chunks are in view {}, run B's in view {}", a.0, b.0),
            Refused::Invalid { run, detail } => write!(f, "run {run:?}: {detail}"),
        }
    }
}

impl std::error::Error for Refused {}

/// Compares the cuts of two runs. A cut is a chunk's start at its level
/// under its parent, the parent named by its own start, so a chunk whose
/// parent moved is a disagreement under each parent.
pub fn compare<'a>(a: Run<'a>, b: Run<'a>) -> Result<Report<'a>, Refused> {
    check(a, Side::A)?;
    check(b, Side::B)?;
    let (va, vb) = (a.chunks.spans.view(), b.chunks.spans.view());
    if va != vb {
        return Err(Refused::Views { a: va, b: vb });
    }
    let (pa, pb) = (places(a.chunks), places(b.chunks));
    let mut report = Report { a: a.execution, b: b.execution, matched: 0, disagreements: Vec::new() };
    let mut open = Open { group: None, after: None, a: Vec::new(), b: Vec::new() };
    let (mut i, mut j) = (0, 0);
    while i < pa.len() || j < pb.len() {
        let next = match (pa.get(i), pb.get(j)) {
            (Some(x), Some(y)) => x.0.cmp(&y.0),
            (Some(_), None) => std::cmp::Ordering::Less,
            _ => std::cmp::Ordering::Greater,
        };
        let place = if next.is_le() { pa[i].0 } else { pb[j].0 };
        if open.group != Some(place.group()) {
            open.close(a, b, None, &mut report);
            open.group = Some(place.group());
            open.after = None;
        }
        match next {
            std::cmp::Ordering::Equal => {
                open.close(a, b, Some(place.start), &mut report);
                open.after = Some((place.start, pa[i].1, pb[j].1));
                report.matched += 1;
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => {
                open.a.push(pa[i].1);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                open.b.push(pb[j].1);
                j += 1;
            }
        }
    }
    open.close(a, b, None, &mut report);
    Ok(report)
}

/// Where a cut is: its level, its parent's start and its own start. The
/// order is the report's.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Place {
    level: u8,
    parent: Option<u32>,
    start: u32,
}

impl Place {
    fn group(self) -> (u8, Option<u32>) {
        (self.level, self.parent)
    }
}

/// Every chunk's place with its row, sorted; ties keep row order.
fn places(c: &Chunks) -> Vec<(Place, usize)> {
    let start = |i: usize| c.spans.get(Ix::new(i as u32)).start;
    let mut v: Vec<(Place, usize)> = (0..c.spans.len())
        .map(|i| (Place { level: c.level[i], parent: c.parent[i].map(|p| start(p.get())), start: start(i) }, i))
        .collect();
    v.sort_unstable();
    v
}

/// The cuts seen since the last shared one in the current group.
struct Open {
    group: Option<(u8, Option<u32>)>,
    /// The last shared cut: its offset and its row in A and in B.
    after: Option<(u32, usize, usize)>,
    a: Vec<usize>,
    b: Vec<usize>,
}

impl Open {
    fn close(&mut self, a: Run<'_>, b: Run<'_>, before: Option<u32>, report: &mut Report<'_>) {
        if self.a.is_empty() && self.b.is_empty() {
            return;
        }
        let (level, parent) = self.group.expect("cuts belong to a group");
        let cut = |run: Run<'_>, row: usize| {
            let start = run.chunks.spans.get(Ix::new(row as u32)).start;
            CutAt {
                start,
                reason: run.chunks.cut[row],
                cost: run.chunks.cost[row],
                over_budget: run.chunks.over_budget[row],
                boundary_a: a.pricing.and_then(|p| margin::boundary(&p, start)),
                boundary_b: b.pricing.and_then(|p| margin::boundary(&p, start)),
            }
        };
        let priced = level == 0;
        let (margin_a, margin_b) = match self.after {
            Some((_, ra, rb)) if priced => (
                margin::margin(a.pricing, before, (a.chunks, ra, &self.a[..]), (b.chunks, rb, &self.b[..])),
                margin::margin(b.pricing, before, (b.chunks, rb, &self.b[..]), (a.chunks, ra, &self.a[..])),
            ),
            _ => (None, None),
        };
        report.disagreements.push(Disagreement {
            level,
            parent,
            after: self.after.map(|s| s.0),
            before,
            a: self.a.iter().map(|&r| cut(a, r)).collect(),
            b: self.b.iter().map(|&r| cut(b, r)).collect(),
            margin_a,
            margin_b,
        });
        self.a.clear();
        self.b.clear();
    }
}

fn check(run: Run<'_>, side: Side) -> Result<(), Refused> {
    let invalid = |detail: String| Refused::Invalid { run: side, detail };
    run.chunks.check().map_err(|d| invalid(format!("its chunks need {d}")))?;
    if let Some(c) = run.pricing {
        if c.sentences.spans.view() != run.chunks.spans.view() {
            return Err(invalid("its sentences and chunks are in different views".into()));
        }
        if c.boundaries.cost.len() != c.sentences.spans.len() {
            return Err(invalid("a boundary cost for every sentence".into()));
        }
        if !c.sentences.spans.is_ordered() {
            return Err(invalid("its sentences in text order".into()));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
