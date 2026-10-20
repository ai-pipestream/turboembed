//! The sweep: the chunker run over every setting of a grid, on documents
//! with a gold segmentation, every setting scored, and the settings ranked
//! into Pareto fronts.
//!
//! A setting is a budget (target, max and size weight; the overhead is the
//! sweep's) and the weights of three boundary signals beside the
//! structure's, which always counts at `Weight::ONE`: drift, sentiment and
//! coreference. The gap after sentence i costs
//! `(structure + drift * wd + sentiment * ws + coref * wc) / 1000`, summed
//! in i128, rounded down and saturated to i32, the rule
//! `weigh_with_sentiment` sums three signals by; the reasons are its
//! reasons. A setting whose target is over its max is skipped.
//!
//! Each setting is scored on four objectives, and a setting dominates
//! another when it is no worse on all four and better on one:
//!
//! - WindowDiff, lower is better, the mean over documents;
//! - boundary F1 within one sentence, higher is better, the same mean;
//! - the budget gap, `|1 - mean utilization|`, lower is better;
//! - byte span IoU, higher is better, the mean over evidence spans.
//!
//! Span IoU here retrieves, for each gold evidence span, the leaf chunks
//! that overlap it: the chunks an oracle would retrieve, so it measures
//! how tightly the cuts wrap the evidence, not how well an embedder ranks.
//!
//! Rank 0 is the Pareto frontier: the settings nothing dominates. Rank r
//! is the frontier once ranks below r are removed. The frontier is listed
//! by WindowDiff, then F1 (higher first), then the gap, then IoU (higher
//! first), then grid order. The grid is walked in a fixed order (target,
//! max, size weight, then the drift, sentiment and coreference weights,
//! each in the order given), so a sweep is the same on every run.

use std::cmp::Ordering;

use turbo_analysis::{Error, Span};
use turbo_analysis_layers::{Boundaries, Sentences, Structure, Subwords};

use super::boundary::score;
use super::budget::{BudgetReport, Preservation, Protected, leaves};
use super::coverage::{coverage, token_range};
use super::{Segmentation, invalid};
use crate::{Budget, Chunker, Weight, no_term, weigh_with_sentiment};

/// The boundary signals of one document, one cost per sentence each, as
/// the steps that write them compute them: the structure's costs, the
/// drift's, the sentiment shift's and the coreference continuity costs.
pub struct Signals {
    pub structure: Boundaries,
    pub drift: Boundaries,
    pub sentiment: Boundaries,
    pub coref: Boundaries,
}

/// One document of a sweep: its layers, its signals and its gold.
pub struct SweepDoc {
    pub sentences: Sentences,
    pub subwords: Subwords,
    pub structure: Option<Structure>,
    pub signals: Signals,
    /// The gold segmentation of the sentences.
    pub gold: Segmentation,
    pub protected: Vec<Protected>,
    /// Gold evidence spans, in bytes of the source.
    pub evidence: Vec<Span>,
}

/// The values each setting is drawn from.
#[derive(Clone, Debug, PartialEq)]
pub struct Grid {
    pub target: Vec<u32>,
    pub max: Vec<u32>,
    pub size_weight: Vec<u32>,
    pub drift_weight: Vec<Weight>,
    pub sentiment_weight: Vec<Weight>,
    pub coref_weight: Vec<Weight>,
}

/// One point of the grid.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Setting {
    pub budget: Budget,
    pub drift: Weight,
    pub sentiment: Weight,
    pub coref: Weight,
}

/// What one setting scored over every document.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Scores {
    pub pk: f64,
    pub window_diff: f64,
    pub f1_exact: f64,
    pub f1_within_1: f64,
    pub f1_within_2: f64,
    pub budget: BudgetReport,
    pub iou: f64,
    pub token_iou: f64,
    pub contained: f64,
    pub redundancy: f64,
    pub preservation: Preservation,
}

impl Scores {
    /// The four objectives, each turned so that lower is better.
    pub fn objectives(&self) -> [f64; 4] {
        [self.window_diff, -self.f1_within_1, (1.0 - self.budget.mean_utilization).abs(), -self.iou]
    }

    /// Whether these scores dominate `other`.
    pub fn dominates(&self, other: &Scores) -> bool {
        let (a, b) = (self.objectives(), other.objectives());
        a.iter().zip(&b).all(|(x, y)| x <= y) && a.iter().zip(&b).any(|(x, y)| x < y)
    }
}

/// A scored setting and its Pareto rank.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SweepPoint {
    pub setting: Setting,
    pub scores: Scores,
    pub rank: usize,
}

/// Every setting of a sweep in grid order, the frontier as indexes into
/// them in frontier order, and how many settings were skipped.
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkerEvalReport {
    pub points: Vec<SweepPoint>,
    pub frontier: Vec<usize>,
    pub skipped: usize,
}

/// A sweep over `grid`, every chunk carrying `overhead` tokens.
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkerSweep {
    pub grid: Grid,
    pub overhead: u32,
}

impl ChunkerSweep {
    /// Runs the chunker at every setting over `docs` and ranks the
    /// settings. Refused: an empty grid axis, no documents, a document
    /// whose signals or gold do not have one entry per sentence.
    pub fn run(&self, docs: &[SweepDoc]) -> Result<ChunkerEvalReport, Error> {
        let g = &self.grid;
        if g.target.is_empty()
            || g.max.is_empty()
            || g.size_weight.is_empty()
            || g.drift_weight.is_empty()
            || g.sentiment_weight.is_empty()
            || g.coref_weight.is_empty()
        {
            return Err(invalid("a sweep grid with an empty axis".into()));
        }
        if docs.is_empty() {
            return Err(invalid("a sweep over no documents".into()));
        }
        for (d, doc) in docs.iter().enumerate() {
            check(d, doc)?;
        }
        // Each document's weighed costs, once per weight triple, in the
        // order the inner loops walk them.
        let mut weights = Vec::new();
        for &wd in &g.drift_weight {
            for &ws in &g.sentiment_weight {
                for &wc in &g.coref_weight {
                    weights.push((wd, ws, wc));
                }
            }
        }
        let costs: Vec<Vec<Boundaries>> =
            docs.iter().map(|doc| weights.iter().map(|&w| combine(&doc.signals, w)).collect()).collect();
        let (mut points, mut skipped) = (Vec::new(), 0);
        for &target in &g.target {
            for &max in &g.max {
                if target > max {
                    skipped += g.size_weight.len() * weights.len();
                    continue;
                }
                for &size_weight in &g.size_weight {
                    let budget = Budget { max, target, size_weight, overhead: self.overhead };
                    for (w, &(drift, sentiment, coref)) in weights.iter().enumerate() {
                        let setting = Setting { budget, drift, sentiment, coref };
                        let scores = evaluate(docs, budget, |d| &costs[d][w])?;
                        points.push(SweepPoint { setting, scores, rank: 0 });
                    }
                }
            }
        }
        let frontier = rank(&mut points);
        Ok(ChunkerEvalReport { points, frontier, skipped })
    }
}

fn check(d: usize, doc: &SweepDoc) -> Result<(), Error> {
    let n = doc.sentences.spans.len();
    let s = &doc.signals;
    for (name, b) in
        [("structure", &s.structure), ("drift", &s.drift), ("sentiment", &s.sentiment), ("coref", &s.coref)]
    {
        if b.cost.len() != n || b.reason.len() != n {
            return Err(invalid(format!("document {d}: {} {name} costs for {n} sentences", b.cost.len())));
        }
    }
    if doc.gold.units() != n {
        return Err(invalid(format!("document {d}: a gold of {} units for {n} sentences", doc.gold.units())));
    }
    Ok(())
}

/// The four signals weighed into one cost per gap. The lengths are
/// checked before.
fn combine(s: &Signals, (wd, ws, wc): (Weight, Weight, Weight)) -> Boundaries {
    let reason = weigh_with_sentiment(&s.structure, &s.drift, &s.sentiment, (Weight::ONE, wd, ws))
        .map(|b| b.reason)
        .unwrap_or_default();
    let cost = (0..s.structure.cost.len())
        .map(|i| {
            let sum = i128::from(s.structure.cost[i]) * i128::from(Weight::ONE.0)
                + i128::from(s.drift.cost[i]) * i128::from(wd.0)
                + i128::from(s.sentiment.cost[i]) * i128::from(ws.0)
                + i128::from(s.coref.cost[i]) * i128::from(wc.0);
            sum.div_euclid(1000).clamp(i128::from(i32::MIN), i128::from(i32::MAX)) as i32
        })
        .collect();
    Boundaries { cost, reason }
}

/// One setting over every document: boundary measures averaged over
/// documents, budget pooled over chunks, coverage averaged over evidence
/// spans, preservation counts summed.
fn evaluate<'a>(docs: &[SweepDoc], budget: Budget, costs: impl Fn(usize) -> &'a Boundaries) -> Result<Scores, Error> {
    let chunker = Chunker { budget };
    let (mut pk, mut wd, mut f0, mut f1, mut f2) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let mut rows = Vec::new();
    let mut preservation = Preservation::default();
    let (mut iou, mut token_iou, mut contained, mut redundancy, mut spans) = (0.0, 0.0, 0.0, 0.0, 0usize);
    for (d, doc) in docs.iter().enumerate() {
        let n = doc.sentences.spans.len();
        let chunks =
            chunker.chunk(super::ID, &doc.sentences, &doc.subwords, costs(d), doc.structure.as_ref(), |_| no_term)?;
        let cuts = Segmentation::of_chunks(&chunks, n)?;
        let s = score(&doc.gold, &cuts)?;
        (pk, wd) = (pk + s.pk, wd + s.window_diff);
        (f0, f1, f2) = (f0 + s.exact.f1, f1 + s.within_1.f1, f2 + s.within_2.f1);
        rows.extend(leaves(&chunks));
        preservation.add(Preservation::of(&doc.protected, &cuts));
        let leaf: Vec<Span> =
            chunks.spans.iter().filter(|&(i, _)| chunks.level[i.get()] == 0).map(|(_, s)| s).collect();
        for &e in &doc.evidence {
            let hit: Vec<Span> = leaf.iter().copied().filter(|c| c.start < e.end && e.start < c.end).collect();
            let bytes = |s: Span| u64::from(s.start)..u64::from(s.end);
            let c = coverage(&[bytes(e)], &hit.iter().map(|&s| bytes(s)).collect::<Vec<_>>());
            let tokens: Vec<_> = hit.iter().map(|&s| token_range(&doc.subwords.spans, s)).collect();
            let t = coverage(&[token_range(&doc.subwords.spans, e)], &tokens);
            (iou, token_iou, contained, redundancy) =
                (iou + c.iou, token_iou + t.iou, contained + c.contained, redundancy + c.redundancy);
            spans += 1;
        }
    }
    let per_doc = docs.len() as f64;
    let per_span = |x: f64| if spans == 0 { 1.0 } else { x / spans as f64 };
    Ok(Scores {
        pk: pk / per_doc,
        window_diff: wd / per_doc,
        f1_exact: f0 / per_doc,
        f1_within_1: f1 / per_doc,
        f1_within_2: f2 / per_doc,
        budget: BudgetReport::of(rows, budget),
        iou: per_span(iou),
        token_iou: per_span(token_iou),
        contained: per_span(contained),
        redundancy: per_span(redundancy),
        preservation,
    })
}

/// Sets every point's Pareto rank by non-dominated sorting (Deb et al.,
/// 2002): who each point dominates and how many dominate it, then peel
/// the fronts. Time is the square of the points. Returns the frontier in
/// frontier order.
fn rank(points: &mut [SweepPoint]) -> Vec<usize> {
    let n = points.len();
    let mut beats: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut beaten = vec![0usize; n];
    for i in 0..n {
        for j in i + 1..n {
            if points[i].scores.dominates(&points[j].scores) {
                beats[i].push(j);
                beaten[j] += 1;
            } else if points[j].scores.dominates(&points[i].scores) {
                beats[j].push(i);
                beaten[i] += 1;
            }
        }
    }
    let mut front: Vec<usize> = (0..n).filter(|&i| beaten[i] == 0).collect();
    let mut frontier = front.clone();
    let mut r = 0;
    while !front.is_empty() {
        let mut next = Vec::new();
        for &i in &front {
            points[i].rank = r;
            for &j in &beats[i] {
                beaten[j] -= 1;
                if beaten[j] == 0 {
                    next.push(j);
                }
            }
        }
        next.sort_unstable();
        front = next;
        r += 1;
    }
    frontier.sort_by(|&a, &b| order(&points[a].scores, &points[b].scores).then(a.cmp(&b)));
    frontier
}

/// The frontier order: the objectives in turn, each lower first.
fn order(a: &Scores, b: &Scores) -> Ordering {
    let (a, b) = (a.objectives(), b.objectives());
    a.iter()
        .zip(&b)
        .map(|(x, y)| x.partial_cmp(y).unwrap_or(Ordering::Equal))
        .find(|o| o.is_ne())
        .unwrap_or(Ordering::Equal)
}
