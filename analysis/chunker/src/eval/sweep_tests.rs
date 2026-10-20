use turbo_analysis::{Annotator, IxRange, ModelRef, Span, Spans, ViewId};
use turbo_analysis_layers::{Block, Boundaries, Cut, Sentences, Structure, Subwords};

use super::Segmentation;
use super::budget::{Protect, Protected, coref_costs};
use super::sweep::{ChunkerEvalReport, ChunkerSweep, Grid, Signals, SweepDoc};
use crate::{Costs, StructureBoundaries, Weight};

/// Twelve sentences "aaaa bbbb." of two subwords each, a space apart, in
/// two sections of six. Each section's topic shifts once, after its
/// second sentence in the first and its fourth in the second, so the gold
/// cuts at 2, 6 and 10. The drift signal makes those two shifts cheap;
/// one coreference chain spans sentences 3 to 5; the evidence is those
/// three sentences.
fn doc() -> SweepDoc {
    let n = 12u32;
    let (mut sentences, mut subwords) = (Spans::new(ViewId::SOURCE), Spans::new(ViewId::SOURCE));
    for k in 0..n {
        sentences.push(Span::new(11 * k, 11 * k + 10));
        subwords.push(Span::new(11 * k, 11 * k + 4));
        subwords.push(Span::new(11 * k + 5, 11 * k + 10));
    }
    let sentences = Sentences { spans: sentences };
    let model = ModelRef { id: "words".into(), revision: "0".into(), sha256: "0".into() };
    let subwords = Subwords { spans: subwords, ids: (0..2 * n as i32).collect(), model, token: None };
    let mut blocks = Spans::new(ViewId::SOURCE);
    blocks.push(Span::new(0, 65));
    blocks.push(Span::new(66, 131));
    let st = Structure { spans: blocks, kind: vec![Block::Section; 2], parent: vec![None; 2], depth: vec![1; 2] };
    let (structure,) = StructureBoundaries { costs: Costs::default() }.annotate((&sentences, Some(&st))).unwrap();
    let shift = |g: usize| g == 1 || g == 9;
    let drift = Boundaries {
        cost: (0..12).map(|g| if shift(g) { -2000 } else { 0 }).collect(),
        reason: (0..12).map(|g| if shift(g) { Cut::TopicShift } else { Cut::Sentence }).collect(),
    };
    let sentiment = Boundaries { cost: vec![0; 12], reason: vec![Cut::Sentence; 12] };
    let coref = coref_costs(&[IxRange::new(3, 6)], 12, Boundaries::UNIT);
    SweepDoc {
        sentences,
        subwords,
        structure: Some(st),
        signals: Signals { structure, drift, sentiment, coref },
        gold: Segmentation::new(12, vec![2, 6, 10]).unwrap(),
        protected: vec![Protected { kind: Protect::Coref, sentences: 3..6 }],
        evidence: vec![Span::new(33, 65)],
    }
}

fn grid() -> Grid {
    Grid {
        target: vec![4, 6, 9],
        max: vec![6, 12, 3],
        size_weight: vec![0, 1000],
        drift_weight: vec![Weight(0), Weight::ONE],
        sentiment_weight: vec![Weight(0), Weight(500)],
        coref_weight: vec![Weight(0), Weight::ONE],
    }
}

fn sweep() -> ChunkerEvalReport {
    ChunkerSweep { grid: grid(), overhead: 0 }.run(&[doc(), doc()]).unwrap()
}

#[test]
fn the_sweep_walks_the_grid_in_order_and_skips_targets_over_max() {
    let r = sweep();
    // Of nine (target, max) pairs, four have the target over the max.
    assert_eq!((r.points.len(), r.skipped), (5 * 16, 4 * 16));
    let first = r.points[0].setting;
    assert_eq!((first.budget.target, first.budget.max, first.budget.size_weight), (4, 6, 0));
    assert_eq!((first.drift, first.sentiment, first.coref), (Weight(0), Weight(0), Weight(0)));
    assert_eq!(r.points[1].setting.coref, Weight::ONE);
    assert_eq!(r.points[2].setting.sentiment, Weight(500));
    assert_eq!(r.points[16].setting.budget.max, 12);
    assert!(r.points.iter().all(|p| p.setting.budget.target <= p.setting.budget.max));
}

#[test]
fn the_sweep_is_deterministic() {
    assert_eq!(sweep(), sweep());
}

#[test]
fn the_cheap_topic_shifts_give_the_gold_cuts() {
    let r = sweep();
    let p = r
        .points
        .iter()
        .find(|p| {
            let s = p.setting;
            (s.budget.target, s.budget.max, s.budget.size_weight, s.drift) == (6, 12, 1000, Weight::ONE)
        })
        .unwrap();
    let s = p.scores;
    assert_eq!((s.pk, s.window_diff, s.f1_exact, s.f1_within_1), (0.0, 0.0, 1.0, 1.0));
    // The gold chunk [2, 6) is bytes [22, 65): the evidence is 32 of them.
    assert_eq!((s.iou, s.contained, s.redundancy), (32.0 / 43.0, 1.0, 1.0));
    // Six tokens a chunk: [2, 6) and [6, 10) hold eight, the others four.
    assert_eq!((s.budget.chunks, s.budget.within_max, s.budget.mean_utilization), (8, 1.0, 1.0));
    assert_eq!(s.preservation.coref.rate(), 1.0);
    // No drift: the topic shifts cost what every gap costs.
    let flat = r.points.iter().find(|q| q.setting == crate::eval::sweep::Setting { drift: Weight(0), ..p.setting });
    assert!(flat.unwrap().scores.f1_exact < 1.0);
}

#[test]
fn the_frontier_is_what_nothing_dominates() {
    let r = sweep();
    let pts = &r.points;
    assert!(!r.frontier.is_empty());
    let on: Vec<usize> = (0..pts.len()).filter(|&i| pts[i].rank == 0).collect();
    let mut sorted = r.frontier.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, on);
    for (i, p) in pts.iter().enumerate() {
        // Nothing of the same rank or a later one dominates a point, and
        // some point of the rank just before dominates every point past
        // rank 0.
        assert!(pts.iter().all(|q| q.rank < p.rank || !q.scores.dominates(&p.scores)), "point {i}");
        if p.rank > 0 {
            assert!(pts.iter().any(|q| q.rank == p.rank - 1 && q.scores.dominates(&p.scores)), "point {i}");
        }
    }
    // The frontier is listed by its objectives, best WindowDiff first.
    for w in r.frontier.windows(2) {
        let (a, b) = (pts[w[0]].scores.objectives(), pts[w[1]].scores.objectives());
        assert!(a <= b, "{a:?} {b:?}");
    }
    let best = pts[r.frontier[0]].scores;
    assert_eq!((best.window_diff, best.f1_within_1), (0.0, 1.0));
}

#[test]
fn the_sweep_refuses_what_it_cannot_score() {
    let s = ChunkerSweep { grid: grid(), overhead: 0 };
    assert!(s.run(&[]).is_err());
    let mut short = doc();
    short.signals.coref.cost.pop();
    assert!(s.run(&[short]).is_err());
    let mut wrong = doc();
    wrong.gold = Segmentation::no_cut(11);
    assert!(s.run(&[wrong]).is_err());
    let empty = ChunkerSweep { grid: Grid { coref_weight: vec![], ..grid() }, overhead: 0 };
    assert!(empty.run(&[doc()]).is_err());
}
