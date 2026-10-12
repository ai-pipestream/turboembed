use turbo_analysis::{IxRange, Span, Spans};

use super::*;
use crate::{Lengths, no_term, penalty, segment};

const B: Budget = Budget { max: 400, target: 300, size_weight: 1000, overhead: 0 };

/// Sentences of ten bytes each, back to back.
fn sentences(n: u32) -> Sentences {
    let mut spans = Spans::new(ViewId::SOURCE);
    for k in 0..n {
        spans.push(Span::new(10 * k, 10 * k + 10));
    }
    Sentences { spans }
}

/// A chunk by hand: sentences `[s, e)` of ten bytes each.
struct Row {
    s: u32,
    e: u32,
    level: u8,
    parent: Option<u32>,
    cost: i64,
    tokens: u32,
}

fn row(s: u32, e: u32) -> Row {
    Row { s, e, level: 0, parent: None, cost: 0, tokens: 0 }
}

fn chunks(view: ViewId, rows: &[Row]) -> Chunks {
    let mut c = Chunks {
        spans: Spans::new(view),
        level: Vec::new(),
        parent: Vec::new(),
        sentences: Vec::new(),
        cut: Vec::new(),
        cost: Vec::new(),
        model_tokens: Vec::new(),
        over_budget: Vec::new(),
    };
    for r in rows {
        c.spans.push(Span::new(10 * r.s, 10 * r.e));
        c.level.push(r.level);
        c.parent.push(r.parent.map(Ix::new));
        c.sentences.push(IxRange::new(r.s, r.e));
        c.cut.push(if r.s == 0 { Cut::DocumentStart } else { Cut::Sentence });
        c.cost.push(r.cost);
        c.model_tokens.push(r.tokens);
        c.over_budget.push(false);
    }
    c
}

/// What the chunker writes for a run with no structure.
fn cut(costs: &[i32], tokens: &[u32]) -> Chunks {
    let l = Lengths::of(tokens).unwrap();
    let rows: Vec<Row> = segment(costs, |i, j| l.run(i, j), B, no_term)
        .iter()
        .map(|s| Row { cost: s.cost as i64, tokens: s.tokens as u32, ..row(s.start as u32, s.end as u32) })
        .collect();
    chunks(ViewId::SOURCE, &rows)
}

fn run(chunks: &Chunks) -> Run<'_> {
    Run { chunks, execution: None, pricing: None }
}

#[test]
fn identical_runs_report_nothing() {
    let c = cut(&[1000, 1000, 100, 1000, 1000, 1000], &[100; 6]);
    let r = compare(run(&c), run(&c)).unwrap();
    assert!(r.is_identical());
    assert_eq!(r.matched, 2);
}

#[test]
fn a_moved_cut_is_reported_with_both_margins_and_both_tiers() {
    // The topic shift after the third sentence is cheap in A; in B, whose
    // vectors were computed at a lower precision, the cheap boundary is
    // after the fourth.
    let (ca, cb) = ([1000, 1000, 100, 1000, 1000, 1000], [1000, 1000, 1000, 100, 1000, 1000]);
    let (a, b) = (cut(&ca, &[100; 6]), cut(&cb, &[100; 6]));
    let s = sentences(6);
    let (ba, bb) = (
        Boundaries { cost: ca.to_vec(), reason: vec![Cut::Sentence; 6] },
        Boundaries { cost: cb.to_vec(), reason: vec![Cut::Sentence; 6] },
    );
    let fp32 = Execution { backend: "cpu".into(), device: "toy".into(), precision: "f32 (model)".into() };
    let fp16 = Execution { backend: "gpu".into(), device: "toy".into(), precision: "f16 (fastest)".into() };
    let ra = Run {
        chunks: &a,
        execution: Some(&fp32),
        pricing: Some(Pricing { sentences: &s, boundaries: &ba, budget: B }),
    };
    let rb = Run {
        chunks: &b,
        execution: Some(&fp16),
        pricing: Some(Pricing { sentences: &s, boundaries: &bb, budget: B }),
    };
    let r = compare(ra, rb).unwrap();
    assert_eq!(r.a.map(|e| e.precision.as_str()), Some("f32 (model)"));
    assert_eq!(r.b.map(|e| e.precision.as_str()), Some("f16 (fastest)"));
    assert_eq!(r.matched, 1);
    let [d] = &r.disagreements[..] else { panic!("one disagreement, not {:?}", r.disagreements) };
    assert_eq!((d.level, d.parent, d.after, d.before), (0, None, Some(0), None));
    // Each is its run's last chunk, so its cost is its size alone: none
    // for A's 300 tokens, a ninth of the weight, rounded down, for B's 200.
    let at = |start, cost, boundary_a, boundary_b| CutAt {
        start,
        reason: Cut::Sentence,
        cost,
        over_budget: false,
        boundary_a: Some(boundary_a),
        boundary_b: Some(boundary_b),
    };
    assert_eq!((&d.a[..], &d.b[..]), (&[at(30, 0, 100, 1000)][..], &[at(40, 111, 1000, 100)][..]));
    // A's own choice costs 100 (two chunks at the target); B's under A's
    // costs pays 1000 at the fourth boundary and 111 for each chunk a
    // third off the target. Under B's costs, A's choice pays 1000 and B's
    // own 100 plus the two 111s.
    assert_eq!(d.margin_a, Some(1000 + 2 * 111 - 100));
    assert_eq!(d.margin_b, Some(1000 - (100 + 2 * 111)));
    // The prices are the optimizer's: A's chunks cost 100 in the layer.
    assert_eq!(a.cost.iter().sum::<i64>(), 100);
}

#[test]
fn an_added_and_a_removed_cut_are_two_disagreements() {
    // A cuts at 30 and B does not; both cut at 50; B cuts at 70 and A
    // does not.
    let a = chunks(ViewId::SOURCE, &[row(0, 3), row(3, 5), row(5, 10)]);
    let b = chunks(ViewId::SOURCE, &[row(0, 5), row(5, 7), row(7, 10)]);
    let r = compare(run(&a), run(&b)).unwrap();
    assert_eq!(r.matched, 2);
    let got: Vec<_> = r
        .disagreements
        .iter()
        .map(|d| {
            let starts = |c: &[CutAt]| c.iter().map(|c| c.start).collect::<Vec<_>>();
            (d.after, d.before, starts(&d.a), starts(&d.b), d.margin_a, d.margin_b)
        })
        .collect();
    // Without costs there are no margins.
    assert_eq!(
        got,
        [(Some(0), Some(50), vec![30], vec![], None, None), (Some(50), None, vec![], vec![70], None, None)]
    );
}

#[test]
fn levels_are_compared_apart_and_children_follow_their_parent() {
    // Two sections, the second starting at 50 in A and at 40 in B. The
    // chunk at 60 is in both, but under different parents.
    let child = |s, e, p| Row { parent: Some(p), ..row(s, e) };
    let section = |s, e| Row { level: 1, ..row(s, e) };
    let a = chunks(ViewId::SOURCE, &[section(0, 5), child(0, 5, 0), section(5, 8), child(5, 6, 2), child(6, 8, 2)]);
    let b = chunks(ViewId::SOURCE, &[section(0, 4), child(0, 4, 0), section(4, 8), child(4, 6, 2), child(6, 8, 2)]);
    let s = sentences(8);
    let costs = Boundaries { cost: vec![1000; 8], reason: vec![Cut::Sentence; 8] };
    let with =
        |c| Run { chunks: c, execution: None, pricing: Some(Pricing { sentences: &s, boundaries: &costs, budget: B }) };
    let r = compare(with(&a), with(&b)).unwrap();
    let got: Vec<_> = r
        .disagreements
        .iter()
        .map(|d| (d.level, d.parent, d.a.iter().map(|c| c.start).collect(), d.b.iter().map(|c| c.start).collect()))
        .collect::<Vec<(u8, Option<u32>, Vec<u32>, Vec<u32>)>>();
    assert_eq!(
        got,
        [(0, Some(40), vec![], vec![40, 60]), (0, Some(50), vec![50, 60], vec![]), (1, None, vec![50], vec![40]),]
    );
    // The first chunks at 0, and the section at 0, are shared.
    assert_eq!(r.matched, 2);
    // No shared cut fixes where these stretches open, and the optimizer
    // does not cut sections, so none has a margin.
    assert!(r.disagreements.iter().all(|d| d.margin_a.is_none() && d.margin_b.is_none()));
}

#[test]
fn runs_in_different_views_are_refused() {
    let a = chunks(ViewId::SOURCE, &[row(0, 2)]);
    let b = chunks(ViewId(1), &[row(0, 2)]);
    assert_eq!(compare(run(&a), run(&b)), Err(Refused::Views { a: ViewId::SOURCE, b: ViewId(1) }));
}

#[test]
fn costs_that_do_not_fit_the_chunks_are_refused() {
    let a = chunks(ViewId::SOURCE, &[row(0, 2)]);
    let s = sentences(2);
    let short = Boundaries { cost: vec![1000], reason: vec![Cut::Sentence] };
    let b =
        Run { chunks: &a, execution: None, pricing: Some(Pricing { sentences: &s, boundaries: &short, budget: B }) };
    let Err(Refused::Invalid { run: Side::B, .. }) = compare(run(&a), b) else { panic!("refused") };
}

#[test]
fn a_cut_at_no_sentence_start_of_the_other_run_has_no_margin() {
    // B's sentences split differently, so its cut at 25 is no place A's
    // costs can price; A's cut at 30 is one B's can.
    let a = chunks(ViewId::SOURCE, &[Row { tokens: 300, ..row(0, 3) }, Row { tokens: 300, ..row(3, 6) }]);
    let mut b = chunks(ViewId::SOURCE, &[Row { tokens: 300, ..row(0, 3) }, Row { tokens: 300, ..row(3, 6) }]);
    b.spans = Spans::new(ViewId::SOURCE);
    b.spans.push(Span::new(0, 25));
    b.spans.push(Span::new(25, 60));
    let s = sentences(6);
    let mut sb = sentences(6);
    sb.spans = Spans::new(ViewId::SOURCE);
    for (x, y) in [(0, 10), (10, 25), (25, 30), (30, 40), (40, 50), (50, 60)] {
        sb.spans.push(Span::new(x, y));
    }
    let costs = Boundaries { cost: vec![1000; 6], reason: vec![Cut::Sentence; 6] };
    let ra =
        Run { chunks: &a, execution: None, pricing: Some(Pricing { sentences: &s, boundaries: &costs, budget: B }) };
    let rb =
        Run { chunks: &b, execution: None, pricing: Some(Pricing { sentences: &sb, boundaries: &costs, budget: B }) };
    let r = compare(ra, rb).unwrap();
    let [d] = &r.disagreements[..] else { panic!("one disagreement") };
    assert_eq!((d.a[0].start, d.b[0].start), (30, 25));
    assert_eq!((d.a[0].boundary_a, d.a[0].boundary_b), (Some(1000), Some(1000)));
    assert_eq!((d.b[0].boundary_a, d.b[0].boundary_b), (None, Some(1000)));
    assert_eq!(d.margin_a, None);
    assert_eq!(d.margin_b, Some(0));
}

#[test]
fn the_margin_is_what_the_optimizer_minimizes() {
    // On generated runs, every change of one gap to the optimizer's own
    // cut (a cut added or removed) costs at least as much, and the margin
    // is exactly the difference of the two totals, each summed here from
    // the boundary costs and `penalty` as the optimizer defines a chunk.
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut rand = |m: u64| {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) % m
    };
    let mut checked = 0;
    for case in 0..2000 {
        let n = 2 + rand(9) as usize;
        let tokens: Vec<u32> = (0..n).map(|_| rand(80) as u32).collect();
        let cost: Vec<i32> = (0..n).map(|_| rand(4001) as i32 - 1500).collect();
        let b = Budget {
            max: 20 + rand(200) as u32,
            target: rand(150) as u32,
            size_weight: [0, 1000, 1_000_000][rand(3) as usize],
            overhead: rand(4) as u32,
        };
        let l = Lengths::of(&tokens).unwrap();
        let size = |i: usize, j: usize| u64::from(l.run(i, j)) + u64::from(b.overhead);
        let total = |ends: &[usize]| {
            let mut i = 0;
            let mut t = 0i128;
            for &j in ends {
                t += i128::from(if j < n { cost[j - 1] } else { 0 }) + penalty(size(i, j), b);
                i = j;
            }
            t
        };
        let layer = |ends: &[usize]| {
            let mut i = 0;
            let rows: Vec<Row> = ends
                .iter()
                .map(|&j| {
                    let r = Row { tokens: size(i, j) as u32, ..row(i as u32, j as u32) };
                    i = j;
                    r
                })
                .collect();
            chunks(ViewId::SOURCE, &rows)
        };
        let segments = segment(&cost, |i, j| l.run(i, j), b, no_term);
        let best: Vec<usize> = segments.iter().map(|s| s.end).collect();
        assert_eq!(total(&best), segments.iter().map(|s| s.cost).sum::<i128>(), "case {case}");
        let (s, bounds) = (sentences(n as u32), Boundaries { cost: cost.clone(), reason: vec![Cut::Sentence; n] });
        let a = layer(&best);
        let pricing = Some(Pricing { sentences: &s, boundaries: &bounds, budget: b });
        for g in 1..n {
            let mut other: Vec<usize> = best.iter().copied().filter(|&e| e != g).collect();
            if other.len() == best.len() {
                other.push(g);
                other.sort_unstable();
            }
            let mut i = 0;
            let fits = other.iter().all(|&j| {
                let ok = j - i == 1 || size(i, j) <= u64::from(b.max);
                i = j;
                ok
            });
            if !fits {
                continue;
            }
            let o = layer(&other);
            let r = compare(Run { chunks: &a, execution: None, pricing }, Run { chunks: &o, execution: None, pricing })
                .unwrap();
            let [d] = &r.disagreements[..] else { panic!("case {case}, gap {g}: {:?}", r.disagreements) };
            let want = total(&other) - total(&best);
            assert!(want >= 0, "case {case}, gap {g}: the optimizer's cut is not the least");
            assert_eq!((d.margin_a, d.margin_b), (Some(want), Some(-want)), "case {case}, gap {g}");
            checked += 1;
        }
    }
    assert!(checked > 1000, "{checked}");
}
