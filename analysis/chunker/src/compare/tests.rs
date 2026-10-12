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
    // chunks at 0 and 60 are in both, but under parents of different
    // spans.
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
        .collect::<Vec<(u8, Option<Span>, Vec<u32>, Vec<u32>)>>();
    assert_eq!(
        got,
        [
            (0, Some(Span::new(0, 40)), vec![], vec![0]),
            (0, Some(Span::new(0, 50)), vec![0], vec![]),
            (0, Some(Span::new(40, 80)), vec![], vec![40, 60]),
            (0, Some(Span::new(50, 80)), vec![50, 60], vec![]),
            (1, None, vec![50], vec![40]),
        ]
    );
    // Only the section at 0 is shared: the first chunks start alike, but
    // under parents of different spans.
    assert_eq!(r.matched, 1);
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
    let refused = compare(run(&a), b).unwrap_err();
    assert_eq!(refused.to_string(), "run B: its costs need a boundary cost for every sentence");
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

fn priced<'a>(chunks: &'a Chunks, sentences: &'a Sentences, boundaries: &'a Boundaries, budget: Budget) -> Run<'a> {
    Run { chunks, execution: None, pricing: Some(Pricing { sentences, boundaries, budget }) }
}

#[test]
fn a_stretch_can_hold_several_cuts() {
    // A cuts at 20 and 40, B only at 30: one choice, not three.
    let t = |s, e| Row { tokens: 100 * (e - s), ..row(s, e) };
    let a = chunks(ViewId::SOURCE, &[t(0, 2), t(2, 4), t(4, 6)]);
    let b = chunks(ViewId::SOURCE, &[t(0, 3), t(3, 6)]);
    let (s, costs) = (sentences(6), Boundaries { cost: vec![1000; 6], reason: vec![Cut::Sentence; 6] });
    let r = compare(priced(&a, &s, &costs, B), priced(&b, &s, &costs, B)).unwrap();
    let [d] = &r.disagreements[..] else { panic!("one disagreement, not {:?}", r.disagreements) };
    let starts = |c: &[CutAt]| c.iter().map(|c| c.start).collect::<Vec<_>>();
    assert_eq!((d.after, d.before, starts(&d.a), starts(&d.b)), (Some(0), None, vec![20, 40], vec![30]));
    // A's three chunks of 200 pay 111 each and two boundaries of 1000;
    // B's two of 300 pay one boundary.
    assert_eq!((d.margin_a, d.margin_b), (Some(1000 - 2333), Some(2333 - 1000)));
}

#[test]
fn a_stretch_in_a_section_that_is_not_the_last_is_priced_to_its_end() {
    // Two sections of 30 bytes, cut alike but for the first. Sentences
    // hold 100 tokens; the boundary after the first costs 500.
    let section = |s, e| Row { level: 1, tokens: 100 * (e - s), ..row(s, e) };
    let child = |s, e, p| Row { parent: Some(p), tokens: 100 * (e - s), ..row(s, e) };
    let second = [section(3, 6), child(3, 4, 3), child(4, 6, 3)];
    let mut ra = vec![section(0, 3), child(0, 1, 0), child(1, 3, 0)];
    let mut rb = vec![section(0, 3), child(0, 2, 0), child(2, 3, 0)];
    ra.extend(second.iter().map(|r| Row { ..*r }));
    rb.extend(second.iter().map(|r| Row { ..*r }));
    let (a, b) = (chunks(ViewId::SOURCE, &ra), chunks(ViewId::SOURCE, &rb));
    let s = sentences(6);
    let costs = Boundaries { cost: vec![500, 1000, 1000, 1000, 1000, 1000], reason: vec![Cut::Sentence; 6] };
    let r = compare(priced(&a, &s, &costs, B), priced(&b, &s, &costs, B)).unwrap();
    let [d] = &r.disagreements[..] else { panic!("one disagreement, not {:?}", r.disagreements) };
    assert_eq!((d.parent, d.after, d.before), (Some(Span::new(0, 30)), Some(0), None));
    // A: 500 after the first sentence, 444 for a chunk of 100, 111 for
    // one of 200, and nothing at the section's end. B: 1000, 111, 444.
    assert_eq!((d.margin_a, d.margin_b), (Some(1555 - 1055), Some(1055 - 1555)));
}

#[test]
fn each_choice_is_sized_by_its_own_run_and_priced_by_the_others_budget() {
    // B's tokenizer counts 150 tokens a sentence to A's 100, and B's
    // target is 200 to A's 300.
    let (a, b) = (
        chunks(ViewId::SOURCE, &[Row { tokens: 100, ..row(0, 1) }, Row { tokens: 200, ..row(1, 3) }]),
        chunks(ViewId::SOURCE, &[Row { tokens: 300, ..row(0, 2) }, Row { tokens: 150, ..row(2, 3) }]),
    );
    let (s, costs) = (sentences(3), Boundaries { cost: vec![1000; 3], reason: vec![Cut::Sentence; 3] });
    let bb = Budget { target: 200, ..B };
    let r = compare(priced(&a, &s, &costs, B), priced(&b, &s, &costs, bb)).unwrap();
    let [d] = &r.disagreements[..] else { panic!("one disagreement, not {:?}", r.disagreements) };
    // Under A's budget B's chunks of 300 and 150 pay 0 and 250, A's own
    // of 100 and 200 pay 444 and 111. Under B's, A's pay 250 and 0, B's
    // own 250 and 62.
    assert_eq!(d.margin_a, Some(1250 - 1555));
    assert_eq!(d.margin_b, Some(1250 - 1312));
}

#[test]
fn children_of_parents_with_different_spans_have_no_margin() {
    // A has one section of six sentences cut in three; B has two
    // sections, of four and two, with the same three chunks. Matched by
    // start alone, A's choice would be priced to 60 and B's to 40.
    let section = |s, e| Row { level: 1, tokens: 100 * (e - s), ..row(s, e) };
    let child = |s, e, p| Row { parent: Some(p), tokens: 100 * (e - s), ..row(s, e) };
    let a = chunks(ViewId::SOURCE, &[section(0, 6), child(0, 2, 0), child(2, 4, 0), child(4, 6, 0)]);
    let b = chunks(ViewId::SOURCE, &[section(0, 4), child(0, 2, 0), child(2, 4, 0), section(4, 6), child(4, 6, 3)]);
    let (s, costs) = (sentences(6), Boundaries { cost: vec![1000; 6], reason: vec![Cut::Sentence; 6] });
    let r = compare(priced(&a, &s, &costs, B), priced(&b, &s, &costs, B)).unwrap();
    assert!(!r.is_identical());
    assert!(r.disagreements.iter().all(|d| d.margin_a.is_none() && d.margin_b.is_none()), "{r:?}");
}

#[test]
fn two_rows_at_one_place_or_overlapping_are_refused() {
    let two = chunks(ViewId::SOURCE, &[row(0, 3), row(3, 6), row(3, 6)]);
    let one = chunks(ViewId::SOURCE, &[row(0, 3), row(3, 6)]);
    let refused = compare(run(&two), run(&one)).unwrap_err();
    assert_eq!(
        refused.to_string(),
        "run A: its chunks need one row per place, and rows 1 and 2 both start at 30 on level 0"
    );
    let overlapping = chunks(ViewId::SOURCE, &[row(0, 4), row(3, 6)]);
    let refused = compare(run(&one), run(&overlapping)).unwrap_err();
    assert_eq!(
        refused.to_string(),
        "run B: its chunks need level 0 rows that do not overlap under one parent, and row 0 ends at 40 past row \
         1's start at 30"
    );
}

#[test]
fn chunks_that_do_not_start_and_end_at_sentences_are_refused() {
    let mut a = chunks(ViewId::SOURCE, &[row(0, 3), row(3, 6)]);
    a.spans = Spans::new(ViewId::SOURCE);
    a.spans.push(Span::new(0, 25));
    a.spans.push(Span::new(25, 60));
    let (s, costs) = (sentences(6), Boundaries { cost: vec![1000; 6], reason: vec![Cut::Sentence; 6] });
    let refused = compare(priced(&a, &s, &costs, B), run(&a)).unwrap_err();
    assert_eq!(
        refused.to_string(),
        "run A: its chunks need to start and end where its sentences do, and row 0 spans 0..25"
    );
}

#[test]
fn the_margin_is_what_the_optimizer_minimizes() {
    // On generated runs, with one section or two, the optimizer's own cut
    // is changed at one to three gaps (cuts added or removed). Every
    // stretch's margin is at least zero, B's is its negation, and their
    // sum is exactly the difference of the two totals, each summed here
    // from the boundary costs and `penalty` as the optimizer defines a
    // chunk, with a section's end free.
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut rand = |m: u64| {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) % m
    };
    let (mut checked, mut several, mut sectioned) = (0, 0, 0);
    for case in 0..3000 {
        let n = 2 + rand(11) as usize;
        let tokens: Vec<u32> = (0..n).map(|_| rand(80) as u32).collect();
        let cost: Vec<i32> = (0..n).map(|_| rand(4001) as i32 - 1500).collect();
        let b = Budget {
            max: 20 + rand(200) as u32,
            target: rand(150) as u32,
            size_weight: [0, 1000, 1_000_000][rand(3) as usize],
            overhead: rand(4) as u32,
        };
        let split = if n >= 4 && rand(2) == 0 { Some(2 + rand(n as u64 - 3) as usize) } else { None };
        let runs: Vec<std::ops::Range<usize>> = match split {
            Some(k) => vec![0..k, k..n],
            None => std::iter::once(0..n).collect(),
        };
        let l = Lengths::of(&tokens).unwrap();
        let size = |i: usize, j: usize| u64::from(l.run(i, j)) + u64::from(b.overhead);
        let total = |ends: &[Vec<usize>]| {
            let mut t = 0i128;
            for (r, ends) in runs.iter().zip(ends) {
                let mut i = r.start;
                for &j in ends {
                    t += i128::from(if j < r.end { cost[j - 1] } else { 0 }) + penalty(size(i, j), b);
                    i = j;
                }
            }
            t
        };
        let fits = |ends: &[Vec<usize>]| {
            runs.iter().zip(ends).all(|(r, ends)| {
                let mut i = r.start;
                ends.iter().all(|&j| {
                    let ok = j - i == 1 || size(i, j) <= u64::from(b.max);
                    i = j;
                    ok
                })
            })
        };
        let layer = |ends: &[Vec<usize>]| {
            let mut rows = Vec::new();
            for (r, ends) in runs.iter().zip(ends) {
                let parent = split.map(|_| {
                    rows.push(Row {
                        level: 1,
                        tokens: size(r.start, r.end) as u32,
                        ..row(r.start as u32, r.end as u32)
                    });
                    rows.len() as u32 - 1
                });
                let mut i = r.start;
                for &j in ends {
                    rows.push(Row { parent, tokens: size(i, j) as u32, ..row(i as u32, j as u32) });
                    i = j;
                }
            }
            chunks(ViewId::SOURCE, &rows)
        };
        let mut best = Vec::new();
        let mut optimizer = 0i128;
        for r in &runs {
            let segments = segment(&cost[r.clone()], |i, j| l.run(r.start + i, r.start + j), b, no_term);
            optimizer += segments.iter().map(|s| s.cost).sum::<i128>();
            best.push(segments.iter().map(|s| r.start + s.end).collect::<Vec<usize>>());
        }
        assert_eq!(total(&best), optimizer, "case {case}");
        let (s, bounds) = (sentences(n as u32), Boundaries { cost: cost.clone(), reason: vec![Cut::Sentence; n] });
        let a = layer(&best);
        let pricing = Some(Pricing { sentences: &s, boundaries: &bounds, budget: b });
        // The gaps inside a run, where a cut may be added or removed.
        let gaps: Vec<(usize, usize)> =
            runs.iter().enumerate().flat_map(|(k, r)| (r.start + 1..r.end).map(move |g| (k, g))).collect();
        for _ in 0..6 {
            let mut other = best.clone();
            let changes = 1 + rand(3) as usize;
            for _ in 0..changes {
                let (k, g) = gaps[rand(gaps.len() as u64) as usize];
                match other[k].binary_search(&g) {
                    Ok(at) => drop(other[k].remove(at)),
                    Err(at) => other[k].insert(at, g),
                }
            }
            if other == best || !fits(&other) {
                continue;
            }
            let o = layer(&other);
            let r = compare(Run { chunks: &a, execution: None, pricing }, Run { chunks: &o, execution: None, pricing })
                .unwrap();
            let mut sum = 0i128;
            for d in &r.disagreements {
                let m = d.margin_a.unwrap_or_else(|| panic!("case {case}: no margin in {d:?}"));
                assert!(m >= 0, "case {case}: the optimizer's cut is not the least in {d:?}");
                assert_eq!(d.margin_b, Some(-m), "case {case}");
                sum += m;
            }
            assert_eq!(sum, total(&other) - total(&best), "case {case}: {best:?} {other:?}");
            checked += 1;
            several += usize::from(r.disagreements.len() > 1 || changes > 1);
            sectioned += usize::from(split.is_some());
        }
    }
    assert!(checked > 5000 && several > 1000 && sectioned > 1000, "{checked} {several} {sectioned}");
}
