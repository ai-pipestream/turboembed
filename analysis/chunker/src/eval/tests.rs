use turbo_analysis::{Ix, IxRange, Span, Spans, ViewId};
use turbo_analysis_layers::{Block, Cut, Sentences, Structure, Subwords};

use super::boundary::{pk, prf, probe_distance, score, window_diff};
use super::budget::{BudgetReport, Kept, Preservation, Protect, Protected, coref_costs, structure_protected};
use super::coverage::{coverage, token_range};
use super::*;
use crate::Budget;

fn seg(lengths: &[usize]) -> Segmentation {
    Segmentation::of_lengths(lengths).unwrap()
}

/// A deterministic generator for the property checks.
fn rng(seed: u64) -> impl FnMut(u64) -> u64 {
    let mut state = seed;
    move |m| {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) % m
    }
}

fn random_seg(rand: &mut impl FnMut(u64) -> u64, units: usize) -> Segmentation {
    let b = (1..units).filter(|_| rand(3) == 0).collect();
    Segmentation::new(units, b).unwrap()
}

#[test]
fn a_segmentation_checks_its_boundaries() {
    assert!(Segmentation::new(5, vec![0]).is_err());
    assert!(Segmentation::new(5, vec![5]).is_err());
    assert!(Segmentation::new(5, vec![3, 2]).is_err());
    assert!(Segmentation::new(5, vec![2, 2]).is_err());
    assert!(Segmentation::of_lengths(&[2, 0, 1]).is_err());
    let s = seg(&[3, 4, 2]);
    assert_eq!((s.units(), s.boundaries(), s.segments()), (9, &[3, 7][..], 3));
    assert_eq!((s.between(0, 3), s.between(3, 7), s.between(2, 8), s.between(5, 1)), (1, 1, 2, 0));
    assert_eq!(Segmentation::no_cut(0).segments(), 0);
}

#[test]
fn a_perfect_segmentation_scores_zero_error_and_full_f1() {
    let gold = seg(&[3, 4, 2, 5, 1, 6]);
    let s = score(&gold, &gold).unwrap();
    assert_eq!((s.pk, s.window_diff), (0.0, 0.0));
    for p in [s.exact, s.within_1, s.within_2] {
        assert_eq!((p.precision, p.recall, p.f1), (1.0, 1.0, 1.0));
    }
}

#[test]
fn the_no_cut_and_all_cut_baselines() {
    // Four segments of three: k = 12 / 8 = 1, eleven probes, three of
    // them across a reference boundary.
    let gold = seg(&[3, 3, 3, 3]);
    assert_eq!(probe_distance(&gold), 1);
    let none = score(&gold, &Segmentation::no_cut(12)).unwrap();
    assert_eq!((none.pk, none.window_diff), (3.0 / 11.0, 3.0 / 11.0));
    assert_eq!((none.exact.precision, none.exact.recall, none.exact.f1), (1.0, 0.0, 0.0));
    let all = score(&gold, &Segmentation::all_cut(12)).unwrap();
    assert_eq!((all.pk, all.window_diff), (8.0 / 11.0, 8.0 / 11.0));
    assert_eq!((all.exact.matched, all.exact.precision, all.exact.recall), (3, 3.0 / 11.0, 1.0));
    assert!((all.exact.f1 - 3.0 / 7.0).abs() < 1e-15);
    // No boundary on either side is a perfect score.
    let p = prf(&[], &[], 0);
    assert_eq!((p.precision, p.recall, p.f1), (1.0, 1.0, 1.0));
}

#[test]
fn a_near_miss_by_hand() {
    // Ten units, one reference boundary at 5, a hypothesis one at 6:
    // k = 10 / 4 = 2, eight probes, and they disagree at i = 3 and i = 5.
    let (r, h) = (Segmentation::new(10, vec![5]).unwrap(), Segmentation::new(10, vec![6]).unwrap());
    let s = score(&r, &h).unwrap();
    assert_eq!((s.k, s.pk, s.window_diff), (2, 0.25, 0.25));
    assert_eq!((s.exact.f1, s.within_1.f1, s.within_2.f1), (0.0, 1.0, 1.0));
    assert!(score(&r, &Segmentation::no_cut(9)).is_err());
}

#[test]
fn tolerance_matches_each_boundary_once() {
    let p = prf(&[5, 10], &[6, 12], 0);
    assert_eq!(p.matched, 0);
    assert_eq!(prf(&[5, 10], &[6, 12], 1).matched, 1);
    assert_eq!(prf(&[5, 10], &[6, 12], 2).matched, 2);
    // Two hypothesis boundaries either side of one reference boundary:
    // only one matches.
    let p = prf(&[5], &[4, 6], 1);
    assert_eq!((p.matched, p.precision, p.recall), (1, 0.5, 1.0));
}

/// The largest matching by augmenting paths, to check the greedy one.
fn largest(reference: &[usize], hypothesis: &[usize], t: usize) -> usize {
    fn augment(h: usize, r: &[usize], hy: &[usize], t: usize, owner: &mut [Option<usize>], seen: &mut [bool]) -> bool {
        for j in 0..r.len() {
            if hy[h].abs_diff(r[j]) <= t && !seen[j] {
                seen[j] = true;
                if owner[j].is_none_or(|o| augment(o, r, hy, t, owner, seen)) {
                    owner[j] = Some(h);
                    return true;
                }
            }
        }
        false
    }
    let mut owner = vec![None; reference.len()];
    (0..hypothesis.len())
        .filter(|&h| augment(h, reference, hypothesis, t, &mut owner, &mut vec![false; reference.len()]))
        .count()
}

#[test]
fn the_greedy_matching_is_a_largest_one() {
    let mut rand = rng(7);
    for case in 0..2000 {
        let units = 2 + rand(30) as usize;
        let (r, h) = (random_seg(&mut rand, units), random_seg(&mut rand, units));
        for t in 0..4 {
            let got = prf(r.boundaries(), h.boundaries(), t).matched;
            assert_eq!(got, largest(r.boundaries(), h.boundaries(), t), "case {case} t {t}: {r:?} {h:?}");
        }
    }
}

#[test]
fn pk_and_window_diff_follow_their_definitions() {
    // Straight from the definitions: same segment means no boundary
    // between the probes; WindowDiff compares the boundary counts.
    let direct = |r: &Segmentation, h: &Segmentation, k: usize| {
        let n = r.units();
        if n <= k {
            return (0.0, 0.0);
        }
        let count = |s: &Segmentation, i: usize| s.boundaries().iter().filter(|&&b| i < b && b <= i + k).count();
        let pk = (0..n - k).filter(|&i| (count(r, i) == 0) != (count(h, i) == 0)).count();
        let wd = (0..n - k).filter(|&i| count(r, i) != count(h, i)).count();
        (pk as f64 / (n - k) as f64, wd as f64 / (n - k) as f64)
    };
    let mut rand = rng(11);
    for case in 0..2000 {
        let units = 1 + rand(40) as usize;
        let (r, h) = (random_seg(&mut rand, units), random_seg(&mut rand, units));
        let k = 1 + rand(6) as usize;
        let (p, w) = (pk(&r, &h, k).unwrap(), window_diff(&r, &h, k).unwrap());
        assert_eq!((p, w), direct(&r, &h, k), "case {case}: {r:?} {h:?} k {k}");
        assert!((0.0..=1.0).contains(&p) && (0.0..=1.0).contains(&w) && p <= w);
        let s = score(&r, &h).unwrap();
        assert_eq!((s.pk, s.window_diff), direct(&r, &h, probe_distance(&r)));
    }
}

/// `coverage` of `(start, end)` pairs.
fn cov(gold: &[(u64, u64)], chunks: &[(u64, u64)]) -> super::coverage::Coverage {
    let r = |v: &[(u64, u64)]| v.iter().map(|&(a, b)| a..b).collect::<Vec<_>>();
    coverage(&r(gold), &r(chunks))
}

#[test]
fn span_coverage_by_hand() {
    // Evidence split across two chunks: covered, never contained.
    let c = cov(&[(10, 20)], &[(0, 15), (15, 30)]);
    assert_eq!((c.iou, c.contained, c.redundancy), (1.0 / 3.0, 0.0, 1.0));
    let c = cov(&[(10, 20)], &[(5, 25)]);
    assert_eq!((c.iou, c.contained, c.redundancy), (0.5, 1.0, 1.0));
    // Overlapping chunks retrieve [10, 20) twice.
    let c = cov(&[(10, 20), (12, 14)], &[(0, 20), (10, 30)]);
    assert_eq!((c.iou, c.contained, c.redundancy), (10.0 / 30.0, 1.0, 40.0 / 30.0));
    // Nothing retrieved, nothing to find.
    assert_eq!(cov(&[(3, 9)], &[]).iou, 0.0);
    let c = cov(&[], &[]);
    assert_eq!((c.iou, c.contained, c.redundancy), (1.0, 1.0, 1.0));
    // A chunk that starts before the evidence but ends short of it, beside
    // one that starts inside it: neither contains it.
    assert_eq!(cov(&[(10, 20)], &[(0, 18), (12, 25)]).contained, 0.0);
}

#[test]
fn a_byte_span_becomes_the_subwords_it_touches() {
    let mut subwords = Spans::<Subwords>::new(ViewId::SOURCE);
    for (a, b) in [(0, 3), (4, 9), (9, 11), (12, 16)] {
        subwords.push(Span::new(a, b));
    }
    assert_eq!(token_range(&subwords, Span::new(0, 3)), 0..1);
    assert_eq!(token_range(&subwords, Span::new(5, 10)), 1..3);
    assert_eq!(token_range(&subwords, Span::new(11, 12)), 3..3);
    assert_eq!(token_range(&subwords, Span::new(0, 16)), 0..4);
}

#[test]
fn budget_compliance_by_hand() {
    let b = Budget { max: 400, target: 300, size_weight: 1000, overhead: 2 };
    let r = BudgetReport::of([(100, false), (200, false), (300, false), (500, true), (2, false), (50, false)], b);
    assert_eq!((r.chunks, r.within_max, r.over_budget), (6, 5.0 / 6.0, 1.0 / 6.0));
    assert_eq!((r.empty, r.undersized), (1.0 / 6.0, 1.0 / 6.0));
    assert_eq!(r.mean_utilization, 1152.0 / 6.0 / 300.0);
    // The nearest rank of six is the sixth.
    assert_eq!((r.p95_utilization, r.max_utilization), (500.0 / 300.0, 500.0 / 300.0));
    let r = BudgetReport::of((1..=100).map(|t| (t, false)), Budget { target: 100, ..b });
    assert_eq!((r.p95_utilization, r.within_max), (0.95, 1.0));
    assert_eq!(BudgetReport::of([], b).chunks, 0);
}

#[test]
fn a_protected_span_is_parted_only_by_a_cut_inside_it() {
    let p = |kind, s, e| Protected { kind, sentences: s..e };
    let spans =
        [p(Protect::Structure, 2, 5), p(Protect::Coref, 0, 3), p(Protect::Clause, 5, 7), p(Protect::Coref, 6, 7)];
    let cuts = Segmentation::new(8, vec![3, 5]).unwrap();
    let kept = Preservation::of(&spans, &cuts);
    // [2, 5) is cut at 3; [0, 3) ends at the cut; [5, 7) starts at one.
    assert_eq!(kept.structure, Kept { total: 1, intact: 0 });
    assert_eq!(kept.coref, Kept { total: 2, intact: 2 });
    assert_eq!(kept.clause, Kept { total: 1, intact: 1 });
    assert_eq!(kept.rate(), 0.75);
    assert_eq!(Preservation::default().rate(), 1.0);
}

#[test]
fn code_blocks_tables_and_quotes_are_protected() {
    let mut spans = Spans::new(ViewId::SOURCE);
    for k in 0..6 {
        spans.push(Span::new(10 * k, 10 * k + 9));
    }
    let sentences = Sentences { spans };
    let mut blocks = Spans::new(ViewId::SOURCE);
    let kinds = [(0, 59, Block::Section), (12, 35, Block::Code), (40, 49, Block::Table), (40, 59, Block::Paragraph)];
    for &(a, b, _) in &kinds {
        blocks.push(Span::new(a, b));
    }
    let st = Structure {
        spans: blocks,
        kind: kinds.iter().map(|k| k.2).collect(),
        parent: vec![None, Some(Ix::new(0)), Some(Ix::new(0)), Some(Ix::new(0))],
        depth: vec![0; 4],
    };
    // The code block has bytes of sentences 1 to 3; the table holds one
    // sentence and has no gap to part; a paragraph is not protected.
    assert_eq!(structure_protected(&sentences, &st), [Protected { kind: Protect::Structure, sentences: 1..4 }]);
}

#[test]
fn coref_costs_charge_the_gaps_a_chain_spans() {
    let chains = [IxRange::new(0, 3), IxRange::new(1, 2), IxRange::new(2, 5), IxRange::new(3, 9)];
    let b = coref_costs(&chains, 6, 500);
    // [0, 3) spans gaps 0 and 1, [2, 5) gaps 2 and 3, [3, 9) is cut to
    // [3, 6): gaps 3 and 4; a one-sentence chain spans none.
    assert_eq!(b.cost, [500, 500, 500, 1000, 500, 0]);
    assert_eq!(b.reason, [Cut::Sentence; 6]);
    assert_eq!(coref_costs(&[IxRange::new(0, 4)], 4, i32::MAX).cost, [i32::MAX, i32::MAX, i32::MAX, 0]);
}
