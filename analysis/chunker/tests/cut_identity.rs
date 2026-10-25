//! DP runner-up receipts and cross-backend partition comparison.
use turbo_analysis::{Annotator, ModelRef, Span, Spans, ViewId};
use turbo_analysis_layers::{Block, Boundaries, Chunks, Cut, Sentences, Structure, Subwords};
use turbo_chunker::eval::compare_chunk_cuts;
use turbo_chunker::{Budget, Chunker, no_term, segment, segment_with_trace};

fn budget(max: u32) -> Budget {
    Budget { max, target: 2, size_weight: 0, overhead: 0 }
}
fn spans(pairs: &[(u32, u32)]) -> Spans<Chunks> {
    let mut s = Spans::new(ViewId::SOURCE);
    for &(start, end) in pairs {
        s.push(Span::new(start, end));
    }
    s
}

#[test]
fn ties_choose_earliest_unit_and_record_zero_with_no_alternative_at_end() {
    let t = segment_with_trace(&[0; 4], |i, j| (j - i) as u32, budget(4), no_term);
    assert_eq!(t.segments.iter().map(|s| s.end).collect::<Vec<_>>(), [1, 2, 3, 4]);
    assert_eq!(t.decisions.iter().map(|d| d.margin()).collect::<Vec<_>>(), [Some(0), Some(0), Some(0), None]);
    assert!(segment_with_trace(&[], |_, _| 0, budget(1), no_term).segments.is_empty());
}

// Exhaustively price all suffix partitions, independent of the DP. Group
// by the first cut: the second cheapest distinct first cut is the runner-up.
fn brute(i: usize, n: usize, cut: &[i32], max: usize) -> Vec<(i128, usize)> {
    (i + 1..=n.min(i + max))
        .map(|j| {
            let rest = if j == n { 0 } else { brute(j, n, cut, max).iter().map(|x| x.0).min().unwrap() };
            (rest + if j == n { 0 } else { i128::from(cut[j - 1]) }, j)
        })
        .collect()
}

#[test]
fn runner_up_is_the_next_cut_total_including_the_optimal_suffix() {
    for n in 1..=7 {
        for max in 1..=n {
            for seed in 0..21 {
                let costs: Vec<i32> = (0..n).map(|i| ((seed + 3 * i) % 7) as i32 - 3).collect();
                let t = segment_with_trace(&costs, |i, j| (j - i) as u32, budget(max as u32), no_term);
                assert_eq!(t.segments, segment(&costs, |i, j| (j - i) as u32, budget(max as u32), no_term));
                for (s, d) in t.segments.iter().zip(t.decisions) {
                    let mut alternatives = brute(s.start, n, &costs, max);
                    alternatives.sort();
                    assert_eq!((d.best_cost, s.end), alternatives[0]);
                    assert_eq!(d.second_best_cost, alternatives.get(1).map(|x| x.0));
                }
            }
        }
    }
}

#[test]
fn identical_empty_and_gapped_partitions() {
    let empty = spans(&[]);
    let r = compare_chunk_cuts(&empty, &[], &empty, &[], 0.).unwrap();
    assert!(r.is_identical());
    assert_eq!(r.agreement_rate, 1.);
    let a = spans(&[(0, 3), (4, 8), (9, 12)]);
    let r = compare_chunk_cuts(&a, &[Some(-3), Some(0), None], &a, &[None; 3], 0.).unwrap();
    assert_eq!(r.matching_cuts, 2);
    assert_eq!(r.total_chunks, [3, 3]);
    assert_eq!(r.agreement_rate, 1.);
    assert_eq!(r.first_differing_offset, None);
}

#[test]
fn small_margins_flag_precision_sensitivity_but_large_or_unknown_ones_do_not() {
    let a = spans(&[(0, 3), (4, 12)]);
    let b = spans(&[(0, 5), (6, 12)]);
    let r = compare_chunk_cuts(&a, &[Some(-2), None], &b, &[Some(-1), None], 2.).unwrap();
    assert_eq!(r.first_differing_offset, Some(3));
    assert_eq!(r.agreement_rate, 0.);
    assert_eq!(r.differences[0].a.unwrap().margin, Some(-2));
    assert_eq!(r.differences[0].b.unwrap().margin, Some(-1));
    assert!(r.all_within_epsilon());
    assert!(!r.is_identical());
    for margins in [[Some(-3), None], [None, None], [Some(i128::MIN), None]] {
        let r = compare_chunk_cuts(&a, &margins, &b, &[Some(-1), None], 2.).unwrap();
        assert!(!r.all_within_epsilon());
    }
    assert!(!compare_chunk_cuts(&a, &[Some(-2), None], &b, &[Some(0), None], 1.99).unwrap().all_within_epsilon());
}

#[test]
fn insertions_do_not_shift_later_shared_cuts_and_gaps_are_compared() {
    let a = spans(&[(0, 3), (4, 6), (7, 9), (10, 12)]);
    let b = spans(&[(0, 6), (7, 9), (10, 12)]);
    let r = compare_chunk_cuts(&a, &[Some(0); 4], &b, &[Some(0); 3], 1.).unwrap();
    assert_eq!(r.matching_cuts, 2);
    assert_eq!(r.total_cuts, 3);
    assert_eq!(r.differences.len(), 1);
    assert!(r.differences[0].b.is_none());
    assert!(!r.all_within_epsilon());
    let gap = spans(&[(0, 5), (7, 9), (10, 12)]);
    assert_eq!(compare_chunk_cuts(&b, &[None; 3], &gap, &[None; 3], 0.).unwrap().first_differing_offset, Some(5));
}

#[test]
fn invalid_records_and_epsilon_return_errors() {
    let a = spans(&[(0, 3), (4, 8)]);
    for e in [-1., f64::NAN, f64::INFINITY] {
        assert!(compare_chunk_cuts(&a, &[None; 2], &a, &[None; 2], e).is_err());
    }
    assert!(compare_chunk_cuts(&a, &[], &a, &[None; 2], 0.).is_err());
    assert!(compare_chunk_cuts(&a, &[Some(1), None], &a, &[None; 2], 0.).is_err());
    for bad in [spans(&[(0, 3), (2, 8)]), spans(&[(0, 0), (4, 8)]), spans(&[(0, 9)])] {
        assert!(compare_chunk_cuts(&a, &[None; 2], &bad, &vec![None; bad.len()], 0.).is_err());
    }
    let mut other = Spans::new(ViewId(1));
    other.push(Span::new(0, 8));
    assert!(compare_chunk_cuts(&a, &[None; 2], &other, &[None], 0.).is_err());
}

fn layers() -> (Sentences, Subwords, Boundaries) {
    let mut s = Spans::new(ViewId::SOURCE);
    let mut w = Spans::new(ViewId::SOURCE);
    for i in 0..4 {
        s.push(Span::new(i * 10, i * 10 + 9));
        w.push(Span::new(i * 10, i * 10 + 9));
    }
    (
        Sentences { spans: s },
        Subwords {
            spans: w,
            ids: vec![0; 4],
            token: None,
            model: ModelRef { id: "test".into(), revision: "1".into(), sha256: "0".repeat(64) },
        },
        Boundaries { cost: vec![0; 4], reason: vec![Cut::Sentence; 4] },
    )
}

#[test]
fn trace_matches_real_annotator_and_forced_sections_have_no_margin() {
    let (s, w, b) = layers();
    let c = Chunker { budget: budget(4) };
    let (normal,) = c.annotate((&s, &w, &b, None, None, None)).unwrap();
    let traced = c.chunk_with_trace(&s, &w, &b, None, None).unwrap();
    assert_eq!(normal.spans.columns(), traced.chunks().spans.columns());
    assert_eq!(normal.cost, traced.chunks().cost);
    let (flat, margins) = traced.leaf_cuts();
    assert_eq!(flat.columns().0, [0, 10, 20, 30]);
    assert_eq!(margins, [Some(0), Some(0), Some(0), None]);
    let mut sections = Spans::new(ViewId::SOURCE);
    sections.push(Span::new(0, 19));
    sections.push(Span::new(20, 39));
    let st = Structure { spans: sections, kind: vec![Block::Section; 2], parent: vec![None; 2], depth: vec![0; 2] };
    let traced = c.chunk_with_trace(&s, &w, &b, Some(&st), None).unwrap();
    assert_eq!(traced.decisions().len(), 6);
    assert_eq!(traced.leaf_cuts().1, [Some(0), None, Some(0), None]);
}

#[test]
fn clause_fallbacks_do_not_inherit_a_sentence_margin() {
    use turbo_chunker::{ClauseBreak, ClauseCuts, SentenceClauseCuts};
    let mut ss = Spans::new(ViewId::SOURCE);
    ss.push(Span::new(0, 39));
    let (_, w, _) = layers();
    let s = Sentences { spans: ss };
    let b = Boundaries { cost: vec![0], reason: vec![Cut::Sentence] };
    let cl = ClauseCuts {
        sentences: vec![SentenceClauseCuts {
            tokens: (0..4).map(|i| Span::new(i * 10, i * 10 + 9)).collect(),
            breaks: vec![ClauseBreak::Token(0); 3],
        }],
    };
    let c = Chunker { budget: budget(1) };
    let traced = c.chunk_with_trace(&s, &w, &b, None, Some(&cl)).unwrap();
    assert_eq!(traced.chunks().spans.len(), 4);
    assert_eq!(traced.leaf_cuts().1, [None; 4]);
    assert!(traced.decisions().iter().all(Option::is_none));
}

#[test]
fn large_integer_margins_do_not_round_down_into_epsilon() {
    let a = spans(&[(0, 3), (4, 8)]);
    let b = spans(&[(0, 4), (5, 8)]);
    let exact = 1i128 << 54;
    let r = compare_chunk_cuts(&a, &[Some(-exact - 1), None], &b, &[Some(0), None], exact as f64).unwrap();
    assert!(!r.all_within_epsilon());
}
