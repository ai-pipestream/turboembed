//! Port of OpenNLP's `CorefScorerTest`: the worked example of Pradhan et
//! al. (ACL 2014), section 4, corpus accumulation and refusals.

use std::collections::HashSet;

use turbo_coref::scorer::{CorefScorer, CorefScores, Score, maximum_assignment};
use turbo_jvm::random::JavaRandom;

const TOLERANCE: f64 = 1e-9;

fn set(items: &[&'static str]) -> HashSet<&'static str> {
    items.iter().copied().collect()
}

fn key() -> Vec<HashSet<&'static str>> {
    vec![set(&["a", "b", "c"]), set(&["d", "e", "f", "g"])]
}

fn response() -> Vec<HashSet<&'static str>> {
    vec![set(&["a", "b"]), set(&["c", "d"]), set(&["f", "g", "h", "i"])]
}

fn example() -> CorefScores {
    let mut scorer = CorefScorer::new();
    scorer.add(&key(), &response()).unwrap();
    scorer.scores()
}

fn close(a: f64, b: f64, tolerance: f64) {
    assert!((a - b).abs() <= tolerance, "{a} != {b}");
}

#[test]
fn muc_matches_reference_example() {
    let muc = example().muc();
    close(muc.recall(), 0.40, TOLERANCE);
    close(muc.precision(), 0.40, TOLERANCE);
    close(muc.f1(), 0.40, TOLERANCE);
}

#[test]
fn b_cubed_matches_reference_example() {
    let b = example().b_cubed();
    close(b.recall(), 35.0 / 84.0, TOLERANCE);
    close(b.precision(), 0.50, TOLERANCE);
    close(b.f1(), 2.0 * (35.0 / 84.0) * 0.5 / (35.0 / 84.0 + 0.5), TOLERANCE);
}

#[test]
fn ceaf_m_and_ceaf_e_match_reference_example() {
    let s = example();
    close(s.ceaf_m().recall(), 4.0 / 7.0, TOLERANCE);
    close(s.ceaf_m().precision(), 0.50, TOLERANCE);
    close(s.ceaf_e().recall(), 0.65, TOLERANCE);
    close(s.ceaf_e().precision(), 1.3 / 3.0, TOLERANCE);
    close(s.ceaf_e().f1(), 0.52, 0.005);
}

#[test]
fn mention_detection_and_conll_average() {
    let s = example();
    close(s.mentions().recall(), 6.0 / 7.0, TOLERANCE);
    close(s.mentions().precision(), 6.0 / 8.0, TOLERANCE);
    close(s.conll(), (s.muc().f1() + s.b_cubed().f1() + s.ceaf_e().f1()) / 3.0, TOLERANCE);
}

#[test]
fn identical_partitions_score_perfectly() {
    let mut scorer = CorefScorer::new();
    scorer.add(&key(), &key()).unwrap();
    let s = scorer.scores();
    for f1 in [s.muc().f1(), s.b_cubed().f1(), s.ceaf_m().f1(), s.ceaf_e().f1(), s.mentions().f1(), s.conll()] {
        close(f1, 1.0, TOLERANCE);
    }
}

#[test]
fn singletons_carry_no_muc_links_but_count_elsewhere() {
    let mut scorer = CorefScorer::new();
    scorer.add(&[set(&["a"])], &[set(&["a"])]).unwrap();
    let s = scorer.scores();
    close(s.muc().f1(), 0.0, TOLERANCE);
    close(s.b_cubed().f1(), 1.0, TOLERANCE);
    close(s.ceaf_e().f1(), 1.0, TOLERANCE);
}

#[test]
fn empty_response_and_nothing_added_score_zero_not_nan() {
    let mut scorer = CorefScorer::new();
    scorer.add(&key(), &[]).unwrap();
    let s = scorer.scores();
    close(s.muc().precision(), 0.0, TOLERANCE);
    close(s.b_cubed().precision(), 0.0, TOLERANCE);
    close(s.ceaf_e().precision(), 0.0, TOLERANCE);
    close(s.conll(), 0.0, TOLERANCE);
    close(CorefScorer::new().scores().conll(), 0.0, TOLERANCE);
}

#[test]
fn corpus_scores_sum_numerators_across_documents() {
    let mut scorer = CorefScorer::new();
    scorer.add(&key(), &response()).unwrap();
    scorer.add(&[set(&["x", "y"])], &[set(&["x", "y"])]).unwrap();
    close(scorer.scores().muc().precision(), 3.0 / 6.0, TOLERANCE);
    close(scorer.scores().muc().recall(), 3.0 / 6.0, TOLERANCE);
}

#[test]
fn rejects_invalid_partitions() {
    let mut scorer = CorefScorer::new();
    assert!(scorer.add(&[HashSet::new()], &response()).is_err());
    assert!(scorer.add(&[set(&["a", "b"]), set(&["b", "c"])], &response()).is_err());
    assert!(scorer.add(&key(), &[HashSet::new()]).is_err());
}

fn best_by_permutation(similarity: &[Vec<f64>], used: &mut [bool], row: usize) -> f64 {
    if row == similarity.len() {
        return 0.0;
    }
    let mut best = f64::NEG_INFINITY;
    for j in 0..similarity.len() {
        if !used[j] {
            used[j] = true;
            best = best.max(similarity[row][j] + best_by_permutation(similarity, used, row + 1));
            used[j] = false;
        }
    }
    best
}

#[test]
fn assignment_matches_exhaustive_search() {
    let mut random = JavaRandom::new(17);
    for _ in 0..200 {
        let n = 1 + random.next_int_bounded(6) as usize;
        let similarity: Vec<Vec<f64>> =
            (0..n).map(|_| (0..n).map(|_| random.next_int_bounded(5) as f64 / 4.0).collect()).collect();
        let exhaustive = best_by_permutation(&similarity, &mut vec![false; n], 0);
        close(maximum_assignment(&similarity), exhaustive, TOLERANCE);
    }
}

#[test]
fn score_rejects_values_outside_unit_interval_and_inconsistent_derived_values() {
    assert!(Score::new(1.5, 0.0, 0.0).is_err());
    assert!(Score::new(0.0, f64::NAN, 0.0).is_err());
    assert!(Score::new(1.0, 1.0, 0.0).is_err());
    let perfect = Score::new(1.0, 1.0, 1.0).unwrap();
    assert!(CorefScores::new(perfect, perfect, perfect, perfect, perfect, 0.0).is_err());
    assert!(CorefScores::new(perfect, perfect, perfect, perfect, perfect, 1.0).is_ok());
}
