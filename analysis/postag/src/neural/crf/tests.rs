use super::*;

#[test]
fn score_table_word_promotion_and_differences() {
    let mut table = CrfScoreTable::new(3);
    let mut exponents = [0i32; 4];
    let mut mantissas = [0i64; 4];

    let mut count = table.add_term(&mut exponents, &mut mantissas, 0, 0, 1e100).unwrap();
    count = table.add_term(&mut exponents, &mut mantissas, count, 0, 1e-100).unwrap();
    table.store(0, &exponents, &mantissas, count);

    count = table.add_term(&mut exponents, &mut mantissas, 0, 1, 1e100).unwrap();
    table.store(1, &exponents, &mantissas, count);

    assert_eq!(table.difference(0, 1), 1e-100);
    assert_eq!(table.difference(1, 0), -1e-100);
    assert_eq!(table.difference(0, 0), 0.0);

    table.copy_from(2, &table.clone(), 1);
    count = table.add_term(&mut exponents, &mut mantissas, 1, 2, -2.5e-100).unwrap();
    table.store_with_base(2, 2, &exponents, &mantissas, count);
    assert_eq!(table.difference(0, 2), 3.5e-100);
}

#[test]
fn score_table_rescaling_and_subnormal_cancel() {
    let mut first = CrfScoreTable::new(3);
    let mut exponents = [0i32; 4];
    let mut mantissas = [0i64; 4];

    let count = first.add_term(&mut exponents, &mut mantissas, 0, 0, 8.0).unwrap();
    first.store(0, &exponents, &mantissas, count);
    let count = first.add_term(&mut exponents, &mut mantissas, 0, 1, 0.25).unwrap();
    first.store(1, &exponents, &mantissas, count);
    assert_eq!(first.difference(0, 1), 7.75);

    let mut second = CrfScoreTable::new(1);
    let count = second.add_term(&mut exponents, &mut mantissas, 0, 0, 1e-200).unwrap();
    second.store(0, &exponents, &mantissas, count);
    first.copy_from(2, &second, 0);
    assert_eq!(first.difference(0, 2), 8.0);
}

#[test]
fn viterbi_matches_brute_force_and_marginals_sum_to_one() {
    let emissions = vec![vec![0.4, -0.3, 0.1], vec![-0.2, 0.5, 0.2], vec![0.3, 0.1, -0.4], vec![-0.1, 0.2, 0.6]];
    let transitions = vec![vec![0.1, -0.5, 0.3], vec![0.4, 0.2, -0.2], vec![-0.3, 0.5, 0.1]];
    let start = vec![0.2, -0.1, 0.05];
    let end = vec![-0.15, 0.25, 0.1];

    let best = CrfScorer::viterbi(&emissions, &transitions, &start, &end).unwrap();
    let mut best_brute = vec![0usize; 4];
    let mut best_score = f64::NEG_INFINITY;
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                for d in 0..3 {
                    let s = start[a]
                        + emissions[0][a]
                        + transitions[a][b]
                        + emissions[1][b]
                        + transitions[b][c]
                        + emissions[2][c]
                        + transitions[c][d]
                        + emissions[3][d]
                        + end[d];
                    if s > best_score {
                        best_score = s;
                        best_brute = vec![a, b, c, d];
                    }
                }
            }
        }
    }
    assert_eq!(best, best_brute);

    let marginals = CrfScorer::marginals(&emissions, &transitions, &start, &end).unwrap();
    assert_eq!(marginals.len(), 4);
    for row in &marginals {
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }
}

#[test]
fn precision_cancellation_preserves_small_delta() {
    let large = CrfScoreTable::scalb(1.0, 100);
    let delta = CrfScoreTable::scalb(1.0, -20);
    let emissions = vec![vec![large, large], vec![delta, 0.0]];
    let transitions = vec![vec![0.0, -large], vec![-large, 0.0]];
    let start = vec![0.0, 0.0];
    let end = vec![0.0, 0.0];

    let path = CrfScorer::viterbi(&emissions, &transitions, &start, &end).unwrap();
    assert_eq!(path, vec![0, 0]);
    let marginals = CrfScorer::marginals(&emissions, &transitions, &start, &end).unwrap();
    let expected_first = 1.0 / (1.0 + (-delta).exp());
    assert!((marginals[0][0] - expected_first).abs() <= 2.0 * f64::EPSILON);
    assert!((marginals[1][0] - expected_first).abs() <= 2.0 * f64::EPSILON);
}
