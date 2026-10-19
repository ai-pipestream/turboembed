//! `CrfScorer` boundaries: dimension mismatches, empty tag sets, out-of-range gold indices,
//! and gradient accumulators that do not match the scored shape fail with typed errors,
//! never panic, and never write into a caller buffer before failing.

use turbo_postag::neural::NeuralError;
use turbo_postag::neural::crf::CrfScorer;

// ---- CrfScorer dimensions ---------------------------------------------------

type CrfInputs = (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>, Vec<f64>);

fn crf_inputs() -> CrfInputs {
    let emissions = vec![vec![0.4, -0.3], vec![-0.2, 0.5]];
    let transitions = vec![vec![0.1, -0.5], vec![0.4, 0.2]];
    let start = vec![0.2, -0.1];
    let end = vec![-0.15, 0.25];
    (emissions, transitions, start, end)
}

#[test]
fn viterbi_rejects_ragged_or_mismatched_dimensions() {
    let (emissions, transitions, start, end) = crf_inputs();

    // An emission row short of the tag count.
    let mut bad = emissions.clone();
    bad[1] = vec![0.1];
    assert!(matches!(CrfScorer::viterbi(&bad, &transitions, &start, &end), Err(NeuralError::InvalidArgument(_))));
    assert!(matches!(CrfScorer::marginals(&bad, &transitions, &start, &end), Err(NeuralError::InvalidArgument(_))));

    // A transition row count or width that is not the tag count.
    let bad_tr = vec![vec![0.1, -0.5]];
    assert!(matches!(CrfScorer::viterbi(&emissions, &bad_tr, &start, &end), Err(NeuralError::InvalidArgument(_))));
    let bad_tr = vec![vec![0.1, -0.5, 0.0], vec![0.4, 0.2, 0.0]];
    assert!(matches!(CrfScorer::viterbi(&emissions, &bad_tr, &start, &end), Err(NeuralError::InvalidArgument(_))));

    // Start and end of a length other than the tag count.
    assert!(matches!(CrfScorer::viterbi(&emissions, &transitions, &[0.0], &end), Err(NeuralError::InvalidArgument(_))));
    assert!(matches!(
        CrfScorer::viterbi(&emissions, &transitions, &start, &[0.0, 0.0, 0.0]),
        Err(NeuralError::InvalidArgument(_))
    ));

    // No tags at all, with tokens to score: an error, not a made-up path.
    let two_rows = vec![vec![0.1], vec![0.2]];
    assert!(matches!(CrfScorer::viterbi(&two_rows, &[], &[], &[]), Err(NeuralError::InvalidArgument(_))));
    assert!(matches!(CrfScorer::marginals(&two_rows, &[], &[], &[]), Err(NeuralError::InvalidArgument(_))));

    // Empty input is still an empty path, and honest inputs still decode.
    assert_eq!(CrfScorer::viterbi(&[], &transitions, &start, &end).unwrap(), Vec::<usize>::new());
    assert_eq!(CrfScorer::viterbi(&emissions, &transitions, &start, &end).unwrap().len(), 2);
}

#[test]
fn viterbi_rejects_non_finite_scores() {
    let (emissions, transitions, start, end) = crf_inputs();
    let mut bad = emissions.clone();
    bad[0][1] = f64::NAN;
    assert!(matches!(CrfScorer::viterbi(&bad, &transitions, &start, &end), Err(NeuralError::NonFiniteScore(_))));
    let mut bad_tr = transitions.clone();
    bad_tr[0][0] = f64::INFINITY;
    assert!(matches!(CrfScorer::viterbi(&emissions, &bad_tr, &start, &end), Err(NeuralError::NonFiniteScore(_))));
}

#[test]
fn loss_and_gradients_rejects_bad_gold() {
    let (emissions, transitions, start, end) = crf_inputs();
    let d_emissions = vec![vec![0.0; 2]; 2];
    let d_transitions = vec![vec![0.0; 2]; 2];
    let d_start = vec![0.0; 2];
    let d_end = vec![0.0; 2];
    let run = |gold: &[usize]| {
        CrfScorer::loss_and_gradients(
            &emissions,
            &transitions,
            &start,
            &end,
            gold,
            &mut d_emissions.clone(),
            &mut d_transitions.clone(),
            &mut d_start.clone(),
            &mut d_end.clone(),
        )
    };
    // A gold sequence short of the token count.
    assert!(matches!(run(&[1]), Err(NeuralError::InvalidArgument(_))));
    // A gold tag outside the tag set.
    assert!(matches!(run(&[0, 2]), Err(NeuralError::InvalidArgument(_))));
    // A valid gold sequence still scores.
    assert!(run(&[0, 1]).is_ok());
}

// ---- CrfScorer gradient accumulators ----------------------------------------

/// Accumulators sized for the two-token, two-tag inputs of `crf_inputs`, filled with a
/// sentinel so a partial write before the error is visible.
type Accumulators = (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>, Vec<f64>);

const SENTINEL: f64 = 7.25;

fn accumulators() -> Accumulators {
    (vec![vec![SENTINEL; 2]; 2], vec![vec![SENTINEL; 2]; 2], vec![SENTINEL; 2], vec![SENTINEL; 2])
}

/// Runs `loss_and_gradients` on `acc` and asserts it fails with `InvalidArgument` and leaves
/// every remaining accumulator value at the sentinel.
fn assert_rejected_untouched(mut acc: Accumulators, case: &str) {
    let (emissions, transitions, start, end) = crf_inputs();
    let result = CrfScorer::loss_and_gradients(
        &emissions,
        &transitions,
        &start,
        &end,
        &[0, 1],
        &mut acc.0,
        &mut acc.1,
        &mut acc.2,
        &mut acc.3,
    );
    assert!(matches!(result, Err(NeuralError::InvalidArgument(_))), "{case}: {result:?}");
    let untouched = acc.0.iter().chain(acc.1.iter()).flatten().chain(acc.2.iter()).chain(acc.3.iter());
    assert!(untouched.copied().all(|v| v == SENTINEL), "{case}: a caller buffer was written before the error");
}

#[test]
fn loss_and_gradients_rejects_short_emission_accumulators_without_writing() {
    let mut acc = accumulators();
    acc.0.pop();
    assert_rejected_untouched(acc, "d_emissions short of the token count");
    let mut acc = accumulators();
    acc.0[1].pop();
    assert_rejected_untouched(acc, "d_emissions row short of the tag count");
    let mut acc = accumulators();
    acc.0[0].push(SENTINEL);
    assert_rejected_untouched(acc, "d_emissions row wider than the tag count");
}

#[test]
fn loss_and_gradients_rejects_short_transition_accumulators_without_writing() {
    let mut acc = accumulators();
    acc.1[0].pop();
    assert_rejected_untouched(acc, "d_transitions row short of the tag count");
    let mut acc = accumulators();
    acc.1.pop();
    assert_rejected_untouched(acc, "d_transitions short of the tag count");
}

#[test]
fn loss_and_gradients_rejects_short_start_and_end_accumulators_without_writing() {
    let mut acc = accumulators();
    acc.2.pop();
    assert_rejected_untouched(acc, "d_start short of the tag count");
    let mut acc = accumulators();
    acc.3.pop();
    assert_rejected_untouched(acc, "d_end short of the tag count");
}

#[test]
fn loss_and_gradients_accumulates_into_well_formed_buffers() {
    let (emissions, transitions, start, end) = crf_inputs();
    let (mut de, mut dt, mut ds, mut dn) = accumulators();
    let loss = CrfScorer::loss_and_gradients(
        &emissions,
        &transitions,
        &start,
        &end,
        &[0, 1],
        &mut de,
        &mut dt,
        &mut ds,
        &mut dn,
    )
    .unwrap();
    assert!(loss.is_finite() && loss > 0.0);
    // Emission gradients per token sum to zero (marginals sum to one, minus the gold one-hot).
    for row in &de {
        let shifted: f64 = row.iter().map(|v| v - SENTINEL).sum();
        assert!(shifted.abs() < 1e-9, "emission gradient row sums to {shifted}");
    }
}
