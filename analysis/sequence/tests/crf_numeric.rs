use turbo_sequence::crf::{CrfConstraints, CrfLattice, CrfScores};
use turbo_sequence::namefind::NameCodec;

#[test]
fn common_start_end_transition_offsets_do_not_change_probabilities() {
    let mask = CrfConstraints::new(&["other".into(), "p-unit".into()], NameCodec::Bilou).unwrap();
    let emissions = vec![vec![0.0; 2]; 3];
    let lattice = CrfLattice::new(
        CrfScores { emissions: &emissions, transitions: &[1e300; 4], start: &[1e300; 2], end: &[1e300; 2] },
        &mask,
    )
    .unwrap();
    for row in lattice.marginals() {
        assert!(row.iter().all(|p| (p - 0.5).abs() < 1e-12));
    }
    assert!((lattice.loss_gradient(&[0, 0, 0]).unwrap().loss - 8f64.ln()).abs() < 1e-12);
    assert!((lattice.span_probability(0, &[0, 0], None).unwrap() - 0.25).abs() < 1e-12);
    assert!((lattice.transition_marginal(1, 0, 1).unwrap() - 0.25).abs() < 1e-12);
    assert!((lattice.best_paths(1).unwrap()[0].log_probability + 8f64.ln()).abs() < 1e-12);
}
#[test]
fn unreachable_span_is_zero_even_with_large_finite_weights() {
    let mask = CrfConstraints::new(&["other".into(), "p-start".into(), "p-cont".into()], NameCodec::Bio).unwrap();
    let emissions = vec![vec![0.0; 3]; 2];
    let lattice = CrfLattice::new(
        CrfScores { emissions: &emissions, transitions: &[1e300; 9], start: &[1e300; 3], end: &[1e300; 3] },
        &mask,
    )
    .unwrap();
    assert_eq!(lattice.span_probability(0, &[2], None).unwrap(), 0.0);
}
