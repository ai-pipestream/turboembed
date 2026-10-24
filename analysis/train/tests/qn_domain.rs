//! A differentiable objective on a bounded domain, without a model stub.
use turbo_train::{QnParams, qn::minimize_objective};
#[test]
fn rejected_trial_does_not_poison_a_successful_line_search() {
    // The first unit-length step leaves the domain. Armijo halves it
    // until the finite minimum at 0.25 is reached.
    let objective = |x: &[f64], g: &mut [f64]| {
        if x[0] >= 0.5 {
            return Err("outside objective domain".to_owned());
        }
        g[0] = 2.0 * (x[0] - 0.25);
        Ok((x[0] - 0.25).powi(2))
    };
    let params = QnParams { l1_cost: 0.0, l2_cost: 0.0, ..QnParams::default() };
    let (x, _) = minimize_objective(1, &params, &objective).unwrap();
    assert!((x[0] - 0.25).abs() < 1e-12);
    let invalid_origin = |x: &[f64], g: &mut [f64]| {
        g[0] = -1.0 / x[0];
        Ok(-x[0].ln())
    };
    assert!(minimize_objective(1, &params, &invalid_origin).is_err());
}

#[test]
fn unrecoverable_domain_error_is_returned() {
    let objective = |x: &[f64], g: &mut [f64]| {
        if x[0] != 0.0 {
            return Err("outside objective domain".to_owned());
        }
        g[0] = -1.0;
        Ok(1.0)
    };
    let params = QnParams { l1_cost: 0.0, l2_cost: 0.0, ..QnParams::default() };
    assert!(minimize_objective(1, &params, &objective).unwrap_err().contains("line search made no progress"));
}

#[test]
fn nonfinite_trial_gradient_can_recover() {
    let objective = |x: &[f64], g: &mut [f64]| {
        g[0] = if x[0] >= 0.5 { f64::NAN } else { 2.0 * (x[0] - 0.25) };
        Ok((x[0] - 0.25).powi(2))
    };
    let params = QnParams { l1_cost: 0.0, l2_cost: 0.0, ..QnParams::default() };
    let (x, _) = minimize_objective(1, &params, &objective).unwrap();
    assert!((x[0] - 0.25).abs() < 1e-12);
}
