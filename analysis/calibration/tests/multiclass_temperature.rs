use turbo_calibration::Transform;

#[test]
fn square_root_temperature_matches_log_space_reference() {
    for p in [vec![0.0, 0.25, 0.75], vec![1e-310, 0.0, 1.0], vec![1.0 / 7.0; 7]] {
        let max = p.iter().copied().fold(0.0f64, f64::max).ln();
        let mut expected: Vec<_> =
            p.iter().map(|&x| if x == 0.0 { 0.0 } else { ((x.ln() - max) / 2.0).exp() }).collect();
        let sum: f64 = expected.iter().sum();
        for x in &mut expected {
            *x /= sum;
        }
        let mut got = vec![0.0; p.len()];
        Transform::Temperature(2.0).apply(&p, &mut got).unwrap();
        for (a, b) in got.iter().zip(&expected) {
            assert!((a - b).abs() <= 1e-13 * b.abs().max(f64::MIN_POSITIVE), "{a} != {b}");
        }
    }
    for p in [vec![f64::NAN, 0.0, 1.0], vec![0.0; 3]] {
        assert!(Transform::Temperature(2.0).apply(&p, &mut [0.0; 3]).is_err());
    }
}
