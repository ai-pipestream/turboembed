use turbo_calibration::*;
fn identity() -> Identity {
    Identity {
        model: "test-model".into(),
        revision: "1".into(),
        sha256: digest(b"model"),
        task: Task::Categorization,
        domain: "held-out".into(),
        labels: vec!["a".into(), "b".into()],
        inference_hash: 3,
    }
}
fn artifact(t: Transform, scores: Vec<f64>) -> Calibration {
    Calibration::new(
        identity(),
        t,
        DataIdentity { split: "fit".into(), sha256: digest(b"fit") },
        DataIdentity { split: "conformal".into(), sha256: digest(b"conformal") },
        scores,
    )
    .unwrap()
}
fn policy() -> Policy {
    Policy { coverage: 0.8, max_set_size: 2, min_confidence: 0.0 }
}
#[test]
fn transforms_are_finite_at_endpoints_and_extremes() {
    for t in [
        Transform::Temperature(1e-6),
        Transform::Temperature(1e6),
        Transform::Platt { slope: 1e6, intercept: -1e6 },
        Transform::Isotonic { upper: vec![0.0, 1.0], value: vec![0.0, 0.0] },
    ] {
        for p in [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [1e-300, 1.0]] {
            let mut out = [0.0; 2];
            t.apply(&p, &mut out).unwrap();
            check_distribution(&out).unwrap();
        }
    }
    let mut out = [0.0; 2];
    Transform::Temperature(2.0).apply(&[0.9, 0.1], &mut out).unwrap();
    assert!((out[0] - 0.75).abs() < 1e-14);
}
#[test]
fn bad_parameters_and_inputs_fail() {
    for t in [
        Transform::Temperature(0.0),
        Transform::Temperature(f64::NAN),
        Transform::Platt { slope: f64::INFINITY, intercept: 0.0 },
        Transform::Isotonic { upper: vec![0.5, 0.5], value: vec![0.0, 1.0] },
        Transform::Isotonic { upper: vec![0.0, 1.0], value: vec![1.0, 0.0] },
    ] {
        assert!(t.check().is_err());
    }
    for p in [[f64::NAN, 1.0], [-0.1, 1.1], [0.2, 0.2], [f64::INFINITY, 0.0]] {
        assert!(Transform::Temperature(1.0).apply(&p, &mut [0.0; 2]).is_err());
    }
    assert!(Transform::Temperature(1.0).apply(&[0.5, 0.5], &mut [0.0; 1]).is_err());
    for p in [
        Policy { coverage: 1.0, ..policy() },
        Policy { coverage: f64::NAN, ..policy() },
        Policy { max_set_size: 0, ..policy() },
        Policy { min_confidence: f64::INFINITY, ..policy() },
    ] {
        assert!(p.check().is_err());
    }
}
#[test]
fn conformal_extra_rank_and_abstention_reasons() {
    let a = artifact(Transform::Temperature(1.0), vec![0.1]);
    let p = Predictor::new(a, &identity(), Policy { max_set_size: 1, ..policy() }).unwrap();
    let r = p.predict(&[1.0, 0.0]).unwrap();
    assert_eq!(r.set, vec![0, 1]);
    assert_eq!(r.abstention, Some(Abstention::SetTooLarge));
    let a = artifact(Transform::Temperature(1.0), vec![0.1; 20]);
    let p = Predictor::new(a.clone(), &identity(), policy()).unwrap();
    assert_eq!(p.predict(&[0.5, 0.5]).unwrap().abstention, Some(Abstention::EmptySet));
    let q = Predictor::new(a, &identity(), Policy { min_confidence: 0.99, ..policy() }).unwrap();
    assert_eq!(q.predict(&[0.95, 0.05]).unwrap().abstention, Some(Abstention::BelowThreshold));
    assert_ne!(p.config_hash(), q.config_hash());
}
#[test]
fn artifact_roundtrip_identity_and_malformed_counts() {
    for t in [
        Transform::Temperature(2.0),
        Transform::Platt { slope: 0.5, intercept: 0.2 },
        Transform::Isotonic { upper: vec![0.2, 0.9], value: vec![0.1, 0.8] },
    ] {
        let a = artifact(t, vec![0.1, 0.2, 0.4]);
        let bytes = a.to_bytes();
        let b = Calibration::from_bytes(&bytes, &identity()).unwrap();
        assert_eq!(a, b);
        assert_eq!(bytes, b.to_bytes());
        assert_eq!(a.id(), b.id());
        for n in 0..bytes.len() {
            assert!(Calibration::from_bytes(&bytes[..n], &identity()).is_err(), "{n}");
        }
        let mut extra = bytes.clone();
        extra.push(0);
        assert!(Calibration::from_bytes(&extra, &identity()).is_err());
        let mut count = bytes.clone();
        count[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(Calibration::from_bytes(&count, &identity()).is_err());
        for i in 0..7 {
            let mut bad = identity();
            match i {
                0 => bad.model.push('x'),
                1 => bad.revision.push('x'),
                2 => bad.sha256 = digest(b"other"),
                3 => bad.domain.push('x'),
                4 => bad.labels.reverse(),
                5 => bad.task = Task::TokenTagging,
                _ => bad.inference_hash += 1,
            };
            let e = Calibration::from_bytes(&bytes, &bad).unwrap_err();
            assert!(e.contains(&a.id()));
            assert!(e.contains("mismatch"));
        }
    }
}
#[test]
fn metrics_are_known_and_do_not_hide_abstention() {
    let m = metrics::reliability(&[vec![0.8, 0.2], vec![0.4, 0.6]], &[0, 0], 10).unwrap();
    assert!((m.ece - 0.4).abs() < 1e-14);
    assert!((m.brier - 0.4).abs() < 1e-14);
    assert_eq!(m.accuracy, 0.5);
    let p = Predictor::new(
        artifact(Transform::Temperature(1.0), vec![0.1]),
        &identity(),
        Policy { max_set_size: 1, ..policy() },
    )
    .unwrap();
    let c = metrics::coverage(&[p.predict(&[0.8, 0.2]).unwrap()], &[0]).unwrap();
    assert_eq!(c.set_coverage, 1.0);
    assert_eq!(c.delivered_coverage, 0.0);
    assert_eq!(c.abstention_rate, 1.0);
}

#[test]
fn prepared_tables_validate_each_input_and_owned_predictions_match() {
    let t = Transform::Isotonic { upper: vec![0.2, 0.8], value: vec![0.1, 0.9] };
    let prepared = t.prepare().unwrap();
    let mut a = [0.0; 2];
    let mut b = a;
    prepared.apply(&[0.3, 0.7], &mut a).unwrap();
    t.apply(&[0.3, 0.7], &mut b).unwrap();
    assert_eq!(a, b);
    assert!(prepared.apply(&[f64::NAN, 0.5], &mut a).is_err());
    let p = Predictor::new(artifact(t, vec![0.2; 20]), &identity(), policy()).unwrap();
    assert_eq!(p.predict(&[0.3, 0.7]).unwrap(), p.predict_owned(vec![0.3, 0.7]).unwrap());
}

#[test]
fn binary_temperature_matches_log_space_reference() {
    for t in [1e-6, 0.13, 1.0, 2.0, 13.0, 1e6] {
        for p in [0.0, f64::from_bits(1), 1e-300, 0.001, 0.13, 0.5, 0.9, 1.0] {
            let raw = [p, 1.0 - p];
            let max = raw[0].max(raw[1]).ln();
            let weights = raw.map(|v| if v == 0.0 { 0.0 } else { ((v.ln() - max) / t).exp() });
            let sum = weights.iter().sum::<f64>();
            let expected = weights.map(|v| v / sum);
            let mut actual = [0.0; 2];
            Transform::Temperature(t).apply(&raw, &mut actual).unwrap();
            for i in 0..2 {
                assert!((actual[i] - expected[i]).abs() < 1e-13, "t={t}, p={p}: {actual:?} / {expected:?}");
            }
        }
    }
}
#[test]
fn scratch_predictions_reuse_storage_and_reject_bad_dimensions() {
    for transform in [
        Transform::Temperature(2.0),
        Transform::Platt { slope: 0.7, intercept: 0.2 },
        Transform::Isotonic { upper: vec![0.2, 0.7, 1.0], value: vec![0.1, 0.4, 0.9] },
    ] {
        let p = Predictor::new(artifact(transform, vec![0.6; 20]), &identity(), policy()).unwrap();
        let mut scores = [0.0; 2];
        let mut set = [99; 2];
        for raw in [[0.1, 0.9], [0.8, 0.2], [0.0, 1.0]] {
            let expected = p.predict(&raw).unwrap();
            let (n, reason) = p.predict_into(&raw, &mut scores, &mut set).unwrap();
            assert_eq!(scores.as_slice(), expected.calibrated);
            assert_eq!(&set[..n], expected.set);
            assert_eq!(reason, expected.abstention);
        }
        assert!(p.predict_into(&[0.5, 0.5], &mut scores, &mut []).is_err());
        assert!(p.predict_into(&[0.5, 0.5], &mut [], &mut set).is_err());
        assert!(p.predict_into(&[1.0], &mut scores, &mut set).is_err());
        assert!(p.predict_into(&[f64::NAN, 0.5], &mut scores, &mut set).is_err());
    }
}
