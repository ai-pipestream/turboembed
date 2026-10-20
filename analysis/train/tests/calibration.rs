use turbo_calibration::{digest, metrics};
use turbo_train::calibration::*;
fn identity() -> Identity {
    Identity {
        model: "numeric-proof".into(),
        revision: "1".into(),
        sha256: digest(b"proof"),
        task: Task::Categorization,
        domain: "independent-splits".into(),
        labels: vec!["a".into(), "b".into()],
        inference_hash: 0,
    }
}
#[test]
fn heldout_fitting_reduces_ece_and_test_sets_cover() {
    // Independent deterministic draws from a known overconfident binary
    // source. Distinct seeds for fitting, conformal and test samples.
    fn data(mut seed: u64, n: usize) -> (Vec<Vec<f64>>, Vec<usize>) {
        let mut rows = Vec::new();
        let mut gold = Vec::new();
        for i in 0..n {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let confidence = 0.5 + 0.49 * ((seed >> 32) as f64 / u32::MAX as f64);
            let raw = 1.0 / (1.0 + (-3.0 * (confidence / (1.0 - confidence)).ln()).exp());
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let wrong = usize::from((seed >> 32) as f64 / u32::MAX as f64 >= confidence);
            let flip = i % 2;
            rows.push(if flip == 0 { vec![raw, 1.0 - raw] } else { vec![1.0 - raw, raw] });
            gold.push(wrong ^ flip);
        }
        (rows, gold)
    }
    let (fit_rows, fit_gold) = data(1, 1200);
    let (conf_rows, conf_gold) = data(123, 1201);
    let (test, test_gold) = data(998, 4000);
    for method in [Method::Temperature, Method::Platt, Method::Isotonic] {
        let a = fit(
            identity(),
            method,
            Samples { split: "fit", probabilities: &fit_rows, gold: &fit_gold },
            Samples { split: "conformal", probabilities: &conf_rows, gold: &conf_gold },
        )
        .unwrap();
        let before = metrics::reliability(&test, &test_gold, 10).unwrap();
        let p = Predictor::new(a.clone(), &identity(), Policy { coverage: 0.9, max_set_size: 2, min_confidence: 0.0 })
            .unwrap();
        let predictions: Vec<_> = test.iter().map(|r| p.predict(r).unwrap()).collect();
        let after =
            metrics::reliability(&predictions.iter().map(|r| r.calibrated.clone()).collect::<Vec<_>>(), &test_gold, 10)
                .unwrap();
        let coverage = metrics::coverage(&predictions, &test_gold).unwrap();
        assert!(after.ece < before.ece * 0.25, "{method:?}: {before:?} {after:?}");
        assert!((coverage.set_coverage - 0.9).abs() <= 0.04, "{coverage:?}");
        println!(
            "{method:?}: before ECE {:.6}, after {:.6}, Brier {:.6}, coverage {:.6}, artifact {}",
            before.ece,
            after.ece,
            after.brier,
            coverage.set_coverage,
            a.id()
        );
        let again = fit(
            identity(),
            method,
            Samples { split: "fit", probabilities: &fit_rows, gold: &fit_gold },
            Samples { split: "conformal", probabilities: &conf_rows, gold: &conf_gold },
        )
        .unwrap();
        assert_eq!(a.to_bytes(), again.to_bytes());
    }
}
#[test]
fn fitting_rejects_bad_rows_and_split_reuse() {
    let rows = vec![vec![0.8, 0.2], vec![0.3, 0.7]];
    let gold = vec![0, 1];
    assert!(
        fit(
            identity(),
            Method::Temperature,
            Samples { split: "same", probabilities: &rows, gold: &gold },
            Samples { split: "same", probabilities: &rows, gold: &gold }
        )
        .is_err()
    );
    assert!(
        fit_transform(Method::Isotonic, &Samples { split: "bad", probabilities: &[vec![f64::NAN, 0.5]], gold: &[0] })
            .is_err()
    );
    assert!(fit_transform(Method::Platt, &Samples { split: "bad", probabilities: &rows, gold: &[2, 0] }).is_err());
}
#[test]
fn isotonic_ties_are_grouped_before_pooling() {
    let rows = vec![vec![0.2, 0.8]; 3];
    let a = [0, 1, 1];
    let b = [1, 0, 1];
    let fit = |gold: &[usize]| {
        fit_transform(Method::Isotonic, &Samples { split: "fit", probabilities: &rows, gold }).unwrap()
    };
    assert_eq!(fit(&a), fit(&b));
}
