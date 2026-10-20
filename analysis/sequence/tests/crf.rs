use turbo_maxent::JavaString;
use turbo_sequence::crf::{CrfConstraints, CrfLattice, CrfScores, CrfSequenceModel};
use turbo_sequence::namefind::NameCodec;

fn labels(codec: NameCodec) -> Vec<String> {
    let names = if codec == NameCodec::Bio {
        vec!["other", "p-start", "p-cont"]
    } else {
        vec!["other", "p-start", "p-cont", "p-last", "p-unit"]
    };
    names.into_iter().map(str::to_owned).collect()
}
#[test]
fn lattice_and_k_best_equal_exhaustive_enumeration() {
    for codec in [NameCodec::Bio, NameCodec::Bilou] {
        let labels = labels(codec);
        let n = labels.len();
        let mask = CrfConstraints::new(&labels, codec).unwrap();
        for t in 1..=4 {
            let emissions: Vec<Vec<f64>> =
                (0..t).map(|i| (0..n).map(|y| ((i * n + y) as f64 * 0.7).sin()).collect()).collect();
            let trans: Vec<_> = (0..n * n).map(|i| (i as f64 * 0.3).cos()).collect();
            let start = vec![0.1; n];
            let end: Vec<_> = (0..n).map(|i| i as f64 * 0.13).collect();
            let lattice = CrfLattice::new(
                CrfScores { emissions: &emissions, transitions: &trans, start: &start, end: &end },
                &mask,
            )
            .unwrap();
            let mut paths = Vec::new();
            for mut code in 0..n.pow(t as u32) {
                let mut path = vec![0; t];
                for y in &mut path {
                    *y = code % n;
                    code /= n;
                }
                if !mask.accepts(&path) {
                    continue;
                }
                let mut score = start[path[0]] + end[path[t - 1]];
                for i in 0..t {
                    score += emissions[i][path[i]];
                    if i > 0 {
                        score += trans[path[i - 1] * n + path[i]];
                    }
                }
                paths.push((path, score));
            }
            let z = paths.iter().map(|(_, s)| s.exp()).sum::<f64>();
            assert!((lattice.log_partition().unwrap() - z.ln()).abs() < 1e-12);
            let marginal = lattice.marginals();
            for (i, row) in marginal.iter().enumerate() {
                for (y, &actual) in row.iter().enumerate() {
                    let expected = paths.iter().filter(|(p, _)| p[i] == y).map(|(_, s)| s.exp() / z).sum::<f64>();
                    assert!((actual - expected).abs() < 1e-12);
                }
            }
            paths.sort_by(|a, b| b.1.total_cmp(&a.1));
            let best = lattice.best_paths(20).unwrap();
            for (got, (_, s)) in best.iter().zip(&paths) {
                assert!(mask.accepts(&got.labels));
                assert!((got.log_probability - (s - z.ln())).abs() < 1e-12);
            }
            let selected = &paths[0].0;
            let expected = paths.iter().filter(|(p, _)| p[0] == selected[0]).map(|(_, s)| s.exp() / z).sum::<f64>();
            assert!((lattice.span_probability(0, &selected[..1], None).unwrap() - expected).abs() < 1e-12);
        }
    }
}
#[test]
fn masks_refuse_open_bilou_entities_and_invalid_types() {
    let mask = CrfConstraints::new(&labels(NameCodec::Bilou), NameCodec::Bilou).unwrap();
    assert!(!mask.accepts(&[1]));
    assert!(!mask.accepts(&[0, 2, 3]));
    assert!(mask.accepts(&[1, 2, 3]));
    let bio = CrfConstraints::new(&["p-start".into(), "q-cont".into()], NameCodec::Bio).unwrap();
    assert!(!bio.accepts(&[0, 1]));
    assert!(CrfConstraints::new(&["bad".into()], NameCodec::Bio).is_err());
}
#[test]
fn wire_round_trip_and_all_truncations() {
    let labels = labels(NameCodec::Bio);
    let model = CrfSequenceModel::new(labels, vec![JavaString::new("x")], NameCodec::Bio, vec![0.25; 18]).unwrap();
    let bytes = model.to_bytes();
    assert_eq!(CrfSequenceModel::from_bytes(&bytes).unwrap().to_bytes(), bytes);
    for i in 0..bytes.len() {
        assert!(CrfSequenceModel::from_bytes(&bytes[..i]).is_err(), "{i}");
    }
    let mut extra = bytes.clone();
    extra.push(0);
    assert!(CrfSequenceModel::from_bytes(&extra).is_err());
    let mut count = bytes.clone();
    count[10..14].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(CrfSequenceModel::from_bytes(&count).is_err());
    let mut nan = bytes.clone();
    let end = nan.len();
    nan[end - 8..].copy_from_slice(&f64::NAN.to_le_bytes());
    assert!(CrfSequenceModel::from_bytes(&nan).is_err());
}
#[test]
fn large_common_emissions_keep_probabilities_normalized() {
    let mask = CrfConstraints::new(&labels(NameCodec::Bio), NameCodec::Bio).unwrap();
    let e = vec![vec![1e300; 3]];
    let l =
        CrfLattice::new(CrfScores { emissions: &e, transitions: &[0.0; 9], start: &[0.0; 3], end: &[0.0; 3] }, &mask)
            .unwrap();
    assert_eq!(l.marginals()[0], vec![0.5, 0.5, 0.0]);
    assert!(l.loss_gradient(&[2]).is_err());
    assert!((l.loss_gradient(&[0]).unwrap().loss - 2f64.ln()).abs() < 1e-12);
}

#[test]
fn model_labels_must_round_trip_through_name_codec() {
    for terminator in ['\n', '\r', '\u{85}', '\u{2028}', '\u{2029}'] {
        let label = format!("person{terminator}-start");
        assert!(CrfConstraints::new(&["other".into(), label], NameCodec::Bio).is_err());
    }
}
