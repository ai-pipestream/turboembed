use turbo_sequence::PatternMode;
use turbo_sequence::crf::CrfSequenceModel;
use turbo_sequence::namefind::{CrfNameFinder, DefaultNameContext, NameCodec};

#[test]
fn typed_and_untyped_span_marginals_equal_complete_path_enumeration() {
    for codec in [NameCodec::Bio, NameCodec::Bilou] {
        for prefix in ["", "person-"] {
            let suffixes =
                if codec == NameCodec::Bio { vec!["start", "cont"] } else { vec!["start", "cont", "last", "unit"] };
            let mut labels = vec!["other".to_owned()];
            labels.extend(suffixes.iter().map(|s| format!("{prefix}{s}")));
            let n = labels.len();
            let model = CrfSequenceModel::new(labels, vec![], codec, vec![0.0; n * n + 2 * n]).unwrap();
            for length in 1..=3 {
                let context = DefaultNameContext::new(vec![], PatternMode::Utf16Units);
                let mut finder = CrfNameFinder::new(&model, context);
                let paths = finder.best_sequences(&vec!["word"; length], &[], &[], 4096).unwrap();
                let total: f64 = paths.iter().map(|p| p.log_probability.exp()).sum();
                assert!((total - 1.0).abs() < 1e-12);
                for path in &paths {
                    for name in &path.names {
                        let expected = paths
                            .iter()
                            .filter(|p| p.names.iter().any(|n| n.span == name.span))
                            .map(|p| p.log_probability.exp())
                            .sum::<f64>();
                        assert!(
                            (name.prob - expected).abs() < 1e-12,
                            "{codec:?} {prefix} {:?}: {} != {expected}",
                            name.span,
                            name.prob
                        );
                    }
                }
                if codec == NameCodec::Bio && prefix.is_empty() && length == 2 {
                    let name =
                        paths.iter().flat_map(|p| &p.names).find(|n| n.span.start == 0 && n.span.end == 1).unwrap();
                    assert!((name.prob - 0.4).abs() < 1e-12);
                }
            }
        }
    }
}
