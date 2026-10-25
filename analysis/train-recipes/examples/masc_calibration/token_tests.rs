use super::*;
use std::sync::Arc;
use turbo_formats::{
    NameSample,
    ner::{GermEvalLayer, GermEvalReader, GermEvalTypes},
};
use turbo_namefind::calibrated::identity;
use turbo_train::{
    Key, TrainingConfiguration, TrainingParameters,
    calibration::{Method, Policy, Samples, fit},
};
use turbo_train_components::{
    namefind::{TokenNameFinderFactory, train_name_model},
    package::Stamp,
};

#[test]
fn later_real_corpus_sentences_match_fresh_adapter_calls() {
    // Keep GermEval's real words and labels, grouping the samples into one
    // adaptive stream to exercise later sentences with clear=false. This is
    // a state-protocol regression, not an accuracy/coverage measurement.
    let raw = include_str!("../../../goldens/data/inputs/ner/germeval-excerpt.txt");
    let samples: Vec<_> = GermEvalReader::new(raw, GermEvalTypes::ALL, GermEvalLayer::Outer)
        .unwrap()
        .enumerate()
        .map(|(i, s)| {
            let s = s.unwrap();
            NameSample::new(s.tokens().to_vec(), s.names().to_vec(), i == 0, s.source(), 0).unwrap()
        })
        .collect();
    assert_eq!(samples.len(), 105);
    assert!(samples.iter().skip(1).all(|s| !s.clear_adaptive_data()));
    let factory = TokenNameFinderFactory::new("de", None, NameCodec::Bio).unwrap();
    let mut params = TrainingParameters::new();
    params.put(Key::ITERATIONS, 50);
    params.put(Key::CUTOFF, 0);
    let trained =
        train_name_model(&samples, &factory, None, &params, &mut TrainingConfiguration::default(), Stamp::at(0))
            .unwrap();
    let file = Arc::new(ModelFile::open(trained.package.to_bytes()).unwrap());
    let id = identity(&file, "germeval-state-regression").unwrap();
    let (rows, gold) = collect(&file, &id.labels, &samples).unwrap();
    assert_eq!(rows.len(), 2022);
    let mid = rows.len() / 2;
    let cal = fit(
        id.clone(),
        Method::Temperature,
        Samples { split: "regression-fit", probabilities: &rows[..mid], gold: &gold[..mid] },
        Samples { split: "regression-conf", probabilities: &rows[mid..], gold: &gold[mid..] },
    )
    .unwrap();
    let ann = CalibratedNameAnnotator::new(
        file,
        cal,
        "germeval-state-regression",
        Policy { coverage: 0.90, max_set_size: id.labels.len(), min_confidence: 0.0 },
    )
    .unwrap();
    let mut mismatches = 0;
    let count = visit_sidecars(&ann, &samples, |i, row| {
        mismatches += usize::from(row.raw != rows[i]);
        Ok(())
    })
    .unwrap();
    assert_eq!(count, rows.len());
    assert_eq!(mismatches, 0, "later sentences carried adaptive state across sentence-scoped adapter calls");
}
