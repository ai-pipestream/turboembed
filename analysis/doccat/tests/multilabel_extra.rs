//! The cardinality bounds and the evaluation metrics of the multi-label
//! categorizer: `min_labels` back-filling the threshold emission (with
//! `min_score` a hard floor), the format's backward compatibility, and
//! micro/macro/sample precision, recall, and F1 plus exact match against
//! hand-computed values.

use std::collections::BTreeSet;

use turbo_doccat::multilabel::{
    FORMAT_VERSION_V1, LinearHead, MultiLabelCategorizer, MultiLabelDoccatModel, MultiLabelError, Prf,
    ThresholdCalibrator, evaluate, f_beta,
};

use turbo_doccat::{BagOfWordsFeatureGenerator, BuiltinFeatureGenerator, DoccatContextGenerator};

/// Three labels: `team` fires `sports`, `stock` fires `finance`, `rain`
/// fires `weather`.
fn model() -> MultiLabelDoccatModel {
    let heads = vec![
        LinearHead::new(-1.0, vec![("bow=team".into(), 4.0)]).unwrap(),
        LinearHead::new(-2.0, vec![("bow=stock".into(), 4.5)]).unwrap(),
        LinearHead::new(-1.5, vec![("bow=rain".into(), 3.0)]).unwrap(),
    ];
    MultiLabelDoccatModel::new(
        "en",
        &["sports", "finance", "weather"],
        heads,
        vec![0.5, 0.5, 0.5],
        DoccatContextGenerator::from_builtins(vec![BuiltinFeatureGenerator::BagOfWords(
            BagOfWordsFeatureGenerator::new(),
        )]),
    )
    .unwrap()
}

fn predicted(cat: &MultiLabelCategorizer, text: &[&str]) -> Vec<usize> {
    let scores = cat.categorize(text);
    cat.predicted_tags(&scores).iter().map(|t| t.0 as usize).collect()
}

#[test]
fn min_labels_back_fills_the_top_scorers() {
    // "team" alone clears only the sports threshold; min_labels 2 fills
    // the weather label, the next highest score.
    let cat = MultiLabelCategorizer::new(model().with_min_labels(Some(2)).unwrap());
    assert_eq!(predicted(&cat, &["team"]), [0, 2]);

    // Nothing clears at all: the two best scores emit anyway.
    let cat = MultiLabelCategorizer::new(model().with_min_labels(Some(2)).unwrap());
    assert_eq!(predicted(&cat, &[]), [0, 2]);

    // min_labels 1 emits the single best label.
    let cat = MultiLabelCategorizer::new(model().with_min_labels(Some(1)).unwrap());
    assert_eq!(predicted(&cat, &[]), [0]);

    // Where the thresholds emit enough, the fill does nothing.
    let cat = MultiLabelCategorizer::new(model().with_min_labels(Some(2)).unwrap());
    assert_eq!(predicted(&cat, &["team", "stock"]), [0, 1]);
}

#[test]
fn min_score_stays_a_hard_floor_under_a_forced_fill() {
    // With min_score 0.2 only sports passes; the fill cannot use finance
    // (0.119) or weather (0.182), so the item keeps one label.
    let model = model().with_min_score(0.2).unwrap().with_min_labels(Some(2)).unwrap();
    let cat = MultiLabelCategorizer::new(model);
    assert_eq!(predicted(&cat, &["team"]), [0]);
}

#[test]
fn min_labels_and_max_labels_bind_together() {
    let capped = model().with_max_labels(Some(2)).unwrap().with_min_labels(Some(2)).unwrap();
    let cat = MultiLabelCategorizer::new(capped);
    // All three clear; the cap keeps the top two and the fill is moot.
    assert_eq!(predicted(&cat, &["team", "stock", "rain"]), [0, 1]);

    let filled = model().with_max_labels(Some(3)).unwrap().with_min_labels(Some(3)).unwrap();
    let cat = MultiLabelCategorizer::new(filled);
    assert_eq!(predicted(&cat, &["team"]), [0, 1, 2], "the fill reaches min_labels");
}

#[test]
fn the_bounds_are_validated() {
    let m = model();
    assert!(m.clone().with_min_labels(Some(0)).is_err(), "no forced emission is the None default");
    assert!(m.clone().with_min_labels(Some(4)).is_err(), "more labels than the model has");
    let capped = m.clone().with_max_labels(Some(1)).unwrap();
    assert!(capped.with_min_labels(Some(2)).is_err(), "min over max crosses the bounds");
    let floored = m.with_min_labels(Some(2)).unwrap();
    assert!(floored.with_max_labels(Some(1)).is_err(), "max under min crosses the bounds");
}

#[test]
fn the_codec_round_trips_min_labels() {
    let model =
        model().with_max_labels(Some(3)).unwrap().with_min_labels(Some(2)).unwrap().with_min_score(0.15).unwrap();
    let opened = MultiLabelDoccatModel::open(&model.write().unwrap()).unwrap();
    assert_eq!(opened.min_labels(), Some(2));
    assert_eq!(opened.max_labels(), Some(3));
    assert_eq!(opened.min_score(), 0.15);
    let cat = MultiLabelCategorizer::new(opened);
    assert_eq!(predicted(&cat, &["team"]), [0, 2]);
}

#[test]
fn version_one_entries_still_open() {
    // The version 1 layout: no min_labels between max_labels and
    // min_score. Hand-built with the same head data as `model`.
    let mut out = Vec::new();
    let utf8 = |out: &mut Vec<u8>, s: &str| {
        out.extend_from_slice(&(s.len() as u16).to_be_bytes());
        out.extend_from_slice(s.as_bytes());
    };
    utf8(&mut out, "MultiLabel");
    out.extend_from_slice(&FORMAT_VERSION_V1.to_be_bytes());
    utf8(&mut out, "en");
    utf8(&mut out, "opennlp.tools.doccat.BagOfWordsFeatureGenerator");
    out.extend_from_slice(&(-1i32).to_be_bytes()); // max_labels: none
    out.extend_from_slice(&0.0f64.to_be_bytes()); // min_score
    out.extend_from_slice(&3i32.to_be_bytes()); // labels
    let head = |out: &mut Vec<u8>, name: &str, threshold: f64, bias: f64, feature: &str, weight: f64| {
        utf8(out, name);
        out.extend_from_slice(&threshold.to_be_bytes());
        out.extend_from_slice(&bias.to_be_bytes());
        out.extend_from_slice(&1i32.to_be_bytes()); // one weight
        utf8(out, feature);
        out.extend_from_slice(&weight.to_be_bytes());
    };
    head(&mut out, "sports", 0.5, -1.0, "bow=team", 4.0);
    head(&mut out, "finance", 0.5, -2.0, "bow=stock", 4.5);
    head(&mut out, "weather", 0.5, -1.5, "bow=rain", 3.0);
    let crc = turbo_train::crc32c(&out);
    out.extend_from_slice(&crc.to_be_bytes());

    let opened = MultiLabelDoccatModel::open(&out).unwrap();
    assert_eq!(opened.min_labels(), None, "version 1 has no min_labels");
    assert_eq!(opened.max_labels(), None);
    assert_eq!(opened.label(2), Some("weather"));
    let cat = MultiLabelCategorizer::new(opened);
    assert_eq!(predicted(&cat, &["rain"]), [2]);
    // A version this reader does not write is refused. The magic is 12
    // bytes, then the version's four big-endian bytes end the field.
    let mut future = out.clone();
    future[15] = 9;
    assert!(matches!(MultiLabelDoccatModel::open(&future), Err(MultiLabelError::UnsupportedVersion(9))));
}

/// The hand-computed fixture: three labels, four documents.
fn report() -> turbo_doccat::multilabel::MultiLabelReport {
    let predicted = [BTreeSet::from([0, 1]), BTreeSet::from([0]), BTreeSet::new(), BTreeSet::from([2])];
    let gold = [BTreeSet::from([0, 2]), BTreeSet::from([0, 1]), BTreeSet::from([1]), BTreeSet::from([2])];
    evaluate(&predicted, &gold, 3).unwrap()
}

fn assert_prf(prf: Prf, precision: f64, recall: f64, f1: f64) {
    let eps = 1e-12;
    assert!((prf.precision - precision).abs() < eps, "precision {} != {precision}", prf.precision);
    assert!((prf.recall - recall).abs() < eps, "recall {} != {recall}", prf.recall);
    assert!((prf.f1 - f1).abs() < eps, "f1 {} != {f1}", prf.f1);
}

#[test]
fn metrics_match_the_hand_computed_values() {
    let report = report();
    // Micro over the summed counts (tp 3, fp 1, fn 3).
    assert_prf(report.micro, 0.75, 0.5, 0.6);
    // Macro over the labels: label 0 perfect, label 1 silent, label 2
    // half recalled.
    assert_prf(report.r#macro, 2.0 / 3.0, 0.5, 5.0 / 9.0);
    // Sample averages over the four documents.
    assert_prf(report.samples, 0.875, 0.5, 13.0 / 24.0);
    // Only the last document is predicted exactly.
    assert!((report.exact_match - 0.25).abs() < 1e-12);
}

#[test]
fn metric_boundaries_follow_the_documented_conventions() {
    // All predictions empty over nonempty gold: nothing predicted is
    // precision 1, nothing found is recall 0.
    let empty = vec![BTreeSet::new(), BTreeSet::new()];
    let gold = vec![BTreeSet::from([0]), BTreeSet::from([1])];
    let report = evaluate(&empty, &gold, 2).unwrap();
    assert_prf(report.micro, 1.0, 0.0, 0.0);
    assert_prf(report.r#macro, 1.0, 0.0, 0.0);
    assert_prf(report.samples, 1.0, 0.0, 0.0);
    assert_eq!(report.exact_match, 0.0);

    // Perfect predictions.
    let perfect = gold.clone();
    let report = evaluate(&perfect, &gold, 2).unwrap();
    assert_prf(report.micro, 1.0, 1.0, 1.0);
    assert_prf(report.r#macro, 1.0, 1.0, 1.0);
    assert_prf(report.samples, 1.0, 1.0, 1.0);
    assert_eq!(report.exact_match, 1.0);

    // No positives anywhere, none predicted: everything is vacuously 1.
    let report = evaluate(&empty, &empty, 2).unwrap();
    assert_prf(report.micro, 1.0, 1.0, 1.0);
    assert_eq!(report.exact_match, 1.0);
}

#[test]
fn the_metrics_validate_their_inputs() {
    let one = [BTreeSet::from([0])];
    assert!(evaluate(&one, &[], 1).is_err(), "predictions and gold differ in length");
    assert!(evaluate(&[], &[], 1).is_err(), "nothing to average over");
    assert!(evaluate(&one, &one, 0).is_err(), "no labels to evaluate over");
    assert!(evaluate(&[BTreeSet::from([1])], &one, 1).is_err(), "a prediction outside the labels");
    assert!(evaluate(&one, &[BTreeSet::from([1])], 1).is_err(), "a gold set outside the labels");
}

#[test]
fn repeated_features_count_at_inference_as_they_did_in_training() {
    // Training sums a feature's counts; inference must too. One head,
    // weight 1 on bow=x, bias 0: ["x","x"] scores sigmoid(2), ["x"]
    // sigmoid(1).
    let model = MultiLabelDoccatModel::new(
        "en",
        &["a"],
        vec![LinearHead::new(0.0, vec![("bow=x".into(), 1.0)]).unwrap()],
        vec![0.5],
        DoccatContextGenerator::from_builtins(vec![BuiltinFeatureGenerator::BagOfWords(
            BagOfWordsFeatureGenerator::new(),
        )]),
    )
    .unwrap();
    let single = model.scores(&["x"])[0];
    let double = model.scores(&["x", "x"])[0];
    let sigmoid = |z: f64| 1.0 / (1.0 + (-z).exp());
    assert!((single - sigmoid(1.0)).abs() < 1e-12, "one token: {}", single);
    assert!((double - sigmoid(2.0)).abs() < 1e-12, "the count weights: {double}");
    assert!(double > single);
}

#[test]
fn calibration_rejects_bad_probabilities_and_clamps_into_the_open_interval() {
    // A validation set of all-1.0 scores fits threshold 1.0, which the
    // open interval (0.0, 1.0) refuses; the fitted value must clamp so
    // fit plus apply round-trips.
    let scores = vec![vec![1.0, 1.0, 0.3]];
    let truth = [BTreeSet::from([0])];
    let calibrator = ThresholdCalibrator::new();
    let fitted = calibrator.fit(&scores, &truth).expect("fit clamps into the open interval");
    assert_eq!(fitted.len(), 3);
    assert!(fitted.iter().all(|&t| (0.0..1.0).contains(&t)), "thresholds {fitted:?}");
    let model = model().with_thresholds(fitted.clone()).expect("the calibrated model applies");
    assert_eq!(model.thresholds(), fitted);

    // NaN and out-of-range scores are probabilities no model emits.
    assert!(calibrator.fit(&[vec![f64::NAN, 0.5]], &truth).is_err());
    assert!(calibrator.fit(&[vec![1.5, 0.5]], &truth).is_err());
    assert!(calibrator.fit(&[vec![-0.1, 0.5]], &truth).is_err());
    assert!(calibrator.fit(&[vec![f64::INFINITY, 0.5]], &truth).is_err());
}

#[test]
fn f_beta_stays_finite_for_extreme_beta() {
    let (p, r, f) = f_beta(1, 1, 1, 1e300);
    assert_eq!((p, r), (0.5, 0.5));
    assert_eq!(f, 0.5, "no NaN from beta * beta overflowing");
    // Small betas keep the plain formula's values.
    let (_, _, f) = f_beta(1, 1, 1, 1.0);
    assert_eq!(f, 0.5);
    let (_, _, f) = f_beta(2, 1, 1, 0.5);
    assert!(f.is_finite());
}

#[test]
fn wire_sized_bounds_and_unique_labels_are_enforced() {
    // i32::MAX + 1 truncates on the wire; it is refused up front.
    assert!(model().with_max_labels(Some(i32::MAX as usize + 1)).is_err());
    assert!(model().with_min_labels(Some(i32::MAX as usize + 1)).is_err());
    assert!(model().with_max_labels(Some(i32::MAX as usize)).is_ok());

    // new() enforces what open() enforces: names non-empty and unique.
    let head = LinearHead::new(0.0, vec![]).unwrap();
    let cg = DoccatContextGenerator::default();
    let dup =
        MultiLabelDoccatModel::new("en", &["a", "a"], vec![head.clone(), head.clone()], vec![0.5, 0.5], cg.clone());
    assert!(dup.is_err(), "duplicate label names");
    let blank =
        MultiLabelDoccatModel::new("en", &["a", ""], vec![head.clone(), head.clone()], vec![0.5, 0.5], cg.clone());
    assert!(blank.is_err(), "an empty label name");
    let fine = MultiLabelDoccatModel::new("en", &["a", "b"], vec![head.clone(), head], vec![0.5, 0.5], cg);
    assert!(fine.is_ok());
}

#[test]
fn extreme_training_parameters_error_instead_of_panicking() {
    use turbo_doccat::DoccatFactory;
    use turbo_doccat::multilabel::{MultiLabelSample, MultiLabelTrainingParameters, train_multilabel};
    let factory = DoccatFactory::new();
    let samples = [MultiLabelSample::new(["a"], ["x"])];
    for params in [
        MultiLabelTrainingParameters { learning_rate: f64::MAX, ..MultiLabelTrainingParameters::default() },
        MultiLabelTrainingParameters { l2: f64::MAX, ..MultiLabelTrainingParameters::default() },
    ] {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            train_multilabel("en", &samples, &factory, &params)
        }));
        let err = result.expect("training must not panic");
        assert!(err.is_err(), "non-finite weights are an error, not a panic");
    }
}

#[test]
fn feature_generator_settings_round_trip_and_custom_generators_error() {
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use turbo_doccat::features::PatternMode;
    use turbo_doccat::multilabel::{
        MultiLabelSample, MultiLabelTrainingParameters, open_package, package_multilabel_model, train_multilabel,
    };
    use turbo_doccat::{DoccatFactory, DoccatFeatureGenerator, NGramFeatureGenerator};

    let samples = [
        MultiLabelSample::new(["sports"], ["team", "123", "wins", "match"]),
        MultiLabelSample::new(["finance"], ["stock", "456", "market", "rally"]),
    ];
    let factory = DoccatFactory::with_feature_generators(vec![
        BuiltinFeatureGenerator::BagOfWords(
            BagOfWordsFeatureGenerator::only_all_letter_tokens().with_pattern_mode(PatternMode::CodePoints),
        ),
        BuiltinFeatureGenerator::NGram(NGramFeatureGenerator::new(1, 3).unwrap()),
    ]);
    let trained = train_multilabel("en", &samples, &factory, &MultiLabelTrainingParameters::default()).unwrap();

    let default_cg_model = trained.clone().with_context_generator(DoccatContextGenerator::from_builtins(vec![
        BuiltinFeatureGenerator::BagOfWords(BagOfWordsFeatureGenerator::new()),
        BuiltinFeatureGenerator::NGram(NGramFeatureGenerator::default()),
    ]));
    assert_ne!(
        trained.model_ref().sha256,
        default_cg_model.model_ref().sha256,
        "distinct generator settings must produce distinct entry sha256"
    );

    let probe = &["team", "123", "wins", "match"];
    assert_ne!(
        trained.scores(probe),
        default_cg_model.scores(probe),
        "non-default BOW/NGram settings must change inference scores on probe"
    );

    let entry_bytes = trained.write().unwrap();
    let opened_entry = MultiLabelDoccatModel::open(&entry_bytes).unwrap();
    assert_eq!(opened_entry.context_generator().builtins(), trained.context_generator().builtins());
    assert_eq!(opened_entry.scores(probe), trained.scores(probe));
    assert_eq!(opened_entry.model_ref().sha256, trained.model_ref().sha256);

    let pkg = package_multilabel_model(&trained, 1_700_000_000_000).unwrap();
    let opened_pkg = open_package(pkg.to_bytes()).unwrap();
    assert_eq!(opened_pkg.context_generator().builtins(), trained.context_generator().builtins());
    assert_eq!(opened_pkg.scores(probe), trained.scores(probe));

    #[derive(Debug)]
    struct CustomGen;
    impl DoccatFeatureGenerator for CustomGen {
        fn extract_features(&self, _text: &[&str], _extra: &BTreeMap<String, String>, out: &mut Vec<String>) {
            out.push("custom=1".into());
        }
    }
    let custom_cg = DoccatContextGenerator::from_custom(vec![Arc::new(CustomGen)]);
    let with_custom = trained.with_context_generator(custom_cg.clone());
    assert!(with_custom.write().is_err(), "write must fail when custom generators are present");
    assert!(
        package_multilabel_model(&with_custom, 1_700_000_000_000).is_err(),
        "package_multilabel_model must fail when custom generators are present"
    );
    assert!(
        !with_custom.model_ref().sha256.is_empty(),
        "in-memory custom-generator models must still record a non-empty provenance sha256"
    );
    let custom_other_weights = MultiLabelDoccatModel::new(
        "en",
        &["finance", "sports"],
        vec![
            LinearHead::new(0.25, vec![("custom=1".into(), 1.5)]).unwrap(),
            LinearHead::new(-0.25, vec![("custom=1".into(), -1.5)]).unwrap(),
        ],
        vec![0.5, 0.5],
        custom_cg,
    )
    .unwrap();
    assert_ne!(
        with_custom.model_ref().sha256,
        custom_other_weights.model_ref().sha256,
        "custom-generator models with different head weights must have distinct sha256"
    );
}

#[test]
fn hash_compat_categorizer_annotators_report_distinct_config_hashes() {
    use turbo_analysis::Annotator;
    use turbo_analysis_layers::{Chunks, Sentences};
    use turbo_doccat::features::PatternMode;
    use turbo_doccat::{DoccatModel, DocumentCategorizerME, NGramFeatureGenerator, SpanCategorizer};
    use turbo_maxent::MaxentModel;
    use turbo_train::{Event, Key as Param, TrainingConfiguration, TrainingParameters, trainer};

    let ml_base = model();
    let ml_cat = MultiLabelCategorizer::new(ml_base.clone());
    let h_base = ml_cat.config_hash();
    assert_ne!(h_base, 0);

    let ml_thresh = MultiLabelCategorizer::new(ml_base.clone().with_thresholds(vec![0.4, 0.5, 0.6]).unwrap());
    let ml_max = MultiLabelCategorizer::new(ml_base.clone().with_max_labels(Some(2)).unwrap());
    let ml_min = MultiLabelCategorizer::new(ml_base.clone().with_min_labels(Some(1)).unwrap());
    let ml_score = MultiLabelCategorizer::new(ml_base.clone().with_min_score(0.25).unwrap());
    let ml_cg =
        MultiLabelCategorizer::new(ml_base.with_context_generator(DoccatContextGenerator::from_builtins(vec![
            BuiltinFeatureGenerator::NGram(NGramFeatureGenerator::new(1, 3).unwrap()),
        ])));
    for other in [
        ml_thresh.config_hash(),
        ml_max.config_hash(),
        ml_min.config_hash(),
        ml_score.config_hash(),
        ml_cg.config_hash(),
    ] {
        assert_ne!(other, 0);
        assert_ne!(other, h_base);
    }
    let ml_spans_sent: turbo_doccat::multilabel::MultiLabelSpanCategorizer<Sentences> = ml_cat.for_spans();
    let ml_spans_chunk: turbo_doccat::multilabel::MultiLabelSpanCategorizer<Chunks> = ml_cat.for_spans();
    assert_ne!(ml_spans_sent.config_hash(), 0);
    assert_ne!(ml_spans_sent.config_hash(), ml_spans_chunk.config_hash());
    assert_ne!(ml_spans_sent.config_hash(), ml_thresh.for_spans::<Sentences>().config_hash());

    let events = vec![Event::from_strs("pos", &["bow=good"]), Event::from_strs("neg", &["bow=bad"])];
    let mut params = TrainingParameters::new();
    params.put(Param::ALGORITHM, "MAXENT");
    params.put(Param::ITERATIONS, 20);
    params.put(Param::CUTOFF, 0);
    let (trained, _) = trainer::train_events(&events, &params, &mut TrainingConfiguration::default()).unwrap();
    let maxent = MaxentModel::read(&trained.write().unwrap()).unwrap();
    let cg1 = DoccatContextGenerator::from_builtins(vec![BuiltinFeatureGenerator::BagOfWords(
        BagOfWordsFeatureGenerator::new(),
    )]);
    let cg2 = DoccatContextGenerator::from_builtins(vec![BuiltinFeatureGenerator::BagOfWords(
        BagOfWordsFeatureGenerator::only_all_letter_tokens().with_pattern_mode(PatternMode::CodePoints),
    )]);
    let doc_cat1 = DocumentCategorizerME::new(DoccatModel::from_maxent("en", maxent.clone(), cg1).unwrap());
    let doc_cat2 = DocumentCategorizerME::new(DoccatModel::from_maxent("en", maxent, cg2).unwrap());
    assert_ne!(doc_cat1.config_hash(), 0);
    assert_ne!(doc_cat1.config_hash(), doc_cat2.config_hash());
    let span1: SpanCategorizer<Sentences> = doc_cat1.for_spans();
    let span2: SpanCategorizer<Sentences> = doc_cat2.for_spans();
    let span_chunk: SpanCategorizer<Chunks> = doc_cat1.for_spans();
    assert_ne!(span1.config_hash(), 0);
    assert_ne!(span1.config_hash(), span2.config_hash());
    assert_ne!(span1.config_hash(), span_chunk.config_hash());
    let hashes = [
        h_base,
        ml_thresh.config_hash(),
        ml_max.config_hash(),
        ml_min.config_hash(),
        ml_score.config_hash(),
        ml_cg.config_hash(),
        ml_spans_sent.config_hash(),
        ml_spans_chunk.config_hash(),
        doc_cat1.config_hash(),
        doc_cat2.config_hash(),
        span1.config_hash(),
        span2.config_hash(),
        span_chunk.config_hash(),
    ];
    // Fixed values from the previous implementation, before migration.
    assert_eq!(
        hashes,
        [
            2406797393651371811,
            14589794838530488386,
            8067442491044145641,
            2163154702620036970,
            2483390473169040282,
            5785359090408478346,
            902727703835235239,
            5322199426368607831,
            14593223505865006032,
            15804804405288898674,
            11971614223886549790,
            18330262021213611080,
            10610587880484796636
        ]
    );
}
