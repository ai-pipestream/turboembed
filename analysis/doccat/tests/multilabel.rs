//! The multi-label categorizer on hand-built and trained models:
//! independent sigmoid scores, per-label threshold emission, abstention
//! into `Categories` (best category `Tag::NONE`, never a forced argmax),
//! threshold calibration against a flat 0.5, and model and package
//! round-trips with truncation checks.

use std::collections::BTreeSet;

use turbo_analysis::{Annotator, Document, Layer, Span, Spans, Tag, TagSet, View, ViewId};
use turbo_analysis_layers::{Categories, Sentences, Tokens};
use turbo_doccat::multilabel::{
    LinearHead, MultiLabelCategorizer, MultiLabelDoccatModel, MultiLabelError, MultiLabelSample,
    MultiLabelTrainingParameters, ThresholdCalibrator, f_beta, macro_f1, open_package, package_multilabel_model,
    precision_recall, train_multilabel,
};
use turbo_doccat::{BagOfWordsFeatureGenerator, BuiltinFeatureGenerator, DoccatContextGenerator, DoccatFactory};

/// Two labels with separable keywords and one shared noise word: `stock`
/// fires `finance` but also appears in sports-only documents.
fn hand_built_model() -> MultiLabelDoccatModel {
    let heads = vec![
        LinearHead::new(-1.0, vec![("bow=team".into(), 4.0)]).unwrap(),
        LinearHead::new(-2.0, vec![("bow=stock".into(), 4.5)]).unwrap(),
    ];
    MultiLabelDoccatModel::new(
        "en",
        &["sports", "finance"],
        heads,
        vec![0.5, 0.5],
        DoccatContextGenerator::from_builtins(vec![BuiltinFeatureGenerator::BagOfWords(
            BagOfWordsFeatureGenerator::new(),
        )]),
    )
    .unwrap()
}

fn token_spans(items: &[(u32, u32)]) -> Spans<Tokens> {
    let mut s = Spans::new(ViewId::SOURCE);
    for &(start, end) in items {
        s.push(Span::new(start, end));
    }
    s
}

#[test]
fn scores_are_independent_and_do_not_sum_to_one() {
    let model = hand_built_model();
    let scores = model.scores(&["team", "stock"]);
    assert_eq!(scores.len(), 2);
    for &s in &scores {
        assert!((0.0..=1.0).contains(&s));
    }
    assert!((scores[0] - 0.9526).abs() < 1e-3, "sports: {}", scores[0]);
    assert!((scores[1] - 0.9241).abs() < 1e-3, "finance: {}", scores[1]);
    assert!((scores[0] + scores[1] - 1.0).abs() > 1e-6, "scores summed to 1");

    // No keyword: both heads sit at their biases, still independent.
    let empty = model.scores(&["unrelated"]);
    assert!((empty[0] - 0.2689).abs() < 1e-3);
    assert!((empty[1] - 0.1192).abs() < 1e-3);
}

#[test]
fn every_label_clearing_its_threshold_is_emitted() {
    let cat = MultiLabelCategorizer::new(hand_built_model());
    let scores = cat.categorize(&["team", "stock"]);
    let predicted = cat.predicted_labels(&["team", "stock"]);
    assert_eq!(predicted, [("sports", scores[0]), ("finance", scores[1])]);
    assert_eq!(cat.predicted_tags(&scores).iter().map(|t| t.0).collect::<Vec<_>>(), [0, 1]);

    // Only finance clears: a single-label emission, not an argmax.
    let scores = cat.categorize(&["stock"]);
    assert_eq!(cat.predicted_tags(&scores).iter().map(|t| t.0).collect::<Vec<_>>(), [1]);
}

#[test]
fn abstention_emits_no_labels_and_no_best_category() {
    let model = hand_built_model();
    let cat = MultiLabelCategorizer::new(model.clone());

    // "cooking" text reaches neither threshold: empty prediction.
    let scores = cat.categorize(&["recipe", "kitchen"]);
    assert!(cat.predicted_tags(&scores).is_empty());

    // The annotator writes the abstention into the layer: one item, no
    // best category, an empty selected set, every score present, the
    // layer's invariants hold.
    let doc = Document::new("recipe kitchen");
    let view = doc.get::<View>().unwrap();
    let tokens = Tokens { spans: token_spans(&[(0, 6), (7, 14)]), sentence: None };
    let (cats,) = cat.annotate((view, &tokens)).unwrap();
    assert_eq!(cats.len(), 1);
    assert_eq!(cats.category, [Tag::NONE]);
    assert!(cats.selected(0).is_empty(), "abstention selects nothing");
    assert!(cats.check().is_ok());
    assert_eq!(cats.best(), None);
    assert_eq!(cats.best_category(0), "");
    assert_eq!(cats.row_scores(0), scores.as_slice());
    assert_eq!(cats.ranked().len(), 2, "the scores are still all there");

    // What ran is still recorded on the layer's provenance path.
    assert!(cat.model().model_ref().id.starts_with("turbo:MultiLabelDoccat:"));
}

#[test]
fn the_annotation_keeps_the_full_selected_label_set() {
    // The regression: label sets {a, b} (thresholds [0.5, 0.5]) and {a}
    // (thresholds [0.5, 0.9]) share best and scores, so the layer must
    // carry the selected set to tell them apart.
    let cat = MultiLabelCategorizer::new(hand_built_model());
    let doc = Document::new("team stock");
    let view = doc.get::<View>().unwrap();
    let tokens = Tokens { spans: token_spans(&[(0, 4), (5, 10)]), sentence: None };
    let (both,) = cat.annotate((view, &tokens)).unwrap();
    assert_eq!(both.selected(0), [Tag(0), Tag(1)]);
    assert_eq!(both.get(0).unwrap().selected, [Tag(0), Tag(1)]);

    let raised = hand_built_model().with_thresholds(vec![0.5, 0.95]).unwrap();
    let cat = MultiLabelCategorizer::new(raised);
    let (one,) = cat.annotate((view, &tokens)).unwrap();
    assert_eq!(one.selected(0), [Tag(0)]);

    assert_eq!(both.category, one.category, "same best, same scores");
    assert_eq!(both.row_scores(0), one.row_scores(0));
    assert_ne!(both, one, "the selected sets differ and the layer shows it");
}

#[test]
fn max_labels_caps_and_min_score_floors_the_emission() {
    let model = hand_built_model().with_max_labels(Some(1)).unwrap();
    let cat = MultiLabelCategorizer::new(model);
    let scores = cat.categorize(&["team", "stock"]);
    let tags = cat.predicted_tags(&scores);
    assert_eq!(tags.len(), 1, "one label at most");
    assert_eq!(tags[0].0, 0, "the higher score wins the cap");

    let model = hand_built_model().with_min_score(0.97).unwrap();
    let cat = MultiLabelCategorizer::new(model);
    let scores = cat.categorize(&["team", "stock"]);
    assert!(cat.predicted_tags(&scores).is_empty(), "neither score reaches 0.97");

    assert!(hand_built_model().with_max_labels(Some(0)).is_err());
    assert!(hand_built_model().with_min_score(1.5).is_err());
    assert!(hand_built_model().with_min_score(f64::NAN).is_err());
}

#[test]
fn span_categorizer_abstains_per_span_and_stays_aligned() {
    let cat = MultiLabelCategorizer::new(hand_built_model());
    let span_cat = cat.for_spans::<Sentences>();

    let doc = Document::new("team wins. recipe time.");
    let view = doc.get::<View>().unwrap();
    let tokens = Tokens { spans: token_spans(&[(0, 4), (5, 9), (11, 17), (18, 22)]), sentence: None };
    let mut sentence_spans = Spans::new(ViewId::SOURCE);
    sentence_spans.push(Span::new(0, 9));
    sentence_spans.push(Span::new(11, 22));
    let sentences = Sentences { spans: sentence_spans };
    let (cats,) = span_cat.annotate((view, &tokens, &sentences)).unwrap();

    assert_eq!(cats.len(), 2, "one item per sentence, the abstaining one included");
    assert_eq!(cats.category, [Tag(0), Tag::NONE]);
    assert_eq!(cats.best_category(0), "sports");
    assert_eq!(cats.best_category(1), "");
    assert!(cats.check().is_ok());
}

#[test]
fn the_layer_records_abstention_with_no_best() {
    let mut cats = Categories::new(TagSet::new("turbo:multilabel-doccat", &["a"]));
    cats.push(Tag::NONE, &[0.4]);
    assert!(cats.check().is_ok());
    assert_eq!(cats.best(), None);
    assert_eq!(cats.best_category(0), "");
    assert_eq!(cats.get(0).unwrap().best_score(), 0.0);
}

#[test]
fn the_model_validates_its_parts() {
    let head = || LinearHead::new(0.0, vec![("bow=a".into(), 1.0)]).unwrap();
    let cg = DoccatContextGenerator::default();
    let build = |heads: Vec<LinearHead>, thresholds: Vec<f64>| {
        MultiLabelDoccatModel::new("en", &["a", "b"], heads, thresholds, cg.clone())
    };
    assert!(build(vec![head()], vec![0.5, 0.5]).is_err(), "one head for two labels");
    assert!(build(vec![head(), head()], vec![0.5]).is_err(), "one threshold for two labels");
    assert!(build(vec![head(), head()], vec![0.5, 0.0]).is_err(), "0.0 is outside the open interval");
    assert!(build(vec![head(), head()], vec![0.5, 1.0]).is_err(), "1.0 is outside the open interval");
    assert!(build(vec![head(), head()], vec![0.5, f64::NAN]).is_err());

    assert!(LinearHead::new(f64::NAN, vec![]).is_err());
    assert!(LinearHead::new(0.0, vec![("bow=b".into(), 1.0), ("bow=a".into(), 1.0)]).is_err(), "unsorted");
    assert!(LinearHead::new(0.0, vec![("bow=a".into(), 1.0), ("bow=a".into(), 2.0)]).is_err(), "duplicate");
    assert!(LinearHead::new(0.0, vec![("bow=a".into(), f64::INFINITY)]).is_err());
}

#[test]
fn the_entry_bytes_round_trip_and_truncation_errors() {
    let model = hand_built_model()
        .with_max_labels(Some(3))
        .unwrap()
        .with_min_score(0.1)
        .unwrap()
        .with_thresholds(vec![0.4, 0.6])
        .unwrap();
    let bytes = model.write();

    let opened = MultiLabelDoccatModel::open(&bytes).unwrap();
    assert_eq!(opened.thresholds(), [0.4, 0.6]);
    assert_eq!(opened.max_labels(), Some(3));
    assert_eq!(opened.min_score(), 0.1);
    assert_eq!(opened.num_labels(), 2);
    assert_eq!(opened.label(0), Some("sports"));
    assert_eq!(opened.head(1).unwrap().weights(), &[("bow=stock".to_owned(), 4.5)]);
    assert_eq!(opened.scores(&["team"]), model.scores(&["team"]));
    let scores = model.scores(&["team", "stock"]);
    assert_eq!(opened.predicted_tags(&scores), model.predicted_tags(&scores));

    // Every truncation must error, never panic.
    for len in 0..bytes.len() {
        assert!(MultiLabelDoccatModel::open(&bytes[..len]).is_err(), "a truncation to {len} bytes opened");
    }
    // A flipped checksum byte is a checksum error, not an open model.
    let mut corrupt = bytes.clone();
    let n = corrupt.len();
    corrupt[n - 1] ^= 0x01;
    assert!(matches!(MultiLabelDoccatModel::open(&corrupt), Err(MultiLabelError::Checksum { .. })));
}

#[test]
fn the_package_round_trips_the_calibrated_thresholds() {
    let model = hand_built_model().with_thresholds(vec![0.3, 0.7]).unwrap();
    let package = package_multilabel_model(&model, 1_700_000_000_000);
    let opened = open_package(package.to_bytes()).unwrap();
    assert_eq!(opened.thresholds(), [0.3, 0.7]);
    assert_eq!(opened.max_labels(), None);
    assert_eq!(opened.label(1), Some("finance"));
    assert_eq!(opened.scores(&["stock"]), model.scores(&["stock"]));
    assert_eq!(opened.model_ref().id, model.model_ref().id);

    // A package of another component is refused.
    let mut manifest = turbo_train::Manifest::new("DocumentCategorizerME", "en", 0);
    manifest.set("factory", "opennlp.tools.doccat.DoccatFactory");
    let mut other = turbo_train::ModelPackage::new(manifest);
    other.add_entry("doccat.model", model.write());
    assert!(open_package(other.to_bytes()).is_err());
}

#[test]
fn f_beta_conventions_and_macro_f1() {
    assert_eq!(precision_recall(0, 0, 3), (1.0, 0.0), "nothing predicted is precision 1");
    assert_eq!(precision_recall(0, 2, 0), (0.0, 1.0), "no positive is recall 1");
    let (p, r, f1) = f_beta(1, 1, 1, 1.0);
    assert_eq!((p, r, f1), (0.5, 0.5, 0.5));
    assert_eq!(f_beta(0, 2, 2, 1.0).2, 0.0, "precision and recall both 0");

    // Macro F1 of thresholding: label 0 thresholded at 0.5 predicts both
    // rows; label 1 at 0.5 predicts none.
    let scores = vec![vec![0.9, 0.2], vec![0.4, 0.1]];
    let truth = [BTreeSet::from([0]), BTreeSet::from([1])];
    let f1 = macro_f1(&scores, &truth, &[0.5, 0.5]);
    assert!((f1 - 0.5).abs() < 1e-12, "label 0 is perfect, label 1 predicts nothing: {f1}");
}

#[test]
fn the_calibrator_sweeps_scores_and_breaks_ties_upward() {
    // One label, five validation rows. F_1 peaks at 0.9 and 0.8 (both
    // 2/3); the tie breaks toward the highest threshold, 0.9.
    let scores = vec![vec![0.9], vec![0.85], vec![0.84], vec![0.8], vec![0.75]];
    let truth = [BTreeSet::from([0]), BTreeSet::new(), BTreeSet::new(), BTreeSet::from([0]), BTreeSet::new()];
    let fitted = ThresholdCalibrator::new().fit(&scores, &truth).unwrap();
    assert_eq!(fitted, [0.9]);

    // The optimum away from 0.5: at 0.7 both positives clear and one
    // negative is admitted (F_1 0.8, the maximum); a flat 0.5 cannot
    // single this cut out.
    let scores = vec![vec![0.9], vec![0.8], vec![0.7]];
    let truth = [BTreeSet::from([0]), BTreeSet::new(), BTreeSet::from([0])];
    assert_eq!(ThresholdCalibrator::new().fit(&scores, &truth).unwrap(), [0.7]);
}

#[test]
fn the_calibrator_honors_beta_the_precision_floor_and_zero_positives() {
    // F_1 peaks at the precise candidate, F_2 at the high-recall one.
    let scores = vec![vec![0.9], vec![0.8], vec![0.7], vec![0.6], vec![0.55]];
    let truth = [BTreeSet::from([0]), BTreeSet::new(), BTreeSet::new(), BTreeSet::new(), BTreeSet::from([0])];
    assert_eq!(ThresholdCalibrator::new().fit(&scores, &truth).unwrap(), [0.9]);
    let f2 = ThresholdCalibrator::new().with_beta(2.0).unwrap();
    assert_eq!(f2.fit(&scores, &truth).unwrap(), [0.55]);

    // The floor restricts the candidates; with none reaching it the
    // fallback is the candidate of the highest precision.
    let scores = vec![vec![0.9], vec![0.8]];
    let truth = [BTreeSet::new(), BTreeSet::from([0])];
    let floored = ThresholdCalibrator::new().with_min_precision(0.9).unwrap();
    assert_eq!(floored.fit(&scores, &truth).unwrap(), [0.8]);

    // A label with no positive in the validation set keeps the default.
    let scores = vec![vec![0.2], vec![0.1]];
    let truth = [BTreeSet::new(), BTreeSet::new()];
    assert_eq!(ThresholdCalibrator::new().fit(&scores, &truth).unwrap(), [0.5]);

    assert!(ThresholdCalibrator::new().with_beta(0.0).is_err());
    assert!(ThresholdCalibrator::new().with_min_precision(0.0).is_err());
}

#[test]
fn the_calibrator_validates_its_input() {
    let cal = ThresholdCalibrator::new();
    let scores = vec![vec![0.9, 0.1], vec![0.2, 0.3]];
    let truth = [BTreeSet::from([0]), BTreeSet::from([1])];
    assert!(cal.fit(&scores, &truth).is_ok());
    assert!(cal.fit(&scores, &truth[..1]).is_err(), "scores and truth differ in length");
    assert!(cal.fit(&[vec![0.9], vec![0.1, 0.2]], &truth).is_err(), "ragged rows");
    let truth = [BTreeSet::from([2]), BTreeSet::from([1])];
    assert!(cal.fit(&scores, &truth).is_err(), "a truth index outside the labels");
}

/// The multi-label fixture: imbalanced labels. `sports` documents are
/// frequent and clean; `finance` documents are rare, and `stock`, the
/// finance keyword, also appears in sports-only documents, so the rare
/// label's optimal threshold is far from 0.5. A few documents hold both
/// labels.
struct Fixture {
    train: Vec<MultiLabelSample>,
    validation: Vec<MultiLabelSample>,
}

fn fixture() -> Fixture {
    let sports_docs: &[&[&str]] = &[
        &["team", "wins", "match"],
        &["goal", "team", "match"],
        &["coach", "team", "wins"],
        &["match", "goal", "team"],
        &["team", "match", "crowd"],
        &["goal", "wins", "team"],
    ];
    let finance_docs: &[&[&str]] =
        &[&["stock", "rises", "market"], &["market", "rally", "earnings"], &["earnings", "beat", "market"]];
    let mixed_docs: &[&[&str]] = &[&["team", "stock", "stadium"], &["market", "team", "club"]];
    let noise_docs: &[&[&str]] = &[&["team", "stock", "trade"], &["goal", "stock", "match"]];

    let mut train = Vec::new();
    for _ in 0..8 {
        for doc in sports_docs {
            train.push(MultiLabelSample::new(["sports"], doc.iter().copied()));
        }
    }
    for _ in 0..3 {
        for doc in noise_docs {
            train.push(MultiLabelSample::new(["sports"], doc.iter().copied()));
        }
    }
    for _ in 0..2 {
        for doc in finance_docs {
            train.push(MultiLabelSample::new(["finance"], doc.iter().copied()));
        }
    }
    for doc in mixed_docs {
        train.push(MultiLabelSample::new(["sports", "finance"], doc.iter().copied()));
    }

    // The held-out validation set: same distribution, fresh documents,
    // and "report" documents that hold the finance label with no direct
    // keyword, so their scores sit under the noise.
    let mut validation = Vec::new();
    for (i, doc) in sports_docs.iter().enumerate() {
        validation.push(MultiLabelSample::new(["sports"], doc.iter().chain(["final"].iter()).copied()));
        if i % 2 == 0 {
            validation.push(MultiLabelSample::new(["sports"], doc.iter().chain(["stock"].iter()).copied()));
        }
    }
    for doc in noise_docs {
        validation.push(MultiLabelSample::new(["sports"], doc.iter().copied()));
    }
    for doc in finance_docs {
        validation.push(MultiLabelSample::new(["finance"], doc.iter().copied()));
        validation.push(MultiLabelSample::new(["finance"], doc.iter().take(1).chain(["report"].iter()).copied()));
    }
    validation.push(MultiLabelSample::new(["sports", "finance"], ["team", "market"]));
    Fixture { train, validation }
}

fn validation_matrix(model: &MultiLabelDoccatModel, fixture: &Fixture) -> (Vec<Vec<f64>>, Vec<BTreeSet<usize>>) {
    let truth: Vec<BTreeSet<usize>> = fixture
        .validation
        .iter()
        .map(|s| s.labels().iter().map(|l| model.index(l).expect("unknown label")).collect())
        .collect();
    let scores: Vec<Vec<f64>> = fixture
        .validation
        .iter()
        .map(|s| model.scores(&s.tokens().iter().map(String::as_str).collect::<Vec<_>>()))
        .collect();
    (scores, truth)
}

#[test]
fn training_then_calibration_beats_a_flat_threshold_on_macro_f1() {
    let fixture = fixture();
    let factory = DoccatFactory::with_feature_generators(vec![BuiltinFeatureGenerator::BagOfWords(
        BagOfWordsFeatureGenerator::new(),
    )]);
    // Under-trained on purpose: the rare label's scores straddle 0.5, so
    // the flat threshold both misses weak positives and admits the
    // sports-only noise.
    let params = MultiLabelTrainingParameters { iterations: 60, ..MultiLabelTrainingParameters::default() };
    let model = train_multilabel("en", &fixture.train, &factory, &params).unwrap();
    assert_eq!(model.num_labels(), 2);
    assert_eq!(model.thresholds(), [0.5, 0.5], "training leaves the defaults for calibration");

    let (scores, truth) = validation_matrix(&model, &fixture);
    let flat = macro_f1(&scores, &truth, model.thresholds());

    let calibrator = ThresholdCalibrator::new();
    let calibrated = calibrator.calibrate(&model, &scores, &truth).unwrap();
    let tuned = macro_f1(&scores, &truth, calibrated.thresholds());
    println!("flat macro-F1 {flat}, calibrated macro-F1 {tuned}, thresholds {:?}", calibrated.thresholds());

    assert!(tuned > flat, "calibrated macro-F1 {tuned} must beat the flat 0.5 threshold's {flat}");
    assert!(tuned - flat > 0.05, "the improvement is a real margin, not noise: {tuned} vs {flat}");
    let finance = model.index("finance").unwrap();
    assert_ne!(calibrated.thresholds()[finance], 0.5, "the rare label moved off the default");

    // The calibrated thresholds serialize inside the package and come
    // back: the acceptance claim survives a save and an open.
    let opened = open_package(package_multilabel_model(&calibrated, 1_700_000_000_000).to_bytes()).unwrap();
    assert_eq!(opened.thresholds(), calibrated.thresholds());
    assert_eq!(macro_f1(&scores, &truth, opened.thresholds()), tuned);
}

#[test]
fn the_trainer_validates_its_inputs() {
    let factory = DoccatFactory::new();
    let params = MultiLabelTrainingParameters::default();
    assert!(train_multilabel("en", &[], &factory, &params).is_err(), "no samples");
    let unlabeled = [MultiLabelSample::new(Vec::<String>::new(), ["team"])];
    assert!(train_multilabel("en", &unlabeled, &factory, &params).is_err(), "no labels");
    let sample = [MultiLabelSample::new(["sports"], ["team"])];
    let bad = MultiLabelTrainingParameters { learning_rate: 0.0, ..params.clone() };
    assert!(train_multilabel("en", &sample, &factory, &bad).is_err());
    let bad = MultiLabelTrainingParameters { l2: f64::NAN, ..params };
    assert!(train_multilabel("en", &sample, &factory, &bad).is_err());
}

#[test]
fn a_trained_model_classifies_and_package_round_trips() {
    let fixture = fixture();
    let factory = DoccatFactory::new();
    let model = train_multilabel("en", &fixture.train, &factory, &MultiLabelTrainingParameters::default()).unwrap();

    let sports = model.scores(&["team", "wins"])[model.index("sports").unwrap()];
    let finance = model.scores(&["team", "wins"])[model.index("finance").unwrap()];
    assert!(sports > 0.9 && sports <= 1.0, "sports: {sports}");
    assert!(finance < 0.1, "finance: {finance}");

    let bytes = model.write();
    let opened = MultiLabelDoccatModel::open(&bytes).unwrap();
    assert_eq!(opened.scores(&["team", "wins"]), model.scores(&["team", "wins"]));

    let package = package_multilabel_model(&model, 1_700_000_000_000);
    assert_eq!(package.manifest().component_name(), "MultiLabelDoccat");
    let bytes = package.to_bytes();
    let opened = open_package(bytes.clone()).unwrap();
    assert_eq!(opened.label(0), model.label(0));
    assert_eq!(opened.scores(&["stock", "market"]), model.scores(&["stock", "market"]));
    // Opening from a package records the package's identity: the SHA-256
    // of the archive's bytes, not of the bare entry.
    assert_eq!(opened.model_ref().sha256, turbo_maxent::sha256(&bytes).to_hex());
}
