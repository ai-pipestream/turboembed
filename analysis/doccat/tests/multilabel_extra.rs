//! The cardinality bounds and the evaluation metrics of the multi-label
//! categorizer: `min_labels` back-filling the threshold emission (with
//! `min_score` a hard floor), the format's backward compatibility, and
//! micro/macro/sample precision, recall, and F1 plus exact match against
//! hand-computed values.

use std::collections::BTreeSet;

use turbo_doccat::multilabel::{
    FORMAT_VERSION_V1, LinearHead, MultiLabelCategorizer, MultiLabelDoccatModel, MultiLabelError, Prf, evaluate,
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
    let opened = MultiLabelDoccatModel::open(&model.write()).unwrap();
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
