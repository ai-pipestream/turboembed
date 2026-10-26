//! Native numeric validation, without device or performance claims.
use std::path::Path;

use turbo_analysis_embed::calibrate::{Calibration, CalibrationOptions, Cost, Point, Side};
use turbo_analysis_embed::vectors::Shape;
use turbo_analysis_embed::{Device, Precision};

fn point() -> Point {
    Point {
        max_seq: 24,
        max_batch: 1,
        min_cosine: 0.99,
        device_ms: 1.0,
        cpu_ms: 2.0,
        short_device_ms: 0.5,
        short_cpu_ms: 1.0,
        doc_device_ms: 1.0,
        doc_cpu_ms: 2.0,
    }
}

fn calibration() -> Calibration {
    Calibration { key: "numeric-control".into(), floor: 0.93, points: vec![point()] }
}

#[test]
fn invalid_kept_values_are_rejected() {
    let original = calibration().to_text();
    let lines: Vec<&str> = original.lines().collect();
    let fields: Vec<&str> = lines[3].split('\t').collect();
    let mut accepted = Vec::new();
    for column in 4..=9 {
        for bad in ["NaN", "inf", "-inf", "-1", "0", "-0"] {
            let mut row = fields.clone();
            row[column] = bad;
            let text = format!("{}\n{}\n{}\n{}\n", lines[0], lines[1], lines[2], row.join("\t"));
            if Calibration::parse(&text).is_some() {
                accepted.push(format!("time {column}: {bad}"));
            }
        }
    }
    for bad in ["NaN", "inf", "-inf"] {
        if Calibration::parse(&original.replace("floor\t0.93", &format!("floor\t{bad}"))).is_some() {
            accepted.push(format!("floor: {bad}"));
        }
    }
    for (column, bad) in [(1, "0"), (2, "0"), (3, "NaN"), (3, "inf")] {
        let mut row = fields.clone();
        row[column] = bad;
        let text = format!("{}\n{}\n{}\n{}\n", lines[0], lines[1], lines[2], row.join("\t"));
        if Calibration::parse(&text).is_some() {
            accepted.push(format!("point {column}: {bad}"));
        }
    }
    if Calibration::parse(&format!("{original}{}\n", lines[3])).is_some() {
        accepted.push("duplicate shape".into());
    }
    assert!(accepted.is_empty(), "invalid kept values accepted: {accepted:?}");
}

#[test]
fn undefined_cosine_sentinel_and_deliberate_high_floor_still_round_trip() {
    let mut c = calibration();
    c.floor = 1.5;
    c.points[0].min_cosine = f64::NEG_INFINITY;
    assert_eq!(Calibration::parse(&c.to_text()), Some(c.clone()));
    assert_eq!(c.side(24, 1, None).unwrap(), Side::Cpu);
    c.floor = 0.93;
    c.points[0].min_cosine = -1.0000000000000002;
    assert_eq!(Calibration::parse(&c.to_text()), Some(c));
}

#[test]
fn cost_fit_does_not_discard_invalid_samples() {
    let shape = Shape { runs: 1, rows: 1, tokens: 1 };
    assert!(Cost::fit(&[(shape, 1.0)]).is_some());
    let accepted: Vec<_> = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0, 0.0]
        .into_iter()
        .filter(|&bad| Cost::fit(&[(shape, 1.0), (shape, bad)]).is_some())
        .collect();
    assert!(accepted.is_empty(), "invalid fit samples accepted: {accepted:?}");
}

#[test]
fn overflowing_normalization_cannot_create_a_nan_cost() {
    let shape = Shape { runs: 1, rows: 1, tokens: 1 };
    let cost = Cost::fit(&[(shape, 1e-320)]);
    assert!(cost.is_none(), "unrepresentable fit returned {cost:?}");
}

#[test]
fn invalid_cache_is_replaced_by_actual_cpu_measurement() {
    let bundle = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle");
    let options = CalibrationOptions {
        device: Device::Cpu,
        precision: Precision::Exact,
        windows: vec![24],
        batches: vec![1],
        floor: Some(0.93),
        reps: 1,
    };
    let texts = vec![("hello".into(), vec![(0, 5)])];
    let mut c = calibration();
    c.key = Calibration::key(&bundle, &options, &texts).unwrap();
    c.points[0].device_ms = -1.0;
    let dir = std::env::temp_dir().join(format!("turbo-calibration-values-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = Calibration::file(&dir, &c.key);
    std::fs::write(&path, c.to_text()).unwrap();
    let loaded = Calibration::load_or_measure(&dir, &bundle, &options, &texts).unwrap();
    assert!(loaded.points[0].device_ms > 0.0, "invalid cache was reused: {loaded:?}");
    assert_eq!(loaded.key, c.key);
    assert_eq!(Calibration::parse(&std::fs::read_to_string(&path).unwrap()), Some(loaded));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn nonfinite_requested_floor_fails_before_opening_the_model() {
    let options = CalibrationOptions {
        device: Device::Cpu,
        precision: Precision::Exact,
        windows: vec![24],
        batches: vec![1],
        floor: None,
        reps: 1,
    };
    for floor in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let o = CalibrationOptions { floor: Some(floor), ..options.clone() };
        let err = Calibration::key(Path::new("absent-calibration-model"), &o, &[]).unwrap_err();
        assert!(err.message.contains("floor"), "invalid floor reached model opening: {}", err.message);
    }
}
