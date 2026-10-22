//! `Calibration` on the small BERT of testdata/tiny-bert-bundle, with the
//! CPU standing in for the device: the windows it measures, the widest
//! one under a floor, the kept form, and that a kept calibration is read
//! back rather than measured again. On a device the environment names,
//! it measures a real bundle and prints what it decided.

use std::path::{Path, PathBuf};

use turbo_analysis_embed::calibrate::{Calibration, CalibrationOptions, Point, Text};
use turbo_analysis_embed::{Device, Precision, VectorOptions};

const SENTENCES: [&str; 8] = [
    "Embedding models turn a passage of text into a vector.",
    "Passages with similar meaning land near each other.",
    "A search system embeds every document once.",
    "It stores the vectors in an index for later.",
    "Each query is embedded as it arrives.",
    "The nearest documents by cosine are the candidates.",
    "Long documents are split into chunks first.",
    "Normalization scales that vector to unit length.",
];

fn bert() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle")
}

/// The sentences as one text, and as one text per sentence.
fn texts() -> Vec<Text> {
    let mut text = String::new();
    let mut spans = Vec::new();
    for s in SENTENCES {
        if !text.is_empty() {
            text.push(' ');
        }
        let a = text.len() as u32;
        text.push_str(s);
        spans.push((a, text.len() as u32));
    }
    let mut all = vec![(text, spans)];
    all.extend(SENTENCES.iter().map(|s| (s.to_string(), vec![(0, s.len() as u32)])));
    all
}

fn options(floor: Option<f64>) -> CalibrationOptions {
    CalibrationOptions { device: Device::Cpu, precision: Precision::Model, windows: vec![24, 64], floor, reps: 1 }
}

#[test]
fn the_widest_window_over_the_floor_is_kept() {
    let o = options(None);
    let c = Calibration::measure(&bert(), &o, &texts()).unwrap();
    assert_eq!(c.points.iter().map(|p| p.max_seq).collect::<Vec<_>>(), vec![24, 64]);
    // The CPU against itself: the F32 tolerance, and every window clears it.
    assert_eq!(c.floor, turbo::record::tolerance(turbo::TURBO_DTYPE_F32).unwrap().min_cosine);
    assert!(c.points.iter().all(|p| p.min_cosine > 0.9999 && p.device_ms > 0.0 && p.cpu_ms > 0.0), "{c:?}");
    assert_eq!(c.max_seq, 64);
    assert!(c.key.contains(" | cpu | "), "{}", c.key);
}

#[test]
fn no_window_over_the_floor_keeps_the_cpu() {
    let o = options(Some(1.5));
    let c = Calibration::measure(&bert(), &o, &texts()).unwrap();
    assert_eq!((c.max_seq, c.use_device), (0, false));
    let v = c.options(&CalibrationOptions { device: Device::Backend("cuda".into()), ..o }, &VectorOptions::default());
    assert!(matches!(v.encoder.device, Device::Cpu), "{v:?}");
}

#[test]
fn the_device_runs_at_the_chosen_window_when_it_is_faster() {
    let o = CalibrationOptions { device: Device::Backend("metal".into()), ..options(None) };
    let c = Calibration {
        key: String::new(),
        floor: 0.93,
        max_seq: 96,
        use_device: true,
        points: vec![Point { max_seq: 96, min_cosine: 0.95, device_ms: 1.0, cpu_ms: 2.0 }],
    };
    let v = c.options(&o, &VectorOptions::default());
    assert!(matches!(&v.encoder.device, Device::Backend(b) if b == "metal"), "{v:?}");
    assert_eq!(v.encoder.max_seq, 96);
}

#[test]
fn the_kept_form_reads_back() {
    let c = Calibration {
        key: "m | abc | hailo | Hailo-8 | 4.23.0 | x | model | 1".into(),
        floor: 0.93,
        max_seq: 64,
        use_device: true,
        points: vec![
            Point { max_seq: 64, min_cosine: 0.9403431657324838, device_ms: 95.3, cpu_ms: 869.2 },
            Point { max_seq: 128, min_cosine: 0.8351824182316789, device_ms: 63.4, cpu_ms: 882.2 },
        ],
    };
    assert_eq!(Calibration::parse(&c.to_text()), Some(c));
    assert_eq!(Calibration::parse("calibration 2\n"), None);
    assert_eq!(Calibration::parse("calibration 1\nfloor\tx\n"), None);
}

#[test]
fn a_kept_calibration_is_read_back_not_measured_again() {
    let dir = std::env::temp_dir().join(format!("turbo-calibrate-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let o = options(None);
    let first = Calibration::load_or_measure(&dir, &bert(), &o, &texts()).unwrap();
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
    // Timings differ from run to run, so equal ones were read, not measured.
    let second = Calibration::load_or_measure(&dir, &bert(), &o, &texts()).unwrap();
    assert_eq!(first, second);
    // Another precision is another key, measured and kept beside it.
    let exact = CalibrationOptions { precision: Precision::Exact, ..o };
    let third = Calibration::load_or_measure(&dir, &bert(), &exact, &texts()).unwrap();
    assert_ne!(third.key, first.key);
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 2);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// On the device TURBO_TEST_DEVICE names, with the bundle
/// TURBO_TEST_VECTORS_BUNDLE: windows of 64, 96 and 128 tokens, printed.
#[test]
fn on_the_device_turbo_test_device_names() {
    let (Ok(bundle), Ok(want)) = (std::env::var("TURBO_TEST_VECTORS_BUNDLE"), std::env::var("TURBO_TEST_DEVICE"))
    else {
        println!("skipped: TURBO_TEST_VECTORS_BUNDLE and TURBO_TEST_DEVICE are not both set");
        return;
    };
    let device = match want.parse() {
        Ok(i) => Device::Index(i),
        Err(_) => Device::Backend(want),
    };
    let o =
        CalibrationOptions { device, precision: Precision::Model, windows: vec![64, 96, 128], floor: None, reps: 5 };
    // Short texts and one long one, so the widest windows fill their rows.
    let mut many = texts();
    let (mut long, mut spans) = (String::new(), Vec::new());
    for s in SENTENCES.iter().cycle().take(3 * SENTENCES.len()) {
        if !long.is_empty() {
            long.push(' ');
        }
        let a = long.len() as u32;
        long.push_str(s);
        spans.push((a, long.len() as u32));
    }
    for _ in 0..4 {
        many.push((long.clone(), spans.clone()));
    }
    let c = Calibration::measure(Path::new(&bundle), &o, &many).unwrap();
    print!("{}", c.to_text());
    assert!(c.points.iter().all(|p| p.min_cosine > 0.0));
}
