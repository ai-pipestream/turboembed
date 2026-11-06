//! The dispatch preference: how `TURBO_CALIBRATION_PREFER` reads, the
//! side each preference picks at and around its margin, and a dispatch
//! opened from a kept calibration choosing by it on the CPU alone, with the
//! small BERT of testdata/tiny-bert-bundle standing in for the device.

use std::path::{Path, PathBuf};

use turbo_analysis_embed::calibrate::{
    Calibration, CalibrationOptions, DEFAULT_MARGIN, MAX_MARGIN, PREFER_VAR, Point, Preference, PreferenceError, Side,
    Text, shape_of,
};
use turbo_analysis_embed::{Device, Dispatch, EncoderOptions, Precision, VectorAnnotator, VectorOptions};

mod held_out;
use held_out::held_out;

fn bert() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle")
}

fn at(max_seq: u32, max_batch: u32) -> VectorOptions {
    let encoder =
        EncoderOptions { device: Device::Cpu, precision: Precision::Model, max_seq, max_batch, ..Default::default() };
    VectorOptions { encoder, ..VectorOptions::default() }
}

#[test]
fn unset_and_offload_are_the_default_margin() {
    let default = Preference::Offload { margin: DEFAULT_MARGIN };
    assert_eq!(DEFAULT_MARGIN, 0.25);
    assert_eq!(Preference::default(), default);
    assert_eq!(Preference::from_var(None), Ok(default));
    assert_eq!(Preference::from_var(Some("offload")), Ok(default));
    assert_eq!(Preference::from_var(Some("latency")), Ok(Preference::Latency));
    assert_eq!(Preference::from_var(Some("offload=0.1")), Ok(Preference::Offload { margin: 0.1 }));
    assert_eq!(Preference::from_var(Some("offload=1")), Ok(Preference::Offload { margin: MAX_MARGIN }));
}

#[test]
fn a_margin_of_zero_is_latency() {
    assert_eq!(Preference::from_var(Some("offload=0")), Ok(Preference::Latency));
    assert_eq!(Preference::from_var(Some("offload=0.0")), Ok(Preference::Latency));
    assert_eq!(Preference::Latency.margin(), 0.0);
}

#[test]
fn other_values_are_typed_errors_naming_the_variable() {
    for v in ["", "Latency", "fast", "device", "offload:0.1", "latency=0.1"] {
        let e = Preference::from_var(Some(v)).unwrap_err();
        assert_eq!(e, PreferenceError::Unknown(v.to_string()), "{v:?}");
        assert!(e.to_string().contains(PREFER_VAR), "{e}");
    }
    for m in ["", "-0.1", "1.01", "2", "nan", "inf", "-inf", "x", "0.25 "] {
        let v = format!("offload={m}");
        let e = Preference::from_var(Some(&v)).unwrap_err();
        assert_eq!(e, PreferenceError::Margin(m.to_string()), "{v:?}");
        assert!(e.to_string().contains(PREFER_VAR), "{e}");
    }
}

#[test]
fn latency_takes_the_device_only_when_it_is_faster() {
    let p = Preference::Latency;
    assert!(p.device(9.999, 10.0));
    assert!(!p.device(10.0, 10.0));
    assert!(!p.device(10.001, 10.0));
}

#[test]
fn offload_takes_the_device_inside_its_margin_and_the_cpu_at_or_past_it() {
    let p = Preference::default();
    // Faster, equal and up to just under 1.25 times the CPU: the device.
    for device_ms in [5.0, 10.0, 12.0, 12.499] {
        assert!(p.device(device_ms, 10.0), "{device_ms}");
    }
    // At 1.25 times the CPU and past it: the CPU.
    for device_ms in [12.5, 12.501, 20.0] {
        assert!(!p.device(device_ms, 10.0), "{device_ms}");
    }
    // The 64x1 document of #1328 on a quiet Raspberry Pi 5: the device at
    // 28.156 ms, the CPU at about 27.8 ms.
    assert!(!Preference::Latency.device(28.156, 27.8));
    assert!(p.device(28.156, 27.8));
    // The widest margin: up to just under twice the CPU's time.
    let wide = Preference::Offload { margin: MAX_MARGIN };
    assert!(wide.device(19.999, 10.0));
    assert!(!wide.device(20.0, 10.0));
}

#[test]
fn a_measured_call_runs_on_the_side_the_default_margin_picks() {
    let dir = std::env::temp_dir().join(format!("turbo-dispatch-prefer-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let o = CalibrationOptions {
        device: Device::Cpu,
        precision: Precision::Model,
        windows: vec![24],
        batches: vec![4],
        floor: None,
        reps: 1,
    };
    let texts = held_out();
    let short: Vec<Text> = vec![(texts[0].0[..texts[0].1[0].1 as usize].to_string(), vec![texts[0].1[0]])];
    let probe = VectorAnnotator::open(&bert(), &at(24, 4)).unwrap();
    let shape = |t: &[Text]| shape_of(&bert(), &probe, t, 4).unwrap();
    let (all, one, doc) = (shape(&texts), shape(&short), shape(&texts[..1]));
    assert!(all != one && all != doc && one != doc, "{all:?} {one:?} {doc:?}");
    // The CPU at 100 ms for each; the device at 1.1 times that for the
    // whole set (inside the margin), 1.25 times for one sentence (at it)
    // and 0.9 times for the first text (faster).
    let point = Point {
        max_seq: 24,
        max_batch: 4,
        min_cosine: 1.0,
        device_ms: 110.0,
        cpu_ms: 100.0,
        short_device_ms: 125.0,
        short_cpu_ms: 100.0,
        doc_device_ms: 90.0,
        doc_cpu_ms: 100.0,
    };
    let key = Calibration::key(&bert(), &o, &texts).unwrap();
    let kept = Calibration { key: key.clone(), floor: 0.999, points: vec![point.clone()] };
    std::fs::write(Calibration::file(&dir, &key), kept.to_text()).unwrap();
    let d = Dispatch::open(&dir, &bert(), &o, &texts, &at(24, 4)).unwrap();
    assert_eq!(d.preference(), Preference::default(), "run without {PREFER_VAR} set");
    assert_eq!([d.side(all), d.side(one), d.side(doc)], [Side::Device, Side::Cpu, Side::Device]);
    // A device whose vectors miss the floor runs nothing, inside the margin
    // or faster.
    let kept = Calibration { key: key.clone(), floor: 0.999, points: vec![Point { min_cosine: 0.5, ..point }] };
    std::fs::write(Calibration::file(&dir, &key), kept.to_text()).unwrap();
    let d = Dispatch::open(&dir, &bert(), &o, &texts, &at(24, 4)).unwrap();
    assert_eq!([d.side(all), d.side(one), d.side(doc)], [Side::Cpu, Side::Cpu, Side::Cpu]);
    std::fs::remove_dir_all(&dir).unwrap();
}
