//! `Calibration` on the small BERT of testdata/tiny-bert-bundle, with the
//! CPU standing in for the device: the windows and batches it measures,
//! the side it picks for each, the kept form, and that a kept calibration
//! is read back rather than measured again. On a device the environment
//! names, it measures a real bundle, prints the table, and checks that the
//! step opened for each configuration runs on the side that won it.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use turbo_analysis_embed::calibrate::{
    Calibration, CalibrationOptions, Point, Side, Text, lowest_cosine, time, vectors,
};
use turbo_analysis_embed::{Device, EncoderOptions, Precision, VectorAnnotator, VectorOptions};

mod held_out;
use held_out::held_out;

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
    CalibrationOptions {
        device: Device::Cpu,
        precision: Precision::Model,
        windows: vec![24, 64],
        batches: vec![1, 4],
        floor,
        reps: 1,
    }
}

fn point(max_seq: u32, max_batch: u32, min_cosine: f64, device_ms: f64, cpu_ms: f64) -> Point {
    Point { max_seq, max_batch, min_cosine, device_ms, cpu_ms, short_device_ms: 14.1, short_cpu_ms: 7.5 }
}

fn at(max_seq: u32, max_batch: u32) -> VectorOptions {
    let encoder = EncoderOptions { max_seq, max_batch, ..EncoderOptions::default() };
    VectorOptions { encoder, ..VectorOptions::default() }
}

#[test]
fn every_window_and_batch_is_measured_in_order() {
    let c = Calibration::measure(&bert(), &options(None), &texts()).unwrap();
    let got: Vec<(u32, u32)> = c.points.iter().map(|p| (p.max_seq, p.max_batch)).collect();
    assert_eq!(got, vec![(24, 1), (24, 4), (64, 1), (64, 4)]);
    // The CPU against itself: the F32 tolerance, and every point clears it.
    assert_eq!(c.floor, turbo::record::tolerance(turbo::TURBO_DTYPE_F32).unwrap().min_cosine);
    assert!(c.points.iter().all(|p| p.min_cosine > 0.9999 && p.device_ms > 0.0 && p.cpu_ms > 0.0), "{c:?}");
    assert!(c.key.contains(" | cpu | "), "{}", c.key);
}

#[test]
fn under_the_floor_the_cpu_runs_every_configuration() {
    let o = options(Some(1.5));
    let c = Calibration::measure(&bert(), &o, &texts()).unwrap();
    for p in &c.points {
        assert_eq!(c.side(p.max_seq, p.max_batch, None).unwrap(), Side::Cpu, "{p:?}");
    }
    let o = CalibrationOptions { device: Device::Backend("cuda".into()), ..o };
    let v = c.options(&o, &at(64, 4)).unwrap();
    assert!(matches!(v.encoder.device, Device::Cpu), "{v:?}");
}

#[test]
fn the_device_runs_where_it_is_faster_within_the_floor() {
    let c = Calibration {
        key: String::new(),
        floor: 0.93,
        points: vec![
            point(32, 1, 0.95, 14.0, 5.0),
            point(32, 32, 0.95, 90.0, 160.0),
            point(128, 32, 0.92, 90.0, 900.0),
            point(128, 8, 0.95, 30.0, 30.0),
        ],
    };
    let side = |w, b| c.side(w, b, None).unwrap();
    // Slower, faster, faster under the floor, and a tie.
    assert_eq!(
        [side(32, 1), side(32, 32), side(128, 32), side(128, 8)],
        [Side::Cpu, Side::Device, Side::Cpu, Side::Cpu]
    );
    let o = CalibrationOptions { device: Device::Backend("hailo".into()), ..options(None) };
    let v = c.options(&o, &at(32, 32)).unwrap();
    assert!(matches!(&v.encoder.device, Device::Backend(b) if b == "hailo"), "{v:?}");
    assert_eq!((v.encoder.max_seq, v.encoder.max_batch), (32, 32));
    // The CPU side runs at the precision it was timed at, whatever base asks.
    let v = c.options(&o, &at(32, 1)).unwrap();
    assert!(matches!(v.encoder.device, Device::Cpu), "{v:?}");
    assert_eq!(v.encoder.precision, Precision::Exact);
    // A configuration not measured, the model's own (0), and a force that
    // names neither side are errors; a force names the side.
    for (w, b) in [(64, 32), (0, 0), (32, 0)] {
        let e = c.side(w, b, None).unwrap_err().to_string();
        assert!(e.contains("no calibration point") && e.contains("32x1, 32x32, 128x32, 128x8"), "{e}");
    }
    assert!(c.options(&o, &VectorOptions::default()).is_err());
    assert_eq!(c.side(32, 1, Some("device")).unwrap(), Side::Device);
    assert_eq!(c.side(32, 32, Some("cpu")).unwrap(), Side::Cpu);
    for v in ["", "hailo", "CPU"] {
        let e = c.side(32, 1, Some(v)).unwrap_err().to_string();
        assert!(e.contains("TURBO_CALIBRATION_FORCE"), "{v:?}: {e}");
    }
}

#[test]
fn a_configuration_not_asked_is_refused_before_anything_is_measured() {
    let dir = std::env::temp_dir().join(format!("turbo-calibrate-unasked-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    for (w, b) in [(96, 4), (24, 8), (0, 0)] {
        let e = Calibration::annotator(&dir, &bert(), &options(None), &texts(), &at(w, b)).err().unwrap().to_string();
        assert!(e.contains("no calibration point") && e.contains("24x1, 24x4, 64x1, 64x4"), "{e}");
    }
    assert!(!dir.exists());
}

#[test]
fn a_document_alone_gets_the_vectors_it_gets_among_the_rest() {
    // The side is chosen per configuration, so a document's vectors do not
    // depend on what else a call sends.
    let mut base = at(24, 4);
    base.encoder.precision = Precision::Exact;
    let a = Arc::new(VectorAnnotator::open(&bert(), &base).unwrap());
    let all = vectors(&bert(), &a, &texts()).unwrap();
    for i in 0..texts().len() {
        let alone = vectors(&bert(), &a, &texts()[i..i + 1]).unwrap();
        let at = vectors(&bert(), &a, &texts()[..i]).unwrap().len();
        let same = lowest_cosine(&alone, &all[at..at + alone.len()], a.dim() as usize);
        assert!(same >= 1.0 - 1e-6, "text {i}: {same}");
    }
}

#[test]
fn the_kept_form_reads_back() {
    let c = Calibration {
        key: "m | abc | hailo | Hailo-8 | 4.23.0 | x | model | 1".into(),
        floor: 0.93,
        points: vec![point(64, 1, 0.9403431657324838, 95.3, 869.2), point(128, 32, 0.8351824182316789, 63.4, 882.2)],
    };
    let text = c.to_text();
    assert_eq!(Calibration::parse(&text), Some(c));
    assert_eq!(Calibration::parse("calibration 1\n"), None);
    assert_eq!(Calibration::parse("calibration 4\nfloor\tx\n"), None);
    // The second form timed the CPU at Precision::Model, and the third
    // has no times of one sentence alone: neither is read.
    for old in ["calibration 2", "calibration 3"] {
        assert_eq!(Calibration::parse(&text.replacen("calibration 4", old, 1)), None);
    }
    // Every line short of the key, the floor and a point is refused, and
    // so is a record given twice or a point of the first form's four fields.
    let lines: Vec<&str> = text.lines().collect();
    for n in 1..lines.len() - 1 {
        assert_eq!(Calibration::parse(&lines[..n].join("\n")), None, "{n} lines");
    }
    assert_eq!(Calibration::parse(&format!("{text}floor\t0.9\n")), None);
    assert_eq!(Calibration::parse(&format!("{text}point\t64\t0.95\t1\t2\n")), None);
}

#[test]
fn a_cosine_that_is_not_a_number_is_the_lowest() {
    assert_eq!(lowest_cosine(&[f32::NAN, 0.0], &[1.0, 0.0], 2), f64::NEG_INFINITY);
    assert_eq!(lowest_cosine(&[1.0, 0.0, f32::NAN, 0.0], &[1.0, 0.0, 1.0, 0.0], 2), f64::NEG_INFINITY);
    assert_eq!(lowest_cosine(&[1.0, 0.0, 0.0, 1.0], &[1.0, 0.0, 1.0, 0.0], 2), 0.0);
    // So a device that gives one is never chosen.
    assert!(!point(64, 1, f64::NEG_INFINITY, 1.0, 2.0).device_wins(0.93));
}

#[test]
fn windows_and_batches_are_each_given_once_and_in_range() {
    for windows in [vec![], vec![24, 24], vec![0], vec![65]] {
        let o = CalibrationOptions { windows: windows.clone(), ..options(None) };
        assert!(Calibration::measure(&bert(), &o, &texts()).is_err(), "{windows:?}");
    }
    for batches in [vec![], vec![4, 4], vec![0]] {
        let o = CalibrationOptions { batches: batches.clone(), ..options(None) };
        let e = Calibration::measure(&bert(), &o, &texts()).unwrap_err().to_string();
        assert!(e.contains("calibration batches"), "{batches:?}: {e}");
    }
}

#[test]
fn malformed_sentence_bounds_are_an_error_not_a_panic() {
    let o = CalibrationOptions { windows: vec![24], ..options(None) };
    for (text, spans) in
        [("hello", vec![(4, 1)]), ("hello", vec![(0, 6)]), ("héllo", vec![(0, 2)]), ("hello", vec![(0, 5), (7, 9)])]
    {
        let t = vec![(text.to_string(), spans.clone())];
        let e = Calibration::measure(&bert(), &o, &t).unwrap_err().to_string();
        assert!(e.contains("not a span"), "{text:?} {spans:?}: {e}");
        let a = Arc::new(VectorAnnotator::open(&bert(), &at(24, 1)).unwrap());
        assert!(time(&bert(), &a, &t, 1).unwrap_err().to_string().contains("not a span"));
        assert!(vectors(&bert(), &a, &t).unwrap_err().to_string().contains("not a span"));
    }
}

#[test]
fn the_key_names_what_was_asked() {
    let k = |o: &CalibrationOptions, t: &[Text]| Calibration::key(&bert(), o, t).unwrap();
    let base = k(&options(None), &texts());
    assert!(base.contains(turbo::tuning::LIBRARY_BUILD), "{base}");
    assert_ne!(base, k(&options(Some(0.97)), &texts()));
    assert_ne!(base, k(&CalibrationOptions { windows: vec![24], ..options(None) }, &texts()));
    assert_ne!(base, k(&CalibrationOptions { batches: vec![1], ..options(None) }, &texts()));
    assert_ne!(base, k(&options(None), &texts()[1..]));
    assert_eq!(base, k(&CalibrationOptions { reps: 7, ..options(None) }, &texts()));
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
    let exact = CalibrationOptions { precision: Precision::Exact, ..o.clone() };
    let third = Calibration::load_or_measure(&dir, &bert(), &exact, &texts()).unwrap();
    assert_ne!(third.key, first.key);
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 2);
    // A kept file cut short is measured again and replaced.
    let file = Calibration::file(&dir, &first.key);
    let text = first.to_text();
    std::fs::write(&file, &text[..text.find("floor").unwrap()]).unwrap();
    let again = Calibration::load_or_measure(&dir, &bert(), &o, &texts()).unwrap();
    assert_eq!(again.key, first.key);
    assert_ne!(again.points, first.points);
    assert_eq!(Calibration::parse(&std::fs::read_to_string(&file).unwrap()), Some(again.clone()));
    // Cut between its points, a file still parses, but lacks a point asked.
    let text = again.to_text();
    let cut = &text[..text.rfind("point").unwrap()];
    assert!(Calibration::parse(cut).is_some());
    std::fs::write(&file, cut).unwrap();
    let whole = Calibration::load_or_measure(&dir, &bert(), &o, &texts()).unwrap();
    assert_eq!(whole.points.len(), 4);
    assert_ne!(whole.points, again.points);
    // The step opened through it runs where its point says, here the CPU.
    let a = Calibration::annotator(&dir, &bert(), &o, &texts(), &at(64, 4)).unwrap();
    assert_eq!(a.execution().backend, "cpu");
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 2);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// How much slower than the faster side's measured time a chosen
/// configuration may run when timed again: the spread of the Pi 5 CPU's
/// medians from run to run is about 5%.
const NOISE: f64 = 1.10;

/// On the device TURBO_TEST_DEVICE names, with the bundle
/// TURBO_TEST_VECTORS_BUNDLE: windows of 16 to 128 tokens and 1 to 32
/// rows a run, over the held-out documents three times over, so the widest
/// windows fill their runs. Prints the table, then opens the step at each
/// configuration through the calibration and checks, on that one
/// instance: that it runs on the side that won; that a document sent
/// alone gets the vectors it gets among all the others; and that the whole
/// set runs within NOISE of the faster side's time, timed up to three
/// times. It prints what one sentence sent alone takes on each side.
#[test]
fn on_the_device_turbo_test_device_names() {
    let (Ok(bundle), Ok(want)) = (std::env::var("TURBO_TEST_VECTORS_BUNDLE"), std::env::var("TURBO_TEST_DEVICE"))
    else {
        println!("skipped: TURBO_TEST_VECTORS_BUNDLE and TURBO_TEST_DEVICE are not both set");
        return;
    };
    let bundle = Path::new(&bundle);
    let device = match want.parse() {
        Ok(i) => Device::Index(i),
        Err(_) => Device::Backend(want),
    };
    let o = CalibrationOptions {
        device,
        precision: Precision::Model,
        windows: vec![16, 32, 64, 128],
        batches: vec![1, 8, 32],
        floor: None,
        reps: 5,
    };
    let texts: Vec<Text> = (0..3).flat_map(|_| held_out()).collect();
    let (first, first_spans) = &texts[0];
    let end = first_spans[0].1;
    let sentence: Vec<Text> = vec![(first[..end as usize].to_string(), vec![(0, end)])];
    let c = Calibration::measure(bundle, &o, &texts).unwrap();
    print!("{}", c.to_text());
    assert!(c.points.iter().all(|p| p.min_cosine > 0.0));
    let dir = std::env::temp_dir().join(format!("turbo-calibrate-device-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(Calibration::file(&dir, &c.key), c.to_text()).unwrap();
    println!("window\tbatch\tmin cosine\tdevice ms\tcpu ms\tchosen\tchosen ms\tone sentence: chosen ms\tother ms");
    for p in &c.points {
        let base = at(p.max_seq, p.max_batch);
        let a = Arc::new(Calibration::annotator(&dir, bundle, &o, &texts, &base).unwrap());
        let side = c.side(p.max_seq, p.max_batch, None).unwrap();
        let backend = a.execution().backend.clone();
        assert_eq!(backend == "cpu", side == Side::Cpu, "{p:?}: {backend}");
        assert_eq!(side == Side::Device, p.device_wins(c.floor), "{p:?}");
        let all = vectors(bundle, &a, &texts).unwrap();
        let alone = vectors(bundle, &a, &texts[..1]).unwrap();
        let dim = a.dim() as usize;
        let same = lowest_cosine(&alone, &all[..alone.len()], dim);
        assert!(same >= 1.0 - 1e-6, "{p:?}: a document alone and among the rest: cosine {same}");
        let best = p.device_ms.min(p.cpu_ms);
        let mut ms = f64::INFINITY;
        for _ in 0..3 {
            ms = time(bundle, &a, &texts, o.reps).unwrap();
            if ms <= best * NOISE {
                break;
            }
        }
        assert!(ms <= best * NOISE, "{p:?}: the chosen side took {ms:.1} ms, over {NOISE} times {best:.1}");
        let other = match side {
            Side::Device => VectorOptions {
                encoder: EncoderOptions { device: Device::Cpu, precision: Precision::Exact, ..base.encoder.clone() },
                ..base.clone()
            },
            Side::Cpu => VectorOptions {
                encoder: EncoderOptions { device: o.device.clone(), precision: o.precision, ..base.encoder.clone() },
                ..base.clone()
            },
        };
        let other = Arc::new(VectorAnnotator::open(bundle, &other).unwrap());
        let one = time(bundle, &a, &sentence, 20).unwrap();
        let one_other = time(bundle, &other, &sentence, 20).unwrap();
        println!(
            "{}\t{}\t{:.4}\t{:.1}\t{:.1}\t{backend}\t{ms:.1}\t{one:.2}\t{one_other:.2}",
            p.max_seq, p.max_batch, p.min_cosine, p.device_ms, p.cpu_ms
        );
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
