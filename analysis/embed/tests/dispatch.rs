//! `Dispatch`: the sentence vector step on the device or on the CPU, chosen
//! per call. On the small BERT of testdata/tiny-bert-bundle, with the CPU
//! at `Precision::Model` standing in for the device: the choice between two
//! measured batches, what it refuses before measuring, and that each
//! document records the side its call ran on and gets that side's vectors.
//! On a device the environment names, every submission size through one
//! instance runs within the timing noise of the faster side.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use turbo_analysis::{Analyzer, Document, Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_embed::calibrate::{
    Calibration, CalibrationOptions, Cost, Point, Side, Text, lowest_cosine, shape_of,
};
use turbo_analysis_embed::vectors::{Input, Shape};
use turbo_analysis_embed::{
    Device, Dispatch, Embedder, EncoderOptions, Precision, SentenceVectors, SubwordAnnotator, VectorAnnotator,
    VectorOptions,
};
use turbo_analysis_layers::{Sentences, Subwords, Vectors};

mod held_out;
use held_out::held_out;

fn bert() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle")
}

fn at(device: Device, precision: Precision, max_seq: u32, max_batch: u32) -> VectorOptions {
    let encoder = EncoderOptions { device, precision, max_seq, max_batch, ..EncoderOptions::default() };
    VectorOptions { encoder, ..VectorOptions::default() }
}

/// The sentence vector step on `e`.
fn step<E: Embedder>(bundle: &Path, e: Arc<E>) -> Analyzer {
    Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .add(SubwordAnnotator::open(bundle).unwrap())
        .add(SentenceVectors(e))
        .build()
        .unwrap()
}

/// A document for each text, with its sentences.
fn documents(texts: &[Text]) -> Vec<Document> {
    texts
        .iter()
        .map(|(text, spans)| {
            let mut d = Document::new(text.as_str());
            let mut s = Spans::new(ViewId::SOURCE);
            for &(a, b) in spans {
                s.push(Span::new(a, b));
            }
            d.add(&Key::DEFAULT, Sentences { spans: s }, Provenance::by("test")).unwrap();
            d
        })
        .collect()
}

/// `texts` through `a`: each document's vectors and the execution its
/// layer records.
fn run(a: &Analyzer, texts: &[Text]) -> Vec<(Vec<f32>, String)> {
    let mut docs = documents(texts);
    assert!(a.run(&mut docs).iter().all(|r| r.is_ok()));
    let key = Key::<Vectors<Sentences>>::DEFAULT.id();
    docs.iter()
        .map(|d| {
            let e = d.provenance(key).unwrap().execution.clone().unwrap();
            (d.get::<Vectors<Sentences>>().unwrap().data.clone(), format!("{} {}", e.backend, e.precision))
        })
        .collect()
}

/// The median time in ms of `reps` runs of `texts` through `a`.
fn median_ms(a: &Analyzer, texts: &[Text], reps: usize) -> f64 {
    run(a, texts);
    let mut t: Vec<f64> = (0..reps)
        .map(|_| {
            let s = Instant::now();
            run(a, texts);
            s.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    t.sort_by(f64::total_cmp);
    t[reps / 2]
}

/// The middle of three or more times.
fn middle(t: &[f64]) -> f64 {
    let mut t = t.to_vec();
    t.sort_by(f64::total_cmp);
    t[t.len() / 2]
}

/// The lowest and highest of some times.
fn low_high(t: &[f64]) -> (f64, f64) {
    t.iter().fold((f64::INFINITY, 0f64), |(l, h), &x| (l.min(x), h.max(x)))
}

/// Times as their middle, then their lowest and highest.
fn spread(t: &[f64]) -> String {
    let (lo, hi) = low_high(t);
    format!("{:.2} ({lo:.2}-{hi:.2})", middle(t))
}

/// The lowest cosine between two runs' vectors, document by document.
fn same(a: &[(Vec<f32>, String)], b: &[(Vec<f32>, String)], dim: usize) -> f64 {
    a.iter().zip(b).map(|(x, y)| lowest_cosine(&x.0, &y.0, dim)).fold(1.0, f64::min)
}

fn shape(runs: u64, rows: u64, tokens: u64) -> Shape {
    Shape { runs, rows, tokens }
}

#[test]
fn a_cost_is_fitted_to_runs_rows_and_tokens() {
    // Samples of 2 ms a run, 0.5 a row and 0.25 a token, fitted exactly.
    let truth = Cost { run: 2.0, row: 0.5, token: 0.25 };
    let shapes = [shape(1, 1, 12), shape(210, 210, 3000), shape(27, 210, 3300), shape(7, 210, 3400)];
    let samples: Vec<(Shape, f64)> = shapes.iter().map(|&s| (s, truth.ms(s))).collect();
    let c = Cost::fit(&samples).unwrap();
    for (got, want) in [(c.run, 2.0), (c.row, 0.5), (c.token, 0.25)] {
        assert!((got - want).abs() < 1e-9, "{c:?}");
    }
    // As many runs as rows in every sample: a run and a row cannot be told
    // apart, and either way a call of as many runs as rows is the same.
    let one_a_run = [shape(1, 1, 12), shape(210, 210, 3000), shape(50, 50, 2000)];
    let samples: Vec<(Shape, f64)> = one_a_run.iter().map(|&s| (s, truth.ms(s))).collect();
    let c = Cost::fit(&samples).unwrap();
    let call = shape(9, 9, 400);
    assert!((c.ms(call) - truth.ms(call)).abs() < 1e-9, "{c:?}");
    // No cost is below 0, even where the least squares one would be: a
    // device that pays for each row whatever its length gets no cost a
    // token, so one short sentence costs it a whole row.
    let frames = [
        (shape(1, 1, 12), 14.08),
        (shape(213, 213, 3000), 2998.6),
        (shape(27, 213, 3300), 1241.7),
        (shape(7, 213, 3400), 640.4),
    ];
    let c = Cost::fit(&frames).unwrap();
    assert!(c.run >= 0.0 && c.row >= 0.0 && c.token >= 0.0, "{c:?}");
    assert!((c.ms(shape(1, 1, 12)) - 14.08).abs() < 1.0, "{c:?}");
    // Nothing to tell the costs apart by.
    assert_eq!(Cost::fit(&[]), None);
}

#[test]
fn what_dispatch_cannot_choose_from_is_refused_before_anything_is_measured() {
    let dir = std::env::temp_dir().join(format!("turbo-dispatch-refused-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let o = CalibrationOptions {
        device: Device::Cpu,
        precision: Precision::Model,
        windows: vec![24],
        batches: vec![1, 4],
        floor: None,
        reps: 1,
    };
    for (w, b) in [(32, 4), (24, 8)] {
        let base = at(Device::Cpu, Precision::Model, w, b);
        let e = Dispatch::open(&dir, &bert(), &o, &held_out(), &base).err().unwrap().to_string();
        assert!(e.contains("needs the calibration to ask for"), "{e}");
    }
    assert!(!dir.exists());
}

#[test]
fn each_document_records_the_side_its_call_ran_on() {
    let dir = std::env::temp_dir().join(format!("turbo-dispatch-placed-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let o = CalibrationOptions {
        device: Device::Cpu,
        precision: Precision::Model,
        windows: vec![24],
        batches: vec![1, 4],
        floor: None,
        reps: 1,
    };
    let texts = held_out();
    // A kept calibration with a Hailo-8's costs: 11 ms a run and 3 a row
    // on the device, 0.6 a token on the CPU, at the shapes the texts, their
    // first sentence and their first text have here.
    let key = Calibration::key(&bert(), &o, &texts).unwrap();
    let probe = VectorAnnotator::open(&bert(), &at(Device::Cpu, Precision::Model, 24, 4)).unwrap();
    let short: Vec<Text> = vec![(texts[0].0[..texts[0].1[0].1 as usize].to_string(), vec![texts[0].1[0]])];
    let (frames, tokens) = (Cost { run: 11.0, row: 3.0, token: 0.0 }, Cost { run: 0.0, row: 0.0, token: 0.6 });
    let points = [1, 4]
        .map(|b| {
            let at = |t: &[Text]| shape_of(&bert(), &probe, t, b).unwrap();
            let (all, one, doc) = (at(&texts), at(&short), at(&texts[..1]));
            Point {
                max_seq: 24,
                max_batch: b,
                min_cosine: 1.0,
                device_ms: frames.ms(all),
                cpu_ms: tokens.ms(all),
                short_device_ms: frames.ms(one),
                short_cpu_ms: tokens.ms(one),
                doc_device_ms: frames.ms(doc),
                doc_cpu_ms: tokens.ms(doc),
            }
        })
        .to_vec();
    let kept = Calibration { key: key.clone(), floor: 0.999, points };
    std::fs::write(Calibration::file(&dir, &key), kept.to_text()).unwrap();
    let d = Arc::new(Dispatch::open(&dir, &bert(), &o, &texts, &at(Device::Cpu, Precision::Model, 24, 4)).unwrap());
    let a = step(&bert(), d.clone());
    let fixed = |p| step(&bert(), Arc::new(VectorAnnotator::open(&bert(), &at(Device::Cpu, p, 24, 4)).unwrap()));
    let (device, cpu) = (fixed(Precision::Model), fixed(Precision::Exact));
    let dim = d.on(Side::Cpu).dim() as usize;
    // One sentence is one row: the CPU, at Exact.
    let one: Vec<Text> = vec![(texts[0].0[..texts[0].1[0].1 as usize].to_string(), vec![texts[0].1[0]])];
    let got = run(&a, &one);
    assert_eq!(got[0].1, "cpu f32 (exact)");
    assert!(same(&got, &run(&cpu, &one), dim) >= 1.0 - 1e-6);
    // Every document of a call of many rows records the device's run.
    let got = run(&a, &texts);
    assert!(got.iter().all(|g| g.1 == "cpu f32 (model)"), "{:?}", got.iter().map(|g| &g.1).collect::<Vec<_>>());
    assert!(same(&got, &run(&device, &texts), dim) >= 1.0 - 1e-6);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// What `shape` predicts for `texts` on `a`, then what a real call made.
fn predicted_and_done(a: &Arc<VectorAnnotator>, texts: &[Text]) -> (Shape, Shape) {
    let words = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .add(SubwordAnnotator::open(&bert()).unwrap())
        .build()
        .unwrap();
    let mut docs = documents(texts);
    assert!(words.run(&mut docs).iter().all(|r| r.is_ok()));
    let inputs: Vec<Input<'_>> = docs
        .iter()
        .map(|d| Input {
            subwords: d.get::<Subwords>().unwrap(),
            sentences: Some(&d.get::<Sentences>().unwrap().spans),
            chunks: None,
            levels: None,
        })
        .collect();
    let predicted = a.shape("test", &inputs);
    // Limits of 0 are refused, not packed forever.
    for (b, r) in [(1, 0), (0, 1), (0, 0)] {
        let e = a.shape_at("test", &inputs, b, r).unwrap_err().to_string();
        assert!(e.contains("each must be at least 1"), "{e}");
    }
    assert_eq!(a.shape_at("test", &inputs, a.info().max_batch, a.info().max_ranges).unwrap(), predicted);
    run(&step(&bert(), a.clone()), texts);
    (predicted, a.last_shape())
}

#[test]
fn the_shape_predicted_is_the_shape_run() {
    // One text of 40 one-word sentences: 40 ranges in one window, so the
    // ranges a run holds, not its rows, split it.
    let mut text = String::new();
    let mut spans = Vec::new();
    for i in 0..40 {
        if i > 0 {
            text.push(' ');
        }
        let a = text.len() as u32;
        text.push_str("word.");
        spans.push((a, text.len() as u32));
    }
    let many: Vec<Text> = vec![(text, spans)];
    for (max_batch, max_ranges) in [(1, 0), (1, 1), (4, 0), (4, 3), (64, 0)] {
        let mut o = at(Device::Cpu, Precision::Exact, 64, max_batch);
        o.encoder.max_ranges = max_ranges;
        let a = Arc::new(VectorAnnotator::open(&bert(), &o).unwrap());
        for texts in [&many, &held_out()] {
            let (predicted, done) = predicted_and_done(&a, texts);
            assert_eq!(predicted, done, "{max_batch} rows and {max_ranges} ranges a run");
        }
    }
    // The case the cap alone gets wrong: one row a run, one range a row.
    let mut o = at(Device::Cpu, Precision::Exact, 64, 1);
    o.encoder.max_ranges = 1;
    let a = Arc::new(VectorAnnotator::open(&bert(), &o).unwrap());
    assert_eq!(predicted_and_done(&a, &many).1.runs, 40);
}

/// On the device TURBO_TEST_DEVICE names, with the bundle
/// TURBO_TEST_VECTORS_BUNDLE: the calibration of tests/calibrate.rs, then,
/// at each window and cap, one sentence, one document and the whole set
/// sent through one dispatched instance. Every document of a submission
/// records one side, its vectors are that side's, and the side is the
/// faster of the step fixed on the device and fixed on the CPU, each timed
/// three times in turn: the one with the lower middle time, unless the two
/// sides' times overlap, where neither is faster. Prints each time's
/// middle and spread, the dispatched instance's too.
#[test]
fn on_the_device_every_submission_runs_at_the_faster_side() {
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
        device: device.clone(),
        precision: Precision::Model,
        windows: vec![16, 32, 64, 128],
        batches: vec![1, 8, 32],
        floor: None,
        reps: 5,
    };
    let texts: Vec<Text> = (0..3).flat_map(|_| held_out()).collect();
    let (first, spans) = &texts[0];
    let sentence: Vec<Text> = vec![(first[..spans[0].1 as usize].to_string(), vec![spans[0]])];
    let submissions: [(&str, &[Text], usize); 3] =
        [("one sentence", &sentence, 20), ("one document", &texts[..1], 10), ("all", &texts, 5)];
    let dir = std::env::temp_dir().join(format!("turbo-dispatch-device-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    println!("window\tcap\tsubmission\tdispatched\tfaster\tdispatched ms\tdevice ms\tcpu ms (median, low-high of 3)");
    for &max_seq in &o.windows {
        for &cap in &o.batches {
            let base = at(device.clone(), Precision::Model, max_seq, cap);
            let d = Arc::new(Dispatch::open(&dir, bundle, &o, &texts, &base).unwrap());
            let dim = d.on(Side::Cpu).dim() as usize;
            let a = step(bundle, d.clone());
            println!("{max_seq}x{cap} device {:?}, cpu {:?}", d.costs().0, d.costs().1);
            let open = |dev: Device, p| {
                step(bundle, Arc::new(VectorAnnotator::open(bundle, &at(dev, p, max_seq, cap)).unwrap()))
            };
            let (on_device, on_cpu) = (open(device.clone(), Precision::Model), open(Device::Cpu, Precision::Exact));
            for (what, sub, reps) in submissions {
                let got = run(&a, sub);
                let side = &got[0].1;
                assert!(got.iter().all(|g| &g.1 == side), "{max_seq}x{cap} {what}: one side per call");
                let fixed = if side.starts_with("cpu") { &on_cpu } else { &on_device };
                assert!(same(&got, &run(fixed, sub), dim) >= 1.0 - 1e-6, "{max_seq}x{cap} {what}");
                // Three medians of each, taking turns, so drift falls on all.
                let mut t = [Vec::new(), Vec::new(), Vec::new()];
                for _ in 0..3 {
                    for (k, x) in [&a, &on_device, &on_cpu].into_iter().enumerate() {
                        t[k].push(median_ms(x, sub, reps));
                    }
                }
                let [dev_ms, cpu_ms] = [middle(&t[1]), middle(&t[2])];
                let ((dl, dh), (cl, ch)) = (low_high(&t[1]), low_high(&t[2]));
                let tie = dl <= ch && cl <= dh;
                let faster = if tie {
                    "either"
                } else if dev_ms < cpu_ms {
                    "hailo"
                } else {
                    "cpu"
                };
                println!(
                    "{max_seq}\t{cap}\t{what}\t{side}\t{faster}\t{}\t{}\t{}",
                    spread(&t[0]),
                    spread(&t[1]),
                    spread(&t[2])
                );
                assert!(
                    tie || side.starts_with(faster),
                    "{max_seq}x{cap} {what}: ran on {side}, the {faster} is faster"
                );
            }
        }
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
