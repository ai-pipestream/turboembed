//! The real GLiNER checkpoint through the backend table against its own
//! package on the CPU (#883): GLINER_PACKAGE names the converted package
//! (gliner-convert) and GLINER_ENCODER_BUNDLE the core bundle sealed from
//! the same checkpoint's encoder (bundle/recipes/gliner-small-v2.1-encoder.json,
//! docs/gliner.md). The device is TURBO_TEST_DEVICE's (`cuda`, `levelzero`,
//! `metal`, `cpu`) at TURBO_TEST_PRECISION (model or exact). With
//! TURBO_TEST_REQUIRE_CUDA, TURBO_TEST_REQUIRE_LEVELZERO or TURBO_TEST_REQUIRE_METAL set, a test fails
//! unless it runs on that backend. Ignored: they need the checkpoint.

use std::path::PathBuf;
use std::time::Instant;

use turbo_analysis_embed::{Device, EncoderOptions, Precision};
use turbo_gliner::{Candidate, Decode, ExtractionSchema, Model, decode, package::Manifest};

/// The largest difference from the CPU path of a logit and of an encoder
/// row the heads read: the F32 device bound, as tests/device.rs and the
/// core's device tests hold it.
const BOUND: f32 = 1e-4;
const ROW_BOUND: f32 = 1e-4;

/// The backend a TURBO_TEST_REQUIRE_* variable requires, if any.
fn required() -> Option<&'static str> {
    let set = |v: &str| std::env::var(v).is_ok_and(|v| !v.is_empty() && v != "0");
    [
        ("TURBO_TEST_REQUIRE_CUDA", "cuda"),
        ("TURBO_TEST_REQUIRE_LEVELZERO", "levelzero"),
        ("TURBO_TEST_REQUIRE_METAL", "metal"),
    ]
    .into_iter()
    .find(|(v, _)| set(v))
    .map(|(_, b)| b)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let n = |v: &[f32]| v.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
    dot / (n(a) * n(b))
}

const LABELS: [&str; 32] = [
    "person",
    "organization",
    "location",
    "date",
    "city",
    "country",
    "company",
    "product",
    "event",
    "time",
    "money",
    "law",
    "language",
    "group",
    "animal",
    "food",
    "color",
    "sport",
    "book",
    "film",
    "song",
    "game",
    "drug",
    "disease",
    "school",
    "river",
    "planet",
    "ship",
    "car",
    "phone",
    "street",
    "award",
];
const TEXT: [&str; 8] = ["the", "cat", "sat", "on", "a", "mat", "and", "slept"];
const SENTENCE: &str = "Barack Obama met Angela Merkel in Berlin on 3 May 2015 to discuss the Paris climate accord , \
                        while Apple announced a new iPhone in Cupertino .";

fn env_path(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("{name} names the directory")))
}

fn options() -> EncoderOptions {
    let device = match std::env::var("TURBO_TEST_DEVICE").as_deref() {
        Err(_) | Ok("cpu") => Device::Cpu,
        Ok(d) => Device::Backend(d.to_owned()),
    };
    let precision = match std::env::var("TURBO_TEST_PRECISION").as_deref() {
        Err(_) | Ok("model") => Precision::Model,
        Ok("exact") => Precision::Exact,
        Ok(p) => panic!("TURBO_TEST_PRECISION={p}: model or exact"),
    };
    EncoderOptions { device, precision, ..EncoderOptions::default() }
}

/// The package on the CPU, and the same over the encoder bundle on the
/// device.
fn models() -> (Model, Model) {
    let (package, bundle) = (env_path("GLINER_PACKAGE"), env_path("GLINER_ENCODER_BUNDLE"));
    let o = options();
    if let Some(b) = required() {
        assert!(!matches!(o.device, Device::Cpu), "{b} is required but TURBO_TEST_DEVICE names no device");
    }
    let cpu = Manifest::load(&package).unwrap();
    let device = Manifest::open_on(&package, &bundle, &o).unwrap();
    if let Some(b) = required() {
        assert_eq!(device.execution().unwrap().backend, b, "the session runs on the backend required");
    }
    (cpu, device)
}

fn schema(labels: usize) -> ExtractionSchema {
    ExtractionSchema::new(LABELS[..labels].iter().map(|s| s.to_string()).collect(), 0.5, Decode::Flat).unwrap()
}

fn words(n: usize) -> Vec<&'static str> {
    (0..n).map(|i| TEXT[i % TEXT.len()]).collect()
}

#[test]
#[ignore = "needs GLINER_PACKAGE and GLINER_ENCODER_BUNDLE; see docs/gliner.md"]
fn the_real_checkpoint_on_the_backend_table_scores_as_its_package() {
    let (cpu, device) = models();
    let execution = device.execution().unwrap().clone();
    let sentence: Vec<&str> = SENTENCE.split(' ').collect();
    let (mut worst, mut row_worst, mut lowest) = (0f32, 0f32, 1f64);
    for (text, labels) in [(sentence.clone(), 4), (sentence, 32), (words(128), 4), (words(384), 32)] {
        let s = schema(labels);
        let p = cpu.pack(&text, &s).unwrap();
        assert_eq!(device.pack(&text, &s).unwrap().ids, p.ids, "packed ids");
        // The rows the heads read, compared directly.
        let (a, b) = (cpu.states(&p, &s).unwrap(), device.states(&p, &s).unwrap());
        assert_eq!(a.len(), b.len(), "states");
        let width = a.len() / (p.prompts.len() + p.words.len());
        assert_eq!(width * (p.prompts.len() + p.words.len()), a.len(), "a state a prompt or word");
        for (x, y) in a.chunks(width).zip(b.chunks(width)) {
            let d = x.iter().zip(y).map(|(u, v)| (u - v).abs()).fold(0f32, f32::max);
            row_worst = row_worst.max(d);
            lowest = lowest.min(cosine(x, y));
        }
        let (want, got) = (cpu.score_packed(&p, &s).unwrap(), device.score_packed(&p, &s).unwrap());
        assert_eq!(want.len(), got.len());
        for (a, b) in want.iter().zip(&got) {
            assert_eq!((a.start, a.end, a.label), (b.start, b.end, b.label));
            worst = worst.max((a.logit - b.logit).abs());
        }
        // Decoded entities alike, but for a candidate within the bound of
        // the threshold on either side.
        let near = |c: &Candidate| (c.score - s.threshold()).abs() < BOUND;
        let keep = |v: Vec<Candidate>| -> Vec<(usize, usize, usize)> {
            decode(v, &s).iter().filter(|c| !near(c)).map(|c| (c.start, c.end, c.label)).collect()
        };
        assert_eq!(keep(want), keep(got), "{} words, {labels} labels: decoded entities", text.len());
    }
    println!(
        "gliner_small-v2.1 on {} {} ({}): marker and first-subword rows: largest difference {row_worst:e}, \
         lowest cosine {lowest:.9}; largest logit difference {worst:e}",
        execution.backend, execution.device, execution.precision
    );
    assert!(row_worst < ROW_BOUND && lowest >= 0.9999, "rows: {row_worst:e}, cosine {lowest}");
    assert!(worst < BOUND, "largest logit difference {worst:e}");
}

/// min, p50, mean and p99 of milliseconds.
fn stats(mut ms: Vec<f64>) -> String {
    ms.sort_by(f64::total_cmp);
    let n = ms.len();
    let at = |q: f64| ms[((q * (n - 1) as f64).round() as usize).min(n - 1)];
    format!("min {:.2} p50 {:.2} mean {:.2} p99 {:.2}", ms[0], at(0.5), ms.iter().sum::<f64>() / n as f64, at(0.99))
}

/// score_packed, and the encoder alone (states), on the package's CPU
/// encoder and on the device, interleaved, GLINER_REPS runs each (default
/// 50) after one untimed.
#[test]
#[ignore = "needs GLINER_PACKAGE and GLINER_ENCODER_BUNDLE; see docs/gliner.md"]
fn real_checkpoint_timing() {
    let (cpu, device) = models();
    let execution = device.execution().unwrap().clone();
    let reps: usize = std::env::var("GLINER_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(50);
    for n in [64, 128, 384] {
        for labels in [4, 32] {
            let s = schema(labels);
            let text = words(n);
            let p = cpu.pack(&text, &s).unwrap();
            // The full call, and the encoder alone (the rows the heads read),
            // on each side: the difference is the heads' share.
            let mut times = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
            for run in 0..=reps {
                for (i, m) in [&cpu, &device].into_iter().enumerate() {
                    let t = Instant::now();
                    std::hint::black_box(m.score_packed(&p, &s).unwrap());
                    let full = t.elapsed().as_secs_f64() * 1e3;
                    let t = Instant::now();
                    std::hint::black_box(m.states(&p, &s).unwrap());
                    let encoder = t.elapsed().as_secs_f64() * 1e3;
                    if run > 0 {
                        times[i].push(full);
                        times[2 + i].push(encoder);
                    }
                }
            }
            let [c, d, ce, de] = times;
            let ids = p.ids.len();
            let (backend, precision) = (&execution.backend, &execution.precision);
            println!("timing words={n} labels={labels} ids={ids} cpu score_packed: {}", stats(c));
            println!("timing words={n} labels={labels} ids={ids} cpu encoder: {}", stats(ce));
            println!("timing words={n} labels={labels} ids={ids} {backend} ({precision}) score_packed: {}", stats(d));
            println!("timing words={n} labels={labels} ids={ids} {backend} ({precision}) encoder: {}", stats(de));
        }
    }
}
