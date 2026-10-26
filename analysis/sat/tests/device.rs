//! The SaT step on a device against the same step on the CPU with the
//! same bundle. TURBO_TEST_DEVICE names the device and
//! TURBO_TEST_PRECISION its tier, as for tests/parity.rs; the CPU runs at
//! `model`. Every token's span must be the CPU's, its boundary
//! probability within the tier's bound of the CPU's, and over the
//! threshold where the CPU's is, except tokens whose CPU probability is
//! within that bound of the threshold, which are counted and listed. The
//! sentences of a text with no such token must be the CPU's. With no
//! device named both sides run on the CPU.
//!
//! TURBO_TEST_SAT_BUNDLE adds a second bundle, as for tests/parity.rs
//! (the released `sat-1l-sm`, one layer of width 768).

use std::path::{Path, PathBuf};

use turbo::bundle::Bundle;
use turbo_analysis::{Annotator, Span};
use turbo_analysis_embed::{Device, EncoderOptions};
use turbo_sat::{SatAnnotator, SatOptions, Scored, cuts};

mod common;

/// The largest probability difference allowed at f32: the core's
/// largest absolute difference for an f32 session's values
/// (`turbo::record::tolerance`).
const F32_BOUND: f64 = 1e-4;

/// The largest probability difference allowed at f16 and bf16, whose
/// record bounds only the cosine: the 1e-3 tests/parity.rs allows
/// against wtpsplit, which an f16 linear layer already reaches there.
const F16_BOUND: f64 = 1e-2;

/// Threshold, stride and block of the fixtures in tests/data.
const SETTINGS: [(f64, u32, u32); 3] = [(0.5, 16, 62), (0.5, 7, 30), (0.25, 62, 62)];

fn bundles() -> Vec<PathBuf> {
    let mut b = vec![Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-sat-bundle")];
    b.extend(std::env::var_os("TURBO_TEST_SAT_BUNDLE").map(PathBuf::from));
    b
}

fn texts() -> Vec<String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/texts.jsonl");
    std::fs::read_to_string(path).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect()
}

fn cpu() -> EncoderOptions {
    EncoderOptions { device: Device::Cpu, ..EncoderOptions::default() }
}

fn open(bundle: &Path, encoder: EncoderOptions, (threshold, stride, block): (f64, u32, u32)) -> SatAnnotator {
    let o = SatOptions { encoder, threshold: Some(threshold), stride, block_size: block, ..SatOptions::default() };
    SatAnnotator::open(bundle, &o).unwrap_or_else(|e| panic!("{}: {e}", bundle.display()))
}

/// The bound for the dtype the device computes in.
fn bound(a: &SatAnnotator) -> f64 {
    if a.execution().unwrap().precision.starts_with("f32") { F32_BOUND } else { F16_BOUND }
}

fn sentences(a: &SatAnnotator, text: &str, (spans, scores): &Scored) -> Vec<Span> {
    let mut out = Vec::new();
    a.sentences(text, spans, scores, &mut out);
    out
}

/// The largest probability difference, and each (text, token, CPU
/// probability) the threshold puts on the other side on the device.
struct Seen {
    largest: f64,
    near: Vec<(usize, usize, f64)>,
}

fn compare(what: &str, texts: &[String], cpu: (&SatAnnotator, &[Scored]), dev: (&SatAnnotator, &[Scored])) -> Seen {
    let (bound, threshold) = (bound(dev.0), dev.0.threshold());
    let mut seen = Seen { largest: 0.0, near: Vec::new() };
    for (t, ((c, d), text)) in cpu.1.iter().zip(dev.1).zip(texts).enumerate() {
        assert_eq!(c.0, d.0, "{what}: text {t}: the spans differ");
        let near = seen.near.len();
        for (k, (&x, &y)) in c.1.iter().zip(&d.1).enumerate() {
            let (p, q) = (f64::from(cuts::sigmoid(x)), f64::from(cuts::sigmoid(y)));
            let diff = (p - q).abs();
            assert!(diff <= bound, "{what}: text {t} token {k}: {q} on the device, {p} on the CPU, over {bound}");
            seen.largest = seen.largest.max(diff);
            if (p > threshold) != (q > threshold) {
                assert!((p - threshold).abs() <= bound, "{what}: text {t} token {k}: {q} and {p} across {threshold}");
                seen.near.push((t, k, p));
            }
        }
        if seen.near.len() == near {
            assert_eq!(sentences(cpu.0, text, c), sentences(dev.0, text, d), "{what}: text {t}: the sentences differ");
        }
    }
    seen
}

fn scored(a: &SatAnnotator, texts: &[&str]) -> Vec<Scored> {
    a.scores(texts).into_iter().map(|r| r.unwrap()).collect()
}

#[test]
fn a_device_scores_as_the_cpu_does() {
    let texts = texts();
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    for bundle in bundles() {
        for s in SETTINGS {
            let (c, d) = (open(&bundle, cpu(), s), open(&bundle, common::encoder(), s));
            let what = format!("{} at {s:?} on {:?}", bundle.display(), d.execution());
            let seen = compare(&what, &texts, (&c, &scored(&c, &refs)), (&d, &scored(&d, &refs)));
            eprintln!("{what}: largest difference {:.1e}, near the threshold {:?}", seen.largest, seen.near);
        }
    }
}

/// Documents of a few kilobytes, as a corpus holds them, read together
/// so that runs fill: with block 62 every token of a full run is a
/// range, for the tiny bundle 3968 of the 4096 the session holds. Each
/// document must read as the CPU reads it alone, with the bundle's
/// max_batch and with 3 rows a run, which splits documents across runs.
#[test]
fn windows_packed_to_capacity_read_as_alone() {
    let texts = texts();
    let docs: Vec<String> = (0..4).flat_map(|_| texts.chunks(10).map(|c| c.join("\n"))).collect();
    let refs: Vec<&str> = docs.iter().map(String::as_str).collect();
    let s = SETTINGS[2];
    for bundle in bundles() {
        let c = open(&bundle, cpu(), s);
        let alone: Vec<Scored> = refs.iter().flat_map(|d| scored(&c, &[d])).collect();
        let mut windows = Vec::new();
        for (_, scores) in &alone {
            cuts::windows(scores.len() as u32, c.block_size(), c.stride(), &mut windows);
        }
        let most = Bundle::open(&bundle).unwrap().manifest.embed.unwrap().max_batch;
        for max_batch in [0, 3] {
            let d = open(&bundle, EncoderOptions { max_batch, ..common::encoder() }, s);
            let what = format!("{} max_batch {max_batch} on {:?}", bundle.display(), d.execution());
            let full = windows.len() / if max_batch == 0 { most } else { max_batch } as usize;
            assert!(full >= 2, "{what}: {} windows fill no runs", windows.len());
            let seen = compare(&what, &docs, (&c, &alone), (&d, &scored(&d, &refs)));
            eprintln!(
                "{what}: {} windows, largest difference {:.1e}, near {:?}",
                windows.len(),
                seen.largest,
                seen.near
            );
        }
    }
}

/// A Hailo session pools no ranges, so SaT is refused there with the
/// range-pooling error and never runs on the CPU instead. Runs when
/// TURBO_TEST_DEVICE names the hailo backend.
#[test]
fn a_hailo_session_refuses_sat() {
    if std::env::var("TURBO_TEST_DEVICE").as_deref() != Ok("hailo") {
        eprintln!("TURBO_TEST_DEVICE is not hailo: nothing to check");
        return;
    }
    let o = SatOptions { encoder: common::encoder(), ..SatOptions::default() };
    let Err(e) = SatAnnotator::open(&bundles()[0], &o) else { panic!("a hailo session opened for SaT") };
    assert!(e.to_string().contains("the hailo backend pools no ranges"), "{e}");
}
