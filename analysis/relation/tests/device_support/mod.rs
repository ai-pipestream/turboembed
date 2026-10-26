//! What the device tests of GLiNER2 and GLiREL share (#920): the tiny
//! DeBERTa-v2 bundle of testdata/tiny-deberta-bundle as the encoder, on
//! the CPU encoder from its own weights and through the backend table on
//! the device TURBO_TEST_DEVICE names (cpu, cuda, levelzero; the CPU when
//! it is not set). With TURBO_TEST_REQUIRE_CUDA or
//! TURBO_TEST_REQUIRE_LEVELZERO set, a test fails unless it runs there.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use turbo_relation::encoder::{DebertaEncoder, Device, EncoderOptions, EncoderStates};

/// The largest head output difference from the CPU encoder: the F32 device
/// bound the encoder itself is held to (#883).
pub const BOUND: f32 = 1e-4;

pub fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

pub fn bundle() -> PathBuf {
    root().join("testdata/tiny-deberta-bundle")
}

pub fn spm() -> Vec<u8> {
    std::fs::read(root().join("testdata/sentencepiece/unigram-identity.model")).unwrap()
}

/// The backend a TURBO_TEST_REQUIRE_* variable requires, if any.
pub fn required() -> Option<&'static str> {
    let set = |v: &str| std::env::var(v).is_ok_and(|v| !v.is_empty() && v != "0");
    [("TURBO_TEST_REQUIRE_CUDA", "cuda"), ("TURBO_TEST_REQUIRE_LEVELZERO", "levelzero")]
        .into_iter()
        .find(|(v, _)| set(v))
        .map(|(_, b)| b)
}

/// The tiny bundle's encoder on the CPU, from its own weights.
pub fn cpu() -> DebertaEncoder {
    let e = DebertaEncoder::open_cpu(&bundle()).unwrap();
    assert!(e.execution().is_none(), "the CPU encoder itself");
    assert_eq!(e.config().max_sequence, 128, "the bundle's max_seq");
    e
}

/// The tiny bundle through the backend table, on the device the test names,
/// checked to run on the backend required.
pub fn device() -> DebertaEncoder {
    device_on(&bundle())
}

/// The bundle at `dir` through the backend table, as [`device`] opens it.
pub fn device_on(dir: &Path) -> DebertaEncoder {
    let device = match std::env::var("TURBO_TEST_DEVICE").as_deref() {
        Err(_) | Ok("cpu") => Device::Cpu,
        Ok(d) => Device::Backend(d.to_owned()),
    };
    if let Some(b) = required() {
        assert!(!matches!(device, Device::Cpu), "{b} is required but TURBO_TEST_DEVICE names no device");
    }
    let e = DebertaEncoder::open(dir, &EncoderOptions { device, ..EncoderOptions::default() })
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    let x = e.execution().expect("the backend table");
    if let Some(b) = required() {
        assert_eq!(x.backend, b, "the session runs on the backend required");
    }
    e
}

/// min, p50, mean and p99 of `v`, in ms.
pub fn summary(v: &mut [f64]) -> String {
    v.sort_by(f64::total_cmp);
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let p = |q: f64| v[((v.len() - 1) as f64 * q).round() as usize];
    format!("min {:.3} p50 {:.3} mean {:.3} p99 {:.3}", v[0], p(0.5), mean, p(0.99))
}

/// The largest difference of two outputs of one length; equal infinities
/// (the heads' slots past the last word) count as none.
pub fn worst(a: &[f32], b: &[f32], what: &str) -> f32 {
    assert_eq!(a.len(), b.len(), "{what}: lengths");
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            if x == y {
                0.0
            } else {
                assert!(x.is_finite() && y.is_finite(), "{what}: {x} against {y}");
                (x - y).abs()
            }
        })
        .fold(0.0, f32::max)
}

pub const TEXTS: [&str; 4] = [
    "Alice visited Paris in May.",
    "Zoë",
    "Apple and Microsoft announced a joint venture in Seattle on Tuesday",
    "the quick brown fox jumps over the lazy dog while seven wizards judge a match",
];
