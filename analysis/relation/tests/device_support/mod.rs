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

/// The CPU encoder of the DeBERTa package at `dir`, as production loads it.
pub fn package_cpu(dir: &Path) -> DebertaEncoder {
    let (_, e) = turbo::deberta::package::PackageManifest::load(dir)
        .unwrap_or_else(|e| panic!("{}: {}", dir.display(), e.message));
    e.into()
}

/// The bundle at `dir` bound to the DeBERTa package at `package`, on the
/// device the test names, as [`device_on`] opens it.
pub fn bound(package: &Path, dir: &Path) -> DebertaEncoder {
    let options = options();
    let e = DebertaEncoder::open_bound(package, dir, &options).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    check(&e);
    e
}

/// A GLiREL package's two encoders as the real device test opens them: the
/// CPU encoder of its `encoder/` DeBERTa package, and the bundle bound to
/// that package on the device the test names, each under the package's
/// heads and `spm`. The tiny test opens a tiny package the same way, so the
/// paths are traced without the licensed checkpoint.
pub fn glirel_pair(package: &Path, bundle: &Path, spm: &Path) -> [turbo_relation::GlirelRelations; 2] {
    let load = |e: DebertaEncoder| {
        turbo_relation::GlirelRelations::load_with(e, package, spm)
            .unwrap_or_else(|e| panic!("{}: {e}", package.display()))
    };
    let encoder = package.join("encoder");
    [load(package_cpu(&encoder)), load(bound(&encoder, bundle))]
}

/// The tiny bundle through the backend table, on the device the test names,
/// checked to run on the backend required.
pub fn device() -> DebertaEncoder {
    device_on(&bundle())
}

/// The bundle at `dir` through the backend table, as [`device`] opens it.
pub fn device_on(dir: &Path) -> DebertaEncoder {
    let e = DebertaEncoder::open(dir, &options()).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    check(&e);
    e
}

/// The options of the device the test names: TURBO_TEST_DEVICE, else the
/// CPU through the backend table.
fn options() -> EncoderOptions {
    let device = match std::env::var("TURBO_TEST_DEVICE").as_deref() {
        Err(_) | Ok("cpu") => Device::Cpu,
        Ok(d) => Device::Backend(d.to_owned()),
    };
    if let Some(b) = required() {
        assert!(!matches!(device, Device::Cpu), "{b} is required but TURBO_TEST_DEVICE names no device");
    }
    EncoderOptions { device, ..EncoderOptions::default() }
}

/// An encoder on the backend table, on the backend required.
fn check(e: &DebertaEncoder) {
    let x = e.execution().expect("the backend table");
    if let Some(b) = required() {
        assert_eq!(x.backend, b, "the session runs on the backend required");
    }
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

/// The largest difference of two outputs as a fraction of the CPU's value,
/// and of 1 below it: `|a - b| / max(1, |a|)`. Absolute at logits up to 1,
/// relative above, where an F32 logit of a few hundred has no absolute
/// 1e-4 to give (its own spacing is 3e-5 at 256 and 1.2e-4 at 1024).
pub fn worst_scaled(a: &[f32], b: &[f32], what: &str) -> f32 {
    assert_eq!(a.len(), b.len(), "{what}: lengths");
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            if x == y {
                0.0
            } else {
                assert!(x.is_finite() && y.is_finite(), "{what}: {x} against {y}");
                (x - y).abs() / x.abs().max(1.0)
            }
        })
        .fold(0.0, f32::max)
}

/// The choice of each of `rows` (logits over the choices; non-finite ones
/// are no choice) whose CPU margin, the best less the next, is over twice
/// `bound` (of the best, above a best of 1, as worst_scaled measures): the
/// decisions a device within `bound` must make the same.
/// `None` for a row whose margin is not.
pub fn decisions(rows: &[&[f32]], bound: f32) -> Vec<Option<usize>> {
    rows.iter()
        .map(|r| {
            let mut best: Option<(usize, f32)> = None;
            let mut next = f32::NEG_INFINITY;
            for (i, &v) in r.iter().enumerate().filter(|(_, v)| v.is_finite()) {
                match best {
                    Some((_, b)) if v <= b => next = next.max(v),
                    _ => {
                        if let Some((_, b)) = best {
                            next = next.max(b);
                        }
                        best = Some((i, v));
                    }
                }
            }
            best.filter(|&(_, b)| b - next > 2.0 * bound * b.abs().max(1.0)).map(|(i, _)| i)
        })
        .collect()
}

/// The device's choices against the CPU's: every decision the CPU's margin
/// holds is the device's too. Returns the choices held, one per decision;
/// a test asks for enough of them, and at least two different choices,
/// over all its inputs ([`varied`]).
pub fn same_decisions(cpu: &[&[f32]], device: &[&[f32]], bound: f32, what: &str) -> Vec<usize> {
    assert_eq!(cpu.len(), device.len(), "{what}: decisions");
    let held = decisions(cpu, bound);
    let theirs = decisions(device, 0.0);
    let mut picked = Vec::new();
    for (i, (h, t)) in held.iter().zip(&theirs).enumerate() {
        if let Some(h) = h {
            assert_eq!(Some(*h), *t, "{what}: decision {i}");
            picked.push(*h);
        }
    }
    picked
}

/// The decisions a test held over all its inputs: some, and at least two
/// different choices among them.
pub fn varied(picked: &[usize], what: &str) {
    let distinct: std::collections::BTreeSet<_> = picked.iter().collect();
    assert!(!picked.is_empty(), "{what}: no decision has a margin over twice the bound");
    assert!(distinct.len() >= 2, "{what}: the {} decisions held all pick {distinct:?}", picked.len());
}

/// A threshold halfway between the two scores either side of the median
/// of `scores`, so about half of them pass it, none within a hair of it.
pub fn midpoint(mut scores: Vec<f32>) -> f32 {
    scores.retain(|s| s.is_finite());
    scores.sort_by(f32::total_cmp);
    scores.dedup();
    let m = scores.len() / 2;
    assert!(m > 0, "too few distinct scores for a threshold");
    (scores[m - 1] + scores[m]) / 2.0
}
