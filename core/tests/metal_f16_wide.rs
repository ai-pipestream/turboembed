//! FASTEST's F16 kernels on a generated bundle whose widths take the
//! F16 path at full tile size (#1597): hidden 128 and intermediate
//! 256, heads of 64, so every linear tile and both attention kernels
//! run whole tiles, past the small-kernel token line. The session
//! must report F16 as its compute dtype, the F16 tensors stored
//! between kernels must be nonzero and inside F16's range, and every
//! row stays within FASTEST's floor of the CPU.
//!
//! The tiny sealed bundles (widths 32 and 48) and the generated
//! fixture (8 and 16) cannot show this: their tiles are mostly
//! padding, and a backend that quietly ran them in F32 would still
//! pass a cosine check. The peaks assertion is the part that cannot
//! pass on an F32 fallback: a tensor stored in F32 reports 0.
//!
//! Built with the `metal` feature only, on macOS. A test that needs a
//! device that runs the kernels says it was skipped, and passes, when
//! the backend lists none. With TURBO_TEST_REQUIRE_METAL=1 it fails
//! instead. docs/metal.md says how to run them.

#![cfg(feature = "metal")]

mod common;
mod metal_support;

use common::*;
use metal_support::*;
use serde_json::json;
use turbo::*;

/// FASTEST's floor, as metal_f16.rs holds it.
const FASTEST_FLOOR: f64 = 0.999;

/// F16's largest finite value: a stored F16 tensor stays below it.
const F16_MAX: f32 = 65504.0;

/// A session on `m` with `choices` read in place of TURBO_METAL_CHOICES,
/// at 8 rows of 256 tokens.
fn make(m: *mut turbo_model, choices: Option<&str>, precision: u32) -> Result<Session, Failure> {
    turbo::metal::use_choices(choices);
    let s = Session::create(m, Some(&session_desc(8, 256, precision)));
    turbo::metal::use_choices(None);
    s
}

/// Rows of the given lengths, with ids spread over the vocabulary.
fn rows_of(dir: &std::path::Path, lens: &[usize]) -> Tokens {
    let vocab = Tok::create(dir).unwrap().info().vocab_size as usize;
    let rows: Vec<Vec<i32>> = lens
        .iter()
        .enumerate()
        .map(|(r, &len)| (0..len).map(|p| (1000 + (r * 131 + p * 17) % (vocab - 1000)) as i32).collect())
        .collect();
    Tokens::new(&rows, 0)
}

/// The session's vectors for `t`, each row within FASTEST_FLOOR of the
/// CPU's; returns the lowest cosine.
fn against_cpu(s: &Session, cpu: &Session, t: &Tokens, what: &str) -> f64 {
    s.write_tokens(&t.batch(), None).unwrap();
    let got = s.run().unwrap().rows();
    cpu.write_tokens(&t.batch(), None).unwrap();
    let want = cpu.run().unwrap().rows();
    assert!(got.iter().flatten().all(|v| v.is_finite()), "{what}: a value is not finite");
    let mut lowest = 1f64;
    for (r, (a, b)) in got.iter().zip(&want).enumerate() {
        let c = cosine(a, b);
        lowest = lowest.min(c);
        assert!(c >= FASTEST_FLOOR, "{what}: row {r}: cosine {c} with the cpu");
    }
    lowest
}

#[test]
fn wide_widths_take_the_f16_path_at_fastest() {
    let _t = turn();
    let Some(_) = metal_device("wide_widths_take_the_f16_path_at_fastest") else { return };
    let mut m = model_manifest();
    m["architecture"]["layers"] = json!(2);
    m["architecture"]["hidden"] = json!(128);
    m["architecture"]["heads"] = json!(2);
    m["architecture"]["intermediate"] = json!(256);
    m["embed"]["dim"] = json!(128);
    let mut f = Fixture::new("metal-f16-wide", m);
    f.weights("weights/model.safetensors", &bert_weights(2, 128, 256, 11, 0.05));
    f.write();
    let (g, c) = (on_metal(&f.dir), Loaded::load(&f.dir).unwrap());

    // The device and the session both say F16 at FASTEST and F32 at
    // the other precisions, for this model.
    let rt = std::mem::ManuallyDrop::new(Rt(g.rt));
    let i = metal(g.rt);
    for (p, dtype) in [
        (TURBO_PRECISION_MODEL, TURBO_DTYPE_F32),
        (TURBO_PRECISION_EXACT, TURBO_DTYPE_F32),
        (TURBO_PRECISION_FASTEST, TURBO_DTYPE_F16),
    ] {
        assert_eq!(rt.capability(i, p).dtype, dtype, "precision {p}");
        assert_eq!(make(g.m, None, p).unwrap().info().compute_dtype, dtype, "precision {p}");
    }

    let cpu = Session::create(c.m, Some(&session_desc(8, 256, TURBO_PRECISION_MODEL))).unwrap();
    // 306 packed tokens: past the small F16 kernel's 64, so the tiled
    // kernels run; whole 128-wide tiles over hidden and intermediate.
    let tiled = rows_of(&f.dir, &[128, 96, 65, 17]);
    let few = rows_of(&f.dir, &[9, 1, 30]);

    // The default FASTEST session, with the F16 tensors it stores
    // between kernels tracked: every one is stored (nonzero) and
    // finite in F16's range, which an F32 fallback cannot show.
    turbo::metal::track_peaks(true);
    let s = make(g.m, None, TURBO_PRECISION_FASTEST);
    turbo::metal::track_peaks(false);
    let s = s.unwrap();
    assert_eq!(field(&s.info().choices), "linear=auto;attn=auto;forced=");
    let lowest = against_cpu(&s, &cpu, &tiled, "auto, tiled shape");
    println!("auto, tiled shape: 1 - lowest cosine with the cpu {:.3e}", 1.0 - lowest);
    for (name, peak) in turbo::metal::PEAK_NAMES.iter().zip(turbo::metal::peaks()) {
        println!("  {name}: largest magnitude {peak:.1}");
        assert!(
            peak > 0.0 && peak < F16_MAX,
            "{name}: largest magnitude {peak}, not an F16 tensor stored between kernels"
        );
    }
    let lowest = against_cpu(&s, &cpu, &few, "auto, few tokens");
    println!("auto, few tokens: 1 - lowest cosine with the cpu {:.3e}", 1.0 - lowest);

    // Every F16 tile and attention kernel, forced, on the tiled shape.
    for (line, reported) in [
        ("linear=h32", "linear=h32;attn=auto;forced=linear"),
        ("linear=h64", "linear=h64;attn=auto;forced=linear"),
        ("linear=h128", "linear=h128;attn=auto;forced=linear"),
        ("linear=h32n32", "linear=h32n32;attn=auto;forced=linear"),
        ("linear=h64n32", "linear=h64n32;attn=auto;forced=linear"),
        ("attn=staged", "linear=auto;attn=staged;forced=attn"),
        ("attn=registers", "linear=auto;attn=registers;forced=attn"),
    ] {
        let s = make(g.m, Some(line), TURBO_PRECISION_FASTEST).unwrap();
        let info = s.info();
        assert_eq!((info.compute_dtype, field(&info.choices)), (TURBO_DTYPE_F16, reported.to_owned()), "{line}");
        let lowest = against_cpu(&s, &cpu, &tiled, line);
        println!("{line}, tiled shape: 1 - lowest cosine with the cpu {:.3e}", 1.0 - lowest);
    }
}
