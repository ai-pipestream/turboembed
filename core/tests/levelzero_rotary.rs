//! ModernBERT and nomic-bert (#1461) on Level Zero against the CPU backend
//! on the same generated bundle (tests/common/rotary.rs): each value within
//! 1e-4 of the CPU's at MODEL, EXACT and FASTEST, over rows that reach
//! past ModernBERT's window, masked holes, left padding, every pooling and
//! token ranges. The generated bundles' widths (8 and 12) are not
//! multiples of the matrix engines' steps, so FASTEST computes them in
//! F32; the F16 path at FASTEST is held to upstream by the conformance
//! test on the real bundles, and here on the overflow bundles at widths
//! 32 and 64, whose gated middle product passes F16's range.
//!
//! Built with the `levelzero` feature only. TURBO_TEST_DEVICE=levelzero
//! runs the tests on the first levelzero device; without it they say they
//! were skipped, as the other device tests do. With
//! TURBO_TEST_REQUIRE_LEVELZERO=1 a test that finds no levelzero device,
//! or is not told to use one, fails instead.

#![cfg(feature = "levelzero")]

mod common;
#[path = "common/rotary.rs"]
mod rotary;

use std::ptr;
use std::sync::{Mutex, MutexGuard};

use common::*;
use serde_json::json;
use turbo::*;

/// The largest difference allowed from the CPU: the F32 device bound.
const BOUND: f32 = 1e-4;

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn turn() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner())
}

fn required() -> bool {
    std::env::var("TURBO_TEST_REQUIRE_LEVELZERO").is_ok_and(|v| !v.is_empty() && v != "0")
}

/// Whether the test runs: TURBO_TEST_DEVICE names levelzero and the
/// backend lists a device; else it says it was skipped.
fn device(test: &str) -> bool {
    match std::env::var("TURBO_TEST_DEVICE").as_deref() {
        Ok("levelzero") => {}
        Ok(d) => panic!("{test}: TURBO_TEST_DEVICE={d}: this test runs on levelzero alone"),
        Err(_) => {
            assert!(!required(), "{test}: TURBO_TEST_REQUIRE_LEVELZERO=1 and TURBO_TEST_DEVICE is not levelzero");
            println!("{test}: skipped: TURBO_TEST_DEVICE is not set");
            return false;
        }
    }
    let mut rt = ptr::null_mut();
    assert_eq!(unsafe { turbo_runtime_create(ptr::null(), &mut rt, &mut new_error()) }, 0);
    let d = first_of(rt, "levelzero");
    unsafe { turbo_runtime_release(rt) };
    if d.is_none() {
        assert!(!required(), "{test}: TURBO_TEST_REQUIRE_LEVELZERO=1 and the levelzero backend lists no device");
        println!("{test}: skipped: the levelzero backend lists no device");
    }
    d.is_some()
}

/// The bundle on the levelzero device, checked to be one.
fn on_gpu(f: &Fixture) -> Loaded {
    let l = f.load_on(|rt| first_of(rt, "levelzero").expect("a levelzero device")).unwrap_or_else(|e| panic!("{e:?}"));
    let mut info: turbo_device_info = unsafe { std::mem::zeroed() };
    info.struct_size = size_of::<turbo_device_info>() as u32;
    let at = first_of(l.rt, "levelzero").unwrap();
    assert_eq!(unsafe { turbo_runtime_device_info(l.rt, at, &mut info, ptr::null_mut()) }, 0);
    assert_eq!(field(&info.backend), "levelzero", "the session runs on the levelzero backend");
    l
}

fn ids(row: usize, len: usize) -> Vec<i32> {
    (0..len).map(|p| 1000 + ((row * 7919 + p * 104_729) % 28_000) as i32).collect()
}

/// Rows of the given lengths, right padded to the longest; `left` rows
/// padded on the left instead, and `holes` (row, column) masked.
fn rows(lens: &[usize], left: &[usize], holes: &[(usize, usize)]) -> Tokens {
    let seq = *lens.iter().max().unwrap();
    let mut t = Tokens::new(&vec![vec![0; seq]; lens.len()], 0);
    t.mask.fill(0);
    for (r, &len) in lens.iter().enumerate() {
        let start = if left.contains(&r) { seq - len } else { 0 };
        for (p, id) in ids(r, len).into_iter().enumerate() {
            t.ids[r * seq + start + p] = id;
            t.mask[r * seq + start + p] = 1;
        }
    }
    for &(r, c) in holes {
        t.mask[r * seq + c] = 0;
    }
    t
}

fn options(pooling: u32, normalize: u32) -> turbo_embed_options {
    turbo_embed_options { pooling, normalize, ..embed_options() }
}

fn worst(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .flat_map(|(x, y)| {
            assert_eq!(x.len(), y.len());
            x.iter().zip(y).map(|(p, q)| if p.is_nan() || q.is_nan() { f32::INFINITY } else { (p - q).abs() })
        })
        .fold(0f32, f32::max)
}

/// `f`'s bundle on the device and on the CPU, every case within BOUND at
/// MODEL, EXACT and FASTEST, each computing in F32 at these widths.
fn matches_the_cpu(test: &str, mut f: Fixture) {
    let _t = turn();
    if !device(test) {
        return;
    }
    f.manifest["artifacts"][0]["backends"] = json!(["levelzero", "cpu"]);
    let (g, c) = (on_gpu(&f), f.load().unwrap());
    let seq = 64usize;
    for precision in [TURBO_PRECISION_MODEL, TURBO_PRECISION_EXACT, TURBO_PRECISION_FASTEST] {
        let desc = session_desc(8, seq as u32, precision);
        let gs = Session::create(g.m, Some(&desc)).unwrap();
        assert_eq!(gs.info().compute_dtype, TURBO_DTYPE_F32, "a rotary session computes in F32");
        let cs = Session::create(c.m, Some(&desc)).unwrap();
        let mixed = rows(&[17, 5, 40, 9, 1], &[3], &[(2, 11), (0, 3)]);
        let cases: [(&str, Tokens, turbo_embed_options); 6] = [
            ("mixed lengths, padded, mean and L2", rows(&[17, 5, 40, 9, 1], &[], &[]), embed_options()),
            ("left padding and masked holes, mean", mixed, options(TURBO_POOLING_MEAN, TURBO_NORMALIZE_NONE)),
            (
                "left padding and masked holes, CLS",
                rows(&[17, 5, 40, 9, 1], &[3], &[(2, 11), (0, 3)]),
                options(TURBO_POOLING_CLS, TURBO_NORMALIZE_NONE),
            ),
            ("one token, CLS and L2", rows(&[1], &[], &[]), options(TURBO_POOLING_CLS, TURBO_NORMALIZE_L2)),
            ("a row of max_seq tokens", rows(&[seq, 3], &[], &[]), embed_options()),
            ("the last token", rows(&[33, 2, 12], &[], &[(0, 31)]), options(TURBO_POOLING_LAST, TURBO_NORMALIZE_NONE)),
        ];
        for (what, t, o) in &cases {
            gs.write_tokens(&t.batch(), Some(o)).unwrap();
            let got = gs.run().unwrap().rows();
            cs.write_tokens(&t.batch(), Some(o)).unwrap();
            let want = cs.run().unwrap().rows();
            let d = worst(&got, &want);
            println!("{test}: precision {precision}: {what}: largest difference {d:e}");
            assert!(d < BOUND, "precision {precision}: {what}: {d:e}");
        }
        let t = rows(&[40, 12, seq], &[], &[(0, 20)]);
        let ranges =
            [range(0, 0, 40), range(0, 1, 21), range(1, 3, 9), range(2, 30, seq as u32), range(2, 50, seq as u32)];
        let o = options(0, TURBO_NORMALIZE_L2);
        gs.write_ranges(&t.batch(), &ranges, Some(&o)).unwrap();
        let got = gs.run().unwrap().rows();
        cs.write_ranges(&t.batch(), &ranges, Some(&o)).unwrap();
        let d = worst(&got, &cs.run().unwrap().rows());
        println!("{test}: precision {precision}: token ranges: largest difference {d:e}");
        assert!(d < BOUND, "precision {precision}: token ranges: {d:e}");
    }
}

#[test]
fn a_modernbert_matches_the_cpu() {
    matches_the_cpu("a_modernbert_matches_the_cpu", rotary::modernbert("lz-rotary-modernbert"));
}

#[test]
fn a_nomic_bert_matches_the_cpu() {
    matches_the_cpu("a_nomic_bert_matches_the_cpu", rotary::nomic("lz-rotary-nomic"));
}

fn least_cosine(a: &[Vec<f32>], b: &[Vec<f32>]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let dot: f64 = x.iter().zip(y).map(|(p, q)| *p as f64 * *q as f64).sum();
            let n = |v: &[f32]| v.iter().map(|p| *p as f64 * *p as f64).sum::<f64>().sqrt();
            dot / (n(x) * n(y))
        })
        .fold(1f64, f64::min)
}

/// The overflow fixtures (tests/common/rotary.rs), at widths 32 and 64
/// that FASTEST runs in F16 on the matrix engines, whose layer 0 gated
/// middle product passes F16's 65504 on every token while finite in F32:
/// narrowed to F16 for the feed-forward output it saturates at 65504, as
/// on cuda and metal, so every value is finite. The CPU computes the
/// middle in F32 without saturating, so the rows are held to it by
/// cosine: nomic-bert's post-norm LayerNorm takes the saturated sum back
/// to the CPU's within 0.999, while ModernBERT's pre-norm residual carries
/// 65504 in place of 90000 into the later layers, still at 0.75. Without
/// the clamp every value was NaN.
#[test]
fn a_gated_middle_past_f16_s_range_stays_finite_at_fastest() {
    let test = "a_gated_middle_past_f16_s_range_stays_finite_at_fastest";
    let _t = turn();
    if !device(test) {
        return;
    }
    for (name, mut f, floor) in [
        ("modernbert", rotary::overflow_modernbert("lz-rotary-overflow-mb"), 0.75),
        ("nomic", rotary::overflow_nomic("lz-rotary-overflow-nb"), 0.999),
    ] {
        f.manifest["artifacts"][0]["backends"] = json!(["levelzero", "cpu"]);
        let (g, c) = (on_gpu(&f), f.load().unwrap());
        let desc = session_desc(4, 64, TURBO_PRECISION_FASTEST);
        let gs = Session::create(g.m, Some(&desc)).unwrap();
        assert_eq!(gs.info().compute_dtype, TURBO_DTYPE_F16, "{name}: FASTEST takes the F16 path");
        let cs = Session::create(c.m, Some(&desc)).unwrap();
        for lens in [&[9usize, 1, 32][..], &[32usize, 32][..], &[64usize, 33, 17][..]] {
            let t = rows(lens, &[], &[]);
            gs.write_tokens(&t.batch(), None).unwrap();
            let got = gs.run().unwrap().rows();
            let bad = got.iter().flatten().filter(|v| !v.is_finite()).count();
            assert_eq!(bad, 0, "{name}: rows {lens:?}: {bad} values not finite");
            cs.write_tokens(&t.batch(), None).unwrap();
            let cos = least_cosine(&got, &cs.run().unwrap().rows());
            println!("{test}: {name}: rows {lens:?}: least cosine to the CPU {cos}");
            assert!(cos >= floor, "{name}: rows {lens:?}: least cosine to the CPU {cos}, under {floor}");
        }
    }
}
