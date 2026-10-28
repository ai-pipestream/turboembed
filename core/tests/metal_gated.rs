//! ModernBERT and nomic-bert on Metal (#1461) against the CPU backend on the
//! sealed tiny bundles (testdata/tiny-modernbert-bundle, tiny-nomic-bundle):
//! every value within 1e-4 of the CPU's at MODEL and at EXACT, on rows long
//! enough that a local layer's window leaves keys out (the tiny ModernBERT's
//! window is 8, its positions 64, and its layer 0 is global and 1 and 2
//! local); each bundle's reference vectors through the conformance check;
//! repeated runs giving the same bits; a left-padded row finite; and
//! FASTEST refused.
//!
//! Built with the `metal` feature only, on macOS: the tests run on the
//! first metal device, and say they were skipped when the backend lists
//! none, as tests/metal.rs's do. With TURBO_TEST_REQUIRE_METAL=1 a test
//! that finds no metal device fails instead.

#![cfg(feature = "metal")]

mod common;

use std::path::PathBuf;
use std::ptr;
use std::sync::{Mutex, MutexGuard};

use common::*;
use turbo::*;

/// The largest difference allowed from the CPU: the F32 device bound.
const BOUND: f32 = 1e-4;

/// The tests take turns on the one device.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn turn() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner())
}

fn required() -> bool {
    std::env::var("TURBO_TEST_REQUIRE_METAL").is_ok_and(|v| !v.is_empty() && v != "0")
}

/// The first metal device, or None after saying the test is skipped.
/// Without TURBO_TEST_DEVICE=metal every test here is skipped; with it, a
/// runtime listing no metal device fails the test when
/// TURBO_TEST_REQUIRE_METAL is set.
fn device(test: &str) -> Option<u32> {
    match std::env::var("TURBO_TEST_DEVICE").as_deref() {
        Ok("metal") => {}
        Ok(d) => panic!("{test}: TURBO_TEST_DEVICE={d}: this test runs on metal alone"),
        Err(_) => {
            assert!(!required(), "{test}: TURBO_TEST_REQUIRE_METAL=1 needs TURBO_TEST_DEVICE=metal");
            println!("{test}: skipped: TURBO_TEST_DEVICE is not metal");
            return None;
        }
    }
    let mut rt = ptr::null_mut();
    assert_eq!(unsafe { turbo_runtime_create(ptr::null(), &mut rt, &mut new_error()) }, 0);
    let d = first_of(rt, "metal");
    unsafe { turbo_runtime_release(rt) };
    if d.is_none() {
        assert!(!required(), "{test}: TURBO_TEST_REQUIRE_METAL=1 and the metal backend lists no device");
        println!("{test}: skipped: the metal backend lists no device");
    }
    d
}

fn metal(rt: *mut turbo_runtime) -> u32 {
    first_of(rt, "metal").expect("a metal device")
}

fn bundle(name: &str) -> PathBuf {
    testdata().join(name)
}

/// The bundle on the metal device, checked to be one, and on the CPU.
fn both(name: &str) -> (Loaded, Loaded) {
    let g = Loaded::load_on(&bundle(name), metal).unwrap_or_else(|e| panic!("{e:?}"));
    let mut info: turbo_device_info = unsafe { std::mem::zeroed() };
    info.struct_size = size_of::<turbo_device_info>() as u32;
    assert_eq!(unsafe { turbo_runtime_device_info(g.rt, metal(g.rt), &mut info, ptr::null_mut()) }, 0);
    assert_eq!(field(&info.backend), "metal", "the model is on the metal backend");
    (g, Loaded::load(&bundle(name)).unwrap_or_else(|e| panic!("{e:?}")))
}

/// Token ids from the place, inside `vocab` and clear of the low ids.
fn ids(row: usize, len: usize, vocab: usize) -> Vec<i32> {
    (0..len).map(|p| 10 + ((row * 7919 + p * 104_729) % (vocab - 20)) as i32).collect()
}

/// Rows of the given lengths, right padded to the longest; `left` rows are
/// padded on the left instead, and `holes` are (row, column) masked inside
/// a row. With `typed`, every other row's tokens are type 1.
fn rows(lens: &[usize], left: &[usize], holes: &[(usize, usize)], vocab: usize, typed: bool) -> Tokens {
    let seq = *lens.iter().max().unwrap();
    let mut t = Tokens::new(&vec![vec![0; seq]; lens.len()], 0);
    t.mask.fill(0);
    let mut types = vec![0; lens.len() * seq];
    for (r, &len) in lens.iter().enumerate() {
        let start = if left.contains(&r) { seq - len } else { 0 };
        for (p, id) in ids(r, len, vocab).into_iter().enumerate() {
            t.ids[r * seq + start + p] = id;
            t.mask[r * seq + start + p] = 1;
            types[r * seq + start + p] = (typed && r % 2 == 1) as i32;
        }
    }
    for &(r, c) in holes {
        t.mask[r * seq + c] = 0;
    }
    if typed {
        t.types = Some(types);
    }
    t
}

fn options(pooling: u32, normalize: u32) -> turbo_embed_options {
    turbo_embed_options { pooling, normalize, ..embed_options() }
}

/// The largest difference between two runs' vectors; infinite where
/// either is NaN.
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

/// The bundle's cases on the device and the CPU at `precision`, every
/// value within BOUND; the largest difference of each case printed.
fn matches_the_cpu(test: &str, name: &str, vocab: usize, typed: bool, precision: u32) {
    let _t = turn();
    let Some(_) = device(test) else { return };
    let (g, c) = both(name);
    let desc = session_desc(8, 64, precision);
    let gs = Session::create(g.m, Some(&desc)).unwrap();
    let cs = Session::create(c.m, Some(&desc)).unwrap();
    assert_eq!(gs.info().compute_dtype, TURBO_DTYPE_F32);
    let r = |lens: &[usize], left: &[usize], holes: &[(usize, usize)]| rows(lens, left, holes, vocab, typed);
    let cases: [(&str, Tokens, turbo_embed_options); 7] = [
        (
            "mixed lengths, mean and L2",
            r(&[17, 5, 40, 9, 1], &[], &[]),
            options(TURBO_POOLING_MEAN, TURBO_NORMALIZE_L2),
        ),
        ("rows of 64, past the window", r(&[64, 64, 33], &[], &[]), options(TURBO_POOLING_MEAN, TURBO_NORMALIZE_NONE)),
        (
            "masked holes, mean",
            r(&[40, 12, 64], &[], &[(0, 3), (2, 30)]),
            options(TURBO_POOLING_MEAN, TURBO_NORMALIZE_NONE),
        ),
        ("CLS", r(&[64, 20, 7], &[], &[]), options(TURBO_POOLING_CLS, TURBO_NORMALIZE_NONE)),
        ("the last token", r(&[33, 2, 64], &[], &[]), options(TURBO_POOLING_LAST, TURBO_NORMALIZE_NONE)),
        ("one token, CLS and L2", r(&[1], &[], &[]), options(TURBO_POOLING_CLS, TURBO_NORMALIZE_L2)),
        ("eight rows", r(&[64, 63, 50, 33, 32, 31, 9, 2], &[], &[]), options(TURBO_POOLING_MEAN, TURBO_NORMALIZE_L2)),
    ];
    let p = if precision == TURBO_PRECISION_EXACT { "EXACT" } else { "MODEL" };
    for (what, t, o) in &cases {
        gs.write_tokens(&t.batch(), Some(o)).unwrap();
        let got = gs.run().unwrap().rows();
        cs.write_tokens(&t.batch(), Some(o)).unwrap();
        let d = worst(&got, &cs.run().unwrap().rows());
        println!("{test}: {p}: {what}: largest difference {d:e}");
        assert!(d < BOUND, "{p}: {what}: {d:e}");
    }
    // Token ranges inside rows that pass the window.
    let t = r(&[64, 40, 12], &[], &[(1, 20)]);
    let ranges = [range(0, 0, 64), range(0, 10, 30), range(1, 1, 21), range(1, 38, 40), range(2, 3, 9)];
    let o = options(0, TURBO_NORMALIZE_L2);
    gs.write_ranges(&t.batch(), &ranges, Some(&o)).unwrap();
    let got = gs.run().unwrap().rows();
    cs.write_ranges(&t.batch(), &ranges, Some(&o)).unwrap();
    let d = worst(&got, &cs.run().unwrap().rows());
    println!("{test}: {p}: token ranges: largest difference {d:e}");
    assert!(d < BOUND, "{p}: token ranges: {d:e}");
}

#[test]
fn the_tiny_modernbert_matches_the_cpu_at_model() {
    let t = "the_tiny_modernbert_matches_the_cpu_at_model";
    matches_the_cpu(t, "tiny-modernbert-bundle", 1200, false, TURBO_PRECISION_MODEL);
}

#[test]
fn the_tiny_modernbert_matches_the_cpu_at_exact() {
    let t = "the_tiny_modernbert_matches_the_cpu_at_exact";
    matches_the_cpu(t, "tiny-modernbert-bundle", 1200, false, TURBO_PRECISION_EXACT);
}

#[test]
fn the_tiny_nomic_matches_the_cpu_at_model() {
    let t = "the_tiny_nomic_matches_the_cpu_at_model";
    matches_the_cpu(t, "tiny-nomic-bundle", 30528, true, TURBO_PRECISION_MODEL);
}

#[test]
fn the_tiny_nomic_matches_the_cpu_at_exact() {
    let t = "the_tiny_nomic_matches_the_cpu_at_exact";
    matches_the_cpu(t, "tiny-nomic-bundle", 30528, true, TURBO_PRECISION_EXACT);
}

/// Each bundle's sealed reference vectors, through the conformance check
/// on the metal device, at MODEL and EXACT.
#[test]
fn the_tiny_bundles_match_their_references() {
    let _t = turn();
    let Some(_) = device("the_tiny_bundles_match_their_references") else { return };
    for name in ["tiny-modernbert-bundle", "tiny-nomic-bundle"] {
        for precision in [TURBO_PRECISION_MODEL, TURBO_PRECISION_EXACT] {
            conformance::check(&bundle(name), metal, precision);
        }
    }
}

/// Five runs of the same rows give the same bits.
#[test]
fn repeated_runs_give_the_same_bits() {
    let _t = turn();
    let Some(_) = device("repeated_runs_give_the_same_bits") else { return };
    for (name, vocab, typed) in [("tiny-modernbert-bundle", 1200, false), ("tiny-nomic-bundle", 30528, true)] {
        let (g, _c) = both(name);
        let s = Session::create(g.m, Some(&session_desc(8, 64, TURBO_PRECISION_MODEL))).unwrap();
        let t = rows(&[64, 17, 40, 1], &[], &[(0, 5)], vocab, typed);
        let bits = |v: Vec<Vec<f32>>| v.into_iter().flatten().map(f32::to_bits).collect::<Vec<u32>>();
        s.write_tokens(&t.batch(), None).unwrap();
        let first = bits(s.run().unwrap().rows());
        for run in 1..5 {
            s.write_tokens(&t.batch(), None).unwrap();
            assert_eq!(bits(s.run().unwrap().rows()), first, "{name}: run {run}");
        }
    }
}

/// A left-padded row on a sliding-window layer: its pad queries' windows
/// hold no live key, and each gets a context of 0, as on the CPU. Every
/// value is finite and every row matches the CPU's.
#[test]
fn a_left_padded_row_stays_finite() {
    let _t = turn();
    let Some(_) = device("a_left_padded_row_stays_finite") else { return };
    let (g, c) = both("tiny-modernbert-bundle");
    let desc = session_desc(8, 64, TURBO_PRECISION_MODEL);
    let (gs, cs) = (Session::create(g.m, Some(&desc)).unwrap(), Session::create(c.m, Some(&desc)).unwrap());
    let t = rows(&[17, 5, 40, 9, 1], &[3], &[], 1200, false);
    let o = options(TURBO_POOLING_MEAN, TURBO_NORMALIZE_NONE);
    gs.write_tokens(&t.batch(), Some(&o)).unwrap();
    let got = gs.run().unwrap().rows();
    assert!(got.iter().flatten().all(|v| v.is_finite()), "a value is not finite: {got:?}");
    cs.write_tokens(&t.batch(), Some(&o)).unwrap();
    let d = worst(&got, &cs.run().unwrap().rows());
    println!("a_left_padded_row_stays_finite: every row, largest difference {d:e}");
    assert!(d < BOUND, "{d:e}");
}

/// FASTEST is refused, naming the precision and the family; it never runs
/// in F32 in its place.
#[test]
fn fastest_is_refused_on_a_rotary_model() {
    let _t = turn();
    let Some(_) = device("fastest_is_refused_on_a_rotary_model") else { return };
    for (name, family) in [("tiny-modernbert-bundle", "FAMILY_MODERNBERT"), ("tiny-nomic-bundle", "FAMILY_NOMIC_BERT")]
    {
        let (g, _c) = both(name);
        let e = Session::create(g.m, Some(&session_desc(4, 64, TURBO_PRECISION_FASTEST))).err().unwrap();
        assert!(e.is(status::UNSUPPORTED_OPTION, &format!("FASTEST on {family}")), "{e:?}");
    }
}
