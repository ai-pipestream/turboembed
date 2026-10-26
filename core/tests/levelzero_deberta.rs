//! DeBERTa-v2 on Level Zero against the CPU backend on the same bundle:
//! the tiny model of tests/deberta_load.rs and one at GLiNER-small's size,
//! each value within 1e-4 of the CPU's at MODEL; and FASTEST refused.
//!
//! Built with the `levelzero` feature only. TURBO_TEST_DEVICE=levelzero
//! runs the tests on the first levelzero device; without it they say they
//! were skipped, as the other device tests do. With
//! TURBO_TEST_REQUIRE_LEVELZERO=1 a test that finds no levelzero device,
//! or is not told to use one, fails instead.

#![cfg(feature = "levelzero")]

mod common;
#[path = "common/deberta.rs"]
mod deberta_common;

use std::ptr;
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

use common::*;
use deberta_common::{SMALL, Shape, TINY, fixture};
use turbo::*;

/// The largest difference allowed from the CPU: the F32 device bound.
const BOUND: f32 = 1e-4;

/// The tests take turns on the one device.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn turn() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner())
}

fn required() -> bool {
    std::env::var("TURBO_TEST_REQUIRE_LEVELZERO").is_ok_and(|v| !v.is_empty() && v != "0")
}

/// The levelzero device TURBO_TEST_DEVICE asks for, or None after saying
/// the test is skipped.
fn device(test: &str) -> Option<u32> {
    match std::env::var("TURBO_TEST_DEVICE").as_deref() {
        Ok("levelzero") => {}
        Ok(d) => panic!("{test}: TURBO_TEST_DEVICE={d}: this test runs on levelzero alone"),
        Err(_) => {
            assert!(!required(), "{test}: TURBO_TEST_REQUIRE_LEVELZERO=1 and TURBO_TEST_DEVICE is not levelzero");
            println!("{test}: skipped: TURBO_TEST_DEVICE is not set");
            return None;
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
    d
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

/// Token ids from the place, inside the vocabulary and clear of the
/// special tokens.
fn ids(row: usize, len: usize) -> Vec<i32> {
    (0..len).map(|p| 1000 + ((row * 7919 + p * 104_729) % 28_000) as i32).collect()
}

/// Rows of the given lengths, right padded to the longest; `left` rows
/// are padded on the left instead, and `holes` are (row, column) masked
/// inside a row.
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

/// The largest difference between two runs' vectors, which must agree in
/// count and width.
fn worst(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .flat_map(|(x, y)| {
            assert_eq!(x.len(), y.len());
            x.iter().zip(y).map(|(p, q)| (p - q).abs())
        })
        .fold(0f32, f32::max)
}

/// Each case of `s`'s bundle on the device and the CPU at MODEL, every
/// value within BOUND; the largest difference of each case printed.
fn matches_the_cpu(test: &str, name: &str, s: &Shape) {
    let _t = turn();
    let Some(_) = device(test) else { return };
    let f = fixture(name, s);
    let (g, c) = (on_gpu(&f), f.load().unwrap());
    let desc = session_desc(8, 512, TURBO_PRECISION_MODEL);
    let (gs, cs) = (Session::create(g.m, Some(&desc)).unwrap(), Session::create(c.m, Some(&desc)).unwrap());
    assert_eq!(gs.info().compute_dtype, TURBO_DTYPE_F32);
    let mixed = rows(&[17, 5, 40, 9, 1], &[3], &[(2, 11), (0, 3)]);
    let cases: [(&str, Tokens, turbo_embed_options); 6] = [
        ("mixed lengths, padded, mean and L2", rows(&[17, 5, 40, 9, 1], &[], &[]), embed_options()),
        ("left padding and masked holes, mean and L2", rows(&[17, 5, 40, 9, 1], &[3], &[(2, 11), (0, 3)]), {
            options(TURBO_POOLING_MEAN, TURBO_NORMALIZE_L2)
        }),
        ("left padding and masked holes, CLS", mixed, options(TURBO_POOLING_CLS, TURBO_NORMALIZE_NONE)),
        ("one token, CLS and L2", rows(&[1], &[], &[]), options(TURBO_POOLING_CLS, TURBO_NORMALIZE_L2)),
        ("a row of 512 tokens", rows(&[512, 3], &[], &[]), embed_options()),
        ("the last token", rows(&[33, 2, 12], &[], &[(0, 31)]), options(TURBO_POOLING_LAST, TURBO_NORMALIZE_NONE)),
    ];
    for (what, t, o) in &cases {
        gs.write_tokens(&t.batch(), Some(o)).unwrap();
        let got = gs.run().unwrap().rows();
        cs.write_tokens(&t.batch(), Some(o)).unwrap();
        let want = cs.run().unwrap().rows();
        let d = worst(&got, &want);
        println!("{test}: {what}: largest difference {d:e}");
        assert!(d < BOUND, "{what}: {d:e}");
    }
    // Token ranges: whole rows, pieces, a masked hole inside one, and one
    // past a row's end, pooled by the mean.
    let t = rows(&[40, 12, 512], &[], &[(0, 20)]);
    let ranges =
        [range(0, 0, 40), range(0, 1, 21), range(1, 3, 9), range(1, 11, 12), range(2, 100, 512), range(2, 500, 512)];
    let o = options(0, TURBO_NORMALIZE_L2);
    gs.write_ranges(&t.batch(), &ranges, Some(&o)).unwrap();
    let got = gs.run().unwrap().rows();
    cs.write_ranges(&t.batch(), &ranges, Some(&o)).unwrap();
    let d = worst(&got, &cs.run().unwrap().rows());
    println!("{test}: token ranges: largest difference {d:e}");
    assert!(d < BOUND, "token ranges: {d:e}");
}

#[test]
fn the_tiny_deberta_matches_the_cpu() {
    matches_the_cpu("the_tiny_deberta_matches_the_cpu", "lz-deberta-tiny", &TINY);
}

#[test]
fn a_gliner_small_sized_deberta_matches_the_cpu() {
    matches_the_cpu("a_gliner_small_sized_deberta_matches_the_cpu", "lz-deberta-small", &SMALL);
}

/// FASTEST is refused, naming the precision and the family; it never runs
/// in F32 in its place.
#[test]
fn fastest_is_refused_on_a_deberta_model() {
    let _t = turn();
    let Some(_) = device("fastest_is_refused_on_a_deberta_model") else { return };
    let f = fixture("lz-deberta-fastest", &TINY);
    let g = on_gpu(&f);
    let e = Session::create(g.m, Some(&session_desc(4, 64, TURBO_PRECISION_FASTEST))).err().unwrap();
    assert!(e.is(status::UNSUPPORTED_OPTION, "FASTEST on FAMILY_DEBERTA_V2"), "{e:?}");
    let exact = Session::create(g.m, Some(&session_desc(4, 64, TURBO_PRECISION_EXACT))).unwrap();
    assert_eq!(exact.info().compute_dtype, TURBO_DTYPE_F32);
}

/// Milliseconds a pass at the GLiNER-small size, the CPU at MODEL and
/// levelzero at MODEL interleaved, TURBO_SPEED_RUNS runs a side (default
/// 5) after one untimed, at batch 32 of 128 tokens and batch 8 of 512.
/// Ignored, since it measures rather than checks.
#[test]
#[ignore]
fn deberta_speed() {
    let _t = turn();
    let Some(_) = device("deberta_speed") else { return };
    let runs: usize = std::env::var("TURBO_SPEED_RUNS").ok().and_then(|r| r.parse().ok()).unwrap_or(5);
    let f = fixture("lz-deberta-speed", &SMALL);
    let (g, c) = (on_gpu(&f), f.load().unwrap());
    for (batch, seq) in [(32usize, 128usize), (8, 512)] {
        let desc = session_desc(batch as u32, seq as u32, TURBO_PRECISION_MODEL);
        let sides = [
            ("cpu", Session::create(c.m, Some(&desc)).unwrap()),
            ("levelzero", { Session::create(g.m, Some(&desc)).unwrap() }),
        ];
        let t = rows(&vec![seq; batch], &[], &[]);
        let mut times = [Vec::new(), Vec::new()];
        for run in 0..=runs {
            for (i, (_, s)) in sides.iter().enumerate() {
                let start = Instant::now();
                s.write_tokens(&t.batch(), None).unwrap();
                let _ = s.run().unwrap().rows();
                if run > 0 {
                    times[i].push(start.elapsed().as_secs_f64() * 1e3);
                }
            }
        }
        for ((name, _), mut ms) in sides.iter().zip(times) {
            ms.sort_by(f64::total_cmp);
            println!(
                "deberta_speed: {name} batch {batch} seq {seq}: ms a pass min {:.2} median {:.2} max {:.2} over {runs}",
                ms[0],
                ms[ms.len() / 2],
                ms[ms.len() - 1]
            );
        }
    }
}
