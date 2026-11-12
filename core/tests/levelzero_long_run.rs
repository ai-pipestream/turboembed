//! A run longer than the compute engine's job timeout on Level Zero
//! (#1765): one run of a generated ModernBERT whose device time passes
//! `job_timeout_ms` of the card's compute engine completes with finite
//! vectors, and the same session then runs again. The kernel's job
//! timeout does not bound a run: the Intel runtime keeps its queues in
//! long-running mode, where the driver applies no job timeout
//! (docs/levelzero.md, A lost device).
//!
//! The model's head width (48) is one the tiled attention does not take,
//! so its three global layers run the general kernel, a group per query,
//! and 24 rows of 4096 tokens keep the device for about eleven seconds
//! on an Arc Pro B70.
//!
//! Built with the `levelzero` feature only. TURBO_TEST_DEVICE=levelzero
//! runs the test on the first levelzero device; without it the test says
//! it was skipped. With TURBO_TEST_REQUIRE_LEVELZERO=1 a test that finds
//! no levelzero device, or is not told to use one, fails instead.

#![cfg(feature = "levelzero")]

mod common;
#[path = "common/rotary.rs"]
mod rotary;

use std::ptr;
use std::time::Instant;

use common::*;
use serde_json::json;
use turbo::*;

const TEST: &str = "a_run_past_the_job_timeout_keeps_the_device";
const ROWS: usize = 24;
const SEQ: usize = 4096;

fn required() -> bool {
    std::env::var("TURBO_TEST_REQUIRE_LEVELZERO").is_ok_and(|v| !v.is_empty() && v != "0")
}

/// Whether the test runs: TURBO_TEST_DEVICE names levelzero and the
/// backend lists a device; else it says it was skipped.
fn device() -> bool {
    match std::env::var("TURBO_TEST_DEVICE").as_deref() {
        Ok("levelzero") => {}
        Ok(d) => panic!("{TEST}: TURBO_TEST_DEVICE={d}: this test runs on levelzero alone"),
        Err(_) => {
            assert!(!required(), "{TEST}: TURBO_TEST_REQUIRE_LEVELZERO=1 and TURBO_TEST_DEVICE is not levelzero");
            println!("{TEST}: skipped: TURBO_TEST_DEVICE is not set");
            return false;
        }
    }
    let mut rt = ptr::null_mut();
    assert_eq!(unsafe { turbo_runtime_create(ptr::null(), &mut rt, &mut new_error()) }, 0);
    let d = first_of(rt, "levelzero");
    unsafe { turbo_runtime_release(rt) };
    if d.is_none() {
        assert!(!required(), "{TEST}: TURBO_TEST_REQUIRE_LEVELZERO=1 and the levelzero backend lists no device");
        println!("{TEST}: skipped: the levelzero backend lists no device");
    }
    d.is_some()
}

/// The longest job timeout, in ms, of a compute engine of the cards the
/// `xe` driver drives, from `/sys/class/drm`; None where no card says.
fn job_timeout_ms() -> Option<u64> {
    let cards = std::fs::read_dir("/sys/class/drm").ok()?;
    cards
        .flatten()
        .filter_map(|card| {
            let at = card.path().join("device/tile0/gt0/engines/ccs/job_timeout_ms");
            std::fs::read_to_string(at).ok()?.trim().parse().ok()
        })
        .max()
}

fn tokens(rows: usize, seq: usize) -> Tokens {
    let ids = |r: usize| (0..seq).map(|p| 1000 + ((r * 7919 + p * 104_729) % 28_000) as i32).collect();
    Tokens::new(&(0..rows).map(ids).collect::<Vec<Vec<i32>>>(), 0)
}

#[test]
fn a_run_past_the_job_timeout_keeps_the_device() {
    if !device() {
        return;
    }
    let Some(timeout) = job_timeout_ms() else {
        assert!(!required(), "{TEST}: TURBO_TEST_REQUIRE_LEVELZERO=1 and no xe card gives a compute job timeout");
        println!("{TEST}: skipped: no xe card gives a compute job timeout");
        return;
    };
    let mut f = rotary::long_modernbert("lz-long-run", 384, 8, (128, 1), SEQ as u64);
    f.manifest["artifacts"][0]["backends"] = json!(["levelzero", "cpu"]);
    let g = f.load_on(|rt| first_of(rt, "levelzero").expect("a levelzero device")).unwrap_or_else(|e| panic!("{e:?}"));
    let s = Session::create(g.m, Some(&session_desc(ROWS as u32, SEQ as u32, TURBO_PRECISION_MODEL))).unwrap();
    let o = turbo_embed_options { pooling: TURBO_POOLING_MEAN, normalize: TURBO_NORMALIZE_NONE, ..embed_options() };

    let long = tokens(ROWS, SEQ);
    let start = Instant::now();
    s.write_tokens(&long.batch(), Some(&o)).unwrap();
    let out = s.run().unwrap_or_else(|e| panic!("{TEST}: the long run: {e:?}"));
    let rows = out.rows();
    let ms = start.elapsed().as_millis() as u64;
    drop(out);
    println!("{TEST}: {ROWS} rows of {SEQ} tokens: {ms} ms on the device, job timeout {timeout} ms");
    assert_eq!(rows.len(), ROWS);
    assert!(rows.iter().flatten().all(|v| v.is_finite()), "the long run's vectors are finite");
    assert!(ms > timeout, "the run took {ms} ms, within the job timeout of {timeout} ms: it bounds nothing");

    // The session and its context still run.
    let short = tokens(2, 64);
    s.write_tokens(&short.batch(), Some(&o)).unwrap();
    let again = s.run().unwrap_or_else(|e| panic!("{TEST}: the run after it: {e:?}")).rows();
    assert!(again.len() == 2 && again.iter().flatten().all(|v| v.is_finite()));
}
