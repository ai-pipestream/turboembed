//! Releasing a Hailo session after a frame did not come back (#939).
//! With TURBO_HAILO_FRAME_TIMEOUT_MS=0 a run's wait for a frame gives up
//! before the device answers, so the run fails as it does on a device that
//! stopped answering, with frames still in flight. The release must then
//! return at once instead of waiting for them, and the device must run a
//! new session afterwards.
//!
//! Its own test binary: the variable is read when a session is made, and
//! would reach the sessions of any test running beside it. Built with the
//! `hailo` feature, ignored unless asked for, with the bundle from
//! TURBO_TEST_BUNDLE (docs/hailo.md).

#![cfg(feature = "hailo")]

mod common;

use std::time::{Duration, Instant};

use common::*;
use turbo::status::{INVALID_STATE, RUNTIME};
use turbo::*;

#[test]
#[ignore = "needs a Hailo device and TURBO_TEST_BUNDLE with a HEF for it"]
fn a_session_whose_frames_did_not_come_back_is_released_at_once() {
    let bundle = std::path::PathBuf::from(std::env::var_os("TURBO_TEST_BUNDLE").expect("TURBO_TEST_BUNDLE is not set"));
    let l = Loaded::load_on(&bundle, |rt| first_of(rt, "hailo").expect("a hailo device"))
        .unwrap_or_else(|e| panic!("{e:?}"));
    let batch = l.info().max_batch;
    let texts: Vec<String> = (0..batch).map(|i| format!("sentence number {i} about item {}", i * 7 % 13)).collect();
    let texts: Vec<&str> = texts.iter().map(String::as_str).collect();

    // SAFETY: this binary's one test, so no other thread reads the
    // environment.
    unsafe { std::env::set_var("TURBO_HAILO_FRAME_TIMEOUT_MS", "0") };
    let s = Session::create(l.m, Some(&session_desc(batch, 0, TURBO_PRECISION_MODEL))).unwrap();
    unsafe { std::env::remove_var("TURBO_HAILO_FRAME_TIMEOUT_MS") };
    s.write_text(&texts, None).unwrap();
    let Err(f) = s.run() else { panic!("a frame that did not come back in 0 ms fails the run") };
    assert_eq!(f.code, RUNTIME, "{f:?}");
    let Err(f) = s.write_text(&texts[..1], None) else { panic!("the session is not used again") };
    assert_eq!(f.code, INVALID_STATE, "{f:?}");

    let t = Instant::now();
    drop(s);
    let took = t.elapsed();
    println!("released in {:.3} ms with frames in flight", took.as_secs_f64() * 1e3);
    // The device here does answer, so a release that waits for the frames
    // returns too, once they finish: on a Hailo-8 that took 163 ms, where
    // one that does not wait took 4 to 21 ms.
    assert!(took < Duration::from_millis(80), "the release took {took:?}");

    let fresh = Session::create(l.m, Some(&session_desc(batch, 0, TURBO_PRECISION_MODEL))).unwrap();
    let v = fresh.embed(&texts, None).unwrap();
    assert_eq!((v.len(), v[0].len()), (batch as usize, l.info().dim as usize));
}
