//! Releasing a Hailo session after a frame did not come back (#939).
//! With TURBO_HAILO_FRAME_TIMEOUT_MS=0 a run's wait for a frame gives up
//! before the device answers, so the run fails as it does on a device that
//! stopped answering, with frames still in flight. The release must then
//! return at once instead of waiting for them, and the device must run a
//! new session afterwards.
//!
//! The frames such a release leaves with the device are kept by the model
//! and freed once they come back, so timeouts and recoveries repeated do
//! not keep memory; and with the model released at once after such a
//! session, the last release returns as soon as the device answers.
//!
//! Its own test binary: the variable is read when a session is made, and
//! would reach the sessions of any test running beside it. Built with the
//! `hailo` feature, ignored unless asked for, with the bundle from
//! TURBO_TEST_BUNDLE (docs/hailo.md).

#![cfg(feature = "hailo")]

mod common;

use std::time::{Duration, Instant};

use common::*;
use turbo::status::{INVALID_ARGUMENT, INVALID_STATE, RUNTIME};
use turbo::*;

unsafe extern "C" {
    /// Frame sets released sessions left with the device and their models
    /// have not freed yet (core/hailo/backend.cpp).
    fn turbo_hailo_retired_frames() -> u32;
}

fn retired() -> u32 {
    unsafe { turbo_hailo_retired_frames() }
}

fn bundle() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var_os("TURBO_TEST_BUNDLE").expect("TURBO_TEST_BUNDLE is not set"))
}

/// A session made with the frame timeout set to `ms`, or its refusal.
fn session_at(l: &Loaded, batch: u32, ms: &str) -> Result<Session, Failure> {
    // SAFETY: this binary runs its one test, so no other thread reads the
    // environment.
    unsafe { std::env::set_var("TURBO_HAILO_FRAME_TIMEOUT_MS", ms) };
    let s = Session::create(l.m, Some(&session_desc(batch, 0, TURBO_PRECISION_MODEL)));
    unsafe { std::env::remove_var("TURBO_HAILO_FRAME_TIMEOUT_MS") };
    s
}

/// A run whose frames miss a timeout of 0: it fails, and the session is
/// not used again.
fn timed_out(l: &Loaded, texts: &[&str]) -> Session {
    let s = session_at(l, texts.len() as u32, "0").unwrap();
    s.write_text(texts, None).unwrap();
    let Err(f) = s.run() else { panic!("a frame that did not come back in 0 ms fails the run") };
    assert_eq!(f.code, RUNTIME, "{f:?}");
    let Err(f) = s.write_text(&texts[..1], None) else { panic!("the session is not used again") };
    assert_eq!(f.code, INVALID_STATE, "{f:?}");
    s
}

#[test]
#[ignore = "needs a Hailo device and TURBO_TEST_BUNDLE with a HEF for it"]
fn a_session_whose_frames_did_not_come_back_is_released_at_once() {
    let l = Loaded::load_on(&bundle(), |rt| first_of(rt, "hailo").expect("a hailo device"))
        .unwrap_or_else(|e| panic!("{e:?}"));
    let batch = l.info().max_batch;
    let texts: Vec<String> = (0..batch).map(|i| format!("sentence number {i} about item {}", i * 7 % 13)).collect();
    let texts: Vec<&str> = texts.iter().map(String::as_str).collect();

    // A timeout that is not a whole number of ms up to ten minutes refuses
    // the session, rather than reaching HailoRT as a negative duration.
    for ms in ["", "-1", "10x", "600001", "18446744073709551616"] {
        let Err(f) = session_at(&l, batch, ms) else { panic!("{ms:?} is refused") };
        assert_eq!(f.code, INVALID_ARGUMENT, "{ms:?}: {f:?}");
    }

    let before = retired();
    let s = timed_out(&l, &texts);
    let t = Instant::now();
    drop(s);
    let took = t.elapsed();
    println!(
        "released in {:.3} ms with frames in flight, {} frame sets kept",
        took.as_secs_f64() * 1e3,
        retired() - before
    );
    // The device here does answer, so a release that waits for the frames
    // returns too, once they finish: on a Hailo-8 that took 163 ms, where
    // one that does not wait took 4 to 21 ms.
    assert!(took < Duration::from_millis(80), "the release took {took:?}");

    let fresh = Session::create(l.m, Some(&session_desc(batch, 0, TURBO_PRECISION_MODEL))).unwrap();
    let v = fresh.embed(&texts, None).unwrap();
    assert_eq!((v.len(), v[0].len()), (batch as usize, l.info().dim as usize));
    drop(fresh);

    // Timeouts and recoveries repeated: the frames kept come back and are
    // freed when a session is made or released, so none build up.
    for _ in 0..5 {
        drop(timed_out(&l, &texts));
        let fresh = Session::create(l.m, Some(&session_desc(batch, 0, TURBO_PRECISION_MODEL))).unwrap();
        fresh.embed(&texts, None).unwrap();
    }
    let deadline = Instant::now() + Duration::from_secs(2);
    while retired() > before && Instant::now() < deadline {
        drop(Session::create(l.m, Some(&session_desc(batch, 0, TURBO_PRECISION_MODEL))).unwrap());
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(retired(), before, "frame sets kept after their frames came back");

    // The model released at once after a session whose frames had not come
    // back: the last release runs HailoRT's shutdown of the configured
    // model, which has a wait of its own; on a device that answers it
    // returns as soon as the frames do.
    let s = timed_out(&l, &texts);
    let t = Instant::now();
    drop(s);
    drop(l);
    let took = t.elapsed();
    println!("session and model released in {:.3} ms with frames in flight", took.as_secs_f64() * 1e3);
    assert!(took < Duration::from_secs(1), "the release took {took:?}");
}
