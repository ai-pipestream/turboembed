//! Group `threading`, Rust layer: distinct sessions on one model run
//! concurrently; one session is single-owner and reports `TURBO_E_BUSY`
//! instead of corrupting state (PLAN.md section 4.6). No sleeps are used:
//! every overlap is forced with a barrier or by holding a result lease.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

use turbo::abi::*;
use turbo::provider::{EmbedOptions, RunOptions, SessionDesc};
use turbo_conformance::{assert_err, read_f32, BundleKind, Target};

const TEXTS: [&str; 4] = ["hello world", "a second document", "", "héllo wörld"];

fn sequential_rows(t: &Target) -> Vec<Vec<f32>> {
    let (_m, session) = t.session(BundleKind::Embedding);
    TEXTS
        .iter()
        .map(|text| {
            session.write_text(&[text], &EmbedOptions::default()).expect("write");
            read_f32(&session.run(&RunOptions::default()).expect("run"), 0)
        })
        .collect()
}

#[test]
fn threading_two_sessions_run_concurrently_with_the_same_results() {
    let t = Target::from_env();
    let expected = sequential_rows(&t);
    let model = t.model(BundleKind::Embedding);
    let a = model.create_session(&SessionDesc::default()).expect("session a");
    let b = model.create_session(&SessionDesc::default()).expect("session b");
    let barrier = Arc::new(Barrier::new(2));

    let results: Vec<Vec<Vec<f32>>> = thread::scope(|scope| {
        let handles: Vec<_> = [a, b]
            .into_iter()
            .map(|session| {
                let barrier = barrier.clone();
                scope.spawn(move || {
                    barrier.wait();
                    let mut rows = Vec::new();
                    for _ in 0..25 {
                        for text in TEXTS {
                            session.write_text(&[text], &EmbedOptions::default()).expect("write");
                            let result = session.run(&RunOptions::default()).expect("run");
                            rows.push(read_f32(&result, 0));
                        }
                    }
                    rows
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("thread")).collect()
    });

    for rows in &results {
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(row, &expected[i % TEXTS.len()], "concurrent run {i} differs from the sequential run");
        }
    }
}

#[test]
fn threading_a_held_result_makes_another_thread_busy() {
    let t = Target::from_env();
    let (_m, session) = t.session(BundleKind::Embedding);
    session.write_text(&["hello world"], &EmbedOptions::default()).expect("write");
    // Thread A holds the lease for the whole of thread B's attempt.
    let result = session.run(&RunOptions::default()).expect("run");
    let barrier = Arc::new(Barrier::new(2));
    thread::scope(|scope| {
        let session = &session;
        let barrier2 = barrier.clone();
        let other = scope.spawn(move || {
            barrier2.wait();
            let run = session.run(&RunOptions::default());
            let write = session.write_text(&["x"], &EmbedOptions::default());
            (run.err().map(|e| e.code()), write.err().map(|e| e.code()))
        });
        barrier.wait();
        let (run, write) = other.join().expect("thread");
        assert_eq!(run, Some(TURBO_E_BUSY), "a run while a result is leased must be TURBO_E_BUSY");
        assert_eq!(write, Some(TURBO_E_BUSY), "a write while a result is leased must be TURBO_E_BUSY");
    });
    drop(result);
    session.run(&RunOptions::default()).expect("the session recovers once the lease is returned");
}

#[test]
fn threading_a_write_while_a_result_is_held_is_busy() {
    let t = Target::from_env();
    let (_m, session) = t.session(BundleKind::Embedding);
    session.write_text(&["hello world"], &EmbedOptions::default()).expect("write");
    let result = session.run(&RunOptions::default()).expect("run");
    assert_err!(session.write_text(&["y"], &EmbedOptions::default()), TURBO_E_BUSY);
    assert_err!(session.run(&RunOptions::default()), TURBO_E_BUSY);
    // Reading the result while it is held is always allowed.
    let v = read_f32(&result, 0);
    assert!(!v.is_empty());
    drop(result);
    session.write_text(&["y"], &EmbedOptions::default()).expect("write after the lease is returned");
}

#[test]
fn threading_one_session_from_many_threads_never_corrupts_or_lies() {
    let t = Target::from_env();
    let expected = sequential_rows(&t);
    let (_m, session) = t.session(BundleKind::Embedding);
    let barrier = Arc::new(Barrier::new(4));
    let busy = Arc::new(AtomicUsize::new(0));
    let ok = Arc::new(AtomicUsize::new(0));

    thread::scope(|scope| {
        for _ in 0..4 {
            let session = &session;
            let barrier = barrier.clone();
            let busy = busy.clone();
            let ok = ok.clone();
            let expected = &expected;
            scope.spawn(move || {
                barrier.wait();
                for _ in 0..200 {
                    // Every status must be OK or BUSY: never a corrupted state.
                    match session.write_text(&[TEXTS[0]], &EmbedOptions::default()) {
                        Ok(()) => {}
                        Err(e) => {
                            assert_eq!(e.code(), TURBO_E_BUSY, "unexpected write status: {}", e.code_name());
                            busy.fetch_add(1, Ordering::Relaxed);
                            continue;
                        }
                    }
                    match session.run(&RunOptions::default()) {
                        Ok(result) => {
                            ok.fetch_add(1, Ordering::Relaxed);
                            let row = read_f32(&result, 0);
                            assert_eq!(row.len(), expected[0].len(), "a concurrent run produced a wrong shape");
                        }
                        Err(e) => {
                            assert!(
                                e.code() == TURBO_E_BUSY || e.code() == TURBO_E_INVALID_STATE,
                                "unexpected run status: {} ({})",
                                e.code_name(),
                                e.message()
                            );
                            busy.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            });
        }
    });

    assert!(ok.load(Ordering::Relaxed) > 0, "no concurrent run succeeded");
    // Whatever happened, the session is still usable and still correct.
    session.write_text(&[TEXTS[0]], &EmbedOptions::default()).expect("write after the storm");
    let result = session.run(&RunOptions::default()).expect("run after the storm");
    assert_eq!(read_f32(&result, 0), expected[0], "the session was corrupted by concurrent use");
}

#[test]
fn threading_results_are_deterministic_across_runs_and_sessions() {
    let t = Target::from_env();
    if !t.has(TURBO_CAP_DETERMINISTIC) {
        println!("threading_results_are_deterministic: device does not claim TURBO_CAP_DETERMINISTIC");
        return;
    }
    let expected = sequential_rows(&t);
    // Same session, repeated runs.
    let (model, session) = t.session(BundleKind::Embedding);
    for _ in 0..10 {
        for (i, text) in TEXTS.iter().enumerate() {
            session.write_text(&[text], &EmbedOptions::default()).expect("write");
            let row = read_f32(&session.run(&RunOptions::default()).expect("run"), 0);
            assert_eq!(row, expected[i], "run-to-run determinism failed for {text:?}");
        }
    }
    // A second session on the same model.
    let other = model.create_session(&SessionDesc::default()).expect("second session");
    for (i, text) in TEXTS.iter().enumerate() {
        other.write_text(&[text], &EmbedOptions::default()).expect("write");
        let row = read_f32(&other.run(&RunOptions::default()).expect("run"), 0);
        assert_eq!(row, expected[i], "session-to-session determinism failed for {text:?}");
    }
    // A second model on a second context.
    let (_m2, fresh) = t.session(BundleKind::Embedding);
    for (i, text) in TEXTS.iter().enumerate() {
        fresh.write_text(&[text], &EmbedOptions::default()).expect("write");
        let row = read_f32(&fresh.run(&RunOptions::default()).expect("run"), 0);
        assert_eq!(row, expected[i], "model-to-model determinism failed for {text:?}");
    }
    // Batched and one-at-a-time must agree.
    let all: Vec<&str> = TEXTS.to_vec();
    fresh.write_text(&all, &EmbedOptions::default()).expect("batched write");
    let batched = read_f32(&fresh.run(&RunOptions::default()).expect("run"), 0);
    let dim = model.info().dim as usize;
    for (i, row) in batched.chunks(dim).enumerate() {
        assert_eq!(row, &expected[i][..], "batching changed row {i}");
    }
}

#[test]
fn threading_models_and_runtimes_are_shared_safely() {
    let t = Target::from_env();
    let model = t.model(BundleKind::Embedding);
    let expected = sequential_rows(&t);
    let barrier = Arc::new(Barrier::new(3));
    thread::scope(|scope| {
        for _ in 0..3 {
            let model = model.clone();
            let barrier = barrier.clone();
            let expected = &expected;
            scope.spawn(move || {
                barrier.wait();
                // Creating sessions from several threads on one model is allowed.
                for _ in 0..10 {
                    let session = model.create_session(&SessionDesc::default()).expect("session");
                    session.write_text(&[TEXTS[1]], &EmbedOptions::default()).expect("write");
                    let result = session.run(&RunOptions::default()).expect("run");
                    assert_eq!(read_f32(&result, 0), expected[1]);
                    // Reading immutable model state concurrently is allowed.
                    assert!(model.info().dim > 0);
                }
            });
        }
    });
}
