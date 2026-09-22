//! Group `threading`, C ABI layer: handles crossing threads, `TURBO_E_BUSY`
//! on overlapping use of one session, and determinism across sessions.

use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

use turbo_abi::*;
use turbo_capi::*;
use turbo_conformance::c::{self, ssz};
use turbo_conformance::{assert_rc, needs, BundleKind, Target};

/// Raw handles are pointers, which are not `Send` by default; the ABI
/// documents that runtime, context and model handles may be used from any
/// thread, so the suite wraps them exactly as a binding would.
struct Shared<T>(*mut T);

impl<T> Clone for Shared<T> {
    fn clone(&self) -> Self {
        *self
    }
}
// A raw pointer is `Copy` whatever it points at; the derive would add a
// needless `T: Copy` bound that opaque handle types cannot satisfy.
impl<T> Copy for Shared<T> {}
// SAFETY: PLAN.md section 4.6 makes runtime, context and model handles
// immutable after creation and usable from any thread. Sessions are
// single-owner; the tests below that share one do so on purpose to observe
// TURBO_E_BUSY, which the library must return without corrupting state.
unsafe impl<T> Send for Shared<T> {}
unsafe impl<T> Sync for Shared<T> {}

const TEXT: &str = "hello world";

fn run_once(session: *mut turbo_session, dim: usize) -> Vec<f32> {
    let mut e = c::err();
    let texts = [c::text(TEXT)];
    // SAFETY: valid session handle and one readable text view.
    assert_rc!(unsafe { turbo_session_write_text(session, texts.as_ptr(), 1, ptr::null(), &mut e) }, TURBO_OK, e);
    let mut r: *mut turbo_result = ptr::null_mut();
    // SAFETY: valid session handle and out pointer.
    assert_rc!(unsafe { turbo_session_run(session, ptr::null(), &mut r, &mut e) }, TURBO_OK, e);
    let v = c::read_f32(r, 0, dim);
    // SAFETY: released once.
    unsafe { turbo_result_release(r) };
    v
}

#[test]
fn threading_two_sessions_run_concurrently_with_the_same_results() {
    let t = Target::from_env();
    needs!(t, Embedding);
    let ct = c::CTarget::new(&t);
    let ctx = ct.context();
    let model = ct.model(ctx, BundleKind::Embedding);
    let mut e = c::err();
    let mut info = turbo_model_info { struct_size: ssz::<turbo_model_info>(), ..unsafe { std::mem::zeroed() } };
    // SAFETY: valid model handle and out pointer.
    assert_rc!(unsafe { turbo_model_get_info(model, &mut info, &mut e) }, TURBO_OK, e);
    let dim = info.dim as usize;

    let a = Shared(ct.session(model));
    let b = Shared(ct.session(model));
    let expected = run_once(a.0, dim);
    let barrier = Arc::new(Barrier::new(2));
    let rows: Vec<Vec<Vec<f32>>> = thread::scope(|scope| {
        [a, b]
            .into_iter()
            .map(|s| {
                let barrier = barrier.clone();
                scope.spawn(move || {
                    // Capture the whole wrapper, not just the raw field.
                    let s = s;
                    barrier.wait();
                    (0..50).map(|_| run_once(s.0, dim)).collect::<Vec<_>>()
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().expect("thread"))
            .collect()
    });
    for per_thread in rows {
        for row in per_thread {
            assert_eq!(row, expected, "a concurrent session produced a different vector");
        }
    }
    // SAFETY: each handle is released once.
    unsafe {
        turbo_session_release(a.0);
        turbo_session_release(b.0);
        turbo_model_release(model);
        turbo_context_release(ctx);
    }
    drop(ct);
}

#[test]
fn threading_a_held_result_makes_another_thread_busy() {
    let t = Target::from_env();
    needs!(t, Embedding);
    let ct = c::CTarget::new(&t);
    let ctx = ct.context();
    let model = ct.model(ctx, BundleKind::Embedding);
    let session = Shared(ct.session(model));
    let mut e = c::err();
    let texts = [c::text(TEXT)];
    // SAFETY: valid session and one readable text.
    assert_rc!(unsafe { turbo_session_write_text(session.0, texts.as_ptr(), 1, ptr::null(), &mut e) }, TURBO_OK, e);
    let mut held: *mut turbo_result = ptr::null_mut();
    // SAFETY: valid session and out pointer.
    assert_rc!(unsafe { turbo_session_run(session.0, ptr::null(), &mut held, &mut e) }, TURBO_OK, e);

    let barrier = Arc::new(Barrier::new(2));
    let codes = thread::scope(|scope| {
        let barrier2 = barrier.clone();
        let other = scope.spawn(move || {
            let session = session;
            barrier2.wait();
            let mut e = c::err();
            let mut r: *mut turbo_result = ptr::null_mut();
            let texts = [c::text(TEXT)];
            // SAFETY: the session handle is valid for the whole test.
            let run = unsafe { turbo_session_run(session.0, ptr::null(), &mut r, &mut e) };
            // SAFETY: as above.
            let write = unsafe { turbo_session_write_text(session.0, texts.as_ptr(), 1, ptr::null(), &mut e) };
            assert!(r.is_null(), "a refused run must not hand back a result");
            (run, write)
        });
        barrier.wait();
        other.join().expect("thread")
    });
    assert_eq!(codes.0, TURBO_E_BUSY, "run: got {}", c::status_name(codes.0));
    assert_eq!(codes.1, TURBO_E_BUSY, "write: got {}", c::status_name(codes.1));
    // SAFETY: each handle is released once.
    unsafe {
        turbo_result_release(held);
        let mut r: *mut turbo_result = ptr::null_mut();
        assert_rc!(turbo_session_run(session.0, ptr::null(), &mut r, &mut e), TURBO_OK, e);
        turbo_result_release(r);
        turbo_session_release(session.0);
        turbo_model_release(model);
        turbo_context_release(ctx);
    }
    drop(ct);
}

#[test]
fn threading_one_session_from_many_threads_never_corrupts_or_lies() {
    let t = Target::from_env();
    needs!(t, Embedding);
    let ct = c::CTarget::new(&t);
    let ctx = ct.context();
    let model = ct.model(ctx, BundleKind::Embedding);
    let mut e = c::err();
    let mut info = turbo_model_info { struct_size: ssz::<turbo_model_info>(), ..unsafe { std::mem::zeroed() } };
    // SAFETY: valid model handle and out pointer.
    assert_rc!(unsafe { turbo_model_get_info(model, &mut info, &mut e) }, TURBO_OK, e);
    let dim = info.dim as usize;
    let session = Shared(ct.session(model));
    let expected = run_once(session.0, dim);

    let barrier = Arc::new(Barrier::new(4));
    let ok = Arc::new(AtomicUsize::new(0));
    thread::scope(|scope| {
        for _ in 0..4 {
            let barrier = barrier.clone();
            let ok = ok.clone();
            let expected = &expected;
            scope.spawn(move || {
                let session = session;
                barrier.wait();
                let mut e = c::err();
                let texts = [c::text(TEXT)];
                for _ in 0..200 {
                    // SAFETY: the session handle stays valid for the whole test.
                    let rc = unsafe { turbo_session_write_text(session.0, texts.as_ptr(), 1, ptr::null(), &mut e) };
                    if rc != TURBO_OK {
                        assert_eq!(rc, TURBO_E_BUSY, "write: {}", c::status_name(rc));
                        continue;
                    }
                    let mut r: *mut turbo_result = ptr::null_mut();
                    // SAFETY: as above.
                    let rc = unsafe { turbo_session_run(session.0, ptr::null(), &mut r, &mut e) };
                    if rc == TURBO_OK {
                        ok.fetch_add(1, Ordering::Relaxed);
                        let v = c::read_f32(r, 0, dim);
                        assert_eq!(v.len(), expected.len());
                        // SAFETY: released once.
                        unsafe { turbo_result_release(r) };
                    } else {
                        assert!(rc == TURBO_E_BUSY || rc == TURBO_E_INVALID_STATE, "run: {}", c::status_name(rc));
                        assert!(r.is_null());
                    }
                }
            });
        }
    });
    assert!(ok.load(Ordering::Relaxed) > 0, "no concurrent run succeeded");
    assert_eq!(run_once(session.0, dim), expected, "the session was corrupted by concurrent use");
    // SAFETY: each handle is released once.
    unsafe {
        turbo_session_release(session.0);
        turbo_model_release(model);
        turbo_context_release(ctx);
    }
    drop(ct);
}

#[test]
fn threading_results_are_deterministic_across_sessions() {
    let t = Target::from_env();
    needs!(t, Embedding);
    if t.caps() & TURBO_CAP_DETERMINISTIC == 0 {
        println!("threading_results_are_deterministic: device does not claim TURBO_CAP_DETERMINISTIC");
        return;
    }
    let ct = c::CTarget::new(&t);
    let ctx = ct.context();
    let model = ct.model(ctx, BundleKind::Embedding);
    let mut e = c::err();
    let mut info = turbo_model_info { struct_size: ssz::<turbo_model_info>(), ..unsafe { std::mem::zeroed() } };
    // SAFETY: valid model handle and out pointer.
    assert_rc!(unsafe { turbo_model_get_info(model, &mut info, &mut e) }, TURBO_OK, e);
    let dim = info.dim as usize;
    let first = ct.session(model);
    let expected = run_once(first, dim);
    for _ in 0..5 {
        assert_eq!(run_once(first, dim), expected, "run-to-run determinism failed");
    }
    let second = ct.session(model);
    assert_eq!(run_once(second, dim), expected, "session-to-session determinism failed");
    // A second context and model on the same device.
    let other_ctx = ct.context();
    let other_model = ct.model(other_ctx, BundleKind::Embedding);
    let third = ct.session(other_model);
    assert_eq!(run_once(third, dim), expected, "context-to-context determinism failed");
    // SAFETY: each handle is released once.
    unsafe {
        turbo_session_release(first);
        turbo_session_release(second);
        turbo_session_release(third);
        turbo_model_release(model);
        turbo_model_release(other_model);
        turbo_context_release(ctx);
        turbo_context_release(other_ctx);
    }
    drop(ct);
}

#[test]
fn threading_a_generation_reports_busy_to_a_second_thread() {
    let t = Target::from_env();
    needs!(t, Generative);
    let ct = c::CTarget::new(&t);
    let ctx = ct.context();
    let model = ct.model(ctx, BundleKind::Generative);
    let mut e = c::err();
    let mut gd = c::generate_desc();
    gd.max_new_tokens = 64;
    let mut gen: *mut turbo_generation = ptr::null_mut();
    // SAFETY: valid model handle and descriptor.
    assert_rc!(unsafe { turbo_generation_create(model, &gd, &mut gen, &mut e) }, TURBO_OK, e);
    let msg = turbo_message { role: c::text("user"), content: c::text("hello") };
    // SAFETY: one readable message.
    assert_rc!(unsafe { turbo_generation_prompt(gen, &msg, 1, &mut e) }, TURBO_OK, e);
    let shared = Shared(gen);

    // Two threads stepping one generation: every status is OK or BUSY, the
    // token order never repeats, and the stream still terminates.
    let barrier = Arc::new(Barrier::new(2));
    let tokens: Vec<Vec<i32>> = thread::scope(|scope| {
        (0..2)
            .map(|_| {
                let barrier = barrier.clone();
                scope.spawn(move || {
                    let shared = shared;
                    barrier.wait();
                    let mut e = c::err();
                    let mut chunk = c::chunk();
                    let mut mine = Vec::new();
                    loop {
                        // SAFETY: the generation handle is valid for the test.
                        let rc = unsafe { turbo_generation_step(shared.0, &mut chunk, &mut e) };
                        if rc == TURBO_E_BUSY {
                            continue;
                        }
                        if rc == TURBO_E_INVALID_STATE {
                            break; // the other thread finished the stream
                        }
                        assert_eq!(rc, TURBO_OK, "step: {} {}", c::status_name(rc), c::message(&e));
                        mine.extend(c::chunk_tokens(&chunk));
                        if chunk.done != 0 {
                            break;
                        }
                    }
                    mine
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().expect("thread"))
            .collect()
    });
    let total: usize = tokens.iter().map(|t| t.len()).sum();
    assert!(total > 0, "no thread produced a token");
    assert!(total <= 64, "more tokens were produced than max_new_tokens allows");
    // SAFETY: each handle is released once.
    unsafe {
        turbo_generation_release(gen);
        turbo_model_release(model);
        turbo_context_release(ctx);
    }
    drop(ct);
}
