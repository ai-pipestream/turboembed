//! Group `contract`, C ABI layer: the status codes a caller can only reach
//! when something fails, plus the panic boundary.
//!
//! The Rust half is `honesty_faults_rust.rs`; the codes and the mock's
//! `fault` option are described there. This file adds what only the C ABI
//! can show: a panic inside a provider call becomes `TURBO_E_PANIC` at the
//! boundary instead of unwinding into C (`docs/architecture.md`), and the
//! error record still carries a message and a field.

use std::ptr;

use turbo_abi::*;
use turbo_capi::*;
use turbo_conformance::c::{self, ssz};
use turbo_conformance::{assert_rc, needs, BundleKind, Target};

/// One `fault` option, as a `turbo_kv` array.
fn fault_kv(value: &str) -> [turbo_kv; 1] {
    [turbo_kv { key: c::text("fault"), value: c::text(value) }]
}

fn context_desc(kvs: &[turbo_kv; 1]) -> turbo_context_desc {
    turbo_context_desc {
        struct_size: ssz::<turbo_context_desc>(),
        flags: 0,
        n_options: 1,
        reserved: 0,
        options: kvs.as_ptr(),
        next: ptr::null(),
    }
}

fn model_desc(kvs: &[turbo_kv; 1]) -> turbo_model_desc {
    turbo_model_desc { struct_size: ssz::<turbo_model_desc>(), n_options: 1, options: kvs.as_ptr(), next: ptr::null() }
}

fn session_desc(kvs: &[turbo_kv; 1]) -> turbo_session_desc {
    turbo_session_desc {
        struct_size: ssz::<turbo_session_desc>(),
        max_batch: 0,
        max_seq: 0,
        n_options: 1,
        options: kvs.as_ptr(),
        next: ptr::null(),
    }
}

/// The code a provider without the mock's fault hook must return for the
/// option: an unknown key, refused by its 1-based index.
const UNKNOWN: i32 = TURBO_E_INVALID_ARGUMENT;

#[test]
fn faults_context_creation_reports_device_unavailable() {
    let t = Target::from_env();
    let ct = c::CTarget::new(&t);
    let mut e = c::err();
    let kvs = fault_kv("context=device_unavailable");
    let desc = context_desc(&kvs);
    let mut ctx: *mut turbo_context = ptr::null_mut();
    // SAFETY: valid runtime handle; the descriptor and its strings outlive the call.
    let rc = unsafe { turbo_context_create(ct.rt, ct.device, &desc, &mut ctx, &mut e) };
    if t.is_mock() {
        assert_eq!(rc, TURBO_E_DEVICE_UNAVAILABLE, "got {} ({})", c::status_name(rc), c::message(&e));
        assert!(c::message(&e).contains("device_unavailable"), "{}", c::message(&e));
    } else {
        assert_eq!(rc, UNKNOWN, "an unknown context option: got {}", c::status_name(rc));
        assert_eq!(e.field, 1, "{}", c::message(&e));
    }
    assert!(ctx.is_null(), "a refused context must not be handed back");
    drop(ct);
}

#[test]
fn faults_model_load_reports_unsupported_dtype() {
    let t = Target::from_env();
    needs!(t, Embedding);
    let ct = c::CTarget::new(&t);
    let ctx = ct.context();
    let mut e = c::err();
    let path = ct.bundle(BundleKind::Embedding);
    let kvs = fault_kv("load=unsupported_dtype");
    let desc = model_desc(&kvs);
    let mut m: *mut turbo_model = ptr::null_mut();
    // SAFETY: valid context handle; `path` and the descriptor outlive the call.
    let rc = unsafe { turbo_model_load(ctx, c::text(&path), &desc, &mut m, &mut e) };
    if t.is_mock() {
        assert_eq!(rc, TURBO_E_UNSUPPORTED_DTYPE, "got {} ({})", c::status_name(rc), c::message(&e));
        assert!(c::message(&e).contains("unsupported_dtype"), "{}", c::message(&e));
    } else {
        assert_eq!(rc, UNKNOWN, "an unknown model option: got {}", c::status_name(rc));
        assert_eq!(e.field, 1, "{}", c::message(&e));
    }
    assert!(m.is_null(), "a refused load must not hand back a model");
    // SAFETY: released once.
    unsafe { turbo_context_release(ctx) };
    drop(ct);
}

#[test]
fn faults_run_reports_overloaded_runtime_and_internal() {
    let t = Target::from_env();
    needs!(t, Embedding);
    let ct = c::CTarget::new(&t);
    let ctx = ct.context();
    let model = ct.model(ctx, BundleKind::Embedding);
    let mut e = c::err();
    let texts = [c::text("hello world")];
    for (value, code) in
        [("run=overloaded", TURBO_E_OVERLOADED), ("run=runtime", TURBO_E_RUNTIME), ("run=internal", TURBO_E_INTERNAL)]
    {
        let kvs = fault_kv(value);
        let desc = session_desc(&kvs);
        let mut s: *mut turbo_session = ptr::null_mut();
        // SAFETY: valid model handle; the descriptor outlives the call.
        let rc = unsafe { turbo_session_create(model, &desc, &mut s, &mut e) };
        if !t.is_mock() {
            assert_eq!(rc, UNKNOWN, "an unknown session option: got {}", c::status_name(rc));
            assert_eq!(e.field, 1, "{}", c::message(&e));
            assert!(s.is_null());
            continue;
        }
        assert_eq!(rc, TURBO_OK, "`{value}` is a session fault, not a refusal: {}", c::message(&e));
        let mut r: *mut turbo_result = ptr::null_mut();
        // SAFETY: valid session handle; one readable text view.
        unsafe {
            assert_rc!(turbo_session_write_text(s, texts.as_ptr(), 1, ptr::null(), &mut e), TURBO_OK, e);
            let rc = turbo_session_run(s, ptr::null(), &mut r, &mut e);
            assert_eq!(rc, code, "`{value}`: got {} ({})", c::status_name(rc), c::message(&e));
            assert_eq!(e.code, code, "the error record carries the same code as the return value");
            assert_eq!(e.field, 0, "a failure of the run names no descriptor field: {}", c::message(&e));
            assert!(!c::message(&e).is_empty(), "the failure must say something");
            assert!(r.is_null(), "a failed run must not lease a result");
            turbo_session_release(s);
        }
    }
    // SAFETY: each handle is released once.
    unsafe {
        turbo_model_release(model);
        turbo_context_release(ctx);
    }
    drop(ct);
}

#[test]
fn faults_a_panicking_provider_is_turbo_e_panic() {
    // docs/architecture.md: "A Rust panic crossing the C boundary is caught
    // by catch_unwind in `boundary` and turned into TURBO_E_PANIC rather
    // than unwinding into C." The session it happened on is then spent, the
    // way `Session::lock` documents; the model and the runtime are not.
    let t = Target::from_env();
    needs!(t, Embedding);
    let ct = c::CTarget::new(&t);
    let ctx = ct.context();
    let model = ct.model(ctx, BundleKind::Embedding);
    let mut e = c::err();
    let kvs = fault_kv("run=panic");
    let desc = session_desc(&kvs);
    let mut s: *mut turbo_session = ptr::null_mut();
    // SAFETY: valid model handle; the descriptor outlives the call.
    let rc = unsafe { turbo_session_create(model, &desc, &mut s, &mut e) };
    if !t.is_mock() {
        assert_eq!(rc, UNKNOWN, "an unknown session option: got {}", c::status_name(rc));
        assert_eq!(e.field, 1, "{}", c::message(&e));
        assert!(s.is_null());
        // SAFETY: each handle is released once.
        unsafe {
            turbo_model_release(model);
            turbo_context_release(ctx);
        }
        return;
    }
    assert_eq!(rc, TURBO_OK, "`run=panic` is a session fault, not a refusal: {}", c::message(&e));
    let texts = [c::text("hello world")];
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let mut r: *mut turbo_result = ptr::null_mut();
    // SAFETY: valid session handle; one readable text view. The panic is
    // caught inside the library, so nothing unwinds through this call.
    let rc = unsafe {
        assert_rc!(turbo_session_write_text(s, texts.as_ptr(), 1, ptr::null(), &mut e), TURBO_OK, e);
        turbo_session_run(s, ptr::null(), &mut r, &mut e)
    };
    std::panic::set_hook(previous);
    assert_eq!(rc, TURBO_E_PANIC, "got {} ({})", c::status_name(rc), c::message(&e));
    assert_eq!(e.code, TURBO_E_PANIC);
    let message = c::message(&e);
    assert!(message.contains("C boundary"), "the message must say where it was caught: {message}");
    assert!(message.contains("panic"), "the message must carry the payload: {message}");
    assert!(r.is_null(), "a panicking run must not lease a result");

    // SAFETY: the handles are still valid; the session is the spent one.
    unsafe {
        let rc = turbo_session_run(s, ptr::null(), &mut r, &mut e);
        assert_eq!(rc, TURBO_E_INVALID_STATE, "a session spent by a panic: got {}", c::status_name(rc));
        assert!(c::message(&e).contains("inconsistent"), "{}", c::message(&e));
        turbo_session_release(s);
        // A new session on the same model is unaffected.
        let mut fresh: *mut turbo_session = ptr::null_mut();
        assert_rc!(turbo_session_create(model, ptr::null(), &mut fresh, &mut e), TURBO_OK, e);
        assert_rc!(turbo_session_write_text(fresh, texts.as_ptr(), 1, ptr::null(), &mut e), TURBO_OK, e);
        assert_rc!(turbo_session_run(fresh, ptr::null(), &mut r, &mut e), TURBO_OK, e);
        turbo_result_release(r);
        turbo_session_release(fresh);
        turbo_model_release(model);
        turbo_context_release(ctx);
    }
    drop(ct);
}

#[test]
fn faults_a_planned_cell_is_reported_but_not_offered() {
    // The C view of the rule the Rust layer asserts in
    // `faults_a_planned_cell_is_reported_but_not_offered`: a PLANNED cell is
    // reported as PLANNED and refuses the work anyway.
    let t = Target::from_env();
    needs!(t, Embedding);
    let ct = c::CTarget::new(&t);
    let mut e = c::err();
    let mut cell = turbo_capability {
        struct_size: ssz::<turbo_capability>(),
        status: 99,
        dtype: 0,
        reference_dtype: 0,
        cosine_floor: 0.0,
        max_abs_error: 0.0,
        deterministic: 0,
        reserved: 0,
        notes: [0; 128],
    };
    // SAFETY: valid runtime handle and out pointer.
    assert_rc!(
        unsafe { turbo_runtime_capability(ct.rt, ct.device, TURBO_TASK_CHUNK, TURBO_MODALITY_TEXT, &mut cell, &mut e) },
        TURBO_OK,
        e
    );
    if t.is_mock() {
        assert_eq!(cell.status, TURBO_CAP_PLANNED, "the mock's CHUNK x TEXT cell is the planned one");
        assert_eq!(cell.dtype, 0, "a planned cell names no compute dtype");
        assert!(!c::fixed(&cell.notes).is_empty(), "a planned cell says what is planned");
    } else {
        assert!(cell.status <= TURBO_CAP_PLANNED, "{} offers CHUNK x TEXT (status {})", t.provider_id(), cell.status);
    }
    let path = ct.bundle(BundleKind::Embedding);
    // SAFETY: valid runtime handle; `path` outlives the call.
    let rc = unsafe { turbo_can_run(ct.rt, ct.device, c::text(&path), TURBO_TASK_CHUNK, TURBO_MODALITY_TEXT, &mut e) };
    assert_eq!(
        rc,
        TURBO_E_UNSUPPORTED_TASK,
        "a cell with status {} must refuse: got {}",
        cell.status,
        c::status_name(rc)
    );
    assert!(c::message(&e).contains("Chunk"), "{}", c::message(&e));
    drop(ct);
}
