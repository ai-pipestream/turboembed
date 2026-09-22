//! Group `lifetime`, C ABI layer: `turbo_*_release` on a parent never
//! invalidates a child, in any release order, and the result lease is held by
//! the result and by every buffer view of it.

use std::ptr;

use turbo_abi::*;
use turbo_capi::*;
use turbo_conformance::c::{self, ssz};
use turbo_conformance::{assert_rc, needs, permutations, BundleKind, Target};

const TEXT: &str = "hello world";

/// Build a fresh runtime → context → model → session → result chain on the
/// device under test and return the handles plus the reference vector.
struct Chain {
    ctx: *mut turbo_context,
    model: *mut turbo_model,
    session: *mut turbo_session,
    dim: usize,
}

fn chain(t: &Target) -> (c::CTarget, Chain) {
    let ct = c::CTarget::new(t);
    let ctx = ct.context();
    let model = ct.model(ctx, BundleKind::Embedding);
    let session = ct.session(model);
    let mut e = c::err();
    let mut info = turbo_model_info { struct_size: ssz::<turbo_model_info>(), ..unsafe { std::mem::zeroed() } };
    // SAFETY: valid model handle and out pointer.
    assert_rc!(unsafe { turbo_model_get_info(model, &mut info, &mut e) }, TURBO_OK, e);
    (ct, Chain { ctx, model, session, dim: info.dim as usize })
}

fn write_and_run(session: *mut turbo_session) -> *mut turbo_result {
    let mut e = c::err();
    let texts = [c::text(TEXT)];
    // SAFETY: one readable text view and a valid session.
    assert_rc!(unsafe { turbo_session_write_text(session, texts.as_ptr(), 1, ptr::null(), &mut e) }, TURBO_OK, e);
    let mut r: *mut turbo_result = ptr::null_mut();
    assert_rc!(unsafe { turbo_session_run(session, ptr::null(), &mut r, &mut e) }, TURBO_OK, e);
    assert!(!r.is_null());
    r
}

#[test]
fn lifetime_releasing_parents_in_every_order_keeps_the_result_readable() {
    let t = Target::from_env();
    let mut expected: Option<Vec<f32>> = None;
    for perm in permutations(4) {
        let (ct, ch) = chain(&t);
        let result = write_and_run(ch.session);
        // `CTarget` must not release the runtime a second time.
        let rt = ct.rt;
        std::mem::forget(ct);
        let releases: [Box<dyn Fn()>; 4] = [
            Box::new(move || unsafe { turbo_runtime_release(rt) }),
            Box::new(move || unsafe { turbo_context_release(ch.ctx) }),
            Box::new(move || unsafe { turbo_model_release(ch.model) }),
            Box::new(move || unsafe { turbo_session_release(ch.session) }),
        ];
        for &i in &perm {
            releases[i]();
        }
        let v = c::read_f32(result, 0, ch.dim);
        match &expected {
            None => expected = Some(v),
            Some(want) => assert_eq!(&v, want, "release order {perm:?} changed the result"),
        }
        // Result metadata survives too.
        let ri = c::result_info(result);
        assert_eq!(ri.batch, 1);
        assert_eq!(ri.dim as usize, ch.dim);
        // SAFETY: released exactly once, after the last read.
        unsafe { turbo_result_release(result) };
    }
}

#[test]
fn lifetime_releasing_parents_in_every_order_keeps_the_session_usable() {
    let t = Target::from_env();
    let mut expected: Option<Vec<f32>> = None;
    for perm in permutations(3) {
        let (ct, ch) = chain(&t);
        let rt = ct.rt;
        std::mem::forget(ct);
        let releases: [Box<dyn Fn()>; 3] = [
            Box::new(move || unsafe { turbo_runtime_release(rt) }),
            Box::new(move || unsafe { turbo_context_release(ch.ctx) }),
            Box::new(move || unsafe { turbo_model_release(ch.model) }),
        ];
        for &i in &perm {
            releases[i]();
        }
        let result = write_and_run(ch.session);
        let v = c::read_f32(result, 0, ch.dim);
        match &expected {
            None => expected = Some(v),
            Some(want) => assert_eq!(&v, want, "release order {perm:?} changed the result"),
        }
        // SAFETY: each handle is released exactly once here.
        unsafe {
            turbo_result_release(result);
            turbo_session_release(ch.session);
        }
    }
}

#[test]
fn lifetime_result_view_keeps_the_lease_after_the_result_is_released() {
    let t = Target::from_env();
    let (ct, ch) = chain(&t);
    let result = write_and_run(ch.session);
    let mut e = c::err();
    let mut view: *mut turbo_buffer = ptr::null_mut();
    // SAFETY: valid result handle and out pointer.
    assert_rc!(unsafe { turbo_result_buffer(result, 0, &mut view, &mut e) }, TURBO_OK, e);
    assert!(!view.is_null());
    // SAFETY: the view keeps the lease; the result handle itself is done.
    unsafe { turbo_result_release(result) };

    let mut second: *mut turbo_result = ptr::null_mut();
    let texts = [c::text(TEXT)];
    // SAFETY: the session is alive; both calls must report the lease.
    unsafe {
        assert_rc!(turbo_session_run(ch.session, ptr::null(), &mut second, &mut e), TURBO_E_BUSY, e);
        assert_rc!(turbo_session_write_text(ch.session, texts.as_ptr(), 1, ptr::null(), &mut e), TURBO_E_BUSY, e);
        // The leased memory is still described and readable.
        let mut desc = turbo_buffer_desc {
            struct_size: ssz::<turbo_buffer_desc>(),
            placement: 0,
            dtype: 0,
            ndim: 0,
            shape: [0; TURBO_MAX_RANK],
            strides: [0; TURBO_MAX_RANK],
            bytes: 0,
            next: ptr::null(),
        };
        assert_rc!(turbo_buffer_get_desc(view, &mut desc, &mut e), TURBO_OK, e);
        assert_eq!(desc.dtype, TURBO_DTYPE_F32);
        turbo_buffer_release(view);
        // With the last view gone the session accepts work again.
        assert_rc!(turbo_session_run(ch.session, ptr::null(), &mut second, &mut e), TURBO_OK, e);
        turbo_result_release(second);
        turbo_session_release(ch.session);
        turbo_model_release(ch.model);
        turbo_context_release(ch.ctx);
    }
    drop(ct);
}

#[test]
fn lifetime_releasing_the_last_of_several_views_returns_the_lease() {
    let t = Target::from_env();
    let (ct, ch) = chain(&t);
    let result = write_and_run(ch.session);
    let mut e = c::err();
    let mut views = [ptr::null_mut::<turbo_buffer>(); 3];
    for v in views.iter_mut() {
        // SAFETY: valid result handle and out pointer.
        assert_rc!(unsafe { turbo_result_buffer(result, 0, v, &mut e) }, TURBO_OK, e);
    }
    // SAFETY: released once; the views hold the lease.
    unsafe { turbo_result_release(result) };
    let mut second: *mut turbo_result = ptr::null_mut();
    for (i, v) in views.iter().enumerate() {
        // SAFETY: the session is alive and still leased.
        unsafe { assert_rc!(turbo_session_run(ch.session, ptr::null(), &mut second, &mut e), TURBO_E_BUSY, e) };
        // SAFETY: each view is released exactly once.
        unsafe { turbo_buffer_release(*v) };
        if i + 1 < views.len() {
            // SAFETY: views remain, so the lease is still out.
            unsafe { assert_rc!(turbo_session_run(ch.session, ptr::null(), &mut second, &mut e), TURBO_E_BUSY, e) };
        }
    }
    // SAFETY: every view is gone, so the lease came back.
    unsafe {
        assert_rc!(turbo_session_run(ch.session, ptr::null(), &mut second, &mut e), TURBO_OK, e);
        turbo_result_release(second);
        turbo_session_release(ch.session);
        turbo_model_release(ch.model);
        turbo_context_release(ch.ctx);
    }
    drop(ct);
}

#[test]
fn lifetime_generation_outlives_its_model_and_context() {
    let t = Target::from_env();
    needs!(t, Generative);
    let ct = c::CTarget::new(&t);
    let ctx = ct.context();
    let model = ct.model(ctx, BundleKind::Generative);
    let mut e = c::err();
    let mut gd = c::generate_desc();
    gd.max_new_tokens = 3;
    let mut gen: *mut turbo_generation = ptr::null_mut();
    // SAFETY: valid model handle and descriptor.
    assert_rc!(unsafe { turbo_generation_create(model, &gd, &mut gen, &mut e) }, TURBO_OK, e);
    let msg = turbo_message { role: c::text("user"), content: c::text("hello") };
    // SAFETY: one readable message.
    assert_rc!(unsafe { turbo_generation_prompt(gen, &msg, 1, &mut e) }, TURBO_OK, e);
    // SAFETY: the generation retains its model and context.
    unsafe {
        turbo_model_release(model);
        turbo_context_release(ctx);
    }
    let mut chunk = c::chunk();
    let mut steps = 0;
    loop {
        // SAFETY: valid generation handle and chunk.
        assert_rc!(unsafe { turbo_generation_step(gen, &mut chunk, &mut e) }, TURBO_OK, e);
        steps += 1;
        if chunk.done != 0 {
            assert_eq!(chunk.generated_tokens, 3);
            assert_eq!(chunk.finish_reason, TURBO_FINISH_LENGTH);
            break;
        }
        assert!(steps < 10, "generation did not finish");
    }
    // SAFETY: released once.
    unsafe { turbo_generation_release(gen) };
    drop(ct);
}

#[test]
fn lifetime_buffer_outlives_its_context() {
    let t = Target::from_env();
    let ct = c::CTarget::new(&t);
    let ctx = ct.context();
    let mut e = c::err();
    let desc = turbo_buffer_desc {
        struct_size: ssz::<turbo_buffer_desc>(),
        placement: TURBO_PLACE_HOST,
        dtype: TURBO_DTYPE_F32,
        ndim: 1,
        shape: [4, 0, 0, 0, 0, 0, 0, 0],
        strides: [0; TURBO_MAX_RANK],
        bytes: 0,
        next: ptr::null(),
    };
    let mut buf: *mut turbo_buffer = ptr::null_mut();
    // SAFETY: valid context and descriptor.
    assert_rc!(unsafe { turbo_buffer_alloc(ctx, &desc, &mut buf, &mut e) }, TURBO_OK, e);
    // SAFETY: the buffer retains the context.
    unsafe { turbo_context_release(ctx) };
    let mut got = turbo_buffer_desc { struct_size: ssz::<turbo_buffer_desc>(), ..desc };
    // SAFETY: the buffer handle is still valid.
    unsafe {
        assert_rc!(turbo_buffer_get_desc(buf, &mut got, &mut e), TURBO_OK, e);
        assert_eq!(got.bytes, 16);
        let mut host: *mut std::ffi::c_void = ptr::null_mut();
        assert_rc!(turbo_buffer_host_ptr(buf, &mut host, &mut e), TURBO_OK, e);
        assert!(!host.is_null());
        turbo_buffer_release(buf);
    }
    drop(ct);
}

#[test]
fn lifetime_generation_step_after_finish_is_invalid_state() {
    let t = Target::from_env();
    needs!(t, Generative);
    let ct = c::CTarget::new(&t);
    let ctx = ct.context();
    let model = ct.model(ctx, BundleKind::Generative);
    let mut e = c::err();
    let mut gd = c::generate_desc();
    gd.max_new_tokens = 1;
    let mut gen: *mut turbo_generation = ptr::null_mut();
    // SAFETY: valid handles throughout.
    unsafe {
        assert_rc!(turbo_generation_create(model, &gd, &mut gen, &mut e), TURBO_OK, e);
        let msg = turbo_message { role: c::text("user"), content: c::text("hello") };
        assert_rc!(turbo_generation_prompt(gen, &msg, 1, &mut e), TURBO_OK, e);
        let mut chunk = c::chunk();
        loop {
            assert_rc!(turbo_generation_step(gen, &mut chunk, &mut e), TURBO_OK, e);
            if chunk.done != 0 {
                break;
            }
        }
        assert_rc!(turbo_generation_step(gen, &mut chunk, &mut e), TURBO_E_INVALID_STATE, e);
        assert_rc!(turbo_generation_step(gen, &mut chunk, &mut e), TURBO_E_INVALID_STATE, e);
        turbo_generation_release(gen);
        turbo_model_release(model);
        turbo_context_release(ctx);
    }
    drop(ct);
}

#[test]
fn lifetime_binding_a_buffer_from_another_context_is_invalid_argument() {
    let t = Target::from_env();
    needs!(t, Generic);
    let ct = c::CTarget::new(&t);
    let first = ct.context();
    let second = ct.context();
    let model = ct.model(first, BundleKind::Generic);
    let session = ct.session(model);
    let mut e = c::err();
    let desc = turbo_buffer_desc {
        struct_size: ssz::<turbo_buffer_desc>(),
        placement: TURBO_PLACE_HOST,
        dtype: TURBO_DTYPE_F32,
        ndim: 2,
        shape: [2, 3, 0, 0, 0, 0, 0, 0],
        strides: [0; TURBO_MAX_RANK],
        bytes: 0,
        next: ptr::null(),
    };
    let mut foreign: *mut turbo_buffer = ptr::null_mut();
    let mut own: *mut turbo_buffer = ptr::null_mut();
    // SAFETY: valid contexts, descriptor, and out pointers.
    unsafe {
        assert_rc!(turbo_buffer_alloc(second, &desc, &mut foreign, &mut e), TURBO_OK, e);
        assert_rc!(turbo_buffer_alloc(first, &desc, &mut own, &mut e), TURBO_OK, e);
        assert_rc!(turbo_session_bind(session, c::text("x"), foreign, &mut e), TURBO_E_INVALID_ARGUMENT, e);
        assert_rc!(turbo_session_bind(session, c::text("x"), own, &mut e), TURBO_OK, e);
        // An unknown tensor name is refused by name, not silently ignored.
        assert_rc!(turbo_session_bind(session, c::text("nope"), own, &mut e), TURBO_E_INVALID_ARGUMENT, e);
        // A NULL buffer handle is a handle error.
        assert_rc!(turbo_session_bind(session, c::text("x"), ptr::null_mut(), &mut e), TURBO_E_INVALID_HANDLE, e);
        // An empty name is an argument error.
        assert_rc!(turbo_session_bind(session, c::text(""), own, &mut e), TURBO_E_INVALID_ARGUMENT, e);
        turbo_buffer_release(foreign);
        turbo_buffer_release(own);
        turbo_session_release(session);
        turbo_model_release(model);
        turbo_context_release(first);
        turbo_context_release(second);
    }
    drop(ct);
}

#[test]
fn lifetime_two_contexts_on_one_device_are_independent() {
    let t = Target::from_env();
    needs!(t, Embedding);
    let ct = c::CTarget::new(&t);
    let first = ct.context();
    let second = ct.context();
    let a = ct.model(first, BundleKind::Embedding);
    let b = ct.model(second, BundleKind::Embedding);
    let sa = ct.session(a);
    let sb = ct.session(b);
    let ra = write_and_run(sa);
    let rb = write_and_run(sb);
    let ia = c::result_info(ra);
    let va = c::read_f32(ra, 0, ia.dim as usize);
    let vb = c::read_f32(rb, 0, ia.dim as usize);
    assert_eq!(va, vb, "two contexts must agree on the same input");
    let mut e = c::err();
    let mut out: *mut turbo_result = ptr::null_mut();
    // SAFETY: `ra` still leases session `sa` only.
    unsafe {
        assert_rc!(turbo_session_run(sa, ptr::null(), &mut out, &mut e), TURBO_E_BUSY, e);
        turbo_result_release(rb);
        let texts = [c::text(TEXT)];
        assert_rc!(turbo_session_write_text(sb, texts.as_ptr(), 1, ptr::null(), &mut e), TURBO_OK, e);
        turbo_result_release(ra);
        turbo_session_release(sa);
        turbo_session_release(sb);
        turbo_model_release(a);
        turbo_model_release(b);
        turbo_context_release(first);
        turbo_context_release(second);
    }
    drop(ct);
}
