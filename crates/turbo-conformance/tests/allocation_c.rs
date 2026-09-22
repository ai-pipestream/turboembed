//! Group `allocation`, C ABI layer: `turbo_session_get_stats` reports the
//! run-path counters, and a steady stream of same-shape runs moves none of
//! them but `runs`.

use std::ptr;

use turbo_abi::*;
use turbo_capi::*;
use turbo_conformance::c::{self, ssz};
use turbo_conformance::{assert_rc, BundleKind, Target};

const ITERATIONS: u64 = 100;

fn stats(session: *mut turbo_session) -> turbo_session_stats {
    let mut e = c::err();
    let mut st = turbo_session_stats { struct_size: ssz::<turbo_session_stats>(), ..Default::default() };
    // SAFETY: valid session handle and out pointer.
    assert_rc!(unsafe { turbo_session_get_stats(session, &mut st, &mut e) }, TURBO_OK, e);
    st
}

fn steady_state(name: &str, session: *mut turbo_session, mut step: impl FnMut()) {
    step();
    let warm = stats(session);
    assert_eq!(warm.runs, 1, "{name}: the warmup run must be counted");
    // The allocation counters are cumulative totals of the session's whole
    // life, so a provider that counts them at all may only ever let them
    // grow. A counter that drops has been reset or recomputed behind the
    // caller's back, which would make every delta taken across two reads
    // (what `turbo-bench` reports per run) a fiction. `u64::MAX` is the
    // "not counted" sentinel, and a counter may not switch sides mid-run.
    let mut previous = (warm.host_allocs, warm.provider_allocs);
    for i in 0..ITERATIONS {
        step();
        let now = stats(session);
        for (label, before, after) in
            [("host_allocs", previous.0, now.host_allocs), ("provider_allocs", previous.1, now.provider_allocs)]
        {
            if before == u64::MAX {
                assert_eq!(after, u64::MAX, "{name}: {label} started counting at run {i}");
                continue;
            }
            assert_ne!(after, u64::MAX, "{name}: {label} stopped counting at run {i}");
            assert!(after >= before, "{name}: {label} went backwards at run {i}: {before} then {after}");
        }
        previous = (now.host_allocs, now.provider_allocs);
    }
    let end = stats(session);
    assert_eq!(end.runs, ITERATIONS + 1, "{name}: runs must count every completed run");
    if warm.host_allocs != u64::MAX {
        assert_eq!(end.host_allocs, warm.host_allocs, "{name}: the adapter allocated on the run path");
    } else {
        assert_eq!(end.host_allocs, u64::MAX, "{name}: host_allocs switched from not counted to counted");
    }
    assert_eq!(end.input_bytes, warm.input_bytes, "{name}: input storage grew after warmup");
    assert_eq!(end.output_bytes, warm.output_bytes, "{name}: output storage grew after warmup");
    if warm.provider_allocs == u64::MAX || end.provider_allocs == u64::MAX {
        println!("{name}: the provider does not report its own allocations");
        return;
    }
    assert_eq!(
        end.provider_allocs,
        warm.provider_allocs,
        "{name}: the provider allocated {} time(s) across {ITERATIONS} steady-state runs",
        end.provider_allocs - warm.provider_allocs
    );
    assert_eq!(end.provider_allocs, 0, "{name}: provider_allocs must be 0 after warmup");
    println!(
        "{name}: runs={} host_allocs={} provider_allocs={} in={} out={}",
        end.runs, end.host_allocs, end.provider_allocs, end.input_bytes, end.output_bytes
    );
}

struct Chain {
    _ct: c::CTarget,
    ctx: *mut turbo_context,
    model: *mut turbo_model,
    session: *mut turbo_session,
}

impl Chain {
    fn new(t: &Target, kind: BundleKind) -> Self {
        let ct = c::CTarget::new(t);
        let ctx = ct.context();
        let model = ct.model(ctx, kind);
        let session = ct.session(model);
        Self { _ct: ct, ctx, model, session }
    }
}

impl Drop for Chain {
    fn drop(&mut self) {
        // SAFETY: each handle is released exactly once.
        unsafe {
            turbo_session_release(self.session);
            turbo_model_release(self.model);
            turbo_context_release(self.ctx);
        }
    }
}

fn run_and_release(session: *mut turbo_session) {
    let mut e = c::err();
    let mut r: *mut turbo_result = ptr::null_mut();
    // SAFETY: valid session handle and out pointer.
    assert_rc!(unsafe { turbo_session_run(session, ptr::null(), &mut r, &mut e) }, TURBO_OK, e);
    // SAFETY: released once, returning the lease.
    unsafe { turbo_result_release(r) };
}

#[test]
fn allocation_embed_is_allocation_free_after_warmup() {
    let t = Target::from_env();
    let chain = Chain::new(&t, BundleKind::Embedding);
    let texts = [c::text("hello world"), c::text("a second document")];
    let mut e = c::err();
    steady_state("embed", chain.session, || {
        // SAFETY: valid session and two readable text views.
        assert_rc!(
            unsafe { turbo_session_write_text(chain.session, texts.as_ptr(), 2, ptr::null(), &mut e) },
            TURBO_OK,
            e
        );
        run_and_release(chain.session);
    });
}

#[test]
fn allocation_rerank_is_allocation_free_after_warmup() {
    let t = Target::from_env();
    let chain = Chain::new(&t, BundleKind::Reranker);
    let query = c::text("alpha gamma");
    let docs = [c::text("alpha beta"), c::text("gamma delta"), c::text("epsilon")];
    let mut e = c::err();
    steady_state("rerank", chain.session, || {
        // SAFETY: valid session; the arrays outlive the call.
        assert_rc!(
            unsafe { turbo_session_write_pairs(chain.session, &query, docs.as_ptr(), 3, ptr::null(), &mut e) },
            TURBO_OK,
            e
        );
        run_and_release(chain.session);
    });
}

#[test]
fn allocation_classify_is_allocation_free_after_warmup() {
    let t = Target::from_env();
    let chain = Chain::new(&t, BundleKind::Classifier);
    let texts = [c::text("hello world"), c::text("another line")];
    let mut e = c::err();
    steady_state("classify", chain.session, || {
        // SAFETY: valid session and two readable text views.
        assert_rc!(
            unsafe { turbo_session_write_text_classify(chain.session, texts.as_ptr(), 2, ptr::null(), &mut e) },
            TURBO_OK,
            e
        );
        run_and_release(chain.session);
    });
}

#[test]
fn allocation_token_classify_is_allocation_free_after_warmup() {
    let t = Target::from_env();
    let chain = Chain::new(&t, BundleKind::TokenClassifier);
    let texts = [c::text("Alice went to Paris"), c::text("Bob stayed home")];
    let mut e = c::err();
    steady_state("token_classify", chain.session, || {
        // SAFETY: valid session and two readable text views.
        assert_rc!(
            unsafe { turbo_session_write_text_classify(chain.session, texts.as_ptr(), 2, ptr::null(), &mut e) },
            TURBO_OK,
            e
        );
        run_and_release(chain.session);
    });
}

#[test]
fn allocation_generic_run_is_allocation_free_after_warmup() {
    let t = Target::from_env();
    let chain = Chain::new(&t, BundleKind::Generic);
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
    let mut x: *mut turbo_buffer = ptr::null_mut();
    let mut y: *mut turbo_buffer = ptr::null_mut();
    // SAFETY: valid context handle and descriptor.
    unsafe {
        assert_rc!(turbo_buffer_alloc(chain.ctx, &desc, &mut x, &mut e), TURBO_OK, e);
        assert_rc!(turbo_buffer_alloc(chain.ctx, &desc, &mut y, &mut e), TURBO_OK, e);
        assert_rc!(turbo_session_bind(chain.session, c::text("x"), x, &mut e), TURBO_OK, e);
        assert_rc!(turbo_session_bind(chain.session, c::text("y"), y, &mut e), TURBO_OK, e);
    }
    steady_state("run", chain.session, || run_and_release(chain.session));
    // SAFETY: each buffer is released once.
    unsafe {
        turbo_buffer_release(x);
        turbo_buffer_release(y);
    }
}

#[test]
fn allocation_stats_descriptor_is_validated() {
    let t = Target::from_env();
    let chain = Chain::new(&t, BundleKind::Embedding);
    let mut e = c::err();
    let mut st = turbo_session_stats { struct_size: ssz::<turbo_session_stats>() + 8, ..Default::default() };
    // SAFETY: valid session handle and out pointer.
    assert_rc!(unsafe { turbo_session_get_stats(chain.session, &mut st, &mut e) }, TURBO_E_INVALID_STRUCT_SIZE, e);
    // An older caller asking only for `runs` gets just that prefix.
    let mut short = turbo_session_stats { struct_size: 16, ..Default::default() };
    short.host_allocs = 0xDEAD_BEEF;
    // SAFETY: as above.
    assert_rc!(unsafe { turbo_session_get_stats(chain.session, &mut short, &mut e) }, TURBO_OK, e);
    assert_eq!(short.host_allocs, 0xDEAD_BEEF, "fields past the declared struct_size must not be written");
    // A fresh session has run nothing.
    let full = stats(chain.session);
    assert_eq!(full.runs, 0);
    assert_eq!(full.reserved, 0, "reserved is always zero");
}
