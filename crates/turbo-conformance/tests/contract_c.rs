//! Group `contract`, C ABI layer: the rules only raw pointers can break --
//! text views, handles, out-pointers, enumerations and token batches.
//! `struct_size`, the error record and the status vocabulary are in
//! `contract_c_errors.rs`.

use std::ptr;

use turbo_abi::*;
use turbo_capi::*;
use turbo_conformance::c::{self, ssz};
use turbo_conformance::{assert_rc, BundleKind, Target};

struct Fixture {
    _target: Target,
    ct: c::CTarget,
    ctx: *mut turbo_context,
    model: *mut turbo_model,
    session: *mut turbo_session,
}

impl Fixture {
    fn new() -> Self {
        let target = Target::from_env();
        let ct = c::CTarget::new(&target);
        let ctx = ct.context();
        let model = ct.model(ctx, BundleKind::Embedding);
        let session = ct.session(model);
        Self { _target: target, ct, ctx, model, session }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        // SAFETY: each handle was produced by this fixture and is released once.
        unsafe {
            turbo_session_release(self.session);
            turbo_model_release(self.model);
            turbo_context_release(self.ctx);
        }
    }
}

#[test]
fn contract_invalid_utf8_is_rejected() {
    let f = Fixture::new();
    let mut e = c::err();
    let bad: [u8; 3] = [0x68, 0xff, 0xfe];
    let view = turbo_text { ptr: bad.as_ptr().cast(), len: 3 };
    // SAFETY: the view covers three readable bytes.
    assert_rc!(unsafe { turbo_session_write_text(f.session, &view, 1, ptr::null(), &mut e) }, TURBO_E_INVALID_UTF8, e);
    // A lone continuation byte and a truncated sequence are equally rejected.
    for bytes in [vec![0x80u8], vec![0xE2, 0x82]] {
        let view = turbo_text { ptr: bytes.as_ptr().cast(), len: bytes.len() as u64 };
        assert_rc!(
            unsafe { turbo_session_write_text(f.session, &view, 1, ptr::null(), &mut e) },
            TURBO_E_INVALID_UTF8,
            e
        );
    }
}

#[test]
fn contract_null_text_with_nonzero_len_is_invalid_argument() {
    let f = Fixture::new();
    let mut e = c::err();
    let view = turbo_text { ptr: ptr::null(), len: 5 };
    // SAFETY: the library must not dereference a NULL view; it reports the error.
    assert_rc!(
        unsafe { turbo_session_write_text(f.session, &view, 1, ptr::null(), &mut e) },
        TURBO_E_INVALID_ARGUMENT,
        e
    );
    // A NULL pointer with length 0 is the empty string and is accepted.
    let empty = turbo_text { ptr: ptr::null(), len: 0 };
    assert_rc!(unsafe { turbo_session_write_text(f.session, &empty, 1, ptr::null(), &mut e) }, TURBO_OK, e);
}

#[test]
fn contract_null_text_array_with_nonzero_count_is_invalid_argument() {
    let f = Fixture::new();
    let mut e = c::err();
    // SAFETY: count > 0 with a NULL array must be detected before any read.
    assert_rc!(
        unsafe { turbo_session_write_text(f.session, ptr::null(), 2, ptr::null(), &mut e) },
        TURBO_E_INVALID_ARGUMENT,
        e
    );
}

/// A field index paired with the mutation that makes that field invalid.
type OptionCase = (u32, fn(&mut turbo_embed_options));

#[test]
fn contract_unknown_enum_values_report_their_field() {
    let f = Fixture::new();
    let mut e = c::err();
    let s = "hello";
    let texts = [c::text(s)];
    // (field index, mutator) pairs follow the 1-based ABI field order.
    let cases: [OptionCase; 5] = [
        (2, |o| o.truncate = 77),
        (4, |o| o.prompt_role = 77),
        (5, |o| o.normalize = 77),
        (6, |o| o.pooling = 77),
        (8, |o| o.output_dtype = 77),
    ];
    for (field, mutate) in cases {
        let mut o = c::embed_options();
        mutate(&mut o);
        // SAFETY: valid session, one readable text, well-formed options.
        assert_rc!(
            unsafe { turbo_session_write_text(f.session, texts.as_ptr(), 1, &o, &mut e) },
            TURBO_E_INVALID_ENUM,
            e,
            field = field
        );
    }
}

#[test]
fn contract_null_handles_are_invalid_handle() {
    let mut e = c::err();
    let mut out_u32 = 0u32;
    let mut out_ptr: *mut turbo_context = ptr::null_mut();
    let s = "x";
    let texts = [c::text(s)];
    // SAFETY: every call below is given a NULL handle on purpose.
    unsafe {
        assert_rc!(turbo_runtime_device_count(ptr::null_mut(), &mut out_u32, &mut e), TURBO_E_INVALID_HANDLE, e);
        assert_rc!(
            turbo_context_create(ptr::null_mut(), 0, ptr::null(), &mut out_ptr, &mut e),
            TURBO_E_INVALID_HANDLE,
            e
        );
        let mut m: *mut turbo_model = ptr::null_mut();
        assert_rc!(
            turbo_model_load(ptr::null_mut(), c::text("/nonexistent"), ptr::null(), &mut m, &mut e),
            TURBO_E_INVALID_HANDLE,
            e
        );
        let mut sess: *mut turbo_session = ptr::null_mut();
        assert_rc!(turbo_session_create(ptr::null_mut(), ptr::null(), &mut sess, &mut e), TURBO_E_INVALID_HANDLE, e);
        assert_rc!(
            turbo_session_write_text(ptr::null_mut(), texts.as_ptr(), 1, ptr::null(), &mut e),
            TURBO_E_INVALID_HANDLE,
            e
        );
        let mut r: *mut turbo_result = ptr::null_mut();
        assert_rc!(turbo_session_run(ptr::null_mut(), ptr::null(), &mut r, &mut e), TURBO_E_INVALID_HANDLE, e);
        let mut written = 0u64;
        let mut dst = [0u8; 8];
        assert_rc!(
            turbo_result_read(ptr::null_mut(), 0, dst.as_mut_ptr().cast(), 8, &mut written, &mut e),
            TURBO_E_INVALID_HANDLE,
            e
        );
        let mut gen: *mut turbo_generation = ptr::null_mut();
        assert_rc!(turbo_generation_create(ptr::null_mut(), ptr::null(), &mut gen, &mut e), TURBO_E_INVALID_HANDLE, e);
        assert_rc!(turbo_generation_cancel(ptr::null_mut(), &mut e), TURBO_E_INVALID_HANDLE, e);
        // Releasing a NULL handle is a documented no-op, not a crash.
        turbo_runtime_release(ptr::null_mut());
        turbo_context_release(ptr::null_mut());
        turbo_model_release(ptr::null_mut());
        turbo_session_release(ptr::null_mut());
        turbo_result_release(ptr::null_mut());
        turbo_generation_release(ptr::null_mut());
        turbo_buffer_release(ptr::null_mut());
    }
}

#[test]
fn contract_null_out_pointers_are_invalid_argument() {
    let f = Fixture::new();
    let mut e = c::err();
    // SAFETY: NULL out-pointers must be reported, never written through.
    unsafe {
        assert_rc!(turbo_runtime_create(ptr::null(), ptr::null_mut(), &mut e), TURBO_E_INVALID_ARGUMENT, e);
        assert_rc!(turbo_runtime_device_count(f.ct.rt, ptr::null_mut(), &mut e), TURBO_E_INVALID_ARGUMENT, e);
        assert_rc!(
            turbo_runtime_device_info(f.ct.rt, f.ct.device, ptr::null_mut(), &mut e),
            TURBO_E_INVALID_ARGUMENT,
            e
        );
        assert_rc!(
            turbo_runtime_select_device(f.ct.rt, ptr::null(), ptr::null_mut(), &mut e),
            TURBO_E_INVALID_ARGUMENT,
            e
        );
        assert_rc!(
            turbo_context_create(f.ct.rt, f.ct.device, ptr::null(), ptr::null_mut(), &mut e),
            TURBO_E_INVALID_ARGUMENT,
            e
        );
        assert_rc!(turbo_model_get_info(f.model, ptr::null_mut(), &mut e), TURBO_E_INVALID_ARGUMENT, e);
        assert_rc!(turbo_session_create(f.model, ptr::null(), ptr::null_mut(), &mut e), TURBO_E_INVALID_ARGUMENT, e);
        assert_rc!(turbo_session_run(f.session, ptr::null(), ptr::null_mut(), &mut e), TURBO_E_INVALID_ARGUMENT, e);
        assert_rc!(turbo_session_get_stats(f.session, ptr::null_mut(), &mut e), TURBO_E_INVALID_ARGUMENT, e);
    }
}

#[test]
fn contract_zero_count_batch_is_invalid_argument() {
    let f = Fixture::new();
    let mut e = c::err();
    let s = "x";
    let texts = [c::text(s)];
    // SAFETY: a zero count with a valid array is still an empty batch.
    assert_rc!(
        unsafe { turbo_session_write_text(f.session, texts.as_ptr(), 0, ptr::null(), &mut e) },
        TURBO_E_INVALID_ARGUMENT,
        e
    );
}

#[test]
fn contract_batch_above_session_max_is_capacity() {
    let f = Fixture::new();
    let mut e = c::err();
    let mut info = turbo_model_info { struct_size: ssz::<turbo_model_info>(), ..unsafe { std::mem::zeroed() } };
    assert_rc!(unsafe { turbo_model_get_info(f.model, &mut info, &mut e) }, TURBO_OK, e);
    let s = "x";
    let texts = vec![c::text(s); info.max_batch as usize + 1];
    // SAFETY: the array holds `max_batch + 1` readable views.
    assert_rc!(
        unsafe { turbo_session_write_text(f.session, texts.as_ptr(), texts.len() as u32, ptr::null(), &mut e) },
        TURBO_E_CAPACITY,
        e
    );
}

#[test]
fn contract_oversized_buffer_shapes_are_invalid_shape() {
    let f = Fixture::new();
    let mut e = c::err();
    let mut out: *mut turbo_buffer = ptr::null_mut();
    let base = turbo_buffer_desc {
        struct_size: ssz::<turbo_buffer_desc>(),
        placement: TURBO_PLACE_HOST,
        dtype: TURBO_DTYPE_F32,
        ndim: 2,
        shape: [u64::MAX, 2, 0, 0, 0, 0, 0, 0],
        strides: [0; TURBO_MAX_RANK],
        bytes: 0,
        next: ptr::null(),
    };
    // SAFETY: valid context and descriptor pointer.
    unsafe {
        assert_rc!(turbo_buffer_alloc(f.ctx, &base, &mut out, &mut e), TURBO_E_INVALID_SHAPE, e);
        // Rank above TURBO_MAX_RANK.
        let deep = turbo_buffer_desc { ndim: (TURBO_MAX_RANK + 1) as u32, shape: [1; TURBO_MAX_RANK], ..base };
        assert_rc!(turbo_buffer_alloc(f.ctx, &deep, &mut out, &mut e), TURBO_E_INVALID_SHAPE, e);
        // Byte count smaller than the shape needs.
        let short = turbo_buffer_desc { shape: [4, 4, 0, 0, 0, 0, 0, 0], bytes: 8, ..base };
        assert_rc!(turbo_buffer_alloc(f.ctx, &short, &mut out, &mut e), TURBO_E_INVALID_SHAPE, e);
        // Strides that reach past the allocation.
        let strided = turbo_buffer_desc {
            shape: [4, 4, 0, 0, 0, 0, 0, 0],
            strides: [1024, 4, 0, 0, 0, 0, 0, 0],
            bytes: 64,
            ..base
        };
        assert_rc!(turbo_buffer_alloc(f.ctx, &strided, &mut out, &mut e), TURBO_E_INVALID_SHAPE, e);
        assert!(out.is_null(), "a rejected descriptor must not produce a buffer");
        // An unknown dtype or placement is an enum error, not a shape error.
        let bad_dtype = turbo_buffer_desc { dtype: 99, shape: [2, 2, 0, 0, 0, 0, 0, 0], ..base };
        assert_rc!(turbo_buffer_alloc(f.ctx, &bad_dtype, &mut out, &mut e), TURBO_E_INVALID_ENUM, e);
        let bad_place = turbo_buffer_desc { placement: 99, shape: [2, 2, 0, 0, 0, 0, 0, 0], ..base };
        assert_rc!(turbo_buffer_alloc(f.ctx, &bad_place, &mut out, &mut e), TURBO_E_INVALID_ENUM, e);
        // NULL descriptor.
        assert_rc!(turbo_buffer_alloc(f.ctx, ptr::null(), &mut out, &mut e), TURBO_E_INVALID_ARGUMENT, e);
    }
}

#[test]
fn contract_token_batch_validation_reports_exact_codes() {
    let f = Fixture::new();
    let mut e = c::err();
    let mut info = turbo_model_info { struct_size: ssz::<turbo_model_info>(), ..unsafe { std::mem::zeroed() } };
    assert_rc!(unsafe { turbo_model_get_info(f.model, &mut info, &mut e) }, TURBO_OK, e);

    let ids: Vec<i32> = vec![1, 2, 3, 4];
    let mask: Vec<i32> = vec![1, 1, 1, 1];
    let base = turbo_token_batch {
        struct_size: ssz::<turbo_token_batch>(),
        batch: 2,
        seq: 2,
        row_stride: 2,
        ids: ids.as_ptr(),
        mask: mask.as_ptr(),
        types: ptr::null(),
    };
    // SAFETY: the arrays hold the elements each descriptor declares.
    unsafe {
        assert_rc!(turbo_session_write_tokens(f.session, &base, &mut e), TURBO_OK, e);
        // Zero dimensions.
        assert_rc!(
            turbo_session_write_tokens(f.session, &turbo_token_batch { batch: 0, ..base }, &mut e),
            TURBO_E_INVALID_SHAPE,
            e
        );
        assert_rc!(
            turbo_session_write_tokens(f.session, &turbo_token_batch { seq: 0, ..base }, &mut e),
            TURBO_E_INVALID_SHAPE,
            e
        );
        // row_stride below seq would overlap rows.
        assert_rc!(
            turbo_session_write_tokens(f.session, &turbo_token_batch { seq: 2, row_stride: 1, ..base }, &mut e),
            TURBO_E_INVALID_SHAPE,
            e
        );
        // NULL arrays name their field.
        assert_rc!(
            turbo_session_write_tokens(f.session, &turbo_token_batch { ids: ptr::null(), ..base }, &mut e),
            TURBO_E_INVALID_ARGUMENT,
            e,
            field = 5
        );
        assert_rc!(
            turbo_session_write_tokens(f.session, &turbo_token_batch { mask: ptr::null(), ..base }, &mut e),
            TURBO_E_INVALID_ARGUMENT,
            e,
            field = 6
        );
        // Out-of-vocabulary and negative ids.
        let oob: Vec<i32> = vec![1, info.vocab_size as i32, 3, 4];
        assert_rc!(
            turbo_session_write_tokens(f.session, &turbo_token_batch { ids: oob.as_ptr(), ..base }, &mut e),
            TURBO_E_INVALID_ARGUMENT,
            e
        );
        let negative: Vec<i32> = vec![1, -1, 3, 4];
        assert_rc!(
            turbo_session_write_tokens(f.session, &turbo_token_batch { ids: negative.as_ptr(), ..base }, &mut e),
            TURBO_E_INVALID_ARGUMENT,
            e
        );
        // Mask values other than 0 and 1.
        let bad_mask: Vec<i32> = vec![1, 2, 1, 1];
        assert_rc!(
            turbo_session_write_tokens(f.session, &turbo_token_batch { mask: bad_mask.as_ptr(), ..base }, &mut e),
            TURBO_E_INVALID_ARGUMENT,
            e
        );
        // Token types other than 0 and 1.
        let bad_types: Vec<i32> = vec![0, 0, 7, 0];
        assert_rc!(
            turbo_session_write_tokens(f.session, &turbo_token_batch { types: bad_types.as_ptr(), ..base }, &mut e),
            TURBO_E_INVALID_ARGUMENT,
            e
        );
        // Sequence beyond the session's maximum.
        let long_ids: Vec<i32> = vec![1; info.max_seq as usize + 1];
        let long_mask: Vec<i32> = vec![1; info.max_seq as usize + 1];
        let long = turbo_token_batch {
            batch: 1,
            seq: info.max_seq + 1,
            row_stride: info.max_seq + 1,
            ids: long_ids.as_ptr(),
            mask: long_mask.as_ptr(),
            ..base
        };
        assert_rc!(turbo_session_write_tokens(f.session, &long, &mut e), TURBO_E_CAPACITY, e);
        // A NULL descriptor and a bad struct_size.
        assert_rc!(turbo_session_write_tokens(f.session, ptr::null(), &mut e), TURBO_E_INVALID_ARGUMENT, e);
        let big = turbo_token_batch { struct_size: ssz::<turbo_token_batch>() + 8, ..base };
        assert_rc!(turbo_session_write_tokens(f.session, &big, &mut e), TURBO_E_INVALID_STRUCT_SIZE, e);
    }
}
