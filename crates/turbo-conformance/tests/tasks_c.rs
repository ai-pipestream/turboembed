//! Group `tasks`, C ABI layer: the output surface of each task, including
//! `turbo_result_*`, `turbo_model_io_info`, and the generic RUN bindings.

use std::ptr;

use turbo_abi::*;
use turbo_capi::*;
use turbo_conformance::c::{self, ssz};
use turbo_conformance::{assert_rc, needs, BundleKind, Target};

const DOCS: [&str; 5] = ["alpha beta", "beta gamma", "alpha beta", "zzz", "alpha beta gamma delta"];

struct Chain {
    _ct: c::CTarget,
    ctx: *mut turbo_context,
    model: *mut turbo_model,
    session: *mut turbo_session,
    info: turbo_model_info,
}

impl Chain {
    fn new(t: &Target, kind: BundleKind) -> Self {
        let ct = c::CTarget::new(t);
        let ctx = ct.context();
        let model = ct.model(ctx, kind);
        let session = ct.session(model);
        let mut e = c::err();
        let mut info = turbo_model_info { struct_size: ssz::<turbo_model_info>(), ..unsafe { std::mem::zeroed() } };
        // SAFETY: valid model handle and out pointer.
        assert_rc!(unsafe { turbo_model_get_info(model, &mut info, &mut e) }, TURBO_OK, e);
        Self { _ct: ct, ctx, model, session, info }
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

fn output_info(r: *mut turbo_result, index: u32) -> turbo_tensor_info {
    let mut e = c::err();
    let mut ti = turbo_tensor_info { struct_size: ssz::<turbo_tensor_info>(), ..unsafe { std::mem::zeroed() } };
    // SAFETY: valid result handle and out pointer.
    assert_rc!(unsafe { turbo_result_output_info(r, index, &mut ti, &mut e) }, TURBO_OK, e);
    ti
}

#[test]
fn tasks_rerank_reports_scores_and_a_sorted_index() {
    let t = Target::from_env();
    needs!(t, Reranker);
    let chain = Chain::new(&t, BundleKind::Reranker);
    let mut e = c::err();
    let query = c::text("alpha beta");
    let docs: Vec<turbo_text> = DOCS.iter().map(|d| c::text(d)).collect();
    let mut o = c::rerank_options();
    o.return_sorted = if t.caps() & TURBO_CAP_OPT_TOP_N != 0 { 1 } else { 0 };
    // SAFETY: valid session; the arrays outlive the call.
    assert_rc!(
        unsafe { turbo_session_write_pairs(chain.session, &query, docs.as_ptr(), docs.len() as u32, &o, &mut e) },
        TURBO_OK,
        e
    );
    let mut r: *mut turbo_result = ptr::null_mut();
    // SAFETY: valid session handle and out pointer.
    assert_rc!(unsafe { turbo_session_run(chain.session, ptr::null(), &mut r, &mut e) }, TURBO_OK, e);

    let ri = c::result_info(r);
    assert_eq!(ri.batch, DOCS.len() as u32, "one score per document");
    assert_eq!(ri.dim, 1, "scores are one column wide");
    assert_eq!(ri.dtype, TURBO_DTYPE_F32);
    assert_eq!(ri.bytes, (DOCS.len() * 4) as u64);
    let scores = c::read_f32(r, 0, DOCS.len());
    assert_eq!(scores[0], scores[2], "identical documents must score identically");
    assert!(scores[0] > scores[3], "a matching document must outrank an unrelated one: {scores:?}");

    let first = output_info(r, 0);
    assert_eq!(c::fixed(&first.name), "scores");
    assert_eq!(first.ndim, 1);
    assert_eq!(first.shape[0], DOCS.len() as i64);

    if o.return_sorted == 1 {
        assert_eq!(ri.n_outputs, 2);
        let second = output_info(r, 1);
        assert_eq!(c::fixed(&second.name), "sorted");
        assert_eq!(second.dtype, TURBO_DTYPE_I32);
        let mut sorted = vec![0i32; DOCS.len()];
        let mut written = 0u64;
        // SAFETY: `sorted` holds DOCS.len() writable i32 values.
        assert_rc!(
            unsafe {
                turbo_result_read(r, 1, sorted.as_mut_ptr().cast(), (DOCS.len() * 4) as u64, &mut written, &mut e)
            },
            TURBO_OK,
            e
        );
        assert_eq!(written, (DOCS.len() * 4) as u64);
        for pair in sorted.windows(2) {
            let (a, b) = (pair[0] as usize, pair[1] as usize);
            assert!(scores[a] >= scores[b], "sorted output is not descending: {sorted:?}");
            if scores[a] == scores[b] {
                assert!(a < b, "ties must keep input order");
            }
        }
    }
    // An output index past the end is refused on every accessor.
    let mut ti = turbo_tensor_info { struct_size: ssz::<turbo_tensor_info>(), ..unsafe { std::mem::zeroed() } };
    let mut buf: *mut turbo_buffer = ptr::null_mut();
    let mut written = 0u64;
    // SAFETY: valid result handle; the indices are deliberately out of range.
    unsafe {
        assert_rc!(turbo_result_output_info(r, 9, &mut ti, &mut e), TURBO_E_INVALID_ARGUMENT, e);
        assert_rc!(turbo_result_buffer(r, 9, &mut buf, &mut e), TURBO_E_INVALID_ARGUMENT, e);
        let mut dst = [0u8; 8];
        assert_rc!(
            turbo_result_read(r, 9, dst.as_mut_ptr().cast(), 8, &mut written, &mut e),
            TURBO_E_INVALID_ARGUMENT,
            e
        );
        turbo_result_release(r);
    }
}

#[test]
fn tasks_classify_rows_sum_to_one_unless_raw() {
    let t = Target::from_env();
    needs!(t, Classifier);
    let chain = Chain::new(&t, BundleKind::Classifier);
    let mut e = c::err();
    let n_labels = chain.info.n_labels as usize;
    assert!(n_labels > 0, "a classifier declares labels");
    let texts = [c::text("hello world"), c::text("another line")];
    // SAFETY: valid session and two readable text views.
    assert_rc!(
        unsafe { turbo_session_write_text_classify(chain.session, texts.as_ptr(), 2, ptr::null(), &mut e) },
        TURBO_OK,
        e
    );
    let mut r: *mut turbo_result = ptr::null_mut();
    // SAFETY: valid session handle and out pointer.
    assert_rc!(unsafe { turbo_session_run(chain.session, ptr::null(), &mut r, &mut e) }, TURBO_OK, e);
    let ri = c::result_info(r);
    assert_eq!(ri.batch, 2);
    assert_eq!(ri.dim as usize, n_labels);
    let values = c::read_f32(r, 0, 2 * n_labels);
    for (i, row) in values.chunks(n_labels).enumerate() {
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "row {i} sums to {sum}: {row:?}");
    }
    // SAFETY: released once.
    unsafe { turbo_result_release(r) };

    let mut raw = c::classify_options();
    raw.raw_scores = 1;
    // SAFETY: as above.
    assert_rc!(
        unsafe { turbo_session_write_text_classify(chain.session, texts.as_ptr(), 2, &raw, &mut e) },
        TURBO_OK,
        e
    );
    // SAFETY: valid session handle and out pointer.
    assert_rc!(unsafe { turbo_session_run(chain.session, ptr::null(), &mut r, &mut e) }, TURBO_OK, e);
    let raw_values = c::read_f32(r, 0, 2 * n_labels);
    let any_unnormalized = raw_values.chunks(n_labels).any(|r| (r.iter().sum::<f32>() - 1.0).abs() > 1e-4);
    assert!(any_unnormalized, "raw_scores must return logits: {raw_values:?}");
    // SAFETY: released once.
    unsafe { turbo_result_release(r) };

    // Labels are readable by index and bounded.
    let mut label = c::null_text();
    // SAFETY: valid model handle and out pointer.
    unsafe {
        assert_rc!(turbo_model_label(chain.model, 0, &mut label, &mut e), TURBO_OK, e);
        assert!(label.len > 0 && !label.ptr.is_null(), "label 0 is empty");
        assert_rc!(
            turbo_model_label(chain.model, chain.info.n_labels, &mut label, &mut e),
            TURBO_E_INVALID_ARGUMENT,
            e
        );
    }
}

#[test]
fn tasks_token_classify_spans_slice_the_input() {
    let t = Target::from_env();
    needs!(t, TokenClassifier);
    let chain = Chain::new(&t, BundleKind::TokenClassifier);
    let mut e = c::err();
    let texts_src = ["Alice went to Paris", "Bob"];
    let texts: Vec<turbo_text> = texts_src.iter().map(|s| c::text(s)).collect();
    // SAFETY: valid session; the array outlives the call.
    assert_rc!(
        unsafe { turbo_session_write_text_classify(chain.session, texts.as_ptr(), 2, ptr::null(), &mut e) },
        TURBO_OK,
        e
    );
    let mut r: *mut turbo_result = ptr::null_mut();
    // SAFETY: valid session handle and out pointer.
    assert_rc!(unsafe { turbo_session_run(chain.session, ptr::null(), &mut r, &mut e) }, TURBO_OK, e);

    // The scores tensor is [batch, seq, n_labels].
    let ti = output_info(r, 0);
    assert_eq!(ti.ndim, 3, "the score tensor is [batch, seq, n_labels]");
    assert_eq!(ti.shape[0], 2);
    assert_eq!(ti.shape[2], chain.info.n_labels as i64);
    let ri = c::result_info(r);
    assert_eq!(ri.dim as i64, ti.shape[1] * ti.shape[2], "result_info.dim is the row width");

    // Asking with a zero capacity reports the count without writing.
    let mut count = 0u32;
    // SAFETY: NULL with capacity 0 is the documented "count only" form.
    assert_rc!(unsafe { turbo_result_spans(r, ptr::null_mut(), 0, &mut count, &mut e) }, TURBO_OK, e);
    assert!(count > 0, "token classification must report spans");

    // A short buffer copies a prefix and still reports the total.
    let mut one = vec![turbo_span::default(); 1];
    let mut short_count = 0u32;
    // SAFETY: `one` holds one writable span.
    assert_rc!(unsafe { turbo_result_spans(r, one.as_mut_ptr(), 1, &mut short_count, &mut e) }, TURBO_OK, e);
    assert_eq!(short_count, count, "the total is reported whatever the capacity");

    let mut spans = vec![turbo_span::default(); count as usize];
    // SAFETY: `spans` holds `count` writable entries.
    assert_rc!(unsafe { turbo_result_spans(r, spans.as_mut_ptr(), count, &mut count, &mut e) }, TURBO_OK, e);
    assert_eq!(spans[0].byte_start, one[0].byte_start, "the prefix copy matches the full copy");
    for s in &spans {
        let row = s.row as usize;
        assert!(row < texts_src.len(), "span row {row} is out of range");
        let text = texts_src[row];
        let (start, end) = (s.byte_start as usize, s.byte_end as usize);
        assert!(start < end && end <= text.len(), "span {s:?} does not lie inside row {row}");
        assert!(text.is_char_boundary(start) && text.is_char_boundary(end), "span {s:?} splits a character");
        let slice = &text[start..end];
        assert!(!slice.trim().is_empty(), "span {s:?} is whitespace");
        assert!(start == 0 || text[..start].ends_with(char::is_whitespace), "span {s:?} starts mid-word");
        assert!(end == text.len() || text[end..].starts_with(char::is_whitespace), "span {s:?} ends mid-word");
        assert!(s.label < chain.info.n_labels, "span {s:?} names no label");
        assert_eq!(s.reserved, 0, "reserved must be zero");
    }
    // A NULL count pointer is an argument error.
    // SAFETY: valid result handle; the out pointer is deliberately NULL.
    unsafe {
        assert_rc!(turbo_result_spans(r, spans.as_mut_ptr(), 1, ptr::null_mut(), &mut e), TURBO_E_INVALID_ARGUMENT, e);
        assert_rc!(turbo_result_spans(r, ptr::null_mut(), 4, &mut count, &mut e), TURBO_E_INVALID_ARGUMENT, e);
        turbo_result_release(r);
    }
}

#[test]
fn tasks_generic_run_computes_y_equals_two_x() {
    let t = Target::from_env();
    needs!(t, Generic);
    let chain = Chain::new(&t, BundleKind::Generic);
    let mut e = c::err();
    // The model's named I/O is discoverable.
    let mut ti = turbo_tensor_info { struct_size: ssz::<turbo_tensor_info>(), ..unsafe { std::mem::zeroed() } };
    // SAFETY: valid model handle and out pointer.
    unsafe {
        assert_rc!(turbo_model_io_info(chain.model, TURBO_IO_INPUT, 0, &mut ti, &mut e), TURBO_OK, e);
        assert_eq!(c::fixed(&ti.name), "x");
        assert_eq!(ti.dtype, TURBO_DTYPE_F32);
        assert_rc!(turbo_model_io_info(chain.model, TURBO_IO_OUTPUT, 0, &mut ti, &mut e), TURBO_OK, e);
        assert_eq!(c::fixed(&ti.name), "y");
        assert_rc!(turbo_model_io_info(chain.model, TURBO_IO_INPUT, 99, &mut ti, &mut e), TURBO_E_INVALID_ARGUMENT, e);
        assert_rc!(turbo_model_io_info(chain.model, 99, 0, &mut ti, &mut e), TURBO_E_INVALID_ENUM, e);
    }

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
    let values: Vec<f32> = (0..6).map(|i| i as f32 - 2.0).collect();
    // SAFETY: valid context handle, descriptor, and out pointers.
    unsafe {
        assert_rc!(turbo_buffer_alloc(chain.ctx, &desc, &mut x, &mut e), TURBO_OK, e);
        assert_rc!(turbo_buffer_alloc(chain.ctx, &desc, &mut y, &mut e), TURBO_OK, e);
        let mut host: *mut std::ffi::c_void = ptr::null_mut();
        assert_rc!(turbo_buffer_host_ptr(x, &mut host, &mut e), TURBO_OK, e);
        assert!(!host.is_null());
        std::ptr::copy_nonoverlapping(values.as_ptr(), host.cast::<f32>(), values.len());

        assert_rc!(turbo_session_bind(chain.session, c::text("x"), x, &mut e), TURBO_OK, e);
        assert_rc!(turbo_session_bind(chain.session, c::text("y"), y, &mut e), TURBO_OK, e);
        let mut r: *mut turbo_result = ptr::null_mut();
        assert_rc!(turbo_session_run(chain.session, ptr::null(), &mut r, &mut e), TURBO_OK, e);
        let out = c::read_f32(r, 0, values.len());
        for (i, v) in values.iter().enumerate() {
            assert_eq!(out[i], v * 2.0, "y[{i}] must be 2x");
        }
        // The caller's own buffer was written in place.
        let mut y_host: *mut std::ffi::c_void = ptr::null_mut();
        assert_rc!(turbo_buffer_host_ptr(y, &mut y_host, &mut e), TURBO_OK, e);
        let written = std::slice::from_raw_parts(y_host.cast::<f32>(), values.len());
        assert_eq!(written, &out[..], "the bound output must hold the result");
        // The result's shape follows the input.
        let ti = output_info(r, 0);
        assert_eq!(ti.ndim, 2);
        assert_eq!([ti.shape[0], ti.shape[1]], [2, 3]);
        turbo_result_release(r);
        turbo_buffer_release(x);
        turbo_buffer_release(y);
    }
}

#[test]
fn tasks_generic_run_rejects_a_small_or_aliased_output() {
    let t = Target::from_env();
    needs!(t, Generic);
    let chain = Chain::new(&t, BundleKind::Generic);
    let mut e = c::err();
    let big = turbo_buffer_desc {
        struct_size: ssz::<turbo_buffer_desc>(),
        placement: TURBO_PLACE_HOST,
        dtype: TURBO_DTYPE_F32,
        ndim: 2,
        shape: [2, 3, 0, 0, 0, 0, 0, 0],
        strides: [0; TURBO_MAX_RANK],
        bytes: 0,
        next: ptr::null(),
    };
    let small = turbo_buffer_desc { ndim: 1, shape: [2, 0, 0, 0, 0, 0, 0, 0], ..big };
    let mut x: *mut turbo_buffer = ptr::null_mut();
    let mut y_small: *mut turbo_buffer = ptr::null_mut();
    let mut r: *mut turbo_result = ptr::null_mut();
    // SAFETY: valid context and session handles; descriptors are initialized.
    unsafe {
        assert_rc!(turbo_buffer_alloc(chain.ctx, &big, &mut x, &mut e), TURBO_OK, e);
        assert_rc!(turbo_buffer_alloc(chain.ctx, &small, &mut y_small, &mut e), TURBO_OK, e);
        assert_rc!(turbo_session_bind(chain.session, c::text("x"), x, &mut e), TURBO_OK, e);
        assert_rc!(turbo_session_bind(chain.session, c::text("y"), y_small, &mut e), TURBO_OK, e);
        assert_rc!(turbo_session_run(chain.session, ptr::null(), &mut r, &mut e), TURBO_E_CAPACITY, e);
        assert!(r.is_null(), "a refused run must not hand back a result");
        // Aliasing the input and the output is refused, not silently allowed.
        assert_rc!(turbo_session_bind(chain.session, c::text("y"), x, &mut e), TURBO_OK, e);
        assert_rc!(turbo_session_run(chain.session, ptr::null(), &mut r, &mut e), TURBO_E_INVALID_ARGUMENT, e);
        assert!(r.is_null());
        turbo_buffer_release(x);
        turbo_buffer_release(y_small);
    }
}

#[test]
fn tasks_result_read_into_a_small_buffer_is_capacity() {
    let t = Target::from_env();
    needs!(t, Embedding);
    let chain = Chain::new(&t, BundleKind::Embedding);
    let mut e = c::err();
    let texts = [c::text("hello world")];
    // SAFETY: valid session and one readable text view.
    assert_rc!(unsafe { turbo_session_write_text(chain.session, texts.as_ptr(), 1, ptr::null(), &mut e) }, TURBO_OK, e);
    let mut r: *mut turbo_result = ptr::null_mut();
    // SAFETY: valid session handle and out pointer.
    assert_rc!(unsafe { turbo_session_run(chain.session, ptr::null(), &mut r, &mut e) }, TURBO_OK, e);
    let ri = c::result_info(r);
    let needed = ri.bytes;
    assert_eq!(needed, (chain.info.dim as u64) * 4);
    let mut dst = vec![0u8; needed as usize + 8];
    let mut written = 0u64;
    // SAFETY: `dst` is large enough for every capacity used below.
    unsafe {
        assert_rc!(
            turbo_result_read(r, 0, dst.as_mut_ptr().cast(), needed - 1, &mut written, &mut e),
            TURBO_E_CAPACITY,
            e
        );
        assert_rc!(turbo_result_read(r, 0, ptr::null_mut(), 0, &mut written, &mut e), TURBO_E_CAPACITY, e);
        // A NULL destination with a non-zero capacity is an argument error.
        assert_rc!(turbo_result_read(r, 0, ptr::null_mut(), needed, &mut written, &mut e), TURBO_E_INVALID_ARGUMENT, e);
        // Exactly enough, and more than enough, both report the logical size.
        assert_rc!(turbo_result_read(r, 0, dst.as_mut_ptr().cast(), needed, &mut written, &mut e), TURBO_OK, e);
        assert_eq!(written, needed);
        written = 0;
        assert_rc!(turbo_result_read(r, 0, dst.as_mut_ptr().cast(), needed + 8, &mut written, &mut e), TURBO_OK, e);
        assert_eq!(written, needed);
        // The `written` out-pointer is optional.
        assert_rc!(turbo_result_read(r, 0, dst.as_mut_ptr().cast(), needed, ptr::null_mut(), &mut e), TURBO_OK, e);
        turbo_result_release(r);
    }
}

#[test]
fn tasks_result_info_reports_the_output_shape() {
    let t = Target::from_env();
    needs!(t, Embedding);
    let chain = Chain::new(&t, BundleKind::Embedding);
    let mut e = c::err();
    let texts = [c::text("a"), c::text("b"), c::text("c")];
    // SAFETY: valid session; the array outlives the call.
    assert_rc!(unsafe { turbo_session_write_text(chain.session, texts.as_ptr(), 3, ptr::null(), &mut e) }, TURBO_OK, e);
    let mut r: *mut turbo_result = ptr::null_mut();
    // SAFETY: valid session handle and out pointer.
    assert_rc!(unsafe { turbo_session_run(chain.session, ptr::null(), &mut r, &mut e) }, TURBO_OK, e);
    let ri = c::result_info(r);
    assert_eq!(ri.n_outputs, 1);
    assert_eq!(ri.batch, 3);
    assert_eq!(ri.dim, chain.info.dim);
    assert_eq!(ri.dtype, TURBO_DTYPE_F32);
    let expected = if t.has(TURBO_CAP_DEVICE_RESULT) { TURBO_PLACE_DEVICE } else { TURBO_PLACE_HOST };
    assert_eq!(ri.placement, expected, "placement follows the device's TURBO_CAP_DEVICE_RESULT bit");
    assert_eq!(ri.bytes, 3 * chain.info.dim as u64 * 4);
    let ti = output_info(r, 0);
    assert_eq!(ti.ndim, 2);
    assert_eq!(ti.shape[0], 3);
    assert_eq!(ti.shape[1], chain.info.dim as i64);
    assert_eq!(c::fixed(&ti.name), "embeddings");
    // A result buffer view describes the same memory.
    let mut view: *mut turbo_buffer = ptr::null_mut();
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
    // SAFETY: valid result handle and out pointers.
    unsafe {
        assert_rc!(turbo_result_buffer(r, 0, &mut view, &mut e), TURBO_OK, e);
        assert_rc!(turbo_buffer_get_desc(view, &mut desc, &mut e), TURBO_OK, e);
        assert_eq!(desc.dtype, TURBO_DTYPE_F32);
        assert_eq!(desc.placement, expected, "the view's placement follows the device's TURBO_CAP_DEVICE_RESULT bit");
        assert!(desc.bytes >= ri.bytes, "the view covers at least the logical bytes");
        let mut native =
            turbo_native_handle { struct_size: ssz::<turbo_native_handle>(), kind: 0, handle: 0, aux: 0, offset: 0 };
        let copied = c::read_f32(r, 0, 3 * chain.info.dim as usize);
        if expected == TURBO_PLACE_HOST {
            // A host-visible result can be read without a copy.
            let mut host: *mut std::ffi::c_void = ptr::null_mut();
            assert_rc!(turbo_buffer_host_ptr(view, &mut host, &mut e), TURBO_OK, e);
            let direct = std::slice::from_raw_parts(host.cast::<f32>(), 3 * chain.info.dim as usize);
            assert_eq!(direct, &copied[..], "the zero-copy view and the explicit read must agree");
            // Exporting a host buffer yields the same pointer.
            assert_rc!(turbo_buffer_export(view, TURBO_HANDLE_HOST_PTR, &mut native, &mut e), TURBO_OK, e);
            assert_eq!(native.handle as usize, host as usize);
            assert_rc!(turbo_buffer_export(view, TURBO_HANDLE_CUDA_PTR, &mut native, &mut e), TURBO_E_UNSUPPORTED, e);
        } else {
            // A device-resident result has no host pointer; a read copies it
            // back, and the export of its own handle kind is non-zero.
            let mut host: *mut std::ffi::c_void = ptr::null_mut();
            assert_rc!(turbo_buffer_host_ptr(view, &mut host, &mut e), TURBO_E_UNSUPPORTED_PLACEMENT, e);
            assert_rc!(turbo_buffer_export(view, TURBO_HANDLE_HOST_PTR, &mut native, &mut e), TURBO_E_UNSUPPORTED, e);
            assert_eq!(copied.len(), 3 * chain.info.dim as usize);
            assert!(copied.iter().any(|x| *x != 0.0), "the copy of a device result must hold the vectors");
        }
        assert_rc!(turbo_buffer_export(view, 99, &mut native, &mut e), TURBO_E_INVALID_ENUM, e);
        turbo_buffer_release(view);
        turbo_result_release(r);
    }
}

#[test]
fn tasks_a_task_the_model_does_not_offer_is_unsupported_task() {
    let t = Target::from_env();
    needs!(t, Embedding, Generative);
    let chain = Chain::new(&t, BundleKind::Embedding);
    let mut e = c::err();
    let query = c::text("q");
    let docs = [c::text("d")];
    let texts = [c::text("x")];
    // SAFETY: valid session handle; every array outlives its call.
    unsafe {
        assert_rc!(
            turbo_session_write_pairs(chain.session, &query, docs.as_ptr(), 1, ptr::null(), &mut e),
            TURBO_E_UNSUPPORTED_TASK,
            e
        );
        assert_rc!(
            turbo_session_write_text_classify(chain.session, texts.as_ptr(), 1, ptr::null(), &mut e),
            TURBO_E_UNSUPPORTED_TASK,
            e
        );
        // A generative bundle has no session-level task: the session itself
        // is refused, for every provider, before any write.
        let gen_model = chain._ct.model(chain.ctx, BundleKind::Generative);
        let mut gen_session: *mut turbo_session = ptr::null_mut();
        assert_rc!(turbo_session_create(gen_model, ptr::null(), &mut gen_session, &mut e), TURBO_E_UNSUPPORTED_TASK, e);
        assert!(gen_session.is_null(), "a refused session leaves the out pointer NULL");
        // An embedder is not a generative model either.
        let mut g: *mut turbo_generation = ptr::null_mut();
        assert_rc!(turbo_generation_create(chain.model, ptr::null(), &mut g, &mut e), TURBO_E_UNSUPPORTED_TASK, e);
        turbo_model_release(gen_model);
    }
}
