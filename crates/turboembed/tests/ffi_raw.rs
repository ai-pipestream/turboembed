//! Raw `extern "C"` edge cases for the frozen ABI v1 surface declared in
//! `include/turboembed.h`, called directly through `turboembed::ffi`
//! against the deterministic mock engine. Only `TURBOEMBED_DEVICE_MOCK`
//! and explicit CPU are used; the mock never impersonates a catalog model.
//!
//! Complements `input_bounds.rs` (count/span bounds) and `options.rs`
//! (option validation). This file covers lifecycle NULL semantics, alias
//! length handling, NULL-vs-default option equality, stream equivalence,
//! the name queries, and free/destroy ordering.
//!
//! Known drift between the Linux C++ stub and the Apple Swift dylib, kept
//! Linux-precise and macOS-tolerant below:
//!   * a fresh explicit-CPU engine reports the mock alias `ready = 0` on
//!     the stub, `ready = 1` on the Swift dylib (pre-loaded);
//!   * `load_model` with an empty NUL-terminated alias is
//!     INVALID_ARGUMENT ("alias is empty") on the stub, NOT_IMPLEMENTED on
//!     the Swift dylib;
//!   * the empty-batch error text differs ("texts must not be empty" vs
//!     "null embed argument"); the status code is the same.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::ptr;

use turboembed::ffi::{
    turboembed_buffer_free, turboembed_device, turboembed_device_name, turboembed_embed,
    turboembed_embed_one, turboembed_embed_options, turboembed_embed_result,
    turboembed_embed_result_free, turboembed_embed_stream, turboembed_engine,
    turboembed_engine_create, turboembed_engine_destroy, turboembed_last_error,
    turboembed_list_models, turboembed_load_model, turboembed_model_info,
    turboembed_model_list_free, turboembed_output_format, turboembed_pooling,
    turboembed_register_provider, turboembed_status, turboembed_status_name, turboembed_str,
};

const MOCK_ALIAS: &[u8] = b"mock-embed";
const MOCK_DIM: u32 = 8;

// Raw `i32` entry points so out-of-range enum representations can be
// probed without materializing an invalid Rust enum value. Both ABI
// implementations switch over the int and fall through to a default.
unsafe extern "C" {
    #[link_name = "turboembed_status_name"]
    fn turboembed_status_name_raw(code: i32) -> *const c_char;

    #[link_name = "turboembed_device_name"]
    fn turboembed_device_name_raw(value: i32) -> *const c_char;
}

#[test]
fn create_and_destroy_with_null_config_path() {
    unsafe {
        let engine = mock_engine();
        // Header contract: the pointer is never NULL, even without an engine.
        let message = turboembed_last_error(ptr::null());
        assert!(!message.is_null());
        // A successful create clears the thread-local create error.
        assert_eq!(c_lossy(message), "");
        turboembed_engine_destroy(engine);
        // Documented: destroying NULL is a no-op.
        turboembed_engine_destroy(ptr::null_mut());
    }
}

#[test]
fn create_accepts_nul_terminated_config_path() {
    let config = CString::new("/nonexistent/turboembed-ffi-raw.toml").unwrap();
    unsafe {
        let mut engine = ptr::null_mut();
        let status = turboembed_engine_create(
            turboembed_device::TURBOEMBED_DEVICE_MOCK,
            config.as_ptr(),
            &mut engine,
        );
        assert_eq!(status, turboembed_status::TURBOEMBED_OK);
        assert!(!engine.is_null());
        turboembed_engine_destroy(engine);
    }
}

#[cfg(not(target_os = "macos"))]
#[test]
fn last_error_null_engine_explains_failed_create_then_clears() {
    unsafe {
        let mut engine = ptr::null_mut();
        let status = turboembed_engine_create(
            turboembed_device::TURBOEMBED_DEVICE_METAL,
            ptr::null(),
            &mut engine,
        );
        // No Metal on the Linux stub: create fails, never falls back to CPU.
        assert_eq!(status, turboembed_status::TURBOEMBED_ERR_UNAVAILABLE);
        assert!(engine.is_null());
        let message = c_lossy(turboembed_last_error(ptr::null()));
        assert!(
            message.contains("refusing CPU fallback"),
            "unexpected create error: {message}"
        );

        // A later successful create clears the thread-local message.
        let engine = mock_engine();
        assert_eq!(c_lossy(turboembed_last_error(ptr::null())), "");
        turboembed_engine_destroy(engine);
    }
}

#[test]
fn last_error_engine_view_tracks_most_recent_call() {
    unsafe {
        let engine = loaded_mock_engine();
        // No error yet: non-NULL, empty message.
        assert_eq!(last_error_lossy(engine), "");

        let text = b"one";
        let view = view(text);
        let mut out = ptr::null_mut();
        let status = turboembed_embed(
            engine,
            MOCK_ALIAS.as_ptr().cast(),
            MOCK_ALIAS.len(),
            &view,
            0,
            ptr::null(),
            &mut out,
        );
        assert_eq!(status, turboembed_status::TURBOEMBED_ERR_INVALID_ARGUMENT);
        assert!(!last_error_lossy(engine).is_empty());

        // The next mutating call replaces the message; success resets it.
        let status = turboembed_embed(
            engine,
            MOCK_ALIAS.as_ptr().cast(),
            MOCK_ALIAS.len(),
            &view,
            1,
            ptr::null(),
            &mut out,
        );
        assert_eq!(
            status,
            turboembed_status::TURBOEMBED_OK,
            "{}",
            last_error_lossy(engine)
        );
        assert!(!out.is_null());
        assert_eq!(last_error_lossy(engine), "");
        turboembed_embed_result_free(out);
        turboembed_engine_destroy(engine);
    }
}

#[test]
fn list_models_round_trip_reports_mock_catalog_entry() {
    unsafe {
        let engine = mock_engine();
        let (infos, count) = list_models(engine);
        assert_eq!(count, 1);
        let info = *infos;
        assert_eq!(info.dim, MOCK_DIM);
        assert_eq!(info.device, turboembed_device::TURBOEMBED_DEVICE_MOCK);
        assert_eq!(view_bytes(&info.alias), MOCK_ALIAS);
        // The mock device serves its alias from creation.
        assert_eq!(info.ready, 1);
        turboembed_model_list_free(infos, count);

        // A second round trip proves the first free did not corrupt state.
        let (infos, count) = list_models(engine);
        assert_eq!(count, 1);
        assert_eq!(view_bytes(&(*infos).alias), MOCK_ALIAS);
        turboembed_model_list_free(infos, count);
        turboembed_engine_destroy(engine);
    }
}

#[test]
fn cpu_engine_loads_mock_and_reports_ready() {
    unsafe {
        let engine = cpu_engine();
        let status = turboembed_load_model(engine, MOCK_ALIAS.as_ptr().cast(), MOCK_ALIAS.len());
        assert_eq!(
            status,
            turboembed_status::TURBOEMBED_OK,
            "{}",
            last_error_lossy(engine)
        );

        let (infos, count) = list_models(engine);
        assert_eq!(count, 1);
        let info = *infos;
        assert_eq!(info.dim, MOCK_DIM);
        assert_eq!(info.device, turboembed_device::TURBOEMBED_DEVICE_MOCK);
        assert_eq!(view_bytes(&info.alias), MOCK_ALIAS);
        assert_eq!(info.ready, 1, "explicit CPU serves the mock once loaded");
        turboembed_model_list_free(infos, count);

        let text = b"explicit cpu row";
        let result = embed_one_text(engine, text, ptr::null());
        assert_eq!(snapshot(result).0, MOCK_DIM);
        turboembed_embed_result_free(result);
        turboembed_engine_destroy(engine);
    }
}

#[cfg(not(target_os = "macos"))]
#[test]
fn cpu_engine_starts_with_mock_not_loaded() {
    // Stub contract: explicit CPU does not serve the mock until
    // turboembed_load_model runs. The Swift dylib pre-loads the mock on
    // explicit CPU, so this asserts the stub's stricter contract only.
    unsafe {
        let engine = cpu_engine();
        let (infos, count) = list_models(engine);
        assert_eq!(count, 1);
        assert_eq!((*infos).ready, 0);
        turboembed_model_list_free(infos, count);

        let text = b"too early";
        let view = view(text);
        let mut out = ptr::null_mut();
        let status = turboembed_embed(
            engine,
            MOCK_ALIAS.as_ptr().cast(),
            MOCK_ALIAS.len(),
            &view,
            1,
            ptr::null(),
            &mut out,
        );
        assert_eq!(status, turboembed_status::TURBOEMBED_ERR_NOT_FOUND);
        assert!(out.is_null());
        assert!(
            last_error_lossy(engine).contains("not loaded"),
            "unexpected error: {}",
            last_error_lossy(engine)
        );
        turboembed_engine_destroy(engine);
    }
}

#[test]
fn load_model_accepts_nul_terminated_and_explicit_length_aliases() {
    unsafe {
        let engine = mock_engine();

        // alias_len == 0 means the alias is a NUL-terminated C string.
        let nul_terminated = b"mock-embed\0";
        let status = turboembed_load_model(engine, nul_terminated.as_ptr().cast(), 0);
        assert_eq!(
            status,
            turboembed_status::TURBOEMBED_OK,
            "{}",
            last_error_lossy(engine)
        );

        // The short alias works the same way.
        let short = b"mock\0";
        let status = turboembed_load_model(engine, short.as_ptr().cast(), 0);
        assert_eq!(
            status,
            turboembed_status::TURBOEMBED_OK,
            "{}",
            last_error_lossy(engine)
        );

        // An explicit length must not read a terminator: the same alias
        // backed by exactly `len` bytes (no NUL) still loads.
        let exact = b"mock-embed";
        let status = turboembed_load_model(engine, exact.as_ptr().cast(), exact.len());
        assert_eq!(
            status,
            turboembed_status::TURBOEMBED_OK,
            "{}",
            last_error_lossy(engine)
        );

        #[cfg(not(target_os = "macos"))]
        {
            // The degenerate NUL-terminated alias is rejected, not strlen-probed
            // into whatever happens to follow in memory.
            let empty = b"\0";
            let status = turboembed_load_model(engine, empty.as_ptr().cast(), 0);
            assert_eq!(status, turboembed_status::TURBOEMBED_ERR_INVALID_ARGUMENT);
            assert!(
                last_error_lossy(engine).contains("alias is empty"),
                "unexpected error: {}",
                last_error_lossy(engine)
            );
        }
        turboembed_engine_destroy(engine);
    }
}

#[test]
fn embed_rejects_empty_batch() {
    unsafe {
        let engine = loaded_mock_engine();
        let text = b"nothing to embed";
        let view = view(text);
        let mut out = ptr::null_mut();
        let status = turboembed_embed(
            engine,
            MOCK_ALIAS.as_ptr().cast(),
            MOCK_ALIAS.len(),
            &view,
            0,
            ptr::null(),
            &mut out,
        );
        assert_eq!(status, turboembed_status::TURBOEMBED_ERR_INVALID_ARGUMENT);
        assert!(out.is_null());
        #[cfg(not(target_os = "macos"))]
        assert!(
            last_error_lossy(engine).contains("empty"),
            "unexpected error: {}",
            last_error_lossy(engine)
        );
        turboembed_engine_destroy(engine);
    }
}

#[test]
fn embed_null_opts_match_explicit_default_opts() {
    let default_opts = turboembed_embed_options {
        pooling: turboembed_pooling::TURBOEMBED_POOLING_DEFAULT,
        normalize: -1,
        truncate_to: 0,
        output_format: turboembed_output_format::TURBOEMBED_OUTPUT_TYPED,
    };
    unsafe {
        let engine = loaded_mock_engine();
        let texts: [&[u8]; 3] = [b"alpha", b"beta", b"alpha"];
        let views = views(&texts);

        let with_null = embed_views(engine, &views, ptr::null());
        let with_defaults = embed_views(engine, &views, &default_opts);
        assert_eq!(snapshot(with_null), snapshot(with_defaults));
        turboembed_embed_result_free(with_null);
        turboembed_embed_result_free(with_defaults);
        turboembed_engine_destroy(engine);
    }
}

#[test]
fn embed_one_matches_single_text_batch() {
    unsafe {
        let engine = loaded_mock_engine();
        let text = b"one text, two paths";
        let batch_views = views(&[text.as_slice()]);

        let one = embed_one_text(engine, text, ptr::null());
        let batch = embed_views(engine, &batch_views, ptr::null());
        let (dim, count, values) = snapshot(one);
        assert_eq!(count, 1);
        assert_eq!(dim, MOCK_DIM);
        assert_eq!(values.len(), MOCK_DIM as usize);
        assert_eq!((dim, count, values), snapshot(batch));
        turboembed_embed_result_free(one);
        turboembed_embed_result_free(batch);
        turboembed_engine_destroy(engine);
    }
}

#[test]
fn stream_with_null_callback_matches_embed() {
    unsafe {
        let engine = loaded_mock_engine();
        let texts: [&[u8]; 2] = [b"stream me", b"second row"];
        let views = views(&texts);

        let mut plain = ptr::null_mut();
        let status = turboembed_embed(
            engine,
            MOCK_ALIAS.as_ptr().cast(),
            MOCK_ALIAS.len(),
            views.as_ptr(),
            views.len(),
            ptr::null(),
            &mut plain,
        );
        assert_eq!(
            status,
            turboembed_status::TURBOEMBED_OK,
            "{}",
            last_error_lossy(engine)
        );

        let mut streamed = ptr::null_mut();
        let status = turboembed_embed_stream(
            engine,
            MOCK_ALIAS.as_ptr().cast(),
            MOCK_ALIAS.len(),
            views.as_ptr(),
            views.len(),
            ptr::null(),
            None,
            ptr::null_mut(),
            &mut streamed,
        );
        assert_eq!(
            status,
            turboembed_status::TURBOEMBED_OK,
            "{}",
            last_error_lossy(engine)
        );
        assert!(!streamed.is_null());
        assert_eq!(snapshot(plain), snapshot(streamed));
        turboembed_embed_result_free(plain);
        turboembed_embed_result_free(streamed);
        turboembed_engine_destroy(engine);
    }
}

struct RowCapture {
    rows: Vec<(u32, bool, Vec<f32>)>,
}

unsafe extern "C" fn capture_row(
    user_data: *mut c_void,
    index: u32,
    values: *const f32,
    dim: u32,
    is_final: i32,
) {
    unsafe {
        let capture = &mut *(user_data as *mut RowCapture);
        let row = std::slice::from_raw_parts(values, dim as usize).to_vec();
        capture.rows.push((index, is_final != 0, row));
    }
}

#[test]
fn stream_callback_reports_each_row_once_in_order() {
    unsafe {
        let engine = loaded_mock_engine();
        let texts: [&[u8]; 3] = [b"row zero", b"row one", b"row two"];
        let views = views(&texts);
        let mut capture = RowCapture { rows: Vec::new() };

        let mut out = ptr::null_mut();
        let status = turboembed_embed_stream(
            engine,
            MOCK_ALIAS.as_ptr().cast(),
            MOCK_ALIAS.len(),
            views.as_ptr(),
            views.len(),
            ptr::null(),
            Some(capture_row),
            (&mut capture as *mut RowCapture).cast(),
            &mut out,
        );
        assert_eq!(
            status,
            turboembed_status::TURBOEMBED_OK,
            "{}",
            last_error_lossy(engine)
        );
        assert!(!out.is_null());

        let (dim, count, values) = snapshot(out);
        assert_eq!(count, 3);
        assert_eq!(capture.rows.len(), count as usize);
        for (i, (index, is_final, row)) in capture.rows.iter().enumerate() {
            assert_eq!(*index, i as u32);
            assert_eq!(*is_final, i + 1 == count as usize, "is_final on row {i}");
            let start = i * dim as usize;
            assert_eq!(row, &values[start..start + dim as usize]);
        }
        turboembed_embed_result_free(out);
        turboembed_engine_destroy(engine);
    }
}

#[test]
fn stream_with_null_out_invokes_callback_then_releases_result() {
    // Current raw-surface contract: with `out == NULL` the engine still
    // invokes the callback once per row, releases the batch result itself,
    // and returns OK. No result is handed to the caller.
    unsafe {
        let engine = loaded_mock_engine();
        let texts: [&[u8]; 2] = [b"a", b"b"];
        let views = views(&texts);
        let mut capture = RowCapture { rows: Vec::new() };

        let status = turboembed_embed_stream(
            engine,
            MOCK_ALIAS.as_ptr().cast(),
            MOCK_ALIAS.len(),
            views.as_ptr(),
            views.len(),
            ptr::null(),
            Some(capture_row),
            (&mut capture as *mut RowCapture).cast(),
            ptr::null_mut(),
        );
        assert_eq!(
            status,
            turboembed_status::TURBOEMBED_OK,
            "{}",
            last_error_lossy(engine)
        );
        assert_eq!(capture.rows.len(), 2);
        assert!(!capture.rows[0].1, "only the last row is final");
        assert!(capture.rows[1].1, "last row is final");
        turboembed_engine_destroy(engine);
    }
}

#[test]
fn status_name_covers_all_abi_codes() {
    let expected = [
        (0, "OK"),
        (1, "INVALID_ARGUMENT"),
        (2, "NOT_FOUND"),
        (3, "NOT_IMPLEMENTED"),
        (4, "UNAVAILABLE"),
        (5, "INTERNAL"),
        (6, "OUT_OF_MEMORY"),
        (7, "UNSUPPORTED_DEVICE"),
    ];
    for (code, name) in expected {
        let status: turboembed_status = unsafe { std::mem::transmute(code) };
        let ptr = unsafe { turboembed_status_name(status) };
        assert!(!ptr.is_null(), "status_name({code}) returned NULL");
        assert_eq!(unsafe { c_lossy(ptr) }, name, "status code {code}");
    }

    // Out-of-range codes hit the C switch default; both ABI implementations
    // return a stable "UNKNOWN" placeholder instead of crashing.
    let ptr = unsafe { turboembed_status_name_raw(99) };
    assert!(!ptr.is_null());
    assert_eq!(unsafe { c_lossy(ptr) }, "UNKNOWN");
}

#[test]
fn device_name_covers_all_values() {
    let expected = [
        (0, "auto"),
        (1, "cpu"),
        (2, "cuda"),
        (3, "tensorrt"),
        (4, "openvino-cpu"),
        (5, "openvino-gpu"),
        (6, "openvino-npu"),
        (7, "metal"),
        (8, "mock"),
        (9, "hailo"),
    ];
    for (value, name) in expected {
        let device: turboembed_device = unsafe { std::mem::transmute(value) };
        let ptr = unsafe { turboembed_device_name(device) };
        assert!(!ptr.is_null(), "device_name({value}) returned NULL");
        assert_eq!(unsafe { c_lossy(ptr) }, name, "device value {value}");
    }

    let ptr = unsafe { turboembed_device_name_raw(99) };
    assert!(!ptr.is_null());
    assert_eq!(unsafe { c_lossy(ptr) }, "unknown");
}

#[test]
fn register_provider_null_vtbl_stays_not_implemented() {
    unsafe {
        let status = turboembed_register_provider(ptr::null());
        assert_eq!(status, turboembed_status::TURBOEMBED_ERR_NOT_IMPLEMENTED);
        // Both implementations record the reservation message where
        // last_error(NULL) can see it.
        let message = c_lossy(turboembed_last_error(ptr::null()));
        assert!(
            message.contains("reserved"),
            "unexpected reservation message: {message}"
        );
    }
}

#[test]
fn embed_result_free_before_engine_destroy_is_safe() {
    unsafe {
        let engine = loaded_mock_engine();
        let text = b"own me first";
        let result = embed_one_text(engine, text, ptr::null());

        // Copy the payload, release the result, then the engine: the
        // documented free-before-destroy order must not crash.
        let (dim, count, values) = snapshot(result);
        turboembed_embed_result_free(result);
        turboembed_engine_destroy(engine);

        assert_eq!(dim, MOCK_DIM);
        assert_eq!(count, 1);
        assert_eq!(values.len(), MOCK_DIM as usize);
        assert!(values.iter().any(|v| *v != 0.0));
    }
}

#[test]
fn buffer_free_null_is_documented_noop() {
    // turboembed_buffer_free: "NULL is a no-op."
    unsafe { turboembed_buffer_free(ptr::null_mut()) };
}

fn view(text: &[u8]) -> turboembed_str {
    turboembed_str {
        ptr: text.as_ptr().cast(),
        len: text.len(),
    }
}

fn views(texts: &[&[u8]]) -> Vec<turboembed_str> {
    texts.iter().copied().map(view).collect()
}

unsafe fn view_bytes(view: &turboembed_str) -> &[u8] {
    std::slice::from_raw_parts(view.ptr.cast::<u8>(), view.len)
}

unsafe fn c_lossy(ptr: *const c_char) -> String {
    assert!(!ptr.is_null(), "ABI returned a NULL string pointer");
    CStr::from_ptr(ptr).to_string_lossy().into_owned()
}

unsafe fn mock_engine() -> *mut turboembed_engine {
    create_engine(turboembed_device::TURBOEMBED_DEVICE_MOCK)
}

unsafe fn loaded_mock_engine() -> *mut turboembed_engine {
    let engine = mock_engine();
    let status = turboembed_load_model(engine, MOCK_ALIAS.as_ptr().cast(), MOCK_ALIAS.len());
    assert_eq!(
        status,
        turboembed_status::TURBOEMBED_OK,
        "{}",
        last_error_lossy(engine)
    );
    engine
}

unsafe fn cpu_engine() -> *mut turboembed_engine {
    create_engine(turboembed_device::TURBOEMBED_DEVICE_CPU)
}

unsafe fn create_engine(device: turboembed_device) -> *mut turboembed_engine {
    let mut engine = ptr::null_mut();
    let status = turboembed_engine_create(device, ptr::null(), &mut engine);
    assert_eq!(status, turboembed_status::TURBOEMBED_OK);
    assert!(!engine.is_null());
    engine
}

unsafe fn last_error_lossy(engine: *mut turboembed_engine) -> String {
    c_lossy(turboembed_last_error(engine))
}

unsafe fn embed_views(
    engine: *mut turboembed_engine,
    texts: &[turboembed_str],
    opts: *const turboembed_embed_options,
) -> *mut turboembed_embed_result {
    let mut out = ptr::null_mut();
    let status = turboembed_embed(
        engine,
        MOCK_ALIAS.as_ptr().cast(),
        MOCK_ALIAS.len(),
        texts.as_ptr(),
        texts.len(),
        opts,
        &mut out,
    );
    assert_eq!(
        status,
        turboembed_status::TURBOEMBED_OK,
        "{}",
        last_error_lossy(engine)
    );
    assert!(!out.is_null());
    out
}

unsafe fn embed_one_text(
    engine: *mut turboembed_engine,
    text: &[u8],
    opts: *const turboembed_embed_options,
) -> *mut turboembed_embed_result {
    let mut out = ptr::null_mut();
    let status = turboembed_embed_one(
        engine,
        MOCK_ALIAS.as_ptr().cast(),
        MOCK_ALIAS.len(),
        text.as_ptr().cast(),
        text.len(),
        opts,
        &mut out,
    );
    assert_eq!(
        status,
        turboembed_status::TURBOEMBED_OK,
        "{}",
        last_error_lossy(engine)
    );
    assert!(!out.is_null());
    out
}

unsafe fn snapshot(result: *const turboembed_embed_result) -> (u32, u32, Vec<f32>) {
    assert!(!result.is_null());
    let dim = (*result).dim;
    let count = (*result).count;
    let values = std::slice::from_raw_parts((*result).values, dim as usize * count as usize);
    (dim, count, values.to_vec())
}

unsafe fn list_models(engine: *mut turboembed_engine) -> (*mut turboembed_model_info, usize) {
    let mut infos = ptr::null_mut();
    let mut count = 0usize;
    let status = turboembed_list_models(engine, &mut infos, &mut count);
    assert_eq!(
        status,
        turboembed_status::TURBOEMBED_OK,
        "{}",
        last_error_lossy(engine)
    );
    assert!(!infos.is_null());
    (infos, count)
}
