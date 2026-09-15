use std::ffi::CStr;
use std::ptr;

use turboembed::ffi::{
    turboembed_device, turboembed_embed_options, turboembed_embed_result,
    turboembed_embed_result_free, turboembed_engine, turboembed_engine_create,
    turboembed_engine_destroy, turboembed_last_error, turboembed_output_format, turboembed_pooling,
    turboembed_status, turboembed_str,
};

#[repr(C)]
struct RawEmbedOptions {
    pooling: i32,
    normalize: i32,
    truncate_to: u32,
    output_format: i32,
}

unsafe extern "C" {
    #[link_name = "turboembed_embed"]
    fn turboembed_embed_raw(
        engine: *mut turboembed_engine,
        alias: *const std::ffi::c_char,
        alias_len: usize,
        texts: *const turboembed_str,
        n_texts: usize,
        opts: *const RawEmbedOptions,
        out: *mut *mut turboembed_embed_result,
    ) -> turboembed_status;
}

#[test]
fn invalid_raw_option_values_are_rejected() {
    for options in [
        RawEmbedOptions {
            pooling: 99,
            normalize: -1,
            truncate_to: 0,
            output_format: 0,
        },
        RawEmbedOptions {
            pooling: 0,
            normalize: 2,
            truncate_to: 0,
            output_format: 0,
        },
        RawEmbedOptions {
            pooling: 0,
            normalize: -1,
            truncate_to: 0,
            output_format: 99,
        },
    ] {
        let (status, out, message) = unsafe { embed_raw(&options) };
        assert_eq!(status, turboembed_status::TURBOEMBED_ERR_INVALID_ARGUMENT);
        assert!(out.is_null());
        assert!(message.contains("invalid"), "unexpected error: {message}");
    }
}

#[test]
fn mock_rejects_mathematical_options_it_does_not_implement() {
    for options in [
        options(turboembed_pooling::TURBOEMBED_POOLING_MEAN, -1, 0),
        options(turboembed_pooling::TURBOEMBED_POOLING_DEFAULT, 1, 0),
        options(turboembed_pooling::TURBOEMBED_POOLING_DEFAULT, -1, 12),
    ] {
        let (status, out, message) = unsafe { embed(&options) };
        assert_eq!(status, turboembed_status::TURBOEMBED_ERR_NOT_IMPLEMENTED);
        assert!(out.is_null());
        assert!(message.contains("mock path"), "unexpected error: {message}");
    }
}

#[test]
fn mock_accepts_both_output_views_as_aliases() {
    for output_format in [
        turboembed_output_format::TURBOEMBED_OUTPUT_TYPED,
        turboembed_output_format::TURBOEMBED_OUTPUT_PACKED_BYTES,
    ] {
        let options = turboembed_embed_options {
            pooling: turboembed_pooling::TURBOEMBED_POOLING_DEFAULT,
            normalize: -1,
            truncate_to: 0,
            output_format,
        };
        unsafe { assert_successful_aliases(&options) };
    }
}

unsafe fn assert_successful_aliases(options: &turboembed_embed_options) {
    let engine = unsafe { mock_engine() };
    let alias = b"mock-embed";
    let text = b"option validation";
    let view = turboembed_str {
        ptr: text.as_ptr().cast(),
        len: text.len(),
    };
    let mut out = ptr::null_mut();
    let status = unsafe {
        turboembed_embed_raw(
            engine,
            alias.as_ptr().cast(),
            alias.len(),
            &view,
            1,
            (options as *const turboembed_embed_options).cast(),
            &mut out,
        )
    };
    let message = unsafe { CStr::from_ptr(turboembed_last_error(engine)) }.to_string_lossy();
    assert_eq!(status, turboembed_status::TURBOEMBED_OK, "{message}");
    assert!(!out.is_null());
    unsafe {
        assert_eq!((*out).packed.cast::<f32>(), (*out).values);
        assert_eq!(
            (*out).packed_len,
            (*out).count as usize * (*out).dim as usize * 4
        );
        turboembed_embed_result_free(out);
        turboembed_engine_destroy(engine);
    }
}

fn options(
    pooling: turboembed_pooling,
    normalize: i32,
    truncate_to: u32,
) -> turboembed_embed_options {
    turboembed_embed_options {
        pooling,
        normalize,
        truncate_to,
        output_format: turboembed_output_format::TURBOEMBED_OUTPUT_TYPED,
    }
}

unsafe fn embed(
    options: &turboembed_embed_options,
) -> (turboembed_status, *mut turboembed_embed_result, String) {
    unsafe { embed_with(options as *const turboembed_embed_options) }
}

unsafe fn embed_raw(
    options: &RawEmbedOptions,
) -> (turboembed_status, *mut turboembed_embed_result, String) {
    unsafe { embed_with(options) }
}

unsafe fn embed_with<T>(
    options: *const T,
) -> (turboembed_status, *mut turboembed_embed_result, String) {
    let engine = unsafe { mock_engine() };
    let alias = b"mock-embed";
    let text = b"option validation";
    let view = turboembed_str {
        ptr: text.as_ptr().cast(),
        len: text.len(),
    };
    let mut out = ptr::null_mut();
    let status = unsafe {
        turboembed_embed_raw(
            engine,
            alias.as_ptr().cast(),
            alias.len(),
            &view,
            1,
            options.cast(),
            &mut out,
        )
    };
    let message = unsafe { CStr::from_ptr(turboembed_last_error(engine)) }
        .to_string_lossy()
        .into_owned();
    unsafe { turboembed_engine_destroy(engine) };
    (status, out, message)
}

unsafe fn mock_engine() -> *mut turboembed_engine {
    let mut engine = ptr::null_mut();
    let status = unsafe {
        turboembed_engine_create(
            turboembed_device::TURBOEMBED_DEVICE_MOCK,
            ptr::null(),
            &mut engine,
        )
    };
    assert_eq!(status, turboembed_status::TURBOEMBED_OK);
    engine
}
