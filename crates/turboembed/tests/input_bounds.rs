#![cfg(target_pointer_width = "64")]

use std::ffi::CStr;
use std::ptr;

use turboembed::ffi::{
    turboembed_device, turboembed_embed, turboembed_embed_result, turboembed_embed_stream,
    turboembed_engine, turboembed_engine_create, turboembed_engine_destroy, turboembed_last_error,
    turboembed_status, turboembed_str,
};

#[test]
fn embed_rejects_unrepresentable_count_before_reading_views() {
    unsafe {
        let engine = mock_engine();
        let alias = b"mock-embed";
        let text = b"only one view exists";
        let view = turboembed_str {
            ptr: text.as_ptr().cast(),
            len: text.len(),
        };
        let mut out: *mut turboembed_embed_result = ptr::null_mut();

        let status = turboembed_embed(
            engine,
            alias.as_ptr().cast(),
            alias.len(),
            &view,
            u32::MAX as usize + 1,
            ptr::null(),
            &mut out,
        );

        assert_eq!(status, turboembed_status::TURBOEMBED_ERR_INVALID_ARGUMENT);
        assert!(out.is_null());
        assert!(CStr::from_ptr(turboembed_last_error(engine))
            .to_string_lossy()
            .contains("count"));
        turboembed_engine_destroy(engine);
    }
}

#[test]
fn stream_rejects_unrepresentable_count_before_reading_views() {
    unsafe {
        let engine = mock_engine();
        let alias = b"mock-embed";
        let view = turboembed_str {
            ptr: ptr::null(),
            len: 0,
        };
        let mut out: *mut turboembed_embed_result = ptr::null_mut();

        let status = turboembed_embed_stream(
            engine,
            alias.as_ptr().cast(),
            alias.len(),
            &view,
            u32::MAX as usize + 1,
            ptr::null(),
            None,
            ptr::null_mut(),
            &mut out,
        );

        assert_eq!(status, turboembed_status::TURBOEMBED_ERR_INVALID_ARGUMENT);
        assert!(out.is_null());
        assert!(CStr::from_ptr(turboembed_last_error(engine))
            .to_string_lossy()
            .contains("count"));
        turboembed_engine_destroy(engine);
    }
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
    assert!(!engine.is_null());
    engine
}
