//! Raw `extern "C"` surface matching `include/turboembed.h`.
//!
//! Keep this 1:1 with the header. Safe wrappers live in `lib.rs`.

#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_void};

pub const TURBOEMBED_ABI_VERSION: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum turboembed_status {
    TURBOEMBED_OK = 0,
    TURBOEMBED_ERR_INVALID_ARGUMENT = 1,
    TURBOEMBED_ERR_NOT_FOUND = 2,
    TURBOEMBED_ERR_NOT_IMPLEMENTED = 3,
    TURBOEMBED_ERR_UNAVAILABLE = 4,
    TURBOEMBED_ERR_INTERNAL = 5,
    TURBOEMBED_ERR_OUT_OF_MEMORY = 6,
    TURBOEMBED_ERR_UNSUPPORTED_DEVICE = 7,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum turboembed_device {
    TURBOEMBED_DEVICE_AUTO = 0,
    TURBOEMBED_DEVICE_CPU = 1,
    TURBOEMBED_DEVICE_CUDA = 2,
    TURBOEMBED_DEVICE_TENSORRT = 3,
    TURBOEMBED_DEVICE_OPENVINO_CPU = 4,
    TURBOEMBED_DEVICE_OPENVINO_GPU = 5,
    TURBOEMBED_DEVICE_OPENVINO_NPU = 6,
    TURBOEMBED_DEVICE_METAL = 7,
    TURBOEMBED_DEVICE_MOCK = 8,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum turboembed_pooling {
    TURBOEMBED_POOLING_DEFAULT = 0,
    TURBOEMBED_POOLING_MEAN = 1,
    TURBOEMBED_POOLING_CLS = 2,
    TURBOEMBED_POOLING_LAST = 3,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum turboembed_output_format {
    TURBOEMBED_OUTPUT_TYPED = 0,
    TURBOEMBED_OUTPUT_PACKED_BYTES = 1,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct turboembed_str {
    pub ptr: *const c_char,
    pub len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct turboembed_embed_options {
    pub pooling: turboembed_pooling,
    pub normalize: i32,
    pub truncate_to: u32,
    pub output_format: turboembed_output_format,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct turboembed_model_info {
    pub alias: turboembed_str,
    pub dim: u32,
    pub device: turboembed_device,
    pub ready: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct turboembed_embed_result {
    pub dim: u32,
    pub count: u32,
    pub values: *const f32,
    pub packed: *const u8,
    pub packed_len: usize,
}

#[repr(C)]
pub struct turboembed_engine {
    _opaque: [u8; 0],
}

pub type turboembed_stream_cb = Option<
    unsafe extern "C" fn(
        user_data: *mut c_void,
        index: u32,
        values: *const f32,
        dim: u32,
        is_final: i32,
    ),
>;

#[repr(C)]
pub struct turboembed_provider_vtbl {
    pub id: *const c_char,
    pub load: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            alias: *const c_char,
            alias_len: usize,
        ) -> turboembed_status,
    >,
    pub embed: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            texts: *const turboembed_str,
            n_texts: usize,
            opts: *const turboembed_embed_options,
            out: *mut *mut turboembed_embed_result,
        ) -> turboembed_status,
    >,
    pub ctx: *mut c_void,
}

unsafe extern "C" {
    pub fn turboembed_engine_create(
        device: turboembed_device,
        config_path: *const c_char,
        out: *mut *mut turboembed_engine,
    ) -> turboembed_status;

    pub fn turboembed_engine_destroy(engine: *mut turboembed_engine);

    pub fn turboembed_abi_version() -> u32;

    pub fn turboembed_status_name(status: turboembed_status) -> *const c_char;

    pub fn turboembed_device_name(device: turboembed_device) -> *const c_char;

    pub fn turboembed_last_error(engine: *const turboembed_engine) -> *const c_char;

    pub fn turboembed_list_models(
        engine: *mut turboembed_engine,
        out_infos: *mut *mut turboembed_model_info,
        out_count: *mut usize,
    ) -> turboembed_status;

    pub fn turboembed_model_list_free(infos: *mut turboembed_model_info, count: usize);

    pub fn turboembed_load_model(
        engine: *mut turboembed_engine,
        alias: *const c_char,
        alias_len: usize,
    ) -> turboembed_status;

    pub fn turboembed_embed_one(
        engine: *mut turboembed_engine,
        alias: *const c_char,
        alias_len: usize,
        text: *const c_char,
        text_len: usize,
        opts: *const turboembed_embed_options,
        out: *mut *mut turboembed_embed_result,
    ) -> turboembed_status;

    pub fn turboembed_embed(
        engine: *mut turboembed_engine,
        alias: *const c_char,
        alias_len: usize,
        texts: *const turboembed_str,
        n_texts: usize,
        opts: *const turboembed_embed_options,
        out: *mut *mut turboembed_embed_result,
    ) -> turboembed_status;

    pub fn turboembed_embed_stream(
        engine: *mut turboembed_engine,
        alias: *const c_char,
        alias_len: usize,
        texts: *const turboembed_str,
        n_texts: usize,
        opts: *const turboembed_embed_options,
        cb: turboembed_stream_cb,
        user_data: *mut c_void,
        out: *mut *mut turboembed_embed_result,
    ) -> turboembed_status;

    pub fn turboembed_embed_result_free(result: *mut turboembed_embed_result);

    pub fn turboembed_buffer_free(ptr: *mut c_void);

    pub fn turboembed_register_provider(vtbl: *const turboembed_provider_vtbl)
        -> turboembed_status;
}
