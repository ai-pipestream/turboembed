//! Raw C ABI bindings for `include/turborerank.h`.

#![allow(non_camel_case_types)]

use std::os::raw::c_char;

pub const TURBORERANK_ABI_VERSION: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct turborerank_status(pub u32);

impl turborerank_status {
    pub const TURBORERANK_OK: Self = Self(0);
    pub const TURBORERANK_ERR_INVALID_ARGUMENT: Self = Self(1);
    pub const TURBORERANK_ERR_NOT_FOUND: Self = Self(2);
    pub const TURBORERANK_ERR_NOT_IMPLEMENTED: Self = Self(3);
    pub const TURBORERANK_ERR_UNAVAILABLE: Self = Self(4);
    pub const TURBORERANK_ERR_INTERNAL: Self = Self(5);
    pub const TURBORERANK_ERR_OUT_OF_MEMORY: Self = Self(6);
    pub const TURBORERANK_ERR_UNSUPPORTED_DEVICE: Self = Self(7);
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct turborerank_device(pub u32);

impl turborerank_device {
    pub const TURBORERANK_DEVICE_AUTO: Self = Self(0);
    pub const TURBORERANK_DEVICE_CPU: Self = Self(1);
    pub const TURBORERANK_DEVICE_CUDA: Self = Self(2);
    pub const TURBORERANK_DEVICE_TENSORRT: Self = Self(3);
    pub const TURBORERANK_DEVICE_OPENVINO_CPU: Self = Self(4);
    pub const TURBORERANK_DEVICE_OPENVINO_GPU: Self = Self(5);
    pub const TURBORERANK_DEVICE_OPENVINO_NPU: Self = Self(6);
    pub const TURBORERANK_DEVICE_METAL: Self = Self(7);
    pub const TURBORERANK_DEVICE_MOCK: Self = Self(8);
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct turborerank_truncation(pub u32);

impl turborerank_truncation {
    pub const TURBORERANK_TRUNC_LONGEST_FIRST: Self = Self(0);
    pub const TURBORERANK_TRUNC_QUERY_PRIORITY: Self = Self(1);
    pub const TURBORERANK_TRUNC_ERROR: Self = Self(2);
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct turborerank_activation(pub u32);

impl turborerank_activation {
    pub const TURBORERANK_ACT_SIGMOID: Self = Self(0);
    pub const TURBORERANK_ACT_IDENTITY: Self = Self(1);
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct turborerank_str {
    pub ptr: *const c_char,
    pub len: usize,
}

#[repr(C)]
pub struct turborerank_buffer {
    pub input_ids: *mut i32,
    pub attention_mask: *mut i32,
    pub token_type_ids: *mut i32,
    pub position_ids: *mut i32,
    pub batch: u32,
    pub seq: u32,
    pub row_stride: u32,
    pub device: turborerank_device,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct turborerank_score_options {
    pub truncation: turborerank_truncation,
    pub activation: turborerank_activation,
    pub max_length: u32,
}

#[repr(C)]
pub struct turborerank_model_info {
    pub alias: turborerank_str,
    pub max_length: u32,
    pub hidden_size: u32,
    pub device: turborerank_device,
    pub ready: i32,
}

#[repr(C)]
pub struct turborerank_engine {
    _private: [u8; 0],
}

extern "C" {
    pub fn turborerank_engine_create(
        device: turborerank_device,
        config_path: *const c_char,
        out: *mut *mut turborerank_engine,
    ) -> turborerank_status;

    pub fn turborerank_engine_destroy(engine: *mut turborerank_engine);

    pub fn turborerank_abi_version() -> u32;

    pub fn turborerank_status_name(status: turborerank_status) -> *const c_char;

    pub fn turborerank_device_name(device: turborerank_device) -> *const c_char;

    pub fn turborerank_last_error(engine: *const turborerank_engine) -> *const c_char;

    pub fn turborerank_list_models(
        engine: *mut turborerank_engine,
        out_infos: *mut *mut turborerank_model_info,
        out_count: *mut usize,
    ) -> turborerank_status;

    pub fn turborerank_model_list_free(infos: *mut turborerank_model_info, count: usize);

    pub fn turborerank_load_model(
        engine: *mut turborerank_engine,
        alias: *const c_char,
        alias_len: usize,
    ) -> turborerank_status;

    pub fn turborerank_buffer_alloc(
        device: turborerank_device,
        batch: u32,
        seq: u32,
        out: *mut *mut turborerank_buffer,
    ) -> turborerank_status;

    pub fn turborerank_buffer_free(buffer: *mut turborerank_buffer);

    pub fn turborerank_pack_ids(
        buffer: *mut turborerank_buffer,
        row: u32,
        query_ids: *const i32,
        n_query: usize,
        doc_ids: *const i32,
        n_doc: usize,
        truncation: turborerank_truncation,
        max_length: u32,
    ) -> turborerank_status;

    pub fn turborerank_pack_text(
        engine: *mut turborerank_engine,
        buffer: *mut turborerank_buffer,
        row: u32,
        query: turborerank_str,
        document: turborerank_str,
        truncation: turborerank_truncation,
        max_length: u32,
    ) -> turborerank_status;

    pub fn turborerank_forward(
        engine: *mut turborerank_engine,
        buffer: *const turborerank_buffer,
        n_rows: u32,
        activation: turborerank_activation,
        scores_out: *mut f32,
    ) -> turborerank_status;

    pub fn turborerank_score(
        engine: *mut turborerank_engine,
        alias: *const c_char,
        alias_len: usize,
        query: turborerank_str,
        documents: *const turborerank_str,
        n_documents: usize,
        opts: *const turborerank_score_options,
        scores_out: *mut f32,
    ) -> turborerank_status;
}
