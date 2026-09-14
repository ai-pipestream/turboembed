//! Raw declarations for the additive TurboEmbed prepared execution ABI v1.

#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_float};

pub const TE_PREPARED_VERSION: u32 = 1;
pub const TE_OK: u32 = 0;
pub const TE_INVALID_ARGUMENT: u32 = 1;
pub const TE_NOT_FOUND: u32 = 2;
pub const TE_NOT_IMPLEMENTED: u32 = 3;
pub const TE_UNAVAILABLE: u32 = 4;
pub const TE_INTERNAL: u32 = 5;
pub const TE_OUT_OF_MEMORY: u32 = 6;
pub const TE_BUSY: u32 = 7;
pub const TE_ABI_MISMATCH: u32 = 8;
pub const TE_INTEGRITY_ERROR: u32 = 9;

pub const TE_DEVICE_AUTO: u32 = 0;
pub const TE_DEVICE_OPENVINO_GPU: u32 = 1;
pub const TE_DEVICE_OPENVINO_CPU: u32 = 2;

pub const TE_CAP_TEXT: u64 = 1;
pub const TE_CAP_PREPARED_I32: u64 = 2;
pub const TE_CAP_OPENCL_RESULT: u64 = 4;
pub const TE_CAP_HOST_READ: u64 = 8;

#[repr(C)]
pub struct te_context {
    _private: [u8; 0],
}

#[repr(C)]
pub struct te_model {
    _private: [u8; 0],
}

#[repr(C)]
pub struct te_slot {
    _private: [u8; 0],
}

#[repr(C)]
pub struct te_result {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct te_error {
    pub code: u32,
    pub message: [c_char; 508],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct te_context_options {
    pub struct_size: u32,
    pub version: u32,
    pub device: u32,
    pub ordinal: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct te_context_info {
    pub struct_size: u32,
    pub version: u32,
    pub device: u32,
    pub ordinal: u32,
    pub capabilities: u64,
    pub device_name: [c_char; 128],
    pub runtime_version: [c_char; 128],
    pub driver_version: [c_char; 128],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct te_model_info {
    pub struct_size: u32,
    pub version: u32,
    pub dimension: u32,
    pub vocab_size: u32,
    pub max_sequence_length: u32,
    pub max_batch_size: u32,
    pub normalized: u32,
    pub reserved: u32,
    pub model_id: [c_char; 128],
    pub revision: [c_char; 64],
    pub tokenizer_sha256: [c_char; 65],
    pub pooling: [c_char; 15],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct te_slot_options {
    pub struct_size: u32,
    pub version: u32,
    pub batch: u32,
    pub sequence_length: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct te_result_info {
    pub struct_size: u32,
    pub version: u32,
    pub batch: u32,
    pub dimension: u32,
    pub byte_size: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct te_opencl_view {
    pub struct_size: u32,
    pub version: u32,
    pub context: usize,
    pub queue: usize,
    pub buffer: usize,
    pub byte_size: u64,
    pub batch: u32,
    pub dimension: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct te_slot_stats {
    pub struct_size: u32,
    pub version: u32,
    pub executions: u64,
    pub input_write_bytes: u64,
    pub output_read_bytes: u64,
    pub owned_input_bytes: u64,
    pub owned_output_bytes: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct te_text {
    pub ptr: *const c_char,
    pub byte_length: u64,
}

unsafe extern "C" {
    pub fn turboembed_prepared_v1_version() -> u32;
    pub fn turboembed_prepared_v1_context_create(
        options: *const te_context_options,
        out: *mut *mut te_context,
        error: *mut te_error,
    ) -> u32;
    pub fn turboembed_prepared_v1_context_info(
        context: *const te_context,
        out: *mut te_context_info,
        error: *mut te_error,
    ) -> u32;
    pub fn turboembed_prepared_v1_context_release(context: *mut te_context);
    pub fn turboembed_prepared_v1_model_load(
        context: *const te_context,
        bundle_path: *const c_char,
        path_length: u64,
        out: *mut *mut te_model,
        error: *mut te_error,
    ) -> u32;
    pub fn turboembed_prepared_v1_model_info(
        model: *const te_model,
        out: *mut te_model_info,
        error: *mut te_error,
    ) -> u32;
    pub fn turboembed_prepared_v1_model_release(model: *mut te_model);
    pub fn turboembed_prepared_v1_slot_create(
        model: *const te_model,
        options: *const te_slot_options,
        out: *mut *mut te_slot,
        error: *mut te_error,
    ) -> u32;
    pub fn turboembed_prepared_v1_slot_release(slot: *mut te_slot);
    pub fn turboembed_prepared_v1_slot_write_tokens(
        slot: *mut te_slot,
        ids: *const i32,
        mask: *const i32,
        types: *const i32,
        count: u64,
        error: *mut te_error,
    ) -> u32;
    pub fn turboembed_prepared_v1_slot_write_text(
        slot: *mut te_slot,
        texts: *const te_text,
        count: u64,
        error: *mut te_error,
    ) -> u32;
    pub fn turboembed_prepared_v1_slot_execute(
        slot: *mut te_slot,
        out: *mut *mut te_result,
        error: *mut te_error,
    ) -> u32;
    pub fn turboembed_prepared_v1_slot_stats(
        slot: *const te_slot,
        out: *mut te_slot_stats,
        error: *mut te_error,
    ) -> u32;
    pub fn turboembed_prepared_v1_result_info(
        result: *const te_result,
        out: *mut te_result_info,
        error: *mut te_error,
    ) -> u32;
    pub fn turboembed_prepared_v1_result_read(
        result: *const te_result,
        out: *mut c_float,
        capacity: u64,
        error: *mut te_error,
    ) -> u32;
    pub fn turboembed_prepared_v1_result_opencl(
        result: *const te_result,
        out: *mut te_opencl_view,
        error: *mut te_error,
    ) -> u32;
    pub fn turboembed_prepared_v1_result_release(
        result: *mut te_result,
        error: *mut te_error,
    ) -> u32;
}

#[cfg(all(test, target_os = "linux", target_arch = "x86_64"))]
mod layout_tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};

    #[test]
    fn prepared_v1_layout_matches_c_header() {
        assert_eq!((size_of::<te_error>(), align_of::<te_error>()), (512, 4));
        assert_eq!(offset_of!(te_error, code), 0);
        assert_eq!(offset_of!(te_error, message), 4);
        assert_eq!(
            (
                size_of::<te_context_options>(),
                align_of::<te_context_options>()
            ),
            (16, 4)
        );
        assert_eq!(offset_of!(te_context_options, struct_size), 0);
        assert_eq!(offset_of!(te_context_options, version), 4);
        assert_eq!(offset_of!(te_context_options, device), 8);
        assert_eq!(offset_of!(te_context_options, ordinal), 12);
        assert_eq!(
            (size_of::<te_context_info>(), align_of::<te_context_info>()),
            (408, 8)
        );
        assert_eq!(offset_of!(te_context_info, struct_size), 0);
        assert_eq!(offset_of!(te_context_info, version), 4);
        assert_eq!(offset_of!(te_context_info, device), 8);
        assert_eq!(offset_of!(te_context_info, ordinal), 12);
        assert_eq!(offset_of!(te_context_info, capabilities), 16);
        assert_eq!(offset_of!(te_context_info, device_name), 24);
        assert_eq!(offset_of!(te_context_info, runtime_version), 152);
        assert_eq!(offset_of!(te_context_info, driver_version), 280);
        assert_eq!(
            (size_of::<te_model_info>(), align_of::<te_model_info>()),
            (304, 4)
        );
        assert_eq!(offset_of!(te_model_info, struct_size), 0);
        assert_eq!(offset_of!(te_model_info, version), 4);
        assert_eq!(offset_of!(te_model_info, dimension), 8);
        assert_eq!(offset_of!(te_model_info, vocab_size), 12);
        assert_eq!(offset_of!(te_model_info, max_sequence_length), 16);
        assert_eq!(offset_of!(te_model_info, max_batch_size), 20);
        assert_eq!(offset_of!(te_model_info, normalized), 24);
        assert_eq!(offset_of!(te_model_info, reserved), 28);
        assert_eq!(offset_of!(te_model_info, model_id), 32);
        assert_eq!(offset_of!(te_model_info, revision), 160);
        assert_eq!(offset_of!(te_model_info, tokenizer_sha256), 224);
        assert_eq!(offset_of!(te_model_info, pooling), 289);
        assert_eq!(
            (size_of::<te_slot_options>(), align_of::<te_slot_options>()),
            (16, 4)
        );
        assert_eq!(offset_of!(te_slot_options, struct_size), 0);
        assert_eq!(offset_of!(te_slot_options, version), 4);
        assert_eq!(offset_of!(te_slot_options, batch), 8);
        assert_eq!(offset_of!(te_slot_options, sequence_length), 12);
        assert_eq!(
            (size_of::<te_result_info>(), align_of::<te_result_info>()),
            (24, 8)
        );
        assert_eq!(offset_of!(te_result_info, struct_size), 0);
        assert_eq!(offset_of!(te_result_info, version), 4);
        assert_eq!(offset_of!(te_result_info, batch), 8);
        assert_eq!(offset_of!(te_result_info, dimension), 12);
        assert_eq!(offset_of!(te_result_info, byte_size), 16);
        assert_eq!(
            (size_of::<te_opencl_view>(), align_of::<te_opencl_view>()),
            (48, 8)
        );
        assert_eq!(offset_of!(te_opencl_view, struct_size), 0);
        assert_eq!(offset_of!(te_opencl_view, version), 4);
        assert_eq!(offset_of!(te_opencl_view, context), 8);
        assert_eq!(offset_of!(te_opencl_view, queue), 16);
        assert_eq!(offset_of!(te_opencl_view, buffer), 24);
        assert_eq!(offset_of!(te_opencl_view, byte_size), 32);
        assert_eq!(offset_of!(te_opencl_view, batch), 40);
        assert_eq!(offset_of!(te_opencl_view, dimension), 44);
        assert_eq!(
            (size_of::<te_slot_stats>(), align_of::<te_slot_stats>()),
            (48, 8)
        );
        assert_eq!(offset_of!(te_slot_stats, struct_size), 0);
        assert_eq!(offset_of!(te_slot_stats, version), 4);
        assert_eq!(offset_of!(te_slot_stats, executions), 8);
        assert_eq!(offset_of!(te_slot_stats, input_write_bytes), 16);
        assert_eq!(offset_of!(te_slot_stats, output_read_bytes), 24);
        assert_eq!(offset_of!(te_slot_stats, owned_input_bytes), 32);
        assert_eq!(offset_of!(te_slot_stats, owned_output_bytes), 40);
        assert_eq!((size_of::<te_text>(), align_of::<te_text>()), (16, 8));
        assert_eq!(offset_of!(te_text, ptr), 0);
        assert_eq!(offset_of!(te_text, byte_length), 8);
    }
}
