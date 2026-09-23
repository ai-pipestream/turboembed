//! Provider plugin ABI: the vtable a provider library exports.
//!
//! A provider library exports one symbol, `turbo_provider_get`, with the
//! signature [`turbo_provider_get_fn`]. The core calls it with its own ABI
//! version; the provider returns a pointer to a static [`turbo_provider_vtbl`]
//! or NULL if it cannot serve that version. Every entry point takes the
//! provider's opaque handles (`void *`) and the shared descriptor structs from
//! the public ABI, and returns a status code with the caller-owned
//! `turbo_error`.
//!
//! Ownership at this boundary:
//! - Handles returned through `void **` are owned by the core until the
//!   matching `*_release`. The core guarantees release order (children
//!   before parents) and single-owner access to sessions and generations,
//!   so providers need no locking of their own for those.
//! - Output buffers in [`turbo_provider_result`] are borrowed from the
//!   session. They stay valid until the next `session_run` or
//!   `session_release`, whichever comes first. The core enforces that no
//!   run happens while a result is outstanding.
//! - Text and array pointers in arguments are valid for the call only.
//! - Provider functions must not unwind or throw across this boundary.
//!   Catch everything and return `TURBO_E_PANIC` with a message.

use core::ffi::{c_char, c_void};

use crate::{
    turbo_buffer_desc, turbo_capability, turbo_classify_options, turbo_context_desc, turbo_device_info,
    turbo_embed_options, turbo_error, turbo_generate_desc, turbo_generation_chunk, turbo_message, turbo_model_desc,
    turbo_model_info, turbo_native_handle, turbo_rerank_options, turbo_run_options, turbo_session_desc,
    turbo_session_stats, turbo_span, turbo_tensor_info, turbo_text, turbo_token_batch,
};

/// Provider ABI version. Equals the public ABI version; bumped together
/// (a test asserts this).
pub const TURBO_PROVIDER_ABI_VERSION: u32 = 2;

/// A buffer the provider allocated or imported.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct turbo_provider_buffer {
    /// `sizeof(turbo_provider_buffer)`.
    pub struct_size: u32,
    /// Reserved; 0.
    pub reserved: u32,
    /// Provider handle, passed to `buffer_read`, `buffer_export`, `buffer_release`.
    pub handle: *mut c_void,
    /// Stable host pointer for host-visible placements, else NULL.
    pub host_ptr: *mut c_void,
    /// Description as allocated (placement, dtype, shape, strides, bytes).
    pub desc: turbo_buffer_desc,
}

/// One output of a run, borrowed from the session.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct turbo_provider_output {
    /// `sizeof(turbo_provider_output)`.
    pub struct_size: u32,
    /// Rank of the logical shape.
    pub ndim: u32,
    /// Output name, valid for the session's lifetime.
    pub name: turbo_text,
    /// The backing buffer (capacity may exceed the logical shape).
    pub buffer: turbo_provider_buffer,
    /// Logical shape.
    pub shape: [u64; crate::TURBO_MAX_RANK],
}

/// Result of `session_run`. Arrays are borrowed until the next run or release.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct turbo_provider_result {
    /// `sizeof(turbo_provider_result)`.
    pub struct_size: u32,
    /// Number of outputs (at least 1 on success).
    pub n_outputs: u32,
    /// Outputs; output 0 is primary.
    pub outputs: *const turbo_provider_output,
    /// Number of spans (token classification).
    pub n_spans: u32,
    /// Reserved; 0.
    pub reserved: u32,
    /// Spans.
    pub spans: *const turbo_span,
}

/// The provider vtable. All function pointers except those documented as
/// optional must be non-NULL; the core rejects a vtable with a NULL required
/// entry at load time with `TURBO_E_PROVIDER_LOAD`.
#[repr(C)]
pub struct turbo_provider_vtbl {
    /// `sizeof(turbo_provider_vtbl)`.
    pub struct_size: u32,
    /// Must equal `TURBO_PROVIDER_ABI_VERSION`.
    pub abi_version: u32,
    /// Stable provider id, NUL-terminated (`cuda`, `openvino`, `metal`, `hailo`, `ggml`, `static`, `mock`).
    pub id: *const c_char,
    /// Provider version, NUL-terminated.
    pub version: *const c_char,
    /// Provider-global state passed to the provider-level functions.
    pub state: *mut c_void,

    /// Number of devices this provider offers.
    pub device_count: Option<unsafe extern "C" fn(state: *mut c_void, out: *mut u32, err: *mut turbo_error) -> i32>,
    /// Static device info for an ordinal.
    pub device_info: Option<
        unsafe extern "C" fn(
            state: *mut c_void,
            ordinal: u32,
            out: *mut turbo_device_info,
            err: *mut turbo_error,
        ) -> i32,
    >,
    /// Capability cell.
    pub capability: Option<
        unsafe extern "C" fn(
            state: *mut c_void,
            ordinal: u32,
            task: u32,
            modality: u32,
            out: *mut turbo_capability,
            err: *mut turbo_error,
        ) -> i32,
    >,
    /// Per-request feasibility for a bundle directory.
    pub can_run: Option<
        unsafe extern "C" fn(
            state: *mut c_void,
            ordinal: u32,
            bundle_dir: turbo_text,
            task: u32,
            modality: u32,
            err: *mut turbo_error,
        ) -> i32,
    >,
    /// Create a context on a device.
    pub context_create: Option<
        unsafe extern "C" fn(
            state: *mut c_void,
            ordinal: u32,
            desc: *const turbo_context_desc,
            out: *mut *mut c_void,
            err: *mut turbo_error,
        ) -> i32,
    >,
    /// Release a context (all its models and buffers are already released).
    pub context_release: Option<unsafe extern "C" fn(ctx: *mut c_void)>,

    /// Allocate a buffer.
    pub buffer_alloc: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            desc: *const turbo_buffer_desc,
            out: *mut turbo_provider_buffer,
            err: *mut turbo_error,
        ) -> i32,
    >,
    /// Import caller memory. Optional; NULL means unsupported.
    pub buffer_import: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            desc: *const turbo_buffer_desc,
            handle: *const turbo_native_handle,
            out: *mut turbo_provider_buffer,
            err: *mut turbo_error,
        ) -> i32,
    >,
    /// Blocking copy of the whole buffer to host memory of `bytes` bytes (must equal `desc.bytes`).
    pub buffer_read:
        Option<unsafe extern "C" fn(buffer: *mut c_void, dst: *mut c_void, bytes: u64, err: *mut turbo_error) -> i32>,
    /// Export a native handle. Optional.
    pub buffer_export: Option<
        unsafe extern "C" fn(
            buffer: *mut c_void,
            kind: u32,
            out: *mut turbo_native_handle,
            err: *mut turbo_error,
        ) -> i32,
    >,
    /// Release a buffer obtained from `buffer_alloc` or `buffer_import`.
    pub buffer_release: Option<unsafe extern "C" fn(buffer: *mut c_void)>,

    /// Load a verified bundle. The core has already verified hashes and
    /// checked the capability cell; the provider re-reads what it needs.
    pub model_load: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            bundle_dir: turbo_text,
            desc: *const turbo_model_desc,
            out: *mut *mut c_void,
            err: *mut turbo_error,
        ) -> i32,
    >,
    /// What loaded.
    pub model_info:
        Option<unsafe extern "C" fn(model: *mut c_void, out: *mut turbo_model_info, err: *mut turbo_error) -> i32>,
    /// Label by index (classifiers). The view is valid for the model's lifetime.
    pub model_label: Option<
        unsafe extern "C" fn(model: *mut c_void, index: u32, out: *mut turbo_text, err: *mut turbo_error) -> i32,
    >,
    /// Named tensor info (RUN). Optional when the provider loads no generic models.
    pub model_io_info: Option<
        unsafe extern "C" fn(
            model: *mut c_void,
            direction: u32,
            index: u32,
            out: *mut turbo_tensor_info,
            err: *mut turbo_error,
        ) -> i32,
    >,
    /// Release a model.
    pub model_release: Option<unsafe extern "C" fn(model: *mut c_void)>,

    /// Create a session. `desc` maxima are already resolved and validated.
    pub session_create: Option<
        unsafe extern "C" fn(
            model: *mut c_void,
            desc: *const turbo_session_desc,
            out: *mut *mut c_void,
            err: *mut turbo_error,
        ) -> i32,
    >,
    /// Write texts for embedding. Options are already validated by the core.
    pub session_write_text: Option<
        unsafe extern "C" fn(
            session: *mut c_void,
            texts: *const turbo_text,
            count: u32,
            opts: *const turbo_embed_options,
            err: *mut turbo_error,
        ) -> i32,
    >,
    /// Write prepared tokens (already validated).
    pub session_write_tokens: Option<
        unsafe extern "C" fn(session: *mut c_void, batch: *const turbo_token_batch, err: *mut turbo_error) -> i32,
    >,
    /// Write a query and documents.
    pub session_write_pairs: Option<
        unsafe extern "C" fn(
            session: *mut c_void,
            query: *const turbo_text,
            docs: *const turbo_text,
            count: u32,
            opts: *const turbo_rerank_options,
            err: *mut turbo_error,
        ) -> i32,
    >,
    /// Write texts for classification.
    pub session_write_text_classify: Option<
        unsafe extern "C" fn(
            session: *mut c_void,
            texts: *const turbo_text,
            count: u32,
            opts: *const turbo_classify_options,
            err: *mut turbo_error,
        ) -> i32,
    >,
    /// Bind a named tensor (RUN). `buffer` is a handle from this provider's `buffer_alloc`/`buffer_import`.
    pub session_bind: Option<
        unsafe extern "C" fn(session: *mut c_void, name: turbo_text, buffer: *mut c_void, err: *mut turbo_error) -> i32,
    >,
    /// Execute. The result arrays are borrowed until the next run or release.
    pub session_run: Option<
        unsafe extern "C" fn(
            session: *mut c_void,
            opts: *const turbo_run_options,
            out: *mut turbo_provider_result,
            err: *mut turbo_error,
        ) -> i32,
    >,
    /// Counters.
    pub session_stats:
        Option<unsafe extern "C" fn(session: *mut c_void, out: *mut turbo_session_stats, err: *mut turbo_error) -> i32>,
    /// Release a session.
    pub session_release: Option<unsafe extern "C" fn(session: *mut c_void)>,

    /// Create a generation. Optional; NULL means the provider does not generate.
    pub generation_create: Option<
        unsafe extern "C" fn(
            model: *mut c_void,
            desc: *const turbo_generate_desc,
            out: *mut *mut c_void,
            err: *mut turbo_error,
        ) -> i32,
    >,
    /// Apply the chat template and tokenize.
    pub generation_prompt: Option<
        unsafe extern "C" fn(
            generation: *mut c_void,
            messages: *const turbo_message,
            count: u32,
            err: *mut turbo_error,
        ) -> i32,
    >,
    /// Use caller-supplied prompt tokens.
    pub generation_prompt_tokens: Option<
        unsafe extern "C" fn(generation: *mut c_void, ids: *const i32, count: u32, err: *mut turbo_error) -> i32,
    >,
    /// Next chunk. Pointers in `out` stay valid until the next call on this generation.
    pub generation_step: Option<
        unsafe extern "C" fn(generation: *mut c_void, out: *mut turbo_generation_chunk, err: *mut turbo_error) -> i32,
    >,
    /// Cancel.
    pub generation_cancel: Option<unsafe extern "C" fn(generation: *mut c_void)>,
    /// Release a generation.
    pub generation_release: Option<unsafe extern "C" fn(generation: *mut c_void)>,
}

/// Signature of the exported `turbo_provider_get` symbol. `core_abi_version`
/// is the core's `TURBO_ABI_VERSION`; return NULL if unsupported.
pub type turbo_provider_get_fn = Option<unsafe extern "C" fn(core_abi_version: u32) -> *const turbo_provider_vtbl>;

#[cfg(test)]
mod tests {
    #[test]
    fn provider_abi_tracks_public_abi() {
        assert_eq!(super::TURBO_PROVIDER_ABI_VERSION, crate::TURBO_ABI_VERSION);
    }
}
