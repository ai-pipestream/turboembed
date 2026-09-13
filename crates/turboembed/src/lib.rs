//! Safe zero-copy wrapper over the TurboEmbed C ABI
//! (`include/turboembed.h`).
//!
//! # Ownership
//!
//! * **Inputs** (`&str`, `&[&str]`) are pointer+length *views*. The C ABI
//!   borrows them for the duration of the call only. The caller keeps the
//!   allocation. This crate does not copy text unless the C layer does.
//! * **Outputs** (`Embeddings`, `ModelList`) own the engine-allocated
//!   buffer. Dropping them calls the matching `*_free`. A `&[f32]` from
//!   [`Embeddings::values`] is valid only while the `Embeddings` value
//!   lives. Destroying [`Engine`] also invalidates outstanding results —
//!   drop results first.
//! * **Packed bytes** ([`Embeddings::packed`]) alias the same allocation
//!   as the typed floats (little-endian FP32). One free releases both.
//! * The ABI is **not thread-safe** on a single engine. `Engine` is `Send`
//!   (move it to another thread) but not `Sync`.
//!
//! The linked implementation is **arch-specific**:
//!
//! * **macOS:** `libTurboEmbed.dylib` (`@_cdecl` → mlx-swift `MlxEngine`).
//!   `Device::Metal` / `Device::Auto` + `minilm` is FP MiniLM, hidden-state
//!   **mean + L2** on Apple GPU. Not the BERT NSP pooler
//!   (`tanh(dense(CLS))`), not the 8-d mock stub, not Python.
//! * **elsewhere:** C++ stub (`native/turboembed`). `mock-embed` works
//!   on explicit [`Device::Mock`] (and explicit CPU for ABI smoke).
//!   Catalog aliases return [`Error::NotImplemented`] unless a real
//!   provider feature is compiled in.
//!
//! `--features genai` links `ov::genai::TextEmbeddingPipeline` with the
//! official device string (`"GPU"`, `"CPU"`, or `"NPU"`).
//! `Engine::create(Device::OpenVinoGpu)` fails if the GPU plugin is
//! missing (no CPU swap). Same for [`Device::OpenVinoNpu`].
//! `Device::OpenVinoCpu` / `Device::Cpu` compile
//! `"CPU"` and return real embeds. No OVMS. No Python.
//!
//! `--features ort-cuda` registers the ONNX Runtime CUDA EP with
//! `error_on_failure`, binds inputs/outputs on `AllocationDevice::CUDA`
//! via IoBinding, and mean+L2 pools on the host after the device→host
//! copy. `Engine::create(Device::Cuda)` / [`Device::Auto`] then
//! `load_model("minilm")` is the NVIDIA CUDA proof path. A CUDA/AUTO
//! request never silently becomes CPU. `Device::Cpu` is an explicit
//! CPU EP path (same ONNX, same mean+L2).
//! [`Device::TensorRt`] loads the ORT TensorRT EP (`error_on_failure`).
//! Missing `libnvinfer.so.10` is a hard error — not a CUDA or CPU
//! session. Live MiniLM proof: `docs/turboembed.md`.
//!
//! # Device policy
//!
//! GPU / accelerator requests (`Auto`, `Cuda`, `TensorRt`, OpenVINO GPU/NPU,
//! `Metal`) **fail** if that device is missing. They never fall back to
//! CPU or the 8-d FNV mock. `Auto` is host-default **GPU** (Metal on Mac),
//! not "CPU if GPU is down". `Cpu` / `OpenVinoCpu` run only when selected.
//! [`Device::Mock`] is ABI smoke only — never a silent substitute for
//! catalog aliases (`minilm`, `bge-*`, …).

#![allow(clippy::result_large_err)]

pub mod ffi;

#[cfg(feature = "ort-cuda")]
mod catalog;
#[cfg(feature = "ort-cuda")]
mod ort_cuda;
#[cfg(feature = "ort-cuda")]
mod ort_cuda_c;

use std::ffi::CStr;
use std::os::raw::c_void;
use std::ptr::{self, NonNull};

use ffi::{
    turboembed_abi_version, turboembed_device, turboembed_device_name, turboembed_embed,
    turboembed_embed_one, turboembed_embed_options, turboembed_embed_result,
    turboembed_embed_result_free, turboembed_embed_stream, turboembed_engine,
    turboembed_engine_create, turboembed_engine_destroy, turboembed_last_error,
    turboembed_list_models, turboembed_load_model, turboembed_model_info,
    turboembed_model_list_free, turboembed_output_format, turboembed_pooling,
    turboembed_register_provider, turboembed_status, turboembed_status_name, turboembed_str,
};

/// Device the stub (and later providers) should target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    Auto,
    Cpu,
    Cuda,
    TensorRt,
    OpenVinoCpu,
    OpenVinoGpu,
    OpenVinoNpu,
    Metal,
    Mock,
}

impl Device {
    fn to_c(self) -> turboembed_device {
        match self {
            Self::Auto => turboembed_device::TURBOEMBED_DEVICE_AUTO,
            Self::Cpu => turboembed_device::TURBOEMBED_DEVICE_CPU,
            Self::Cuda => turboembed_device::TURBOEMBED_DEVICE_CUDA,
            Self::TensorRt => turboembed_device::TURBOEMBED_DEVICE_TENSORRT,
            Self::OpenVinoCpu => turboembed_device::TURBOEMBED_DEVICE_OPENVINO_CPU,
            Self::OpenVinoGpu => turboembed_device::TURBOEMBED_DEVICE_OPENVINO_GPU,
            Self::OpenVinoNpu => turboembed_device::TURBOEMBED_DEVICE_OPENVINO_NPU,
            Self::Metal => turboembed_device::TURBOEMBED_DEVICE_METAL,
            Self::Mock => turboembed_device::TURBOEMBED_DEVICE_MOCK,
        }
    }

    fn from_c(raw: turboembed_device) -> Self {
        match raw {
            turboembed_device::TURBOEMBED_DEVICE_CPU => Self::Cpu,
            turboembed_device::TURBOEMBED_DEVICE_CUDA => Self::Cuda,
            turboembed_device::TURBOEMBED_DEVICE_TENSORRT => Self::TensorRt,
            turboembed_device::TURBOEMBED_DEVICE_OPENVINO_CPU => Self::OpenVinoCpu,
            turboembed_device::TURBOEMBED_DEVICE_OPENVINO_GPU => Self::OpenVinoGpu,
            turboembed_device::TURBOEMBED_DEVICE_OPENVINO_NPU => Self::OpenVinoNpu,
            turboembed_device::TURBOEMBED_DEVICE_METAL => Self::Metal,
            turboembed_device::TURBOEMBED_DEVICE_MOCK => Self::Mock,
            _ => Self::Auto,
        }
    }

    /// ABI device name (`"cuda"`, `"mock"`, …).
    pub fn as_str(self) -> &'static str {
        c_str(unsafe { turboembed_device_name(self.to_c()) })
    }
}

/// Pooling hint forwarded to the provider.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Pooling {
    #[default]
    Default,
    Mean,
    Cls,
    Last,
}

impl Pooling {
    fn to_c(self) -> turboembed_pooling {
        match self {
            Self::Default => turboembed_pooling::TURBOEMBED_POOLING_DEFAULT,
            Self::Mean => turboembed_pooling::TURBOEMBED_POOLING_MEAN,
            Self::Cls => turboembed_pooling::TURBOEMBED_POOLING_CLS,
            Self::Last => turboembed_pooling::TURBOEMBED_POOLING_LAST,
        }
    }
}

/// How the C layer presents the buffer. Typed and packed alias one allocation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OutputFormat {
    #[default]
    Typed,
    PackedBytes,
}

impl OutputFormat {
    fn to_c(self) -> turboembed_output_format {
        match self {
            Self::Typed => turboembed_output_format::TURBOEMBED_OUTPUT_TYPED,
            Self::PackedBytes => turboembed_output_format::TURBOEMBED_OUTPUT_PACKED_BYTES,
        }
    }
}

/// Options for a single embed call. All fields are hints; the stub ignores them.
#[derive(Clone, Copy, Debug, Default)]
pub struct EmbedOptions {
    pub pooling: Pooling,
    /// `None` = provider default.
    pub normalize: Option<bool>,
    /// `None` / 0 = provider default.
    pub truncate_to: Option<u32>,
    pub output_format: OutputFormat,
}

impl EmbedOptions {
    fn to_c(self) -> turboembed_embed_options {
        turboembed_embed_options {
            pooling: self.pooling.to_c(),
            normalize: match self.normalize {
                None => -1,
                Some(false) => 0,
                Some(true) => 1,
            },
            truncate_to: self.truncate_to.unwrap_or(0),
            output_format: self.output_format.to_c(),
        }
    }
}

/// ABI / provider error. `message` is copied out of `turboembed_last_error`.
#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum Error {
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("not implemented: {0}")]
    NotImplemented(String),
    #[error("unavailable: {0}")]
    Unavailable(String),
    #[error("internal: {0}")]
    Internal(String),
    #[error("out of memory: {0}")]
    OutOfMemory(String),
    #[error("unsupported device: {0}")]
    UnsupportedDevice(String),
    #[error("status {code}: {message}")]
    Other { code: i32, message: String },
}

impl Error {
    fn from_status(status: turboembed_status, engine: *const turboembed_engine) -> Self {
        let message = last_error(engine);
        match status {
            turboembed_status::TURBOEMBED_ERR_INVALID_ARGUMENT => Self::InvalidArgument(message),
            turboembed_status::TURBOEMBED_ERR_NOT_FOUND => Self::NotFound(message),
            turboembed_status::TURBOEMBED_ERR_NOT_IMPLEMENTED => Self::NotImplemented(message),
            turboembed_status::TURBOEMBED_ERR_UNAVAILABLE => Self::Unavailable(message),
            turboembed_status::TURBOEMBED_ERR_INTERNAL => Self::Internal(message),
            turboembed_status::TURBOEMBED_ERR_OUT_OF_MEMORY => Self::OutOfMemory(message),
            turboembed_status::TURBOEMBED_ERR_UNSUPPORTED_DEVICE => {
                Self::UnsupportedDevice(message)
            }
            other => Self::Other {
                code: other as i32,
                message,
            },
        }
    }
}

fn c_str<'a>(ptr: *const std::os::raw::c_char) -> &'a str {
    if ptr.is_null() {
        return "";
    }
    unsafe { CStr::from_ptr(ptr) }.to_str().unwrap_or("")
}

fn last_error(engine: *const turboembed_engine) -> String {
    c_str(unsafe { turboembed_last_error(engine) }).to_string()
}

fn check(status: turboembed_status, engine: *const turboembed_engine) -> Result<(), Error> {
    if status == turboembed_status::TURBOEMBED_OK {
        Ok(())
    } else {
        Err(Error::from_status(status, engine))
    }
}

/// Frozen ABI version compiled into the header / stub.
pub fn abi_version() -> u32 {
    unsafe { turboembed_abi_version() }
}

/// Status name from the stub (`"OK"`, `"NOT_IMPLEMENTED"`, …).
pub fn status_name(status: turboembed_status) -> &'static str {
    c_str(unsafe { turboembed_status_name(status) })
}

/// Catalog entry from [`Engine::list_models`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInfo {
    pub alias: String,
    pub dim: u32,
    pub device: Device,
    pub ready: bool,
}

/// Engine-owned model list. Dropping it frees the C allocation.
pub struct ModelList {
    ptr: *mut turboembed_model_info,
    count: usize,
}

unsafe impl Send for ModelList {}

impl ModelList {
    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn get(&self, index: usize) -> Option<ModelInfo> {
        if index >= self.count || self.ptr.is_null() {
            return None;
        }
        let row = unsafe { &*self.ptr.add(index) };
        let alias = unsafe { str_view(row.alias) }.to_string();
        Some(ModelInfo {
            alias,
            dim: row.dim,
            device: Device::from_c(row.device),
            ready: row.ready != 0,
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = ModelInfo> + '_ {
        (0..self.count).filter_map(|i| self.get(i))
    }
}

impl Drop for ModelList {
    fn drop(&mut self) {
        unsafe { turboembed_model_list_free(self.ptr, self.count) }
        self.ptr = ptr::null_mut();
        self.count = 0;
    }
}

/// Engine-owned embed result. Typed floats and packed bytes alias one buffer.
///
/// Drop this **before** the [`Engine`] that produced it.
#[derive(Debug)]
pub struct Embeddings {
    raw: *mut turboembed_embed_result,
}

unsafe impl Send for Embeddings {}

impl Embeddings {
    fn from_raw(raw: *mut turboembed_embed_result) -> Result<Self, Error> {
        if raw.is_null() {
            return Err(Error::Internal("engine returned a null result".into()));
        }
        Ok(Self { raw })
    }

    fn inner(&self) -> &turboembed_embed_result {
        unsafe { &*self.raw }
    }

    pub fn dim(&self) -> usize {
        self.inner().dim as usize
    }

    pub fn count(&self) -> usize {
        self.inner().count as usize
    }

    /// Row-major `count * dim` floats. Valid until `self` is dropped.
    pub fn values(&self) -> &[f32] {
        let inner = self.inner();
        let n = inner.count as usize * inner.dim as usize;
        if inner.values.is_null() || n == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(inner.values, n) }
    }

    /// Little-endian FP32 blob of [`Self::values`]. Same allocation.
    pub fn packed(&self) -> &[u8] {
        let inner = self.inner();
        if inner.packed.is_null() || inner.packed_len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(inner.packed, inner.packed_len) }
    }

    /// One row as a float view. Valid until `self` is dropped.
    pub fn row(&self, index: usize) -> Option<&[f32]> {
        let dim = self.dim();
        let values = self.values();
        if dim == 0 {
            return None;
        }
        let start = index.checked_mul(dim)?;
        values.get(start..start + dim)
    }
}

impl Drop for Embeddings {
    fn drop(&mut self) {
        unsafe { turboembed_embed_result_free(self.raw) }
        self.raw = ptr::null_mut();
    }
}

/// Handle to a TurboEmbed engine (opaque C pointer).
///
/// Not `Sync`. Serialize calls. Drop outstanding [`Embeddings`] first.
pub struct Engine {
    raw: NonNull<turboembed_engine>,
}

unsafe impl Send for Engine {}

impl Engine {
    pub fn create(device: Device) -> Result<Self, Error> {
        Self::create_with_config(device, None)
    }

    pub fn create_with_config(device: Device, config_path: Option<&CStr>) -> Result<Self, Error> {
        let mut out: *mut turboembed_engine = ptr::null_mut();
        let path = config_path.map(|s| s.as_ptr()).unwrap_or(ptr::null());
        let status = unsafe { turboembed_engine_create(device.to_c(), path, &mut out) };
        check(status, ptr::null())?;
        let raw = NonNull::new(out)
            .ok_or_else(|| Error::Internal("create returned OK with a null engine".into()))?;
        Ok(Self { raw })
    }

    fn as_ptr(&self) -> *mut turboembed_engine {
        self.raw.as_ptr()
    }

    /// Raw C engine for Machine B GenAI arena receipts. Not a second ABI.
    #[doc(hidden)]
    pub fn raw_engine(&self) -> *mut turboembed_engine {
        self.as_ptr()
    }

    pub fn last_error(&self) -> String {
        last_error(self.as_ptr())
    }

    pub fn list_models(&self) -> Result<ModelList, Error> {
        let mut infos: *mut turboembed_model_info = ptr::null_mut();
        let mut count: usize = 0;
        let status = unsafe { turboembed_list_models(self.as_ptr(), &mut infos, &mut count) };
        check(status, self.as_ptr())?;
        Ok(ModelList { ptr: infos, count })
    }

    /// Load a catalog alias. Stub: `mock-embed` / `mock` succeed only on
    /// explicit [`Device::Mock`] / [`Device::Cpu`]. Catalog aliases never
    /// resolve to the 8-d FNV mock.
    /// `--features ort-cuda`: `minilm` loads ORT CUDA + IoBinding on
    /// [`Device::Cuda`] / [`Device::Auto`], the CPU EP on [`Device::Cpu`],
    /// or the TensorRT EP on [`Device::TensorRt`] (fails loud if
    /// `libnvinfer.so.10` is missing — not a CUDA/CPU stand-in).
    /// `--features genai`: `minilm` (and other `models/ov/<alias>` dirs)
    /// load `TextEmbeddingPipeline` on `"GPU"` or `"CPU"`.
    /// macOS: `minilm` loads MLX mean+L2 on [`Device::Metal`] / [`Device::Auto`].
    pub fn load_model(&self, alias: &str) -> Result<(), Error> {
        let status =
            unsafe { turboembed_load_model(self.as_ptr(), alias.as_ptr().cast(), alias.len()) };
        check(status, self.as_ptr())
    }

    /// Embed one text. `text` is a view for the duration of the call.
    pub fn embed_one(
        &self,
        alias: &str,
        text: &str,
        opts: &EmbedOptions,
    ) -> Result<Embeddings, Error> {
        let c_opts = opts.to_c();
        let mut out: *mut turboembed_embed_result = ptr::null_mut();
        let status = unsafe {
            turboembed_embed_one(
                self.as_ptr(),
                alias.as_ptr().cast(),
                alias.len(),
                text.as_ptr().cast(),
                text.len(),
                &c_opts,
                &mut out,
            )
        };
        check(status, self.as_ptr())?;
        Embeddings::from_raw(out)
    }

    /// Embed a batch. Each `&str` is a view for the duration of the call.
    pub fn embed(
        &self,
        alias: &str,
        texts: &[&str],
        opts: &EmbedOptions,
    ) -> Result<Embeddings, Error> {
        if texts.is_empty() {
            return Err(Error::InvalidArgument("texts must not be empty".into()));
        }
        let views: Vec<turboembed_str> = texts
            .iter()
            .map(|t| turboembed_str {
                ptr: t.as_ptr().cast(),
                len: t.len(),
            })
            .collect();
        let c_opts = opts.to_c();
        let mut out: *mut turboembed_embed_result = ptr::null_mut();
        let status = unsafe {
            turboembed_embed(
                self.as_ptr(),
                alias.as_ptr().cast(),
                alias.len(),
                views.as_ptr(),
                views.len(),
                &c_opts,
                &mut out,
            )
        };
        check(status, self.as_ptr())?;
        Embeddings::from_raw(out)
    }

    /// Embed then invoke `on_row(index, row, is_final)` per row.
    ///
    /// The `row` slice is valid only for the duration of that callback
    /// invocation (ABI rule). This wrapper copies nothing; `on_row` must
    /// copy if it needs the data later. The returned [`Embeddings`] still
    /// owns the full batch.
    pub fn embed_stream<F>(
        &self,
        alias: &str,
        texts: &[&str],
        opts: &EmbedOptions,
        mut on_row: F,
    ) -> Result<Embeddings, Error>
    where
        F: FnMut(u32, &[f32], bool),
    {
        if texts.is_empty() {
            return Err(Error::InvalidArgument("texts must not be empty".into()));
        }
        let views: Vec<turboembed_str> = texts
            .iter()
            .map(|t| turboembed_str {
                ptr: t.as_ptr().cast(),
                len: t.len(),
            })
            .collect();
        let c_opts = opts.to_c();
        let mut out: *mut turboembed_embed_result = ptr::null_mut();
        let mut cb_state = StreamState {
            on_row: &mut on_row,
        };
        let status = unsafe {
            turboembed_embed_stream(
                self.as_ptr(),
                alias.as_ptr().cast(),
                alias.len(),
                views.as_ptr(),
                views.len(),
                &c_opts,
                Some(stream_trampoline::<F>),
                (&mut cb_state as *mut StreamState<F>).cast::<c_void>(),
                &mut out,
            )
        };
        check(status, self.as_ptr())?;
        Embeddings::from_raw(out)
    }
}

struct StreamState<'a, F> {
    on_row: &'a mut F,
}

unsafe extern "C" fn stream_trampoline<F>(
    user_data: *mut c_void,
    index: u32,
    values: *const f32,
    dim: u32,
    is_final: i32,
) where
    F: FnMut(u32, &[f32], bool),
{
    if user_data.is_null() || values.is_null() {
        return;
    }
    let state = unsafe { &mut *user_data.cast::<StreamState<F>>() };
    let row = unsafe { std::slice::from_raw_parts(values, dim as usize) };
    (state.on_row)(index, row, is_final != 0);
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { turboembed_engine_destroy(self.raw.as_ptr()) }
    }
}

/// Provider registration is reserved. Stub always returns NotImplemented.
pub fn register_provider_stub() -> Result<(), Error> {
    let status = unsafe { turboembed_register_provider(ptr::null()) };
    check(status, ptr::null())
}

unsafe fn str_view(s: turboembed_str) -> &'static str {
    if s.ptr.is_null() || s.len == 0 {
        return "";
    }
    let bytes = unsafe { std::slice::from_raw_parts(s.ptr.cast::<u8>(), s.len) };
    std::str::from_utf8(bytes).unwrap_or("")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::TURBOEMBED_ABI_VERSION;

    #[test]
    fn abi_version_matches_header() {
        assert_eq!(abi_version(), TURBOEMBED_ABI_VERSION);
        assert_eq!(status_name(turboembed_status::TURBOEMBED_OK), "OK");
        assert_eq!(Device::Mock.as_str(), "mock");
    }

    #[test]
    fn apple_header_matches_canonical() {
        let canonical = include_str!("../../../include/turboembed.h");
        let apple = include_str!("../../../swift/Sources/TurboEmbedC/include/turboembed.h");
        assert_eq!(
            canonical, apple,
            "swift/Sources/TurboEmbedC/include/turboembed.h must stay identical to include/turboembed.h"
        );
    }
}
