//! Safe wrapper over the TurboRerank C ABI (`include/turborerank.h`).
//!
//! Token buffers are caller-written (64-byte aligned on CPU; cudaHostAlloc
//! pinned on CUDA; Level Zero USM on OpenVINO; MTLResourceStorageModeShared
//! on Metal). `forward` does not allocate.
//! CUDA / AUTO-with-CUDA run the device MiniLM CE. OpenVINO GPU/CPU run
//! CompiledModel with `ov::Tensor(..., usm_pointer)`. Metal / AUTO-with-Metal
//! run first-party Metal kernels bound to those shared MTLBuffers.
//! TensorRT fails loud — never a mock score.

#![allow(clippy::result_large_err)]

pub mod ffi;

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::Path;
use std::ptr;

use ffi::{
    turborerank_abi_version, turborerank_activation, turborerank_buffer, turborerank_buffer_alloc,
    turborerank_buffer_free, turborerank_device, turborerank_device_name, turborerank_engine,
    turborerank_engine_create, turborerank_engine_destroy, turborerank_forward,
    turborerank_last_error, turborerank_list_models, turborerank_load_model,
    turborerank_model_info, turborerank_model_list_free, turborerank_pack_ids,
    turborerank_pack_text, turborerank_score, turborerank_score_options, turborerank_status,
    turborerank_status_name, turborerank_str, turborerank_truncation,
};

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
    fn to_c(self) -> turborerank_device {
        match self {
            Self::Auto => turborerank_device::TURBORERANK_DEVICE_AUTO,
            Self::Cpu => turborerank_device::TURBORERANK_DEVICE_CPU,
            Self::Cuda => turborerank_device::TURBORERANK_DEVICE_CUDA,
            Self::TensorRt => turborerank_device::TURBORERANK_DEVICE_TENSORRT,
            Self::OpenVinoCpu => turborerank_device::TURBORERANK_DEVICE_OPENVINO_CPU,
            Self::OpenVinoGpu => turborerank_device::TURBORERANK_DEVICE_OPENVINO_GPU,
            Self::OpenVinoNpu => turborerank_device::TURBORERANK_DEVICE_OPENVINO_NPU,
            Self::Metal => turborerank_device::TURBORERANK_DEVICE_METAL,
            Self::Mock => turborerank_device::TURBORERANK_DEVICE_MOCK,
        }
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Truncation {
    LongestFirst,
    QueryPriority,
    Error,
}

impl Truncation {
    fn to_c(self) -> turborerank_truncation {
        match self {
            Self::LongestFirst => turborerank_truncation::TURBORERANK_TRUNC_LONGEST_FIRST,
            Self::QueryPriority => turborerank_truncation::TURBORERANK_TRUNC_QUERY_PRIORITY,
            Self::Error => turborerank_truncation::TURBORERANK_TRUNC_ERROR,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    Sigmoid,
    Identity,
}

impl Activation {
    fn to_c(self) -> turborerank_activation {
        match self {
            Self::Sigmoid => turborerank_activation::TURBORERANK_ACT_SIGMOID,
            Self::Identity => turborerank_activation::TURBORERANK_ACT_IDENTITY,
        }
    }
}

#[derive(Debug, thiserror::Error)]
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
}

fn cstr<'a>(p: *const c_char) -> &'a str {
    if p.is_null() {
        return "";
    }
    unsafe { CStr::from_ptr(p).to_str().unwrap_or("") }
}

fn map_status(st: turborerank_status, detail: &str) -> Result<(), Error> {
    if st == turborerank_status::TURBORERANK_OK {
        return Ok(());
    }
    let name = cstr(unsafe { turborerank_status_name(st) });
    let msg = if detail.is_empty() {
        name.to_string()
    } else {
        format!("{name}: {detail}")
    };
    Err(match st.0 {
        1 => Error::InvalidArgument(msg),
        2 => Error::NotFound(msg),
        3 => Error::NotImplemented(msg),
        4 => Error::Unavailable(msg),
        5 => Error::Internal(msg),
        6 => Error::OutOfMemory(msg),
        7 => Error::UnsupportedDevice(msg),
        _ => Error::Internal(msg),
    })
}

fn last_err(engine: *const turborerank_engine) -> String {
    cstr(unsafe { turborerank_last_error(engine) }).to_string()
}

pub fn abi_version() -> u32 {
    unsafe { turborerank_abi_version() }
}

#[derive(Debug)]
pub struct Engine {
    raw: *mut turborerank_engine,
}

unsafe impl Send for Engine {}

impl Engine {
    pub fn create(device: Device) -> Result<Self, Error> {
        Self::create_with_config(device, None)
    }

    pub fn create_with_config(device: Device, config: Option<&Path>) -> Result<Self, Error> {
        let c_path = config
            .map(|p| CString::new(p.to_string_lossy().as_bytes()).ok())
            .flatten();
        let ptr = c_path.as_ref().map(|s| s.as_ptr()).unwrap_or(ptr::null());
        let mut out = ptr::null_mut();
        let st = unsafe { turborerank_engine_create(device.to_c(), ptr, &mut out) };
        map_status(st, &last_err(ptr::null()))?;
        if out.is_null() {
            return Err(Error::Internal("engine_create returned null".into()));
        }
        Ok(Self { raw: out })
    }

    pub fn last_error(&self) -> String {
        last_err(self.raw)
    }

    pub fn load_model(&self, alias: &str) -> Result<(), Error> {
        let st = unsafe { turborerank_load_model(self.raw, alias.as_ptr().cast(), alias.len()) };
        map_status(st, &self.last_error())
    }

    pub fn list_models(&self) -> Result<Vec<ModelInfo>, Error> {
        let mut infos = ptr::null_mut();
        let mut n = 0usize;
        let st = unsafe { turborerank_list_models(self.raw, &mut infos, &mut n) };
        map_status(st, &self.last_error())?;
        let mut out = Vec::with_capacity(n);
        if !infos.is_null() {
            let slice = unsafe { std::slice::from_raw_parts(infos, n) };
            for m in slice {
                let alias = unsafe {
                    if m.alias.ptr.is_null() {
                        String::new()
                    } else {
                        std::str::from_utf8(std::slice::from_raw_parts(
                            m.alias.ptr as *const u8,
                            m.alias.len,
                        ))
                        .unwrap_or("")
                        .to_string()
                    }
                };
                out.push(ModelInfo {
                    alias,
                    max_length: m.max_length,
                    hidden_size: m.hidden_size,
                    ready: m.ready != 0,
                });
            }
            unsafe { turborerank_model_list_free(infos, n) };
        }
        Ok(out)
    }

    pub fn pack_text(
        &self,
        buffer: &mut TokenBuffer,
        row: u32,
        query: &str,
        document: &str,
        truncation: Truncation,
        max_length: u32,
    ) -> Result<(), Error> {
        let q = turborerank_str {
            ptr: query.as_ptr().cast(),
            len: query.len(),
        };
        let d = turborerank_str {
            ptr: document.as_ptr().cast(),
            len: document.len(),
        };
        let st = unsafe {
            turborerank_pack_text(
                self.raw,
                buffer.raw,
                row,
                q,
                d,
                truncation.to_c(),
                max_length,
            )
        };
        map_status(st, &self.last_error())
    }

    pub fn forward(
        &self,
        buffer: &TokenBuffer,
        n_rows: u32,
        activation: Activation,
    ) -> Result<Vec<f32>, Error> {
        let mut scores = vec![0.0f32; n_rows as usize];
        let st = unsafe {
            turborerank_forward(
                self.raw,
                buffer.raw,
                n_rows,
                activation.to_c(),
                scores.as_mut_ptr(),
            )
        };
        map_status(st, &self.last_error())?;
        Ok(scores)
    }

    pub fn score(
        &self,
        alias: Option<&str>,
        query: &str,
        documents: &[&str],
        truncation: Truncation,
        activation: Activation,
        max_length: u32,
    ) -> Result<Vec<f32>, Error> {
        if documents.is_empty() {
            return Err(Error::InvalidArgument("documents must not be empty".into()));
        }
        let views: Vec<turborerank_str> = documents
            .iter()
            .map(|d| turborerank_str {
                ptr: d.as_ptr().cast(),
                len: d.len(),
            })
            .collect();
        let q = turborerank_str {
            ptr: query.as_ptr().cast(),
            len: query.len(),
        };
        let opts = turborerank_score_options {
            truncation: truncation.to_c(),
            activation: activation.to_c(),
            max_length,
        };
        let (alias_ptr, alias_len) = match alias {
            Some(a) => (a.as_ptr().cast(), a.len()),
            None => (ptr::null(), 0usize),
        };
        let mut scores = vec![0.0f32; documents.len()];
        let st = unsafe {
            turborerank_score(
                self.raw,
                alias_ptr,
                alias_len,
                q,
                views.as_ptr(),
                views.len(),
                &opts,
                scores.as_mut_ptr(),
            )
        };
        map_status(st, &self.last_error())?;
        Ok(scores)
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { turborerank_engine_destroy(self.raw) };
    }
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub alias: String,
    pub max_length: u32,
    pub hidden_size: u32,
    pub ready: bool,
}

#[derive(Debug)]
pub struct TokenBuffer {
    raw: *mut turborerank_buffer,
}

unsafe impl Send for TokenBuffer {}

impl TokenBuffer {
    pub fn alloc(device: Device, batch: u32, seq: u32) -> Result<Self, Error> {
        let mut out = ptr::null_mut();
        let st = unsafe { turborerank_buffer_alloc(device.to_c(), batch, seq, &mut out) };
        map_status(st, &last_err(ptr::null()))?;
        if out.is_null() {
            return Err(Error::Internal("buffer_alloc returned null".into()));
        }
        Ok(Self { raw: out })
    }

    fn inner(&self) -> &turborerank_buffer {
        unsafe { &*self.raw }
    }

    pub fn batch(&self) -> u32 {
        self.inner().batch
    }

    pub fn seq(&self) -> u32 {
        self.inner().seq
    }

    pub fn device(&self) -> Device {
        match self.inner().device.0 {
            0 => Device::Auto,
            1 => Device::Cpu,
            2 => Device::Cuda,
            3 => Device::TensorRt,
            4 => Device::OpenVinoCpu,
            5 => Device::OpenVinoGpu,
            6 => Device::OpenVinoNpu,
            7 => Device::Metal,
            8 => Device::Mock,
            _ => Device::Cpu,
        }
    }

    pub fn input_ids_mut(&mut self) -> &mut [i32] {
        let b = self.inner();
        unsafe {
            std::slice::from_raw_parts_mut(b.input_ids, (b.batch * b.row_stride) as usize)
        }
    }

    pub fn input_ids(&self) -> &[i32] {
        let b = self.inner();
        unsafe { std::slice::from_raw_parts(b.input_ids, (b.batch * b.row_stride) as usize) }
    }

    pub fn attention_mask(&self) -> &[i32] {
        let b = self.inner();
        unsafe {
            std::slice::from_raw_parts(b.attention_mask, (b.batch * b.row_stride) as usize)
        }
    }

    pub fn token_type_ids(&self) -> &[i32] {
        let b = self.inner();
        unsafe {
            std::slice::from_raw_parts(b.token_type_ids, (b.batch * b.row_stride) as usize)
        }
    }

    pub fn pack_ids(
        &mut self,
        row: u32,
        query: &[i32],
        doc: &[i32],
        truncation: Truncation,
        max_length: u32,
    ) -> Result<(), Error> {
        let st = unsafe {
            turborerank_pack_ids(
                self.raw,
                row,
                query.as_ptr(),
                query.len(),
                doc.as_ptr(),
                doc.len(),
                truncation.to_c(),
                max_length,
            )
        };
        map_status(st, &last_err(ptr::null()))
    }

    pub fn ptr_aligned(&self) -> bool {
        let b = self.inner();
        let a = |p: *mut i32| (p as usize) % 64 == 0;
        a(b.input_ids) && a(b.attention_mask) && a(b.token_type_ids) && a(b.position_ids)
    }
}

impl Drop for TokenBuffer {
    fn drop(&mut self) {
        unsafe { turborerank_buffer_free(self.raw) };
    }
}

pub fn device_name(device: Device) -> &'static str {
    cstr(unsafe { turborerank_device_name(device.to_c()) })
}

/// Workspace root baked in at compile time (same as TurboEmbed).
pub fn workspace_root() -> &'static str {
    option_env!("INFERSTREAM_ROOT").unwrap_or(".")
}

pub fn default_model_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(workspace_root()).join("models/rerank/ms-marco-minilm-l6")
}

pub fn weights_present() -> bool {
    let d = default_model_dir();
    d.join("model.safetensors").is_file() && d.join("vocab.txt").is_file()
}

pub fn default_ov_ir_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(workspace_root()).join("models/ov-rerank/ms-marco-minilm-l6")
}

pub fn ov_ir_present() -> bool {
    let d = default_ov_ir_dir();
    (d.join("openvino_model.xml").is_file() || d.join("model.xml").is_file())
        && (d.join("openvino_model.bin").is_file() || d.join("model.bin").is_file())
        && d.join("vocab.txt").is_file()
}

// Silence unused import of turborerank_model_info in some rustc versions.
#[allow(dead_code)]
fn _keep_model_info(_: &turborerank_model_info) {}
