//! Native MLX runtime handle. On macOS this is in-process Swift MLX via FFI.
//! Off-macOS every call is `Unavailable` so Linux CI still type-checks.

#[cfg(target_os = "macos")]
use std::ffi::{c_void, CString};
#[cfg(target_os = "macos")]
use std::os::raw::c_char;
#[cfg(target_os = "macos")]
use std::sync::Mutex;

use inferstream_backend::BackendError;
use serde::Deserialize;

#[cfg(target_os = "macos")]
use crate::ffi::{
    mlx_engine_create, mlx_engine_destroy, mlx_engine_embed, mlx_engine_free, mlx_engine_generate,
    mlx_engine_ping, take_cstr, MlxEngineHandle, MlxTokenCb,
};

/// Shared native MLX engine. One per process; models stay hot after first use.
pub struct MlxEngine {
    #[cfg(target_os = "macos")]
    inner: Mutex<Option<NativeSession>>,
}

#[cfg(target_os = "macos")]
struct NativeSession {
    handle: *mut MlxEngineHandle,
}

#[cfg(target_os = "macos")]
unsafe impl Send for NativeSession {}
#[cfg(target_os = "macos")]
unsafe impl Sync for NativeSession {}

#[derive(Debug, Deserialize)]
pub struct PingResult {
    pub device: String,
    pub metal_available: bool,
    pub matmul_ok: bool,
    #[serde(default)]
    pub mlx_version: String,
    #[serde(default)]
    pub active_memory: u64,
    #[serde(default)]
    pub peak_memory: u64,
}

#[derive(Debug)]
pub struct EmbedResult {
    pub dimensions: usize,
    pub vectors: Vec<Vec<f32>>,
}

#[derive(Debug)]
pub struct GenerateStats {
    pub tokens: u32,
    pub decode_tps: f64,
}

impl Default for MlxEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl MlxEngine {
    pub fn new() -> Self {
        Self {
            #[cfg(target_os = "macos")]
            inner: Mutex::new(None),
        }
    }

    pub fn from_env() -> Self {
        Self::new()
    }

    #[cfg(not(target_os = "macos"))]
    pub fn ping(&self) -> Result<PingResult, BackendError> {
        Err(unavailable())
    }

    #[cfg(not(target_os = "macos"))]
    pub fn embed(
        &self,
        _model: &str,
        _texts: &[String],
        _normalize: bool,
    ) -> Result<EmbedResult, BackendError> {
        Err(unavailable())
    }

    #[cfg(not(target_os = "macos"))]
    pub fn generate(
        &self,
        _model: &str,
        _prompt: &str,
        _max_tokens: u32,
        _on_token: impl FnMut(String),
    ) -> Result<GenerateStats, BackendError> {
        Err(unavailable())
    }
}

#[cfg(not(target_os = "macos"))]
fn unavailable() -> BackendError {
    BackendError::Unavailable(
        "native MLX (mlx-swift) is macOS-only; this binary is a compile stub".into(),
    )
}

#[cfg(target_os = "macos")]
impl MlxEngine {
    fn session(&self) -> Result<std::sync::MutexGuard<'_, Option<NativeSession>>, BackendError> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| BackendError::Internal("mlx engine lock poisoned".into()))?;
        if guard.is_none() {
            let mut err = std::ptr::null_mut();
            let handle = unsafe { mlx_engine_create(&mut err) };
            if handle.is_null() {
                let msg =
                    unsafe { take_cstr(err) }.unwrap_or_else(|| "mlx_engine_create failed".into());
                return Err(BackendError::Unavailable(msg));
            }
            *guard = Some(NativeSession { handle });
        }
        Ok(guard)
    }

    pub fn ping(&self) -> Result<PingResult, BackendError> {
        let guard = self.session()?;
        let handle = guard.as_ref().expect("session").handle;
        let mut out = std::ptr::null_mut();
        let mut err = std::ptr::null_mut();
        let rc = unsafe { mlx_engine_ping(handle, &mut out, &mut err) };
        if rc != 0 {
            let msg = unsafe { take_cstr(err) }.unwrap_or_else(|| "ping failed".into());
            return Err(BackendError::Internal(msg));
        }
        let json = unsafe { take_cstr(out) }.unwrap_or_else(|| "{}".into());
        serde_json::from_str(&json)
            .map_err(|e| BackendError::Internal(format!("malformed ping: {e}")))
    }

    pub fn embed(
        &self,
        model: &str,
        texts: &[String],
        normalize: bool,
    ) -> Result<EmbedResult, BackendError> {
        let model_c = CString::new(model)
            .map_err(|_| BackendError::InvalidRequest("model path contains NUL".into()))?;
        let text_c: Vec<CString> = texts
            .iter()
            .map(|t| {
                CString::new(t.as_str())
                    .map_err(|_| BackendError::InvalidRequest("text contains NUL".into()))
            })
            .collect::<Result<_, _>>()?;
        let ptrs: Vec<*const c_char> = text_c.iter().map(|s| s.as_ptr()).collect();
        let guard = self.session()?;
        let handle = guard.as_ref().expect("session").handle;
        let mut vectors = std::ptr::null_mut();
        let mut rows = 0usize;
        let mut dim = 0usize;
        let mut err = std::ptr::null_mut();
        let rc = unsafe {
            mlx_engine_embed(
                handle,
                model_c.as_ptr(),
                ptrs.as_ptr(),
                ptrs.len(),
                if normalize { 1 } else { 0 },
                &mut vectors,
                &mut rows,
                &mut dim,
                &mut err,
            )
        };
        if rc != 0 {
            let msg = unsafe { take_cstr(err) }.unwrap_or_else(|| "embed failed".into());
            return Err(BackendError::Internal(msg));
        }
        if vectors.is_null() || rows == 0 || dim == 0 {
            return Err(BackendError::Internal("embed returned empty tensor".into()));
        }
        let total = rows * dim;
        let slice = unsafe { std::slice::from_raw_parts(vectors, total) };
        let mut out = Vec::with_capacity(rows);
        for row in slice.chunks_exact(dim) {
            out.push(row.to_vec());
        }
        unsafe { mlx_engine_free(vectors as *mut c_void) };
        Ok(EmbedResult {
            dimensions: dim,
            vectors: out,
        })
    }

    pub fn generate<F: FnMut(String) + Send + 'static>(
        &self,
        model: &str,
        prompt: &str,
        max_tokens: u32,
        on_token: F,
    ) -> Result<GenerateStats, BackendError> {
        let model_c = CString::new(model)
            .map_err(|_| BackendError::InvalidRequest("model path contains NUL".into()))?;
        let prompt_c = CString::new(prompt)
            .map_err(|_| BackendError::InvalidRequest("prompt contains NUL".into()))?;

        struct Ctx {
            cb: std::sync::Mutex<Box<dyn FnMut(String) + Send>>,
        }
        unsafe extern "C" fn cb(token: *const c_char, user: *mut c_void) {
            if token.is_null() || user.is_null() {
                return;
            }
            let ctx = &*(user as *const Ctx);
            let s = std::ffi::CStr::from_ptr(token)
                .to_string_lossy()
                .into_owned();
            if let Ok(mut f) = ctx.cb.lock() {
                f(s);
            }
        }

        let ctx = Ctx {
            cb: std::sync::Mutex::new(Box::new(on_token)),
        };
        let guard = self.session()?;
        let handle = guard.as_ref().expect("session").handle;
        let mut tokens = 0u32;
        let mut tps = 0.0f64;
        let mut err = std::ptr::null_mut();
        let rc = unsafe {
            mlx_engine_generate(
                handle,
                model_c.as_ptr(),
                prompt_c.as_ptr(),
                max_tokens,
                cb as MlxTokenCb,
                &ctx as *const Ctx as *mut c_void,
                &mut tokens,
                &mut tps,
                &mut err,
            )
        };
        drop(ctx);
        if rc != 0 {
            let msg = unsafe { take_cstr(err) }.unwrap_or_else(|| "generate failed".into());
            return Err(BackendError::Internal(msg));
        }
        Ok(GenerateStats {
            tokens,
            decode_tps: tps,
        })
    }
}

#[cfg(target_os = "macos")]
impl Drop for MlxEngine {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.inner.lock() {
            if let Some(session) = guard.take() {
                unsafe { mlx_engine_destroy(session.handle) };
            }
        }
    }
}
