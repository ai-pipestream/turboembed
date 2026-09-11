//! C ABI for `native/mlx-engine` (Swift mlx-swift + mlx-swift-lm on Metal).
//!
//! Only linked on macOS. Linux CI type-checks the rest of the crate against
//! the stub in [`crate::native`].

use std::os::raw::{c_char, c_int, c_void};

#[repr(C)]
pub struct MlxEngineHandle {
    _private: [u8; 0],
}

pub type MlxTokenCb = unsafe extern "C" fn(*const c_char, *mut c_void);

#[cfg(target_os = "macos")]
unsafe extern "C" {
    pub fn mlx_engine_create(err: *mut *mut c_char) -> *mut MlxEngineHandle;
    pub fn mlx_engine_destroy(engine: *mut MlxEngineHandle);
    pub fn mlx_engine_ping(
        engine: *mut MlxEngineHandle,
        out_json: *mut *mut c_char,
        err: *mut *mut c_char,
    ) -> c_int;
    pub fn mlx_engine_embed(
        engine: *mut MlxEngineHandle,
        model_path: *const c_char,
        texts: *const *const c_char,
        n_texts: usize,
        normalize: c_int,
        out_vectors: *mut *mut f32,
        out_rows: *mut usize,
        out_dim: *mut usize,
        err: *mut *mut c_char,
    ) -> c_int;
    pub fn mlx_engine_generate(
        engine: *mut MlxEngineHandle,
        model_path: *const c_char,
        prompt: *const c_char,
        max_tokens: u32,
        on_token: MlxTokenCb,
        user: *mut c_void,
        out_tokens: *mut u32,
        out_decode_tps: *mut f64,
        err: *mut *mut c_char,
    ) -> c_int;
    pub fn mlx_engine_free(ptr: *mut c_void);
    pub fn mlx_engine_free_str(ptr: *mut c_char);
}

pub unsafe fn take_cstr(ptr: *mut c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    let s = std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned();
    #[cfg(target_os = "macos")]
    mlx_engine_free_str(ptr);
    Some(s)
}
