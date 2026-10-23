//! What the CUDA test files share: the CUDA runtime calls they make from
//! the caller's side of a pointer, the turn lock, the runtime and device
//! helpers, and buffers in every placement.

#![allow(dead_code)]

pub mod nvml;
pub mod rows;

use std::ffi::c_void;
use std::ptr;
use std::sync::{Mutex, MutexGuard};

use crate::common::*;
#[allow(unused_imports)]
pub use rows::*;
use turbo::*;

// The CUDA runtime the library links, for what the tests check from the
// caller's side of a pointer.
unsafe extern "C" {
    pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    pub fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
    pub fn cudaSetDevice(device: i32) -> i32;
    pub fn cudaDeviceGetPCIBusId(bus: *mut std::ffi::c_char, len: i32, device: i32) -> i32;
    pub fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
    pub fn cudaFree(ptr: *mut c_void) -> i32;
}

pub const HOST_TO_DEVICE: i32 = 1;

pub const DEVICE_TO_HOST: i32 = 2;

/// The tests take turns: the allocation counts and the device's free
/// memory are the process's, and another test's work would move them.
pub static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

pub fn turn() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner())
}

/// f with MODEL's GEMMs pinned to the FMA kernels, whatever
/// TURBO_CUDA_TF32 says, for the sessions f makes: what a test holding
/// MODEL to F32's bound (or to EXACT's bits) needs.
pub fn strict<T>(f: impl FnOnce() -> T) -> T {
    turbo::cuda::use_tf32(Some(false));
    let r = f();
    turbo::cuda::use_tf32(None);
    r
}

pub fn null_err() -> *mut turbo_error {
    ptr::null_mut()
}

pub const TEXTS: [&str; 3] = ["The quick brown fox jumps over the lazy dog.", "how do I reset a password", "a"];

pub struct Rt(pub *mut turbo_runtime);

impl Rt {
    pub fn new() -> Rt {
        let mut rt = ptr::null_mut();
        assert_eq!(unsafe { turbo_runtime_create(ptr::null(), &mut rt, null_err()) }, 0);
        Rt(rt)
    }

    pub fn info(&self, i: u32) -> turbo_device_info {
        let mut out: turbo_device_info = unsafe { std::mem::zeroed() };
        out.struct_size = size_of::<turbo_device_info>() as u32;
        let mut err = new_error();
        let rc = unsafe { turbo_runtime_device_info(self.0, i, &mut out, &mut err) };
        assert_eq!(rc, 0, "{:?}", failure(rc, &err));
        out
    }
}

impl Drop for Rt {
    fn drop(&mut self) {
        unsafe { turbo_runtime_release(self.0) };
    }
}

/// TURBO_TEST_REQUIRE_CUDA=1: a test that finds no device fails.
pub fn required() -> bool {
    std::env::var("TURBO_TEST_REQUIRE_CUDA").is_ok_and(|v| v == "1")
}

/// The first CUDA device, or None after saying the test is skipped.
pub fn cuda_device(test: &str) -> Option<u32> {
    let rt = Rt::new();
    let d = first_of(rt.0, "cuda");
    if d.is_none() {
        assert!(!required(), "{test}: TURBO_TEST_REQUIRE_CUDA=1 and the cuda backend lists no device");
        println!("{test}: skipped: the cuda feature is on and the cuda backend lists no device");
    }
    d
}

/// Whether this build links cuBLAS (the `cuda-cublas` feature), after
/// saying that `what` is skipped when it does not: a check against
/// cuBLAS, or a GEMM handed to it, needs that build.
pub fn cublas_or_skip(test: &str, what: &str) -> bool {
    let has = turbo::cuda::has_cublas();
    if !has {
        println!("{test}: {what} skipped: this build has no cuBLAS (the cuda-cublas feature)");
    }
    has
}

pub fn cuda(rt: *mut turbo_runtime) -> u32 {
    first_of(rt, "cuda").expect("a cuda device")
}

pub fn on_cuda(dir: &std::path::Path) -> Loaded {
    Loaded::load_on(dir, cuda).unwrap_or_else(|e| panic!("{e:?}"))
}

pub fn opts(edit: impl FnOnce(&mut turbo_embed_options)) -> turbo_embed_options {
    let mut o = embed_options();
    edit(&mut o);
    o
}

/// A context, released on drop when `owned`.
pub struct Ctx(pub *mut turbo_context, pub bool);

impl Ctx {
    pub fn alloc(&self, placement: u32, n: u64) -> Result<Buf, Failure> {
        let d = desc(placement, n);
        let mut b = ptr::null_mut();
        let mut err = new_error();
        match unsafe { turbo_buffer_alloc(self.0, &d, &mut b, &mut err) } {
            0 => Ok(Buf(b)),
            rc => Err(failure(rc, &err)),
        }
    }

    pub fn import(&self, placement: u32, n: u64, h: turbo_native_handle) -> Result<Buf, Failure> {
        let d = desc(placement, n);
        let mut b = ptr::null_mut();
        let mut err = new_error();
        match unsafe { turbo_buffer_import(self.0, &d, &h, &mut b, &mut err) } {
            0 => Ok(Buf(b)),
            rc => Err(failure(rc, &err)),
        }
    }
}

impl Drop for Ctx {
    fn drop(&mut self) {
        if self.1 {
            unsafe { turbo_context_release(self.0) };
        }
    }
}

#[derive(Debug)]
pub struct Buf(pub *mut turbo_buffer);

impl Buf {
    pub fn host(&self) -> Result<*mut c_void, Failure> {
        let mut p = ptr::null_mut();
        let mut err = new_error();
        match unsafe { turbo_buffer_host_ptr(self.0, &mut p, &mut err) } {
            0 => Ok(p),
            rc => Err(failure(rc, &err)),
        }
    }

    pub fn export(&self, kind: u32) -> Result<turbo_native_handle, Failure> {
        let mut h: turbo_native_handle = unsafe { std::mem::zeroed() };
        h.struct_size = size_of::<turbo_native_handle>() as u32;
        let mut err = new_error();
        match unsafe { turbo_buffer_export(self.0, kind, &mut h, &mut err) } {
            0 => Ok(h),
            rc => Err(failure(rc, &err)),
        }
    }
}

impl Drop for Buf {
    fn drop(&mut self) {
        unsafe { turbo_buffer_release(self.0) };
    }
}

/// n F32 values in one row.
pub fn desc(placement: u32, n: u64) -> turbo_buffer_desc {
    turbo_buffer_desc {
        struct_size: size_of::<turbo_buffer_desc>() as u32,
        placement,
        dtype: TURBO_DTYPE_F32,
        ndim: 1,
        shape: [n, 0],
        bytes: 0,
    }
}

pub fn handle(kind: u32, at: u64, aux: u64, offset: u64) -> turbo_native_handle {
    turbo_native_handle { struct_size: size_of::<turbo_native_handle>() as u32, kind, handle: at, aux, offset }
}

pub fn context(rt: &Rt, device: u32) -> Ctx {
    let mut c = ptr::null_mut();
    let mut err = new_error();
    let rc = unsafe { turbo_context_create(rt.0, device, &mut c, &mut err) };
    assert_eq!(rc, 0, "{:?}", failure(rc, &err));
    Ctx(c, true)
}

pub fn to_device(dst: u64, src: &[f32]) {
    assert_eq!(
        unsafe { cudaMemcpy(dst as *mut c_void, src.as_ptr() as *const c_void, src.len() * 4, HOST_TO_DEVICE) },
        0
    );
}

pub fn from_device(src: u64, n: usize) -> Vec<f32> {
    let mut v = vec![f32::NAN; n];
    assert_eq!(unsafe { cudaMemcpy(v.as_mut_ptr() as *mut c_void, src as *const c_void, n * 4, DEVICE_TO_HOST) }, 0);
    v
}
