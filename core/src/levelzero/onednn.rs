//! oneDNN's kernels for the F16 linear layers (the `levelzero-onednn`
//! feature): the C interface of core/levelzero/onednn.cpp, built by
//! core/build.rs with the oneAPI compiler, over the backend's own device
//! and context. The backend orders oneDNN's queue and its own command
//! list by host synchronisation (encoder.rs).

use std::ffi::{CStr, c_char, c_void};

use super::gpu::{Res, fail};
use crate::status::RUNTIME;

unsafe extern "C" {
    fn turbo_dnnl_open(ze_device: *mut c_void, ze_context: *mut c_void, err: *mut c_char, n: usize) -> *mut c_void;
    fn turbo_dnnl_close(h: *mut c_void);
    fn turbo_dnnl_weights_bytes(
        h: *mut c_void,
        k: i32,
        n: i32,
        bytes: *mut usize,
        err: *mut c_char,
        n_err: usize,
    ) -> i32;
    fn turbo_dnnl_pack(
        h: *mut c_void,
        src: *const c_void,
        k: i32,
        n: i32,
        dst: *mut c_void,
        err: *mut c_char,
        n_err: usize,
    ) -> i32;
    fn turbo_dnnl_matmul(
        h: *mut c_void,
        a: *const c_void,
        packed: *const c_void,
        bias: *const f32,
        residual: *const c_void,
        c: *mut c_void,
        m: i32,
        k: i32,
        n: i32,
        gelu: i32,
        ze_wait: *mut c_void,
        ze_signal: *mut *mut c_void,
        err: *mut c_char,
        n_err: usize,
    ) -> i32;
    fn turbo_dnnl_layer_norm(
        h: *mut c_void,
        src: *const c_void,
        gamma: *const f32,
        beta: *const f32,
        eps: f32,
        dst: *mut c_void,
        m: i32,
        n: i32,
        ze_wait: *mut c_void,
        ze_signal: *mut *mut c_void,
        err: *mut c_char,
        n_err: usize,
    ) -> i32;
    fn turbo_dnnl_wait(h: *mut c_void, err: *mut c_char, n_err: usize) -> i32;
    fn turbo_dnnl_version() -> *const c_char;
}

/// c [m, n] F16 = a [m, k] F16 times the weights packed by `pack`, plus
/// bias (F32, n), plus residual [m, n] F16 where not 0, then GELU where
/// asked. Device addresses.
pub(crate) struct Matmul {
    pub a: u64,
    pub packed: u64,
    pub bias: u64,
    pub residual: u64,
    pub c: u64,
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub gelu: bool,
    /// The backend's event to run after, or 0.
    pub wait: u64,
}

/// dst [m, n] F16 = LayerNorm of src over n with gamma and beta (F32, n).
pub(crate) struct LayerNorm {
    pub src: u64,
    pub gamma: u64,
    pub beta: u64,
    pub eps: f32,
    pub dst: u64,
    pub m: u32,
    pub n: u32,
    /// The event to run after, or 0.
    pub wait: u64,
}

/// oneDNN on one context: a SYCL queue of its own on the backend's Level
/// Zero context, and the primitives it has built.
pub(crate) struct Dnnl(*mut c_void);

// The handle is used from any thread under the context's queue lock.
unsafe impl Send for Dnnl {}
unsafe impl Sync for Dnnl {}

const ERR: usize = 512;

fn check(what: &str, rc: i32, err: &[c_char; ERR]) -> Res<()> {
    if rc == 0 {
        return Ok(());
    }
    let text = unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy().into_owned();
    Err(fail(RUNTIME, format!("levelzero: {what}: {text}")))
}

impl Dnnl {
    /// Opened on the backend's device and context; Err says why not.
    pub fn open(ze_device: *mut c_void, ze_context: *mut c_void) -> Res<Dnnl> {
        let mut err = [0 as c_char; ERR];
        let h = unsafe { turbo_dnnl_open(ze_device, ze_context, err.as_mut_ptr(), ERR) };
        if h.is_null() {
            check("oneDNN open", 1, &err)?;
        }
        Ok(Dnnl(h))
    }

    pub fn version() -> String {
        unsafe { CStr::from_ptr(turbo_dnnl_version()) }.to_string_lossy().into_owned()
    }

    /// Bytes the packed weights of a [k, n] F16 matrix take.
    pub fn weights_bytes(&self, k: u32, n: u32) -> Res<usize> {
        let (mut bytes, mut err) = (0usize, [0 as c_char; ERR]);
        let rc = unsafe { turbo_dnnl_weights_bytes(self.0, k as i32, n as i32, &mut bytes, err.as_mut_ptr(), ERR) };
        check("oneDNN weights size", rc, &err)?;
        Ok(bytes)
    }

    /// `src`, [k, n] F16 row-major on the device, packed into `dst`; waits.
    pub fn pack(&self, src: u64, k: u32, n: u32, dst: u64) -> Res<()> {
        let mut err = [0 as c_char; ERR];
        let rc = unsafe {
            turbo_dnnl_pack(self.0, src as *const c_void, k as i32, n as i32, dst as *mut c_void, err.as_mut_ptr(), ERR)
        };
        check("oneDNN pack", rc, &err)
    }

    /// Queues a matmul after `wait`; the event it signals, valid until
    /// `wait` is called.
    pub fn matmul(&self, mm: &Matmul) -> Res<u64> {
        let (mut signal, mut err) = (std::ptr::null_mut::<c_void>(), [0 as c_char; ERR]);
        let rc = unsafe {
            turbo_dnnl_matmul(
                self.0,
                mm.a as *const c_void,
                mm.packed as *const c_void,
                mm.bias as *const f32,
                mm.residual as *const c_void,
                mm.c as *mut c_void,
                mm.m as i32,
                mm.k as i32,
                mm.n as i32,
                mm.gelu as i32,
                mm.wait as *mut c_void,
                &mut signal,
                err.as_mut_ptr(),
                ERR,
            )
        };
        check("oneDNN matmul", rc, &err)?;
        Ok(signal as u64)
    }

    /// Queues a LayerNorm after `wait`; the event it signals.
    pub fn layer_norm(&self, ln: &LayerNorm) -> Res<u64> {
        let (mut signal, mut err) = (std::ptr::null_mut::<c_void>(), [0 as c_char; ERR]);
        let rc = unsafe {
            turbo_dnnl_layer_norm(
                self.0,
                ln.src as *const c_void,
                ln.gamma as *const f32,
                ln.beta as *const f32,
                ln.eps,
                ln.dst as *mut c_void,
                ln.m as i32,
                ln.n as i32,
                ln.wait as *mut c_void,
                &mut signal,
                err.as_mut_ptr(),
                ERR,
            )
        };
        check("oneDNN layer norm", rc, &err)?;
        Ok(signal as u64)
    }

    /// Waits for everything queued, and drops the events handed out.
    pub fn wait(&self) -> Res<()> {
        let mut err = [0 as c_char; ERR];
        let rc = unsafe { turbo_dnnl_wait(self.0, err.as_mut_ptr(), ERR) };
        check("oneDNN wait", rc, &err)
    }
}

impl Drop for Dnnl {
    fn drop(&mut self) {
        unsafe { turbo_dnnl_close(self.0) };
    }
}
