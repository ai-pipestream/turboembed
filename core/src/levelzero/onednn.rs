//! oneDNN's kernels for the F16 linear layers (the `levelzero-onednn`
//! feature): the C interface of core/levelzero/onednn.cpp, built by
//! core/build.rs with the oneAPI compiler, over the backend's own device
//! and context. The backend orders oneDNN's queue and its own command
//! list by host synchronisation (encoder/mod.rs).

use std::ffi::{CStr, c_char, c_void};
use std::sync::OnceLock;

use super::gpu::{Res, fail};
use crate::status::RUNTIME;

/// The C interface of libturbo_onednn.so, opened once (`api`).
struct Api {
    // Kept open for the life of the process: the pointers below are into it.
    _lib: libloading::Library,
    open: unsafe extern "C" fn(
        ze_device: *mut c_void,
        ze_context: *mut c_void,
        err: *mut c_char,
        n: usize,
    ) -> *mut c_void,
    close: unsafe extern "C" fn(h: *mut c_void),
    weights_bytes:
        unsafe extern "C" fn(h: *mut c_void, k: i32, n: i32, bytes: *mut usize, err: *mut c_char, n_err: usize) -> i32,
    pack: unsafe extern "C" fn(
        h: *mut c_void,
        src: *const c_void,
        k: i32,
        n: i32,
        dst: *mut c_void,
        err: *mut c_char,
        n_err: usize,
    ) -> i32,
    matmul: unsafe extern "C" fn(
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
        f32: i32,
        ze_wait: *mut c_void,
        ze_signal: *mut *mut c_void,
        built: *mut i32,
        err: *mut c_char,
        n_err: usize,
    ) -> i32,
    layer_norm: unsafe extern "C" fn(
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
        built: *mut i32,
        err: *mut c_char,
        n_err: usize,
    ) -> i32,
    wait: unsafe extern "C" fn(h: *mut c_void, err: *mut c_char, n_err: usize) -> i32,
    version: unsafe extern "C" fn() -> *const c_char,
}

/// Where core/build.rs built the library. A copy found through the
/// dynamic linker's search (`LD_LIBRARY_PATH`, the run-time path) is the
/// fallback, for a binary moved off the machine that built it.
const BUILT: &str = env!("TURBO_ONEDNN_LIB");
const NAME: &str = "libturbo_onednn.so";

static API: OnceLock<Result<Api, String>> = OnceLock::new();

/// The library's functions, opened the first time; Err says why not.
fn api() -> Res<&'static Api> {
    API.get_or_init(open_api).as_ref().map_err(|e| fail(RUNTIME, e.clone()))
}

fn open_api() -> Result<Api, String> {
    let lib = match unsafe { libloading::Library::new(BUILT) } {
        Ok(lib) => lib,
        Err(built) => unsafe { libloading::Library::new(NAME) }.map_err(|found| {
            format!("levelzero: oneDNN: cannot open {BUILT} ({built}) or {NAME} on the library path ({found})")
        })?,
    };
    macro_rules! sym {
        ($name:literal) => {{
            let s = unsafe { lib.get(concat!($name, "\0").as_bytes()) }
                .map_err(|e| format!("levelzero: oneDNN: {NAME} has no {}: {e}", $name))?;
            *s
        }};
    }
    Ok(Api {
        open: sym!("turbo_dnnl_open"),
        close: sym!("turbo_dnnl_close"),
        weights_bytes: sym!("turbo_dnnl_weights_bytes"),
        pack: sym!("turbo_dnnl_pack"),
        matmul: sym!("turbo_dnnl_matmul"),
        layer_norm: sym!("turbo_dnnl_layer_norm"),
        wait: sym!("turbo_dnnl_wait"),
        version: sym!("turbo_dnnl_version"),
        _lib: lib,
    })
}

/// c [m, n] = a [m, k] times the weights, plus bias (F32, n) where not 0,
/// plus residual [m, n] where not 0, then GELU where asked: in F16 from the
/// weights packed by `pack`, or with `f32` in F32 from a linear layer's
/// own [n, k] weights. Device addresses.
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
    pub f32: bool,
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
/// Zero context, and the primitives it has built, one per shape, its tokens
/// binned (encoder/mod.rs).
pub(crate) struct Dnnl {
    h: *mut c_void,
    api: &'static Api,
}

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
        let api = api()?;
        let mut err = [0 as c_char; ERR];
        let h = unsafe { (api.open)(ze_device, ze_context, err.as_mut_ptr(), ERR) };
        if h.is_null() {
            check("oneDNN open", 1, &err)?;
        }
        Ok(Dnnl { h, api })
    }

    /// oneDNN's version, as the opened library reports it.
    pub fn version(&self) -> String {
        unsafe { CStr::from_ptr((self.api.version)()) }.to_string_lossy().into_owned()
    }

    /// Bytes the packed weights of a [k, n] F16 matrix take.
    pub fn weights_bytes(&self, k: u32, n: u32) -> Res<usize> {
        let (mut bytes, mut err) = (0usize, [0 as c_char; ERR]);
        let rc = unsafe { (self.api.weights_bytes)(self.h, k as i32, n as i32, &mut bytes, err.as_mut_ptr(), ERR) };
        check("oneDNN weights size", rc, &err)?;
        Ok(bytes)
    }

    /// `src`, [k, n] F16 row-major on the device, packed into `dst`; waits.
    pub fn pack(&self, src: u64, k: u32, n: u32, dst: u64) -> Res<()> {
        let mut err = [0 as c_char; ERR];
        let rc = unsafe {
            (self.api.pack)(self.h, src as *const c_void, k as i32, n as i32, dst as *mut c_void, err.as_mut_ptr(), ERR)
        };
        check("oneDNN pack", rc, &err)
    }

    /// Queues a matmul after `wait`; the event it signals, valid until
    /// `wait` is called, and whether its primitive was built for it.
    pub fn matmul(&self, mm: &Matmul) -> Res<(u64, bool)> {
        let (mut signal, mut built, mut err) = (std::ptr::null_mut::<c_void>(), 0, [0 as c_char; ERR]);
        let rc = unsafe {
            (self.api.matmul)(
                self.h,
                mm.a as *const c_void,
                mm.packed as *const c_void,
                mm.bias as *const f32,
                mm.residual as *const c_void,
                mm.c as *mut c_void,
                mm.m as i32,
                mm.k as i32,
                mm.n as i32,
                mm.gelu as i32,
                mm.f32 as i32,
                mm.wait as *mut c_void,
                &mut signal,
                &mut built,
                err.as_mut_ptr(),
                ERR,
            )
        };
        check("oneDNN matmul", rc, &err)?;
        Ok((signal as u64, built != 0))
    }

    /// Queues a LayerNorm after `wait`; the event it signals, and whether
    /// its primitive was built for it.
    pub fn layer_norm(&self, ln: &LayerNorm) -> Res<(u64, bool)> {
        let (mut signal, mut built, mut err) = (std::ptr::null_mut::<c_void>(), 0, [0 as c_char; ERR]);
        let rc = unsafe {
            (self.api.layer_norm)(
                self.h,
                ln.src as *const c_void,
                ln.gamma as *const f32,
                ln.beta as *const f32,
                ln.eps,
                ln.dst as *mut c_void,
                ln.m as i32,
                ln.n as i32,
                ln.wait as *mut c_void,
                &mut signal,
                &mut built,
                err.as_mut_ptr(),
                ERR,
            )
        };
        check("oneDNN layer norm", rc, &err)?;
        Ok((signal as u64, built != 0))
    }

    /// Waits for everything queued, and drops the events handed out.
    pub fn wait(&self) -> Res<()> {
        let mut err = [0 as c_char; ERR];
        let rc = unsafe { (self.api.wait)(self.h, err.as_mut_ptr(), ERR) };
        check("oneDNN wait", rc, &err)
    }
}

impl Drop for Dnnl {
    fn drop(&mut self) {
        unsafe { (self.api.close)(self.h) };
    }
}
