//! Minimal CUDA runtime bindings: device enumeration, streams, device and
//! pinned host memory, copies, and the provider's kernels. Every call checks
//! its error code; nothing is assumed to succeed.

use std::ffi::{c_char, c_int, c_void, CStr};
use std::ptr::NonNull;

use turbo_core::error::{Error, Result};

/// Raw `cudaStream_t`.
pub type CudaStream = *mut c_void;

#[repr(C)]
#[derive(Clone, Copy)]
struct CudaDeviceProp {
    name: [c_char; 256],
    uuid: [u8; 16],
    luid: [c_char; 8],
    luid_device_node_mask: u32,
    total_global_mem: usize,
    shared_mem_per_block: usize,
    regs_per_block: c_int,
    warp_size: c_int,
    mem_pitch: usize,
    max_threads_per_block: c_int,
    max_threads_dim: [c_int; 3],
    max_grid_size: [c_int; 3],
    clock_rate: c_int,
    total_const_mem: usize,
    major: c_int,
    minor: c_int,
    // The struct continues with many more fields; we allocate a generous
    // buffer and read only the leading fields above, whose layout has been
    // stable across CUDA 11, 12, and 13.
    _tail: [u8; 4096],
}

extern "C" {
    fn cudaGetDeviceCount(count: *mut c_int) -> c_int;
    fn cudaGetDeviceProperties_v2(prop: *mut CudaDeviceProp, device: c_int) -> c_int;
    fn cudaSetDevice(device: c_int) -> c_int;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> c_int;
    fn cudaRuntimeGetVersion(v: *mut c_int) -> c_int;
    fn cudaDriverGetVersion(v: *mut c_int) -> c_int;
    fn cudaStreamCreateWithFlags(stream: *mut CudaStream, flags: u32) -> c_int;
    fn cudaStreamDestroy(stream: CudaStream) -> c_int;
    fn cudaStreamSynchronize(stream: CudaStream) -> c_int;
    fn cudaMalloc(ptr: *mut *mut c_void, bytes: usize) -> c_int;
    fn cudaFree(ptr: *mut c_void) -> c_int;
    fn cudaHostAlloc(ptr: *mut *mut c_void, bytes: usize, flags: u32) -> c_int;
    fn cudaFreeHost(ptr: *mut c_void) -> c_int;
    fn cudaMemsetAsync(ptr: *mut c_void, value: c_int, bytes: usize, stream: CudaStream) -> c_int;
    fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, bytes: usize, kind: c_int, stream: CudaStream) -> c_int;
    fn cudaGetErrorString(e: c_int) -> *const c_char;

    /// Pool `hidden` over `mask` into `out` (mode 0 mean, 1 cls, 2 last), optionally L2-normalized.
    pub fn turbo_cuda_pool(
        hidden: *const f32,
        mask: *const c_void,
        mask_width: c_int,
        out: *mut f32,
        batch: c_int,
        seq: c_int,
        dim: c_int,
        out_dim: c_int,
        mode: c_int,
        normalize: c_int,
        stream: CudaStream,
    ) -> c_int;
    /// Element-wise sigmoid.
    pub fn turbo_cuda_sigmoid(input: *const f32, out: *mut f32, n: c_int, stream: CudaStream) -> c_int;
    /// Row-wise softmax over `width` logits.
    pub fn turbo_cuda_softmax_rows(
        input: *const f32,
        out: *mut f32,
        rows: c_int,
        width: c_int,
        stream: CudaStream,
    ) -> c_int;
}

const CUDA_MEMCPY_H2D: c_int = 1;
const CUDA_MEMCPY_D2H: c_int = 2;
const CUDA_MEMCPY_D2D: c_int = 3;
const CUDA_STREAM_NON_BLOCKING: u32 = 1;
const CUDA_HOST_ALLOC_DEFAULT: u32 = 0;

/// Convert a CUDA status into an error naming the operation.
pub fn check(code: c_int, what: &str) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    // SAFETY: cudaGetErrorString returns a static string for any code.
    let msg = unsafe { CStr::from_ptr(cudaGetErrorString(code)) }.to_string_lossy().into_owned();
    Err(Error::runtime(format!("CUDA {what}: {msg} (code {code})")))
}

/// Static description of one CUDA device.
#[derive(Clone, Debug)]
pub struct DeviceProps {
    /// CUDA device ordinal.
    pub index: i32,
    /// Device name.
    pub name: String,
    /// Total memory in bytes.
    pub total_mem: u64,
    /// Free memory at probe time.
    pub free_mem: u64,
    /// Compute capability major.
    pub major: i32,
    /// Compute capability minor.
    pub minor: i32,
    /// `cudaRuntimeGetVersion`.
    pub runtime_version: i32,
    /// `cudaDriverGetVersion`.
    pub driver_version: i32,
}

/// Enumerate devices. A runtime probe failure (no driver, no device) is an error.
pub fn devices() -> Result<Vec<DeviceProps>> {
    let mut n: c_int = 0;
    // SAFETY: plain FFI with an out-pointer.
    check(unsafe { cudaGetDeviceCount(&mut n) }, "cudaGetDeviceCount")?;
    let mut rt = 0;
    let mut drv = 0;
    check(unsafe { cudaRuntimeGetVersion(&mut rt) }, "cudaRuntimeGetVersion")?;
    check(unsafe { cudaDriverGetVersion(&mut drv) }, "cudaDriverGetVersion")?;
    let mut out = Vec::with_capacity(n as usize);
    for i in 0..n {
        let mut prop: CudaDeviceProp = unsafe { std::mem::zeroed() };
        check(unsafe { cudaGetDeviceProperties_v2(&mut prop, i) }, "cudaGetDeviceProperties")?;
        let name = unsafe { CStr::from_ptr(prop.name.as_ptr()) }.to_string_lossy().into_owned();
        let (mut free, mut total) = (0usize, 0usize);
        check(unsafe { cudaSetDevice(i) }, "cudaSetDevice")?;
        check(unsafe { cudaMemGetInfo(&mut free, &mut total) }, "cudaMemGetInfo")?;
        out.push(DeviceProps {
            index: i,
            name,
            total_mem: total as u64,
            free_mem: free as u64,
            major: prop.major,
            minor: prop.minor,
            runtime_version: rt,
            driver_version: drv,
        });
    }
    Ok(out)
}

/// Make `device` current on this thread.
pub fn set_device(device: i32) -> Result<()> {
    check(unsafe { cudaSetDevice(device) }, "cudaSetDevice")
}

/// Free device memory in bytes right now.
pub fn free_memory(device: i32) -> Result<u64> {
    set_device(device)?;
    let (mut free, mut total) = (0usize, 0usize);
    check(unsafe { cudaMemGetInfo(&mut free, &mut total) }, "cudaMemGetInfo")?;
    Ok(free as u64)
}

/// Render a CUDA version integer (for example 13020) as `13.2`.
pub fn version_string(v: i32) -> String {
    format!("{}.{}", v / 1000, (v % 1000) / 10)
}

/// A CUDA stream owned by a context.
pub struct Stream {
    raw: CudaStream,
    device: i32,
}

// SAFETY: streams are used only under the core's single-owner session rule.
unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

impl Stream {
    /// Create a non-blocking stream on `device`.
    pub fn new(device: i32) -> Result<Self> {
        set_device(device)?;
        let mut raw: CudaStream = std::ptr::null_mut();
        check(unsafe { cudaStreamCreateWithFlags(&mut raw, CUDA_STREAM_NON_BLOCKING) }, "cudaStreamCreate")?;
        Ok(Self { raw, device })
    }

    /// Raw handle.
    pub fn raw(&self) -> CudaStream {
        self.raw
    }

    /// Owning device.
    pub fn device(&self) -> i32 {
        self.device
    }

    /// Block until every operation queued on the stream completes.
    pub fn synchronize(&self) -> Result<()> {
        check(unsafe { cudaStreamSynchronize(self.raw) }, "cudaStreamSynchronize")
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // SAFETY: created by cudaStreamCreateWithFlags; destroyed once.
        let _ = unsafe { cudaStreamDestroy(self.raw) };
    }
}

/// Device memory.
pub struct DeviceMem {
    ptr: NonNull<c_void>,
    bytes: usize,
}
unsafe impl Send for DeviceMem {}
unsafe impl Sync for DeviceMem {}

impl DeviceMem {
    /// Allocate `bytes` on `device`.
    pub fn new(device: i32, bytes: usize) -> Result<Self> {
        if bytes == 0 {
            return Err(Error::invalid_shape("zero-byte device allocation"));
        }
        set_device(device)?;
        let mut p: *mut c_void = std::ptr::null_mut();
        check(unsafe { cudaMalloc(&mut p, bytes) }, &format!("cudaMalloc({bytes} bytes)"))?;
        let ptr = NonNull::new(p).ok_or_else(|| Error::out_of_memory("cudaMalloc returned NULL"))?;
        Ok(Self { ptr, bytes })
    }

    /// Device pointer.
    pub fn ptr(&self) -> *mut c_void {
        self.ptr.as_ptr()
    }

    /// Size in bytes.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Asynchronously zero the allocation.
    pub fn zero(&self, stream: &Stream) -> Result<()> {
        check(unsafe { cudaMemsetAsync(self.ptr.as_ptr(), 0, self.bytes, stream.raw()) }, "cudaMemsetAsync")
    }
}

impl Drop for DeviceMem {
    fn drop(&mut self) {
        let _ = unsafe { cudaFree(self.ptr.as_ptr()) };
    }
}

/// Page-locked host memory.
pub struct PinnedMem {
    ptr: NonNull<u8>,
    bytes: usize,
}
unsafe impl Send for PinnedMem {}
unsafe impl Sync for PinnedMem {}

impl PinnedMem {
    /// Allocate `bytes` of page-locked host memory, zeroed.
    pub fn new(bytes: usize) -> Result<Self> {
        if bytes == 0 {
            return Err(Error::invalid_shape("zero-byte pinned allocation"));
        }
        let mut p: *mut c_void = std::ptr::null_mut();
        check(
            unsafe { cudaHostAlloc(&mut p, bytes, CUDA_HOST_ALLOC_DEFAULT) },
            &format!("cudaHostAlloc({bytes} bytes)"),
        )?;
        let ptr = NonNull::new(p.cast::<u8>()).ok_or_else(|| Error::out_of_memory("cudaHostAlloc returned NULL"))?;
        // SAFETY: freshly allocated, `bytes` long.
        unsafe { std::ptr::write_bytes(ptr.as_ptr(), 0, bytes) };
        Ok(Self { ptr, bytes })
    }

    /// Host pointer.
    pub fn ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Size in bytes.
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for PinnedMem {
    fn drop(&mut self) {
        let _ = unsafe { cudaFreeHost(self.ptr.as_ptr().cast()) };
    }
}

/// Asynchronous host-to-device copy on `stream`.
///
/// # Safety
/// `src` must be readable for `bytes` bytes until the stream has consumed it.
pub unsafe fn copy_h2d(dst: &DeviceMem, src: *const u8, bytes: usize, stream: &Stream) -> Result<()> {
    if bytes > dst.bytes() {
        return Err(Error::capacity(format!(
            "H2D copy of {bytes} bytes exceeds the {} byte device buffer",
            dst.bytes()
        )));
    }
    check(
        unsafe { cudaMemcpyAsync(dst.ptr(), src.cast(), bytes, CUDA_MEMCPY_H2D, stream.raw()) },
        "cudaMemcpyAsync H2D",
    )
}

/// Asynchronous device-to-host copy on `stream`.
///
/// # Safety
/// `dst` must be writable and `src` a device allocation, both for `bytes` bytes.
pub unsafe fn copy_d2h(dst: *mut u8, src: *const c_void, bytes: usize, stream: &Stream) -> Result<()> {
    check(unsafe { cudaMemcpyAsync(dst.cast(), src, bytes, CUDA_MEMCPY_D2H, stream.raw()) }, "cudaMemcpyAsync D2H")
}

/// Asynchronous device-to-device copy on `stream`.
///
/// # Safety
/// Both pointers must be device allocations of at least `bytes` bytes.
pub unsafe fn copy_d2d(dst: *mut c_void, src: *const c_void, bytes: usize, stream: &Stream) -> Result<()> {
    check(unsafe { cudaMemcpyAsync(dst, src, bytes, CUDA_MEMCPY_D2D, stream.raw()) }, "cudaMemcpyAsync D2D")
}
