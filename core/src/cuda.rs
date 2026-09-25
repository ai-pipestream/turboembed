//! The CUDA backend: NVIDIA GPUs through the CUDA runtime. It
//! is written in C++ and CUDA in core/cuda/, compiled by build.rs into a
//! static library, and reached only through its turbo_backend table, like
//! any other backend. docs/cuda.md says how to build and test it.

use crate::backend::turbo_backend;

unsafe extern "C" {
    /// The table core/cuda/backend.cpp fills.
    pub(crate) static turbo_cuda_backend: turbo_backend;
}

/// The backend's table.
pub fn backend() -> &'static turbo_backend {
    // A table the C++ side fills at compile time and never writes.
    unsafe { &turbo_cuda_backend }
}

#[cfg(feature = "internals")]
unsafe extern "C" {
    fn turbo_cuda_allocations(host: *mut u64, device: *mut u64);
    fn turbo_cuda_arch_label(name: *const std::ffi::c_char, out: *mut std::ffi::c_char, len: usize);
    fn turbo_cuda_widened(model: *mut std::ffi::c_void) -> *const std::ffi::c_void;
    fn turbo_cuda_narrowed(model: *mut std::ffi::c_void) -> *const std::ffi::c_void;
    fn turbo_cuda_gemm_check(
        ordinal: u32,
        m: i32,
        n: i32,
        k: i32,
        epilogue: i32,
        half: i32,
        tensor_cores: i32,
        tile: i32,
        blocks: i32,
        heads: i32,
        max_diff: *mut f64,
        max_ref: *mut f64,
    ) -> i32;
    fn turbo_cuda_use_cublas(gemms: i32);
    fn turbo_cuda_use_tile(tile: i32);
    fn turbo_cuda_use_split_attention(split: i32);
    fn turbo_cuda_use_wide_attention(wide: i32);
    fn turbo_cuda_use_separate_layer_norm(separate: i32);
    fn turbo_cuda_use_column_pool(columns: i32);
    fn turbo_cuda_use_tf32(tf32: i32);
    fn turbo_cuda_use_f16_accumulate(f16: i32);
}

/// The epilogues of the backend's own GEMMs, as [`gemm_check`] names them.
#[cfg(feature = "internals")]
#[derive(Clone, Copy, Debug)]
pub enum Epilogue {
    /// + bias, written head-major for attention.
    Qkv = 0,
    /// + bias, then GELU with the error function.
    Gelu = 1,
    /// The bare product, F32.
    Plain = 2,
}

/// The GEMMs' tiles, rows by columns, as TURBO_CUDA_TILE names them.
#[cfg(feature = "internals")]
#[derive(Clone, Copy, Debug)]
pub enum Tile {
    /// The backend's own choice.
    Default = 0,
    /// 64 x 64.
    T64x64 = 1,
    /// 128 x 64.
    T128x64 = 2,
    /// 128 x 128.
    T128x128 = 3,
    /// 128 x 128 with 16 x 8 outputs to a thread for the FMA kernel;
    /// 128 x 128 on the tensor cores.
    T128x128Thread16x8 = 4,
    /// On the tensor cores 128 x 128 over four warps of 64 x 64; 128 x 64
    /// for the FMA kernel.
    T128x128Warps4 = 5,
    /// On the tensor cores 256 x 128 over eight warps of 64 x 64 (128 x 128
    /// over four for an F32 output); 128 x 64 for the FMA kernel.
    T256x128 = 6,
    /// The tensor cores' eight-warp tiles, FASTEST's default: 128 x 128
    /// for QKV and GELU, 128 x 64 for the others.
    EightWarps = 7,
    /// The eight-warp tiles with F16 accumulators over each 64 terms of
    /// k, added into F32 ones; F16 operands only.
    EightWarpsF16Accumulate = 8,
    /// F16 operands on the swizzled kernel (stage rows of 64 bytes, loads
    /// running ahead across tiles, the epilogue from registers): 128 x 128
    /// over four warps of 64 x 64, two blocks to an SM.
    Swizzled = 9,
    /// The swizzled kernel at the eight-warp mix's shapes.
    SwizzledEightWarps = 10,
    /// `Swizzled`, but 256 x 128 over eight warps for GELU.
    Swizzled256x128 = 11,
    /// `SwizzledEightWarps` with F16 accumulators over each 64 terms of k.
    SwizzledEightWarpsF16Accumulate = 12,
    /// `SwizzledEightWarps`, but the attention output and second
    /// feed-forward GEMMs 64 x 384, whole rows, with the residual and the
    /// LayerNorm in their epilogue (hidden widths up to 384, F16).
    SwizzledRows = 13,
}

/// One GEMM of the CUDA backend's own, `[m, k]` by `[n, k]`, on random
/// operands on CUDA device `ordinal`, against cuBLAS's product with the
/// epilogue done on the host: the largest absolute difference and the
/// largest reference value. `half` takes F16 operands; `tensor_cores`
/// computes on the tensor cores (F16, or F32 operands as TF32), else with
/// FMAs; `tile` is the GEMM's tile;
/// `blocks` the launch's blocks, which share the work (0 for as many as
/// the device holds at once, and never more); `heads` the QKV epilogue's,
/// n being three times the hidden width. The GEMM runs twice and must
/// repeat its bits. Built only with `internals`.
#[cfg(feature = "internals")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_check(
    ordinal: u32,
    m: i32,
    n: i32,
    k: i32,
    epilogue: Epilogue,
    half: bool,
    tensor_cores: bool,
    tile: Tile,
    blocks: i32,
    heads: i32,
) -> Result<(f64, f64), i32> {
    let (mut diff, mut reference) = (0.0, 0.0);
    let rc = unsafe {
        turbo_cuda_gemm_check(
            ordinal,
            m,
            n,
            k,
            epilogue as i32,
            i32::from(half),
            i32::from(tensor_cores),
            tile as i32,
            blocks,
            heads,
            &mut diff,
            &mut reference,
        )
    };
    if rc == 0 { Ok((diff, reference)) } else { Err(rc) }
}

/// The GEMMs sessions made from now on compute with cuBLAS, as
/// TURBO_CUDA_CUBLAS names them (1 QKV, 2 attention output, 4 feed-forward
/// input, 8 feed-forward output), or `None` to read the variable again.
/// Built only with `internals`.
#[cfg(feature = "internals")]
pub fn use_cublas(gemms: Option<u32>) {
    unsafe { turbo_cuda_use_cublas(gemms.map_or(-1, |g| g as i32)) };
}

/// Every allocation the CUDA backend has made in this process, host and
/// device, counted where it makes them. Built only with `internals`.
#[cfg(feature = "internals")]
pub fn allocations() -> (u64, u64) {
    let (mut host, mut device) = (0, 0);
    unsafe { turbo_cuda_allocations(&mut host, &mut device) };
    (host, device)
}

/// The arch label a device of this name is listed with. Built only with
/// `internals`.
#[cfg(feature = "internals")]
pub fn arch_label(name: &str) -> String {
    let name = std::ffi::CString::new(name).unwrap();
    let mut out = [0 as std::ffi::c_char; 32];
    unsafe { turbo_cuda_arch_label(name.as_ptr(), out.as_mut_ptr(), out.len()) };
    crate::backend::cstr(&out)
}

/// The device address of the F32 copy of an F16 or BF16 model's weights,
/// once a session made it.
///
/// # Safety
/// `model` is one this backend's model_load returned, not yet released.
#[cfg(feature = "internals")]
pub(crate) unsafe fn widened(model: *mut std::ffi::c_void) -> Option<*const std::ffi::c_void> {
    let p = unsafe { turbo_cuda_widened(model) };
    (!p.is_null()).then_some(p)
}

/// The device address of the F16 copy of an F32 or BF16 model's GEMM
/// weights, once an F16 session made it.
///
/// # Safety
/// `model` is one this backend's model_load returned, not yet released.
#[cfg(feature = "internals")]
pub(crate) unsafe fn narrowed(model: *mut std::ffi::c_void) -> Option<*const std::ffi::c_void> {
    let p = unsafe { turbo_cuda_narrowed(model) };
    (!p.is_null()).then_some(p)
}

/// The GEMMs' tile in sessions made from now on, as TURBO_CUDA_TILE names
/// it, or `None` to read the variable again. Built only with `internals`.
#[cfg(feature = "internals")]
pub fn use_tile(tile: Option<Tile>) {
    unsafe { turbo_cuda_use_tile(tile.map_or(-1, |t| t as i32)) };
}

/// The FMA attention of sessions made from now on: `Some(true)` the
/// kernel that splits each query's keys among four warps, as
/// TURBO_CUDA_ATTENTION=split picks it, `Some(false)` the default, `None`
/// to read the variable again. Built only with `internals`.
#[cfg(feature = "internals")]
pub fn use_split_attention(split: Option<bool>) {
    unsafe { turbo_cuda_use_split_attention(split.map_or(-1, i32::from)) };
}

/// FASTEST's attention on the tensor cores in sessions made from now on:
/// `Some(true)` the kernel of 128 queries to a block, keys and values
/// through cp.async 64 at a time, the default, `Some(false)` the kernel
/// of 64 that TURBO_CUDA_ATTENTION=64 picks, `None` to read the variable
/// again.
/// Built only with `internals`.
#[cfg(feature = "internals")]
pub fn use_wide_attention(wide: Option<bool>) {
    unsafe { turbo_cuda_use_wide_attention(wide.map_or(-1, i32::from)) };
}

/// The LayerNorms of sessions made from now on: `Some(true)` a kernel of
/// their own after each GEMM (the default), `Some(false)` inside the GEMM,
/// as TURBO_CUDA_LAYER_NORM=fused picks it, `None` to read the variable
/// again. Built only with `internals`.
#[cfg(feature = "internals")]
pub fn use_separate_layer_norm(separate: Option<bool>) {
    unsafe { turbo_cuda_use_separate_layer_norm(separate.map_or(-1, i32::from)) };
}

/// The pooling of sessions made from now on: `Some(true)` the kernel of
/// a thread per column, as TURBO_CUDA_POOL=columns picks it, `Some(false)`
/// the default, `None` to read the variable again. Built only with
/// `internals`.
#[cfg(feature = "internals")]
pub fn use_column_pool(columns: Option<bool>) {
    unsafe { turbo_cuda_use_column_pool(columns.map_or(-1, i32::from)) };
}

/// The F32 GEMMs of MODEL sessions made from now on: `Some(true)` TF32 on
/// the tensor cores (sm_80 and newer), as TURBO_CUDA_TF32=1 picks it,
/// `Some(false)` the FMA kernels, the default, `None` to read the
/// variable again. Built only with `internals`.
#[cfg(feature = "internals")]
pub fn use_tf32(tf32: Option<bool>) {
    unsafe { turbo_cuda_use_tf32(tf32.map_or(-1, i32::from)) };
}

/// The F16 GEMMs of FASTEST sessions made from now on: `Some(true)` F16
/// accumulators over each 64 terms of k, added into F32 ones, as
/// TURBO_CUDA_F16_ACCUMULATE=1 picks it, `Some(false)` F32 accumulators
/// throughout, the default, `None` to read the variable again. Built only
/// with `internals`.
#[cfg(feature = "internals")]
pub fn use_f16_accumulate(f16: Option<bool>) {
    unsafe { turbo_cuda_use_f16_accumulate(f16.map_or(-1, i32::from)) };
}
