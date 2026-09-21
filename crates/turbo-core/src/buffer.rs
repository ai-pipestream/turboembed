//! Buffer descriptors with checked arithmetic, and the host buffer used by
//! CPU-side providers.

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ptr::NonNull;
use std::sync::Arc;

use turbo_abi as abi;

use crate::error::{Error, Result};
use crate::types::{DType, HandleKind, Placement};

/// Alignment for host buffers: one cache line pair, also satisfies every SIMD width.
pub const HOST_ALIGN: usize = 128;

/// Validated description of a buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferDesc {
    /// Memory placement.
    pub placement: Placement,
    /// Element type.
    pub dtype: DType,
    /// Extents.
    pub shape: Vec<u64>,
    /// Byte strides per dimension (packed row-major when constructed by [`BufferDesc::packed`]).
    pub strides: Vec<u64>,
    /// Total bytes covered.
    pub bytes: u64,
}

impl BufferDesc {
    /// Packed row-major buffer for `shape`. Fails on overflow or on `Bytes`
    /// dtype without an explicit byte count.
    pub fn packed(placement: Placement, dtype: DType, shape: &[u64]) -> Result<Self> {
        if shape.len() > abi::TURBO_MAX_RANK {
            return Err(Error::invalid_shape(format!(
                "rank {} exceeds the maximum {}",
                shape.len(),
                abi::TURBO_MAX_RANK
            )));
        }
        let elem = dtype.element_size().ok_or_else(|| {
            Error::invalid_shape("dtype `bytes` needs an explicit byte count; use BufferDesc::with_bytes")
        })? as u64;
        let mut strides = vec![0u64; shape.len()];
        let mut acc = elem;
        for i in (0..shape.len()).rev() {
            strides[i] = acc;
            acc = acc.checked_mul(shape[i]).ok_or_else(|| Error::invalid_shape("shape product overflows u64"))?;
        }
        if acc > isize::MAX as u64 {
            return Err(Error::invalid_shape(format!("buffer of {acc} bytes exceeds addressable memory")));
        }
        Ok(Self { placement, dtype, shape: shape.to_vec(), strides, bytes: acc })
    }

    /// Buffer with an explicit byte count (required for `Bytes`).
    pub fn with_bytes(placement: Placement, dtype: DType, shape: &[u64], bytes: u64) -> Result<Self> {
        if shape.len() > abi::TURBO_MAX_RANK {
            return Err(Error::invalid_shape(format!(
                "rank {} exceeds the maximum {}",
                shape.len(),
                abi::TURBO_MAX_RANK
            )));
        }
        if bytes > isize::MAX as u64 {
            return Err(Error::invalid_shape("byte count exceeds addressable memory"));
        }
        if let Some(elem) = dtype.element_size() {
            let needed = shape
                .iter()
                .try_fold(elem as u64, |acc, &d| acc.checked_mul(d))
                .ok_or_else(|| Error::invalid_shape("shape product overflows u64"))?;
            if bytes < needed {
                return Err(Error::invalid_shape(format!(
                    "byte count {bytes} is smaller than the {needed} bytes the shape needs"
                )));
            }
        }
        let packed_strides = if dtype.element_size().is_some() {
            Self::packed(placement, dtype, shape)?.strides
        } else {
            vec![0; shape.len()]
        };
        Ok(Self { placement, dtype, shape: shape.to_vec(), strides: packed_strides, bytes })
    }

    /// Validate an ABI descriptor: known constants, consistent strides, no overflow.
    pub fn from_abi(d: &abi::turbo_buffer_desc) -> Result<Self> {
        let placement = Placement::from_abi(d.placement)?;
        let dtype = DType::from_abi(d.dtype)?;
        if d.ndim as usize > abi::TURBO_MAX_RANK {
            return Err(Error::invalid_shape(format!("ndim {} exceeds {}", d.ndim, abi::TURBO_MAX_RANK)));
        }
        let shape = &d.shape[..d.ndim as usize];
        let explicit_strides = d.strides[..d.ndim as usize].iter().any(|&s| s != 0);
        let mut desc = if d.bytes == 0 {
            Self::packed(placement, dtype, shape)?
        } else {
            Self::with_bytes(placement, dtype, shape, d.bytes)?
        };
        if explicit_strides {
            let strides = &d.strides[..d.ndim as usize];
            // Largest reachable byte offset must fit inside `bytes`.
            let elem = dtype.element_size().unwrap_or(1) as u64;
            let mut last = 0u64;
            for (i, &n) in shape.iter().enumerate() {
                if n == 0 {
                    last = 0;
                    break;
                }
                last = last
                    .checked_add(
                        (n - 1).checked_mul(strides[i]).ok_or_else(|| Error::invalid_shape("stride overflow"))?,
                    )
                    .ok_or_else(|| Error::invalid_shape("stride overflow"))?;
            }
            let end = last.checked_add(elem).ok_or_else(|| Error::invalid_shape("stride overflow"))?;
            if shape.iter().all(|&n| n != 0) && end > desc.bytes {
                return Err(Error::invalid_shape(format!(
                    "strides reach byte {end} but the buffer holds {} bytes",
                    desc.bytes
                )));
            }
            desc.strides = strides.to_vec();
        }
        Ok(desc)
    }

    /// Number of elements (product of shape).
    pub fn element_count(&self) -> u64 {
        self.shape.iter().product()
    }

    /// True when strides are packed row-major.
    pub fn is_packed(&self) -> bool {
        match Self::packed(self.placement, self.dtype, &self.shape) {
            Ok(p) => p.strides == self.strides,
            Err(_) => false,
        }
    }
}

/// Native memory handle for import and export.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeHandle {
    /// Kind.
    pub kind: HandleKind,
    /// Pointer, `cl_mem`, fd, or object handle.
    pub handle: u64,
    /// Kind-specific auxiliary.
    pub aux: u64,
    /// Byte offset.
    pub offset: u64,
}

/// A provider's buffer. Providers implement this over their own memory.
pub trait ProviderBuffer: Send + Sync {
    /// Description.
    fn desc(&self) -> &BufferDesc;

    /// Host pointer when the placement is host-visible, else `None`.
    /// The pointer is stable for the buffer's lifetime.
    fn host_ptr(&self) -> Option<NonNull<u8>>;

    /// Explicit blocking copy of the whole buffer into `dst`. `dst.len()`
    /// must equal `desc().bytes`. Device placements implement the D2H here.
    fn read_to_host(&self, dst: &mut [u8]) -> Result<()>;

    /// Export a native handle of the requested kind, or `TURBO_E_UNSUPPORTED`.
    fn export(&self, kind: HandleKind) -> Result<NativeHandle>;

    /// For buffers that wrap a plugin provider's handle: the provider's vtable
    /// pointer and the handle, so a session of the same provider can bind it.
    fn plugin_handle(&self) -> Option<(*const std::ffi::c_void, *mut std::ffi::c_void)> {
        None
    }
}

/// Aligned, zero-initialized host memory.
pub struct HostBuffer {
    desc: BufferDesc,
    ptr: NonNull<u8>,
    layout: Layout,
}

// The buffer is exclusively owned memory; access is through raw pointers the
// caller synchronizes, exactly as the ABI requires for sessions.
unsafe impl Send for HostBuffer {}
unsafe impl Sync for HostBuffer {}

impl HostBuffer {
    /// Allocate. Placement must be `Host`.
    pub fn new(desc: BufferDesc) -> Result<Arc<Self>> {
        if desc.placement != Placement::Host {
            return Err(Error::unsupported_placement(format!(
                "HostBuffer only holds TURBO_PLACE_HOST, not {:?}",
                desc.placement
            )));
        }
        let size = usize::try_from(desc.bytes).map_err(|_| Error::invalid_shape("byte count exceeds usize"))?;
        let layout = Layout::from_size_align(size.max(1), HOST_ALIGN)
            .map_err(|e| Error::invalid_shape(format!("layout: {e}")))?;
        // SAFETY: layout has non-zero size and valid alignment.
        let raw = unsafe { alloc_zeroed(layout) };
        let ptr =
            NonNull::new(raw).ok_or_else(|| Error::out_of_memory(format!("host allocation of {size} bytes failed")))?;
        Ok(Arc::new(Self { desc, ptr, layout }))
    }

    /// Allocate a packed buffer for `shape`.
    pub fn packed(dtype: DType, shape: &[u64]) -> Result<Arc<Self>> {
        Self::new(BufferDesc::packed(Placement::Host, dtype, shape)?)
    }

    /// Byte view.
    pub fn bytes(&self) -> &[u8] {
        // SAFETY: ptr covers `bytes` initialized bytes for the lifetime of self.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.desc.bytes as usize) }
    }

    /// Mutable byte view.
    ///
    /// # Safety
    /// The caller must hold exclusive access to the buffer (the session lock
    /// with no result lease outstanding); no other reference may be live.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn bytes_mut(&self) -> &mut [u8] {
        // SAFETY: caller guarantees exclusivity; ptr covers `bytes` bytes.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.desc.bytes as usize) }
    }

    /// Typed view of an `f32` buffer.
    pub fn as_f32(&self) -> Result<&[f32]> {
        if self.desc.dtype != DType::F32 {
            return Err(Error::unsupported_dtype("buffer is not f32"));
        }
        let bytes = self.bytes();
        // SAFETY: HOST_ALIGN guarantees f32 alignment; length is a multiple of 4 by construction.
        Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), bytes.len() / 4) })
    }

    /// Mutable typed view of an `f32` buffer.
    ///
    /// # Safety
    /// Same rule as [`HostBuffer::bytes_mut`]: exclusive access, no live references.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn as_f32_mut(&self) -> Result<&mut [f32]> {
        if self.desc.dtype != DType::F32 {
            return Err(Error::unsupported_dtype("buffer is not f32"));
        }
        let bytes = unsafe { self.bytes_mut() };
        Ok(unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast::<f32>(), bytes.len() / 4) })
    }

    /// Mutable typed view of an `i32` buffer.
    ///
    /// # Safety
    /// Same rule as [`HostBuffer::bytes_mut`]: exclusive access, no live references.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn as_i32_mut(&self) -> Result<&mut [i32]> {
        if self.desc.dtype != DType::I32 {
            return Err(Error::unsupported_dtype("buffer is not i32"));
        }
        let bytes = unsafe { self.bytes_mut() };
        Ok(unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast::<i32>(), bytes.len() / 4) })
    }

    /// Typed view of an `i32` buffer.
    pub fn as_i32(&self) -> Result<&[i32]> {
        if self.desc.dtype != DType::I32 {
            return Err(Error::unsupported_dtype("buffer is not i32"));
        }
        let bytes = self.bytes();
        Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<i32>(), bytes.len() / 4) })
    }
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        // SAFETY: allocated with this layout in `new`.
        unsafe { dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

impl ProviderBuffer for HostBuffer {
    fn desc(&self) -> &BufferDesc {
        &self.desc
    }

    fn host_ptr(&self) -> Option<NonNull<u8>> {
        Some(self.ptr)
    }

    fn read_to_host(&self, dst: &mut [u8]) -> Result<()> {
        if dst.len() as u64 != self.desc.bytes {
            return Err(Error::capacity(format!(
                "destination holds {} bytes but the buffer is {} bytes",
                dst.len(),
                self.desc.bytes
            )));
        }
        dst.copy_from_slice(self.bytes());
        Ok(())
    }

    fn export(&self, kind: HandleKind) -> Result<NativeHandle> {
        match kind {
            HandleKind::HostPtr => Ok(NativeHandle { kind, handle: self.ptr.as_ptr() as u64, aux: 0, offset: 0 }),
            other => Err(Error::unsupported(format!(
                "host buffer cannot be exported as {other:?}; only TURBO_HANDLE_HOST_PTR"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_strides_and_bytes() {
        let d = BufferDesc::packed(Placement::Host, DType::F32, &[2, 3]).unwrap();
        assert_eq!(d.strides, vec![12, 4]);
        assert_eq!(d.bytes, 24);
        assert!(d.is_packed());
    }

    #[test]
    fn overflow_is_rejected() {
        let err = BufferDesc::packed(Placement::Host, DType::F64, &[u64::MAX, 2]).unwrap_err();
        assert_eq!(err.code(), abi::TURBO_E_INVALID_SHAPE);
    }

    #[test]
    fn bytes_dtype_needs_explicit_size() {
        assert!(BufferDesc::packed(Placement::Host, DType::Bytes, &[3]).is_err());
        let d = BufferDesc::with_bytes(Placement::Host, DType::Bytes, &[3], 100).unwrap();
        assert_eq!(d.bytes, 100);
    }

    #[test]
    fn abi_strides_must_fit() {
        let mut raw = abi::turbo_buffer_desc {
            struct_size: std::mem::size_of::<abi::turbo_buffer_desc>() as u32,
            placement: abi::TURBO_PLACE_HOST,
            dtype: abi::TURBO_DTYPE_F32,
            ndim: 2,
            shape: [2, 3, 0, 0, 0, 0, 0, 0],
            strides: [16, 4, 0, 0, 0, 0, 0, 0],
            bytes: 0,
            next: std::ptr::null(),
        };
        assert_eq!(BufferDesc::from_abi(&raw).unwrap_err().code(), abi::TURBO_E_INVALID_SHAPE);
        raw.bytes = 32;
        let d = BufferDesc::from_abi(&raw).unwrap();
        assert_eq!(d.strides, vec![16, 4]);
        assert!(!d.is_packed());
    }

    #[test]
    fn host_buffer_is_zeroed_and_aligned() {
        let b = HostBuffer::packed(DType::F32, &[4, 8]).unwrap();
        assert_eq!(b.bytes().len(), 128);
        assert!(b.bytes().iter().all(|&x| x == 0));
        assert_eq!(b.host_ptr().unwrap().as_ptr() as usize % HOST_ALIGN, 0);
        let mut dst = vec![1u8; 128];
        b.read_to_host(&mut dst).unwrap();
        assert!(dst.iter().all(|&x| x == 0));
        assert_eq!(b.read_to_host(&mut [0u8; 3]).unwrap_err().code(), abi::TURBO_E_CAPACITY);
        assert_eq!(b.export(HandleKind::CudaPtr).unwrap_err().code(), abi::TURBO_E_UNSUPPORTED);
    }
}
