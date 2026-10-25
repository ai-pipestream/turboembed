//! Buffers in every placement, and import and export.

use super::*;

// ---- Buffers -------------------------------------------------------------------------------

/// A context, released on drop when `owned`.
pub(crate) struct Ctx(pub(crate) *mut turbo_context, pub(crate) bool);

impl Ctx {
    pub(crate) fn alloc(&self, placement: u32, n: u64) -> Result<Buf, Failure> {
        let d = desc(placement, n);
        let mut b = ptr::null_mut();
        let mut err = new_error();
        match unsafe { turbo_buffer_alloc(self.0, &d, &mut b, &mut err) } {
            0 => Ok(Buf(b)),
            rc => Err(failure(rc, &err)),
        }
    }

    pub(crate) fn import(&self, placement: u32, n: u64, h: turbo_native_handle) -> Result<Buf, Failure> {
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
pub(crate) struct Buf(pub(crate) *mut turbo_buffer);

impl Buf {
    pub(crate) fn host(&self) -> Result<*mut c_void, Failure> {
        let mut p = ptr::null_mut();
        let mut err = new_error();
        match unsafe { turbo_buffer_host_ptr(self.0, &mut p, &mut err) } {
            0 => Ok(p),
            rc => Err(failure(rc, &err)),
        }
    }

    pub(crate) fn export(&self, kind: u32) -> Result<turbo_native_handle, Failure> {
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
fn desc(placement: u32, n: u64) -> turbo_buffer_desc {
    turbo_buffer_desc {
        struct_size: size_of::<turbo_buffer_desc>() as u32,
        placement,
        dtype: TURBO_DTYPE_F32,
        ndim: 1,
        shape: [n, 0],
        bytes: 0,
    }
}

fn handle(kind: u32, at: u64, aux: u64, offset: u64) -> turbo_native_handle {
    turbo_native_handle { struct_size: size_of::<turbo_native_handle>() as u32, kind, handle: at, aux, offset }
}

fn context(rt: &Rt, device: u32) -> Ctx {
    let mut c = ptr::null_mut();
    let mut err = new_error();
    let rc = unsafe { turbo_context_create(rt.0, device, &mut c, &mut err) };
    assert_eq!(rc, 0, "{:?}", failure(rc, &err));
    Ctx(c, true)
}

#[test]
fn each_placement_is_the_memory_it_names() {
    let _t = turn();
    let Some(dev) = gpu_device("each_placement_is_the_memory_it_names") else { return };
    let rt = Rt::new();
    let ctx = context(&rt, dev);
    let values: Vec<f32> = (0..1000).map(|i| i as f32 * 0.5).collect();

    // DEVICE: no host address; its USM pointer is device memory holding
    // the bytes, in the context its aux names.
    let d = ctx.alloc(TURBO_PLACE_DEVICE, 1000).unwrap();
    assert!(d.host().unwrap_err().is(UNSUPPORTED, "no host pointer"));
    let h = d.export(TURBO_HANDLE_ZE_USM).unwrap();
    assert_eq!((h.kind, h.offset), (TURBO_HANDLE_ZE_USM, 0));
    assert_eq!(ze().kind(h.aux, h.handle).0, MEMORY_TYPE_DEVICE);
    ze().write(&h, &values);
    assert_eq!(ze().read(&h, 1000), values);
    assert!(d.export(TURBO_HANDLE_HOST_PTR).unwrap_err().is(UNSUPPORTED, "no host address"));

    // HOST: pageable memory, exported as a host pointer only.
    let b = ctx.alloc(TURBO_PLACE_HOST, 1000).unwrap();
    let p = b.host().unwrap();
    assert_eq!(p as usize % 64, 0, "aligned");
    assert_eq!(b.export(TURBO_HANDLE_HOST_PTR).unwrap().handle, p as u64);
    assert!(b.export(TURBO_HANDLE_ZE_USM).unwrap_err().is(UNSUPPORTED, "did not allocate"));

    // PINNED: host memory the driver allocated, a USM pointer too.
    let b = ctx.alloc(TURBO_PLACE_PINNED, 1000).unwrap();
    let p = b.host().unwrap();
    assert_eq!(p as usize % 64, 0, "aligned");
    assert_eq!(b.export(TURBO_HANDLE_HOST_PTR).unwrap().handle, p as u64);
    let h = b.export(TURBO_HANDLE_ZE_USM).unwrap();
    assert_eq!((h.handle, ze().kind(h.aux, h.handle).0), (p as u64, MEMORY_TYPE_HOST));

    // SHARED: one address, for the host and the device both.
    let s = ctx.alloc(TURBO_PLACE_SHARED, 1000).unwrap();
    let p = s.host().unwrap();
    let h = s.export(TURBO_HANDLE_ZE_USM).unwrap();
    assert_eq!((h.handle, ze().kind(h.aux, h.handle).0), (p as u64, MEMORY_TYPE_SHARED));
    ze().write(&h, &values);
    assert_eq!(unsafe { std::slice::from_raw_parts(p as *const f32, 1000) }, values.as_slice());
    assert_eq!(s.export(TURBO_HANDLE_HOST_PTR).unwrap().handle, p as u64);

    for kind in [TURBO_HANDLE_CUDA_PTR, TURBO_HANDLE_CL_MEM, TURBO_HANDLE_DMABUF_FD] {
        let e = d.export(kind).unwrap_err();
        assert_eq!(e.code, UNSUPPORTED, "{e:?}");
        assert!(e.message.contains("kind: TURBO_HANDLE_"), "the kind is named: {e:?}");
    }
}

#[test]
fn import_wraps_usm_and_host_pointers_without_a_copy() {
    let _t = turn();
    let Some(dev) = gpu_device("import_wraps_usm_and_host_pointers_without_a_copy") else { return };
    let rt = Rt::new();
    let ctx = context(&rt, dev);
    let owner = ctx.alloc(TURBO_PLACE_DEVICE, 64).unwrap();
    let oh = owner.export(TURBO_HANDLE_ZE_USM).unwrap();
    let (at, zc) = (oh.handle, oh.aux);
    let values: Vec<f32> = (0..64).map(|i| i as f32).collect();
    ze().write(&oh, &values);

    // Eight floats in, no copy: the wrapped pointer is the owner's plus 32 bytes.
    let w = ctx.import(TURBO_PLACE_DEVICE, 56, handle(TURBO_HANDLE_ZE_USM, at, zc, 32)).unwrap();
    let h = w.export(TURBO_HANDLE_ZE_USM).unwrap();
    assert_eq!((h.handle, h.aux, h.offset), (at + 32, zc, 0));
    assert_eq!(ze().read(&h, 56), values[8..]);
    assert!(w.host().is_err());

    let e = ctx.import(TURBO_PLACE_DEVICE, 8, handle(TURBO_HANDLE_ZE_USM, at, zc + 1, 0)).unwrap_err();
    assert!(e.is(INVALID_ARGUMENT, "aux: not this context"), "another context is refused: {e:?}");
    let e = ctx.import(TURBO_PLACE_SHARED, 8, handle(TURBO_HANDLE_ZE_USM, at, zc, 0)).unwrap_err();
    assert!(e.is(INVALID_ARGUMENT, "not SHARED"), "device memory is not shared: {e:?}");
    let e = ctx.import(TURBO_PLACE_HOST, 8, handle(TURBO_HANDLE_ZE_USM, at, zc, 0)).unwrap_err();
    assert!(e.is(INVALID_ARGUMENT, "not HOST"), "{e:?}");
    let e = ctx.import(TURBO_PLACE_DEVICE, 8, handle(TURBO_HANDLE_ZE_USM, 0, zc, 0)).unwrap_err();
    assert!(e.is(INVALID_ARGUMENT, "NULL"), "{e:?}");
    let e = ctx.import(TURBO_PLACE_DEVICE, 8, handle(TURBO_HANDLE_ZE_USM, u64::MAX - 4, zc, 0)).unwrap_err();
    assert!(e.is(INVALID_ARGUMENT, "past the address space"), "{e:?}");

    // Host memory as HOST, never as PINNED unless the driver allocated it.
    let mut mine = vec![1.5f32; 16];
    let hp = mine.as_mut_ptr() as u64;
    let b = ctx.import(TURBO_PLACE_HOST, 16, handle(TURBO_HANDLE_HOST_PTR, hp, 0, 0)).unwrap();
    assert_eq!(b.host().unwrap() as u64, hp);
    assert!(b.export(TURBO_HANDLE_ZE_USM).unwrap_err().is(UNSUPPORTED, "did not allocate"));
    let e = ctx.import(TURBO_PLACE_PINNED, 16, handle(TURBO_HANDLE_HOST_PTR, hp, 0, 0)).unwrap_err();
    assert!(e.is(INVALID_ARGUMENT, "not host memory the driver allocated"), "{e:?}");
    let e = ctx.import(TURBO_PLACE_DEVICE, 16, handle(TURBO_HANDLE_HOST_PTR, hp, 0, 0)).unwrap_err();
    assert!(e.is(INVALID_ARGUMENT, "host memory"), "{e:?}");
    let e = ctx.import(TURBO_PLACE_HOST, 16, handle(TURBO_HANDLE_HOST_PTR, at, 0, 0)).unwrap_err();
    assert!(e.is(INVALID_ARGUMENT, "device memory, not host memory"), "{e:?}");
    let pinned = ctx.alloc(TURBO_PLACE_PINNED, 16).unwrap();
    let pp = pinned.host().unwrap() as u64;
    let b = ctx.import(TURBO_PLACE_PINNED, 16, handle(TURBO_HANDLE_HOST_PTR, pp, 0, 0)).unwrap();
    assert_eq!(b.host().unwrap() as u64, pp);
    assert_eq!(b.export(TURBO_HANDLE_ZE_USM).unwrap().handle, pp, "the driver's memory is USM however it came");

    for kind in [TURBO_HANDLE_CUDA_PTR, TURBO_HANDLE_CL_MEM, TURBO_HANDLE_MTL_BUFFER, TURBO_HANDLE_DMABUF_FD] {
        let e = ctx.import(TURBO_PLACE_DEVICE, 8, handle(kind, at, 0, 0)).unwrap_err();
        assert_eq!(e.code, UNSUPPORTED, "{e:?}");
        assert!(e.message.contains("kind: TURBO_HANDLE_"), "the kind is named: {e:?}");
    }
    drop(mine);
}
