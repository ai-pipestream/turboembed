//! Contexts and buffers through the C interface, on the CPU backend this
//! build links. The host is real hardware on every machine the tests run on.

mod common;

use std::ffi::c_void;
use std::ptr;

use common::{Failure, failure, new_error};
use turbo::*;

struct Rt(*mut turbo_runtime);

impl Rt {
    fn new() -> Rt {
        let mut rt = ptr::null_mut();
        let mut err = new_error();
        assert_eq!(unsafe { turbo_runtime_create(ptr::null(), &mut rt, &mut err) }, 0);
        Rt(rt)
    }

    fn count(&self) -> u32 {
        let mut n = 0;
        assert_eq!(unsafe { turbo_runtime_device_count(self.0, &mut n, ptr::null_mut()) }, 0);
        n
    }

    fn cpu(&self) -> u32 {
        (0..self.count())
            .find(|&i| {
                let mut info: turbo_device_info = unsafe { std::mem::zeroed() };
                info.struct_size = size_of::<turbo_device_info>() as u32;
                assert_eq!(unsafe { turbo_runtime_device_info(self.0, i, &mut info, ptr::null_mut()) }, 0);
                info.kind == TURBO_DEVICE_CPU
            })
            .expect("the cpu backend lists the host")
    }

    fn context(&self, device: u32) -> Result<Ctx, Failure> {
        let mut c = ptr::null_mut();
        let mut err = new_error();
        match unsafe { turbo_context_create(self.0, device, &mut c, &mut err) } {
            0 => Ok(Ctx(c)),
            rc => Err(failure(rc, &err)),
        }
    }
}

impl Drop for Rt {
    fn drop(&mut self) {
        unsafe { turbo_runtime_release(self.0) };
    }
}

struct Ctx(*mut turbo_context);

impl Ctx {
    fn alloc(&self, desc: &turbo_buffer_desc) -> Result<Buf, Failure> {
        let mut b = ptr::null_mut();
        let mut err = new_error();
        match unsafe { turbo_buffer_alloc(self.0, desc, &mut b, &mut err) } {
            0 => Ok(Buf(b)),
            rc => Err(failure(rc, &err)),
        }
    }

    fn import(&self, desc: &turbo_buffer_desc, handle: &turbo_native_handle) -> Result<Buf, Failure> {
        let mut b = ptr::null_mut();
        let mut err = new_error();
        match unsafe { turbo_buffer_import(self.0, desc, handle, &mut b, &mut err) } {
            0 => Ok(Buf(b)),
            rc => Err(failure(rc, &err)),
        }
    }
}

impl Drop for Ctx {
    fn drop(&mut self) {
        unsafe { turbo_context_release(self.0) };
    }
}

struct Buf(*mut turbo_buffer);

impl Buf {
    fn desc(&self) -> turbo_buffer_desc {
        let mut d: turbo_buffer_desc = unsafe { std::mem::zeroed() };
        d.struct_size = size_of::<turbo_buffer_desc>() as u32;
        let mut err = new_error();
        let rc = unsafe { turbo_buffer_get_desc(self.0, &mut d, &mut err) };
        assert_eq!(rc, 0, "{:?}", failure(rc, &err));
        d
    }

    fn host_ptr(&self) -> Result<*mut c_void, Failure> {
        let mut p = ptr::null_mut();
        let mut err = new_error();
        match unsafe { turbo_buffer_host_ptr(self.0, &mut p, &mut err) } {
            0 => Ok(p),
            rc => Err(failure(rc, &err)),
        }
    }

    fn export(&self, kind: u32) -> Result<turbo_native_handle, Failure> {
        let mut h = handle(0, 0, 0);
        h.aux = 7;
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

fn desc(placement: u32, dtype: u32, shape: &[u64]) -> turbo_buffer_desc {
    let mut s = [0u64; 2];
    s[..shape.len()].copy_from_slice(shape);
    turbo_buffer_desc {
        struct_size: size_of::<turbo_buffer_desc>() as u32,
        placement,
        dtype,
        ndim: shape.len() as u32,
        shape: s,
        bytes: 0,
    }
}

fn handle(kind: u32, at: u64, offset: u64) -> turbo_native_handle {
    turbo_native_handle { struct_size: size_of::<turbo_native_handle>() as u32, kind, handle: at, aux: 0, offset }
}

fn host_handle<T>(p: *const T, offset: u64) -> turbo_native_handle {
    handle(TURBO_HANDLE_HOST_PTR, p as usize as u64, offset)
}

fn null_err() -> *mut turbo_error {
    ptr::null_mut()
}

#[test]
fn a_context_reports_its_device() {
    let rt = Rt::new();
    let i = rt.cpu();
    let ctx = rt.context(i).unwrap();
    let mut d = 99u32;
    let mut err = new_error();
    assert_eq!(unsafe { turbo_context_device(ctx.0, &mut d, &mut err) }, 0);
    assert_eq!(d, i);
    assert_eq!(err.code, 0);
    assert_eq!(err.message[0], 0);
}

#[test]
fn a_device_index_past_the_list_is_refused() {
    let rt = Rt::new();
    let e = rt.context(rt.count()).err().unwrap();
    assert!(e.is(status::INVALID_ARGUMENT, "device"), "{e:?}");
}

#[test]
fn alloc_gives_aligned_host_memory_the_caller_can_use() {
    let rt = Rt::new();
    let ctx = rt.context(rt.cpu()).unwrap();
    for placement in [TURBO_PLACE_HOST, TURBO_PLACE_PINNED, TURBO_PLACE_SHARED] {
        for (dtype, shape, bytes) in [
            (TURBO_DTYPE_F32, &[3u64, 5][..], 60u64),
            (TURBO_DTYPE_I32, &[7][..], 28),
            (TURBO_DTYPE_F16, &[2, 3][..], 12),
            (TURBO_DTYPE_BF16, &[1][..], 2),
        ] {
            let b = ctx.alloc(&desc(placement, dtype, shape)).unwrap();
            let p = b.host_ptr().unwrap();
            assert!(!p.is_null());
            assert_eq!(p as usize % 64, 0, "64-byte aligned");
            let mem = unsafe { std::slice::from_raw_parts_mut(p as *mut u8, bytes as usize) };
            mem.fill(0xab);
            assert!(mem.iter().all(|&x| x == 0xab));
            let d = b.desc();
            assert_eq!(d.bytes, bytes, "bytes filled in from shape and dtype");
            assert_eq!(d, turbo_buffer_desc { bytes, ..desc(placement, dtype, shape) });
        }
    }
}

#[test]
fn bytes_that_agree_with_the_shape_are_taken() {
    let rt = Rt::new();
    let ctx = rt.context(rt.cpu()).unwrap();
    let mut d = desc(TURBO_PLACE_HOST, TURBO_DTYPE_F32, &[4, 4]);
    d.bytes = 64;
    assert_eq!(ctx.alloc(&d).unwrap().desc(), d);
}

#[test]
fn import_wraps_the_callers_memory_without_a_copy() {
    let rt = Rt::new();
    let ctx = rt.context(rt.cpu()).unwrap();
    let mut mem = [0i32; 16];
    let base = mem.as_mut_ptr();
    let b = ctx.import(&desc(TURBO_PLACE_HOST, TURBO_DTYPE_I32, &[3, 4]), &host_handle(base, 8)).unwrap();
    let p = b.host_ptr().unwrap() as *mut i32;
    assert_eq!(p, unsafe { base.add(2) }, "the caller's pointer plus offset");

    unsafe { *p.add(5) = 41 };
    assert_eq!(mem[7], 41, "a write through the buffer is in the caller's memory");
    mem[3] = -9;
    assert_eq!(unsafe { *p.add(1) }, -9, "a write to the caller's memory is in the buffer");

    let h = b.export(TURBO_HANDLE_HOST_PTR).unwrap();
    assert_eq!(h, host_handle(p, 0), "export gives the same pointer");
    assert_eq!(b.desc().bytes, 48);
    drop(b);
    mem[0] = 1; // still the caller's after release
    assert_eq!(mem[0], 1);
}

#[test]
fn export_of_an_allocation_is_its_host_pointer() {
    let rt = Rt::new();
    let ctx = rt.context(rt.cpu()).unwrap();
    let b = ctx.alloc(&desc(TURBO_PLACE_SHARED, TURBO_DTYPE_F32, &[8])).unwrap();
    let h = b.export(TURBO_HANDLE_HOST_PTR).unwrap();
    assert_eq!(h, host_handle(b.host_ptr().unwrap(), 0));
}

#[test]
fn release_in_any_order_is_safe() {
    let rt = Rt::new();
    let ctx = rt.context(rt.cpu()).unwrap();
    let a = ctx.alloc(&desc(TURBO_PLACE_HOST, TURBO_DTYPE_F32, &[1024])).unwrap();
    let mut mem = [0u8; 32];
    let i = ctx.import(&desc(TURBO_PLACE_HOST, TURBO_DTYPE_F16, &[16]), &host_handle(mem.as_mut_ptr(), 0)).unwrap();
    // The runtime, then the context, then the buffers.
    drop(rt);
    drop(ctx);
    let p = a.host_ptr().unwrap() as *mut f32;
    unsafe { *p.add(1023) = 1.5 };
    assert_eq!(unsafe { *p.add(1023) }, 1.5);
    assert_eq!(i.host_ptr().unwrap(), mem.as_mut_ptr() as *mut c_void);
    assert_eq!(i.desc().bytes, 32);
    drop(a);
    drop(i);
}

#[test]
fn release_of_null_is_a_no_op() {
    unsafe {
        turbo_context_release(ptr::null_mut());
        turbo_buffer_release(ptr::null_mut());
    }
}

#[test]
fn a_wrong_struct_size_is_refused() {
    let rt = Rt::new();
    let ctx = rt.context(rt.cpu()).unwrap();
    let mut d = desc(TURBO_PLACE_HOST, TURBO_DTYPE_F32, &[4]);
    d.struct_size = 8;
    assert!(ctx.alloc(&d).err().unwrap().is(status::INVALID_STRUCT_SIZE, "turbo_buffer_desc"));
    let mut mem = [0f32; 4];
    assert!(ctx.import(&d, &host_handle(mem.as_mut_ptr(), 0)).err().unwrap().is(status::INVALID_STRUCT_SIZE, ""));
    let d = desc(TURBO_PLACE_HOST, TURBO_DTYPE_F32, &[4]);
    let mut h = host_handle(mem.as_mut_ptr(), 0);
    h.struct_size = 8;
    assert!(ctx.import(&d, &h).err().unwrap().is(status::INVALID_STRUCT_SIZE, "turbo_native_handle"));

    let b = ctx.alloc(&d).unwrap();
    let mut out = d;
    out.struct_size = 8;
    assert_eq!(unsafe { turbo_buffer_get_desc(b.0, &mut out, null_err()) }, status::INVALID_STRUCT_SIZE);
    let mut h = handle(0, 0, 0);
    h.struct_size = 8;
    let rc = unsafe { turbo_buffer_export(b.0, TURBO_HANDLE_HOST_PTR, &mut h, null_err()) };
    assert_eq!(rc, status::INVALID_STRUCT_SIZE);

    let mut err = new_error();
    err.struct_size = 8;
    let mut c = ptr::null_mut();
    assert_eq!(unsafe { turbo_context_create(rt.0, rt.cpu(), &mut c, &mut err) }, status::INVALID_STRUCT_SIZE);
    assert!(c.is_null());
}

#[test]
fn null_and_wrong_handles_are_refused() {
    let rt = Rt::new();
    let d = desc(TURBO_PLACE_HOST, TURBO_DTYPE_F32, &[4]);
    let mut mem = [0f32; 4];
    let h = host_handle(mem.as_mut_ptr(), 0);
    let nc = ptr::null_mut::<turbo_context>();
    let nb = ptr::null_mut::<turbo_buffer>();
    let (mut c, mut b, mut u, mut p) = (ptr::null_mut(), ptr::null_mut(), 0u32, ptr::null_mut());
    let mut out = d;
    let mut nh = handle(0, 0, 0);
    let bad = status::INVALID_HANDLE;
    unsafe {
        assert_eq!(turbo_context_create(ptr::null_mut(), 0, &mut c, null_err()), bad);
        assert_eq!(turbo_context_device(nc, &mut u, null_err()), bad);
        assert_eq!(turbo_buffer_alloc(nc, &d, &mut b, null_err()), bad);
        assert_eq!(turbo_buffer_import(nc, &d, &h, &mut b, null_err()), bad);
        assert_eq!(turbo_buffer_get_desc(nb, &mut out, null_err()), bad);
        assert_eq!(turbo_buffer_host_ptr(nb, &mut p, null_err()), bad);
        assert_eq!(turbo_buffer_export(nb, TURBO_HANDLE_HOST_PTR, &mut nh, null_err()), bad);
        // A context where a buffer belongs, and the other way round.
        let ctx = rt.context(rt.cpu()).unwrap();
        let buf = ctx.alloc(&d).unwrap();
        assert_eq!(turbo_buffer_host_ptr(ctx.0 as *mut turbo_buffer, &mut p, null_err()), bad);
        assert_eq!(turbo_context_device(buf.0 as *mut turbo_context, &mut u, null_err()), bad);
        assert_eq!(turbo_buffer_alloc(rt.0 as *mut turbo_context, &d, &mut b, null_err()), bad);
    }
    assert!(c.is_null() && b.is_null());
}

#[test]
fn null_outputs_are_refused() {
    let rt = Rt::new();
    let ctx = rt.context(rt.cpu()).unwrap();
    let d = desc(TURBO_PLACE_HOST, TURBO_DTYPE_F32, &[4]);
    let mut mem = [0f32; 4];
    let h = host_handle(mem.as_mut_ptr(), 0);
    let buf = ctx.alloc(&d).unwrap();
    let arg = status::INVALID_ARGUMENT;
    let mut b = ptr::null_mut();
    unsafe {
        assert_eq!(turbo_context_create(rt.0, rt.cpu(), ptr::null_mut(), null_err()), arg);
        assert_eq!(turbo_context_device(ctx.0, ptr::null_mut(), null_err()), arg);
        assert_eq!(turbo_buffer_alloc(ctx.0, &d, ptr::null_mut(), null_err()), arg);
        assert_eq!(turbo_buffer_import(ctx.0, &d, &h, ptr::null_mut(), null_err()), arg);
        assert_eq!(turbo_buffer_get_desc(buf.0, ptr::null_mut(), null_err()), arg);
        assert_eq!(turbo_buffer_host_ptr(buf.0, ptr::null_mut(), null_err()), arg);
        assert_eq!(turbo_buffer_export(buf.0, TURBO_HANDLE_HOST_PTR, ptr::null_mut(), null_err()), arg);
        // NULL inputs as well.
        assert_eq!(turbo_buffer_alloc(ctx.0, ptr::null(), &mut b, null_err()), arg);
        assert_eq!(turbo_buffer_import(ctx.0, ptr::null(), &h, &mut b, null_err()), arg);
        assert_eq!(turbo_buffer_import(ctx.0, &d, ptr::null(), &mut b, null_err()), arg);
    }
    assert!(b.is_null());
}

#[test]
fn unknown_enumerations_are_refused_by_name() {
    let rt = Rt::new();
    let ctx = rt.context(rt.cpu()).unwrap();
    let mut mem = [0f32; 4];
    for placement in [0, 5, 99] {
        let e = ctx.alloc(&desc(placement, TURBO_DTYPE_F32, &[4])).err().unwrap();
        assert!(e.is(status::INVALID_ENUM, "placement"), "{e:?}");
    }
    for dtype in [0, 9, 13] {
        let e = ctx.alloc(&desc(TURBO_PLACE_HOST, dtype, &[4])).err().unwrap();
        assert!(e.is(status::INVALID_ENUM, "dtype"), "{e:?}");
    }
    let d = desc(TURBO_PLACE_HOST, TURBO_DTYPE_F32, &[4]);
    for kind in [0, 7] {
        let e = ctx.import(&d, &handle(kind, mem.as_mut_ptr() as usize as u64, 0)).err().unwrap();
        assert!(e.is(status::INVALID_ENUM, "kind"), "{e:?}");
        let e = ctx.alloc(&d).unwrap().export(kind).err().unwrap();
        assert!(e.is(status::INVALID_ENUM, "kind"), "{e:?}");
    }
}

#[test]
fn bad_shapes_are_refused() {
    let rt = Rt::new();
    let ctx = rt.context(rt.cpu()).unwrap();
    let mut d = desc(TURBO_PLACE_HOST, TURBO_DTYPE_F32, &[4, 4]);
    for ndim in [0, 3] {
        d.ndim = ndim;
        let e = ctx.alloc(&d).err().unwrap();
        assert!(e.is(status::INVALID_SHAPE, "ndim"), "{e:?}");
    }
    let e = ctx.alloc(&desc(TURBO_PLACE_HOST, TURBO_DTYPE_F32, &[4, 0])).err().unwrap();
    assert!(e.is(status::INVALID_SHAPE, "shape[1] is 0"), "{e:?}");
    // A shape entry past ndim is not read: a desc reused from a 2-D buffer
    // makes a 1-D one, and get_desc reports it as 0.
    let mut d = desc(TURBO_PLACE_HOST, TURBO_DTYPE_F32, &[4]);
    d.shape[1] = 4;
    let b = ctx.alloc(&d).ok().unwrap();
    assert_eq!((b.desc().shape, b.desc().bytes), ([4, 0], 16));
    for shape in [&[u64::MAX][..], &[1 << 32, 1 << 31][..], &[u64::MAX / 4 + 1][..]] {
        let e = ctx.alloc(&desc(TURBO_PLACE_HOST, TURBO_DTYPE_F32, shape)).err().unwrap();
        assert!(e.is(status::INVALID_SHAPE, "2^64"), "{shape:?}: {e:?}");
    }
}

#[test]
fn more_than_the_host_can_address_is_out_of_memory() {
    let rt = Rt::new();
    let ctx = rt.context(rt.cpu()).unwrap();
    let e = ctx.alloc(&desc(TURBO_PLACE_HOST, TURBO_DTYPE_F32, &[1 << 61])).err().unwrap();
    assert!(e.is(status::OUT_OF_MEMORY, "cpu backend, buffer_alloc"), "{e:?}");
}

#[test]
fn bytes_that_disagree_with_the_shape_are_refused() {
    let rt = Rt::new();
    let ctx = rt.context(rt.cpu()).unwrap();
    let mut mem = [0f32; 16];
    for bytes in [1, 63, 65, 128] {
        let mut d = desc(TURBO_PLACE_HOST, TURBO_DTYPE_F32, &[4, 4]);
        d.bytes = bytes;
        let e = ctx.alloc(&d).err().unwrap();
        assert!(e.is(status::INVALID_ARGUMENT, "bytes"), "{e:?}");
        let e = ctx.import(&d, &host_handle(mem.as_mut_ptr(), 0)).err().unwrap();
        assert!(e.is(status::INVALID_ARGUMENT, "bytes"), "{e:?}");
    }
}

#[test]
fn the_cpu_has_no_device_memory() {
    let rt = Rt::new();
    let ctx = rt.context(rt.cpu()).unwrap();
    let d = desc(TURBO_PLACE_DEVICE, TURBO_DTYPE_F32, &[4]);
    let e = ctx.alloc(&d).err().unwrap();
    assert!(e.is(status::UNSUPPORTED, "placement: TURBO_PLACE_DEVICE"), "{e:?}");
    assert!(e.message.starts_with("cpu backend, buffer_alloc"), "{e:?}");
    let mut mem = [0f32; 4];
    let e = ctx.import(&d, &host_handle(mem.as_mut_ptr(), 0)).err().unwrap();
    assert!(e.is(status::UNSUPPORTED, "placement: TURBO_PLACE_DEVICE"), "{e:?}");
}

#[test]
fn the_cpu_imports_and_exports_only_host_pointers() {
    let rt = Rt::new();
    let ctx = rt.context(rt.cpu()).unwrap();
    let d = desc(TURBO_PLACE_HOST, TURBO_DTYPE_F32, &[4]);
    let mut mem = [0f32; 4];
    let buf = ctx.alloc(&d).unwrap();
    for kind in [
        TURBO_HANDLE_CUDA_PTR,
        TURBO_HANDLE_CL_MEM,
        TURBO_HANDLE_ZE_USM,
        TURBO_HANDLE_MTL_BUFFER,
        TURBO_HANDLE_DMABUF_FD,
    ] {
        let e = ctx.import(&d, &handle(kind, mem.as_mut_ptr() as usize as u64, 0)).err().unwrap();
        assert!(e.is(status::UNSUPPORTED, &format!("kind: {kind}")), "{e:?}");
        let e = buf.export(kind).err().unwrap();
        assert!(e.is(status::UNSUPPORTED, &format!("kind: {kind}")), "{e:?}");
    }
}

#[test]
fn an_import_of_nothing_is_refused() {
    let rt = Rt::new();
    let ctx = rt.context(rt.cpu()).unwrap();
    let d = desc(TURBO_PLACE_HOST, TURBO_DTYPE_F32, &[4]);
    let e = ctx.import(&d, &handle(TURBO_HANDLE_HOST_PTR, 0, 0)).err().unwrap();
    assert!(e.is(status::INVALID_ARGUMENT, "NULL"), "{e:?}");
    let e = ctx.import(&d, &handle(TURBO_HANDLE_HOST_PTR, 64, u64::MAX - 8)).err().unwrap();
    assert!(e.is(status::INVALID_ARGUMENT, "address space"), "{e:?}");
}

#[test]
fn a_failed_call_leaves_the_output_alone() {
    let rt = Rt::new();
    let ctx = rt.context(rt.cpu()).unwrap();
    let mut b = 0x10 as *mut turbo_buffer;
    let d = desc(TURBO_PLACE_DEVICE, TURBO_DTYPE_F32, &[4]);
    assert_eq!(unsafe { turbo_buffer_alloc(ctx.0, &d, &mut b, null_err()) }, status::UNSUPPORTED);
    assert_eq!(b as usize, 0x10);
    let buf = ctx.alloc(&desc(TURBO_PLACE_HOST, TURBO_DTYPE_F32, &[4])).unwrap();
    let mut h = handle(9, 9, 9);
    let rc = unsafe { turbo_buffer_export(buf.0, TURBO_HANDLE_CUDA_PTR, &mut h, null_err()) };
    assert_eq!(rc, status::UNSUPPORTED);
    assert_eq!(h, handle(9, 9, 9));
}

#[test]
fn contexts_and_buffers_work_from_many_threads() {
    let rt = Rt::new();
    let ctx = rt.context(rt.cpu()).unwrap();
    let c = ctx.0 as usize;
    std::thread::scope(|s| {
        for t in 0..8u32 {
            s.spawn(move || {
                let ctx = c as *mut turbo_context;
                for n in 1..50u64 {
                    let mut b = ptr::null_mut();
                    let d = desc(TURBO_PLACE_HOST, TURBO_DTYPE_I32, &[n]);
                    assert_eq!(unsafe { turbo_buffer_alloc(ctx, &d, &mut b, null_err()) }, 0);
                    let mut p = ptr::null_mut();
                    assert_eq!(unsafe { turbo_buffer_host_ptr(b, &mut p, null_err()) }, 0);
                    let row = unsafe { std::slice::from_raw_parts_mut(p as *mut u32, n as usize) };
                    row.fill(t);
                    assert!(row.iter().all(|&x| x == t));
                    unsafe { turbo_buffer_release(b) };
                }
            });
        }
    });
}
