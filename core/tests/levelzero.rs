//! The levelzero backend through the C interface: its devices, checked
//! against what the kernel says is plugged in; its buffers in every
//! placement; import and export of USM pointers; a run's vectors left on
//! the device; the bytes a run moves; and a run's allocations, counted by
//! this binary's allocator and by the backend. The encoder's vectors are
//! held to the f64 arithmetic of tests/common and to the CPU backend's;
//! tests/conformance.rs with TURBO_TEST_DEVICE=levelzero holds them to the
//! upstream reference.
//!
//! Built with the `levelzero` feature only. A test that needs a device
//! says it was skipped, and passes, when the backend lists none; nothing is
//! run on anything else in its place. With TURBO_TEST_REQUIRE_LEVELZERO=1
//! it fails instead, so a run on a machine with an Intel GPU cannot pass by
//! finding none. docs/levelzero.md says how to run them.

#![cfg(feature = "levelzero")]

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::ffi::{c_char, c_void};
use std::ptr;
use std::sync::{Mutex, MutexGuard, OnceLock};

use common::*;
use serde_json::json;
use turbo::status::{INVALID_ARGUMENT, UNSUPPORTED, UNSUPPORTED_OPTION};
use turbo::*;

// ---- This binary's allocator ------------------------------------------------------

/// Counts every heap allocation made on the calling thread, so a run's
/// host_allocs can be held to what really happened.
struct Counting;

thread_local! {
    static COUNT: Cell<u64> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        COUNT.with(|c| c.set(c.get() + 1));
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        COUNT.with(|c| c.set(c.get() + 1));
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        COUNT.with(|c| c.set(c.get() + 1));
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

// ---- Level Zero from the caller's side ---------------------------------------------

/// The loader calls a caller makes on memory it was handed: what an
/// address is, and copies to and from it on the same context.
struct Ze {
    _lib: libloading::Library,
    get_alloc_properties: unsafe extern "C" fn(*mut c_void, *const c_void, *mut AllocProps, *mut *mut c_void) -> i32,
    list_create: unsafe extern "C" fn(*mut c_void, *mut c_void, *const QueueDesc, *mut *mut c_void) -> i32,
    list_destroy: unsafe extern "C" fn(*mut c_void) -> i32,
    copy: unsafe extern "C" fn(*mut c_void, *mut c_void, *const c_void, usize, *mut c_void, u32, *mut c_void) -> i32,
    sync: unsafe extern "C" fn(*mut c_void, u64) -> i32,
}

#[repr(C)]
struct AllocProps {
    stype: u32,
    p_next: *mut c_void,
    kind: u32,
    id: u64,
    page_size: u64,
}

#[repr(C)]
struct QueueDesc {
    stype: u32,
    p_next: *const c_void,
    ordinal: u32,
    index: u32,
    flags: u32,
    mode: u32,
    priority: u32,
}

const MEMORY_TYPE_HOST: u32 = 1;
const MEMORY_TYPE_DEVICE: u32 = 2;
const MEMORY_TYPE_SHARED: u32 = 3;

fn ze() -> &'static Ze {
    static ZE: OnceLock<Ze> = OnceLock::new();
    ZE.get_or_init(|| unsafe {
        let lib = libloading::Library::new("libze_loader.so.1").expect("the Level Zero loader");
        Ze {
            get_alloc_properties: *lib.get(b"zeMemGetAllocProperties\0").unwrap(),
            list_create: *lib.get(b"zeCommandListCreateImmediate\0").unwrap(),
            list_destroy: *lib.get(b"zeCommandListDestroy\0").unwrap(),
            copy: *lib.get(b"zeCommandListAppendMemoryCopy\0").unwrap(),
            sync: *lib.get(b"zeCommandListHostSynchronize\0").unwrap(),
            _lib: lib,
        }
    })
}

impl Ze {
    /// What the driver says `p` is in context `ctx`, and its device.
    fn kind(&self, ctx: u64, p: u64) -> (u32, *mut c_void) {
        let mut a = AllocProps { stype: 0x17, p_next: ptr::null_mut(), kind: 0, id: 0, page_size: 0 };
        let mut dev = ptr::null_mut();
        assert_eq!(unsafe { (self.get_alloc_properties)(ctx as *mut c_void, p as *const c_void, &mut a, &mut dev) }, 0);
        (a.kind, dev)
    }

    /// A copy of `bytes` from `src` to `dst` on the device of `on`, in
    /// context `ctx`, waited for.
    fn copy(&self, ctx: u64, on: u64, dst: *mut c_void, src: *const c_void, bytes: usize) {
        let (_, dev) = self.kind(ctx, on);
        let q = QueueDesc { stype: 0xe, p_next: ptr::null(), ordinal: 0, index: 0, flags: 0, mode: 1, priority: 0 };
        let mut list = ptr::null_mut();
        unsafe {
            assert_eq!((self.list_create)(ctx as *mut c_void, dev, &q, &mut list), 0);
            assert_eq!((self.copy)(list, dst, src, bytes, ptr::null_mut(), 0, ptr::null_mut()), 0);
            assert_eq!((self.sync)(list, u64::MAX), 0);
            assert_eq!((self.list_destroy)(list), 0);
        }
    }

    fn write(&self, h: &turbo_native_handle, src: &[f32]) {
        self.copy(h.aux, h.handle, h.handle as *mut c_void, src.as_ptr() as *const c_void, src.len() * 4);
    }

    fn read(&self, h: &turbo_native_handle, n: usize) -> Vec<f32> {
        let mut v = vec![f32::NAN; n];
        self.copy(h.aux, h.handle, v.as_mut_ptr() as *mut c_void, h.handle as *const c_void, n * 4);
        v
    }
}

// ---- Helpers ------------------------------------------------------------------------

/// The tests take turns: the allocation counts are the process's, and
/// another test's work would move them.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn turn() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner())
}

fn null_err() -> *mut turbo_error {
    ptr::null_mut()
}

const TEXTS: [&str; 3] = ["The quick brown fox jumps over the lazy dog.", "how do I reset a password", "a"];

struct Rt(*mut turbo_runtime);

impl Rt {
    fn new() -> Rt {
        let mut rt = ptr::null_mut();
        assert_eq!(unsafe { turbo_runtime_create(ptr::null(), &mut rt, null_err()) }, 0);
        Rt(rt)
    }

    fn count(&self) -> u32 {
        let mut n = 0;
        assert_eq!(unsafe { turbo_runtime_device_count(self.0, &mut n, null_err()) }, 0);
        n
    }

    fn info(&self, i: u32) -> turbo_device_info {
        let mut out: turbo_device_info = unsafe { std::mem::zeroed() };
        out.struct_size = size_of::<turbo_device_info>() as u32;
        let mut err = new_error();
        let rc = unsafe { turbo_runtime_device_info(self.0, i, &mut out, &mut err) };
        assert_eq!(rc, 0, "{:?}", failure(rc, &err));
        out
    }

    /// The indices of the devices the levelzero backend lists.
    fn listed(&self) -> Vec<u32> {
        (0..self.count()).filter(|&i| field(&self.info(i).backend) == "levelzero").collect()
    }
}

impl Drop for Rt {
    fn drop(&mut self) {
        unsafe { turbo_runtime_release(self.0) };
    }
}

/// TURBO_TEST_REQUIRE_LEVELZERO=1: a test that finds no device fails.
fn required() -> bool {
    std::env::var("TURBO_TEST_REQUIRE_LEVELZERO").is_ok_and(|v| v == "1")
}

/// The first levelzero device, or None after saying the test is skipped.
fn gpu_device(test: &str) -> Option<u32> {
    let rt = Rt::new();
    let d = first_of(rt.0, "levelzero");
    if d.is_none() {
        assert!(!required(), "{test}: TURBO_TEST_REQUIRE_LEVELZERO=1 and the levelzero backend lists no device");
        println!("{test}: skipped: the levelzero feature is on and the levelzero backend lists no device");
    }
    d
}

fn gpu(rt: *mut turbo_runtime) -> u32 {
    first_of(rt, "levelzero").expect("a levelzero device")
}

fn on_gpu(dir: &std::path::Path) -> Loaded {
    Loaded::load_on(dir, gpu).unwrap_or_else(|e| panic!("{e:?}"))
}

fn opts(edit: impl FnOnce(&mut turbo_embed_options)) -> turbo_embed_options {
    let mut o = embed_options();
    edit(&mut o);
    o
}

// ---- Without a device ------------------------------------------------------------------

#[test]
fn the_backend_is_linked_before_the_cpu_and_names_itself() {
    let linked = turbo::backend::linked();
    let b = linked.iter().find(|b| b.name() == "levelzero").expect("linked");
    assert_eq!(b.struct_size as usize, size_of::<turbo::backend::turbo_backend>());
    turbo::backend::check_table(b).unwrap();
    assert!(b.buffer_read.is_some() && b.session_run.is_some());
    let at = |name: &str| linked.iter().position(|b| b.name() == name).unwrap();
    assert!(at("levelzero") < at("cpu"), "its devices come before the cpu's");
    let v = unsafe { std::ffi::CStr::from_ptr(b.runtime_version) }.to_str().unwrap();
    assert_eq!(v, "", "nothing is linked at build time; each device names the loader it opened");
}

// ---- Devices -------------------------------------------------------------------------------

/// The PCI device ids of the Intel GPUs the kernel has bound a compute
/// driver to, one per render node.
fn intel_gpus() -> Vec<u32> {
    let mut ids = Vec::new();
    let Ok(dir) = std::fs::read_dir("/sys/class/drm") else { return ids };
    for entry in dir {
        let path = entry.unwrap().path();
        if !path.file_name().unwrap().to_str().unwrap().starts_with("renderD") {
            continue;
        }
        let dev = path.join("device");
        let read = |f: &str| std::fs::read_to_string(dev.join(f)).unwrap_or_default().trim().to_owned();
        let driver = std::fs::read_link(dev.join("driver")).unwrap_or_default();
        let driver = driver.file_name().and_then(|d| d.to_str()).unwrap_or("");
        if read("vendor") == "0x8086" && (driver == "xe" || driver == "i915") {
            ids.push(u32::from_str_radix(read("device").trim_start_matches("0x"), 16).unwrap());
        }
    }
    ids.sort();
    ids
}

/// The version in an installed library's file name: libze_loader.so.1
/// links to libze_loader.so.1.28.2, which is "1.28.2". None where the
/// library is not in one of the usual directories.
fn installed_version(soname: &str) -> Option<String> {
    let lib = ["/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib"]
        .iter()
        .find_map(|d| std::fs::read_link(format!("{d}/{soname}")).ok())?;
    let name = lib.file_name()?.to_str()?;
    Some(name.strip_prefix(soname.trim_end_matches(".1"))?.trim_start_matches('.').to_owned())
}

/// Every Intel GPU the kernel drives is listed, and each reads as the
/// hardware and software it is.
#[test]
fn the_devices_listed_are_the_kernels() {
    let _t = turn();
    let rt = Rt::new();
    let want = intel_gpus();
    let listed = rt.listed();
    if listed.is_empty() {
        // No GPU, or one without the compute runtime: the listing says
        // nothing about the kernel's unless a device is required.
        assert!(!required(), "TURBO_TEST_REQUIRE_LEVELZERO=1 and the levelzero backend lists no device");
        println!("the_devices_listed_are_the_kernels: skipped: the levelzero backend lists no device");
        return;
    }
    assert_eq!(listed.len(), want.len(), "the kernel drives {want:x?}");
    let mut archs: Vec<String> = listed.iter().map(|&i| field(&rt.info(i).arch)).collect();
    archs.sort();
    let mut expect: Vec<String> =
        want.iter().map(|&id| if id == 0xe223 { "b70".to_owned() } else { format!("intel-{id:04x}") }).collect();
    expect.sort();
    assert_eq!(archs, expect);
    let (loader, driver) = (installed_version("libze_loader.so.1"), installed_version("libze_intel_gpu.so.1"));
    for (ordinal, &i) in listed.iter().enumerate() {
        let d = rt.info(i);
        assert_eq!(d.ordinal, ordinal as u32, "numbered within the backend");
        assert!(d.kind == TURBO_DEVICE_GPU || d.kind == TURBO_DEVICE_IGPU);
        assert_eq!(d.unified_memory, u32::from(d.kind == TURBO_DEVICE_IGPU));
        assert_eq!(field(&d.vendor), "Intel");
        assert!(!field(&d.name).is_empty(), "the driver names it");
        if let Some(loader) = &loader {
            assert_eq!(&field(&d.runtime_version), loader, "the loader's version");
        }
        if let Some(build) = driver.as_ref().and_then(|v| v.rsplit('.').next()) {
            assert!(field(&d.driver_version).ends_with(&format!(".{build}")), "the installed driver's build {build}");
        }
        assert!(d.memory_total > 0);
        assert!(d.memory_free > 0 && d.memory_free <= d.memory_total, "sysman reads free memory");
        println!(
            "levelzero device {ordinal}: {}, arch {}, loader {}, driver {}, {} of {} bytes free",
            field(&d.name),
            field(&d.arch),
            field(&d.runtime_version),
            field(&d.driver_version),
            d.memory_free,
            d.memory_total
        );
    }
    let first = listed[0];
    assert_eq!(listed, (first..first + listed.len() as u32).collect::<Vec<_>>(), "a backend's devices are contiguous");
    assert!((first + listed.len() as u32..rt.count()).all(|i| field(&rt.info(i).backend) == "cpu"), "then the cpu");
}

#[test]
fn runtimes_on_many_threads_list_the_same_devices() {
    let infos: Vec<Vec<String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..16)
            .map(|_| {
                scope.spawn(|| {
                    let rt = Rt::new();
                    rt.listed()
                        .into_iter()
                        .map(|i| {
                            let mut info = rt.info(i);
                            info.memory_free = 0;
                            format!("{info:?}")
                        })
                        .collect()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    assert!(infos.iter().all(|i| *i == infos[0]), "{infos:#?}");
}

// ---- Capability and selection --------------------------------------------------------------

#[test]
fn embed_is_offered_in_f32_and_f16_as_sessions_run_it() {
    let _t = turn();
    let Some(_) = gpu_device("embed_is_offered_in_f32_and_f16_as_sessions_run_it") else { return };
    let l = on_gpu(&tiny_bundle());
    for p in [TURBO_PRECISION_MODEL, TURBO_PRECISION_FASTEST, TURBO_PRECISION_EXACT] {
        let mut cap: turbo_capability = unsafe { std::mem::zeroed() };
        cap.struct_size = size_of::<turbo_capability>() as u32;
        assert_eq!(unsafe { turbo_runtime_capability(l.rt, gpu(l.rt), TURBO_TASK_EMBED, p, &mut cap, null_err()) }, 0);
        // A benchmark record compiled into the build for this machine's
        // cell makes it SUPPORTED, naming the record; else it runs
        // unmeasured.
        if cap.status == backend::TURBO_CAP_SUPPORTED {
            assert!(field(&cap.benchmark).starts_with(&format!("{}.levelzero.embed.", field(&arch_of(&l)))));
            assert!(cap.cosine_floor > 0.999 && cap.speed_ratio > 0.0, "a record's numbers");
        } else {
            assert_eq!(cap.status, backend::TURBO_CAP_EXPERIMENTAL, "{}", field(&cap.reason));
        }
        // FASTEST runs the linear layers on the matrix engines in F16.
        let want = if p == TURBO_PRECISION_FASTEST { TURBO_DTYPE_F16 } else { TURBO_DTYPE_F32 };
        assert_eq!(cap.dtype, want, "precision {p}");
        assert_eq!(cap.options_honored, 0b111111, "every field of turbo_embed_options");
        let info = Session::create(l.m, Some(&session_desc(0, 0, p))).unwrap().info();
        assert_eq!(info.compute_dtype, cap.dtype, "precision {p}");
    }
}

/// The arch label of the levelzero device a bundle is loaded on.
fn arch_of(l: &Loaded) -> [c_char; 32] {
    let mut info: turbo_device_info = unsafe { std::mem::zeroed() };
    info.struct_size = size_of::<turbo_device_info>() as u32;
    assert_eq!(unsafe { turbo_runtime_device_info(l.rt, gpu(l.rt), &mut info, null_err()) }, 0);
    info.arch
}

#[test]
fn selection_picks_the_gpu() {
    let _t = turn();
    let Some(dev) = gpu_device("selection_picks_the_gpu") else { return };
    let rt = Rt::new();
    let mut pick = 99u32;
    let mut reason = [0 as c_char; 256];
    let rc = unsafe { turbo_runtime_select(rt.0, TURBO_TASK_EMBED, &mut pick, reason.as_mut_ptr(), 256, null_err()) };
    assert_eq!(rc, 0, "{}", field(&reason));
    assert_eq!(pick, dev, "{}", field(&reason));
    assert!(field(&reason).contains("(levelzero "), "{}", field(&reason));
}

// ---- Buffers -------------------------------------------------------------------------------

/// A context, released on drop when `owned`.
struct Ctx(*mut turbo_context, bool);

impl Ctx {
    fn alloc(&self, placement: u32, n: u64) -> Result<Buf, Failure> {
        let d = desc(placement, n);
        let mut b = ptr::null_mut();
        let mut err = new_error();
        match unsafe { turbo_buffer_alloc(self.0, &d, &mut b, &mut err) } {
            0 => Ok(Buf(b)),
            rc => Err(failure(rc, &err)),
        }
    }

    fn import(&self, placement: u32, n: u64, h: turbo_native_handle) -> Result<Buf, Failure> {
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
struct Buf(*mut turbo_buffer);

impl Buf {
    fn host(&self) -> Result<*mut c_void, Failure> {
        let mut p = ptr::null_mut();
        let mut err = new_error();
        match unsafe { turbo_buffer_host_ptr(self.0, &mut p, &mut err) } {
            0 => Ok(p),
            rc => Err(failure(rc, &err)),
        }
    }

    fn export(&self, kind: u32) -> Result<turbo_native_handle, Failure> {
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

// ---- Runs ----------------------------------------------------------------------------------

/// Rows with padding on either side, a masked token inside a row and token
/// types, as tests/sessions.rs writes them for the CPU.
fn awkward_rows(dir: &std::path::Path) -> Tokens {
    let tok = Tok::create(dir).unwrap();
    let rows: Vec<Vec<i32>> = TEXTS.iter().map(|t| tok.row(t, None).unwrap()).collect();
    let seq = rows.iter().map(Vec::len).max().unwrap() + 2;
    let mut t = Tokens::new(&[vec![0; seq], vec![0; seq], vec![0; seq]], 0);
    t.mask.fill(0);
    let mut types = vec![0; 3 * seq];
    for (r, row) in rows.iter().enumerate() {
        let start = if r == 1 { seq - row.len() } else { 0 };
        for (p, &id) in row.iter().enumerate() {
            t.ids[r * seq + start + p] = id;
            t.mask[r * seq + start + p] = 1;
            if r == 2 && p > 0 {
                types[r * seq + start + p] = 1;
            }
        }
    }
    t.mask[2 * seq + 1] = 0;
    t.types = Some(types);
    t
}

/// Every pooling, with and without normalization, against the f64
/// encoder of tests/common and against the CPU backend.
#[test]
fn every_option_matches_the_arithmetic_and_the_cpu() {
    let _t = turn();
    let Some(_) = gpu_device("every_option_matches_the_arithmetic_and_the_cpu") else { return };
    let dir = tiny_bundle();
    let plain = PlainBert::new(&dir);
    let (g, c) = (on_gpu(&dir), Loaded::load(&dir).unwrap());
    let (gs, cs) = (Session::create(g.m, None).unwrap(), Session::create(c.m, None).unwrap());
    let t = awkward_rows(&dir);
    let seq = t.seq as usize;
    let types = t.types.clone().unwrap();
    let (mut worst, mut lowest) = (0f64, 1f64);
    for pooling in [TURBO_POOLING_MEAN, TURBO_POOLING_CLS, TURBO_POOLING_LAST] {
        for normalize in [TURBO_NORMALIZE_NONE, TURBO_NORMALIZE_L2] {
            let o = opts(|o| {
                o.pooling = pooling;
                o.normalize = normalize;
            });
            gs.write_tokens(&t.batch(), Some(&o)).unwrap();
            let got = gs.run().unwrap().rows();
            cs.write_tokens(&t.batch(), Some(&o)).unwrap();
            let cpu = cs.run().unwrap().rows();
            for (r, row) in got.iter().enumerate() {
                let at = r * seq..(r + 1) * seq;
                let want = plain.embed(
                    &t.ids[at.clone()],
                    &t.mask[at.clone()],
                    &types[at],
                    pooling,
                    32,
                    normalize == TURBO_NORMALIZE_L2,
                );
                for (a, w) in row.iter().zip(&want) {
                    let d = (*a as f64 - w).abs();
                    worst = worst.max(d);
                    assert!(d < 1e-5 * (1.0 + w.abs()), "pooling {pooling} normalize {normalize} row {r}: {a} vs {w}");
                }
                let c = cosine(row, &cpu[r]);
                lowest = lowest.min(c);
                assert!(c >= 0.9999, "pooling {pooling} normalize {normalize} row {r}: cosine {c} with the cpu");
            }
        }
    }
    println!(
        "largest difference from the f64 encoder {worst:.3e}; 1 - lowest cosine with the cpu {:.3e}",
        1.0 - lowest
    );

    // FASTEST: the linear layers on the matrix engines in F16, every
    // option, held to the F16 floor against the cpu.
    let fs = Session::create(g.m, Some(&session_desc(0, 0, TURBO_PRECISION_FASTEST))).unwrap();
    assert_eq!(fs.info().compute_dtype, TURBO_DTYPE_F16);
    let mut lowest = 1f64;
    for pooling in [TURBO_POOLING_MEAN, TURBO_POOLING_CLS, TURBO_POOLING_LAST] {
        for normalize in [TURBO_NORMALIZE_NONE, TURBO_NORMALIZE_L2] {
            let o = opts(|o| {
                o.pooling = pooling;
                o.normalize = normalize;
            });
            fs.write_tokens(&t.batch(), Some(&o)).unwrap();
            let got = fs.run().unwrap().rows();
            cs.write_tokens(&t.batch(), Some(&o)).unwrap();
            for (r, (a, b)) in got.iter().zip(cs.run().unwrap().rows()).enumerate() {
                let c = cosine(a, &b);
                lowest = lowest.min(c);
                assert!(c >= 0.999, "FASTEST pooling {pooling} normalize {normalize} row {r}: cosine {c} with the cpu");
            }
        }
    }
    println!("FASTEST: 1 - lowest cosine with the cpu {:.3e}", 1.0 - lowest);

    // Text, and the same rows as tokens at a wider stride, give the same vectors.
    let from_text = gs.embed(&TEXTS, None).unwrap();
    let tok = Tok::create(&dir).unwrap();
    let rows: Vec<Vec<i32>> = TEXTS.iter().map(|t| tok.row(t, None).unwrap()).collect();
    let packed = Tokens::new(&rows, 0);
    let seq = packed.seq as usize;
    let mut wide = Tokens::new(&rows, 0);
    wide.stride = seq as u32 + 5;
    wide.ids = vec![-9; rows.len() * wide.stride as usize];
    wide.mask = vec![-9; rows.len() * wide.stride as usize];
    for r in 0..rows.len() {
        let at = r * wide.stride as usize;
        wide.ids[at..at + seq].copy_from_slice(&packed.ids[r * seq..(r + 1) * seq]);
        wide.mask[at..at + seq].copy_from_slice(&packed.mask[r * seq..(r + 1) * seq]);
    }
    gs.write_tokens(&wide.batch(), None).unwrap();
    assert_eq!(gs.run().unwrap().rows(), from_text, "the stride's gap is never read");
}

/// The vectors stay on the device: the result's buffer is device memory
/// whose USM pointer holds what turbo_result_read copies back, and the
/// counts are the bytes that crossed.
#[test]
fn a_run_leaves_its_vectors_on_the_device_and_counts_what_crossed() {
    let _t = turn();
    let Some(_) = gpu_device("a_run_leaves_its_vectors_on_the_device_and_counts_what_crossed") else { return };
    let l = on_gpu(&tiny_bundle());
    let mi = l.info();
    let s = Session::create(l.m, None).unwrap();
    let tok = Tok::create(&tiny_bundle()).unwrap();
    let live: u64 = TEXTS.iter().map(|t| tok.row(t, None).unwrap().len() as u64).sum();

    s.write_text(&TEXTS, None).unwrap();
    let r = s.run().unwrap();
    let i = r.info();
    assert_eq!((i.task, i.batch, i.dim), (TURBO_TASK_EMBED, 3, 32));
    assert_eq!((i.dtype, i.compute_dtype, i.placement), (TURBO_DTYPE_F32, TURBO_DTYPE_F32, TURBO_PLACE_DEVICE));
    assert_eq!(i.device, gpu(l.rt));
    assert_eq!(i.bytes, 3 * 32 * 4);
    // The packed rows the lookup kernel reads over the bus: ids, positions
    // and mask for each live token, and the row table.
    assert_eq!(i.h2d_bytes, 3 * live * 4 + 3 * 8, "{live} live tokens");
    assert_eq!(i.d2h_bytes, 0, "nothing came back yet");
    assert_eq!((i.host_allocs, i.device_allocs), (0, 0));
    let (h, d, f, u) = (TURBO_STAGE_HOST, TURBO_STAGE_DEVICE, TURBO_STAGE_FUSED, TURBO_STAGE_UNUSED);
    assert_eq!(i.stage[..7], [h, f, d, d, d, d, u], "tokenize, upload, lookup, encode, pool, normalize, download");
    assert!(i.stage[7..].iter().all(|&s| s == u));
    assert_eq!(field(&i.backend), "levelzero");
    assert_eq!(field(&i.manifest_sha256), field(&mi.manifest_sha256));

    let read = r.rows().concat();
    assert_eq!(r.info().d2h_bytes, i.bytes, "a read is counted");
    r.rows();
    assert_eq!(r.info().d2h_bytes, 2 * i.bytes, "each time");

    let mut buf = ptr::null_mut();
    assert_eq!(unsafe { turbo_result_buffer(r.0, &mut buf, null_err()) }, 0);
    let buf = Buf(buf);
    let mut bd: turbo_buffer_desc = unsafe { std::mem::zeroed() };
    bd.struct_size = size_of::<turbo_buffer_desc>() as u32;
    assert_eq!(unsafe { turbo_buffer_get_desc(buf.0, &mut bd, null_err()) }, 0);
    assert_eq!((bd.placement, bd.shape, bd.bytes), (TURBO_PLACE_DEVICE, [3, 32], 384));
    assert!(buf.host().unwrap_err().is(UNSUPPORTED, "no host pointer"));
    let h = buf.export(TURBO_HANDLE_ZE_USM).unwrap();
    assert_eq!(ze().kind(h.aux, h.handle).0, MEMORY_TYPE_DEVICE);
    assert_eq!(ze().read(&h, read.len()), read, "the device pointer holds the vectors, no copy made");
    drop(buf);
    drop(r);

    // Tokens the caller wrote, with types: three arrays cross; nothing
    // is tokenized, and an unnormalized vector is not normalized.
    let mut b = Tokens::new(&[vec![101, 7592, 102]], 0);
    b.types = Some(vec![0, 1, 1]);
    s.write_tokens(&b.batch(), Some(&opts(|o| o.normalize = TURBO_NORMALIZE_NONE))).unwrap();
    let i = s.run().unwrap().info();
    assert_eq!(i.h2d_bytes, 4 * 3 * 4 + 8, "types too");
    assert_eq!(i.stage[..7], [u, f, d, d, d, u, u]);
    assert_eq!((i.batch, i.d2h_bytes), (1, 0), "a run's count starts again");
}

/// Rows in memory the driver allocated (a PINNED or SHARED buffer's) go to
/// the device from where they are, and give the same vectors.
#[test]
fn rows_in_driver_memory_give_the_same_vectors() {
    let _t = turn();
    let Some(_) = gpu_device("rows_in_driver_memory_give_the_same_vectors") else { return };
    let l = on_gpu(&tiny_bundle());
    let s = Session::create(l.m, None).unwrap();
    let t = awkward_rows(&tiny_bundle());
    s.write_tokens(&t.batch(), None).unwrap();
    let want = s.run().unwrap().rows();

    let ctx = Ctx(l.ctx, false);
    let n = t.ids.len() as u64;
    // Each row's positions through its last live token.
    let packed: u64 =
        t.mask.chunks(t.seq as usize).map(|m| m.iter().rposition(|&v| v != 0).map_or(0, |p| p + 1) as u64).sum();
    for placement in [TURBO_PLACE_PINNED, TURBO_PLACE_SHARED] {
        let bufs: Vec<Buf> = (0..3).map(|_| ctx.alloc(placement, n).unwrap()).collect();
        let at: Vec<*mut i32> = bufs.iter().map(|b| b.host().unwrap() as *mut i32).collect();
        let types = t.types.as_ref().unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(t.ids.as_ptr(), at[0], t.ids.len());
            std::ptr::copy_nonoverlapping(t.mask.as_ptr(), at[1], t.mask.len());
            std::ptr::copy_nonoverlapping(types.as_ptr(), at[2], types.len());
        }
        let mut b = t.batch();
        b.ids = at[0];
        b.mask = at[1];
        b.types = at[2];
        s.write_tokens(&b, None).unwrap();
        let r = s.run().unwrap();
        assert_eq!(r.rows(), want, "placement {placement}");
        assert_eq!(r.info().h2d_bytes, 4 * packed * 4 + 3 * 8);
    }
}

/// A run allocates nothing on the host or the device, cold or warm: this
/// binary's allocator counts nothing on the running thread, the backend's
/// own count does not move, and the result says 0.
#[test]
fn a_run_allocates_nothing() {
    let _t = turn();
    let Some(_) = gpu_device("a_run_allocates_nothing") else { return };
    let l = on_gpu(&tiny_bundle());
    let s = Session::create(l.m, None).unwrap();
    let tok = Tok::create(&tiny_bundle()).unwrap();
    let rows: Vec<Vec<i32>> = TEXTS.iter().map(|t| tok.row(t, None).unwrap()).collect();
    let small = Tokens::new(&rows, 0);
    let long = Tokens::new(&vec![(0..64).map(|i| 1000 + i).collect(); 64], 0);
    let mut dst = vec![0f32; 64 * 32];
    let mut run = |t: &Tokens| {
        let batch = t.batch();
        s.write_tokens(&batch, None).unwrap();
        let mut r = ptr::null_mut();
        let before = COUNT.with(Cell::get);
        assert_eq!(unsafe { turbo_session_run(s.0, &mut r, null_err()) }, 0);
        let counted = COUNT.with(Cell::get) - before;
        let r = Outcome(r);
        let i = r.info();
        let rc = unsafe {
            turbo_result_read(r.0, dst.as_mut_ptr() as *mut _, (dst.len() * 4) as u64, ptr::null_mut(), null_err())
        };
        assert_eq!(rc, 0);
        (i.host_allocs, i.device_allocs, counted)
    };
    let before = turbo::levelzero::allocations();
    for (i, t) in [&small, &long, &small, &long].into_iter().enumerate() {
        let (host, device, counted) = run(t);
        assert_eq!((host, device), (0, 0), "run {i}: the result's count");
        assert_eq!(counted, host, "run {i}: what this thread's allocator counted in turbo_session_run");
    }
    assert_eq!(turbo::levelzero::allocations(), before, "the backend allocated nothing in runs");
}

/// A model stored in F16 or BF16: MODEL is refused, EXACT and FASTEST
/// share one F32 copy on the device, and give the vectors of the same
/// values stored as F32.
#[test]
fn half_weights_compute_in_f32_from_one_shared_copy() {
    let _t = turn();
    let Some(_) = gpu_device("half_weights_compute_in_f32_from_one_shared_copy") else { return };
    for (dtype, narrow) in [("F16", to_f16 as fn(f32) -> u16), ("BF16", to_bf16)] {
        let (n, w): (Vec<Tensor>, Vec<Tensor>) = tiny_weights(0)
            .into_iter()
            .map(|t| {
                let halves: Vec<u16> =
                    t.data.chunks(4).map(|c| narrow(f32::from_le_bytes(c.try_into().unwrap()))).collect();
                let wide = |h: u16| if dtype == "F16" { from_f16(h) } else { f32::from_bits((h as u32) << 16) };
                let nt = Tensor {
                    name: t.name.clone(),
                    dtype,
                    shape: t.shape.clone(),
                    data: halves.iter().flat_map(|h| h.to_le_bytes()).collect(),
                };
                let wt = Tensor { data: halves.iter().flat_map(|&h| wide(h).to_le_bytes()).collect(), ..t };
                (nt, wt)
            })
            .unzip();
        let mut f = Fixture::model(&format!("levelzero-half-{dtype}"));
        f.weights("weights/model.safetensors", &n);
        let l = f.load_on(gpu).unwrap();
        let e = Session::create(l.m, None).err().unwrap();
        assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 3), "{dtype}: {e:?}");
        assert!(unsafe { model_converted_weights(l.m) }.is_none(), "a refused session makes no copy");
        let a = Session::create(l.m, Some(&session_desc(0, 0, TURBO_PRECISION_EXACT))).unwrap();
        let copy = unsafe { model_converted_weights(l.m) }.expect("the first F32 session made the copy");
        let b = Session::create(l.m, Some(&session_desc(0, 0, TURBO_PRECISION_FASTEST))).unwrap();
        assert_eq!(unsafe { model_converted_weights(l.m) }.unwrap(), copy, "one copy, shared");
        assert_eq!(b.info().compute_dtype, TURBO_DTYPE_F32);

        let mut g = Fixture::model(&format!("levelzero-half-{dtype}-wide"));
        g.weights("weights/model.safetensors", &w);
        let lw = g.load_on(gpu).unwrap();
        let want = Session::create(lw.m, None).unwrap().embed(&TEXTS, None).unwrap();
        assert_eq!(a.embed(&TEXTS, None).unwrap(), want, "{dtype}");
        assert_eq!(b.embed(&TEXTS, None).unwrap(), want, "{dtype}");
        assert!(unsafe { model_converted_weights(lw.m) }.is_none(), "F32 weights are used as loaded");

        // The same F16 or BF16 bundle on the CPU, which widens it too.
        let lc = f.load().unwrap();
        let cpu = Session::create(lc.m, Some(&session_desc(0, 0, TURBO_PRECISION_EXACT))).unwrap();
        for (r, (g, c)) in want.iter().zip(cpu.embed(&TEXTS, None).unwrap()).enumerate() {
            let cos = cosine(g, &c);
            assert!(cos >= 0.9999, "{dtype} row {r}: cosine {cos} with the cpu");
        }
    }
}

fn to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    if x == 0.0 {
        return sign;
    }
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    assert!((1..31).contains(&exp), "{x} is not a normal half");
    let man = b & 0x7f_ffff;
    let mut h = ((exp as u32) << 10) | (man >> 13);
    let rest = man & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && h & 1 == 1) {
        h += 1;
    }
    sign | h as u16
}

fn from_f16(h: u16) -> f32 {
    let (sign, exp, man) = ((h as u32 & 0x8000) << 16, (h >> 10) & 0x1f, h as u32 & 0x3ff);
    if exp == 0 && man == 0 {
        return f32::from_bits(sign);
    }
    f32::from_bits(sign | ((exp as u32 + 112) << 23) | (man << 13))
}

fn to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// Sessions on one context, run from several threads at once, each get
/// their own vectors: the context's queue is shared under its lock.
#[test]
fn sessions_on_one_context_run_from_many_threads() {
    let _t = turn();
    let Some(_) = gpu_device("sessions_on_one_context_run_from_many_threads") else { return };
    let l = on_gpu(&tiny_bundle());
    let want: Vec<Vec<Vec<f32>>> = {
        let s = Session::create(l.m, None).unwrap();
        TEXTS.iter().map(|t| s.embed(&[t], None).unwrap()).collect()
    };
    let m = l.m as usize;
    std::thread::scope(|scope| {
        for (i, text) in TEXTS.iter().enumerate() {
            let want = &want[i];
            scope.spawn(move || {
                let s = Session::create(m as *mut turbo_model, None).unwrap();
                for _ in 0..20 {
                    assert_eq!(&s.embed(&[text], None).unwrap(), want, "text {i}");
                }
            });
        }
    });
}

/// An output_dim the bundle lists is cut, then normalized, on the device.
#[test]
fn an_output_dim_is_cut_then_normalized_on_the_device() {
    let _t = turn();
    let Some(_) = gpu_device("an_output_dim_is_cut_then_normalized_on_the_device") else { return };
    let mut f = Fixture::new("levelzero-output-dims", {
        let mut m = model_manifest();
        m["embed"]["output_dims"] = json!([4]);
        m
    });
    f.weights("weights/model.safetensors", &tiny_weights(0));
    let l = f.load_on(gpu).unwrap();
    let s = Session::create(l.m, None).unwrap();
    let got = s.embed(&TEXTS, Some(&opts(|o| o.output_dim = 4))).unwrap();
    let plain = PlainBert::new(&f.dir);
    let tok = Tok::create(&f.dir).unwrap();
    for (t, g) in TEXTS.iter().zip(&got) {
        assert_eq!(g.len(), 4);
        let ids = tok.row(t, None).unwrap();
        let want = plain.embed(&ids, &vec![1; ids.len()], &vec![0; ids.len()], TURBO_POOLING_MEAN, 4, true);
        for (a, b) in g.iter().zip(&want) {
            assert!((*a as f64 - b).abs() < 1e-5, "{a} vs {b}");
        }
    }
}

/// After an append the driver refuses, its immediate list never finishes.
/// The call that saw the failure waits for the work appended before it,
/// which arrives whole, and the context gets a queue that works.
#[test]
fn a_failed_append_waits_for_what_came_before_and_replaces_the_queue() {
    let _t = turn();
    let Some(_) = gpu_device("a_failed_append_waits_for_what_came_before_and_replaces_the_queue") else { return };
    turbo::levelzero::append_failure_recovers().unwrap();
}

// ---- Limits and the largest shape ----------------------------------------------------------

/// A session longer than attention's local memory holds on this device is
/// refused by field 2. 65536 positions need 256 KiB of scores per
/// work-group, more than any device gives one. 32000 need 125 KiB, which
/// fits a B70's 128 KiB only without the 4 KiB its driver keeps for the
/// kernel's reductions: that is refused too, where a launch would fail. A
/// session just inside the limit runs a row of its full length.
#[test]
fn more_tokens_than_attention_holds_are_refused_by_field() {
    let _t = turn();
    let Some(_) = gpu_device("more_tokens_than_attention_holds_are_refused_by_field") else { return };
    let positions = 65536u64;
    let mut m = model_manifest();
    m["architecture"]["max_positions"] = json!(positions);
    m["embed"]["max_seq"] = json!(positions);
    // A case longer than max_seq, which a manifest needs: 600 paragraphs of
    // about 120 tokens.
    m["reference"]["cases"][8]["text"] = json!(vec![PARAGRAPH; 600].join(" "));
    let mut f = Fixture::new("levelzero-positions", m);
    let weights: Vec<Tensor> = tiny_weights(0)
        .into_iter()
        .map(|t| {
            if t.name != "embeddings.position_embeddings.weight" {
                return t;
            }
            let n = positions as usize * 8;
            let data = (0..n).flat_map(|i| (((i * 37 % 101) as f32 - 50.0) / 500.0).to_le_bytes()).collect();
            Tensor { shape: vec![positions, 8], data, ..t }
        })
        .collect();
    f.weights("weights/model.safetensors", &weights);
    let l = f.load_on(gpu).unwrap();
    let e = Session::create(l.m, Some(&session_desc(1, positions as u32, 0))).err().unwrap();
    assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 2), "{e:?}");
    assert!(e.message.contains("local memory"), "{e:?}");
    let e = Session::create(l.m, Some(&session_desc(1, 32000, 0))).err().unwrap();
    assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 2), "{e:?}");
    let s = Session::create(l.m, Some(&session_desc(1, 30000, 0))).unwrap();
    let row: Vec<i32> = (0..30000).map(|i| 1000 + i % 500).collect();
    s.write_tokens(&Tokens::new(&[row], 0).batch(), None).unwrap();
    let v = s.run().unwrap().rows();
    assert!(v[0].iter().all(|x| x.is_finite()), "{:?}", v[0]);
}

/// TURBO_TEST_BUNDLE, a relative path read from the workspace root, as
/// tests/conformance.rs reads it.
fn named_bundle() -> Option<std::path::PathBuf> {
    let p = std::path::PathBuf::from(std::env::var_os("TURBO_TEST_BUNDLE")?);
    Some(if p.is_absolute() { p } else { std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(p) })
}

/// One run at the session's largest shape, max_batch rows of max_seq
/// tokens, on the device and on the CPU: every row's cosine reaches
/// 0.9999. Rows are full or end early, so the padding is exercised too.
fn largest_shape_matches_the_cpu(dir: &std::path::Path) {
    let (g, c) = (on_gpu(dir), Loaded::load(dir).unwrap());
    let cs = Session::create(c.m, None).unwrap();
    // F32 at MODEL, and F16 on the matrix engines at FASTEST, each held
    // to its dtype's floor.
    for (precision, floor) in [(TURBO_PRECISION_MODEL, 0.9999), (TURBO_PRECISION_FASTEST, 0.999)] {
        let gs = Session::create(g.m, Some(&session_desc(0, 0, precision))).unwrap();
        largest_shape_on(dir, &gs, &cs, floor);
    }
}

fn largest_shape_on(dir: &std::path::Path, gs: &Session, cs: &Session, floor: f64) {
    let si = gs.info();
    let (batch, seq) = (si.max_batch as usize, si.max_seq as usize);
    let vocab = Tok::create(dir).unwrap().info().vocab_size as usize;
    let rows: Vec<Vec<i32>> = (0..batch)
        .map(|r| {
            let len = seq - (r * 7) % (seq / 2);
            (0..len).map(|p| (1000 + (r * 131 + p * 17) % (vocab - 1000)) as i32).collect()
        })
        .collect();
    let t = Tokens::new(&rows, 0);
    assert_eq!((t.batch as usize, t.seq as usize), (batch, seq));
    gs.write_tokens(&t.batch(), None).unwrap();
    let got = gs.run().unwrap().rows();
    cs.write_tokens(&t.batch(), None).unwrap();
    let want = cs.run().unwrap().rows();
    let mut lowest = 1f64;
    for (r, (a, b)) in got.iter().zip(&want).enumerate() {
        let cos = cosine(a, b);
        lowest = lowest.min(cos);
        assert!(cos >= floor, "row {r}: cosine {cos} with the cpu");
    }
    println!(
        "{}: {batch} rows of {seq} tokens in dtype {}, 1 - lowest cosine with the cpu {:.3e}",
        dir.display(),
        gs.info().compute_dtype,
        1.0 - lowest
    );
}

#[test]
fn the_largest_shape_matches_the_cpu() {
    let _t = turn();
    let Some(_) = gpu_device("the_largest_shape_matches_the_cpu") else { return };
    largest_shape_matches_the_cpu(&tiny_bundle());
}

#[test]
#[ignore = "needs a real bundle directory in TURBO_TEST_BUNDLE"]
fn the_largest_shape_of_a_real_bundle_matches_the_cpu() {
    let _t = turn();
    let dir = named_bundle().expect("TURBO_TEST_BUNDLE is not set");
    let Some(_) = gpu_device("the_largest_shape_of_a_real_bundle_matches_the_cpu") else { return };
    largest_shape_matches_the_cpu(&dir);
}
