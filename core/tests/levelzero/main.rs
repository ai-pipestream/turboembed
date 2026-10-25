//! The levelzero backend through the C interface: its devices, checked
//! against what the kernel says is plugged in; its buffers in every
//! placement; import and export of USM pointers; a run's vectors left on
//! the device; the bytes a run moves; and a run's allocations, counted by
//! this binary's allocator and by the backend. The encoder's vectors are
//! held to the f64 arithmetic of tests/common, to the CPU backend's, and
//! to the bundle's reference vectors (tests/common/conformance.rs). Range
//! pooling is held to the row path's bits for a whole row and to the CPU
//! backend's range write for every other range.
//!
//! Built with the `levelzero` feature only. A test that needs a device
//! says it was skipped, and passes, when the backend lists none; nothing is
//! run on anything else in its place. With TURBO_TEST_REQUIRE_LEVELZERO=1
//! it fails instead, so a run on a machine with an Intel GPU cannot pass by
//! finding none. docs/levelzero.md says how to run them.

#![cfg(feature = "levelzero")]

mod buffers;
#[path = "../common/mod.rs"]
mod common;
mod devices;
mod limits;
mod models;
mod ranges;
mod runs;

use buffers::{Buf, Ctx};
use runs::awkward_rows;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::ffi::{c_char, c_void};
use std::ptr;
use std::sync::{Mutex, MutexGuard, OnceLock};

use common::*;
use serde_json::json;
use turbo::status::{CAPACITY, INVALID_ARGUMENT, UNSUPPORTED, UNSUPPORTED_OPTION};
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
