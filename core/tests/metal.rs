//! The Metal backend through the C interface: its devices, its buffers in
//! every placement, import and export of Metal buffers and host memory,
//! weights read where the core holds them, a run's vectors left in shared
//! memory, and a warm run's allocation count. The encoder's vectors are
//! held to the f64 arithmetic of tests/common and to the CPU backend's;
//! tests/conformance.rs with TURBO_TEST_DEVICE=metal holds them to the
//! upstream reference.
//!
//! Built with the `metal` feature only, on macOS. A test that needs a
//! device that runs the kernels says it was skipped, and passes, when the
//! backend lists none; nothing is run on anything else in its place. With
//! TURBO_TEST_REQUIRE_METAL=1 it fails instead. docs/metal.md says how to
//! run them.

#![cfg(feature = "metal")]

mod common;

use std::ffi::{CStr, c_char, c_void};
use std::ptr;
use std::sync::{Mutex, MutexGuard};

use common::*;
use serde_json::json;
use turbo::status::{INVALID_ARGUMENT, UNSUPPORTED, UNSUPPORTED_OPTION};
use turbo::*;

// The Objective-C runtime, to look at a Metal buffer from the caller's
// side: each message is sent through objc_msgSend cast to its signature.
#[link(name = "objc")]
unsafe extern "C" {
    fn sel_registerName(name: *const c_char) -> *const c_void;
    fn objc_msgSend();
}

/// [obj sel], for a message that takes nothing and returns a word.
fn send(obj: u64, sel: &CStr) -> u64 {
    let f: unsafe extern "C" fn(u64, *const c_void) -> u64 = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(obj, sel_registerName(sel.as_ptr())) }
}

/// [device newBufferWithLength:n options:options], retained; release it
/// with `send(b, c"release")`.
fn new_buffer(device: u64, n: usize, options: usize) -> u64 {
    let f: unsafe extern "C" fn(u64, *const c_void, usize, usize) -> u64 =
        unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    let b = unsafe { f(device, sel_registerName(c"newBufferWithLength:options:".as_ptr()), n, options) };
    assert_ne!(b, 0);
    b
}

const STORAGE_SHARED: usize = 0;
const STORAGE_PRIVATE: usize = 2;
/// MTLResourceStorageModePrivate, as a resource option.
const OPTIONS_PRIVATE: usize = STORAGE_PRIVATE << 4;

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

    fn capability(&self, i: u32, precision: u32) -> turbo_capability {
        let mut out: turbo_capability = unsafe { std::mem::zeroed() };
        out.struct_size = size_of::<turbo_capability>() as u32;
        let mut err = new_error();
        let rc = unsafe { turbo_runtime_capability(self.0, i, TURBO_TASK_EMBED, precision, &mut out, &mut err) };
        assert_eq!(rc, 0, "{:?}", failure(rc, &err));
        out
    }

    /// The runtime's indexes of the devices the metal backend listed.
    fn metal(&self) -> Vec<u32> {
        (0..self.count()).filter(|&i| field(&self.info(i).backend) == "metal").collect()
    }
}

impl Drop for Rt {
    fn drop(&mut self) {
        unsafe { turbo_runtime_release(self.0) };
    }
}

/// TURBO_TEST_REQUIRE_METAL=1: a test that finds no device fails.
fn required() -> bool {
    std::env::var("TURBO_TEST_REQUIRE_METAL").is_ok_and(|v| v == "1")
}

/// The first Metal device that runs embed, if any.
fn runs_embed(rt: *mut turbo_runtime) -> Option<u32> {
    let rt = std::mem::ManuallyDrop::new(Rt(rt));
    rt.metal().into_iter().find(|&i| rt.capability(i, TURBO_PRECISION_MODEL).status != 0)
}

/// The first Metal device that runs embed, or None after saying the test
/// is skipped.
fn metal_device(test: &str) -> Option<u32> {
    let rt = Rt::new();
    let d = runs_embed(rt.0);
    if d.is_none() {
        assert!(!required(), "{test}: TURBO_TEST_REQUIRE_METAL=1 and no metal device runs embed");
        println!("{test}: skipped: no metal device runs embed");
    }
    d
}

fn metal(rt: *mut turbo_runtime) -> u32 {
    runs_embed(rt).expect("a metal device that runs embed")
}

fn on_metal(dir: &std::path::Path) -> Loaded {
    Loaded::load_on(dir, metal).unwrap_or_else(|e| panic!("{e:?}"))
}

fn opts(edit: impl FnOnce(&mut turbo_embed_options)) -> turbo_embed_options {
    let mut o = embed_options();
    edit(&mut o);
    o
}

fn table() -> &'static backend::turbo_backend {
    turbo::metal::backend()
}

// ---- Devices ---------------------------------------------------------------------

#[test]
fn the_backend_is_linked_before_the_cpu_and_names_itself() {
    let b = table();
    assert_eq!(b.name(), "metal");
    assert_eq!(b.struct_size as usize, size_of::<backend::turbo_backend>());
    backend::check_table(b).unwrap();
    assert!(b.buffer_read.is_some() && b.session_run.is_some());
    let linked = backend::linked();
    let at = |name: &str| linked.iter().position(|t| t.name() == name);
    assert!(at("metal") < at("cpu"), "its devices come before the cpu's");
    let v = unsafe { CStr::from_ptr(b.runtime_version) }.to_str().unwrap();
    let sdk = v.strip_prefix("Metal, macOS SDK ").unwrap_or_else(|| panic!("{v}"));
    assert!(sdk.split('.').all(|p| p.parse::<u32>().is_ok()), "{v}");
}

#[test]
fn the_gpu_is_listed_with_what_metal_says_about_it() {
    let rt = Rt::new();
    let metal = rt.metal();
    if metal.is_empty() {
        assert!(!required(), "TURBO_TEST_REQUIRE_METAL=1 and the metal backend lists no device");
        return println!("skipped: the metal backend lists no device");
    }
    for (ordinal, &i) in metal.iter().enumerate() {
        let info = rt.info(i);
        let name = field(&info.name);
        println!(
            "metal {ordinal}: {name} arch {} vendor {} kind {} memory {}/{} runtime {:?} driver {:?}",
            field(&info.arch),
            field(&info.vendor),
            info.kind,
            info.memory_free,
            info.memory_total,
            field(&info.runtime_version),
            field(&info.driver_version)
        );
        assert_eq!(info.ordinal, ordinal as u32, "ordinals count within the backend");
        assert!(!name.is_empty());
        if let Some(chip) = name.strip_prefix("Apple ") {
            // Apple silicon: one memory, shared with the host.
            assert_eq!(info.kind, TURBO_DEVICE_IGPU);
            assert_eq!(info.unified_memory, 1);
            assert_eq!(field(&info.vendor), "Apple");
            assert_eq!(field(&info.arch), chip.to_lowercase().replace(' ', ""), "{name}");
            assert!(info.memory_free > 0 && info.memory_free <= info.memory_total);
        }
        assert!(!field(&info.arch).is_empty());
        assert!(info.memory_total > 0);
        assert!(info.memory_free <= info.memory_total, "{} {}", info.memory_free, info.memory_total);
        assert_eq!(field(&info.runtime_version), unsafe { CStr::from_ptr(table().runtime_version) }.to_str().unwrap());
        assert!(field(&info.driver_version).starts_with("macOS "), "{}", field(&info.driver_version));
    }
}

#[test]
fn ordinals_name_the_same_device_in_every_runtime() {
    let (a, b) = (Rt::new(), Rt::new());
    let names = |rt: &Rt| rt.metal().iter().map(|&i| field(&rt.info(i).name)).collect::<Vec<_>>();
    assert_eq!(names(&a), names(&b));
}

#[test]
fn nothing_but_memory_free_differs_between_queries() {
    let rt = Rt::new();
    let Some(&i) = rt.metal().first() else {
        assert!(!required(), "TURBO_TEST_REQUIRE_METAL=1 and the metal backend lists no device");
        return println!("skipped: the metal backend lists no device");
    };
    let (mut a, mut b) = (rt.info(i), rt.info(i));
    a.memory_free = 0;
    b.memory_free = 0;
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
}

#[test]
fn the_table_refuses_an_ordinal_it_did_not_list() {
    // The core checks first; the table still answers only for its own.
    let b = table();
    let mut n = 0;
    assert_eq!(unsafe { (b.device_count)(&mut n, null_err()) }, 0);
    let mut info: turbo_device_info = unsafe { std::mem::zeroed() };
    let mut err = new_error();
    let rc = unsafe { (b.device_info)(n, &mut info, &mut err) };
    assert_eq!(rc, INVALID_ARGUMENT);
    assert_eq!(err.code, INVALID_ARGUMENT);
    assert_eq!(field(&err.message), format!("metal device {n}: {n} listed"));
    assert_eq!(unsafe { (b.device_info)(n, &mut info, null_err()) }, INVALID_ARGUMENT, "a NULL error is allowed");
    let (mut st, mut dt, mut oh) = (9, 9, 9);
    let mut reason = [0 as c_char; 8];
    let rc = unsafe {
        (b.capability)(n, TURBO_TASK_EMBED, 0, &mut st, &mut dt, &mut oh, reason.as_mut_ptr(), 8, null_err())
    };
    assert_eq!(rc, INVALID_ARGUMENT);
    assert_eq!((st, dt, oh), (9, 9, 9), "outputs are untouched on failure");
}

#[test]
fn a_reason_fits_the_buffer_it_is_given() {
    let b = table();
    if Rt::new().metal().is_empty() {
        assert!(!required(), "TURBO_TEST_REQUIRE_METAL=1 and the metal backend lists no device");
        return println!("skipped: the metal backend lists no device");
    }
    let (mut st, mut dt, mut oh) = (0, 0, 0);
    let mut reason = [b'x' as c_char; 16];
    let rc = unsafe {
        (b.capability)(0, TURBO_TASK_EMBED, 0, &mut st, &mut dt, &mut oh, reason.as_mut_ptr(), 8, null_err())
    };
    assert_eq!(rc, 0);
    // Empty where the device runs embed, else cut to fit.
    let end = reason.iter().position(|&c| c == 0).expect("terminated");
    assert!(end < 8, "within 8 bytes");
    assert!(reason[8..].iter().all(|&c| c == b'x' as c_char), "nothing written past reason_len");
    let mut reason = [b'x' as c_char; 4];
    let rc = unsafe {
        (b.capability)(0, TURBO_TASK_EMBED, 0, &mut st, &mut dt, &mut oh, reason.as_mut_ptr(), 0, null_err())
    };
    assert_eq!(rc, 0);
    assert!(reason.iter().all(|&c| c == b'x' as c_char), "reason_len 0 writes nothing");
}

// ---- Capability ------------------------------------------------------------------

#[test]
fn embed_is_offered_in_f32_as_sessions_run_it() {
    let _t = turn();
    let Some(_) = metal_device("embed_is_offered_in_f32_as_sessions_run_it") else { return };
    let l = on_metal(&tiny_bundle());
    let rt = std::mem::ManuallyDrop::new(Rt(l.rt));
    for p in [TURBO_PRECISION_MODEL, TURBO_PRECISION_FASTEST, TURBO_PRECISION_EXACT] {
        let cap = rt.capability(metal(l.rt), p);
        assert_eq!(cap.status, backend::TURBO_CAP_EXPERIMENTAL, "{}", field(&cap.reason));
        assert_eq!(field(&cap.reason), "no benchmark record for this cell");
        assert_eq!(cap.dtype, TURBO_DTYPE_F32);
        assert_eq!(cap.options_honored, 0b111111, "every field of turbo_embed_options");
        let info = Session::create(l.m, Some(&session_desc(0, 0, p))).unwrap().info();
        assert_eq!(info.compute_dtype, cap.dtype, "precision {p}");
    }
}

// ---- Buffers ---------------------------------------------------------------------

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

/// Host memory on whole pages, which Metal can map, freed on drop.
struct Pages(*mut u8, std::alloc::Layout);

impl Pages {
    fn new(bytes: usize) -> Pages {
        let page = 16384;
        let layout = std::alloc::Layout::from_size_align(bytes.div_ceil(page) * page, page).unwrap();
        Pages(unsafe { std::alloc::alloc_zeroed(layout) }, layout)
    }
}

impl Drop for Pages {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.0, self.1) };
    }
}

#[test]
fn each_placement_is_the_memory_it_names() {
    let _t = turn();
    let Some(dev) = metal_device("each_placement_is_the_memory_it_names") else { return };
    let rt = Rt::new();
    let ctx = context(&rt, dev);
    let values: Vec<f32> = (0..1000).map(|i| i as f32 * 0.5).collect();

    // DEVICE: a private Metal buffer, with no host address.
    let d = ctx.alloc(TURBO_PLACE_DEVICE, 1000).unwrap();
    assert!(d.host().unwrap_err().is(UNSUPPORTED, "no host pointer"));
    let h = d.export(TURBO_HANDLE_MTL_BUFFER).unwrap();
    assert_eq!((h.kind, h.offset), (TURBO_HANDLE_MTL_BUFFER, 0));
    assert_eq!(send(h.handle, c"storageMode") as usize, STORAGE_PRIVATE);
    assert!(send(h.handle, c"length") >= 4000);
    assert_eq!(send(h.handle, c"device"), h.aux, "aux is the buffer's MTLDevice");
    assert!(d.export(TURBO_HANDLE_HOST_PTR).unwrap_err().is(UNSUPPORTED, "no host address"));
    let device = h.aux;

    // PINNED and SHARED: shared Metal buffers, one address for both.
    for pl in [TURBO_PLACE_PINNED, TURBO_PLACE_SHARED] {
        let b = ctx.alloc(pl, 1000).unwrap();
        let p = b.host().unwrap();
        unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), p as *mut f32, 1000) };
        let h = b.export(TURBO_HANDLE_MTL_BUFFER).unwrap();
        assert_eq!((h.aux, h.offset), (device, 0));
        assert_eq!(send(h.handle, c"storageMode") as usize, STORAGE_SHARED, "placement {pl}");
        assert_eq!(send(h.handle, c"contents"), p as u64, "placement {pl}: the host address is the buffer's");
        assert_eq!(b.export(TURBO_HANDLE_HOST_PTR).unwrap().handle, p as u64);
    }

    // HOST: host memory, not a Metal buffer.
    let b = ctx.alloc(TURBO_PLACE_HOST, 1000).unwrap();
    let p = b.host().unwrap();
    assert_eq!(p as usize % 64, 0, "aligned");
    unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), p as *mut f32, 1000) };
    assert_eq!(b.export(TURBO_HANDLE_HOST_PTR).unwrap().handle, p as u64);
    assert!(b.export(TURBO_HANDLE_MTL_BUFFER).unwrap_err().is(UNSUPPORTED, "not a Metal buffer"));

    for kind in [TURBO_HANDLE_CUDA_PTR, TURBO_HANDLE_CL_MEM, TURBO_HANDLE_DMABUF_FD] {
        let e = d.export(kind).unwrap_err();
        assert_eq!(e.code, UNSUPPORTED, "{e:?}");
        assert!(e.message.contains("TURBO_HANDLE_"), "the kind is named: {e:?}");
    }
}

#[test]
fn import_wraps_metal_buffers_and_host_pages_without_a_copy() {
    let _t = turn();
    let Some(dev) = metal_device("import_wraps_metal_buffers_and_host_pages_without_a_copy") else { return };
    let rt = Rt::new();
    let ctx = context(&rt, dev);
    let device = ctx.alloc(TURBO_PLACE_DEVICE, 1).unwrap().export(TURBO_HANDLE_MTL_BUFFER).unwrap().aux;

    // A shared buffer of the caller's, eight floats in: the host address is
    // the buffer's own plus 32 bytes.
    let shared = new_buffer(device, 256, 0);
    let contents = send(shared, c"contents");
    let w = ctx.import(TURBO_PLACE_SHARED, 56, handle(TURBO_HANDLE_MTL_BUFFER, shared, device, 32)).unwrap();
    assert_eq!(w.host().unwrap() as u64, contents + 32);
    let h = w.export(TURBO_HANDLE_MTL_BUFFER).unwrap();
    assert_eq!((h.handle, h.aux, h.offset), (shared, device, 32), "the same buffer handed back");
    let e = ctx.import(TURBO_PLACE_SHARED, 64, handle(TURBO_HANDLE_MTL_BUFFER, shared, device, 32)).unwrap_err();
    assert!(e.is(INVALID_ARGUMENT, "past the buffer"), "{e:?}");
    let e = ctx.import(TURBO_PLACE_DEVICE, 8, handle(TURBO_HANDLE_MTL_BUFFER, shared, device, 0)).unwrap_err();
    assert!(e.is(INVALID_ARGUMENT, "placement: DEVICE"), "a shared buffer is not DEVICE memory: {e:?}");
    let e = ctx.import(TURBO_PLACE_SHARED, 8, handle(TURBO_HANDLE_MTL_BUFFER, shared, device + 16, 0)).unwrap_err();
    assert!(e.is(INVALID_ARGUMENT, "aux"), "another device is refused: {e:?}");
    let e = ctx.import(TURBO_PLACE_HOST, 8, handle(TURBO_HANDLE_MTL_BUFFER, shared, device, 0)).unwrap_err();
    assert!(e.is(INVALID_ARGUMENT, "placement: HOST"), "{e:?}");
    drop(w);

    // A private buffer as DEVICE, and never as SHARED.
    let private = new_buffer(device, 256, OPTIONS_PRIVATE);
    let w = ctx.import(TURBO_PLACE_DEVICE, 64, handle(TURBO_HANDLE_MTL_BUFFER, private, device, 0)).unwrap();
    assert!(w.host().is_err());
    let e = ctx.import(TURBO_PLACE_SHARED, 8, handle(TURBO_HANDLE_MTL_BUFFER, private, device, 0)).unwrap_err();
    assert!(e.is(INVALID_ARGUMENT, "private"), "{e:?}");
    let e = ctx.import(TURBO_PLACE_DEVICE, 8, handle(TURBO_HANDLE_MTL_BUFFER, 0, device, 0)).unwrap_err();
    assert!(e.is(INVALID_ARGUMENT, "NULL"), "{e:?}");
    drop(w);
    // The caller's buffers are still the caller's to release.
    send(shared, c"release");
    send(private, c"release");

    // Host memory as HOST anywhere, as PINNED or SHARED only on whole pages.
    let mut mine = vec![1.5f32; 16];
    let hp = mine.as_mut_ptr() as u64;
    let b = ctx.import(TURBO_PLACE_HOST, 16, handle(TURBO_HANDLE_HOST_PTR, hp, 0, 0)).unwrap();
    assert_eq!(b.host().unwrap() as u64, hp);
    let e = ctx.import(TURBO_PLACE_PINNED, 16, handle(TURBO_HANDLE_HOST_PTR, hp, 0, 0)).unwrap_err();
    assert!(e.is(INVALID_ARGUMENT, "whole pages"), "{e:?}");
    let e = ctx.import(TURBO_PLACE_DEVICE, 16, handle(TURBO_HANDLE_HOST_PTR, hp, 0, 0)).unwrap_err();
    assert!(e.is(INVALID_ARGUMENT, "host memory"), "{e:?}");
    let pages = Pages::new(16384);
    let pp = pages.0 as u64;
    let b = ctx.import(TURBO_PLACE_SHARED, 4096, handle(TURBO_HANDLE_HOST_PTR, pp, 0, 0)).unwrap();
    assert_eq!(b.host().unwrap() as u64, pp);
    let h = b.export(TURBO_HANDLE_MTL_BUFFER).unwrap();
    assert_eq!(send(h.handle, c"contents"), pp, "Metal reads the caller's pages where they are");
    drop(b);

    for kind in [TURBO_HANDLE_CUDA_PTR, TURBO_HANDLE_CL_MEM, TURBO_HANDLE_ZE_USM, TURBO_HANDLE_DMABUF_FD] {
        let e = ctx.import(TURBO_PLACE_DEVICE, 8, handle(kind, hp, 0, 0)).unwrap_err();
        assert_eq!(e.code, UNSUPPORTED, "{e:?}");
        assert!(e.message.contains("kind: TURBO_HANDLE_"), "the kind is named: {e:?}");
    }
    drop(mine);
}

// ---- Models ----------------------------------------------------------------------

#[test]
fn the_device_reads_the_weights_where_the_core_holds_them() {
    let _t = turn();
    let Some(_) = metal_device("the_device_reads_the_weights_where_the_core_holds_them") else { return };
    let l = on_metal(&tiny_bundle());
    assert_eq!(unsafe { model_metal_in_place(l.m) }, Some(true), "no copy of the weights");
    assert!(unsafe { model_converted_weights(l.m) }.is_none());
}

// ---- Runs ------------------------------------------------------------------------

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
    let Some(_) = metal_device("every_option_matches_the_arithmetic_and_the_cpu") else { return };
    let dir = tiny_bundle();
    let plain = PlainBert::new(&dir);
    let (g, c) = (on_metal(&dir), Loaded::load(&dir).unwrap());
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

/// The vectors stay in the session's shared memory: the result's buffer
/// is a shared Metal buffer whose contents are what turbo_result_read
/// copies, and nothing crosses to or from a device.
#[test]
fn a_run_leaves_its_vectors_in_shared_memory_and_nothing_crosses() {
    let _t = turn();
    let Some(_) = metal_device("a_run_leaves_its_vectors_in_shared_memory_and_nothing_crosses") else { return };
    let l = on_metal(&tiny_bundle());
    let mi = l.info();
    let s = Session::create(l.m, None).unwrap();

    s.write_text(&TEXTS, None).unwrap();
    let r = s.run().unwrap();
    let i = r.info();
    assert_eq!((i.task, i.batch, i.dim), (TURBO_TASK_EMBED, 3, 32));
    assert_eq!((i.dtype, i.compute_dtype, i.placement), (TURBO_DTYPE_F32, TURBO_DTYPE_F32, TURBO_PLACE_SHARED));
    assert_eq!(i.device, metal(l.rt));
    assert_eq!(i.bytes, 3 * 32 * 4);
    assert_eq!((i.h2d_bytes, i.d2h_bytes), (0, 0), "one memory: nothing crosses");
    assert_eq!((i.host_allocs, i.device_allocs), (0, 0));
    let (h, d, f, u) = (TURBO_STAGE_HOST, TURBO_STAGE_DEVICE, TURBO_STAGE_FUSED, TURBO_STAGE_UNUSED);
    assert_eq!(i.stage[..7], [h, u, d, d, d, f, u], "tokenize, upload, lookup, encode, pool, normalize, download");
    assert!(i.stage[7..].iter().all(|&s| s == u));
    assert_eq!(field(&i.backend), "metal");
    assert_eq!(field(&i.manifest_sha256), field(&mi.manifest_sha256));

    let read = r.rows().concat();
    assert_eq!(r.info().d2h_bytes, i.bytes, "a read is counted, as turbo.h says");

    let mut buf = ptr::null_mut();
    assert_eq!(unsafe { turbo_result_buffer(r.0, &mut buf, null_err()) }, 0);
    let buf = Buf(buf);
    let mut bd: turbo_buffer_desc = unsafe { std::mem::zeroed() };
    bd.struct_size = size_of::<turbo_buffer_desc>() as u32;
    assert_eq!(unsafe { turbo_buffer_get_desc(buf.0, &mut bd, null_err()) }, 0);
    assert_eq!((bd.placement, bd.shape, bd.bytes), (TURBO_PLACE_SHARED, [3, 32], 384));
    let p = buf.host().unwrap();
    assert_eq!(unsafe { std::slice::from_raw_parts(p as *const f32, read.len()) }, read.as_slice());
    let mh = buf.export(TURBO_HANDLE_MTL_BUFFER).unwrap();
    assert_eq!(send(mh.handle, c"contents") + mh.offset, p as u64, "the Metal buffer holds the vectors, no copy made");
    drop(buf);
    drop(r);

    // Tokens the caller wrote, with types: nothing is tokenized, and an
    // unnormalized vector is not normalized.
    let mut b = Tokens::new(&[vec![101, 7592, 102]], 0);
    b.types = Some(vec![0, 1, 1]);
    s.write_tokens(&b.batch(), Some(&opts(|o| o.normalize = TURBO_NORMALIZE_NONE))).unwrap();
    let i = s.run().unwrap().info();
    assert_eq!(i.stage[..7], [u, u, d, d, d, u, u]);
    assert_eq!((i.batch, i.h2d_bytes, i.d2h_bytes), (1, 0, 0), "a run's count starts again");
}

/// Rows in a shared buffer the caller got from the context give the same
/// vectors.
#[test]
fn rows_in_shared_memory_give_the_same_vectors() {
    let _t = turn();
    let Some(_) = metal_device("rows_in_shared_memory_give_the_same_vectors") else { return };
    let l = on_metal(&tiny_bundle());
    let s = Session::create(l.m, None).unwrap();
    let t = awkward_rows(&tiny_bundle());
    s.write_tokens(&t.batch(), None).unwrap();
    let want = s.run().unwrap().rows();

    let ctx = Ctx(l.ctx, false);
    let n = t.ids.len() as u64;
    let bufs: Vec<Buf> = (0..3).map(|_| ctx.alloc(TURBO_PLACE_SHARED, n).unwrap()).collect();
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
    assert_eq!(s.run().unwrap().rows(), want);
}

/// A warm run allocates nothing: the backend's own count does not move,
/// and the result says 0.
#[test]
fn a_warm_run_allocates_nothing() {
    let _t = turn();
    let Some(_) = metal_device("a_warm_run_allocates_nothing") else { return };
    let l = on_metal(&tiny_bundle());
    let s = Session::create(l.m, None).unwrap();
    let tok = Tok::create(&tiny_bundle()).unwrap();
    let rows: Vec<Vec<i32>> = TEXTS.iter().map(|t| tok.row(t, None).unwrap()).collect();
    let small = Tokens::new(&rows, 0);
    let long = Tokens::new(&vec![(0..64).map(|i| 1000 + i).collect(); 64], 0);
    let mut dst = vec![0f32; 64 * 32];
    let mut run = |t: &Tokens| {
        s.write_tokens(&t.batch(), None).unwrap();
        let r = s.run().unwrap();
        let i = r.info();
        let rc = unsafe {
            turbo_result_read(r.0, dst.as_mut_ptr() as *mut _, (dst.len() * 4) as u64, ptr::null_mut(), null_err())
        };
        assert_eq!(rc, 0);
        (i.host_allocs, i.device_allocs)
    };
    run(&small);
    run(&long);
    let before = turbo::metal::allocations();
    for (i, t) in [&small, &long, &small, &long].into_iter().enumerate() {
        assert_eq!(run(t), (0, 0), "warm run {i}: the result's count");
    }
    assert_eq!(turbo::metal::allocations(), before, "the backend allocated nothing in warm runs");
}

/// A model stored in F16 or BF16: MODEL is refused, EXACT and FASTEST
/// share one F32 copy on the device, and give the vectors of the same
/// values stored as F32.
#[test]
fn half_weights_compute_in_f32_from_one_shared_copy() {
    let _t = turn();
    let Some(_) = metal_device("half_weights_compute_in_f32_from_one_shared_copy") else { return };
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
        let mut f = Fixture::model(&format!("metal-half-{dtype}"));
        f.weights("weights/model.safetensors", &n);
        let l = f.load_on(metal).unwrap();
        let e = Session::create(l.m, None).err().unwrap();
        assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 3), "{dtype}: {e:?}");
        assert!(unsafe { model_converted_weights(l.m) }.is_none(), "a refused session makes no copy");
        let a = Session::create(l.m, Some(&session_desc(0, 0, TURBO_PRECISION_EXACT))).unwrap();
        let copy = unsafe { model_converted_weights(l.m) }.expect("the first F32 session made the copy");
        let b = Session::create(l.m, Some(&session_desc(0, 0, TURBO_PRECISION_FASTEST))).unwrap();
        assert_eq!(unsafe { model_converted_weights(l.m) }.unwrap(), copy, "one copy, shared");
        assert_eq!(b.info().compute_dtype, TURBO_DTYPE_F32);

        let mut g = Fixture::model(&format!("metal-half-{dtype}-wide"));
        g.weights("weights/model.safetensors", &w);
        let lw = g.load_on(metal).unwrap();
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
    let Some(_) = metal_device("sessions_on_one_context_run_from_many_threads") else { return };
    let l = on_metal(&tiny_bundle());
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
    let Some(_) = metal_device("an_output_dim_is_cut_then_normalized_on_the_device") else { return };
    let mut f = Fixture::new("metal-output-dims", {
        let mut m = model_manifest();
        m["embed"]["output_dims"] = json!([4]);
        m
    });
    f.weights("weights/model.safetensors", &tiny_weights(0));
    let l = f.load_on(metal).unwrap();
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

// ---- Limits and the largest shape ----------------------------------------------------

/// A session longer than attention's threadgroup memory holds on this
/// device is refused by field 2. 65536 positions need 256 KiB of scores
/// per threadgroup, more than any Apple GPU gives one.
#[test]
fn more_tokens_than_attention_holds_are_refused_by_field() {
    let _t = turn();
    let Some(_) = metal_device("more_tokens_than_attention_holds_are_refused_by_field") else { return };
    let positions = 65536u64;
    let mut m = model_manifest();
    m["architecture"]["max_positions"] = json!(positions);
    m["embed"]["max_seq"] = json!(positions);
    // A case longer than max_seq, which a manifest needs: 600 paragraphs of
    // about 120 tokens.
    m["reference"]["cases"][8]["text"] = json!(vec![PARAGRAPH; 600].join(" "));
    let mut f = Fixture::new("metal-positions", m);
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
    let l = f.load_on(metal).unwrap();
    let e = Session::create(l.m, Some(&session_desc(1, positions as u32, 0))).err().unwrap();
    assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 2), "{e:?}");
    assert!(e.message.contains("threadgroup memory"), "{e:?}");
    Session::create(l.m, Some(&session_desc(1, 512, 0))).unwrap();
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
    let (g, c) = (on_metal(dir), Loaded::load(dir).unwrap());
    let (gs, cs) = (Session::create(g.m, None).unwrap(), Session::create(c.m, None).unwrap());
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
        assert!(cos >= 0.9999, "row {r}: cosine {cos} with the cpu");
    }
    println!("{}: {batch} rows of {seq} tokens, 1 - lowest cosine with the cpu {:.3e}", dir.display(), 1.0 - lowest);
}

#[test]
fn the_largest_shape_matches_the_cpu() {
    let _t = turn();
    let Some(_) = metal_device("the_largest_shape_matches_the_cpu") else { return };
    largest_shape_matches_the_cpu(&tiny_bundle());
}

#[test]
#[ignore = "needs a real bundle directory in TURBO_TEST_BUNDLE"]
fn the_largest_shape_of_a_real_bundle_matches_the_cpu() {
    let _t = turn();
    let dir = named_bundle().expect("TURBO_TEST_BUNDLE is not set");
    let Some(_) = metal_device("the_largest_shape_of_a_real_bundle_matches_the_cpu") else { return };
    largest_shape_matches_the_cpu(&dir);
}
