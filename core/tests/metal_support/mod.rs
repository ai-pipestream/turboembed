//! What the Metal tests share: the Objective-C runtime to look at a Metal
//! buffer from the caller's side, the runtime's Metal devices, and finding
//! one that runs embed, or saying the test is skipped.

#![allow(dead_code)]

use std::ffi::{CStr, c_char, c_void};
use std::ptr;
use std::sync::{Mutex, MutexGuard};

use crate::common::*;
use turbo::*;

// The Objective-C runtime, to look at a Metal buffer from the caller's
// side: each message is sent through objc_msgSend cast to its signature.
#[link(name = "objc")]
unsafe extern "C" {
    pub fn sel_registerName(name: *const c_char) -> *const c_void;
    pub fn objc_msgSend();
}

/// [obj sel], for a message that takes nothing and returns a word.
pub fn send(obj: u64, sel: &CStr) -> u64 {
    let f: unsafe extern "C" fn(u64, *const c_void) -> u64 = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(obj, sel_registerName(sel.as_ptr())) }
}

/// [device newBufferWithLength:n options:options], retained; release it
/// with `send(b, c"release")`.
pub fn new_buffer(device: u64, n: usize, options: usize) -> u64 {
    let f: unsafe extern "C" fn(u64, *const c_void, usize, usize) -> u64 =
        unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    let b = unsafe { f(device, sel_registerName(c"newBufferWithLength:options:".as_ptr()), n, options) };
    assert_ne!(b, 0);
    b
}

pub const STORAGE_SHARED: usize = 0;
pub const STORAGE_PRIVATE: usize = 2;
/// MTLResourceStorageModePrivate, as a resource option.
pub const OPTIONS_PRIVATE: usize = STORAGE_PRIVATE << 4;

/// The tests take turns: the allocation counts are the process's, and
/// another test's work would move them.
pub static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

pub fn turn() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner())
}

pub fn null_err() -> *mut turbo_error {
    ptr::null_mut()
}

pub const TEXTS: [&str; 3] = ["The quick brown fox jumps over the lazy dog.", "how do I reset a password", "a"];

pub struct Rt(pub *mut turbo_runtime);

impl Rt {
    pub fn new() -> Rt {
        let mut rt = ptr::null_mut();
        assert_eq!(unsafe { turbo_runtime_create(ptr::null(), &mut rt, null_err()) }, 0);
        Rt(rt)
    }

    pub fn count(&self) -> u32 {
        let mut n = 0;
        assert_eq!(unsafe { turbo_runtime_device_count(self.0, &mut n, null_err()) }, 0);
        n
    }

    pub fn info(&self, i: u32) -> turbo_device_info {
        let mut out: turbo_device_info = unsafe { std::mem::zeroed() };
        out.struct_size = size_of::<turbo_device_info>() as u32;
        let mut err = new_error();
        let rc = unsafe { turbo_runtime_device_info(self.0, i, &mut out, &mut err) };
        assert_eq!(rc, 0, "{:?}", failure(rc, &err));
        out
    }

    pub fn capability(&self, i: u32, precision: u32) -> turbo_capability {
        let mut out: turbo_capability = unsafe { std::mem::zeroed() };
        out.struct_size = size_of::<turbo_capability>() as u32;
        let mut err = new_error();
        let rc = unsafe { turbo_runtime_capability(self.0, i, TURBO_TASK_EMBED, precision, &mut out, &mut err) };
        assert_eq!(rc, 0, "{:?}", failure(rc, &err));
        out
    }

    /// The runtime's indexes of the devices the metal backend listed.
    pub fn metal(&self) -> Vec<u32> {
        (0..self.count()).filter(|&i| field(&self.info(i).backend) == "metal").collect()
    }
}

impl Drop for Rt {
    fn drop(&mut self) {
        unsafe { turbo_runtime_release(self.0) };
    }
}

/// TURBO_TEST_REQUIRE_METAL=1: a test that finds no device fails.
pub fn required() -> bool {
    std::env::var("TURBO_TEST_REQUIRE_METAL").is_ok_and(|v| v == "1")
}

/// The first Metal device that runs embed, if any.
pub fn runs_embed(rt: *mut turbo_runtime) -> Option<u32> {
    let rt = std::mem::ManuallyDrop::new(Rt(rt));
    rt.metal().into_iter().find(|&i| rt.capability(i, TURBO_PRECISION_MODEL).status != 0)
}

/// The first Metal device that runs embed, or None after saying the test
/// is skipped.
pub fn metal_device(test: &str) -> Option<u32> {
    let rt = Rt::new();
    let d = runs_embed(rt.0);
    if d.is_none() {
        assert!(!required(), "{test}: TURBO_TEST_REQUIRE_METAL=1 and no metal device runs embed");
        println!("{test}: skipped: no metal device runs embed");
    }
    d
}

pub fn metal(rt: *mut turbo_runtime) -> u32 {
    runs_embed(rt).expect("a metal device that runs embed")
}

pub fn on_metal(dir: &std::path::Path) -> Loaded {
    Loaded::load_on(dir, metal).unwrap_or_else(|e| panic!("{e:?}"))
}

pub fn opts(edit: impl FnOnce(&mut turbo_embed_options)) -> turbo_embed_options {
    let mut o = embed_options();
    edit(&mut o);
    o
}

pub fn table() -> &'static backend::turbo_backend {
    turbo::metal::backend()
}

/// Rows with padding on either side, a masked token inside a row and token
/// types, as tests/sessions.rs writes them for the CPU.
pub fn awkward_rows(dir: &std::path::Path) -> Tokens {
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

// ---- Buffers ---------------------------------------------------------------------

/// A context, released on drop when `owned`.
pub struct Ctx(pub *mut turbo_context, pub bool);

impl Ctx {
    pub fn alloc(&self, placement: u32, n: u64) -> Result<Buf, Failure> {
        let d = desc(placement, n);
        let mut b = ptr::null_mut();
        let mut err = new_error();
        match unsafe { turbo_buffer_alloc(self.0, &d, &mut b, &mut err) } {
            0 => Ok(Buf(b)),
            rc => Err(failure(rc, &err)),
        }
    }

    pub fn import(&self, placement: u32, n: u64, h: turbo_native_handle) -> Result<Buf, Failure> {
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
pub struct Buf(pub *mut turbo_buffer);

impl Buf {
    pub fn host(&self) -> Result<*mut c_void, Failure> {
        let mut p = ptr::null_mut();
        let mut err = new_error();
        match unsafe { turbo_buffer_host_ptr(self.0, &mut p, &mut err) } {
            0 => Ok(p),
            rc => Err(failure(rc, &err)),
        }
    }

    pub fn export(&self, kind: u32) -> Result<turbo_native_handle, Failure> {
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
pub fn desc(placement: u32, n: u64) -> turbo_buffer_desc {
    turbo_buffer_desc {
        struct_size: size_of::<turbo_buffer_desc>() as u32,
        placement,
        dtype: TURBO_DTYPE_F32,
        ndim: 1,
        shape: [n, 0],
        bytes: 0,
    }
}

pub fn handle(kind: u32, at: u64, aux: u64, offset: u64) -> turbo_native_handle {
    turbo_native_handle { struct_size: size_of::<turbo_native_handle>() as u32, kind, handle: at, aux, offset }
}

pub fn context(rt: &Rt, device: u32) -> Ctx {
    let mut c = ptr::null_mut();
    let mut err = new_error();
    let rc = unsafe { turbo_context_create(rt.0, device, &mut c, &mut err) };
    assert_eq!(rc, 0, "{:?}", failure(rc, &err));
    Ctx(c, true)
}

/// Host memory on whole pages, which Metal can map, freed on drop.
pub struct Pages(pub *mut u8, pub std::alloc::Layout);

impl Pages {
    pub fn new(bytes: usize) -> Pages {
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
