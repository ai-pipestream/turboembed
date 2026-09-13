//! FFI for `include/turbo_buffer.h` used by the ORT embed path.
//!
//! The C++ stub owns the engine arena. This module rents I/O views from
//! that arena and exposes the process-wide alloc counters for tests.

#![allow(non_camel_case_types)]

use std::os::raw::c_void;
use std::ptr;

pub const TURBO_BUFFER_OK: i32 = 0;

pub const TURBO_BUFFER_DTYPE_I32: i32 = 1;
pub const TURBO_BUFFER_DTYPE_F32: i32 = 2;

pub const TURBO_BUFFER_PLACE_HOST: i32 = 1;
pub const TURBO_BUFFER_PLACE_DEVICE: i32 = 2;
pub const TURBO_BUFFER_PLACE_PINNED: i32 = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct turbo_buffer_view {
    pub ptr: *mut c_void,
    pub rows: u32,
    pub cols: u32,
    pub row_stride: u32,
    pub dtype: i32,
    pub device: i32,
    pub placement: i32,
    pub handle: u32,
}

unsafe impl Send for turbo_buffer_view {}
unsafe impl Sync for turbo_buffer_view {}

impl turbo_buffer_view {
    pub fn empty() -> Self {
        Self {
            ptr: ptr::null_mut(),
            rows: 0,
            cols: 0,
            row_stride: 0,
            dtype: 0,
            device: 0,
            placement: 0,
            handle: 0,
        }
    }

}

pub type turbo_buffer_arena = c_void;

unsafe extern "C" {
    pub fn turbo_buffer_arena_rent(
        arena: *mut turbo_buffer_arena,
        dtype: i32,
        placement: i32,
        rows: u32,
        cols: u32,
        row_stride: u32,
        out: *mut turbo_buffer_view,
    ) -> i32;

    pub fn turbo_buffer_arena_return(
        arena: *mut turbo_buffer_arena,
        view: *mut turbo_buffer_view,
    ) -> i32;

    pub fn turbo_buffer_cuda_mapped_device_ptr(
        host_ptr: *const c_void,
        device_ptr: *mut *mut c_void,
    ) -> i32;

    pub fn turbo_buffer_alloc_counter_reset();
    pub fn turbo_buffer_alloc_counter() -> u64;
    pub fn turbo_buffer_cuda_forward_enter();
    pub fn turbo_buffer_cuda_forward_leave();
    pub fn turbo_buffer_cuda_forward_allocs_reset();
    pub fn turbo_buffer_cuda_forward_allocs() -> u64;
    pub fn turbo_buffer_cuda_forward_h2d_reset();
    pub fn turbo_buffer_cuda_forward_h2d_bytes() -> u64;
    pub fn turbo_buffer_cuda_forward_h2d_calls() -> u64;
    pub fn turbo_buffer_last_error(arena: *const turbo_buffer_arena) -> *const i8;
}

pub fn last_error(arena: *const turbo_buffer_arena) -> String {
    unsafe {
        let p = turbo_buffer_last_error(arena);
        if p.is_null() {
            return String::new();
        }
        std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}

pub fn rent(
    arena: *mut turbo_buffer_arena,
    dtype: i32,
    placement: i32,
    rows: u32,
    cols: u32,
    row_stride: u32,
) -> Result<turbo_buffer_view, String> {
    if arena.is_null() {
        return Err("turbo_buffer arena is null".into());
    }
    let mut view = turbo_buffer_view::empty();
    let st = unsafe {
        turbo_buffer_arena_rent(
            arena,
            dtype,
            placement,
            rows,
            cols,
            row_stride,
            &mut view,
        )
    };
    if st != TURBO_BUFFER_OK || view.ptr.is_null() {
        return Err(format!(
            "turbo_buffer_arena_rent failed ({st}): {}",
            last_error(arena)
        ));
    }
    Ok(view)
}

pub fn return_view(arena: *mut turbo_buffer_arena, view: &mut turbo_buffer_view) {
    if arena.is_null() || view.ptr.is_null() {
        *view = turbo_buffer_view::empty();
        return;
    }
    unsafe {
        let _ = turbo_buffer_arena_return(arena, view);
    }
    *view = turbo_buffer_view::empty();
}

pub fn mapped_device_ptr(host: *const c_void) -> Result<*mut c_void, String> {
    let mut dev = ptr::null_mut();
    let hit = unsafe { turbo_buffer_cuda_mapped_device_ptr(host, &mut dev) };
    if hit != 1 || dev.is_null() {
        return Err(
            "PINNED token row is not cudaHostAllocMapped; refusing a \
             convenience H2D stand-in"
                .into(),
        );
    }
    Ok(dev)
}
