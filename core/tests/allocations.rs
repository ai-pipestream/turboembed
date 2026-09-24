//! turbo_result_info.host_allocs against the allocator: this binary counts
//! every heap allocation made on the test's own thread, and a run must
//! report exactly what was counted, which on the CPU is none, warm or
//! cold.

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use common::*;
use turbo::*;

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

/// Heap allocations `f` makes on this thread.
fn counted<T>(f: impl FnOnce() -> T) -> (T, u64) {
    let before = COUNT.with(Cell::get);
    let out = f();
    (out, COUNT.with(Cell::get) - before)
}

#[test]
fn a_run_reports_the_allocations_it_made() {
    let l = Loaded::load(&tiny_bundle()).unwrap();
    let s = Session::create(l.m, None).unwrap();
    let tok = Tok::create(&tiny_bundle()).unwrap();
    let texts = ["The quick brown fox jumps over the lazy dog.", "how do I reset a password", "a"];
    let rows: Vec<Vec<i32>> = texts.iter().map(|t| tok.row(t, None).unwrap()).collect();
    let small = Tokens::new(&rows, 0);
    let long = Tokens::new(&vec![(0..64).map(|i| 1000 + i).collect(); 64], 0);
    let mut dst = vec![0f32; 64 * 32];

    // The first run of the session, then warm runs of the same shape, a
    // smaller one and the session's largest.
    for (i, t) in [&small, &small, &long, &small, &long].into_iter().enumerate() {
        let b = t.batch();
        let (rc, writes) =
            counted(|| unsafe { turbo_embed_write_tokens(s.0, &b, std::ptr::null(), std::ptr::null_mut()) });
        assert_eq!(rc, 0);
        assert_eq!(writes, 0, "run {i}: writing tokens allocates nothing either");
        let mut r = std::ptr::null_mut();
        let (rc, allocs) = counted(|| unsafe { turbo_session_run(s.0, &mut r, std::ptr::null_mut()) });
        assert_eq!(rc, 0);
        let r = Outcome(r);
        let mut info: turbo_result_info = unsafe { std::mem::zeroed() };
        info.struct_size = size_of::<turbo_result_info>() as u32;
        let (rc, reads) = counted(|| unsafe {
            let rc = turbo_result_get_info(r.0, &mut info, std::ptr::null_mut());
            let bytes = (dst.len() * 4) as u64;
            rc | turbo_result_read(r.0, dst.as_mut_ptr() as *mut _, bytes, std::ptr::null_mut(), std::ptr::null_mut())
        });
        assert_eq!(rc, 0);
        assert_eq!(info.host_allocs, allocs, "run {i}: host_allocs is what the allocator counted");
        assert_eq!(allocs, 0, "run {i}");
        assert_eq!(reads, 0, "run {i}: reading the result allocates nothing");
        let (_, releases) = counted(|| drop(r));
        assert_eq!(releases, 0);
    }
    // The count is the allocator's, not a constant: a write of text
    // tokenizes, and that allocates.
    let views = texts.map(text);
    let (rc, n) =
        counted(|| unsafe { turbo_embed_write_text(s.0, views.as_ptr(), 3, std::ptr::null(), std::ptr::null_mut()) });
    assert_eq!(rc, 0);
    assert!(n > 0);
}
