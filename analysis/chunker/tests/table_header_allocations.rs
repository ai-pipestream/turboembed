//! An allocation regression, without a timing claim or instrumentation in production.
mod table_support;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use turbo_analysis_layers::{ChunkTableHeaders, Chunks};

struct Counter;
static COUNTING: AtomicBool = AtomicBool::new(false);
static BYTES: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            BYTES.fetch_add(size, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: Counter = Counter;

#[test]
fn header_stage_allocations_do_not_scale_with_body_cell_count() {
    let measure = |rows| {
        let mut doc = table_support::prepared(&table_support::table(rows));
        table_support::four_chunks(&mut doc);
        assert_eq!(doc.get::<Chunks>().unwrap().spans.len(), 4);
        BYTES.store(0, Ordering::Relaxed);
        COUNTING.store(true, Ordering::Relaxed);
        table_support::headers(&mut doc);
        COUNTING.store(false, Ordering::Relaxed);
        let bytes = BYTES.load(Ordering::Relaxed);
        assert_eq!(doc.get::<ChunkTableHeaders>().unwrap().contexts.len(), 4);
        bytes
    };
    let small = measure(30);
    let large = measure(3000);
    eprintln!("header stage allocated bytes: 30 rows={small}, 3000 rows={large}");
    assert!(large <= small + 4096, "body cells must not be collected or sorted per chunk: {small} -> {large}");
}
