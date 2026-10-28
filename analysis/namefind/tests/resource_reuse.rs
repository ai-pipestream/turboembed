//! Repeated rule lookup must not allocate once per immutable dictionary entry.
mod native_support;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use turbo_analysis::{Annotator, View};

thread_local! { static COUNT: Cell<usize> = const { Cell::new(0) }; }
struct Counting;
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        COUNT.with(|c| c.set(c.get() + 1));
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        COUNT.with(|c| c.set(c.get() + 1));
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        COUNT.with(|c| c.set(c.get() + 1));
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn allocations(entries: usize) -> usize {
    let finder = native_support::finder(entries, "person");
    let (doc, tokens) = native_support::input();
    let view = doc.get::<View>().unwrap();
    let warm = finder.annotate((view, &tokens)).unwrap().0;
    let before = COUNT.get();
    for _ in 0..20 {
        let result = finder.annotate((view, &tokens)).unwrap().0;
        assert_eq!(result.spans.columns(), warm.spans.columns());
        assert!(std::sync::Arc::ptr_eq(&result.kinds, &warm.kinds));
        std::hint::black_box(result);
    }
    COUNT.get() - before
}

#[test]
fn repeated_rule_documents_do_not_allocate_per_resource_entry() {
    let small = allocations(0);
    let large = allocations(2000);
    assert!(large <= small + 2, "resource allocation grew: {small} -> {large}");
}
