//! Real dictionary lookup allocation contract.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
struct Counting;
thread_local! { static COUNT: Cell<usize> = const { Cell::new(0) }; }
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

#[test]
fn lowercase_dictionary_hits_and_misses_allocate_nothing() {
    let dict =
        turbo_lemmatizer::DictionaryLemmatizer::parse("dogs\tNNS\tdog\ncafé\tNN\tcafé\n".as_bytes(), "test").unwrap();
    let before = COUNT.get();
    for _ in 0..1000 {
        assert_eq!(dict.lemma("dogs", "NNS"), "dog");
        assert_eq!(dict.lemma("café", "NN"), "café");
        assert_eq!(dict.lemma("dogs", "missing"), "O");
        assert_eq!(dict.lemma("missing", "NN"), "O");
    }
    assert_eq!(COUNT.get() - before, 0);
}
