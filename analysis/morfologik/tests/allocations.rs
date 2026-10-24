//! Allocation checks using real dictionary tables and compiled FSA bytes.
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

use turbo_morfologik::{
    MorfologikDictionary, MorfologikDictionaryBuilder, MorfologikDictionaryReader, PoliMorfDictionaryReader,
};

#[test]
fn expanded_dictionary_hits_and_misses_borrow_keys() {
    let dict = PoliMorfDictionaryReader::read("dogs\tdog\tNNS\ncafé\tcafé\tNN\n".as_bytes()).unwrap();
    let before = COUNT.get();
    for _ in 0..1000 {
        assert_eq!(dict.lemma("dogs", "NNS"), "dog");
        assert_eq!(dict.lemma("café", "NN"), "café");
        assert_eq!(dict.lemma("dogs", "missing"), "O");
        assert_eq!(dict.lemma("missing", "NN"), "O");
    }
    assert_eq!(COUNT.get() - before, 0);
}

#[test]
fn unused_conversion_rules_do_not_allocate() {
    let info = b"fsa.dict.separator=+\nfsa.dict.encoding=UTF-8\nfsa.dict.encoder=NONE\nfsa.dict.input-conversion=x y\n";
    let bytes = MorfologikDictionaryBuilder::compile_bytes(b"dog+dogs+NNS\n", info).unwrap();
    let dict = MorfologikDictionaryReader::read(&bytes, info).unwrap();
    let before = COUNT.get();
    for _ in 0..1000 {
        assert_eq!(dict.lemma("dogs", "NNS"), "dog");
    }
    assert_eq!(COUNT.get() - before, 0);
}

#[test]
fn first_lemma_does_not_allocate_each_candidate_record() {
    let info = b"fsa.dict.separator=+\nfsa.dict.encoding=UTF-8\nfsa.dict.encoder=NONE\n";
    let table: String = (0..100).map(|i| format!("lemma{i:03}+word+N\n")).collect();
    let bytes = MorfologikDictionaryBuilder::compile_bytes(table.as_bytes(), info).unwrap();
    let dict = MorfologikDictionary::from_bytes(&bytes, info).unwrap();
    let expected = dict.lookup("word");
    assert_eq!(expected.len(), 100);
    let before = COUNT.get();
    let first = dict.first_lemma("word", "N");
    let allocations = COUNT.get() - before;
    assert_eq!(first.as_deref(), Some(expected[0].stem.as_str()));
    // Two traversal buffers, one decoded byte buffer and the returned string.
    // No per-candidate word/tag copies or result vector.
    assert!(allocations <= 4, "{allocations} allocations for the first lemma");
}
