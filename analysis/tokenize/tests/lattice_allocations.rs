//! The lattice tokenizer allocates nothing once its `Scratch` and output
//! have grown to what the texts need. After one analysis of the texts
//! joined, each text and the join again allocate nothing. This binary
//! counts every heap allocation made on the test's thread; it holds one
//! test.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use turbo_goldens::data_dir;
use turbo_tokenize::lattice::Scratch;
use turbo_tokenize::{LatticeTokenizer, MecabDictionary, WhitespaceMode};

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

/// The texts of an input set, one per file.
fn inputs(set: &str) -> Vec<String> {
    let dir = data_dir().join("inputs").join(set);
    let mut paths: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().path()).collect();
    paths.sort();
    paths.iter().map(|p| std::fs::read_to_string(p).unwrap()).collect()
}

#[test]
fn a_grown_scratch_analyzes_without_allocating() {
    for (dictionary, set) in [("mecab-fixture-ja", "ja"), ("mecab-fixture-ko", "ko")] {
        let dir = data_dir().join("fixtures").join(dictionary);
        let d = MecabDictionary::load(&dir, WhitespaceMode::Unicode).unwrap();
        let t = LatticeTokenizer::new(d, WhitespaceMode::Unicode);
        let texts = inputs(set);
        let long = texts.join("\n");
        let mut scratch = Scratch::new();
        let (mut morphemes, mut spans) = (Vec::new(), Vec::new());
        t.analyze_with(&long, &mut scratch, &mut morphemes).unwrap();
        t.tokenize_with(&long, &mut scratch, &mut spans).unwrap();
        let before = COUNT.with(Cell::get);
        for text in texts.iter().map(String::as_str).chain([long.as_str()]) {
            morphemes.clear();
            t.analyze_with(text, &mut scratch, &mut morphemes).unwrap();
            spans.clear();
            t.tokenize_with(text, &mut scratch, &mut spans).unwrap();
        }
        assert_eq!(COUNT.with(Cell::get) - before, 0, "{dictionary}");
    }
}
