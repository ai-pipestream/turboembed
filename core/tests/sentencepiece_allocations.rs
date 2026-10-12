//! SentencePiece encoding allocates nothing once its scratch and output
//! have grown: the normalizer, the unigram search and BPE work in
//! buffers kept between calls. This binary counts every heap allocation
//! made on the test's thread; it holds one test.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::path::Path;

use turbo::sentencepiece::{Scratch, SentencePiece};

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

#[test]
fn a_grown_scratch_encodes_without_allocating() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata");
    let texts: Vec<String> = std::fs::read_to_string(dir.join("tokenizer-texts.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap()["text"].as_str().unwrap().to_owned())
        .collect();
    let long = texts.join("\n");
    // Unigram and BPE, each with and without byte fallback.
    for name in ["t5", "unigram-bytes", "bpe-identity", "bpe-bytes"] {
        let sp =
            SentencePiece::parse(&std::fs::read(dir.join(format!("sentencepiece/{name}.model"))).unwrap()).unwrap();
        let mut scratch = Scratch::default();
        let (mut pieces, mut ids) = (Vec::new(), Vec::new());
        sp.encode(&long, &mut scratch, &mut pieces);
        sp.encode_ids(&long, &mut scratch, &mut ids);
        let before = COUNT.with(Cell::get);
        for t in texts.iter().map(String::as_str).chain([long.as_str()]) {
            pieces.clear();
            sp.encode(t, &mut scratch, &mut pieces);
            ids.clear();
            sp.encode_ids(t, &mut scratch, &mut ids);
        }
        assert_eq!(COUNT.with(Cell::get) - before, 0, "{name}");
    }
}
