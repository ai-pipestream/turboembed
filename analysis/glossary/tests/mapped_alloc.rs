//! Loading an image and matching over it allocate nothing once the
//! caller's buffers have grown to fit, counted by the global allocator.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use turbo_glossary::{Boundaries, Case, Glossary, ImageBuf, MappedGlossaryMatcher, MappedGlossaryWriter, Scratch};

struct Counting;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: forwards every call to the system allocator, counting them.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

#[test]
fn loading_and_matching_allocate_nothing() {
    let mut g = Glossary::new();
    for (i, t) in ["New York", "York", "東京", "Größe", "hot dog", "dog", "he", "she", "hers"].iter().enumerate() {
        g.push(&format!("id{i}"), t).unwrap();
    }
    let image = ImageBuf::copy_from(&MappedGlossaryWriter::write(&g, Case::Insensitive).unwrap());
    let texts = ["ushers in NEW YORK", "a HOT DOG in 東京 of GRÖSSE größe", "", "she sells hers"];
    let bounds: Vec<Boundaries> =
        texts.iter().map(|t| Boundaries::from_offsets(t.len() as u32, 0..=t.len() as u32)).collect();
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    let m = MappedGlossaryMatcher::from_bytes(image.bytes()).unwrap();
    assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), before, "loading allocated");
    let mut scratch = Scratch::new();
    let mut out = Vec::new();
    // Grow the buffers once.
    for (t, b) in texts.iter().zip(&bounds) {
        m.find(t, b, &mut scratch, &mut out);
    }
    out.reserve(1024);
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    let mut found = 0;
    for _ in 0..1000 {
        for (t, b) in texts.iter().zip(&bounds) {
            out.clear();
            m.find(t, b, &mut scratch, &mut out);
            found += out.len();
            m.scan(t, &mut scratch, |_, _| found += 1);
        }
    }
    assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), before, "matching allocated");
    assert!(found > 10_000, "{found}");
    // The count is not blind: the in-heap matcher's find allocates its ring
    // and candidates on every call.
    let heap = turbo_glossary::TextMatcher::new(&g, Case::Insensitive).unwrap();
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    out.clear();
    heap.find(texts[0], &bounds[0], &mut out);
    assert!(ALLOCATIONS.load(Ordering::Relaxed) > before);
}
