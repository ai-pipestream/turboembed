use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use turbo_analysis::{Document, Key, NormalizationXray, Provenance, View, ViewBuilder, ViewId};

struct Count;
thread_local! {
    static WATCH: Cell<bool> = const { Cell::new(false) };
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}
fn allocation() {
    let _ = WATCH.try_with(|watch| {
        if watch.get() {
            let _ = ALLOCS.try_with(|n| n.set(n.get() + 1));
        }
    });
}
unsafe impl GlobalAlloc for Count {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        allocation();
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        allocation();
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        allocation();
        unsafe { System.realloc(p, l, n) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static ALLOCATOR: Count = Count;

#[test]
fn iteration_columns_and_borrowed_slices_allocate_nothing() {
    let mut doc = Document::new("aé中😀z");
    let source = doc.get::<View>().unwrap().clone();
    let mut b = ViewBuilder::new(source.text());
    b.keep(1);
    b.replace(3, "e");
    let (text, alignment) = b.finish();
    let key = Key::named("normalized");
    doc.add(&key, View::derived(&source, text, alignment), Provenance::by("test:edit")).unwrap();
    let child = doc.get_key(&key).unwrap();
    let x = NormalizationXray::between(&doc, ViewId::SOURCE, child.id()).unwrap();
    ALLOCS.with(|n| n.set(0));
    WATCH.with(|n| n.set(true));
    for _ in 0..1000 {
        for (i, r) in x.iter().enumerate() {
            std::hint::black_box((r, x.parent_slice(i, source.text()), x.child_slice(i, child.text())));
        }
        std::hint::black_box((x.parent_start(), x.parent_end(), x.child_start(), x.child_end(), x.kinds(), x.steps()));
    }
    WATCH.with(|n| n.set(false));
    assert_eq!(ALLOCS.with(Cell::get), 0);
}
