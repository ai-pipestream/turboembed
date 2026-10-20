//! Allocation contract for the binary prediction path and flat batches.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use turbo_analysis::Layer;
use turbo_analysis_layers::{Calibrated, Tokens};
use turbo_calibration::*;
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
fn small_rows_allocate_nothing_and_batches_grow_by_columns() {
    let id = Identity {
        model: "m".into(),
        revision: "1".into(),
        sha256: digest(b"m"),
        task: Task::TokenTagging,
        domain: "test".into(),
        labels: vec!["a".into(), "b".into()],
        inference_hash: 1,
    };
    let a = Calibration::new(
        id.clone(),
        Transform::Temperature(2.0),
        DataIdentity { split: "fit".into(), sha256: digest(b"fit") },
        DataIdentity { split: "conf".into(), sha256: digest(b"conf") },
        vec![0.7; 20],
    )
    .unwrap();
    let p = Predictor::new(a, &id, Policy { coverage: 0.9, max_set_size: 2, min_confidence: 0.0 }).unwrap();
    let before = COUNT.get();
    for _ in 0..100 {
        let mut side = Calibrated::<Tokens>::new(&p);
        side.push(&p, &[0.8, 0.2]).unwrap();
        side.check().unwrap();
        std::hint::black_box(side);
    }
    assert_eq!(COUNT.get() - before, 0);
    let mut side = Calibrated::<Tokens>::new(&p);
    let before = COUNT.get();
    for _ in 0..1024 {
        side.push(&p, &[0.8, 0.2]).unwrap();
    }
    // Five flat columns grow geometrically, independent of individual rows.
    assert!(COUNT.get() - before < 64);
    let before = COUNT.get();
    side.check().unwrap();
    assert_eq!(COUNT.get() - before, 0);
    assert_eq!(side.len(), 1024);
}
