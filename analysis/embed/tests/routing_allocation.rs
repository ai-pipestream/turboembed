//! Routing a request allocates nothing, counted on the routing thread only:
//! other threads' allocations, such as the test harness's, are not
//! routing's.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::{path::Path, time::Duration};
use turbo::bundle::Bundle;
use turbo_analysis_embed::routing::*;
use turbo_vector_index::Normalization;
struct Tracked;
thread_local! {
    /// Whether this thread's allocations are counted, and how many.
    /// Const-initialized, so reading them never allocates.
    static TRACK: Cell<bool> = const { Cell::new(false) };
    static COUNT: Cell<usize> = const { Cell::new(0) };
}
unsafe impl GlobalAlloc for Tracked {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACK.try_with(Cell::get).unwrap_or(false) {
            let _ = COUNT.try_with(|c| c.set(c.get() + 1));
        }
        // SAFETY: forward the allocator contract unchanged.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forward the allocator contract unchanged.
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOC: Tracked = Tracked;
#[test]
fn successful_routing_allocates_nothing() {
    let b = Bundle::open(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle")).unwrap();
    let p = ModelProfile::from_bundle(
        &b,
        vec![RoleCapability {
            role: Role::DenseQuery,
            width: 32,
            normalization: Normalization::Unit,
            context_limit: 64,
            cost_per_item: Duration::from_micros(10),
        }],
    )
    .unwrap();
    let router = ModelRouter::new(vec![LoadedModel::new(0, p)], RoutingPolicy::Fastest).unwrap();
    let request = Request {
        task: Task::Query,
        budget: LatencyBudget::MaxLatency(Duration::from_millis(1)),
        items: 1,
        tokens: 16,
        width: Some(32),
        normalization: Some(Normalization::Unit),
        space: None,
    };
    // Another thread allocates the whole time routing runs; none of it is
    // routing's.
    let stop = Arc::new(AtomicBool::new(false));
    let noise = {
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                std::hint::black_box(vec![0u8; 64]);
            }
        })
    };
    TRACK.set(true);
    for _ in 0..10_000 {
        std::hint::black_box(router.route(std::hint::black_box(request)).unwrap());
    }
    TRACK.set(false);
    stop.store(true, Ordering::Relaxed);
    noise.join().unwrap();
    assert_eq!(COUNT.get(), 0);
}
