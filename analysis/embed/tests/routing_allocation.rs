use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::{path::Path, time::Duration};
use turbo::bundle::Bundle;
use turbo_analysis_embed::routing::*;
use turbo_vector_index::Normalization;
struct Tracked;
static TRACK: AtomicBool = AtomicBool::new(false);
static COUNT: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Tracked {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACK.load(Ordering::Relaxed) {
            COUNT.fetch_add(1, Ordering::Relaxed);
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
    TRACK.store(true, Ordering::Relaxed);
    for _ in 0..10_000 {
        std::hint::black_box(router.route(std::hint::black_box(request)).unwrap());
    }
    TRACK.store(false, Ordering::Relaxed);
    assert_eq!(COUNT.load(Ordering::Relaxed), 0);
}
