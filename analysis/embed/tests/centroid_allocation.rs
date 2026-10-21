//! A short centroid payload cannot reserve storage from an untrusted dimension.
//! The largest request is watched on the parsing thread only.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use turbo_analysis::{ModelRef, TagSet};
use turbo_analysis_embed::centroids::{CentroidNorm, LabelCentroids};

struct Tracked;
thread_local! {
    /// Whether this thread's requests are watched, and the largest one.
    static TRACK: Cell<bool> = const { Cell::new(false) };
    static MAX: Cell<usize> = const { Cell::new(0) };
}
unsafe impl GlobalAlloc for Tracked {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACK.try_with(Cell::get).unwrap_or(false) {
            let _ = MAX.try_with(|m| m.set(m.get().max(layout.size())));
        }
        // SAFETY: delegate the same allocation request to the system allocator.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: ptr and layout are supplied by the allocator contract.
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOC: Tracked = Tracked;

#[test]
fn wire_dimension_does_not_drive_allocation_before_payload_check() {
    let model = ModelRef { id: "test".into(), revision: "1".into(), sha256: "test".into() };
    let head =
        LabelCentroids::from_centroids(model, 1, TagSet::new("test", &["a"]), vec![1.0], 1.0, CentroidNorm::None)
            .unwrap();
    let mut bytes = head.to_bytes();
    bytes[8..12].copy_from_slice(&1_000_000u32.to_le_bytes());
    MAX.set(0);
    TRACK.set(true);
    let result = LabelCentroids::from_bytes(&bytes);
    TRACK.set(false);
    assert!(result.is_err());
    assert!(MAX.get() < 4096, "allocation of {} bytes", MAX.get());
}
