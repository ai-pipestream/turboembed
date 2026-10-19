//! A short centroid payload cannot reserve storage from an untrusted dimension.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use turbo_analysis::{ModelRef, TagSet};
use turbo_analysis_embed::centroids::{CentroidNorm, LabelCentroids};

struct Tracked;
static TRACK: AtomicBool = AtomicBool::new(false);
static MAX: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Tracked {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACK.load(Ordering::Relaxed) {
            MAX.fetch_max(layout.size(), Ordering::Relaxed);
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
    MAX.store(0, Ordering::Relaxed);
    TRACK.store(true, Ordering::Relaxed);
    let result = LabelCentroids::from_bytes(&bytes);
    TRACK.store(false, Ordering::Relaxed);
    assert!(result.is_err());
    assert!(MAX.load(Ordering::Relaxed) < 4096, "allocation of {} bytes", MAX.load(Ordering::Relaxed));
}
