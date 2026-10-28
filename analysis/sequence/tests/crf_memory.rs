//! Measures actual allocator traffic around numerical decoding, not a mock.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use turbo_analysis::{Tag, TagSet};
use turbo_sequence::crf::{CrfScores, EntityDecoder, EntityEncoding, EntityLabel};

struct Tracking;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn added(bytes: usize) {
    let live = LIVE.fetch_add(bytes, Relaxed) + bytes;
    PEAK.fetch_max(live, Relaxed);
}
// This binary has one test. Account for successful System allocations and
// frees, including reallocations, without allocating inside the allocator.
unsafe impl GlobalAlloc for Tracking {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            added(layout.size());
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Relaxed);
        unsafe { System.dealloc(ptr, layout) };
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let new = unsafe { System.realloc(ptr, layout, size) };
        if !new.is_null() {
            if size >= layout.size() {
                added(size - layout.size());
            } else {
                LIVE.fetch_sub(layout.size() - size, Relaxed);
            }
        }
        new
    }
}
#[global_allocator]
static ALLOCATOR: Tracking = Tracking;

#[test]
fn discarded_candidates_do_not_accumulate_in_saved_cells() {
    let names: Vec<_> = (0..16).map(|i| format!("kind{i}")).collect();
    let refs: Vec<_> = names.iter().map(String::as_str).collect();
    let mut labels = vec![EntityLabel::Outside];
    for kind in 0..16 {
        labels.extend([EntityLabel::Begin(Tag(kind)), EntityLabel::Inside(Tag(kind))]);
    }
    let n = labels.len();
    let decoder = EntityDecoder::new(EntityEncoding::Bio, TagSet::new("entities", &refs), labels).unwrap();
    let emissions = vec![vec![0.0; n]; 24];
    let transitions = vec![0.0; n * n];
    let endpoints = vec![0.0; n];
    let lattice = decoder
        .lattice(CrfScores { emissions: &emissions, transitions: &transitions, start: &endpoints, end: &endpoints })
        .unwrap();
    let baseline = LIVE.load(Relaxed);
    PEAK.store(baseline, Relaxed);
    let paths = lattice.numeric().best_paths(8).unwrap();
    let extra = PEAK.load(Relaxed).saturating_sub(baseline);
    println!("peak additional decode allocation: {extra} bytes");
    assert_eq!(paths.len(), 8);
    assert!(paths.iter().all(|p| p.labels.len() == 24));
    // Selected cells require about 152 KiB on 64-bit platforms. Allow space
    // for scratch, marginals and returned paths, but not n*k per saved cell.
    assert!(extra < 512 * 1024, "decoding retained too much memory: {extra} bytes");
}
