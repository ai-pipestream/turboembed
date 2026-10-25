//! A warmed `turbo::cpu::lstm::Work` makes a run of the same or a smaller
//! shape allocate nothing, as its doc says: counted with a global
//! allocator that counts this thread's allocations.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use turbo::cpu::lstm::{Lstm, Weights, Work};

struct Counting;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.with(|n| n.set(n.get() + 1));
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

fn allocations(f: impl FnOnce()) -> usize {
    let before = ALLOCATIONS.with(Cell::get);
    f();
    ALLOCATIONS.with(Cell::get) - before
}

#[test]
fn a_warmed_work_runs_the_same_or_a_smaller_shape_without_allocating() {
    let (input, hidden) = (3, 2);
    let w: Vec<f32> = (0..4 * hidden * input).map(|i| (i as f32 * 0.37).sin() * 0.5).collect();
    let u: Vec<f32> = (0..4 * hidden * hidden).map(|i| (i as f32 * 0.53).cos() * 0.5).collect();
    let b: Vec<f32> = (0..4 * hidden).map(|i| i as f32 * 0.01).collect();
    let dirs = [false, true].map(|reverse| Weights { w: &w, u: &u, b: &b, reverse });
    let lstm = Lstm::new(input, hidden, &dirs).unwrap();
    let x: Vec<f32> = (0..6 * input).map(|i| (i as f32 * 0.11).sin()).collect();
    let mut out = vec![0.0; 6 * lstm.width()];
    let mut work = Work::default();
    lstm.run(&x, &[4, 2], &mut out, &mut work).unwrap();
    let same = allocations(|| lstm.run(&x, &[4, 2], &mut out, &mut work).unwrap());
    assert_eq!(same, 0, "the same shape again");
    let (x, out) = (&x[..3 * input], &mut out[..3 * lstm.width()]);
    let smaller = allocations(|| lstm.run(x, &[2, 1], out, &mut work).unwrap());
    assert_eq!(smaller, 0, "a smaller shape");
}
