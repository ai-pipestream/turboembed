//! The rule splitter's per-thread buffers.

use std::cell::RefCell;

use super::RuleScratch;

/// The most elements a per-thread buffer keeps between calls. A split
/// that grows one past this hands it back to the allocator afterwards, so
/// a thread that once handled a very large document does not hold that
/// document's peak memory for the rest of its life.
pub(crate) const RETAINED: usize = 1 << 16;

thread_local! {
    /// The buffers of the rule annotator, one set per thread, so that a
    /// pipeline running many documents allocates only while they grow.
    static RULE_SCRATCH: RefCell<RuleScratch> = RefCell::new(RuleScratch::default());
}

/// Runs `f` with this thread's buffers, or with fresh ones when they are
/// already in use further up the stack, then releases any buffer that grew
/// past [`RETAINED`].
pub(crate) fn with_rule_scratch<R>(f: impl FnOnce(&mut RuleScratch) -> R) -> R {
    RULE_SCRATCH.with(|cell| match cell.try_borrow_mut() {
        Ok(mut scratch) => {
            let r = f(&mut scratch);
            scratch.release_large();
            r
        }
        Err(_) => f(&mut RuleScratch::default()),
    })
}

/// The largest capacity among this thread's buffers.
#[cfg(test)]
pub(crate) fn retained_capacity() -> usize {
    RULE_SCRATCH.with_borrow(|s| s.largest_capacity())
}

fn release<T>(v: &mut Vec<T>) {
    if v.capacity() > RETAINED {
        *v = Vec::new();
    }
}

impl RuleScratch {
    fn release_large(&mut self) {
        release(&mut self.units);
        release(&mut self.breaks);
        release(&mut self.depth);
        release(&mut self.stack);
        release(&mut self.found);
        release(&mut self.index);
        release(&mut self.markers);
        if self.lower.capacity() > RETAINED {
            self.lower = String::new();
        }
    }

    #[cfg(test)]
    fn largest_capacity(&self) -> usize {
        [
            self.units.capacity(),
            self.breaks.capacity(),
            self.depth.capacity(),
            self.stack.capacity(),
            self.found.capacity(),
            self.index.capacity(),
            self.markers.capacity(),
            self.lower.capacity(),
        ]
        .into_iter()
        .max()
        .unwrap_or(0)
    }
}
