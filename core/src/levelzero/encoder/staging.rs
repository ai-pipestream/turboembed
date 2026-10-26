//! The pinned staging a model load copies its weights through (#1355): two
//! buffers of STAGE bytes on the context, filled in turn, a buffer written
//! again only after the queue has synchronized the copies out of it. The
//! driver then never pins the caller's pages (a mapped bundle) for the
//! length of a load, which ran a second load at once out of memory.

use std::ffi::c_void;

use super::*;
use crate::levelzero::gpu::{Context, Queue, Res};

/// Bytes of each of the two pinned buffers a load stages its weights in.
const STAGE: usize = 32 << 20;

/// Two pinned buffers on a context, freed with this.
pub(super) struct Staging<'c> {
    c: &'c Context,
    bufs: [*mut u8; 2],
}

impl<'c> Staging<'c> {
    pub(super) fn new(c: &'c Context) -> Res<Staging<'c>> {
        let a = c.alloc_pinned(STAGE)?;
        let b = c.alloc_pinned(STAGE).inspect_err(|_| c.free(a, "the weights' staging"))?;
        Ok(Staging { c, bufs: [a as *mut u8, b as *mut u8] })
    }

    pub(super) fn fill(&self) -> Fill {
        Fill { bufs: self.bufs, at: 0, used: [0; 2], k: 0 }
    }
}

impl Drop for Staging<'_> {
    fn drop(&mut self) {
        for b in self.bufs {
            self.c.free(b as *mut c_void, "the weights' staging");
        }
    }
}

/// Where the next bytes go: buffer `k` from `at`. A buffer is written again
/// only after the queue has synchronized the copies out of it.
pub(super) struct Fill {
    bufs: [*mut u8; 2],
    at: usize,
    used: [usize; 2],
    k: usize,
}

impl Fill {
    /// Appends the copy of `src` to `dst` through the staging, in pieces
    /// of at most STAGE bytes.
    ///
    /// # Safety
    /// `dst` is device memory of `src.len()` bytes on `c`, valid until the
    /// queue synchronizes.
    pub(super) unsafe fn copy(&mut self, c: &Context, q: &mut Queue, dst: *mut u8, src: &[u8]) -> Res<()> {
        let mut done = 0;
        while done < src.len() {
            if self.at == STAGE {
                // The other buffer: wait for its copies before writing it.
                self.k ^= 1;
                self.at = 0;
                if self.used[self.k] > 0 {
                    c.sync(q)?;
                    self.used = [0; 2];
                }
            }
            let n = (src.len() - done).min(STAGE - self.at);
            let stage = unsafe { self.bufs[self.k].add(self.at) };
            unsafe {
                std::ptr::copy_nonoverlapping(src[done..].as_ptr(), stage, n);
                c.copy(q, dst.add(done) as *mut c_void, stage as *const c_void, n)?;
            }
            // The next piece starts aligned, as a copy of a whole tensor did.
            self.at = (self.at + n).next_multiple_of(DEVICE_ALIGN).min(STAGE);
            self.used[self.k] += n;
            done += n;
        }
        Ok(())
    }
}
