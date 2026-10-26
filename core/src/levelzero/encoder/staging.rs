//! The pinned staging a model load copies its weights through (#1355): two
//! buffers of STAGE bytes on the context, filled in turn, a buffer written
//! again only after the queue has synchronized the copies out of it. The
//! driver then never reads the caller's pages (a mapped bundle) for the
//! length of a load. Copied from those pages, two large loads at once
//! failed: the issue saw TURBO_E_OUT_OF_MEMORY from the copy, and its
//! reproduction here a lost device (0x70000001) at the synchronization.

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
        Fill { bufs: self.bufs, cursor: Cursor::new(STAGE) }
    }
}

impl Drop for Staging<'_> {
    fn drop(&mut self) {
        for b in self.bufs {
            self.c.free(b as *mut c_void, "the weights' staging");
        }
    }
}

/// Where the next bytes go: buffer `k` from `at`, buffers of `size`
/// bytes. A buffer is written again only after the queue has synchronized
/// the copies out of it. No device here: [`Fill`] acts on what it says.
#[derive(Debug, Default)]
struct Cursor {
    size: usize,
    at: usize,
    used: [usize; 2],
    k: usize,
}

/// One piece of a copy: `n` bytes into buffer `k` at `at`, after waiting
/// for the queue when `sync`.
#[derive(Debug, PartialEq)]
struct Piece {
    sync: bool,
    k: usize,
    at: usize,
    n: usize,
}

impl Cursor {
    fn new(size: usize) -> Cursor {
        Cursor { size, ..Cursor::default() }
    }

    /// The next piece of a copy with `left` bytes still to go.
    fn next(&mut self, left: usize) -> Piece {
        let mut sync = false;
        if self.at == self.size {
            // The other buffer: wait for its copies before writing it.
            self.k ^= 1;
            self.at = 0;
            if self.used[self.k] > 0 {
                sync = true;
                self.used = [0; 2];
            }
        }
        let (k, at, n) = (self.k, self.at, left.min(self.size - self.at));
        // The next piece starts aligned, as a copy of a whole tensor did.
        self.at = (self.at + n).next_multiple_of(DEVICE_ALIGN).min(self.size);
        self.used[self.k] += n;
        Piece { sync, k, at, n }
    }
}

/// The copies of a load through the staging.
pub(super) struct Fill {
    bufs: [*mut u8; 2],
    cursor: Cursor,
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
            let p = self.cursor.next(src.len() - done);
            if p.sync {
                c.sync(q)?;
            }
            let stage = unsafe { self.bufs[p.k].add(p.at) };
            unsafe {
                std::ptr::copy_nonoverlapping(src[done..].as_ptr(), stage, p.n);
                c.copy(q, dst.add(done) as *mut c_void, stage as *const c_void, p.n)?;
            }
            done += p.n;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every piece of copies of `sizes` through buffers of `size`.
    fn pieces(size: usize, sizes: &[usize]) -> Vec<Piece> {
        let mut c = Cursor::new(size);
        let mut out = Vec::new();
        for &len in sizes {
            let mut done = 0;
            while done < len {
                let p = c.next(len - done);
                done += p.n;
                out.push(p);
            }
        }
        out
    }

    #[test]
    fn small_copies_share_a_buffer_at_aligned_offsets() {
        let a = DEVICE_ALIGN;
        let got = pieces(4 * a, &[10, a + 1, 3]);
        let want = [(0, 0, 10), (0, a, a + 1), (0, 3 * a, 3)];
        assert_eq!(got, want.map(|(k, at, n)| Piece { sync: false, k, at, n }));
    }

    #[test]
    fn a_large_copy_is_cut_into_buffers_in_turn_waiting_before_each_reuse() {
        let size = 4 * DEVICE_ALIGN;
        let got = pieces(size, &[5 * size + 7]);
        // Buffer 0, then buffer 1 (unused, no wait), then back to 0, which
        // waits for the queue; that wait covers buffer 1's copies too, so
        // the switch to 1 after it does not wait again.
        let want =
            [(false, 0, size), (false, 1, size), (true, 0, size), (false, 1, size), (true, 0, size), (false, 1, 7)];
        let want: Vec<Piece> = want.into_iter().map(|(sync, k, n)| Piece { sync, k, at: 0, n }).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn no_piece_runs_past_its_buffer_nor_is_written_before_a_wait() {
        let size = 3 * DEVICE_ALIGN + 100;
        let sizes = [1, 700, 2 * DEVICE_ALIGN, 5000, 3, 999, size, 17];
        let got = pieces(size, &sizes);
        assert_eq!(got.iter().map(|p| p.n).sum::<usize>(), sizes.iter().sum::<usize>());
        // Since each buffer's last wait, the pieces written to it lie one
        // after another and inside it.
        let mut end = [0usize; 2];
        for p in &got {
            assert!(p.n > 0 && p.at + p.n <= size, "{p:?}");
            if p.sync {
                end = [0; 2];
            }
            assert!(p.at >= end[p.k], "{p:?} overwrites bytes not yet copied out");
            end[p.k] = p.at + p.n;
        }
    }
}
