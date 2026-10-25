//! Writes of rows and ranges into the staging, and the run.

use std::ffi::c_void;

use super::model::*;
use super::session::*;
use super::*;
use crate::backend::{turbo_backend_embed_rows, turbo_backend_run};
use crate::levelzero::gpu::{Buffer, Context, Res, device_allocs_here, fail, guarded};
use crate::levelzero::ze;
use crate::status::{CAPACITY, INVALID_ARGUMENT, INVALID_SHAPE, INVALID_STATE};
use crate::{
    TURBO_EMBED_STAGE_DOWNLOAD, TURBO_EMBED_STAGE_ENCODE, TURBO_EMBED_STAGE_LOOKUP, TURBO_EMBED_STAGE_NORMALIZE,
    TURBO_EMBED_STAGE_POOL, TURBO_EMBED_STAGE_UPLOAD, TURBO_NORMALIZE_L2, TURBO_PLACE_DEVICE, TURBO_POOLING_MEAN,
    TURBO_STAGE_DEVICE, TURBO_STAGE_FUSED, TURBO_STAGE_UNUSED, turbo_error, turbo_token_range,
};

impl Session {
    #[inline]
    pub(super) fn ctx(&self) -> &Context {
        unsafe { &*self.ctx }
    }

    /// The encoder through `layers` layers over the written rows, waited
    /// for. Returns the oneDNN primitives the run built.
    pub(super) fn encode_and_wait(&self, layers: u32) -> Res<u64> {
        let c = self.ctx();
        let mut q = c.lock_queue()?;
        let encoded = self.encode(&mut q, layers);
        // The queue is left idle whether or not the run finished.
        let synced = c.sync(&mut q);
        // So is oneDNN's: a run that finished has its last event waited for
        // by the backend's last kernel, and one that failed part way may
        // have oneDNN work queued that nothing waits for, still writing to
        // the scratch. Its events are let go.
        #[cfg(feature = "levelzero-onednn")]
        let waited = match c.dnnl() {
            Some(dn) if self.linear.uses(Linear::Onednn) => dn.wait(),
            _ => Ok(()),
        };
        #[cfg(not(feature = "levelzero-onednn"))]
        let waited: Res<()> = Ok(());
        let built = encoded?;
        synced?;
        waited?;
        Ok(built)
    }

    #[inline]
    pub(super) fn model(&self) -> &Model {
        unsafe { &*self.model }
    }

    /// The written rows, packed on the host into the staging the device
    /// reads: each row's positions through its last live token, one after
    /// another, as ids, positions, types (when written) and mask; and the
    /// row table, each row's first packed token and its length. Returns
    /// the live tokens, the longest row, and the bytes the device will
    /// read. Rows in device memory are refused: the core reads rows on the
    /// host.
    ///
    /// # Safety
    /// The arrays hold (batch - 1) * row_stride + seq values each.
    pub(super) unsafe fn pack(&self, r: &turbo_backend_embed_rows) -> Res<(u32, u32, u64)> {
        let c = self.ctx();
        for p in [r.ids, r.mask, r.types] {
            if !p.is_null() && c.memory_type(p as *const c_void).0 == ze::MEMORY_TYPE_DEVICE {
                return Err(fail(INVALID_ARGUMENT, "rows are device memory; the core reads rows on the host"));
            }
        }
        let n = self.max_batch as usize * self.max_seq as usize;
        let staging = self.staging as *mut i32;
        let (ids, positions, types, mask) =
            unsafe { (staging, staging.add(n), staging.add(2 * n), staging.add(3 * n)) };
        let table = unsafe { staging.add(4 * n) };
        let pos0 = self.model().desc.position_offset as usize;
        let (mut t, mut longest) = (0usize, 0u32);
        for b in 0..r.batch as usize {
            let at = b * r.row_stride as usize;
            let m = unsafe { std::slice::from_raw_parts(r.mask.add(at), r.seq as usize) };
            let len = m.iter().rposition(|&v| v != 0).map_or(0, |p| p + 1);
            unsafe {
                *table.add(2 * b) = t as i32;
                *table.add(2 * b + 1) = len as i32;
                std::ptr::copy_nonoverlapping(r.ids.add(at), ids.add(t), len);
                std::ptr::copy_nonoverlapping(r.mask.add(at), mask.add(t), len);
                if !r.types.is_null() {
                    std::ptr::copy_nonoverlapping(r.types.add(at), types.add(t), len);
                }
                for p in 0..len {
                    *positions.add(t + p) = (p + pos0) as i32;
                }
            }
            t += len;
            longest = longest.max(len as u32);
        }
        let arrays = if r.types.is_null() { 3 } else { 4 };
        Ok((t as u32, longest, (arrays * t * 4 + r.batch as usize * 8) as u64))
    }

    /// Where the ranges sit in the staging: after the row table, which
    /// holds max_batch rows.
    pub(super) fn ranges_at(&self) -> *mut turbo_token_range {
        let n = self.max_batch as usize * self.max_seq as usize;
        unsafe { (self.staging as *mut u8).add(4 * n * 4 + self.max_batch as usize * 8) as *mut turbo_token_range }
    }

    /// A write: the rows packed into the staging, and `ranges` copied
    /// after them, none for embed_write, whose run pools each row. A write
    /// that fails leaves nothing to run.
    ///
    /// # Safety
    /// As `pack`; `ranges` holds at most max_ranges ranges.
    pub(super) unsafe fn write(&mut self, r: &turbo_backend_embed_rows, ranges: &[turbo_token_range]) -> Res<()> {
        self.written = false;
        // The core checks the shape against the session; pack writes the
        // staging through raw pointers, so a caller of the table that does
        // not is refused here, before a byte is written.
        if r.batch > self.max_batch || r.seq > self.max_seq {
            return Err(fail(
                CAPACITY,
                format!(
                    "{} rows of {} tokens: the levelzero session holds {} rows of {}",
                    r.batch, r.seq, self.max_batch, self.max_seq
                ),
            ));
        }
        if r.batch > 1 && r.row_stride < r.seq {
            return Err(fail(INVALID_SHAPE, format!("row_stride {} is under seq {}", r.row_stride, r.seq)));
        }
        // No crossing yet: the lookup kernel reads the packed rows, and the
        // pooling kernel the ranges, from the staging when the run starts.
        let (tokens, longest, sent) = unsafe { self.pack(r)? };
        debug_assert!(ranges.len() <= self.max_ranges as usize);
        unsafe { std::ptr::copy_nonoverlapping(ranges.as_ptr(), self.ranges_at(), ranges.len()) };
        self.has_types = !r.types.is_null();
        self.batch = r.batch;
        self.ranges = ranges.len() as u32;
        self.tokens = tokens;
        self.longest = longest;
        // The core writes a range run's pooling as TURBO_POOLING_MEAN.
        self.pooling = if ranges.is_empty() { r.pooling } else { TURBO_POOLING_MEAN };
        self.normalize = r.normalize;
        self.output_dim = r.output_dim;
        self.h2d = sent + (ranges.len() * RANGE_BYTES) as u64;
        self.written = true;
        Ok(())
    }
}

pub(crate) unsafe extern "C" fn embed_write(
    session: *mut c_void,
    rows: *const turbo_backend_embed_rows,
    err: *mut turbo_error,
) -> i32 {
    unsafe { guarded(err, || (*(session as *mut Session)).write(&*rows, &[])) }
}

/// The rows as embed_write takes them, and the ranges the run pools them
/// by in place of each row. The core has checked each range and that
/// there are 1 to max_ranges of them.
pub(crate) unsafe extern "C" fn embed_write_ranges(
    session: *mut c_void,
    rows: *const turbo_backend_embed_rows,
    ranges: *const turbo_token_range,
    count: u32,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        guarded(err, || {
            let s = &mut *(session as *mut Session);
            if count == 0 || count > s.max_ranges {
                return Err(fail(
                    INVALID_ARGUMENT,
                    format!("{count} ranges: the levelzero session takes 1 to its max_ranges, {}", s.max_ranges),
                ));
            }
            s.write(&*rows, std::slice::from_raw_parts(ranges, count as usize))
        })
    }
}

pub(crate) unsafe extern "C" fn session_run(
    session: *mut c_void,
    out: *mut turbo_backend_run,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        guarded(err, || {
            let (s, out) = (&mut *(session as *mut Session), &mut *out);
            if !std::mem::take(&mut s.written) {
                return Err(fail(INVALID_STATE, "the levelzero session has no rows written since its last run"));
            }
            let device0 = device_allocs_here();
            let built = s.encode_and_wait(s.model().desc.layers)?;
            out.placement = TURBO_PLACE_DEVICE;
            out.output = &*s.output as *const Buffer as *mut c_void;
            out.host = std::ptr::null_mut();
            out.h2d_bytes = s.h2d;
            out.d2h_bytes = 0;
            // The run appends launches to memory the session holds: the
            // kernels' arguments are set in place and nothing is allocated,
            // but for oneDNN's primitives the first time at a bin of tokens.
            out.host_allocs = built;
            out.device_allocs = device_allocs_here() - device0;
            let st = &mut out.stage;
            // The lookup kernel reads the packed rows over the bus.
            st[TURBO_EMBED_STAGE_UPLOAD] = TURBO_STAGE_FUSED;
            st[TURBO_EMBED_STAGE_LOOKUP] = TURBO_STAGE_DEVICE;
            st[TURBO_EMBED_STAGE_ENCODE] = TURBO_STAGE_DEVICE;
            st[TURBO_EMBED_STAGE_POOL] = TURBO_STAGE_DEVICE;
            st[TURBO_EMBED_STAGE_NORMALIZE] =
                if s.normalize == TURBO_NORMALIZE_L2 { TURBO_STAGE_DEVICE } else { TURBO_STAGE_UNUSED };
            st[TURBO_EMBED_STAGE_DOWNLOAD] = TURBO_STAGE_UNUSED;
            Ok(())
        })
    }
}
