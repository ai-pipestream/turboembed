//! A session's rows written in and run: no allocation, and no crossing
//! between memories to count (session.rs).

use std::ffi::c_void;

use crate::backend::{refuse, turbo_backend_embed_rows, turbo_backend_run};
use crate::status::INVALID_STATE;
use crate::{
    TURBO_EMBED_STAGE_DOWNLOAD, TURBO_EMBED_STAGE_ENCODE, TURBO_EMBED_STAGE_LOOKUP, TURBO_EMBED_STAGE_NORMALIZE,
    TURBO_EMBED_STAGE_POOL, TURBO_EMBED_STAGE_UPLOAD, TURBO_NORMALIZE_L2, TURBO_PLACE_HOST, TURBO_STAGE_FUSED,
    TURBO_STAGE_HOST, TURBO_STAGE_UNUSED, turbo_error, turbo_token_range,
};

use super::buffers::*;
use super::session::*;

pub(super) unsafe extern "C" fn session_release(session: *mut c_void) {
    drop(unsafe { Box::from_raw(session as *mut Session) });
}

/// The rows, copied into the session: see session.rs.
pub(super) unsafe extern "C" fn embed_write(
    session: *mut c_void,
    rows: *const turbo_backend_embed_rows,
    _err: *mut turbo_error,
) -> i32 {
    let (s, r) = unsafe { (&mut *(session as *mut Session), &*rows) };
    let (ids, mask, types) = unsafe { arrays(r) };
    match &mut s.work {
        Work::Encoder(e) => e.write(r, ids, mask, types),
        // A static model has one token type, so the core passes only 0s.
        Work::Static(t) => t.write(r, ids, mask),
        Work::Deberta(d) => d.write(r, ids, mask),
    }
    s.written = true;
    0
}

/// The rows and the ranges the run pools them by, copied into the
/// session. The core has checked each range, and that there are at most
/// the session's max_ranges.
pub(super) unsafe extern "C" fn embed_write_ranges(
    session: *mut c_void,
    rows: *const turbo_backend_embed_rows,
    ranges: *const turbo_token_range,
    count: u32,
    _err: *mut turbo_error,
) -> i32 {
    let (s, r) = unsafe { (&mut *(session as *mut Session), &*rows) };
    debug_assert!(count >= 1 && count as usize <= s.max_ranges);
    let (ids, mask, types) = unsafe { arrays(r) };
    let ranges = unsafe { std::slice::from_raw_parts(ranges, count as usize) };
    match &mut s.work {
        Work::Encoder(e) => e.write_ranges(r, ids, mask, types, ranges),
        Work::Static(t) => t.write_ranges(r, ids, mask, ranges),
        Work::Deberta(d) => d.write_ranges(r, ids, mask, ranges),
    }
    s.written = true;
    0
}

/// The ids, mask and types `r` points to, through its last row's last
/// column.
///
/// # Safety
/// The arrays hold what `r` says, for as long as the slices are used.
pub(super) unsafe fn arrays<'a>(r: &turbo_backend_embed_rows) -> (&'a [i32], &'a [i32], Option<&'a [i32]>) {
    let span = (r.batch as usize - 1) * r.row_stride as usize + r.seq as usize;
    let (ids, mask) = unsafe { (std::slice::from_raw_parts(r.ids, span), std::slice::from_raw_parts(r.mask, span)) };
    let types = (!r.types.is_null()).then(|| unsafe { std::slice::from_raw_parts(r.types, span) });
    (ids, mask, types)
}

pub(super) unsafe extern "C" fn session_run(
    session: *mut c_void,
    out: *mut turbo_backend_run,
    err: *mut turbo_error,
) -> i32 {
    let (s, out) = unsafe { (&mut *(session as *mut Session), &mut *out) };
    if !std::mem::take(&mut s.written) {
        return unsafe { refuse(err, INVALID_STATE, "the cpu session has no rows written since its last run") };
    }
    let floats = unsafe { std::slice::from_raw_parts_mut(s.output.ptr as *mut f32, s.output_len()) };
    let normalize = match &mut s.work {
        Work::Encoder(e) => {
            e.run(&mut s.pool, floats);
            e.normalize()
        }
        Work::Static(t) => {
            t.run(&mut s.pool, floats);
            t.normalize()
        }
        Work::Deberta(d) => {
            if !d.run(&mut s.pool, floats) {
                return unsafe { refuse(err, INVALID_STATE, "the deberta encoder refused a row the core passed") };
            }
            d.normalize()
        }
    };
    let encoded = !matches!(s.work, Work::Static(_));
    out.placement = TURBO_PLACE_HOST;
    out.output = &*s.output as *const Buffer as *mut c_void;
    out.host = s.output.ptr as *mut c_void;
    out.h2d_bytes = 0;
    out.d2h_bytes = 0;
    // Nothing on the BERT and static paths allocates, on this thread or
    // the pool's: the encoder computes in the memory session_create gave
    // it, on the threads it started. tests/allocations.rs holds this to a
    // counting allocator. A DeBERTa-v2 run allocates each row's workspace
    // (deberta.rs), which these counts do not report.
    out.host_allocs = 0;
    out.device_allocs = 0;
    let st = &mut out.stage;
    st[TURBO_EMBED_STAGE_UPLOAD] = TURBO_STAGE_UNUSED;
    st[TURBO_EMBED_STAGE_LOOKUP] = TURBO_STAGE_HOST;
    // A static model's lookup and pool are one sum: the pool is fused
    // into it, and there is no encoder.
    st[TURBO_EMBED_STAGE_ENCODE] = if encoded { TURBO_STAGE_HOST } else { TURBO_STAGE_UNUSED };
    st[TURBO_EMBED_STAGE_POOL] = if encoded { TURBO_STAGE_HOST } else { TURBO_STAGE_FUSED };
    st[TURBO_EMBED_STAGE_NORMALIZE] =
        if normalize == TURBO_NORMALIZE_L2 { TURBO_STAGE_HOST } else { TURBO_STAGE_UNUSED };
    st[TURBO_EMBED_STAGE_DOWNLOAD] = TURBO_STAGE_UNUSED;
    0
}

impl Session {
    /// F32 values the output buffer holds.
    #[inline]
    pub(super) fn output_len(&self) -> usize {
        self.output.layout.map_or(0, |l| l.size() / 4)
    }
}
