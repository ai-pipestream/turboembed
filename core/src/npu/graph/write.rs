//! Writes of rows and ranges into a session.

use std::ffi::c_void;

use super::host::*;
use super::session::*;
use super::*;
use crate::backend::turbo_backend_embed_rows;
use crate::status::{CAPACITY, INVALID_ARGUMENT, INVALID_SHAPE, UNSUPPORTED_OPTION};
use crate::{TURBO_POOLING_MEAN, turbo_error, turbo_token_range};

/// # Safety
/// As turbo_backend.h says for embed_write.
pub(crate) unsafe extern "C" fn embed_write(
    session: *mut c_void,
    rows: *const turbo_backend_embed_rows,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        guarded(err, || {
            let s = &*(session as *const Session);
            let mut kept = s.rows.lock().unwrap_or_else(|p| p.into_inner());
            keep_rows(s, &*rows, &mut kept)
        })
    }
}

/// # Safety
/// As turbo_backend.h says for embed_write_ranges.
pub(crate) unsafe extern "C" fn embed_write_ranges(
    session: *mut c_void,
    rows: *const turbo_backend_embed_rows,
    ranges: *const turbo_token_range,
    count: u32,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        guarded(err, || {
            let s = &*(session as *const Session);
            let r = &*rows;
            let mut kept = s.rows.lock().unwrap_or_else(|p| p.into_inner());
            keep_rows(s, r, &mut kept)?;
            // The rows are kept; the write stands only once the ranges are.
            kept.state.written = false;
            if count == 0 {
                return Err(fail(INVALID_ARGUMENT, "npu: a write of ranges holds at least one"));
            }
            let ranges = std::slice::from_raw_parts(ranges, count as usize);
            keep_ranges(s.max_ranges, s.max_seq as usize, r, ranges, &mut kept)?;
            kept.state.vectors = Vectors::Ranges;
            kept.state.written = true;
            Ok(())
        })
    }
}

/// The rows of a write into the session's own, at stride max_seq, and the
/// options with them: the state of a write of rows. A refused row leaves
/// nothing written.
///
/// # Safety
/// `r` and the arrays it points to are as turbo_backend.h says for
/// embed_write.
pub(super) unsafe fn keep_rows(s: &Session, r: &turbo_backend_embed_rows, rows: &mut Rows) -> Res<()> {
    let m = s.model();
    let (batch, seq, stride) = (r.batch as usize, r.seq as usize, r.row_stride as usize);
    let dst_stride = s.max_seq as usize;
    rows.state.written = false;
    // The core checks the shape against the session; a caller of the
    // table that does not is refused here, before a row is copied.
    let max_batch = rows.ids.len() / dst_stride.max(1);
    if batch > max_batch || seq > dst_stride {
        return Err(fail(
            CAPACITY,
            format!("npu: {batch} rows of {seq} tokens; the session holds {max_batch} rows of {dst_stride}"),
        ));
    }
    if batch > 1 && stride < seq {
        return Err(fail(INVALID_SHAPE, format!("npu: row_stride {stride} is under seq {seq}")));
    }
    for row in 0..batch {
        let src = row * stride;
        let dst = row * dst_stride;
        let ids = unsafe { std::slice::from_raw_parts(r.ids.add(src), seq) };
        let mask = unsafe { std::slice::from_raw_parts(r.mask.add(src), seq) };
        // The core promises at least one live token per row; a row
        // without one would pool a mean over nothing, so it is refused
        // here rather than returned as NaN.
        if !mask.contains(&1) {
            return Err(fail(INVALID_ARGUMENT, format!("npu: row {row} has no token with mask 1")));
        }
        rows.ids[dst..dst + seq].copy_from_slice(ids);
        rows.mask[dst..dst + seq].copy_from_slice(mask);
        match r.types.is_null() {
            true => rows.types[dst..dst + seq].fill(0),
            false => {
                let types = unsafe { std::slice::from_raw_parts(r.types.add(src), seq) };
                if let Some(why) = type_refusal(&m.inputs)
                    && let Some(at) = types.iter().position(|&t| t != 0)
                {
                    return Err(type_error(row, at, types[at], why));
                }
                rows.types[dst..dst + seq].copy_from_slice(types);
            }
        }
    }
    rows.state = State {
        batch: r.batch,
        seq: r.seq,
        pooling: r.pooling,
        normalize: r.normalize,
        output_dim: r.output_dim,
        vectors: Vectors::Rows,
        written: true,
    };
    Ok(())
}

/// The ranges of a write, after the rows they index were kept. The core
/// checks each range as turbo.h says; they are checked again here
/// because pooling indexes the session's rows by them, and a range
/// without a live token would be a mean over nothing. Copied into
/// memory made with the session, so a write allocates nothing.
pub(super) fn keep_ranges(
    max_ranges: u32,
    stride: usize,
    r: &turbo_backend_embed_rows,
    ranges: &[turbo_token_range],
    rows: &mut Rows,
) -> Res<()> {
    rows.ranges.clear();
    if ranges.len() > max_ranges as usize {
        return Err(fail(
            CAPACITY,
            format!("npu: {} ranges are over the session's max_ranges {max_ranges}", ranges.len()),
        ));
    }
    if r.pooling != TURBO_POOLING_MEAN {
        return Err(fail_field(
            UNSUPPORTED_OPTION,
            5,
            format!("npu: pooling {}: a range is pooled by the mean alone", r.pooling),
        ));
    }
    for (i, g) in ranges.iter().enumerate() {
        if g.row >= r.batch || g.start >= g.end || g.end > r.seq {
            return Err(fail(
                INVALID_ARGUMENT,
                format!(
                    "npu: range {i} is row {} columns [{}, {}), outside {} rows of {} tokens",
                    g.row, g.start, g.end, r.batch, r.seq
                ),
            ));
        }
        let at = g.row as usize * stride;
        if !rows.mask[at + g.start as usize..at + g.end as usize].contains(&1) {
            return Err(fail(INVALID_ARGUMENT, format!("npu: range {i} has no token with mask 1")));
        }
    }
    rows.ranges.extend_from_slice(ranges);
    Ok(())
}
