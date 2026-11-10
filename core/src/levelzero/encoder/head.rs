//! The token head on the device: an extractive QA model's span head, a
//! linear layer of head_out outputs over each range's pooled vector, its
//! sums in F64 (turbo_backend.h, token_head). The core hands the head only
//! to a table that sets token_head, and asks for it on a range write
//! (rows.head).

use std::ffi::c_void;

use super::session::Session;
use crate::backend::turbo_backend_model;
use crate::levelzero::gpu::{Arg, Context, Queue, Res, fail};
use crate::status::INTERNAL;

/// Output values a group of span_head computes, a sub-group each.
pub(super) const HEAD_SUBGROUPS: u32 = 8;

/// The model's head on the device: the [head_out, hidden] F32 weight and
/// then its [head_out] bias, in one allocation; null for a model the core
/// gave no head.
pub(super) fn upload(c: &Context, d: &turbo_backend_model) -> Res<*mut c_void> {
    let covers = d.struct_size as usize >= std::mem::offset_of!(turbo_backend_model, head_out) + size_of::<u32>();
    if !covers || d.head.is_null() {
        return Ok(std::ptr::null_mut());
    }
    // SAFETY: a non-null head is two tensors the core holds while the
    // model is loaded.
    let [w, b] = unsafe { [&*d.head, &*d.head.add(1)] };
    let (out, hidden) = (d.head_out as u64, d.hidden as u64);
    if out == 0 || w.bytes != out * hidden * 4 || b.bytes != out * 4 {
        return Err(fail(
            INTERNAL,
            format!("a token head of {out} outputs whose tensors are not F32 [{out}, {hidden}] and [{out}]"),
        ));
    }
    let (wb, bb) = (w.bytes as usize, b.bytes as usize);
    let head = c.alloc_device(wb + bb)?;
    let copied = (|| {
        let mut q = c.lock_queue()?;
        // SAFETY: both tensors hold their bytes, and head holds both.
        let appended = unsafe {
            c.copy(&mut q, head, w.data, wb).and_then(|_| c.copy(&mut q, head.cast::<u8>().add(wb).cast(), b.data, bb))
        };
        let synced = c.sync(&mut q);
        appended?;
        synced
    })();
    if let Err(e) = copied {
        c.free(head, "the token head");
        return Err(e);
    }
    Ok(head)
}

/// Frees what upload made, if anything.
pub(super) fn release(c: &Context, head: *mut c_void) {
    if !head.is_null() {
        c.free(head, "the token head");
    }
}

impl Session {
    /// The head over the `ranges` vectors pooled, hidden wide, at
    /// `pooled`, into the output: [ranges, head_out] F32.
    pub(super) fn apply_head(&self, q: &mut Queue, pooled: u64) -> Res<()> {
        use Arg::*;
        let (m, k) = (self.model(), &self.kernels);
        let values = self.ranges * m.desc.head_out;
        let args = [
            Ptr(pooled),
            Ptr(m.head as u64),
            I32(m.desc.head_out as i32),
            I32(m.desc.hidden as i32),
            I32(values as i32),
            Ptr(self.output.ptr as u64),
        ];
        k.span_head.launch(self.ctx(), q, "the token head", &args, [values.div_ceil(HEAD_SUBGROUPS), 1, 1])
    }
}
