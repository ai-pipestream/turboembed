//! Sessions: device scratch, host staging and the output, made once.

use std::ffi::c_void;

use super::kernels::*;
use super::model::*;
use super::tune::*;
use super::*;
use crate::backend::{turbo_backend_session_desc, turbo_backend_tuning};
use crate::levelzero::gpu::{Buffer, Context, Res, fail, fail_field, guarded, quietly};
use crate::status::{INVALID_ARGUMENT, INVALID_STRUCT_SIZE, UNSUPPORTED_OPTION, UNSUPPORTED_TASK};
use crate::{
    TURBO_DTYPE_F16, TURBO_DTYPE_F32, TURBO_PRECISION_FASTEST, TURBO_PRECISION_MODEL, TURBO_TASK_EMBED, turbo_error,
};

pub(super) struct Session {
    pub(super) model: *const Model,
    pub(super) ctx: *const Context,
    pub(super) kernels: Kernels,
    pub(super) weights: Weights,
    /// The linear layers' F16 weights, for a session at FASTEST.
    pub(super) half: Option<Transposed>,
    /// The linear layers' F32 weights transposed, for a session in F32
    /// whose widths linear_sgemm takes.
    pub(super) f32t: Option<Transposed>,
    /// The linear layers' kernels.
    pub(super) linear: Linear,
    pub(super) max_batch: u32,
    pub(super) max_seq: u32,
    /// The most ranges one embed_write_ranges may hold.
    pub(super) max_ranges: u32,
    pub(super) scratch: *mut c_void,
    /// Each row's first packed token and its length through its last live
    /// token, [batch, 2] int32, as the lookup kernel copies it from the
    /// staging.
    pub(super) rows: u64,
    // Packed, [tokens, ...]: the mask, the hidden states, the fused
    // projections, the attention context, a projection's output, and the
    // feed-forward block.
    pub(super) packed_mask: u64,
    /// At FASTEST with the LayerNorm-fused kernels, x holds the
    /// embedding output until the last layer writes its own, for the
    /// pooling; the layers between keep their states in xh alone.
    pub(super) x: u64,
    /// At FASTEST, x in F16, for the linear layers that read it.
    pub(super) xh: u64,
    pub(super) qkv: u64,
    pub(super) att: u64,
    pub(super) tmp: u64,
    pub(super) ffn: u64,
    /// [max(max_batch, max_ranges), hidden] F32 on the device, handed out
    /// as the output: a row's vectors after embed_write, a range's after
    /// embed_write_ranges, each output_dim values, packed.
    pub(super) output: Box<Buffer>,
    /// The packed rows, which the device reads directly: ids, positions,
    /// types and mask, [max_batch * max_seq] int32 each, then the row
    /// table, [max_batch, 2], then the ranges, [max_ranges, 3] (row,
    /// start, end).
    pub(super) staging: *mut c_void,
    /// What the last write left.
    pub(super) written: bool,
    pub(super) has_types: bool,
    pub(super) batch: u32,
    /// The ranges embed_write_ranges wrote, which the run pools by; 0
    /// after embed_write, whose run pools each row.
    pub(super) ranges: u32,
    /// Live tokens, packed.
    pub(super) tokens: u32,
    pub(super) longest: u32,
    pub(super) pooling: u32,
    pub(super) normalize: u32,
    pub(super) output_dim: u32,
    pub(super) h2d: u64,
}

unsafe impl Send for Session {}
unsafe impl Sync for Session {}

impl Drop for Session {
    fn drop(&mut self) {
        let c = unsafe { &*self.ctx };
        if !self.scratch.is_null() {
            c.free(self.scratch, "the session's scratch");
        }
        if !self.staging.is_null() {
            c.free(self.staging, "the session's staging");
        }
    }
}

/// What a session is made for: session_open's description, or
/// session_create's arguments with max_batch ranges.
#[derive(Clone, Copy)]
pub(super) struct Shape {
    pub(super) task: u32,
    pub(super) max_batch: u32,
    pub(super) max_seq: u32,
    pub(super) precision: u32,
    pub(super) max_ranges: u32,
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe extern "C" fn session_create(
    model: *mut c_void,
    task: u32,
    max_batch: u32,
    max_seq: u32,
    precision: u32,
    compute_dtype: *mut u32,
    out: *mut *mut c_void,
    err: *mut turbo_error,
) -> i32 {
    let shape = Shape { task, max_batch, max_seq, precision, max_ranges: max_batch };
    unsafe { create(model, shape, compute_dtype, out, err) }
}

/// As turbo_backend.h says for session_open: the session's whole
/// description, made as session_create_tuned makes one, its output sized
/// for max_ranges vectors when those are more than max_batch. The device
/// is not a CPU, so a thread count is refused.
///
/// # Safety
/// As turbo_backend.h says for session_open.
pub(crate) unsafe extern "C" fn session_open(
    model: *mut c_void,
    desc: *const turbo_backend_session_desc,
    tuning: *mut turbo_backend_tuning,
    compute_dtype: *mut u32,
    out: *mut *mut c_void,
    err: *mut turbo_error,
) -> i32 {
    let d = unsafe { &*desc };
    let need = std::mem::offset_of!(turbo_backend_session_desc, max_ranges) + size_of::<u32>();
    if (d.struct_size as usize) < need {
        let message = format!(
            "turbo_backend_session_desc of {} bytes: the levelzero backend reads it through max_ranges, {need} bytes",
            d.struct_size
        );
        return unsafe { guarded(err, || Err(fail(INVALID_STRUCT_SIZE, message))) };
    }
    if d.cpu_threads != 0 {
        let message = format!(
            "cpu_threads {}: the levelzero backend's device is not a CPU; cpu_threads must be 0",
            d.cpu_threads
        );
        return unsafe { guarded(err, || Err(fail_field(UNSUPPORTED_OPTION, 6, message))) };
    }
    let shape = Shape {
        task: d.task,
        max_batch: d.max_batch,
        max_seq: d.max_seq,
        precision: d.precision,
        max_ranges: d.max_ranges,
    };
    unsafe { create_tuned(model, shape, tuning, compute_dtype, out, err) }
}

/// A session of `shape` on `model`, for session_create, and for
/// session_create_tuned and session_open before they tune it.
pub(super) unsafe fn create(
    model: *mut c_void,
    shape: Shape,
    compute_dtype: *mut u32,
    out: *mut *mut c_void,
    err: *mut turbo_error,
) -> i32 {
    let Shape { task, max_batch, max_seq, precision, max_ranges } = shape;
    unsafe {
        guarded(err, || {
            let m = &*(model as *const Model);
            let c = m.ctx();
            let d = &m.desc;
            if task != TURBO_TASK_EMBED {
                return Err(fail(UNSUPPORTED_TASK, format!("task {task}: the levelzero backend runs embed")));
            }
            if precision == TURBO_PRECISION_MODEL && d.dtype != TURBO_DTYPE_F32 {
                return Err(fail_field(
                    UNSUPPORTED_OPTION,
                    3,
                    format!(
                        "precision: MODEL computes in the weights' {}, which the levelzero backend does not; EXACT \
                         computes this model in F32, and FASTEST in F16 where its widths are multiples of {}, else \
                         in F32",
                        dtype_name(d.dtype),
                        DPAS_K.max(DPAS_TN)
                    ),
                ));
            }
            if max_seq as u64 + d.position_offset as u64 > d.max_positions as u64 {
                return Err(fail_field(
                    UNSUPPORTED_OPTION,
                    2,
                    format!(
                        "max_seq {max_seq} from position {} is over the model's {} positions",
                        d.position_offset, d.max_positions
                    ),
                ));
            }
            let head_dim = d.hidden / d.heads;
            // FASTEST runs the linear layers on the matrix engines, whose
            // 2D reads take whole steps of DPAS_K terms and tiles of
            // DPAS_TN outputs, from F16 activations: the attention kernels
            // write an F16 context.
            let xmx = precision == TURBO_PRECISION_FASTEST
                && d.hidden.is_multiple_of(DPAS_K.max(DPAS_TN))
                && d.intermediate.is_multiple_of(DPAS_K.max(DPAS_TN));
            let kernels = Kernels::new(c, d.hidden, d.intermediate, head_dim, xmx)?;
            if let Attention::General(k) = &kernels.attention {
                // The driver keeps some of a work-group's local memory for
                // the kernel's own use (its reductions); the scores get the
                // rest.
                let local = attention_local_bytes(max_seq, head_dim);
                let room = (c.max_local as u64).saturating_sub(k.local_bytes()? as u64);
                if local > room {
                    return Err(fail_field(
                        UNSUPPORTED_OPTION,
                        2,
                        format!(
                            "max_seq {max_seq}: attention needs {local} bytes of local memory per work-group, and \
                             device {} gives it {room}",
                            c.ordinal
                        ),
                    ));
                }
            }
            let tokens = max_batch as usize * max_seq as usize;
            if tokens > i32::MAX as usize / (3 * d.hidden).max(d.intermediate) as usize {
                return Err(fail_field(
                    UNSUPPORTED_OPTION,
                    1,
                    format!("{max_batch} rows of {max_seq} tokens is more than one kernel indexes"),
                ));
            }
            let weights = m.f32_weights()?;
            let half = if xmx { Some(m.f16_weights(&weights)?) } else { None };
            let sgemm = !xmx && d.hidden.is_multiple_of(SGEMM_TN) && d.intermediate.is_multiple_of(SGEMM_TN);
            let f32t = if sgemm { Some(m.f32_transposed(&weights)?) } else { None };
            let ints = round_up(tokens * 4, DEVICE_ALIGN);
            let table = round_up(max_batch as usize * 8, DEVICE_ALIGN);
            let wide = round_up(tokens * d.hidden as usize * 4, DEVICE_ALIGN);
            let narrow = if xmx { round_up(tokens * d.hidden as usize * 2, DEVICE_ALIGN) } else { 0 };
            let ffn = round_up(tokens * d.intermediate as usize * 4, DEVICE_ALIGN);
            // The row path and the range path write their vectors to this
            // one buffer, so it holds the larger of the two counts.
            let vectors = max_batch.max(max_ranges) as usize;
            let output = round_up(vectors * d.hidden as usize * 4, DEVICE_ALIGN);
            // FASTEST's sums are one part, where they are not taken by the
            // LayerNorm-fused kernels.
            let parts = if xmx { 1 } else { SPLITS as usize };
            let scratch = c.alloc_device(ints + table + (5 + parts) * wide + narrow + ffn + output)?;
            let mut s = Box::new(Session {
                model: m,
                ctx: c,
                kernels,
                weights,
                half,
                f32t,
                linear: Linear::Own,
                max_batch,
                max_seq,
                max_ranges,
                scratch,
                rows: 0,
                packed_mask: 0,
                x: 0,
                xh: 0,
                qkv: 0,
                att: 0,
                tmp: 0,
                ffn: 0,
                output: Box::new(Buffer::session_output(c, std::ptr::null_mut(), 0)),
                staging: std::ptr::null_mut(),
                written: false,
                has_types: false,
                batch: 0,
                ranges: 0,
                tokens: 0,
                longest: 0,
                pooling: 0,
                normalize: 0,
                output_dim: 0,
                h2d: 0,
            });
            s.staging = c.alloc_pinned(4 * tokens * 4 + max_batch as usize * 8 + max_ranges as usize * RANGE_BYTES)?;
            let mut p = scratch as u64;
            let mut take = |n: usize| {
                let at = p;
                p += n as u64;
                at
            };
            s.packed_mask = take(ints);
            s.rows = take(table);
            s.x = take(wide);
            // Null in F32, where the LayerNorms write no F16 copy.
            s.xh = if xmx { take(narrow) } else { 0 };
            s.qkv = take(3 * wide);
            s.att = take(wide);
            s.tmp = take(parts * wide);
            s.ffn = take(ffn);
            let out_ptr = take(output) as usize as *mut c_void;
            s.output = Box::new(Buffer::session_output(c, out_ptr, vectors as u64 * d.hidden as u64 * 4));
            s.linear = match forced_linear()? {
                Some(l) if l == Linear::Onednn && !onednn_runs(c, s.half.as_ref()) => {
                    return Err(fail(
                        UNSUPPORTED_OPTION,
                        "TURBO_LEVELZERO_CHOICES: linear=onednn, and oneDNN does not run on this device in this build",
                    ));
                }
                Some(l) => l,
                None => default_linear(c, s.half.as_ref()),
            };
            *compute_dtype = if xmx { TURBO_DTYPE_F16 } else { TURBO_DTYPE_F32 };
            *out = Box::into_raw(s) as *mut c_void;
            Ok(())
        })
    }
}

/// Whether oneDNN can run a session's linear layers: it opened on the
/// context and, for one at FASTEST, packed the F16 weights.
pub(super) fn onednn_runs(c: &Context, half: Option<&Transposed>) -> bool {
    #[cfg(feature = "levelzero-onednn")]
    return c.dnnl().is_some() && half.is_none_or(|h| !h.packed.is_empty());
    #[cfg(not(feature = "levelzero-onednn"))]
    {
        let _ = (c, half);
        false
    }
}

/// The linear layers' kernels of a session nothing forces or tunes:
/// oneDNN's where it runs, at every precision, as the faster on a B70
/// (docs/levelzero.md).
pub(super) fn default_linear(c: &Context, half: Option<&Transposed>) -> Linear {
    if onednn_runs(c, half) { Linear::Onednn } else { Linear::Own }
}

/// A test's choices line in place of TURBO_LEVELZERO_CHOICES (use_choices).
#[cfg(feature = "internals")]
pub(super) static TEST_CHOICES: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

#[cfg(feature = "internals")]
pub(crate) fn use_choices(choices: Option<&str>) {
    *TEST_CHOICES.lock().unwrap_or_else(|p| p.into_inner()) = choices.map(str::to_owned);
}

/// The linear layers' kernels TURBO_LEVELZERO_CHOICES forces, read when a
/// session is made; None when unset or it names no linear item.
pub(super) fn forced_linear() -> Res<Option<Linear>> {
    #[cfg(feature = "internals")]
    if let Some(line) = TEST_CHOICES.lock().unwrap_or_else(|p| p.into_inner()).clone() {
        return parse_linear(&line).map_err(|e| fail(INVALID_ARGUMENT, format!("TURBO_LEVELZERO_CHOICES: {e}")));
    }
    match std::env::var("TURBO_LEVELZERO_CHOICES") {
        Ok(line) => parse_linear(&line).map_err(|e| fail(INVALID_ARGUMENT, format!("TURBO_LEVELZERO_CHOICES: {e}"))),
        Err(_) => Ok(None),
    }
}

/// The linear item of a choices line, `linear=own` or `linear=onednn`,
/// among `;`-separated items; a `forced=` item, as a session reports it,
/// is passed over. None when it names no linear item.
pub(super) fn parse_linear(line: &str) -> Result<Option<Linear>, String> {
    let mut linear = None;
    for item in line.split(';').filter(|i| !i.is_empty() && !i.starts_with("forced=")) {
        linear = match item.split_once('=') {
            Some(("linear", "own")) => Some(Linear::Own),
            Some(("linear", "onednn")) => Some(Linear::Onednn),
            _ => return Err(format!("{item:?} is not linear=own or linear=onednn")),
        };
    }
    Ok(linear)
}

pub(crate) unsafe extern "C" fn session_release(session: *mut c_void) {
    quietly(|| drop(unsafe { Box::from_raw(session as *mut Session) }));
}
