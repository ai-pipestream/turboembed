//! Models and embed sessions on a listed GPU: the BERT encoder in the
//! kernels of core/levelzero/encoder.cl.
//!
//! Loading copies each tensor, in the dtype it is stored in, into one
//! device allocation: that copy is the load, and the core's host bytes are
//! not read again. A model stored in F16 or BF16 gets a second allocation,
//! its weights widened to F32 on the device, made by the first session
//! that computes in F32 and shared by every later one; it goes with the
//! model.
//!
//! A session is device scratch for its largest batch, host staging for the
//! rows, and the buffer its vectors are written to, all allocated when it
//! is made; a run allocates nothing. The rows are computed over the written
//! batch's full [batch, seq] grid: attention skips masked keys and no
//! pooling reads a masked token, so the padding a row carries changes no
//! output. embed_write sends the rows to the device and waits for them;
//! the run leaves the vectors on the device, in the session's DEVICE
//! buffer, which turbo_result_read copies back through buffer_read.

use std::ffi::c_void;
use std::sync::Mutex;

use super::gpu::{
    Arg, Buffer, Context, Kernel, LOG_DEBUG, Queue, Res, device_allocs_here, fail, fail_field, guarded, quietly,
};
use super::ze;
use crate::backend::{
    TURBO_BERT_EMBEDDING_TENSORS, TURBO_BERT_LAYER_TENSORS, TURBO_FAMILY_BERT, turbo_backend_embed_rows,
    turbo_backend_model, turbo_backend_run,
};
use crate::status::{INVALID_ARGUMENT, INVALID_STATE, UNSUPPORTED, UNSUPPORTED_OPTION, UNSUPPORTED_TASK};
use crate::{
    TURBO_DTYPE_F16, TURBO_DTYPE_F32, TURBO_EMBED_STAGE_DOWNLOAD, TURBO_EMBED_STAGE_ENCODE, TURBO_EMBED_STAGE_LOOKUP,
    TURBO_EMBED_STAGE_NORMALIZE, TURBO_EMBED_STAGE_POOL, TURBO_EMBED_STAGE_UPLOAD, TURBO_NORMALIZE_L2,
    TURBO_PLACE_DEVICE, TURBO_PRECISION_FASTEST, TURBO_PRECISION_MODEL, TURBO_STAGE_DEVICE, TURBO_STAGE_FUSED,
    TURBO_STAGE_UNUSED, TURBO_TASK_EMBED, turbo_error,
};

/// TURBO_BERT_* in turbo_backend.h: the embedding tensors, then each
/// layer's at these offsets.
const WORD: usize = 0;
const POSITION: usize = 1;
const TOKEN_TYPE: usize = 2;
const EMB_LN_W: usize = 3;
const EMB_LN_B: usize = 4;
const Q_WEIGHT: u32 = 0;
const Q_BIAS: u32 = 1;
const K_WEIGHT: u32 = 2;
const K_BIAS: u32 = 3;
const V_WEIGHT: u32 = 4;
const V_BIAS: u32 = 5;
const ATTN_OUT_WEIGHT: u32 = 6;
const ATTN_OUT_BIAS: u32 = 7;
const ATTN_LN_WEIGHT: u32 = 8;
const ATTN_LN_BIAS: u32 = 9;
const FFN_IN_WEIGHT: u32 = 10;
const FFN_IN_BIAS: u32 = 11;
const FFN_OUT_WEIGHT: u32 = 12;
const FFN_OUT_BIAS: u32 = 13;
const FFN_LN_WEIGHT: u32 = 14;
const FFN_LN_BIAS: u32 = 15;

/// Tensors and scratch in one device allocation each start on 256 bytes.
const DEVICE_ALIGN: usize = 256;
/// Work-items per group for the row kernels, as encoder.cl's BLOCK.
const BLOCK: u32 = 128;
/// Work-items per group for the elementwise kernels.
const WIDE: u32 = 256;
/// The linear kernel's output tile, as encoder.cl's TILE.
const TILE: u32 = 64;

fn round_up(n: usize, a: usize) -> usize {
    n.div_ceil(a) * a
}

fn dtype_name(d: u32) -> &'static str {
    match d {
        TURBO_DTYPE_F32 => "F32",
        TURBO_DTYPE_F16 => "F16",
        _ => "BF16",
    }
}

/// Groups for an elementwise pass over n values: enough to fill the
/// device, each work-item striding over the rest.
fn elementwise_groups(n: u64) -> u32 {
    n.div_ceil(WIDE as u64).clamp(1, 65535) as u32
}

// ---- Models --------------------------------------------------------------------

pub(crate) struct Model {
    ctx: *const Context,
    desc: turbo_backend_model,
    /// Every tensor as stored, in one allocation, and where each starts.
    stored: *mut c_void,
    offsets: Vec<usize>,
    counts: Vec<u64>,
    /// The weights sessions compute from, made by the first one.
    f32: Mutex<Option<Weights>>,
    /// The linear layers' weights in F16, for the matrix engines, made by
    /// the first session at FASTEST.
    f16: Mutex<Option<Half>>,
}

/// Each layer's linear weights in F16, in one allocation: the fused Q, K
/// and V, the attention output, and the feed-forward input and output.
#[derive(Clone)]
struct Half {
    layers: Vec<[u64; 4]>,
    alloc: *mut c_void,
}

/// A model's weights in F32 on the device: every tensor, into stored for
/// an F32 model and into a widened copy for an F16 or BF16 one; and each
/// layer's Q, K and V weights and biases side by side, [3 * hidden,
/// hidden] and [3 * hidden], for one projection.
#[derive(Clone)]
struct Weights {
    tensors: Vec<u64>,
    qkv: Vec<(u64, u64)>,
    /// The widened copy; null for an F32 model.
    widened: *mut c_void,
    fused: *mut c_void,
}

// The context outlives the model (the core releases models first); the
// device memory is the driver's, for any thread.
unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {
    fn ctx(&self) -> &Context {
        unsafe { &*self.ctx }
    }

    /// The weights in F32, made on first need.
    fn f32_weights(&self) -> Res<Weights> {
        let mut f = self.f32.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(w) = f.as_ref() {
            return Ok(w.clone());
        }
        let c = self.ctx();
        let base = self.stored as u64;
        let (tensors, widened) = if self.desc.dtype == TURBO_DTYPE_F32 {
            (self.offsets.iter().map(|&o| base + o as u64).collect::<Vec<_>>(), std::ptr::null_mut())
        } else {
            self.widen()?
        };
        let fused = match self.fuse(&tensors) {
            Ok(p) => p,
            Err(e) => {
                if !widened.is_null() {
                    c.free(widened, "the widened weights");
                }
                return Err(e);
            }
        };
        let (h, layers) = (self.desc.hidden as u64, self.desc.layers as u64);
        let per_layer = 3 * h * h * 4 + 3 * h * 4;
        let qkv =
            (0..layers).map(|l| (fused as u64 + l * per_layer, fused as u64 + l * per_layer + 3 * h * h * 4)).collect();
        let w = Weights { tensors, qkv, widened, fused };
        *f = Some(w.clone());
        Ok(w)
    }

    /// The linear layers' weights in F16, made on first need from the F32
    /// ones.
    fn f16_weights(&self, w: &Weights) -> Res<Half> {
        let mut f = self.f16.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(h) = f.as_ref() {
            return Ok(h.clone());
        }
        let c = self.ctx();
        let d = &self.desc;
        let (h, i) = (d.hidden as u64, d.intermediate as u64);
        let t = |l: u32, r: u32| w.tensors[(TURBO_BERT_EMBEDDING_TENSORS + l * TURBO_BERT_LAYER_TENSORS + r) as usize];
        // Each layer's four weights: where they are in F32, and their size.
        let mut parts = Vec::with_capacity(4 * d.layers as usize);
        for l in 0..d.layers {
            parts.push((w.qkv[l as usize].0, 3 * h * h));
            parts.push((t(l, ATTN_OUT_WEIGHT), h * h));
            parts.push((t(l, FFN_IN_WEIGHT), i * h));
            parts.push((t(l, FFN_OUT_WEIGHT), h * i));
        }
        let mut at = Vec::with_capacity(parts.len());
        let mut total = 0usize;
        for &(_, n) in &parts {
            at.push(total as u64);
            total += round_up(n as usize * 2, DEVICE_ALIGN);
        }
        let alloc = c.alloc_device(total)?;
        let narrowed = (|| {
            let k = c.kernel("narrow_f16", [WIDE, 1, 1])?;
            let mut q = c.lock_queue()?;
            let appended = (|| {
                for (p, &(src, n)) in parts.iter().enumerate() {
                    let args = [Arg::Ptr(src), Arg::U64(n), Arg::Ptr(alloc as u64 + at[p])];
                    k.launch(c, &mut q, "narrow_f16", &args, [elementwise_groups(n), 1, 1])?;
                }
                Ok(())
            })();
            let synced = c.sync(&mut q);
            appended?;
            synced
        })();
        if let Err(e) = narrowed {
            c.free(alloc, "the F16 weights");
            return Err(e);
        }
        let layers = at.chunks(4).map(|a| [0, 1, 2, 3].map(|j| alloc as u64 + a[j])).collect();
        let half = Half { layers, alloc };
        *f = Some(half.clone());
        Ok(half)
    }

    /// An F16 or BF16 model's tensors widened to F32 in a new allocation.
    fn widen(&self) -> Res<(Vec<u64>, *mut c_void)> {
        let c = self.ctx();
        let base = self.stored as u64;
        let mut at = Vec::with_capacity(self.counts.len());
        let mut total = 0usize;
        for &n in &self.counts {
            at.push(total);
            total += round_up(n as usize * 4, DEVICE_ALIGN);
        }
        let wide = c.alloc_device(total)?;
        let name = if self.desc.dtype == TURBO_DTYPE_F16 { "widen_f16" } else { "widen_bf16" };
        let widened = (|| {
            let k = c.kernel(name, [WIDE, 1, 1])?;
            let mut q = c.lock_queue()?;
            // The queue is left idle whether or not every launch was
            // appended, before a failure frees what they write to.
            let appended = (|| {
                for (i, &n) in self.counts.iter().enumerate() {
                    let args =
                        [Arg::Ptr(base + self.offsets[i] as u64), Arg::U64(n), Arg::Ptr(wide as u64 + at[i] as u64)];
                    k.launch(c, &mut q, name, &args, [elementwise_groups(n), 1, 1])?;
                }
                Ok(())
            })();
            let synced = c.sync(&mut q);
            appended?;
            synced
        })();
        if let Err(e) = widened {
            c.free(wide, "the widened weights");
            return Err(e);
        }
        Ok((at.iter().map(|&a| wide as u64 + a as u64).collect(), wide))
    }

    /// Each layer's Q, K and V weights, then their biases, copied side by
    /// side on the device from the F32 tensors.
    fn fuse(&self, tensors: &[u64]) -> Res<*mut c_void> {
        let c = self.ctx();
        let (h, layers) = (self.desc.hidden as usize, self.desc.layers as usize);
        let (weight, bias) = (h * h * 4, h * 4);
        let fused = c.alloc_device(layers * 3 * (weight + bias))?;
        let copied = (|| {
            let mut q = c.lock_queue()?;
            let appended = (|| {
                for l in 0..layers {
                    let dst = fused as usize + l * 3 * (weight + bias);
                    let t = |r: u32| {
                        tensors[(TURBO_BERT_EMBEDDING_TENSORS + l as u32 * TURBO_BERT_LAYER_TENSORS + r) as usize]
                    };
                    for (i, (w, b)) in
                        [(Q_WEIGHT, Q_BIAS), (K_WEIGHT, K_BIAS), (V_WEIGHT, V_BIAS)].into_iter().enumerate()
                    {
                        unsafe {
                            c.copy(&mut q, (dst + i * weight) as *mut c_void, t(w) as usize as *const c_void, weight)?;
                            c.copy(
                                &mut q,
                                (dst + 3 * weight + i * bias) as *mut c_void,
                                t(b) as usize as *const c_void,
                                bias,
                            )?;
                        }
                    }
                }
                Ok(())
            })();
            let synced = c.sync(&mut q);
            appended?;
            synced
        })();
        if let Err(e) = copied {
            c.free(fused, "the fused projections");
            return Err(e);
        }
        Ok(fused)
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        let c = unsafe { &*self.ctx };
        let f = self.f32.get_mut().unwrap_or_else(|p| p.into_inner());
        if let Some(w) = f.take() {
            if !w.widened.is_null() {
                c.free(w.widened, "the widened weights");
            }
            c.free(w.fused, "the fused projections");
        }
        let h = self.f16.get_mut().unwrap_or_else(|p| p.into_inner());
        if let Some(h) = h.take() {
            c.free(h.alloc, "the F16 weights");
        }
        if !self.stored.is_null() {
            c.free(self.stored, "the weights");
        }
    }
}

pub(crate) unsafe extern "C" fn model_load(
    ctx: *mut c_void,
    desc: *const turbo_backend_model,
    out: *mut *mut c_void,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        guarded(err, || {
            let (c, d) = (&*(ctx as *const Context), &*desc);
            if d.family != TURBO_FAMILY_BERT {
                return Err(fail(
                    UNSUPPORTED,
                    format!("family {}: the levelzero backend holds BERT encoders", d.family),
                ));
            }
            if d.heads == 0 || d.hidden % d.heads != 0 {
                return Err(fail(UNSUPPORTED, format!("hidden {} is not a multiple of heads {}", d.hidden, d.heads)));
            }
            let elem = if d.dtype == TURBO_DTYPE_F32 { 4 } else { 2 };
            let tensors = std::slice::from_raw_parts(d.tensors, d.tensor_count as usize);
            let mut offsets = Vec::with_capacity(tensors.len());
            let mut counts = Vec::with_capacity(tensors.len());
            let mut total = 0usize;
            for t in tensors {
                offsets.push(total);
                counts.push(t.bytes / elem);
                total += round_up(t.bytes as usize, DEVICE_ALIGN);
            }
            let stored = c.alloc_device(total)?;
            // From here the model's Drop frees what it holds.
            let m = Box::new(Model {
                ctx: c,
                desc: turbo_backend_model { tensors: std::ptr::null(), ..*d },
                stored,
                offsets,
                counts,
                f32: Mutex::new(None),
                f16: Mutex::new(None),
            });
            {
                let mut q = c.lock_queue()?;
                // The queue is left idle whether or not every copy was
                // appended, before a failure frees what they write to.
                let appended = (|| {
                    for (i, t) in tensors.iter().enumerate() {
                        let dst = (stored as *mut u8).add(m.offsets[i]) as *mut c_void;
                        c.copy(&mut q, dst, t.data, t.bytes as usize)?;
                    }
                    Ok(())
                })();
                let synced = c.sync(&mut q);
                appended?;
                synced?;
            }
            c.say(
                LOG_DEBUG,
                &format!(
                    "levelzero device {}: a BERT of {} layers, {total} bytes of {} weights",
                    c.ordinal,
                    d.layers,
                    dtype_name(d.dtype)
                ),
            );
            *out = Box::into_raw(m) as *mut c_void;
            Ok(())
        })
    }
}

/// The device address of the F32 copy of an F16 or BF16 model's weights,
/// once a session made it.
///
/// # Safety
/// `model` is one model_load returned, not yet released.
#[cfg(feature = "internals")]
pub(crate) unsafe fn widened(model: *mut c_void) -> Option<*const c_void> {
    let m = unsafe { &*(model as *const Model) };
    let f = m.f32.lock().unwrap_or_else(|p| p.into_inner());
    f.as_ref().map(|w| w.widened as *const c_void).filter(|p| !p.is_null())
}

pub(crate) unsafe extern "C" fn model_release(model: *mut c_void) {
    quietly(|| drop(unsafe { Box::from_raw(model as *mut Model) }));
}

// ---- Sessions ------------------------------------------------------------------

/// The encoder's kernel objects, one set per session.
struct Kernels {
    linear: Kernel,
    /// The linear layers in F32 by sub-group, where the terms come in
    /// sixteens.
    linear_sg: Kernel,
    /// The linear layers on the matrix engines, for a session at FASTEST:
    /// F32 activations to F32, F32 to F16, and F16 to F32; then the same
    /// with a group's sub-groups sharing its tile.
    linear_xmx: Option<[Kernel; 6]>,
    embed_layer_norm: Kernel,
    add_layer_norm: Kernel,
    attention: Attention,
    pool: Kernel,
}

/// Attention for the model's head width: a tiled kernel where one is
/// built for it, else the general one.
enum Attention {
    Tiled(Kernel),
    General(Kernel),
}

impl Kernels {
    fn new(c: &Context, head_dim: u32, xmx: bool) -> Res<Kernels> {
        let row = [BLOCK, 1, 1];
        let attention = match head_dim {
            32 | 64 | 128 => Attention::Tiled(c.kernel(&format!("attention_{head_dim}"), [QUERIES, 1, 1])?),
            _ => Attention::General(c.kernel("attention", row)?),
        };
        Ok(Kernels {
            linear: c.kernel("linear", [16, 16, 1])?,
            linear_sg: c.kernel("linear_sg", [16, 1, 1])?,
            linear_xmx: if xmx {
                let (one, shared) = ([16, 1, 1], [16 * XMX_SUBGROUPS, 1, 1]);
                Some([
                    c.kernel("linear_xmx", one)?,
                    c.kernel("linear_xmx_to_half", one)?,
                    c.kernel("linear_xmx_from_half", one)?,
                    c.kernel("linear_xmx_shared", shared)?,
                    c.kernel("linear_xmx_shared_to_half", shared)?,
                    c.kernel("linear_xmx_shared_from_half", shared)?,
                ])
            } else {
                None
            },
            embed_layer_norm: c.kernel("embed_layer_norm", row)?,
            add_layer_norm: c.kernel("add_layer_norm", row)?,
            attention,
            pool: c.kernel("pool", row)?,
        })
    }
}

/// The parts the feed-forward output's sums are split into, over its
/// terms, so its few output tiles still fill the device; the LayerNorm
/// after it adds them.
const SPLITS: u32 = 4;

/// Operands of the XMX linear kernels: F32 activations to F32, F32 to
/// F16, and F16 to F32.
const XMX_F32: usize = 0;
const XMX_TO_F16: usize = 1;
const XMX_FROM_F16: usize = 2;

/// The F32 sub-group linear kernel's tile, as encoder.cl's SG_N and SG_T.
const SG_N: u32 = 64;
const SG_T: u32 = 8;

/// The XMX linear kernel's output tile, as encoder.cl's XM and XN, and the
/// sub-groups of a group, each summing its own share of the terms, as
/// encoder.cl's KS.
const XMX_TILE: u32 = 32;
const XMX_SUBGROUPS: u32 = 4;
/// Below this many tiles a layer's groups share theirs among sub-groups.
const XMX_FEW_TILES: u32 = 512;

/// The terms each of an XMX group's sub-groups sums: an equal share, a
/// multiple of 16, of k_len. None when no such share covers k_len
/// exactly with the sub-groups there are.
fn xmx_share(k_len: u32) -> Option<u32> {
    let share = k_len.div_ceil(16 * XMX_SUBGROUPS) * 16;
    (k_len.is_multiple_of(share) && k_len / share <= XMX_SUBGROUPS).then_some(share)
}

/// Queries a tiled attention group takes, as encoder.cl's QUERIES.
const QUERIES: u32 = 256;

/// The linear kernel's epilogue, as encoder.cl's LINEAR_*.
const LINEAR_BIAS: i32 = 1;
const LINEAR_GELU: i32 = 2;

/// Bytes of local memory the general attention kernel takes for rows of
/// seq tokens: the query's head, the row's scores and the partial
/// contexts.
fn attention_local_bytes(seq: u32, head_dim: u32) -> u64 {
    let part = if head_dim <= BLOCK { (BLOCK / head_dim) * head_dim } else { 0 };
    4 * (head_dim as u64 + seq as u64 + part as u64)
}

struct Session {
    model: *const Model,
    ctx: *const Context,
    kernels: Kernels,
    weights: Weights,
    /// The linear layers' F16 weights, for a session at FASTEST.
    half: Option<Half>,
    max_batch: u32,
    max_seq: u32,
    scratch: *mut c_void,
    /// Each row's first packed token and its length through its last live
    /// token, [batch, 2] int32, as the lookup kernel copies it from the
    /// staging.
    rows: u64,
    // Packed, [tokens, ...]: the mask, the hidden states, the fused
    // projections, the attention context, a projection's output, and the
    // feed-forward block.
    packed_mask: u64,
    x: u64,
    qkv: u64,
    att: u64,
    tmp: u64,
    ffn: u64,
    /// [max_batch, hidden] F32 on the device, handed out as the output.
    output: Box<Buffer>,
    /// The packed rows, which the device reads directly: ids, positions,
    /// types and mask, [max_batch * max_seq] int32 each, then the row
    /// table, [max_batch, 2].
    staging: *mut c_void,
    /// What the last write left.
    written: bool,
    has_types: bool,
    batch: u32,
    seq: u32,
    /// Live tokens, packed.
    tokens: u32,
    longest: u32,
    pooling: u32,
    normalize: u32,
    output_dim: u32,
    h2d: u64,
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
                        "precision: MODEL computes in the weights' {}, and the levelzero backend computes in F32 \
                         only; EXACT and FASTEST compute this model in F32",
                        dtype_name(d.dtype)
                    ),
                ));
            }
            if max_seq > d.max_positions {
                return Err(fail_field(
                    UNSUPPORTED_OPTION,
                    2,
                    format!("max_seq {max_seq} is over the model's {} positions", d.max_positions),
                ));
            }
            let head_dim = d.hidden / d.heads;
            // FASTEST runs the linear layers on the matrix engines, 16
            // terms at a time.
            let ffn_out_terms =
                if d.intermediate.is_multiple_of(SPLITS * 16) { d.intermediate / SPLITS } else { d.intermediate };
            let xmx = precision == TURBO_PRECISION_FASTEST
                && d.hidden.is_multiple_of(16)
                && d.intermediate.is_multiple_of(16)
                && xmx_share(d.hidden).is_some()
                && xmx_share(d.intermediate).is_some()
                && xmx_share(ffn_out_terms).is_some();
            let kernels = Kernels::new(c, head_dim, xmx)?;
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
            let ints = round_up(tokens * 4, DEVICE_ALIGN);
            let table = round_up(max_batch as usize * 8, DEVICE_ALIGN);
            let wide = round_up(tokens * d.hidden as usize * 4, DEVICE_ALIGN);
            let ffn = round_up(tokens * d.intermediate as usize * 4, DEVICE_ALIGN);
            let output = round_up(max_batch as usize * d.hidden as usize * 4, DEVICE_ALIGN);
            let scratch = c.alloc_device(ints + table + (5 + SPLITS as usize) * wide + ffn + output)?;
            let mut s = Box::new(Session {
                model: m,
                ctx: c,
                kernels,
                weights,
                half,
                max_batch,
                max_seq,
                scratch,
                rows: 0,
                packed_mask: 0,
                x: 0,
                qkv: 0,
                att: 0,
                tmp: 0,
                ffn: 0,
                output: Box::new(Buffer::session_output(c, std::ptr::null_mut(), 0)),
                staging: std::ptr::null_mut(),
                written: false,
                has_types: false,
                batch: 0,
                seq: 0,
                tokens: 0,
                longest: 0,
                pooling: 0,
                normalize: 0,
                output_dim: 0,
                h2d: 0,
            });
            s.staging = c.alloc_pinned(4 * tokens * 4 + max_batch as usize * 8)?;
            let mut p = scratch as u64;
            let mut take = |n: usize| {
                let at = p;
                p += n as u64;
                at
            };
            s.packed_mask = take(ints);
            s.rows = take(table);
            s.x = take(wide);
            s.qkv = take(3 * wide);
            s.att = take(wide);
            s.tmp = take(SPLITS as usize * wide);
            s.ffn = take(ffn);
            let out_ptr = take(output) as usize as *mut c_void;
            s.output = Box::new(Buffer::session_output(c, out_ptr, max_batch as u64 * d.hidden as u64 * 4));
            *compute_dtype = if xmx { TURBO_DTYPE_F16 } else { TURBO_DTYPE_F32 };
            *out = Box::into_raw(s) as *mut c_void;
            Ok(())
        })
    }
}

pub(crate) unsafe extern "C" fn session_release(session: *mut c_void) {
    quietly(|| drop(unsafe { Box::from_raw(session as *mut Session) }));
}

impl Session {
    fn ctx(&self) -> &Context {
        unsafe { &*self.ctx }
    }

    fn model(&self) -> &Model {
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
    unsafe fn pack(&self, r: &turbo_backend_embed_rows) -> Res<(u32, u32, u64)> {
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
                    *positions.add(t + p) = p as i32;
                }
            }
            t += len;
            longest = longest.max(len as u32);
        }
        let arrays = if r.types.is_null() { 3 } else { 4 };
        Ok((t as u32, longest, (arrays * t * 4 + r.batch as usize * 8) as u64))
    }

    /// The encoder over the packed rows, appended to the queue.
    fn encode(&self, q: &mut Queue) -> Res<()> {
        let c = self.ctx();
        let d = &self.model().desc;
        let w = &self.weights.tensors;
        let k = &self.kernels;
        let (batch, tokens) = (self.batch, self.tokens);
        let (h, inter) = (d.hidden, d.intermediate);
        let eps = d.layer_norm_eps as f32;
        let head_dim = h / d.heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let layer = |l: u32, r: u32| w[(TURBO_BERT_EMBEDDING_TENSORS + l * TURBO_BERT_LAYER_TENSORS + r) as usize];
        use Arg::*;
        // A linear layer: `which` of the layer's four weights, for the F16
        // copy at FASTEST, and the F32 weight; its sums split over its
        // terms into `splits` parts; and at FASTEST, which operands.
        let linear = |q: &mut Queue,
                      x: u64,
                      n_in: u32,
                      (l, which, weight): (u32, usize, u64),
                      bias: u64,
                      n_out: u32,
                      y: u64,
                      flags: i32,
                      (splits, operands): (u32, usize),
                      what: &str| {
            // A split takes an equal share of the terms, a multiple of 16.
            let splits = if n_in.is_multiple_of(splits * 16) { splits } else { 1 };
            let k_len = n_in / splits;
            if let (Some(kx), Some(half)) = (&k.linear_xmx, &self.half) {
                let groups = [n_out.div_ceil(XMX_TILE), tokens.div_ceil(XMX_TILE), splits];
                let args = [
                    Ptr(x),
                    Ptr(half.layers[l as usize][which]),
                    Ptr(bias),
                    Ptr(y),
                    I32(tokens as i32),
                    I32(n_out as i32),
                    I32(n_in as i32),
                    I32(flags),
                    I32(k_len as i32),
                    I32(xmx_share(k_len).unwrap_or(k_len) as i32),
                ];
                // Too few tiles to fill the device: each group's
                // sub-groups share one.
                let few = groups[0] * groups[1] * groups[2] < XMX_FEW_TILES;
                let groups = if few { groups } else { [n_out.div_ceil(SG_N), tokens.div_ceil(SG_T), splits] };
                let kernel = &kx[operands + if few { 3 } else { 0 }];
                let args = if few {
                    args
                } else {
                    [args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], I32(k_len as i32)]
                };
                return kernel.launch(c, q, what, &args, groups);
            }
            if k_len.is_multiple_of(16) {
                let groups = [n_out.div_ceil(SG_N), tokens.div_ceil(SG_T), splits];
                let args = [
                    Ptr(x),
                    Ptr(weight),
                    Ptr(bias),
                    Ptr(y),
                    I32(tokens as i32),
                    I32(n_out as i32),
                    I32(n_in as i32),
                    I32(flags),
                    I32(k_len as i32),
                ];
                return k.linear_sg.launch(c, q, what, &args, groups);
            }
            let groups = [n_out.div_ceil(TILE), tokens.div_ceil(TILE), splits];
            let args = [
                Ptr(x),
                Ptr(weight),
                Ptr(bias),
                Ptr(y),
                I32(tokens as i32),
                I32(n_out as i32),
                I32(n_in as i32),
                I32(flags),
                I32(k_len as i32),
            ];
            k.linear.launch(c, q, what, &args, groups)
        };
        let add_ln = |q: &mut Queue, bias: u64, lnw: u64, lnb: u64, parts: u32, what: &str| {
            let args =
                [Ptr(self.x), Ptr(self.tmp), Ptr(bias), Ptr(lnw), Ptr(lnb), F32(eps), I32(h as i32), I32(parts as i32)];
            k.add_layer_norm.launch(c, q, what, &args, [tokens, 1, 1])
        };

        let n = self.max_batch as u64 * self.max_seq as u64 * 4;
        let staging = self.staging as u64;
        let args = [
            Ptr(staging),
            Ptr(staging + n),
            Ptr(staging + 2 * n),
            I32(self.has_types as i32),
            Ptr(staging + 3 * n),
            Ptr(staging + 4 * n),
            I32(batch as i32),
            Ptr(w[WORD]),
            Ptr(w[POSITION]),
            Ptr(w[TOKEN_TYPE]),
            Ptr(w[EMB_LN_W]),
            Ptr(w[EMB_LN_B]),
            F32(eps),
            I32(h as i32),
            Ptr(self.x),
            Ptr(self.packed_mask),
            Ptr(self.rows),
        ];
        k.embed_layer_norm.launch(c, q, "the embedding lookup", &args, [tokens, 1, 1])?;
        for l in 0..d.layers {
            let (qkv_w, qkv_b) = self.weights.qkv[l as usize];
            linear(
                q,
                self.x,
                h,
                (l, 0, qkv_w),
                qkv_b,
                3 * h,
                self.qkv,
                LINEAR_BIAS,
                (1, XMX_F32),
                "the query, key and value projection",
            )?;
            match &k.attention {
                Attention::Tiled(a) => {
                    let args = [
                        Ptr(self.qkv),
                        Ptr(self.packed_mask),
                        Ptr(self.rows),
                        I32(h as i32),
                        F32(scale),
                        Ptr(self.att),
                    ];
                    a.launch(c, q, "attention", &args, [self.longest.div_ceil(QUERIES), d.heads, batch])?;
                }
                Attention::General(a) => {
                    let args = [
                        Ptr(self.qkv),
                        Ptr(self.packed_mask),
                        Ptr(self.rows),
                        I32(h as i32),
                        I32(head_dim as i32),
                        F32(scale),
                        Ptr(self.att),
                        Local(attention_local_bytes(self.longest, head_dim) as usize),
                    ];
                    a.launch(c, q, "attention", &args, [self.longest, d.heads, batch])?;
                }
            }
            let wo = (l, 1, layer(l, ATTN_OUT_WEIGHT));
            linear(q, self.att, h, wo, 0, h, self.tmp, 0, (1, XMX_F32), "the attention output projection")?;
            add_ln(
                q,
                layer(l, ATTN_OUT_BIAS),
                layer(l, ATTN_LN_WEIGHT),
                layer(l, ATTN_LN_BIAS),
                1,
                "the attention LayerNorm",
            )?;
            let (wi, bi) = ((l, 2, layer(l, FFN_IN_WEIGHT)), layer(l, FFN_IN_BIAS));
            linear(
                q,
                self.x,
                h,
                wi,
                bi,
                inter,
                self.ffn,
                LINEAR_BIAS | LINEAR_GELU,
                (1, XMX_TO_F16),
                "the feed-forward input and GELU",
            )?;
            let wf = (l, 3, layer(l, FFN_OUT_WEIGHT));
            linear(q, self.ffn, inter, wf, 0, h, self.tmp, 0, (SPLITS, XMX_FROM_F16), "the feed-forward output")?;
            add_ln(
                q,
                layer(l, FFN_OUT_BIAS),
                layer(l, FFN_LN_WEIGHT),
                layer(l, FFN_LN_BIAS),
                if inter.is_multiple_of(SPLITS * 16) { SPLITS } else { 1 },
                "the feed-forward LayerNorm",
            )?;
        }
        let args = [
            Ptr(self.x),
            Ptr(self.packed_mask),
            Ptr(self.rows),
            I32(h as i32),
            I32(self.output_dim as i32),
            I32(self.pooling as i32),
            I32((self.normalize == TURBO_NORMALIZE_L2) as i32),
            Ptr(self.output.ptr as u64),
        ];
        k.pool.launch(c, q, "pooling", &args, [batch, 1, 1])
    }
}

pub(crate) unsafe extern "C" fn embed_write(
    session: *mut c_void,
    rows: *const turbo_backend_embed_rows,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        guarded(err, || {
            let (s, r) = (&mut *(session as *mut Session), &*rows);
            s.written = false;
            // No crossing yet: the lookup kernel reads the packed rows from
            // the staging when the run starts.
            let (tokens, longest, sent) = s.pack(r)?;
            s.has_types = !r.types.is_null();
            s.batch = r.batch;
            s.seq = r.seq;
            s.tokens = tokens;
            s.longest = longest;
            s.pooling = r.pooling;
            s.normalize = r.normalize;
            s.output_dim = r.output_dim;
            s.h2d = sent;
            s.written = true;
            Ok(())
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
            {
                let c = s.ctx();
                let mut q = c.lock_queue()?;
                let encoded = s.encode(&mut q);
                // The queue is left idle whether or not the run finished.
                let synced = c.sync(&mut q);
                encoded?;
                synced?;
            }
            out.placement = TURBO_PLACE_DEVICE;
            out.output = &*s.output as *const Buffer as *mut c_void;
            out.host = std::ptr::null_mut();
            out.h2d_bytes = s.h2d;
            out.d2h_bytes = 0;
            // The run appends launches to memory the session holds: the
            // kernels' arguments are set in place and nothing is allocated.
            out.host_allocs = 0;
            out.device_allocs = device_allocs_here() - device0;
            let st = &mut out.stage;
            // The lookup kernel reads the packed rows over the bus.
            st[TURBO_EMBED_STAGE_UPLOAD] = TURBO_STAGE_FUSED;
            st[TURBO_EMBED_STAGE_LOOKUP] = TURBO_STAGE_DEVICE;
            st[TURBO_EMBED_STAGE_ENCODE] = TURBO_STAGE_DEVICE;
            st[TURBO_EMBED_STAGE_POOL] = TURBO_STAGE_DEVICE;
            // Normalization is the last step of the pooling kernel.
            st[TURBO_EMBED_STAGE_NORMALIZE] =
                if s.normalize == TURBO_NORMALIZE_L2 { TURBO_STAGE_FUSED } else { TURBO_STAGE_UNUSED };
            st[TURBO_EMBED_STAGE_DOWNLOAD] = TURBO_STAGE_UNUSED;
            Ok(())
        })
    }
}
