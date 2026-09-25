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
    Arg, Buffer, Context, Kernel, LOG_DEBUG, Res, device_allocs_here, fail, fail_field, guarded, quietly,
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
    TURBO_PLACE_DEVICE, TURBO_PRECISION_MODEL, TURBO_STAGE_DEVICE, TURBO_STAGE_FUSED, TURBO_STAGE_UNUSED,
    TURBO_TASK_EMBED, turbo_error,
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
    /// Every tensor's F32 device address: into stored for an F32 model,
    /// else into the widened copy once a session made it.
    f32: Mutex<Option<(Vec<u64>, *mut c_void)>>,
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
    fn f32_weights(&self) -> Res<Vec<u64>> {
        let mut f = self.f32.lock().unwrap_or_else(|p| p.into_inner());
        if let Some((w, _)) = f.as_ref() {
            return Ok(w.clone());
        }
        let c = self.ctx();
        let base = self.stored as u64;
        if self.desc.dtype == TURBO_DTYPE_F32 {
            let w = self.offsets.iter().map(|&o| base + o as u64).collect::<Vec<_>>();
            *f = Some((w.clone(), std::ptr::null_mut()));
            return Ok(w);
        }
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
                    k.launch(c, q.0, name, &args, [elementwise_groups(n), 1, 1])?;
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
        let w = at.iter().map(|&a| wide as u64 + a as u64).collect::<Vec<_>>();
        *f = Some((w.clone(), wide));
        Ok(w)
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        let c = unsafe { &*self.ctx };
        let f = self.f32.get_mut().unwrap_or_else(|p| p.into_inner());
        if let Some((_, wide)) = f.take()
            && !wide.is_null()
        {
            c.free(wide, "the widened weights");
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
            });
            {
                let mut q = c.lock_queue()?;
                // The queue is left idle whether or not every copy was
                // appended, before a failure frees what they write to.
                let appended = (|| {
                    for (i, t) in tensors.iter().enumerate() {
                        let dst = (stored as *mut u8).add(m.offsets[i]) as *mut c_void;
                        c.copy(q.0, dst, t.data, t.bytes as usize)?;
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
    f.as_ref().map(|(_, wide)| *wide as *const c_void).filter(|p| !p.is_null())
}

pub(crate) unsafe extern "C" fn model_release(model: *mut c_void) {
    quietly(|| drop(unsafe { Box::from_raw(model as *mut Model) }));
}

// ---- Sessions ------------------------------------------------------------------

/// The encoder's kernel objects, one set per session.
struct Kernels {
    linear: Kernel,
    embed_layer_norm: Kernel,
    add_layer_norm: Kernel,
    bias_gelu: Kernel,
    attention: Kernel,
    pool: Kernel,
}

impl Kernels {
    fn new(c: &Context) -> Res<Kernels> {
        let row = [BLOCK, 1, 1];
        Ok(Kernels {
            linear: c.kernel("linear", [16, 16, 1])?,
            embed_layer_norm: c.kernel("embed_layer_norm", row)?,
            add_layer_norm: c.kernel("add_layer_norm", row)?,
            bias_gelu: c.kernel("bias_gelu", [WIDE, 1, 1])?,
            attention: c.kernel("attention", row)?,
            pool: c.kernel("pool", row)?,
        })
    }
}

/// Bytes of local memory attention takes for rows of seq tokens: the
/// query's head, the row's scores and the partial contexts.
fn attention_local_bytes(seq: u32, head_dim: u32) -> u64 {
    let part = if head_dim <= BLOCK { (BLOCK / head_dim) * head_dim } else { 0 };
    4 * (head_dim as u64 + seq as u64 + part as u64)
}

struct Session {
    model: *const Model,
    ctx: *const Context,
    kernels: Kernels,
    weights: Vec<u64>,
    max_batch: u32,
    max_seq: u32,
    scratch: *mut c_void,
    ids: u64,
    mask: u64,
    types: u64,
    x: u64,
    q: u64,
    k: u64,
    v: u64,
    att: u64,
    tmp: u64,
    ffn: u64,
    /// [max_batch, hidden] F32 on the device, handed out as the output.
    output: Box<Buffer>,
    /// [3, max_batch * max_seq] int32 the device reads directly.
    staging: *mut c_void,
    /// What the last write left.
    written: bool,
    has_types: bool,
    batch: u32,
    seq: u32,
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
            let kernels = Kernels::new(c)?;
            // The driver keeps some of a work-group's local memory for the
            // kernel's own use (its reductions); the scores get the rest.
            let head_dim = d.hidden / d.heads;
            let local = attention_local_bytes(max_seq, head_dim);
            let room = (c.max_local as u64).saturating_sub(kernels.attention.local_bytes()? as u64);
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
            let tokens = max_batch as usize * max_seq as usize;
            if tokens > i32::MAX as usize / d.intermediate.max(d.hidden) as usize {
                return Err(fail_field(
                    UNSUPPORTED_OPTION,
                    1,
                    format!("{max_batch} rows of {max_seq} tokens is more than one kernel indexes"),
                ));
            }
            let weights = m.f32_weights()?;
            let ints = round_up(tokens * 4, DEVICE_ALIGN);
            let wide = round_up(tokens * d.hidden as usize * 4, DEVICE_ALIGN);
            let ffn = round_up(tokens * d.intermediate as usize * 4, DEVICE_ALIGN);
            let output = round_up(max_batch as usize * d.hidden as usize * 4, DEVICE_ALIGN);
            let scratch = c.alloc_device(3 * ints + 6 * wide + ffn + output)?;
            let mut s = Box::new(Session {
                model: m,
                ctx: c,
                kernels,
                weights,
                max_batch,
                max_seq,
                scratch,
                ids: 0,
                mask: 0,
                types: 0,
                x: 0,
                q: 0,
                k: 0,
                v: 0,
                att: 0,
                tmp: 0,
                ffn: 0,
                output: Box::new(Buffer::session_output(c, std::ptr::null_mut(), 0)),
                staging: std::ptr::null_mut(),
                written: false,
                has_types: false,
                batch: 0,
                seq: 0,
                pooling: 0,
                normalize: 0,
                output_dim: 0,
                h2d: 0,
            });
            s.staging = c.alloc_pinned(3 * tokens * 4)?;
            let mut p = scratch as u64;
            let mut take = |n: usize| {
                let at = p;
                p += n as u64;
                at
            };
            s.ids = take(ints);
            s.mask = take(ints);
            s.types = take(ints);
            s.x = take(wide);
            s.q = take(wide);
            s.k = take(wide);
            s.v = take(wide);
            s.att = take(wide);
            s.tmp = take(wide);
            s.ffn = take(ffn);
            let out_ptr = take(output) as usize as *mut c_void;
            s.output = Box::new(Buffer::session_output(c, out_ptr, max_batch as u64 * d.hidden as u64 * 4));
            *compute_dtype = TURBO_DTYPE_F32;
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

    /// One [batch, seq] array of the rows to dst on the device, row_stride
    /// elements apart in src, returning the bytes sent. Device memory is
    /// refused: the core has read and checked the rows on the host. Host
    /// memory the driver allocated goes straight to the device; any other
    /// is copied into the session's staging first.
    ///
    /// # Safety
    /// src holds (batch - 1) * stride + seq values; the caller holds the
    /// queue and synchronizes it before the arrays go away.
    unsafe fn upload(
        &self,
        queue: ze::Handle,
        r: &turbo_backend_embed_rows,
        src: *const i32,
        slot: usize,
        dst: u64,
    ) -> Res<u64> {
        let c = self.ctx();
        let row = r.seq as usize * 4;
        let (batch, stride) = (r.batch as usize, r.row_stride as usize);
        let (kind, _) = c.memory_type(src as *const c_void);
        match kind {
            ze::MEMORY_TYPE_DEVICE => {
                return Err(fail(INVALID_ARGUMENT, "rows are device memory; the core reads rows on the host"));
            }
            ze::MEMORY_TYPE_HOST | ze::MEMORY_TYPE_SHARED if stride == r.seq as usize => unsafe {
                c.copy(queue, dst as usize as *mut c_void, src as *const c_void, row * batch)?;
            },
            ze::MEMORY_TYPE_HOST | ze::MEMORY_TYPE_SHARED => {
                for b in 0..batch {
                    unsafe {
                        c.copy(
                            queue,
                            (dst as usize + b * row) as *mut c_void,
                            src.add(b * stride) as *const c_void,
                            row,
                        )?;
                    }
                }
            }
            _ => {
                let tokens = self.max_batch as usize * self.max_seq as usize;
                let staging = unsafe { (self.staging as *mut i32).add(slot * tokens) };
                for b in 0..batch {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            src.add(b * stride),
                            staging.add(b * r.seq as usize),
                            r.seq as usize,
                        )
                    };
                }
                unsafe { c.copy(queue, dst as usize as *mut c_void, staging as *const c_void, row * batch)? };
            }
        }
        Ok((row * batch) as u64)
    }

    /// The encoder over the written rows, appended to the queue.
    fn encode(&self, queue: ze::Handle) -> Res<()> {
        let c = self.ctx();
        let d = &self.model().desc;
        let w = &self.weights;
        let k = &self.kernels;
        let (batch, seq) = (self.batch, self.seq);
        let tokens = batch * seq;
        let (h, inter) = (d.hidden, d.intermediate);
        let eps = d.layer_norm_eps as f32;
        let head_dim = h / d.heads;
        let layer = |l: u32, r: u32| w[(TURBO_BERT_EMBEDDING_TENSORS + l * TURBO_BERT_LAYER_TENSORS + r) as usize];
        use Arg::*;
        let linear = |x: u64, n_in: u32, weight: u64, n_out: u32, y: u64, what: &str| {
            let groups = [n_out.div_ceil(TILE), tokens.div_ceil(TILE), 1];
            let args = [Ptr(x), Ptr(weight), Ptr(y), I32(tokens as i32), I32(n_out as i32), I32(n_in as i32)];
            k.linear.launch(c, queue, what, &args, groups)
        };
        let add_ln = |bias: u64, lnw: u64, lnb: u64, what: &str| {
            let args = [Ptr(self.x), Ptr(self.tmp), Ptr(bias), Ptr(lnw), Ptr(lnb), F32(eps), I32(h as i32)];
            k.add_layer_norm.launch(c, queue, what, &args, [tokens, 1, 1])
        };

        let args = [
            Ptr(self.ids),
            Ptr(self.types),
            I32(self.has_types as i32),
            Ptr(w[WORD]),
            Ptr(w[POSITION]),
            Ptr(w[TOKEN_TYPE]),
            Ptr(w[EMB_LN_W]),
            Ptr(w[EMB_LN_B]),
            F32(eps),
            I32(seq as i32),
            I32(h as i32),
            Ptr(self.x),
        ];
        k.embed_layer_norm.launch(c, queue, "the embedding lookup", &args, [tokens, 1, 1])?;
        for l in 0..d.layers {
            linear(self.x, h, layer(l, Q_WEIGHT), h, self.q, "the query projection")?;
            linear(self.x, h, layer(l, K_WEIGHT), h, self.k, "the key projection")?;
            linear(self.x, h, layer(l, V_WEIGHT), h, self.v, "the value projection")?;
            let args = [
                Ptr(self.q),
                Ptr(self.k),
                Ptr(self.v),
                Ptr(layer(l, Q_BIAS)),
                Ptr(layer(l, K_BIAS)),
                Ptr(layer(l, V_BIAS)),
                Ptr(self.mask),
                I32(seq as i32),
                I32(h as i32),
                I32(head_dim as i32),
                F32(1.0 / (head_dim as f32).sqrt()),
                Ptr(self.att),
                Local(attention_local_bytes(seq, head_dim) as usize),
            ];
            k.attention.launch(c, queue, "attention", &args, [seq, d.heads, batch])?;
            linear(self.att, h, layer(l, ATTN_OUT_WEIGHT), h, self.tmp, "the attention output projection")?;
            add_ln(
                layer(l, ATTN_OUT_BIAS),
                layer(l, ATTN_LN_WEIGHT),
                layer(l, ATTN_LN_BIAS),
                "the attention LayerNorm",
            )?;
            linear(self.x, h, layer(l, FFN_IN_WEIGHT), inter, self.ffn, "the feed-forward input")?;
            let n = tokens as u64 * inter as u64;
            let args = [Ptr(self.ffn), Ptr(layer(l, FFN_IN_BIAS)), U64(n), I32(inter as i32)];
            k.bias_gelu.launch(c, queue, "GELU", &args, [elementwise_groups(n), 1, 1])?;
            linear(self.ffn, inter, layer(l, FFN_OUT_WEIGHT), h, self.tmp, "the feed-forward output")?;
            add_ln(
                layer(l, FFN_OUT_BIAS),
                layer(l, FFN_LN_WEIGHT),
                layer(l, FFN_LN_BIAS),
                "the feed-forward LayerNorm",
            )?;
        }
        let args = [
            Ptr(self.x),
            Ptr(self.mask),
            I32(seq as i32),
            I32(h as i32),
            I32(self.output_dim as i32),
            I32(self.pooling as i32),
            I32((self.normalize == TURBO_NORMALIZE_L2) as i32),
            Ptr(self.output.ptr as u64),
        ];
        k.pool.launch(c, queue, "pooling", &args, [batch, 1, 1])
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
            let c = s.ctx();
            let mut sent = 0;
            {
                let mut q = c.lock_queue()?;
                // The queue is left idle whether or not every copy was
                // appended: the caller's arrays are valid for this call only.
                let appended = (|| {
                    sent += s.upload(q.0, r, r.ids, 0, s.ids)?;
                    sent += s.upload(q.0, r, r.mask, 1, s.mask)?;
                    if !r.types.is_null() {
                        sent += s.upload(q.0, r, r.types, 2, s.types)?;
                    }
                    Ok(())
                })();
                let synced = c.sync(&mut q);
                appended?;
                synced?;
            }
            s.has_types = !r.types.is_null();
            s.batch = r.batch;
            s.seq = r.seq;
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
                let encoded = s.encode(q.0);
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
            st[TURBO_EMBED_STAGE_UPLOAD] = TURBO_STAGE_DEVICE;
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
