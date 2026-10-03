//! Contexts, buffers, models and embed sessions on a listed NPU.
//!
//! A context is a Level Zero context on the device and one in-order
//! immediate command list, used under the context's lock. A model is one
//! graph the driver's compiler built from the bundle's OpenVINO IR
//! (npu/ir.rs), initialized when it is loaded: that compile and
//! initialize are the load. A session owns host memory the device reads
//! and writes (zeMemAllocHost): one buffer per graph input, one for the
//! graph's hidden states, and the result vectors. A run binds the
//! session's buffers to the graph's arguments, executes it one frame of
//! fixed_batch rows at a time, and pools and normalizes on the host.
//! An INPUT_EMBEDDINGS graph does not take token ids. The host gathers
//! each token's word row and writes the attention bias the mask asks
//! for, and the graph owns everything after that gather. Argument
//! values live on the graph, so runs on one model are serialized under
//! the model's lock.

use std::ffi::{CStr, CString, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Mutex;

use super::native::{self, BlobError};
use super::ze::{self, GraphExt, Handle};
use super::{Device, Driver, ir};
use crate::backend::{
    TURBO_BERT_EMBEDDING_TENSORS, TURBO_FORMAT_OPENVINO_IR, TURBO_INPUT_EMBEDDINGS, TURBO_INPUT_TOKEN_IDS,
    TURBO_OUTPUT_HIDDEN_STATES, refuse, refuse_field, turbo_backend_embed_rows, turbo_backend_model, turbo_backend_run,
    turbo_backend_tensor,
};
use crate::status::{INVALID_ARGUMENT, INVALID_STATE, OUT_OF_MEMORY, PANIC, RUNTIME, UNSUPPORTED, UNSUPPORTED_OPTION};
use crate::{
    TURBO_DTYPE_F16, TURBO_DTYPE_F32, TURBO_EMBED_STAGE_DOWNLOAD, TURBO_EMBED_STAGE_ENCODE, TURBO_EMBED_STAGE_LOOKUP,
    TURBO_EMBED_STAGE_NORMALIZE, TURBO_EMBED_STAGE_POOL, TURBO_EMBED_STAGE_UPLOAD, TURBO_HANDLE_HOST_PTR,
    TURBO_NORMALIZE_L2, TURBO_PLACE_HOST, TURBO_POOLING_CLS, TURBO_POOLING_LAST, TURBO_POOLING_MEAN,
    TURBO_PRECISION_EXACT, TURBO_STAGE_DEVICE, TURBO_STAGE_HOST, TURBO_TASK_EMBED, turbo_buffer_desc, turbo_error,
    turbo_log_fn, turbo_native_handle, turbo_text,
};

// ---- Failures ----------------------------------------------------------------------

/// A refusal on its way to the caller's turbo_error.
pub(crate) struct Fail {
    pub code: i32,
    pub field: u32,
    pub message: String,
}

pub(crate) type Res<T> = Result<T, Fail>;

pub(crate) fn fail(code: i32, message: impl Into<String>) -> Fail {
    Fail { code, field: 0, message: message.into() }
}

fn fail_field(code: i32, field: u32, message: impl Into<String>) -> Fail {
    Fail { code, field, message: message.into() }
}

/// How long a command list may be waited on. The wait holds the
/// context lock, so a device that never finishes must come back as an
/// error instead of blocking every model on the context.
const DEVICE_WAIT_NS: u64 = 30_000_000_000;

fn wait_for_device(api: &ze::Api, list: Handle, ordinal: u32, what: &str) -> Res<()> {
    let rc = unsafe { (api.command_list_host_synchronize)(list, DEVICE_WAIT_NS) };
    if rc == 0 {
        return Ok(());
    }
    Err(fail(
        RUNTIME,
        format!(
            "npu device {ordinal}: the device did not answer {what} within 30s (zeCommandListHostSynchronize 0x{rc:08x})"
        ),
    ))
}

/// A Level Zero status as a refusal: running out of memory is
/// OUT_OF_MEMORY, anything else RUNTIME with the call and its code.
pub(crate) fn ze_res(what: &str, rc: ze::Status) -> Res<()> {
    match rc {
        0 => Ok(()),
        ze::RESULT_ERROR_OUT_OF_DEVICE_MEMORY | ze::RESULT_ERROR_OUT_OF_HOST_MEMORY => {
            Err(fail(OUT_OF_MEMORY, format!("npu: {what}: out of memory (0x{rc:08x})")))
        }
        _ => Err(fail(RUNTIME, format!("npu: {what} failed with 0x{rc:08x}"))),
    }
}

/// Runs a table entry's body, turning a refusal into the caller's
/// turbo_error and a panic into TURBO_E_PANIC: nothing unwinds across the
/// table.
///
/// # Safety
/// `err` is NULL or valid for the call.
pub(crate) unsafe fn guarded(err: *mut turbo_error, f: impl FnOnce() -> Res<()>) -> i32 {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => unsafe { refuse_field(err, e.code, e.field, &e.message) },
        Err(_) => unsafe { refuse(err, PANIC, "a panic inside the npu backend") },
    }
}

/// A release entry's body: it returns nothing, and a panic stops here.
fn quietly(f: impl FnOnce()) {
    let _ = catch_unwind(AssertUnwindSafe(f));
}

// ---- Contexts ------------------------------------------------------------------------

pub(crate) struct Context {
    api: &'static ze::Api,
    ext: GraphExt,
    device: Handle,
    handle: Handle,
    /// The in-order immediate command list, and the lock every use of it
    /// takes.
    list: Mutex<Shared>,
    compiler: ze::CompilerVersion,
    /// ze_graph_format_t bits and the highest OpenVINO opset the
    /// device's compiler takes, from the listing probe.
    formats_supported: u32,
    max_opset: u32,
    ordinal: u32,
    log: turbo_log_fn,
    log_user_data: *mut c_void,
}

/// A handle the driver lets any thread use.
#[derive(Clone, Copy)]
struct Shared(Handle);

unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

// The handles are the driver's, for any thread; the log function is the
// caller's, which turbo.h says may be called from any thread.
unsafe impl Send for Context {}
unsafe impl Sync for Context {}

const LOG_DEBUG: u32 = 3;

impl Context {
    fn say(&self, level: u32, message: &str) {
        if let Some(f) = self.log {
            let t = turbo_text { ptr: message.as_ptr() as *const _, len: message.len() as u64 };
            unsafe { f(self.log_user_data, level, t) };
        }
    }

    /// `bytes` of host memory the device reads directly.
    fn alloc_host(&self, bytes: usize, what: &str) -> Res<*mut c_void> {
        let desc =
            ze::HostMemAllocDesc { stype: ze::STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, p_next: std::ptr::null(), flags: 0 };
        let mut ptr = std::ptr::null_mut();
        let rc = unsafe { (self.api.mem_alloc_host)(self.handle, &desc, bytes.max(1), 64, &mut ptr) };
        ze_res(&format!("{bytes} bytes of host memory for {what}"), rc)?;
        Ok(ptr)
    }

    /// The immediate command list, held under its lock for the whole of
    /// `f`: an append and the synchronization that waits for it are one
    /// critical section, and the raw handle never leaves it.
    fn with_list<T>(&self, f: impl FnOnce(Handle) -> T) -> T {
        let guard = self.list.lock().unwrap_or_else(|p| p.into_inner());
        f(guard.0)
    }

    /// Runs the graph once over whatever its arguments point at, and
    /// returns when the device is done.
    fn execute(&self, graph: Handle) -> Res<()> {
        let execute = self
            .ext
            .append_graph_execute()
            .ok_or_else(|| fail(UNSUPPORTED, "npu: the driver's graph extension has no pfnAppendGraphExecute"))?;
        self.with_list(|list| {
            let null = std::ptr::null_mut();
            ze_res("pfnAppendGraphExecute", unsafe { execute(list, graph, null, null, 0, std::ptr::null_mut()) })?;
            wait_for_device(self.api, list, self.ordinal, "pfnAppendGraphExecute")
        })
    }
}

/// context_create: a Level Zero context and an immediate command list on
/// the listed device.
pub(crate) fn create(
    d: &'static Driver,
    dev: &'static Device,
    ordinal: u32,
    log: turbo_log_fn,
    log_user_data: *mut c_void,
    out: *mut *mut c_void,
) -> Res<()> {
    let desc = ze::ContextDesc { stype: ze::STRUCTURE_TYPE_CONTEXT_DESC, p_next: std::ptr::null(), flags: 0 };
    let mut handle = std::ptr::null_mut();
    ze_res("zeContextCreate", unsafe { (d.api.context_create)(dev.driver, &desc, &mut handle) })?;
    let qdesc = ze::CommandQueueDesc {
        stype: ze::STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
        p_next: std::ptr::null(),
        ordinal: 0,
        index: 0,
        flags: ze::COMMAND_QUEUE_FLAG_IN_ORDER,
        mode: ze::COMMAND_QUEUE_MODE_ASYNCHRONOUS,
        priority: ze::COMMAND_QUEUE_PRIORITY_NORMAL,
    };
    let mut list = std::ptr::null_mut();
    let rc = unsafe { (d.api.command_list_create_immediate)(handle, dev.handle, &qdesc, &mut list) };
    if let Err(e) = ze_res("zeCommandListCreateImmediate", rc) {
        unsafe { (d.api.context_destroy)(handle) };
        return Err(e);
    }
    let ctx = Box::new(Context {
        api: &d.api,
        ext: dev.ext,
        device: dev.handle,
        handle,
        list: Mutex::new(Shared(list)),
        compiler: dev.compiler,
        formats_supported: dev.formats_supported,
        max_opset: dev.max_opset,
        ordinal,
        log,
        log_user_data,
    });
    unsafe { *out = Box::into_raw(ctx) as *mut c_void };
    Ok(())
}

/// # Safety
/// `ctx` is a context this backend created, released once.
pub(crate) unsafe extern "C" fn context_release(ctx: *mut c_void) {
    quietly(|| {
        let ctx = unsafe { Box::from_raw(ctx as *mut Context) };
        // The box is the last owner: the mutex is consumed, not locked.
        let list = ctx.list.into_inner().unwrap_or_else(|p| p.into_inner()).0;
        unsafe {
            (ctx.api.command_list_destroy)(list);
            (ctx.api.context_destroy)(ctx.handle);
        }
    });
}

// ---- Buffers ---------------------------------------------------------------------------

/// Host memory the device reads directly, 64-byte aligned: the only
/// placement this backend has. The session's result vectors are one too,
/// so buffer_export serves them alike.
pub(crate) struct Buffer {
    /// zeMemFree for a device allocation. None when `ptr` is memory the
    /// caller owns. A test buffer is not a Level Zero allocation, and
    /// Drop does not call the driver for it.
    free: Option<(&'static ze::Api, Handle)>,
    ptr: *mut c_void,
    bytes: u64,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    fn new(ctx: &Context, bytes: u64, what: &str) -> Res<Buffer> {
        let ptr = ctx.alloc_host(bytes as usize, what)?;
        Ok(Buffer { free: Some((ctx.api, ctx.handle)), ptr, bytes })
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if let Some((api, context)) = self.free {
            unsafe { (api.mem_free)(context, self.ptr) };
        }
    }
}

/// # Safety
/// As turbo_backend.h says for buffer_alloc.
pub(crate) unsafe extern "C" fn buffer_alloc(
    ctx: *mut c_void,
    desc: *const turbo_buffer_desc,
    out: *mut *mut c_void,
    host: *mut *mut c_void,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        guarded(err, || {
            let ctx = &*(ctx as *const Context);
            let desc = &*desc;
            if desc.placement != TURBO_PLACE_HOST {
                return Err(fail(UNSUPPORTED, "npu: TURBO_PLACE_HOST only: the device reads host memory"));
            }
            let b = Box::new(Buffer::new(ctx, desc.bytes, "a buffer")?);
            *host = b.ptr;
            *out = Box::into_raw(b) as *mut c_void;
            Ok(())
        })
    }
}

/// # Safety
/// `buf` is a buffer this backend allocated, released once.
pub(crate) unsafe extern "C" fn buffer_release(buf: *mut c_void) {
    quietly(|| drop(unsafe { Box::from_raw(buf as *mut Buffer) }));
}

/// # Safety
/// As turbo_backend.h says for buffer_export.
pub(crate) unsafe extern "C" fn buffer_export(
    buf: *mut c_void,
    kind: u32,
    out: *mut turbo_native_handle,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        guarded(err, || {
            let b = &*(buf as *const Buffer);
            if kind != TURBO_HANDLE_HOST_PTR {
                return Err(fail(UNSUPPORTED, format!("npu: a buffer exports as TURBO_HANDLE_HOST_PTR, not {kind}")));
            }
            let out = &mut *out;
            out.kind = TURBO_HANDLE_HOST_PTR;
            out.handle = b.ptr as u64;
            out.aux = 0;
            out.offset = 0;
            Ok(())
        })
    }
}

// ---- Models ----------------------------------------------------------------------------

/// One argument of the compiled graph.
struct Arg {
    index: u32,
    name: String,
    precision: u32,
    /// Bytes of one element in the argument's device precision.
    elem: usize,
}

/// One compiled argument, with the dims, rank and device layout the
/// driver reported. The layout is what the host buffer must be when
/// `pfnSetArgumentValue` is called.
struct Compiled {
    arg: Arg,
    dims: [u32; 5],
    rank: u32,
    layout: u32,
}

/// What the compiled graph takes, after the boundary check.
enum GraphInputs {
    /// Token ids, the mask, and token types where the graph takes them.
    Tokens { ids: Arg, mask: Arg, types: Option<Arg> },
    /// The Hailo-style cut: gathered word rows and an attention bias.
    /// `table` is the core's F32 `word_embeddings`, alive until model_release.
    Embeddings { rows: Arg, bias: Arg, heads: u32, table: *const f32, vocab: u32 },
}

pub(crate) struct Model {
    ctx: *const Context,
    graph: Handle,
    /// Argument values live on the graph: one run at a time per model.
    run: Mutex<()>,
    inputs: GraphInputs,
    output: Arg,
    /// The shapes compiled in: a frame of rows, each of seq tokens.
    frame_batch: u32,
    seq: u32,
    hidden: u32,
    /// TURBO_DTYPE_* a session computes in.
    pub compute_dtype: u32,
    /// `NGRAPH_LITE` or `NATIVE`: what this graph was initialized as.
    pub graph_format: &'static str,
}

unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {
    fn ctx(&self) -> &Context {
        unsafe { &*self.ctx }
    }
}

/// The packed device layout the build flags name for a rank: NC for a
/// rank-2 token frame, CHW for rank 3 (word rows, or the hidden states),
/// NCHW for a rank-4 attention bias.
fn packed_layout(rank: u32) -> Option<u32> {
    match rank {
        2 => Some(ze::GRAPH_ARGUMENT_LAYOUT_NC),
        3 => Some(ze::GRAPH_ARGUMENT_LAYOUT_CHW),
        4 => Some(ze::GRAPH_ARGUMENT_LAYOUT_NCHW),
        _ => None,
    }
}

fn layout_label(layout: u32) -> String {
    match layout {
        ze::GRAPH_ARGUMENT_LAYOUT_NC => "NC".to_string(),
        ze::GRAPH_ARGUMENT_LAYOUT_NCHW => "NCHW".to_string(),
        ze::GRAPH_ARGUMENT_LAYOUT_CHW => "CHW".to_string(),
        0 => "ANY".to_string(),
        0xC8 => "BLOCKED".to_string(),
        n => format!("0x{n:02x}"),
    }
}

/// Refuse unless `layout` is the packed layout the build flags asked for
/// at this rank. BLOCKED and ANY are not that layout: the host writes
/// packed rows.
fn expect_packed_layout(name: &str, rank: u32, layout: u32) -> Res<()> {
    let Some(want) = packed_layout(rank) else {
        return Err(fail(
            RUNTIME,
            format!("npu: {name:?} has rank {rank}; the build flags name a packed layout for rank 2, 3 or 4"),
        ));
    };
    if layout != want {
        return Err(fail(
            RUNTIME,
            format!(
                "npu: {name:?} has device layout {}; the build flags asked for {}",
                layout_label(layout),
                layout_label(want)
            ),
        ));
    }
    Ok(())
}

/// The leading two dims are the token frame `[batch, seq]`.
fn expect_frame(name: &str, dims: &[u32], batch: u32, seq: u32) -> Res<()> {
    let (d0, d1) = (dims.first().copied().unwrap_or(0), dims.get(1).copied().unwrap_or(0));
    if d0 != batch || d1 != seq {
        return Err(fail(RUNTIME, format!("npu: {name:?} is [{d0}, {d1}, ...]; the token frame is [{batch}, {seq}]")));
    }
    Ok(())
}

/// The TURBO_DTYPE_* the compiled output precision is. FP16 is
/// TURBO_DTYPE_F16 and FP32 is TURBO_DTYPE_F32. A manifest dtype of 0
/// takes that; any other value must be it. The session reports this
/// dtype, not a manifest value that disagrees.
fn session_dtype(manifest: u32, precision: u32) -> Res<u32> {
    let dtype = match precision {
        ze::GRAPH_ARGUMENT_PRECISION_FP16 => TURBO_DTYPE_F16,
        ze::GRAPH_ARGUMENT_PRECISION_FP32 => TURBO_DTYPE_F32,
        p => {
            return Err(fail(
                RUNTIME,
                format!("npu: the graph computes in argument precision 0x{p:02x}; FP16 and FP32 are a session dtype"),
            ));
        }
    };
    if manifest != 0 && manifest != dtype {
        return Err(fail(
            RUNTIME,
            format!(
                "npu: the graph computes in {}; the manifest's compute_dtype is {}",
                dtype_label(dtype),
                dtype_label(manifest)
            ),
        ));
    }
    Ok(dtype)
}

fn dtype_label(d: u32) -> String {
    match d {
        TURBO_DTYPE_F16 => "DTYPE_F16".to_string(),
        TURBO_DTYPE_F32 => "DTYPE_F32".to_string(),
        n => n.to_string(),
    }
}

/// The element size of a graph argument precision this backend moves, or
/// a refusal naming it.
fn elem_bytes(what: &str, precision: u32) -> Res<usize> {
    Ok(match precision {
        ze::GRAPH_ARGUMENT_PRECISION_INT64 | ze::GRAPH_ARGUMENT_PRECISION_UINT64 => 8,
        ze::GRAPH_ARGUMENT_PRECISION_FP32
        | ze::GRAPH_ARGUMENT_PRECISION_INT32
        | ze::GRAPH_ARGUMENT_PRECISION_UINT32 => 4,
        ze::GRAPH_ARGUMENT_PRECISION_FP16 | ze::GRAPH_ARGUMENT_PRECISION_BF16 => 2,
        p => {
            return Err(fail(
                RUNTIME,
                format!("npu: {what} is argument precision 0x{p:02x}, which this backend does not move"),
            ));
        }
    })
}

/// The bias a dropped key gets. exp(-100) underflows to 0 in the softmax,
/// the value the Hailo cut is checked with (core/hailo/embed.cpp).
const MASKED: f32 = -100.0;

/// Why a non-zero token type is refused, or None when the graph takes the
/// type ids and will run them.
fn type_refusal(inputs: &GraphInputs) -> Option<&'static str> {
    match inputs {
        GraphInputs::Tokens { types: None, .. } => Some("the graph takes no token type input and computes type 0 only"),
        GraphInputs::Embeddings { .. } => Some("an INPUT_EMBEDDINGS graph computes token type 0 only"),
        GraphInputs::Tokens { types: Some(_), .. } => None,
    }
}

fn type_error(row: usize, at: usize, ty: i32, why: &str) -> Fail {
    fail(UNSUPPORTED_OPTION, format!("npu: token type {ty} in row {row} position {at}: {why}"))
}

/// F32 as F16, round to nearest even, for writing a gathered tensor the
/// compiler kept in FP16.
fn f32_to_half(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x007f_ffff;
    if exp == 0xff {
        let payload = if frac == 0 { 0 } else { 0x200 };
        return sign | 0x7c00 | payload;
    }
    let half_exp = exp - 127 + 15;
    if half_exp >= 31 {
        return sign | 0x7c00;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let frac = frac | 0x0080_0000;
        let shift = (1 - half_exp) as u32;
        let mut half_frac = frac >> (shift + 13);
        let round = (frac >> (shift + 12)) & 1;
        let sticky = (frac & ((1 << (shift + 12)) - 1)) != 0;
        if round == 1 && (sticky || (half_frac & 1) == 1) {
            half_frac += 1;
        }
        return sign | (half_frac as u16);
    }
    let mut out = ((half_exp as u16) << 10) | ((frac >> 13) as u16);
    let round = (frac >> 12) & 1;
    let sticky = (frac & 0xfff) != 0;
    if round == 1 && (sticky || (out & 1) == 1) {
        out += 1;
    }
    sign | out
}

/// `src` into `dst` at the argument's float precision.
///
/// # Safety
/// `dst` holds `src.len()` elements of that precision.
unsafe fn write_floats(dst: *mut c_void, precision: u32, src: &[f32]) -> Res<()> {
    unsafe {
        match precision {
            ze::GRAPH_ARGUMENT_PRECISION_FP32 => {
                (dst as *mut f32).copy_from_nonoverlapping(src.as_ptr(), src.len());
            }
            ze::GRAPH_ARGUMENT_PRECISION_FP16 => {
                let d = dst as *mut u16;
                for (i, &v) in src.iter().enumerate() {
                    d.add(i).write(f32_to_half(v));
                }
            }
            p => {
                return Err(fail(
                    RUNTIME,
                    format!("npu: a gathered tensor is argument precision 0x{p:02x}; FP32 and FP16 are written"),
                ));
            }
        }
    }
    Ok(())
}

/// One written row, as the Hailo backend gathers it.
struct Gather<'a> {
    table: &'a [f32],
    vocab: usize,
    hidden: usize,
    heads: usize,
    /// The compiled frame length.
    model_seq: usize,
    /// Tokens `embed_write` copied. Past this the frame repeats the first id.
    written: usize,
    ids: &'a [i32],
    mask: &'a [i32],
}

/// `rows_out` is `[model_seq, hidden]`. `bias_out` is `[heads, model_seq, model_seq]`,
/// and every query sees the same keys: 0 where the mask keeps a key, MASKED
/// where it drops one.
fn gather_row(g: &Gather<'_>, rows_out: &mut [f32], bias_out: &mut [f32]) -> Res<()> {
    if g.written == 0 || g.written > g.model_seq || g.ids.len() < g.written || g.mask.len() < g.written {
        return Err(fail(INVALID_ARGUMENT, "npu: a gathered row is longer than the frame or empty"));
    }
    if rows_out.len() != g.model_seq * g.hidden || bias_out.len() != g.heads * g.model_seq * g.model_seq {
        return Err(fail(INVALID_ARGUMENT, "npu: the gather buffers are not the frame"));
    }
    for t in 0..g.model_seq {
        let id = if t < g.written { g.ids[t] } else { g.ids[0] };
        if id < 0 || id as usize >= g.vocab {
            return Err(fail(
                INVALID_ARGUMENT,
                format!("npu: token id {id} is outside the word table of {} rows", g.vocab),
            ));
        }
        let src = id as usize * g.hidden;
        let dst = t * g.hidden;
        rows_out[dst..dst + g.hidden].copy_from_slice(&g.table[src..src + g.hidden]);
    }
    let plane = g.model_seq * g.model_seq;
    for k in 0..g.model_seq {
        let v = if k < g.written && g.mask[k] == 1 { 0.0 } else { MASKED };
        for head in 0..g.heads {
            let base = head * plane;
            for q in 0..g.model_seq {
                bias_out[base + q * g.model_seq + k] = v;
            }
        }
    }
    Ok(())
}

/// The host_weights word table an INPUT_EMBEDDINGS graph gathers from.
/// The core hands it first, TURBO_BERT_WORD_EMBEDDINGS, F32, [vocab, hidden].
fn word_table(desc: &turbo_backend_model) -> Res<(*const f32, u32)> {
    if desc.tensors.is_null() || desc.tensor_count == 0 {
        return Err(fail(
            UNSUPPORTED,
            "npu: INPUT_EMBEDDINGS needs the host_weights word_embeddings table, and none was handed over",
        ));
    }
    if desc.tensor_count != TURBO_BERT_EMBEDDING_TENSORS {
        return Err(fail(
            UNSUPPORTED,
            format!(
                "npu: INPUT_EMBEDDINGS needs the {TURBO_BERT_EMBEDDING_TENSORS} host embedding tensors; {} were handed \
                 over",
                desc.tensor_count
            ),
        ));
    }
    if desc.dtype != TURBO_DTYPE_F32 {
        return Err(fail(
            UNSUPPORTED,
            format!("npu: the host gather reads F32 word rows; the weights are {}", dtype_label(desc.dtype)),
        ));
    }
    let t = unsafe { &*desc.tensors };
    let name = tensor_name(t);
    if t.dtype != TURBO_DTYPE_F32 {
        return Err(fail(
            UNSUPPORTED,
            format!("npu: {name} is {}; the host gather reads F32 word rows", dtype_label(t.dtype)),
        ));
    }
    if t.ndim != 2 || t.shape[0] != desc.vocab_size as u64 || t.shape[1] != desc.hidden as u64 {
        return Err(fail(
            RUNTIME,
            format!(
                "npu: {name} is {:?} of rank {}; word_embeddings is [{}, {}]",
                &t.shape[..t.ndim as usize],
                t.ndim,
                desc.vocab_size,
                desc.hidden
            ),
        ));
    }
    let n = (desc.vocab_size as u64).saturating_mul(desc.hidden as u64);
    if n == 0 || t.bytes != n * 4 || t.data.is_null() {
        return Err(fail(
            RUNTIME,
            format!("npu: {name} is {} bytes; [{}, {}] F32 is {} bytes", t.bytes, desc.vocab_size, desc.hidden, n * 4),
        ));
    }
    if !(t.data as usize).is_multiple_of(4) {
        return Err(fail(INVALID_ARGUMENT, format!("npu: {name} is not aligned to 4 bytes")));
    }
    Ok((t.data as *const f32, desc.vocab_size))
}

fn tensor_name(t: &turbo_backend_tensor) -> String {
    if t.name.is_null() {
        return "<unnamed>".to_string();
    }
    unsafe { CStr::from_ptr(t.name).to_string_lossy().into_owned() }
}

fn expect_float(name: &str, precision: u32) -> Res<()> {
    if !matches!(precision, ze::GRAPH_ARGUMENT_PRECISION_FP32 | ze::GRAPH_ARGUMENT_PRECISION_FP16) {
        return Err(fail(
            RUNTIME,
            format!("npu: {name:?} is argument precision 0x{precision:02x}; FP32 and FP16 are written"),
        ));
    }
    Ok(())
}

/// The compiled batch and seq against the manifest. A dynamic graph is
/// refused before a zero manifest shape, so the message stays about the graph.
fn expect_manifest_frame(frame_batch: u32, seq: u32, hidden: u32, desc: &turbo_backend_model) -> Res<()> {
    if hidden != desc.hidden {
        return Err(fail(RUNTIME, format!("npu: the graph's hidden is {hidden}; the manifest says {}", desc.hidden)));
    }
    if frame_batch == 0 || seq == 0 {
        return Err(fail(
            RUNTIME,
            format!("npu: the graph compiled to [{frame_batch}, {seq}]; a static shape is needed"),
        ));
    }
    if desc.fixed_seq != seq {
        return Err(fail(
            RUNTIME,
            format!("npu: the graph's seq is {seq}; the manifest's fixed_seq is {}", desc.fixed_seq),
        ));
    }
    if desc.fixed_batch != frame_batch {
        return Err(fail(
            RUNTIME,
            format!("npu: the graph's frame is {frame_batch} rows; the manifest's fixed_batch is {}", desc.fixed_batch),
        ));
    }
    Ok(())
}

fn take_hidden(mut outputs: Vec<Compiled>) -> Res<Compiled> {
    if outputs.is_empty() {
        return Err(fail(RUNTIME, "npu: the graph returns nothing"));
    }
    let names: Vec<String> = outputs.iter().map(|c| c.arg.name.clone()).collect();
    match outputs.len() {
        1 => Ok(outputs.remove(0)),
        _ => {
            let at = outputs.iter().position(|c| c.arg.name == "last_hidden_state").ok_or_else(|| {
                fail(RUNTIME, format!("npu: the graph returns {names:?} and none is last_hidden_state"))
            })?;
            Ok(outputs.remove(at))
        }
    }
}

/// The hidden states are FP32 or FP16 `[batch, seq, hidden]`, layout CHW.
fn expect_hidden(output: &Compiled, frame_batch: u32, seq: u32) -> Res<u32> {
    if output.rank != 3 {
        return Err(fail(
            RUNTIME,
            format!(
                "npu: {:?} has rank {}; OUTPUT_HIDDEN_STATES is [batch, seq, hidden]",
                output.arg.name, output.rank
            ),
        ));
    }
    expect_frame(&output.arg.name, &output.dims, frame_batch, seq)?;
    expect_packed_layout(&output.arg.name, output.rank, output.layout)?;
    if !matches!(output.arg.precision, ze::GRAPH_ARGUMENT_PRECISION_FP32 | ze::GRAPH_ARGUMENT_PRECISION_FP16) {
        return Err(fail(
            RUNTIME,
            format!(
                "npu: {:?} is argument precision 0x{:02x}; FP32 and FP16 are read back",
                output.arg.name, output.arg.precision
            ),
        ));
    }
    Ok(output.dims[2])
}

fn arg_names(v: &[Compiled]) -> Vec<String> {
    v.iter().map(|c| c.arg.name.clone()).collect()
}

/// INPUT_TOKEN_IDS: ids, mask, and token types where the graph takes them,
/// each I64 or I32 of one `[batch, seq]`, layout NC.
fn accept_tokens(
    inputs: Vec<Compiled>,
    output: Compiled,
    desc: &turbo_backend_model,
) -> Res<(GraphInputs, Arg, u32, u32, u32)> {
    if !(2..=3).contains(&inputs.len()) {
        return Err(fail(
            RUNTIME,
            format!(
                "npu: the graph takes {:?}; an encoder takes input_ids, attention_mask and optionally token_type_ids",
                arg_names(&inputs)
            ),
        ));
    }
    let (mut ids, mut mask, mut types) = (None, None, None);
    for entry in inputs {
        let slot = match entry.arg.name.as_str() {
            "input_ids" => &mut ids,
            "attention_mask" => &mut mask,
            "token_type_ids" => &mut types,
            other => {
                return Err(fail(
                    RUNTIME,
                    format!(
                        "npu: the graph takes an input named {other:?}, which this backend does not know; it runs \
                         graphs whose inputs are input_ids, attention_mask and optionally token_type_ids"
                    ),
                ));
            }
        };
        if slot.replace(entry).is_some() {
            return Err(fail(RUNTIME, "npu: the graph takes two inputs of one name"));
        }
    }
    let (Some(ids), Some(mask)) = (ids, mask) else {
        return Err(fail(RUNTIME, "npu: the graph takes no input_ids or no attention_mask"));
    };
    if ids.rank != 2 {
        return Err(fail(
            RUNTIME,
            format!("npu: {:?} has rank {}; token ids are [batch, seq]", ids.arg.name, ids.rank),
        ));
    }
    expect_packed_layout(&ids.arg.name, ids.rank, ids.layout)?;
    let (frame_batch, seq) = (ids.dims[0], ids.dims[1]);
    if mask.rank != 2 {
        return Err(fail(
            RUNTIME,
            format!("npu: {:?} has rank {}; the mask is [batch, seq]", mask.arg.name, mask.rank),
        ));
    }
    expect_frame(&mask.arg.name, &mask.dims, frame_batch, seq)?;
    expect_packed_layout(&mask.arg.name, mask.rank, mask.layout)?;
    let types = match types {
        Some(t) => {
            if t.rank != 2 {
                return Err(fail(
                    RUNTIME,
                    format!("npu: {:?} has rank {}; token types are [batch, seq]", t.arg.name, t.rank),
                ));
            }
            expect_frame(&t.arg.name, &t.dims, frame_batch, seq)?;
            expect_packed_layout(&t.arg.name, t.rank, t.layout)?;
            Some(t.arg)
        }
        None => None,
    };
    let hidden = expect_hidden(&output, frame_batch, seq)?;
    let ids = ids.arg;
    let mask = mask.arg;
    for a in [&ids, &mask].into_iter().chain(types.iter()) {
        if !matches!(
            a.precision,
            ze::GRAPH_ARGUMENT_PRECISION_INT64
                | ze::GRAPH_ARGUMENT_PRECISION_INT32
                | ze::GRAPH_ARGUMENT_PRECISION_UINT64
                | ze::GRAPH_ARGUMENT_PRECISION_UINT32
        ) {
            return Err(fail(
                RUNTIME,
                format!(
                    "npu: {:?} is argument precision 0x{:02x}; rows are written as I64 or I32",
                    a.name, a.precision
                ),
            ));
        }
    }
    expect_manifest_frame(frame_batch, seq, hidden, desc)?;
    Ok((GraphInputs::Tokens { ids, mask, types }, output.arg, frame_batch, seq, hidden))
}

/// INPUT_EMBEDDINGS: `word_rows` `[batch, seq, hidden]` and `attn_bias`
/// `[batch, heads, seq, seq]`, the cut hef_compile.py makes, kept at the
/// batch the IR was reshaped to. Layouts are the packed ones the build
/// flags name. The host writes FP32 or FP16, whichever the compiler kept.
fn accept_embeddings(
    inputs: Vec<Compiled>,
    output: Compiled,
    desc: &turbo_backend_model,
) -> Res<(Arg, Arg, Arg, u32, u32, u32)> {
    if inputs.len() != 2 {
        return Err(fail(
            RUNTIME,
            format!(
                "npu: the graph takes {:?}; an INPUT_EMBEDDINGS graph takes word_rows and attn_bias",
                arg_names(&inputs)
            ),
        ));
    }
    let (mut rows, mut bias) = (None, None);
    for entry in inputs {
        let slot = match entry.arg.name.as_str() {
            "word_rows" => &mut rows,
            "attn_bias" => &mut bias,
            other => {
                return Err(fail(
                    RUNTIME,
                    format!(
                        "npu: the graph takes an input named {other:?}; an INPUT_EMBEDDINGS graph takes word_rows and \
                         attn_bias"
                    ),
                ));
            }
        };
        if slot.replace(entry).is_some() {
            return Err(fail(RUNTIME, "npu: the graph takes two inputs of one name"));
        }
    }
    let (Some(rows), Some(bias)) = (rows, bias) else {
        return Err(fail(RUNTIME, "npu: the graph takes no word_rows or no attn_bias"));
    };
    if rows.rank != 3 {
        return Err(fail(
            RUNTIME,
            format!("npu: {:?} has rank {}; word rows are [batch, seq, hidden]", rows.arg.name, rows.rank),
        ));
    }
    expect_packed_layout(&rows.arg.name, rows.rank, rows.layout)?;
    let (frame_batch, seq, hidden) = (rows.dims[0], rows.dims[1], rows.dims[2]);
    if bias.rank != 4 {
        return Err(fail(
            RUNTIME,
            format!("npu: {:?} has rank {}; the attention bias is [batch, heads, seq, seq]", bias.arg.name, bias.rank),
        ));
    }
    expect_packed_layout(&bias.arg.name, bias.rank, bias.layout)?;
    if bias.dims[0] != frame_batch || bias.dims[2] != seq || bias.dims[3] != seq || bias.dims[1] != desc.heads {
        return Err(fail(
            RUNTIME,
            format!(
                "npu: {:?} is [{}, {}, {}, {}]; the attention bias is [{frame_batch}, {}, {seq}, {seq}]",
                bias.arg.name, bias.dims[0], bias.dims[1], bias.dims[2], bias.dims[3], desc.heads
            ),
        ));
    }
    expect_float(&rows.arg.name, rows.arg.precision)?;
    expect_float(&bias.arg.name, bias.arg.precision)?;
    let out_hidden = expect_hidden(&output, frame_batch, seq)?;
    if out_hidden != hidden {
        return Err(fail(
            RUNTIME,
            format!("npu: word_rows hidden is {hidden}; {:?} hidden is {out_hidden}", output.arg.name),
        ));
    }
    expect_manifest_frame(frame_batch, seq, hidden, desc)?;
    Ok((rows.arg, bias.arg, output.arg, frame_batch, seq, hidden))
}

/// The build log's text, read and destroyed. Empty when there is none.
fn build_log(ext: &GraphExt, log: Handle) -> String {
    if log.is_null() {
        return String::new();
    }
    let mut text = String::new();
    if let Some(get) = ext.build_log_get_string2() {
        let mut n = 0u32;
        if unsafe { get(log, &mut n, std::ptr::null_mut()) } == 0 && n > 0 {
            let mut buf = vec![0 as c_char; n as usize];
            if unsafe { get(log, &mut n, buf.as_mut_ptr()) } == 0 {
                text = crate::backend::cstr(&buf);
            }
        }
    }
    if let Some(destroy) = ext.build_log_destroy() {
        unsafe { destroy(log) };
    }
    text
}

/// model_load: the driver's compiler builds the graph from the IR's xml
/// and weights, and the graph is initialized, so a session allocates and
/// compiles nothing. A compile failure is RUNTIME with the compiler's
/// own log.
///
/// # Safety
/// As turbo_backend.h says for model_load.
pub(crate) unsafe extern "C" fn model_load(
    ctx: *mut c_void,
    desc: *const turbo_backend_model,
    out: *mut *mut c_void,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        guarded(err, || {
            let ctx = &*(ctx as *const Context);
            let desc = &*desc;
            load(ctx, desc).map(|m| *out = Box::into_raw(Box::new(m)) as *mut c_void)
        })
    }
}

fn load(ctx: &Context, desc: &turbo_backend_model) -> Res<Model> {
    if desc.format != TURBO_FORMAT_OPENVINO_IR {
        return Err(fail(UNSUPPORTED, format!("npu: model_load takes FORMAT_OPENVINO_IR, not format {}", desc.format)));
    }
    // artifact2, the IR's weights, was appended to the struct: a core
    // from before it cannot hand an IR's two files.
    let covers =
        desc.struct_size as usize >= std::mem::offset_of!(turbo_backend_model, artifact2_bytes) + size_of::<u64>();
    if !covers {
        return Err(fail(UNSUPPORTED, "npu: the core's turbo_backend_model ends before artifact2, the IR's weights"));
    }
    let embeddings = match desc.graph_input {
        TURBO_INPUT_TOKEN_IDS => None,
        TURBO_INPUT_EMBEDDINGS => Some(word_table(desc)?),
        other => {
            return Err(fail(
                UNSUPPORTED,
                format!("npu: graph_input {other} is not INPUT_TOKEN_IDS or INPUT_EMBEDDINGS"),
            ));
        }
    };
    if desc.graph_output != TURBO_OUTPUT_HIDDEN_STATES {
        return Err(fail(
            UNSUPPORTED,
            "npu: only OUTPUT_HIDDEN_STATES artifacts run; this backend pools and normalizes on the host",
        ));
    }
    if desc.artifact.is_null() || desc.artifact2.is_null() {
        return Err(fail(INVALID_ARGUMENT, "npu: an OpenVINO IR is two blocks of bytes, the xml and its weights"));
    }
    let (maj, min) = (ctx.compiler.major, ctx.compiler.minor);
    if (maj, min) < (5, 9) {
        return Err(fail(
            UNSUPPORTED,
            format!("npu: the driver's compiler is {maj}.{min}; this IR path needs 5.9 or later"),
        ));
    }
    if ctx.ext.create2().is_none() {
        return Err(fail(
            UNSUPPORTED,
            format!(
                "npu: the driver's graph extension is {}.{}; compiling an IR needs 1.5 or later",
                ctx.ext.version >> 16,
                ctx.ext.version & 0xffff
            ),
        ));
    }

    // What the device said it compiles, checked before any compile is
    // tried, so a refusal says why in words rather than a status code.
    if ctx.formats_supported & ze::GRAPH_FORMAT_NGRAPH_LITE == 0 {
        return Err(fail(
            UNSUPPORTED,
            "npu: the device's compiler does not take an OpenVINO IR (NGRAPH_LITE is not among its graph formats)",
        ));
    }
    let xml = unsafe { std::slice::from_raw_parts(desc.artifact as *const u8, desc.artifact_bytes as usize) };
    let bin = unsafe { std::slice::from_raw_parts(desc.artifact2 as *const u8, desc.artifact2_bytes as usize) };
    let io = ir::interface(xml).map_err(|e| fail(RUNTIME, e))?;
    if ctx.max_opset != 0 && io.max_opset > ctx.max_opset {
        return Err(fail(
            UNSUPPORTED,
            format!(
                "npu: the IR uses opset {}, and the device's compiler supports up to opset {}; update the NPU \
                 driver or export the IR for an older opset",
                io.max_opset, ctx.max_opset
            ),
        ));
    }
    let flags = ir::build_flags(&io).map_err(|e| fail(RUNTIME, e))?;
    let container = ir::container((maj, min), xml, bin);
    let cflags = CString::new(flags.clone()).map_err(|_| fail(RUNTIME, "npu: a NUL in the build flags"))?;
    let mut graph = create_graph(
        ctx,
        ze::GRAPH_FORMAT_NGRAPH_LITE,
        &container,
        cflags.as_ptr(),
        "the driver's compiler refused the IR",
    )?;
    ctx.say(LOG_DEBUG, &format!("npu device {}: the IR is compiled ({flags})", ctx.ordinal));

    // The graph that runs, when the device lists it, is the blob that
    // compile just produced. The NGRAPH_LITE graph is destroyed once
    // the copy exists. A refusal here does not keep that graph.
    if ctx.formats_supported & ze::GRAPH_FORMAT_NATIVE != 0 {
        graph = native_graph(ctx, graph)?;
    } else {
        ctx.say(
            LOG_DEBUG,
            &format!(
                "npu device {}: the graph is NGRAPH_LITE; ZE_GRAPH_FORMAT_NATIVE is not among the device's graph \
                 formats (0x{:x})",
                ctx.ordinal, ctx.formats_supported
            ),
        );
    }

    let model = describe(ctx, desc, graph, embeddings);
    if model.is_err()
        && let Some(destroy) = ctx.ext.destroy()
    {
        unsafe { destroy(graph) };
    }
    model
}

fn destroy_graph(ctx: &Context, graph: Handle) {
    if let Some(destroy) = ctx.ext.destroy() {
        unsafe { destroy(graph) };
    }
}

/// pfnCreate3 where the extension has it, so a refusal carries the
/// driver's log, else pfnCreate2. `refused` is the start of the
/// TURBO_E_RUNTIME message.
fn create_graph(ctx: &Context, format: u32, input: &[u8], build_flags: *const c_char, refused: &str) -> Res<Handle> {
    let gdesc = if format == ze::GRAPH_FORMAT_NATIVE {
        native::descriptor(input, build_flags)
    } else {
        ze::GraphDesc2 {
            stype: ze::STRUCTURE_TYPE_GRAPH_DESC_2,
            p_next: std::ptr::null(),
            format,
            input_size: input.len(),
            input: input.as_ptr(),
            build_flags,
            flags: 0,
        }
    };
    let mut graph = std::ptr::null_mut();
    match ctx.ext.create3() {
        Some(create3) => {
            let mut log = std::ptr::null_mut();
            let rc = unsafe { create3(ctx.handle, ctx.device, &gdesc, &mut graph, &mut log) };
            if rc != 0 {
                let text = build_log(&ctx.ext, log);
                return Err(fail(RUNTIME, format!("npu: {refused} with 0x{rc:08x}: {text}")));
            }
            build_log(&ctx.ext, log);
        }
        None => {
            let create2 = ctx.ext.create2().ok_or_else(|| {
                fail(UNSUPPORTED, "npu: the driver's graph extension predates 1.5; pfnCreate2 is needed")
            })?;
            ze_res("pfnGraphCreate2", unsafe { create2(ctx.handle, ctx.device, &gdesc, &mut graph) })?;
        }
    }
    Ok(graph)
}

/// The native graph for a compiled IR graph. `lite` is destroyed
/// whether the blob loads or the driver refuses it.
fn native_graph(ctx: &Context, lite: Handle) -> Res<Handle> {
    let blob = match native::copy(&ctx.ext, lite) {
        Ok(blob) => blob,
        Err(e) => {
            destroy_graph(ctx, lite);
            return Err(blob_fail(e));
        }
    };
    match create_graph(ctx, ze::GRAPH_FORMAT_NATIVE, &blob, c"".as_ptr(), "the driver refused the native blob") {
        Ok(native) => {
            destroy_graph(ctx, lite);
            ctx.say(
                LOG_DEBUG,
                &format!("npu device {}: the graph is the native blob, {} bytes", ctx.ordinal, blob.len()),
            );
            Ok(native)
        }
        Err(e) => {
            destroy_graph(ctx, lite);
            Err(e)
        }
    }
}

fn blob_fail(e: BlobError) -> Fail {
    match e {
        BlobError::Unsupported(m) => fail(UNSUPPORTED, m),
        BlobError::Runtime(m) => fail(RUNTIME, m),
    }
}

/// The compiled graph's arguments and shapes, checked against the
/// manifest, and the graph initialized: the weights' move to the device.
fn describe(
    ctx: &Context,
    desc: &turbo_backend_model,
    graph: Handle,
    embeddings: Option<(*const f32, u32)>,
) -> Res<Model> {
    let get_props = ctx.ext.get_properties2().ok_or_else(|| {
        fail(UNSUPPORTED, "npu: the driver's graph extension predates 1.8; pfnGetProperties2 is needed")
    })?;
    let get_arg = ctx.ext.get_argument_properties3().ok_or_else(|| {
        fail(UNSUPPORTED, "npu: the driver's graph extension predates 1.2; pfnGetArgumentProperties3 is needed")
    })?;
    let mut props = ze::GraphProperties2 { stype: ze::STRUCTURE_TYPE_GRAPH_PROPERTIES_2, ..Default::default() };
    ze_res("pfnGetProperties2", unsafe { get_props(graph, &mut props) })?;

    let mut inputs: Vec<Compiled> = Vec::new();
    let mut outputs: Vec<Compiled> = Vec::new();
    for i in 0..props.num_graph_args {
        let mut a = ze::GraphArgumentProperties3 {
            stype: ze::STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES_3,
            ..Default::default()
        };
        ze_res("pfnGetArgumentProperties3", unsafe { get_arg(graph, i, &mut a) })?;
        let name = ze::string(&a.name);
        let arg = Arg {
            index: i,
            precision: a.device_precision,
            elem: elem_bytes(&format!("argument {name:?}"), a.device_precision)?,
            name: name.clone(),
        };
        if a.dims_count == 0 {
            return Err(fail(
                RUNTIME,
                format!("npu: argument {name:?} reports dims_count 0; the rank is not guessed from the dimension list"),
            ));
        }
        let rank = a.dims_count;
        let compiled = Compiled { arg, dims: a.dims, rank, layout: a.device_layout };
        match a.kind {
            ze::GRAPH_ARGUMENT_TYPE_INPUT => inputs.push(compiled),
            _ => outputs.push(compiled),
        }
    }

    // Inputs are taken by name. A graph is never run on a guess about
    // which argument is which. One output is the hidden states whatever
    // its name; among several, only last_hidden_state is.
    let output = take_hidden(outputs)?;
    let (inputs, output, frame_batch, seq, hidden) = match desc.graph_input {
        TURBO_INPUT_TOKEN_IDS => accept_tokens(inputs, output, desc)?,
        TURBO_INPUT_EMBEDDINGS => {
            let (table, vocab) = embeddings
                .ok_or_else(|| fail(RUNTIME, "npu: INPUT_EMBEDDINGS reached the compiler with no word table"))?;
            let (rows, bias, output, frame_batch, seq, hidden) = accept_embeddings(inputs, output, desc)?;
            (GraphInputs::Embeddings { rows, bias, heads: desc.heads, table, vocab }, output, frame_batch, seq, hidden)
        }
        other => {
            return Err(fail(
                UNSUPPORTED,
                format!("npu: graph_input {other} is not INPUT_TOKEN_IDS or INPUT_EMBEDDINGS"),
            ));
        }
    };

    // Initialize now: the weights' move to the device is the load.
    match props.init_stage_required {
        ze::GRAPH_STAGE_INITIALIZE => {
            let init = ctx.ext.graph_initialize().ok_or_else(|| {
                fail(RUNTIME, "npu: the graph asks for pfnGraphInitialize and the driver's table has none")
            })?;
            ze_res("pfnGraphInitialize", unsafe { init(graph) })?;
        }
        ze::GRAPH_STAGE_COMMAND_LIST_INITIALIZE => {
            let init = ctx.ext.append_graph_initialize().ok_or_else(|| {
                fail(RUNTIME, "npu: the graph asks for pfnAppendGraphInitialize and the driver's table has none")
            })?;
            ctx.with_list(|list| {
                ze_res("pfnAppendGraphInitialize", unsafe {
                    init(list, graph, std::ptr::null_mut(), 0, std::ptr::null_mut())
                })?;
                wait_for_device(ctx.api, list, ctx.ordinal, "pfnAppendGraphInitialize")
            })?;
        }
        s => {
            return Err(fail(
                RUNTIME,
                format!("npu: the graph asks for init stage {s}, which this backend does not know"),
            ));
        }
    }

    let compute_dtype = session_dtype(desc.compute_dtype, output.precision)?;
    let graph_format = super::format_name(ctx.formats_supported);
    Ok(Model { ctx, graph, run: Mutex::new(()), inputs, output, frame_batch, seq, hidden, compute_dtype, graph_format })
}

/// # Safety
/// `model` is a model this backend loaded, released once.
pub(crate) unsafe extern "C" fn model_release(model: *mut c_void) {
    quietly(|| {
        let m = unsafe { Box::from_raw(model as *mut Model) };
        if let Some(destroy) = m.ctx().ext.destroy() {
            unsafe { destroy(m.graph) };
        }
    });
}

// ---- Sessions ---------------------------------------------------------------------------

/// What embed_write left for the next run.
struct State {
    batch: u32,
    seq: u32,
    pooling: u32,
    normalize: u32,
    output_dim: u32,
    written: bool,
}

pub(crate) struct Session {
    model: *const Model,
    max_seq: u32,
    /// One host buffer per graph input, a frame each, bound to the graph
    /// at every run.
    frames: Frames,
    /// The graph's hidden states, one frame.
    out_buf: Buffer,
    /// The run's vectors, [batch, output_dim] F32: handed to
    /// buffer_export, released with the session.
    result: Box<Buffer>,
    /// The rows as written, [max_batch, max_seq] at stride max_seq.
    rows: Mutex<Rows>,
}

/// The host buffers of one graph input kind.
enum Frames {
    Tokens { ids: Buffer, mask: Buffer, types: Option<Buffer> },
    Embeddings { rows: Buffer, bias: Buffer },
}

struct Rows {
    ids: Vec<i32>,
    mask: Vec<i32>,
    types: Vec<i32>,
    /// Pooling scratch, one token's sums: no allocation in a run.
    acc: Vec<f64>,
    /// One token's hidden row widened from FP16, so pooling reads f32.
    token: Vec<f32>,
    /// One embeddings frame, filled on the host and then written at the
    /// argument's precision. Empty for a token-id graph, and sized when
    /// the session is made so a run allocates nothing.
    gathered_rows: Vec<f32>,
    gathered_bias: Vec<f32>,
    state: State,
}

unsafe impl Send for Session {}
unsafe impl Sync for Session {}

impl Session {
    fn model(&self) -> &Model {
        unsafe { &*self.model }
    }
}

/// Write `text` into the tuning choices, NUL-terminated.
fn write_choices(dst: &mut [c_char; crate::TURBO_CHOICES_LEN], text: &str) {
    dst.fill(0);
    let n = text.len().min(crate::TURBO_CHOICES_LEN - 1);
    for (i, b) in text.as_bytes().iter().take(n).enumerate() {
        dst[i] = *b as c_char;
    }
}

/// # Safety
/// As turbo_backend.h says for session_create_tuned. The session is the
/// one `session_create` makes. `choices` is the graph format the driver
/// selected (`NGRAPH_LITE` or `NATIVE`). `cached` is not read: nothing
/// in the environment selects the format, and a measured choice is not
/// stored (`tuned` is DEFAULT).
pub(crate) unsafe extern "C" fn session_create_tuned(
    model: *mut c_void,
    task: u32,
    max_batch: u32,
    max_seq: u32,
    precision: u32,
    tuning: *mut crate::backend::turbo_backend_tuning,
    compute_dtype: *mut u32,
    out: *mut *mut c_void,
    err: *mut turbo_error,
) -> i32 {
    let rc = unsafe { session_create(model, task, max_batch, max_seq, precision, compute_dtype, out, err) };
    if rc == 0 && !tuning.is_null() {
        let format = unsafe { (*(model as *const Model)).graph_format };
        let t = unsafe { &mut *tuning };
        t.tuned = crate::TURBO_TUNED_DEFAULT;
        t.tune_ms = 0;
        write_choices(&mut t.choices, format);
    }
    rc
}

/// # Safety
/// As turbo_backend.h says for session_create.
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
            if task != TURBO_TASK_EMBED {
                return Err(fail(UNSUPPORTED, format!("npu: task {task} is not built; embed is")));
            }
            if precision == TURBO_PRECISION_EXACT {
                return Err(fail_field(
                    UNSUPPORTED_OPTION,
                    3,
                    format!(
                        "npu: EXACT asks for F32 throughout, and the graph was compiled to compute in {}",
                        if m.compute_dtype == TURBO_DTYPE_F16 { "F16" } else { "F32" }
                    ),
                ));
            }
            if max_seq > m.seq {
                return Err(fail_field(
                    UNSUPPORTED_OPTION,
                    2,
                    format!("npu: max_seq {max_seq} is over the graph's compiled seq {}", m.seq),
                ));
            }
            let ctx = m.ctx();
            let frame = m.frame_batch as u64 * m.seq as u64;
            let (frames, gathered_rows, gathered_bias) = match &m.inputs {
                GraphInputs::Tokens { ids, mask, types } => (
                    Frames::Tokens {
                        ids: Buffer::new(ctx, frame * ids.elem as u64, "token ids")?,
                        mask: Buffer::new(ctx, frame * mask.elem as u64, "the mask")?,
                        types: match types {
                            Some(t) => Some(Buffer::new(ctx, frame * t.elem as u64, "token types")?),
                            None => None,
                        },
                    },
                    Vec::new(),
                    Vec::new(),
                ),
                GraphInputs::Embeddings { rows, bias, heads, .. } => {
                    let n_rows = frame * m.hidden as u64;
                    let n_bias = m.frame_batch as u64 * u64::from(*heads) * m.seq as u64 * m.seq as u64;
                    (
                        Frames::Embeddings {
                            rows: Buffer::new(ctx, n_rows * rows.elem as u64, "word rows")?,
                            bias: Buffer::new(ctx, n_bias * bias.elem as u64, "the attention bias")?,
                        },
                        vec![0.0; n_rows as usize],
                        vec![0.0; n_bias as usize],
                    )
                }
            };
            let s = Session {
                model: m,
                max_seq,
                frames,
                out_buf: Buffer::new(ctx, frame * m.hidden as u64 * m.output.elem as u64, "hidden states")?,
                result: Box::new(Buffer::new(ctx, max_batch as u64 * m.hidden as u64 * 4, "the vectors")?),
                rows: Mutex::new(Rows {
                    ids: vec![0; (max_batch * max_seq) as usize],
                    mask: vec![0; (max_batch * max_seq) as usize],
                    types: vec![0; (max_batch * max_seq) as usize],
                    acc: vec![0.0; m.hidden as usize],
                    token: vec![0.0; m.hidden as usize],
                    gathered_rows,
                    gathered_bias,
                    state: State { batch: 0, seq: 0, pooling: 0, normalize: 0, output_dim: 0, written: false },
                }),
            };
            *compute_dtype = m.compute_dtype;
            *out = Box::into_raw(Box::new(s)) as *mut c_void;
            Ok(())
        })
    }
}

/// # Safety
/// `session` is a session this backend created, released once.
pub(crate) unsafe extern "C" fn session_release(session: *mut c_void) {
    quietly(|| drop(unsafe { Box::from_raw(session as *mut Session) }));
}

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
            let r = &*rows;
            let m = s.model();
            let (batch, seq, stride) = (r.batch as usize, r.seq as usize, r.row_stride as usize);
            let mut rows = s.rows.lock().unwrap_or_else(|p| p.into_inner());
            let dst_stride = s.max_seq as usize;
            for row in 0..batch {
                let src = row * stride;
                let dst = row * dst_stride;
                let ids = std::slice::from_raw_parts(r.ids.add(src), seq);
                let mask = std::slice::from_raw_parts(r.mask.add(src), seq);
                // The core promises at least one live token per row; a
                // row without one would pool a mean over nothing, so it
                // is refused here rather than returned as NaN.
                if !mask.contains(&1) {
                    rows.state.written = false;
                    return Err(fail(INVALID_ARGUMENT, format!("npu: row {row} has no token with mask 1")));
                }
                rows.ids[dst..dst + seq].copy_from_slice(ids);
                rows.mask[dst..dst + seq].copy_from_slice(mask);
                match r.types.is_null() {
                    true => rows.types[dst..dst + seq].fill(0),
                    false => {
                        let types = std::slice::from_raw_parts(r.types.add(src), seq);
                        if let Some(why) = type_refusal(&m.inputs)
                            && let Some(at) = types.iter().position(|&t| t != 0)
                        {
                            rows.state.written = false;
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
                written: true,
            };
            Ok(())
        })
    }
}

/// An i32 row written into a graph input at the argument's precision.
///
/// # Safety
/// `dst` holds `n` elements of the argument's precision at `at`.
unsafe fn write_tokens(dst: *mut c_void, precision: u32, at: usize, src: &[i32]) {
    unsafe {
        match precision {
            ze::GRAPH_ARGUMENT_PRECISION_INT64 | ze::GRAPH_ARGUMENT_PRECISION_UINT64 => {
                let d = (dst as *mut i64).add(at);
                for (i, &v) in src.iter().enumerate() {
                    d.add(i).write(v as i64);
                }
            }
            _ => {
                let d = (dst as *mut i32).add(at);
                d.copy_from_nonoverlapping(src.as_ptr(), src.len());
            }
        }
    }
}

/// F16's bits as F32, for reading hidden states back.
fn half_to_f32(h: u16) -> f32 {
    let (sign, exp, frac) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as u32, (h & 0x3ff) as u32);
    let bits = match (exp, frac) {
        (0, 0) => sign << 31,
        (0, f) => {
            // Subnormal: normalize it.
            let shift = f.leading_zeros() - 21;
            (sign << 31) | ((127 - 15 - shift + 1) << 23) | ((f << (shift + 13)) & 0x007f_ffff)
        }
        (0x1f, 0) => (sign << 31) | 0x7f80_0000,
        (0x1f, f) => (sign << 31) | 0x7f80_0000 | (f << 13),
        _ => (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

/// half_to_f32 over a row. With F16C the CPU widens eight at a time,
/// exactly, so a value is the same either way; a NaN stays a NaN.
fn widen_f16(src: &[u16], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("f16c") {
        // SAFETY: f16c was detected, and both slices are dst.len() long.
        return unsafe { widen_f16c(src, dst) };
    }
    for (d, &h) in dst.iter_mut().zip(src) {
        *d = half_to_f32(h);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn widen_f16c(src: &[u16], dst: &mut [f32]) {
    use std::arch::x86_64::{__m128i, _mm_loadu_si128, _mm256_cvtph_ps, _mm256_storeu_ps};
    let whole = src.len() / 8 * 8;
    for i in (0..whole).step_by(8) {
        unsafe {
            let h = _mm_loadu_si128(src.as_ptr().add(i) as *const __m128i);
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_cvtph_ps(h));
        }
    }
    for i in whole..src.len() {
        dst[i] = half_to_f32(src[i]);
    }
}

/// # Safety
/// As turbo_backend.h says for session_run.
pub(crate) unsafe extern "C" fn session_run(
    session: *mut c_void,
    out: *mut turbo_backend_run,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        guarded(err, || {
            let s = &*(session as *const Session);
            let mut rows = s.rows.lock().unwrap_or_else(|p| p.into_inner());
            if !rows.state.written {
                return Err(fail(INVALID_STATE, "npu: no rows are written"));
            }
            run(s, &mut rows, &mut *out)
        })
    }
}

/// One embeddings frame: the compiled arguments, the host buffers, and the
/// word table the gather reads.
struct EmbedFrame<'a> {
    rows_arg: &'a Arg,
    bias_arg: &'a Arg,
    rows_buf: &'a Buffer,
    bias_buf: &'a Buffer,
    table: *const f32,
    vocab: u32,
    heads: u32,
    hidden: usize,
    model_seq: usize,
    written: usize,
}

/// Word rows gathered from the table, and the bias, written at the
/// precision the compiler kept.
fn write_embedding_frame(frame: &EmbedFrame<'_>, live: usize, first: usize, stride: usize, rows: &mut Rows) -> Res<()> {
    let vocab = frame.vocab as usize;
    let heads = frame.heads as usize;
    let table = unsafe { std::slice::from_raw_parts(frame.table, vocab * frame.hidden) };
    rows.gathered_rows.fill(0.0);
    rows.gathered_bias.fill(MASKED);
    let row_elems = frame.model_seq * frame.hidden;
    let bias_elems = heads * frame.model_seq * frame.model_seq;
    for r in 0..live {
        let src = (first + r) * stride;
        gather_row(
            &Gather {
                table,
                vocab,
                hidden: frame.hidden,
                heads,
                model_seq: frame.model_seq,
                written: frame.written,
                ids: &rows.ids[src..src + frame.written],
                mask: &rows.mask[src..src + frame.written],
            },
            &mut rows.gathered_rows[r * row_elems..(r + 1) * row_elems],
            &mut rows.gathered_bias[r * bias_elems..(r + 1) * bias_elems],
        )?;
    }
    unsafe {
        write_floats(frame.rows_buf.ptr, frame.rows_arg.precision, &rows.gathered_rows)?;
        write_floats(frame.bias_buf.ptr, frame.bias_arg.precision, &rows.gathered_bias)?;
    }
    Ok(())
}

fn run(s: &Session, rows: &mut Rows, out: &mut turbo_backend_run) -> Res<()> {
    let m = s.model();
    let ctx = m.ctx();
    let set = m
        .ctx()
        .ext
        .set_argument_value()
        .ok_or_else(|| fail(UNSUPPORTED, "npu: the driver's graph extension has no pfnSetArgumentValue"))?;
    let _one_run_at_a_time = m.run.lock().unwrap_or_else(|p| p.into_inner());

    // The session's buffers onto the graph's arguments: values live on
    // the graph, and another session's are whatever it set last.
    let bind = |arg: &Arg, buf: &Buffer| ze_res("pfnSetArgumentValue", unsafe { set(m.graph, arg.index, buf.ptr) });
    let h2d_frame = match (&m.inputs, &s.frames) {
        (GraphInputs::Tokens { ids, mask, types }, Frames::Tokens { ids: ib, mask: mb, types: tb }) => {
            bind(ids, ib)?;
            bind(mask, mb)?;
            if let (Some(t), Some(b)) = (types, tb) {
                bind(t, b)?;
            }
            ib.bytes + mb.bytes + tb.as_ref().map_or(0, |b| b.bytes)
        }
        (GraphInputs::Embeddings { rows, bias, .. }, Frames::Embeddings { rows: rb, bias: bb }) => {
            bind(rows, rb)?;
            bind(bias, bb)?;
            rb.bytes + bb.bytes
        }
        _ => return Err(fail(INVALID_STATE, "npu: the session was not made for this model")),
    };
    bind(&m.output, &s.out_buf)?;

    let (batch, seq) = (rows.state.batch as usize, rows.state.seq as usize);
    let (pooling, normalize, output_dim) = (rows.state.pooling, rows.state.normalize, rows.state.output_dim);
    let (frame_batch, model_seq, hidden) = (m.frame_batch as usize, m.seq as usize, m.hidden as usize);
    let stride = s.max_seq as usize;
    let frames = batch.div_ceil(frame_batch);
    let mut h2d = 0u64;
    let mut d2h = 0u64;
    let result = s.result.ptr as *mut f32;

    for frame in 0..frames {
        let first = frame * frame_batch;
        let live = (batch - first).min(frame_batch);
        // The frame's inputs. Token ids are the live tokens and zeros
        // after. An embeddings graph is gathered here: each token's word
        // row, and the bias the mask asks for. Rows past the batch stay
        // zero (and, for the bias, dropped), and nothing reads them.
        match (&m.inputs, &s.frames) {
            (GraphInputs::Tokens { ids, mask, types }, Frames::Tokens { ids: ib, mask: mb, types: tb }) => unsafe {
                std::ptr::write_bytes(ib.ptr as *mut u8, 0, ib.bytes as usize);
                std::ptr::write_bytes(mb.ptr as *mut u8, 0, mb.bytes as usize);
                if let Some(b) = tb {
                    std::ptr::write_bytes(b.ptr as *mut u8, 0, b.bytes as usize);
                }
                for r in 0..live {
                    let src = (first + r) * stride;
                    let dst = r * model_seq;
                    write_tokens(ib.ptr, ids.precision, dst, &rows.ids[src..src + seq]);
                    write_tokens(mb.ptr, mask.precision, dst, &rows.mask[src..src + seq]);
                    if let (Some(t), Some(b)) = (types, tb) {
                        write_tokens(b.ptr, t.precision, dst, &rows.types[src..src + seq]);
                    }
                }
            },
            (
                GraphInputs::Embeddings { rows: rows_arg, bias: bias_arg, heads, table, vocab },
                Frames::Embeddings { rows: rb, bias: bb },
            ) => {
                write_embedding_frame(
                    &EmbedFrame {
                        rows_arg,
                        bias_arg,
                        rows_buf: rb,
                        bias_buf: bb,
                        table: *table,
                        vocab: *vocab,
                        heads: *heads,
                        hidden,
                        model_seq,
                        written: seq,
                    },
                    live,
                    first,
                    stride,
                    rows,
                )?;
            }
            _ => return Err(fail(INVALID_STATE, "npu: the session was not made for this model")),
        }
        h2d += h2d_frame;

        ctx.execute(m.graph)?;
        d2h += s.out_buf.bytes;

        // Pool, cut and normalize each live row on the host. Each token
        // row is read once, as f32: widened from FP16 or read in place.
        let frame_elems = live * model_seq * hidden;
        let fp16 = m.output.precision == ze::GRAPH_ARGUMENT_PRECISION_FP16;
        for r in 0..live {
            let row = first + r;
            let mask = &rows.mask[row * stride..row * stride + seq];
            let (acc, token) = (&mut rows.acc, &mut rows.token);
            let mut add = |t: usize, acc: &mut [f64]| {
                let at = (r * model_seq + t) * hidden;
                debug_assert!(at + hidden <= frame_elems);
                let values: &[f32] = if fp16 {
                    // SAFETY: out_buf holds the frame at the output's precision.
                    let h = unsafe { std::slice::from_raw_parts((s.out_buf.ptr as *const u16).add(at), hidden) };
                    widen_f16(h, token);
                    token
                } else {
                    unsafe { std::slice::from_raw_parts((s.out_buf.ptr as *const f32).add(at), hidden) }
                };
                for (a, &v) in acc.iter_mut().zip(values) {
                    *a += v as f64;
                }
            };
            acc.fill(0.0);
            match pooling {
                TURBO_POOLING_CLS => add(0, acc),
                TURBO_POOLING_LAST => add(mask.iter().rposition(|&v| v == 1).unwrap_or(0), acc),
                _ => {
                    debug_assert_eq!(pooling, TURBO_POOLING_MEAN);
                    let mut live_tokens = 0f64;
                    for (t, &v) in mask.iter().enumerate() {
                        if v == 1 {
                            live_tokens += 1.0;
                            add(t, acc);
                        }
                    }
                    for a in acc.iter_mut() {
                        *a /= live_tokens;
                    }
                }
            }
            let dim = output_dim as usize;
            let vector = &acc[..dim];
            let norm = match normalize {
                // The same floor the cpu uses: a zero vector stays finite.
                TURBO_NORMALIZE_L2 => vector.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12),
                _ => 1.0,
            };
            for (h, v) in vector.iter().enumerate() {
                unsafe { result.add(row * dim + h).write((v / norm) as f32) };
            }
        }
    }

    out.placement = TURBO_PLACE_HOST;
    out.output = &*s.result as *const Buffer as *mut c_void;
    out.host = s.result.ptr;
    out.h2d_bytes = h2d;
    out.d2h_bytes = d2h;
    out.host_allocs = 0;
    out.device_allocs = 0;
    out.stage[TURBO_EMBED_STAGE_UPLOAD] = TURBO_STAGE_DEVICE;
    out.stage[TURBO_EMBED_STAGE_LOOKUP] = lookup_stage(&m.inputs);
    out.stage[TURBO_EMBED_STAGE_ENCODE] = TURBO_STAGE_DEVICE;
    out.stage[TURBO_EMBED_STAGE_POOL] = TURBO_STAGE_HOST;
    if normalize == TURBO_NORMALIZE_L2 {
        out.stage[TURBO_EMBED_STAGE_NORMALIZE] = TURBO_STAGE_HOST;
    }
    out.stage[TURBO_EMBED_STAGE_DOWNLOAD] = TURBO_STAGE_DEVICE;
    Ok(())
}

/// Where the word-row lookup ran. An embeddings graph gathers on the
/// host. A token-id graph hands ids to the device, and the graph does
/// the lookup.
fn lookup_stage(inputs: &GraphInputs) -> u32 {
    match inputs {
        GraphInputs::Embeddings { .. } => TURBO_STAGE_HOST,
        GraphInputs::Tokens { .. } => TURBO_STAGE_DEVICE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halves_read_back_as_the_f32_they_name() {
        assert_eq!(half_to_f32(0x0000), 0.0);
        assert_eq!(half_to_f32(0x8000), -0.0);
        assert_eq!(half_to_f32(0x3c00), 1.0);
        assert_eq!(half_to_f32(0xbc00), -1.0);
        assert_eq!(half_to_f32(0x4000), 2.0);
        assert_eq!(half_to_f32(0x3555), 0.333_251_95);
        assert_eq!(half_to_f32(0x7bff), 65504.0, "the largest finite half");
        assert_eq!(half_to_f32(0x0001), 5.960_464_5e-8, "the smallest subnormal");
        assert_eq!(half_to_f32(0x03ff), 6.097_555e-5, "the largest subnormal");
        assert_eq!(half_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(half_to_f32(0xfc00), f32::NEG_INFINITY);
        assert!(half_to_f32(0x7e00).is_nan());
    }

    fn ok_dtype(manifest: u32, precision: u32) -> u32 {
        match session_dtype(manifest, precision) {
            Ok(d) => d,
            Err(e) => panic!("{}", e.message),
        }
    }

    #[test]
    fn widen_f16_is_half_to_f32_for_every_half() {
        let all: Vec<u16> = (0..=u16::MAX).collect();
        // An odd length runs the tail after the eight-wide steps.
        let src = &all[..all.len() - 3];
        let mut wide = vec![0.0f32; src.len()];
        widen_f16(src, &mut wide);
        for (&h, &w) in src.iter().zip(&wide) {
            let one = half_to_f32(h);
            if one.is_nan() {
                assert!(w.is_nan(), "{h:#06x}");
            } else {
                assert_eq!(w.to_bits(), one.to_bits(), "{h:#06x}");
            }
        }
    }

    fn err_msg(r: Res<()>) -> String {
        match r {
            Ok(()) => panic!("expected a refusal"),
            Err(e) => e.message,
        }
    }

    fn ok_msg<T>(r: Result<T, Fail>) -> T {
        match r {
            Ok(v) => v,
            Err(e) => panic!("{}", e.message),
        }
    }

    fn err_any<T>(r: Result<T, Fail>) -> String {
        match r {
            Ok(_) => panic!("expected a refusal"),
            Err(e) => e.message,
        }
    }

    #[test]
    fn the_session_dtype_is_the_compiled_precision() {
        assert_eq!(ok_dtype(0, ze::GRAPH_ARGUMENT_PRECISION_FP16), TURBO_DTYPE_F16);
        assert_eq!(ok_dtype(0, ze::GRAPH_ARGUMENT_PRECISION_FP32), TURBO_DTYPE_F32);
        assert_eq!(ok_dtype(TURBO_DTYPE_F16, ze::GRAPH_ARGUMENT_PRECISION_FP16), TURBO_DTYPE_F16);
        assert_eq!(ok_dtype(TURBO_DTYPE_F32, ze::GRAPH_ARGUMENT_PRECISION_FP32), TURBO_DTYPE_F32);
        let e = match session_dtype(TURBO_DTYPE_F32, ze::GRAPH_ARGUMENT_PRECISION_FP16) {
            Ok(d) => panic!("kept manifest dtype {d}"),
            Err(e) => e.message,
        };
        assert!(e.contains("DTYPE_F16") && e.contains("DTYPE_F32"), "{e}");
        let e = match session_dtype(TURBO_DTYPE_F16, ze::GRAPH_ARGUMENT_PRECISION_FP32) {
            Ok(d) => panic!("kept manifest dtype {d}"),
            Err(e) => e.message,
        };
        assert!(e.contains("computes in DTYPE_F32"), "{e}");
        let e = match session_dtype(TURBO_DTYPE_F16, ze::GRAPH_ARGUMENT_PRECISION_BF16) {
            Ok(_) => panic!("BF16 became a session dtype"),
            Err(e) => e.message,
        };
        assert!(e.contains("FP16 and FP32"), "{e}");
    }

    #[test]
    fn the_device_layout_is_the_packed_one_the_flags_named() {
        assert!(expect_packed_layout("input_ids", 2, ze::GRAPH_ARGUMENT_LAYOUT_NC).is_ok());
        assert!(expect_packed_layout("last_hidden_state", 3, ze::GRAPH_ARGUMENT_LAYOUT_CHW).is_ok());
        let e = err_msg(expect_packed_layout("input_ids", 2, 0xC8));
        assert!(e.contains("BLOCKED") && e.contains("NC"), "{e}");
        let e = err_msg(expect_packed_layout("attention_mask", 2, 0));
        assert!(e.contains("ANY"), "{e}");
        assert!(expect_packed_layout("input_ids", 4, ze::GRAPH_ARGUMENT_LAYOUT_NC).is_err());
        assert!(expect_packed_layout("attn_bias", 4, ze::GRAPH_ARGUMENT_LAYOUT_NCHW).is_ok());
        let e = err_msg(expect_packed_layout("attn_bias", 4, ze::GRAPH_ARGUMENT_LAYOUT_NC));
        assert!(e.contains("NCHW"), "{e}");
    }

    fn manifest(graph_input: u32, hidden: u32, heads: u32, seq: u32, batch: u32) -> turbo_backend_model {
        turbo_backend_model {
            struct_size: 0,
            family: 0,
            dtype: 0,
            layers: 0,
            hidden,
            heads,
            intermediate: 0,
            vocab_size: 0,
            max_positions: 0,
            token_types: 0,
            layer_norm_eps: 0.0,
            tensor_count: 0,
            position_offset: 0,
            tensors: std::ptr::null(),
            format: 0,
            graph_input,
            graph_output: 0,
            compute_dtype: 0,
            fixed_seq: seq,
            fixed_batch: batch,
            artifact: std::ptr::null(),
            artifact_bytes: 0,
            artifact2: std::ptr::null(),
            artifact2_bytes: 0,
        }
    }

    fn compiled(name: &str, precision: u32, dims: &[u32], layout: u32) -> Compiled {
        let mut d = [0u32; 5];
        d[..dims.len()].copy_from_slice(dims);
        let elem = if precision == ze::GRAPH_ARGUMENT_PRECISION_FP16 { 2 } else { 4 };
        Compiled { arg: Arg { index: 0, name: name.into(), precision, elem }, dims: d, rank: dims.len() as u32, layout }
    }

    #[test]
    fn the_host_gathers_word_rows_and_a_mask_bias() {
        // Four word rows of hidden 2. Written length 3 of a frame of 4:
        // the last position repeats the first id, and keys the mask drops
        // are MASKED for every head and every query.
        let table = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut rows = vec![0.0; 8];
        let mut bias = vec![0.0; 2 * 16];
        ok_msg(gather_row(
            &Gather {
                table: &table,
                vocab: 4,
                hidden: 2,
                heads: 2,
                model_seq: 4,
                written: 3,
                ids: &[2, 0, 1],
                mask: &[1, 0, 1],
            },
            &mut rows,
            &mut bias,
        ));
        assert_eq!(rows, [5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        for head in 0..2 {
            for q in 0..4 {
                let base = (head * 4 + q) * 4;
                assert_eq!(&bias[base..base + 4], &[0.0, MASKED, 0.0, MASKED], "head {head} query {q}");
            }
        }
    }

    #[test]
    fn a_token_id_outside_the_table_is_refused() {
        let table = [1.0, 2.0];
        let mut rows = vec![0.0; 2];
        let mut bias = vec![0.0; 1];
        let e = err_any(gather_row(
            &Gather { table: &table, vocab: 1, hidden: 2, heads: 1, model_seq: 1, written: 1, ids: &[3], mask: &[1] },
            &mut rows,
            &mut bias,
        ));
        assert!(e.contains("token id 3") && e.contains("1 rows"), "{e}");
    }

    #[test]
    fn halves_round_trip_the_values_the_gather_writes() {
        for v in [0.0, -0.0, 1.0, -1.0, 2.0, MASKED] {
            assert_eq!(half_to_f32(f32_to_half(v)).to_bits(), v.to_bits());
        }
        for bits in 0u16..=0xffff {
            let f = half_to_f32(bits);
            if f.is_nan() {
                assert!(half_to_f32(f32_to_half(f)).is_nan());
            } else {
                assert_eq!(f32_to_half(f), bits, "half {bits:#06x}");
            }
        }
    }

    #[test]
    fn an_embeddings_graph_is_word_rows_and_an_attention_bias() {
        let desc = manifest(TURBO_INPUT_EMBEDDINGS, 2, 2, 4, 1);
        let inputs = vec![
            compiled("word_rows", ze::GRAPH_ARGUMENT_PRECISION_FP32, &[1, 4, 2], ze::GRAPH_ARGUMENT_LAYOUT_CHW),
            compiled("attn_bias", ze::GRAPH_ARGUMENT_PRECISION_FP16, &[1, 2, 4, 4], ze::GRAPH_ARGUMENT_LAYOUT_NCHW),
        ];
        let output =
            compiled("last_hidden_state", ze::GRAPH_ARGUMENT_PRECISION_FP16, &[1, 4, 2], ze::GRAPH_ARGUMENT_LAYOUT_CHW);
        let (rows, bias, out, batch, seq, hidden) = ok_msg(accept_embeddings(inputs, output, &desc));
        assert_eq!(
            (rows.name.as_str(), bias.name.as_str(), out.name.as_str(), batch, seq, hidden),
            ("word_rows", "attn_bias", "last_hidden_state", 1, 4, 2)
        );

        let inputs = vec![
            compiled("input_ids", ze::GRAPH_ARGUMENT_PRECISION_FP32, &[1, 4, 2], ze::GRAPH_ARGUMENT_LAYOUT_CHW),
            compiled("attn_bias", ze::GRAPH_ARGUMENT_PRECISION_FP32, &[1, 2, 4, 4], ze::GRAPH_ARGUMENT_LAYOUT_NCHW),
        ];
        let e = err_any(accept_embeddings(
            inputs,
            compiled("h", ze::GRAPH_ARGUMENT_PRECISION_FP32, &[1, 4, 2], ze::GRAPH_ARGUMENT_LAYOUT_CHW),
            &desc,
        ));
        assert!(e.contains("input_ids") && e.contains("word_rows"), "{e}");

        let inputs = vec![
            compiled("word_rows", ze::GRAPH_ARGUMENT_PRECISION_FP32, &[1, 4, 2], ze::GRAPH_ARGUMENT_LAYOUT_CHW),
            compiled("attn_bias", ze::GRAPH_ARGUMENT_PRECISION_FP32, &[1, 3, 4, 4], ze::GRAPH_ARGUMENT_LAYOUT_NCHW),
        ];
        let e = err_any(accept_embeddings(
            inputs,
            compiled("h", ze::GRAPH_ARGUMENT_PRECISION_FP32, &[1, 4, 2], ze::GRAPH_ARGUMENT_LAYOUT_CHW),
            &desc,
        ));
        assert!(e.contains("[1, 3, 4, 4]") && e.contains("[1, 2, 4, 4]"), "{e}");

        let inputs = vec![
            compiled("word_rows", ze::GRAPH_ARGUMENT_PRECISION_FP32, &[1, 4, 2], ze::GRAPH_ARGUMENT_LAYOUT_NC),
            compiled("attn_bias", ze::GRAPH_ARGUMENT_PRECISION_FP32, &[1, 2, 4, 4], ze::GRAPH_ARGUMENT_LAYOUT_NCHW),
        ];
        let e = err_any(accept_embeddings(
            inputs,
            compiled("h", ze::GRAPH_ARGUMENT_PRECISION_FP32, &[1, 4, 2], ze::GRAPH_ARGUMENT_LAYOUT_CHW),
            &desc,
        ));
        assert!(e.contains("CHW"), "{e}");
    }

    #[test]
    fn a_token_id_graph_still_refuses_an_embeddings_input_name() {
        let desc = manifest(TURBO_INPUT_TOKEN_IDS, 2, 2, 4, 1);
        let inputs = vec![
            compiled("input_ids", ze::GRAPH_ARGUMENT_PRECISION_INT64, &[1, 4], ze::GRAPH_ARGUMENT_LAYOUT_NC),
            compiled("attention_mask", ze::GRAPH_ARGUMENT_PRECISION_INT32, &[1, 4], ze::GRAPH_ARGUMENT_LAYOUT_NC),
        ];
        let (got, out, batch, seq, hidden) = ok_msg(accept_tokens(
            inputs,
            compiled("last_hidden_state", ze::GRAPH_ARGUMENT_PRECISION_FP16, &[1, 4, 2], ze::GRAPH_ARGUMENT_LAYOUT_CHW),
            &desc,
        ));
        assert!(matches!(got, GraphInputs::Tokens { types: None, .. }));
        assert_eq!((batch, seq, hidden, out.name.as_str()), (1, 4, 2, "last_hidden_state"));

        let inputs = vec![
            compiled("word_rows", ze::GRAPH_ARGUMENT_PRECISION_INT64, &[1, 4], ze::GRAPH_ARGUMENT_LAYOUT_NC),
            compiled("attention_mask", ze::GRAPH_ARGUMENT_PRECISION_INT64, &[1, 4], ze::GRAPH_ARGUMENT_LAYOUT_NC),
        ];
        let e = err_any(accept_tokens(
            inputs,
            compiled("h", ze::GRAPH_ARGUMENT_PRECISION_FP32, &[1, 4, 2], ze::GRAPH_ARGUMENT_LAYOUT_CHW),
            &desc,
        ));
        assert!(e.contains("word_rows"), "{e}");
    }

    #[test]
    fn a_nonzero_token_type_on_an_embeddings_graph_names_the_row() {
        let e = type_error(1, 4, 2, "an INPUT_EMBEDDINGS graph computes token type 0 only");
        assert_eq!(e.code, UNSUPPORTED_OPTION);
        assert!(
            e.message.contains("token type 2") && e.message.contains("row 1") && e.message.contains("position 4"),
            "{}",
            e.message
        );
        assert!(
            type_refusal(&GraphInputs::Embeddings {
                rows: Arg { index: 0, name: String::new(), precision: 0, elem: 4 },
                bias: Arg { index: 1, name: String::new(), precision: 0, elem: 4 },
                heads: 1,
                table: std::ptr::null(),
                vocab: 1,
            })
            .unwrap()
            .contains("INPUT_EMBEDDINGS")
        );
    }

    #[test]
    fn the_word_table_is_the_f32_host_tensor() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let name = CString::new("embeddings.word_embeddings.weight").unwrap();
        let tensor = turbo_backend_tensor {
            name: name.as_ptr(),
            data: data.as_ptr() as *const _,
            shape: [2, 2],
            ndim: 2,
            dtype: TURBO_DTYPE_F32,
            bytes: 16,
        };
        let tensors = [tensor, tensor, tensor, tensor, tensor];
        let mut desc = manifest(TURBO_INPUT_EMBEDDINGS, 2, 1, 4, 1);
        desc.vocab_size = 2;
        desc.dtype = TURBO_DTYPE_F32;
        desc.tensor_count = TURBO_BERT_EMBEDDING_TENSORS;
        desc.tensors = tensors.as_ptr();
        let (p, vocab) = ok_msg(word_table(&desc));
        assert_eq!(vocab, 2);
        assert_eq!(unsafe { std::slice::from_raw_parts(p, 4) }, &data);

        desc.dtype = TURBO_DTYPE_F16;
        let e = err_any(word_table(&desc));
        assert!(e.contains("DTYPE_F16"), "{e}");
        desc.tensor_count = 1;
        desc.dtype = TURBO_DTYPE_F32;
        let e = err_any(word_table(&desc));
        assert!(e.contains("5 host embedding tensors") && e.contains("1 were handed"), "{e}");
        desc.tensor_count = 0;
        let e = err_any(word_table(&desc));
        assert!(e.contains("none was handed"), "{e}");
    }

    #[test]
    fn the_mask_and_the_output_share_the_ids_frame() {
        assert!(expect_frame("attention_mask", &[1, 128], 1, 128).is_ok());
        assert!(expect_frame("token_type_ids", &[1, 128, 0, 0, 0], 1, 128).is_ok());
        assert!(expect_frame("last_hidden_state", &[1, 128, 384], 1, 128).is_ok());
        let e = err_msg(expect_frame("attention_mask", &[1, 64], 1, 128));
        assert!(e.contains("[1, 64, ...]"), "{e}");
        let e = err_msg(expect_frame("last_hidden_state", &[2, 128, 384], 1, 128));
        assert!(e.contains("the token frame is [1, 128]"), "{e}");
    }

    #[test]
    fn two_rows_of_one_frame_are_gathered_into_the_host_buffers() {
        // Frame of three. Two rows are written, at a stride longer than
        // the compiled seq, with ids past the written length that must
        // not be read. The third row stays zero word rows and an all
        // MASKED bias, and both land in the host buffers.
        let table = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let hidden = 2usize;
        let model_seq = 4usize;
        let heads = 2usize;
        let stride = 8usize;
        let written = 3usize;
        let frame = 3usize;
        let mut ids = vec![99i32; frame * stride];
        let mut mask = vec![1i32; frame * stride];
        ids[..3].copy_from_slice(&[2, 0, 1]);
        mask[..3].copy_from_slice(&[1, 0, 1]);
        ids[stride..stride + 3].copy_from_slice(&[1, 3, 0]);
        mask[stride..stride + 3].copy_from_slice(&[1, 1, 0]);
        let row_elems = model_seq * hidden;
        let bias_elems = heads * model_seq * model_seq;
        let mut rows = Rows {
            ids,
            mask,
            types: vec![0; frame * stride],
            acc: vec![0.0; hidden],
            token: vec![0.0; hidden],
            gathered_rows: vec![7.0; frame * row_elems],
            gathered_bias: vec![7.0; frame * bias_elems],
            state: State {
                batch: 2,
                seq: written as u32,
                pooling: 0,
                normalize: 0,
                output_dim: hidden as u32,
                written: true,
            },
        };
        let mut host_rows = vec![9.0f32; frame * row_elems];
        let mut host_bias = vec![9.0f32; frame * bias_elems];
        let rows_buf = Buffer { free: None, ptr: host_rows.as_mut_ptr().cast(), bytes: (host_rows.len() * 4) as u64 };
        let bias_buf = Buffer { free: None, ptr: host_bias.as_mut_ptr().cast(), bytes: (host_bias.len() * 4) as u64 };
        let rows_arg =
            Arg { index: 0, name: "word_rows".into(), precision: ze::GRAPH_ARGUMENT_PRECISION_FP32, elem: 4 };
        let bias_arg =
            Arg { index: 1, name: "attn_bias".into(), precision: ze::GRAPH_ARGUMENT_PRECISION_FP32, elem: 4 };
        ok_msg(write_embedding_frame(
            &EmbedFrame {
                rows_arg: &rows_arg,
                bias_arg: &bias_arg,
                rows_buf: &rows_buf,
                bias_buf: &bias_buf,
                table: table.as_ptr(),
                vocab: 4,
                heads: heads as u32,
                hidden,
                model_seq,
                written,
            },
            2,
            0,
            stride,
            &mut rows,
        ));
        assert_eq!(&host_rows[0..8], &[5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "row 0");
        assert_eq!(&host_rows[8..16], &[3.0, 4.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0], "row 1");
        assert!(host_rows[16..].iter().all(|&v| v == 0.0), "the unused row is zero");
        let plane = model_seq * model_seq;
        for head in 0..heads {
            for q in 0..model_seq {
                let base = head * plane + q * model_seq;
                assert_eq!(&host_bias[base..base + 4], &[0.0, MASKED, 0.0, MASKED], "row 0 head {head} query {q}");
                let base = bias_elems + base;
                assert_eq!(&host_bias[base..base + 4], &[0.0, 0.0, MASKED, MASKED], "row 1 head {head} query {q}");
            }
        }
        assert!(host_bias[2 * bias_elems..].iter().all(|&v| v == MASKED), "the unused bias is dropped");
    }

    #[test]
    fn an_embeddings_gather_reports_lookup_on_the_host() {
        let arg =
            |name: &str| Arg { index: 0, name: name.into(), precision: ze::GRAPH_ARGUMENT_PRECISION_FP32, elem: 4 };
        let embeddings = GraphInputs::Embeddings {
            rows: arg("word_rows"),
            bias: arg("attn_bias"),
            heads: 1,
            table: std::ptr::null(),
            vocab: 1,
        };
        assert_eq!(lookup_stage(&embeddings), TURBO_STAGE_HOST);
        let tokens = GraphInputs::Tokens { ids: arg("input_ids"), mask: arg("attention_mask"), types: None };
        assert_eq!(lookup_stage(&tokens), TURBO_STAGE_DEVICE);
    }
}
