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
//! Argument values live on the graph, so runs on one model are
//! serialized under the model's lock.

use std::ffi::{CString, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Mutex;

use super::ze::{self, GraphExt, Handle};
use super::{Device, Driver, ir};
use crate::backend::{
    TURBO_FORMAT_OPENVINO_IR, TURBO_INPUT_TOKEN_IDS, TURBO_OUTPUT_HIDDEN_STATES, refuse, refuse_field,
    turbo_backend_embed_rows, turbo_backend_model, turbo_backend_run,
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
            ze_res("zeCommandListHostSynchronize", unsafe { (self.api.command_list_host_synchronize)(list, u64::MAX) })
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
    api: &'static ze::Api,
    context: Handle,
    ptr: *mut c_void,
    bytes: u64,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    fn new(ctx: &Context, bytes: u64, what: &str) -> Res<Buffer> {
        let ptr = ctx.alloc_host(bytes as usize, what)?;
        Ok(Buffer { api: ctx.api, context: ctx.handle, ptr, bytes })
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe { (self.api.mem_free)(self.context, self.ptr) };
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

pub(crate) struct Model {
    ctx: *const Context,
    graph: Handle,
    /// Argument values live on the graph: one run at a time per model.
    run: Mutex<()>,
    /// The graph's inputs, in argument order: token ids, then the mask,
    /// then token type ids where the graph takes them.
    ids: Arg,
    mask: Arg,
    types: Option<Arg>,
    output: Arg,
    /// The shapes compiled in: a frame of rows, each of seq tokens.
    frame_batch: u32,
    seq: u32,
    hidden: u32,
    /// TURBO_DTYPE_* a session computes in.
    pub compute_dtype: u32,
}

unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {
    fn ctx(&self) -> &Context {
        unsafe { &*self.ctx }
    }
}

/// The packed device layout the build flags name for a rank: NC for the
/// rank-2 token frame, CHW for the rank-3 hidden states.
fn packed_layout(rank: u32) -> Option<u32> {
    match rank {
        2 => Some(ze::GRAPH_ARGUMENT_LAYOUT_NC),
        3 => Some(ze::GRAPH_ARGUMENT_LAYOUT_CHW),
        _ => None,
    }
}

fn layout_label(layout: u32) -> String {
    match layout {
        ze::GRAPH_ARGUMENT_LAYOUT_NC => "NC".to_string(),
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
            format!("npu: {name:?} has rank {rank}; the build flags name a packed layout for rank 2 or 3"),
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
    if desc.graph_input != TURBO_INPUT_TOKEN_IDS {
        return Err(fail(
            UNSUPPORTED,
            "npu: only INPUT_TOKEN_IDS artifacts run; the host-gather path for INPUT_EMBEDDINGS is not built",
        ));
    }
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
            format!("npu: the driver's compiler is {maj}.{min}; indexed IO build flags need 5.9 or later"),
        ));
    }
    let create2 = ctx.ext.create2().ok_or_else(|| {
        fail(
            UNSUPPORTED,
            format!(
                "npu: the driver's graph extension is {}.{}; compiling an IR needs 1.5 or later",
                ctx.ext.version >> 16,
                ctx.ext.version & 0xffff
            ),
        )
    })?;

    // What the device said it compiles, checked before any compile is
    // tried, so a refusal says why in words rather than a status code.
    if ctx.formats_supported & ze::GRAPH_FORMAT_NGRAPH_LITE == 0 {
        return Err(fail(
            UNSUPPORTED,
            "npu: the device's compiler does not take an OpenVINO IR (NGRAPH_LITE is not among its graph formats); \
             only pre-compiled blobs would run, and no bundle format carries one",
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
    let gdesc = ze::GraphDesc2 {
        stype: ze::STRUCTURE_TYPE_GRAPH_DESC_2,
        p_next: std::ptr::null(),
        format: ze::GRAPH_FORMAT_NGRAPH_LITE,
        input_size: container.len(),
        input: container.as_ptr(),
        build_flags: cflags.as_ptr(),
        flags: 0,
    };

    // pfnCreate3 returns the compiler's log beside a failure; without it
    // (extension 1.5 to 1.11) the status code stands alone.
    let mut graph = std::ptr::null_mut();
    let rc = match ctx.ext.create3() {
        Some(create3) => {
            let mut log = std::ptr::null_mut();
            let rc = unsafe { create3(ctx.handle, ctx.device, &gdesc, &mut graph, &mut log) };
            if rc != 0 {
                let text = build_log(&ctx.ext, log);
                return Err(fail(
                    RUNTIME,
                    format!("npu: the driver's compiler refused the IR with 0x{rc:08x}: {text}"),
                ));
            }
            build_log(&ctx.ext, log);
            rc
        }
        None => unsafe { create2(ctx.handle, ctx.device, &gdesc, &mut graph) },
    };
    ze_res("pfnGraphCreate2", rc)?;
    ctx.say(LOG_DEBUG, &format!("npu device {}: the IR is compiled ({flags})", ctx.ordinal));

    let model = describe(ctx, desc, graph);
    if model.is_err()
        && let Some(destroy) = ctx.ext.destroy()
    {
        unsafe { destroy(graph) };
    }
    model
}

/// The compiled graph's arguments and shapes, checked against the
/// manifest, and the graph initialized: the weights' move to the device.
fn describe(ctx: &Context, desc: &turbo_backend_model, graph: Handle) -> Res<Model> {
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
            name,
        };
        let rank = if a.dims_count != 0 { a.dims_count } else { a.dims.iter().take_while(|&&d| d > 0).count() as u32 };
        let compiled = Compiled { arg, dims: a.dims, rank, layout: a.device_layout };
        match a.kind {
            ze::GRAPH_ARGUMENT_TYPE_INPUT => inputs.push(compiled),
            _ => outputs.push(compiled),
        }
    }

    // The encoder's boundary: ids and mask, token types where the graph
    // takes them, and the hidden states back. Each input is taken by its
    // name, the ones a BERT export gives, and an input named anything
    // else is refused: a graph is never run on a guess about which
    // argument is which.
    let names = |v: &[Compiled]| v.iter().map(|c| c.arg.name.clone()).collect::<Vec<_>>();
    if !(2..=3).contains(&inputs.len()) || outputs.is_empty() {
        return Err(fail(
            RUNTIME,
            format!(
                "npu: the graph takes {:?} and returns {:?}; an encoder takes input_ids, attention_mask and \
                 optionally token_type_ids, and returns last_hidden_state",
                names(&inputs),
                names(&outputs)
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
    // One output is the hidden states whatever its name; among several,
    // only the one named last_hidden_state is, and anything else is
    // refused rather than guessed at.
    let output = match outputs.len() {
        1 => outputs.remove(0),
        _ => {
            let at = outputs.iter().position(|c| c.arg.name == "last_hidden_state").ok_or_else(|| {
                fail(RUNTIME, format!("npu: the graph returns {:?} and none is last_hidden_state", names(&outputs)))
            })?;
            outputs.remove(at)
        }
    };

    // Every token input shares the ids frame, and the hidden states'
    // leading dims are that same frame. The device layout is the packed
    // one the build flags asked for: NC on the rank-2 inputs, CHW on the
    // rank-3 output. A blocked layout is refused, because the host
    // writes packed rows.
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
    let hidden = output.dims[2];
    let ids = ids.arg;
    let mask = mask.arg;
    let output = output.arg;
    if !matches!(output.precision, ze::GRAPH_ARGUMENT_PRECISION_FP32 | ze::GRAPH_ARGUMENT_PRECISION_FP16) {
        return Err(fail(
            RUNTIME,
            format!(
                "npu: {:?} is argument precision 0x{:02x}; FP32 and FP16 are read back",
                output.name, output.precision
            ),
        ));
    }
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

    // The compiled shape against the manifest: a mismatch is the
    // bundle's fault, said plainly. A dynamic graph is refused before a
    // zero manifest shape, so the message stays about the graph.
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
                ze_res("zeCommandListHostSynchronize", unsafe {
                    (ctx.api.command_list_host_synchronize)(list, u64::MAX)
                })
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
    Ok(Model { ctx, graph, run: Mutex::new(()), ids, mask, types, output, frame_batch, seq, hidden, compute_dtype })
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
    ids_buf: Buffer,
    mask_buf: Buffer,
    types_buf: Option<Buffer>,
    /// The graph's hidden states, one frame.
    out_buf: Buffer,
    /// The run's vectors, [batch, output_dim] F32: handed to
    /// buffer_export, released with the session.
    result: Box<Buffer>,
    /// The rows as written, [max_batch, max_seq] at stride max_seq.
    rows: Mutex<Rows>,
}

struct Rows {
    ids: Vec<i32>,
    mask: Vec<i32>,
    types: Vec<i32>,
    /// Pooling scratch, one token's sums: no allocation in a run.
    acc: Vec<f64>,
    state: State,
}

unsafe impl Send for Session {}
unsafe impl Sync for Session {}

impl Session {
    fn model(&self) -> &Model {
        unsafe { &*self.model }
    }
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
            let s = Session {
                model: m,
                max_seq,
                ids_buf: Buffer::new(ctx, frame * m.ids.elem as u64, "token ids")?,
                mask_buf: Buffer::new(ctx, frame * m.mask.elem as u64, "the mask")?,
                types_buf: match &m.types {
                    Some(t) => Some(Buffer::new(ctx, frame * t.elem as u64, "token types")?),
                    None => None,
                },
                out_buf: Buffer::new(ctx, frame * m.hidden as u64 * m.output.elem as u64, "hidden states")?,
                result: Box::new(Buffer::new(ctx, max_batch as u64 * m.hidden as u64 * 4, "the vectors")?),
                rows: Mutex::new(Rows {
                    ids: vec![0; (max_batch * max_seq) as usize],
                    mask: vec![0; (max_batch * max_seq) as usize],
                    types: vec![0; (max_batch * max_seq) as usize],
                    acc: vec![0.0; m.hidden as usize],
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
                        if m.types.is_none()
                            && let Some(at) = types.iter().position(|&t| t != 0)
                        {
                            rows.state.written = false;
                            return Err(fail(
                                UNSUPPORTED_OPTION,
                                format!(
                                    "npu: token type {} in row {row} position {at}: the graph takes no token type input and computes type 0 only",
                                    types[at]
                                ),
                            ));
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
    bind(&m.ids, &s.ids_buf)?;
    bind(&m.mask, &s.mask_buf)?;
    if let (Some(t), Some(b)) = (&m.types, &s.types_buf) {
        bind(t, b)?;
    }
    bind(&m.output, &s.out_buf)?;

    let st = &rows.state;
    let (batch, seq) = (st.batch as usize, st.seq as usize);
    let (frame_batch, model_seq, hidden) = (m.frame_batch as usize, m.seq as usize, m.hidden as usize);
    let stride = s.max_seq as usize;
    let frames = batch.div_ceil(frame_batch);
    let mut h2d = 0u64;
    let mut d2h = 0u64;
    let result = s.result.ptr as *mut f32;

    for frame in 0..frames {
        let first = frame * frame_batch;
        let live = (batch - first).min(frame_batch);
        // The frame's inputs: each row's live tokens, zeros after, zeros
        // in rows past the batch (their mask is 0 and nothing reads
        // their output).
        unsafe {
            std::ptr::write_bytes(s.ids_buf.ptr as *mut u8, 0, s.ids_buf.bytes as usize);
            std::ptr::write_bytes(s.mask_buf.ptr as *mut u8, 0, s.mask_buf.bytes as usize);
            if let Some(b) = &s.types_buf {
                std::ptr::write_bytes(b.ptr as *mut u8, 0, b.bytes as usize);
            }
            for r in 0..live {
                let src = (first + r) * stride;
                let dst = r * model_seq;
                write_tokens(s.ids_buf.ptr, m.ids.precision, dst, &rows.ids[src..src + seq]);
                write_tokens(s.mask_buf.ptr, m.mask.precision, dst, &rows.mask[src..src + seq]);
                if let (Some(t), Some(b)) = (&m.types, &s.types_buf) {
                    write_tokens(b.ptr, t.precision, dst, &rows.types[src..src + seq]);
                }
            }
        }
        h2d += s.ids_buf.bytes + s.mask_buf.bytes + s.types_buf.as_ref().map_or(0, |b| b.bytes);

        ctx.execute(m.graph)?;
        d2h += s.out_buf.bytes;

        // Pool, cut and normalize each live row on the host.
        for r in 0..live {
            let row = first + r;
            let mask = &rows.mask[row * stride..row * stride + seq];
            let acc = &mut rows.acc;
            acc.fill(0.0);
            let read = |t: usize, h: usize| -> f64 {
                let at = (r * model_seq + t) * hidden + h;
                match m.output.precision {
                    ze::GRAPH_ARGUMENT_PRECISION_FP16 => {
                        half_to_f32(unsafe { (s.out_buf.ptr as *const u16).add(at).read() }) as f64
                    }
                    _ => (unsafe { (s.out_buf.ptr as *const f32).add(at).read() }) as f64,
                }
            };
            match st.pooling {
                TURBO_POOLING_CLS => {
                    for (h, a) in acc.iter_mut().enumerate() {
                        *a = read(0, h);
                    }
                }
                TURBO_POOLING_LAST => {
                    let last = mask.iter().rposition(|&v| v == 1).unwrap_or(0);
                    for (h, a) in acc.iter_mut().enumerate() {
                        *a = read(last, h);
                    }
                }
                _ => {
                    debug_assert_eq!(st.pooling, TURBO_POOLING_MEAN);
                    let mut live_tokens = 0f64;
                    for (t, &v) in mask.iter().enumerate() {
                        if v == 1 {
                            live_tokens += 1.0;
                            for (h, a) in acc.iter_mut().enumerate() {
                                *a += read(t, h);
                            }
                        }
                    }
                    for a in acc.iter_mut() {
                        *a /= live_tokens;
                    }
                }
            }
            let dim = st.output_dim as usize;
            let vector = &acc[..dim];
            let norm = match st.normalize {
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
    out.stage[TURBO_EMBED_STAGE_LOOKUP] = TURBO_STAGE_DEVICE;
    out.stage[TURBO_EMBED_STAGE_ENCODE] = TURBO_STAGE_DEVICE;
    out.stage[TURBO_EMBED_STAGE_POOL] = TURBO_STAGE_HOST;
    if rows.state.normalize == TURBO_NORMALIZE_L2 {
        out.stage[TURBO_EMBED_STAGE_NORMALIZE] = TURBO_STAGE_HOST;
    }
    out.stage[TURBO_EMBED_STAGE_DOWNLOAD] = TURBO_STAGE_DEVICE;
    Ok(())
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

    fn err_msg(r: Res<()>) -> String {
        match r {
            Ok(()) => panic!("expected a refusal"),
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
}
