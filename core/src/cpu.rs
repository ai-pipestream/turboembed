//! The CPU backend: the host processor, through turbo_backend.h like any
//! other backend. It lists the device and its memory, holds the models
//! loaded on it, and runs embed sessions with the encoder in encoder.rs,
//! or for a static model the table in table.rs.

mod encoder;
mod kernels;
pub(crate) mod pool;
mod table;

use std::alloc::Layout;
use std::ffi::{c_char, c_void};
use std::sync::{Mutex, OnceLock};

use crate::backend::{
    TURBO_BERT_EMBEDDING_TENSORS, TURBO_BERT_LAYER_TENSORS, TURBO_CAP_EXPERIMENTAL, TURBO_FAMILY_BERT,
    TURBO_FAMILY_ROBERTA, TURBO_FAMILY_STATIC, TURBO_FORMAT_SAFETENSORS, format_bit, refuse, refuse_field,
    turbo_backend, turbo_backend_embed_rows, turbo_backend_model, turbo_backend_run, turbo_backend_session_desc,
    turbo_backend_tensor, turbo_backend_tuning,
};
use crate::status::{
    INVALID_ARGUMENT, INVALID_STATE, INVALID_STRUCT_SIZE, OUT_OF_MEMORY, UNSUPPORTED, UNSUPPORTED_OPTION,
    UNSUPPORTED_TASK,
};
use crate::{
    TURBO_DEVICE_CPU, TURBO_DTYPE_F16, TURBO_DTYPE_F32, TURBO_DTYPE_I8, TURBO_EMBED_STAGE_DOWNLOAD,
    TURBO_EMBED_STAGE_ENCODE, TURBO_EMBED_STAGE_LOOKUP, TURBO_EMBED_STAGE_NORMALIZE, TURBO_EMBED_STAGE_POOL,
    TURBO_EMBED_STAGE_UPLOAD, TURBO_HANDLE_HOST_PTR, TURBO_NORMALIZE_L2, TURBO_PLACE_DEVICE, TURBO_PLACE_HOST,
    TURBO_PRECISION_FASTEST, TURBO_PRECISION_MODEL, TURBO_STAGE_FUSED, TURBO_STAGE_HOST, TURBO_STAGE_UNUSED,
    TURBO_TASK_EMBED, turbo_buffer_desc, turbo_device_info, turbo_error, turbo_log_fn, turbo_native_handle,
    turbo_token_range, write_str,
};

pub static BACKEND: turbo_backend = turbo_backend {
    struct_size: size_of::<turbo_backend>() as u32,
    reserved: 0,
    name: c"cpu".as_ptr(),
    runtime_version: c"".as_ptr(),
    device_count,
    device_info,
    capability,
    context_create: Some(context_create),
    context_release: Some(context_release),
    buffer_alloc: Some(buffer_alloc),
    buffer_import: Some(buffer_import),
    buffer_release: Some(buffer_release),
    buffer_export: Some(buffer_export),
    model_load: Some(model_load),
    model_release: Some(model_release),
    session_create: Some(session_create),
    session_release: Some(session_release),
    embed_write: Some(embed_write),
    session_run: Some(session_run),
    // Every buffer here has a host address, which the core reads itself.
    buffer_read: None,
    formats: format_bit(TURBO_FORMAT_SAFETENSORS),
    reserved2: 0,
    session_create_tuned: None,
    session_open: Some(session_open),
    embed_write_ranges: Some(embed_write_ranges),
};

unsafe extern "C" fn device_count(out: *mut u32, _err: *mut turbo_error) -> i32 {
    unsafe { *out = 1 };
    0
}

/// The host is the one device this backend lists.
unsafe fn only(ordinal: u32, err: *mut turbo_error) -> Option<i32> {
    (ordinal != 0).then(|| unsafe { refuse(err, INVALID_ARGUMENT, &format!("cpu device {ordinal}: only 0 is listed")) })
}

unsafe extern "C" fn device_info(ordinal: u32, out: *mut turbo_device_info, err: *mut turbo_error) -> i32 {
    if let Some(rc) = unsafe { only(ordinal, err) } {
        return rc;
    }
    let out = unsafe { &mut *out };
    let host = Host::read();
    out.kind = TURBO_DEVICE_CPU;
    out.ordinal = 0;
    out.unified_memory = 1;
    out.memory_total = host.memory_total;
    out.memory_free = host.memory_free;
    // The instruction set, not the processor: arch[32] cannot hold a model
    // name. A CPU benchmark record is filed under arch and name together,
    // so a record for one x86_64 processor backs no other.
    write_str(&mut out.arch, std::env::consts::ARCH);
    write_str(&mut out.name, &host.name);
    write_str(&mut out.vendor, &host.vendor);
    write_str(&mut out.runtime_version, "");
    write_str(&mut out.driver_version, "");
    0
}

/// Bits of turbo_embed_options the run honors: normalize (4), pooling (5)
/// and output_dim (6), every value of each.
const EMBED_HONORED: u32 = 0b111000;

/// Embed runs at every precision, in F32: the one dtype the encoder
/// computes in. FASTEST is F32 too, and says so. A model whose weights are
/// F16 or BF16 computes in F32 at EXACT and FASTEST, from a converted copy
/// (see session_create); at MODEL it would compute in its storage dtype,
/// which the encoder does not, so that session is refused. A cell cannot
/// say that, since it is not for one model; turbo_session_get_info does.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn capability(
    ordinal: u32,
    _task: u32,
    _precision: u32,
    status: *mut u32,
    dtype: *mut u32,
    options_honored: *mut u32,
    reason: *mut c_char,
    reason_len: u32,
    err: *mut turbo_error,
) -> i32 {
    if let Some(rc) = unsafe { only(ordinal, err) } {
        return rc;
    }
    unsafe {
        *status = TURBO_CAP_EXPERIMENTAL;
        *dtype = TURBO_DTYPE_F32;
        *options_honored = EMBED_HONORED;
        let r = std::slice::from_raw_parts_mut(reason, reason_len as usize);
        write_str(r, "");
    }
    0
}

// ---- Contexts and buffers ----------------------------------------------
//
// The CPU has no memory apart from the host's. HOST, PINNED and SHARED are
// all pageable host memory here: there is no second device to pin for or
// to share with. DEVICE is refused rather than read as host memory, so a
// caller that asked for memory off the host is told it did not get it.

/// Every allocation starts on a 64-byte boundary: a cache line, and the
/// width of an AVX-512 register.
const ALIGN: usize = 64;

/// The host keeps nothing per context and has nothing to warn about.
struct Context;

struct Buffer {
    ptr: *mut u8,
    /// The allocation to free; None for memory the caller owns.
    layout: Option<Layout>,
}

unsafe extern "C" fn context_create(
    ordinal: u32,
    _log: turbo_log_fn,
    _log_user_data: *mut c_void,
    out: *mut *mut c_void,
    err: *mut turbo_error,
) -> i32 {
    if let Some(rc) = unsafe { only(ordinal, err) } {
        return rc;
    }
    unsafe { *out = Box::into_raw(Box::new(Context)) as *mut c_void };
    0
}

unsafe extern "C" fn context_release(ctx: *mut c_void) {
    drop(unsafe { Box::from_raw(ctx as *mut Context) });
}

/// DEVICE placement, refused: see above.
unsafe fn host_only(desc: &turbo_buffer_desc, err: *mut turbo_error) -> Option<i32> {
    (desc.placement == TURBO_PLACE_DEVICE).then(|| unsafe {
        refuse(err, UNSUPPORTED, "placement: TURBO_PLACE_DEVICE: the cpu has no memory apart from the host's")
    })
}

unsafe fn give(buf: Buffer, out: *mut *mut c_void, host: *mut *mut c_void) -> i32 {
    unsafe {
        *host = buf.ptr as *mut c_void;
        *out = Box::into_raw(Box::new(buf)) as *mut c_void;
    }
    0
}

unsafe extern "C" fn buffer_alloc(
    _ctx: *mut c_void,
    desc: *const turbo_buffer_desc,
    out: *mut *mut c_void,
    host: *mut *mut c_void,
    err: *mut turbo_error,
) -> i32 {
    let desc = unsafe { &*desc };
    if let Some(rc) = unsafe { host_only(desc, err) } {
        return rc;
    }
    let too_big =
        || unsafe { refuse(err, OUT_OF_MEMORY, &format!("{} bytes is more than the host can address", desc.bytes)) };
    let Ok(bytes) = usize::try_from(desc.bytes) else {
        return too_big();
    };
    // The allocator may not be asked for no bytes; the core never asks.
    if bytes == 0 {
        return unsafe { refuse(err, INVALID_ARGUMENT, "bytes: 0: a buffer holds at least one byte") };
    }
    let Ok(layout) = Layout::from_size_align(bytes, ALIGN) else {
        return too_big();
    };
    // Not zeroed: turbo.h promises no contents, and the caller writes them.
    match Buffer::alloc(layout) {
        Some(buf) => unsafe { give(buf, out, host) },
        None => unsafe { refuse(err, OUT_OF_MEMORY, &format!("{bytes} bytes of host memory")) },
    }
}

impl Buffer {
    /// Host memory of `layout`, not zeroed: turbo.h promises no contents.
    fn alloc(layout: Layout) -> Option<Buffer> {
        // A zero-size layout is not the allocator's to take: buffer_alloc
        // refuses one, and a session's output is never empty.
        if layout.size() == 0 {
            return None;
        }
        let ptr = unsafe { std::alloc::alloc(layout) };
        (!ptr.is_null()).then_some(Buffer { ptr, layout: Some(layout) })
    }
}

/// The caller's own pointer plus offset. Nothing is copied.
unsafe extern "C" fn buffer_import(
    _ctx: *mut c_void,
    desc: *const turbo_buffer_desc,
    handle: *const turbo_native_handle,
    out: *mut *mut c_void,
    host: *mut *mut c_void,
    err: *mut turbo_error,
) -> i32 {
    let (desc, h) = unsafe { (&*desc, &*handle) };
    if let Some(rc) = unsafe { host_only(desc, err) } {
        return rc;
    }
    if h.kind != TURBO_HANDLE_HOST_PTR {
        let m = format!("kind: {} is not TURBO_HANDLE_HOST_PTR, the one kind the cpu imports", h.kind);
        return unsafe { refuse(err, UNSUPPORTED, &m) };
    }
    if h.handle == 0 {
        return unsafe { refuse(err, INVALID_ARGUMENT, "handle: a NULL host pointer") };
    }
    let end = h.handle.checked_add(h.offset).and_then(|a| a.checked_add(desc.bytes));
    if end.is_none_or(|e| usize::try_from(e).is_err()) {
        let m =
            format!("handle {:#x} + offset {} + {} bytes is past the address space", h.handle, h.offset, desc.bytes);
        return unsafe { refuse(err, INVALID_ARGUMENT, &m) };
    }
    let ptr = (h.handle as usize as *mut u8).wrapping_add(h.offset as usize);
    unsafe { give(Buffer { ptr, layout: None }, out, host) }
}

unsafe extern "C" fn buffer_release(buf: *mut c_void) {
    drop(unsafe { Box::from_raw(buf as *mut Buffer) });
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if let Some(layout) = self.layout {
            unsafe { std::alloc::dealloc(self.ptr, layout) };
        }
    }
}

/// The buffer's own host address; no copy.
unsafe extern "C" fn buffer_export(
    buf: *mut c_void,
    kind: u32,
    out: *mut turbo_native_handle,
    err: *mut turbo_error,
) -> i32 {
    if kind != TURBO_HANDLE_HOST_PTR {
        let m = format!("kind: {kind} is not TURBO_HANDLE_HOST_PTR, the one kind the cpu exports");
        return unsafe { refuse(err, UNSUPPORTED, &m) };
    }
    let (buf, out) = unsafe { (&*(buf as *const Buffer), &mut *out) };
    out.kind = TURBO_HANDLE_HOST_PTR;
    out.handle = buf.ptr as usize as u64;
    out.aux = 0;
    out.offset = 0;
    0
}

// ---- Models --------------------------------------------------------------
//
// The weights stay where the core read them: its one verified host copy of
// each weights file, which it keeps unchanged until model_release returns.
// The CPU reads them in place. A model here is the architecture and a table
// of pointers into those bytes. F16 and BF16 weights are widened once, for
// the sessions that compute in F32. The linear layers' weight matrices are
// also copied once into the panel layout the matrix kernel reads
// (kernels.rs): for all-MiniLM-L6-v2, 42.5 MB beside its 90.9 MB file.

struct Model {
    desc: turbo_backend_model,
    /// Each tensor as the core described it, its name dropped: the name is
    /// valid only for the load.
    tensors: Vec<turbo_backend_tensor>,
    /// For F16 or BF16 weights, each tensor converted to F32: turbo.h's
    /// one resident copy per compute dtype. The first session that
    /// computes in F32 makes it, every later one shares it, and it goes
    /// with the model. It is made outside any run, so no result counts it.
    f32: OnceLock<Vec<Vec<f32>>>,
    /// The linear layers packed for this processor's matrix kernel, made
    /// by the first session and shared by every later one, like `f32`.
    packed: OnceLock<kernels::Packed>,
    /// Held while `packed` is made. A failed pack leaves it unset, and the
    /// next session tries again.
    packing: Mutex<()>,
    /// A static model's table, its weights and mapping widened, made by the
    /// first session and shared by every later one.
    table: OnceLock<std::sync::Arc<table::Table>>,
    /// The table in I8 for FASTEST sessions, made by the first, or why
    /// it cannot be.
    quantized: OnceLock<Result<std::sync::Arc<table::Quantized>, String>>,
}

// The tensors point into the core's weights, which it keeps unchanged
// until model_release; the copy is written once, then only read.
unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {
    /// Each tensor as F32 values, in TURBO_BERT_* order.
    fn f32_tensors(&self) -> Vec<&[f32]> {
        if self.desc.dtype == TURBO_DTYPE_F32 {
            // The loader placed each tensor at a multiple of its element
            // size in a 64-byte aligned file.
            return self
                .tensors
                .iter()
                .map(|t| unsafe { std::slice::from_raw_parts(t.data as *const f32, t.bytes as usize / 4) })
                .collect();
        }
        let copy = self.f32.get_or_init(|| {
            self.tensors
                .iter()
                .map(|t| {
                    let h = unsafe { std::slice::from_raw_parts(t.data as *const u16, t.bytes as usize / 2) };
                    let widen = if self.desc.dtype == TURBO_DTYPE_F16 { f16_to_f32 } else { bf16_to_f32 };
                    h.iter().map(|&v| widen(v)).collect()
                })
                .collect()
        });
        copy.iter().map(Vec::as_slice).collect()
    }

    /// The packed linear layers, made on first use from `tensors`, which
    /// are f32_tensors(). Err is the bytes that could not be allocated.
    fn packed(&self, tensors: &[&[f32]]) -> Result<&kernels::Packed, usize> {
        if let Some(p) = self.packed.get() {
            return Ok(p);
        }
        // Sessions made at once wait here, so the weights are packed once.
        let _one = self.packing.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(p) = self.packed.get() {
            return Ok(p);
        }
        let layer = |l: usize, r: usize| {
            tensors[(TURBO_BERT_EMBEDDING_TENSORS + l as u32 * TURBO_BERT_LAYER_TENSORS) as usize + r]
        };
        let p = kernels::Packed::new(&self.desc, layer, kernels::Isa::detect())?;
        Ok(self.packed.get_or_init(|| p))
    }
}

/// An IEEE half to the single it names exactly.
fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = (h >> 10) & 0x1f;
    let man = (h & 0x3ff) as u32;
    match exp {
        // Zero and subnormals: man * 2^-24, exact in F32.
        0 => sign * man as f32 * f32::from_bits(0x3380_0000),
        0x1f => f32::from_bits(((h as u32 & 0x8000) << 16) | 0x7f80_0000 | (man << 13)),
        e => f32::from_bits(((h as u32 & 0x8000) << 16) | ((e as u32 + 112) << 23) | (man << 13)),
    }
}

/// A bfloat16 is the top half of the single it names.
fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

unsafe extern "C" fn model_load(
    _ctx: *mut c_void,
    desc: *const turbo_backend_model,
    out: *mut *mut c_void,
    err: *mut turbo_error,
) -> i32 {
    let desc = unsafe { *desc };
    if !matches!(desc.family, TURBO_FAMILY_BERT | TURBO_FAMILY_ROBERTA | TURBO_FAMILY_STATIC) {
        let msg = format!("family {}: the cpu holds BERT encoders and static models", desc.family);
        return unsafe { refuse(err, UNSUPPORTED, &msg) };
    }
    let tensors = unsafe { std::slice::from_raw_parts(desc.tensors, desc.tensor_count as usize) }
        .iter()
        .map(|t| turbo_backend_tensor { name: std::ptr::null(), ..*t })
        .collect();
    let desc = turbo_backend_model { tensors: std::ptr::null(), ..desc };
    unsafe {
        *out = Box::into_raw(Box::new(Model {
            desc,
            tensors,
            f32: OnceLock::new(),
            packed: OnceLock::new(),
            packing: Mutex::new(()),
            table: OnceLock::new(),
            quantized: OnceLock::new(),
        })) as *mut c_void
    };
    0
}

unsafe extern "C" fn model_release(model: *mut c_void) {
    drop(unsafe { Box::from_raw(model as *mut Model) });
}

/// Where each tensor of a model this backend loaded is read from, in the
/// order it was handed them.
///
/// # Safety
/// `model` is one model_load returned and model_release has not taken.
#[cfg(feature = "internals")]
pub(crate) unsafe fn tensor_data(model: *mut c_void) -> Vec<*const c_void> {
    let m = unsafe { &*(model as *const Model) };
    debug_assert_eq!(m.desc.tensor_count as usize, m.tensors.len());
    m.tensors.iter().map(|t| t.data).collect()
}

/// Where the F32 copy of each tensor of a model this backend loaded is,
/// if one was made.
///
/// # Safety
/// As for tensor_data.
#[cfg(feature = "internals")]
pub(crate) unsafe fn converted_data(model: *mut c_void) -> Option<Vec<*const c_void>> {
    let m = unsafe { &*(model as *const Model) };
    m.f32.get().map(|c| c.iter().map(|t| t.as_ptr() as *const c_void).collect())
}

// ---- Sessions -------------------------------------------------------------
//
// A session is an encoder sized for its largest batch, the threads it runs
// on, and the buffer its vectors are written to. Every byte they need is
// allocated here, and every thread started; a run reads the rows
// embed_write or embed_write_ranges copied in, computes into that memory
// on those threads, and allocates nothing. Releasing the session stops
// and joins its threads. The CPU's memory is the host's: the rows copied
// into the session are its input, with no crossing to count, and the
// vectors are where the caller reads them. So UPLOAD and DOWNLOAD do not
// run here, and h2d_bytes and d2h_bytes are 0.

struct Session {
    work: Work,
    /// The threads `threads` gives, the caller's among them; see pool.rs.
    pool: pool::Pool,
    /// [max(max_batch, max_ranges), hidden] F32, the buffer handed out as
    /// the run's output: a row's vectors after embed_write, a range's
    /// after embed_write_ranges, each output_dim values, packed.
    output: Box<Buffer>,
    /// The most ranges one embed_write_ranges may hold.
    max_ranges: usize,
    /// Rows are written and not yet run.
    written: bool,
}

/// What a session is made for: session_open's description, or
/// session_create's arguments with no thread count and max_batch ranges.
struct Shape {
    task: u32,
    max_batch: u32,
    max_seq: u32,
    precision: u32,
    /// turbo_session_desc.cpu_threads: 0 leaves the count to `threads`.
    cpu_threads: u32,
    max_ranges: u32,
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn session_create(
    model: *mut c_void,
    task: u32,
    max_batch: u32,
    max_seq: u32,
    precision: u32,
    compute_dtype: *mut u32,
    out: *mut *mut c_void,
    err: *mut turbo_error,
) -> i32 {
    let shape = Shape { task, max_batch, max_seq, precision, cpu_threads: 0, max_ranges: max_batch };
    unsafe { open(model, &shape, compute_dtype, out, err) }
}

/// The session's whole description. The CPU has one path, so `tuning` is
/// left as the core set it.
unsafe extern "C" fn session_open(
    model: *mut c_void,
    desc: *const turbo_backend_session_desc,
    _tuning: *mut turbo_backend_tuning,
    compute_dtype: *mut u32,
    out: *mut *mut c_void,
    err: *mut turbo_error,
) -> i32 {
    let d = unsafe { &*desc };
    let need = std::mem::offset_of!(turbo_backend_session_desc, max_ranges) + size_of::<u32>();
    if (d.struct_size as usize) < need {
        let msg = format!(
            "turbo_backend_session_desc of {} bytes: the cpu reads it through max_ranges, {need} bytes",
            d.struct_size
        );
        return unsafe { refuse(err, INVALID_STRUCT_SIZE, &msg) };
    }
    let shape = Shape {
        task: d.task,
        max_batch: d.max_batch,
        max_seq: d.max_seq,
        precision: d.precision,
        cpu_threads: d.cpu_threads,
        max_ranges: d.max_ranges,
    };
    unsafe { open(model, &shape, compute_dtype, out, err) }
}

/// A session of `shape` on `model`, for session_create and session_open.
unsafe fn open(
    model: *mut c_void,
    shape: &Shape,
    compute_dtype: *mut u32,
    out: *mut *mut c_void,
    err: *mut turbo_error,
) -> i32 {
    let m = unsafe { &*(model as *const Model) };
    let Shape { task, max_batch, max_seq, precision, cpu_threads, max_ranges } = *shape;
    if task != TURBO_TASK_EMBED {
        return unsafe { refuse(err, UNSUPPORTED_TASK, &format!("task {task}: the cpu runs embed")) };
    }
    let threads = match threads(cpu_threads, std::env::var("TURBO_CPU_THREADS").ok().as_deref()) {
        Ok(n) => n,
        Err(refusal) => return unsafe { refusal.refuse(err) },
    };
    let hidden = m.desc.hidden as usize;
    // The row path and the range path write their vectors to this one
    // buffer, so it holds the larger of the two counts.
    let vectors = max_batch.max(max_ranges) as usize;
    let bytes = vectors.saturating_mul(hidden).saturating_mul(4);
    let output = Layout::from_size_align(bytes, ALIGN).ok().and_then(Buffer::alloc);
    let Some(output) = output else {
        return unsafe { refuse(err, OUT_OF_MEMORY, &format!("{bytes} bytes for the session's vectors")) };
    };
    let max_ranges = max_ranges as usize;
    if m.desc.family == TURBO_FAMILY_STATIC {
        let quantize = match static_table(precision, std::env::var("TURBO_CPU_STATIC_TABLE").ok().as_deref()) {
            Ok(q) => q,
            Err(msg) => return unsafe { refuse(err, INVALID_ARGUMENT, &msg) },
        };
        let huge = match static_pages(std::env::var("TURBO_CPU_STATIC_PAGES").ok().as_deref()) {
            Ok(h) => h,
            Err(msg) => return unsafe { refuse(err, INVALID_ARGUMENT, &msg) },
        };
        let work = match static_session(m, quantize, huge, shape, err) {
            Ok(w) => w,
            Err(rc) => return rc,
        };
        let s = Session {
            work,
            pool: pool::Pool::with_spin(threads, pool::SPIN_BATCH),
            output: Box::new(output),
            max_ranges,
            written: false,
        };
        unsafe {
            *compute_dtype = if quantize { TURBO_DTYPE_I8 } else { TURBO_DTYPE_F32 };
            *out = Box::into_raw(Box::new(s)) as *mut c_void;
        }
        return 0;
    }
    if precision == TURBO_PRECISION_MODEL && m.desc.dtype != TURBO_DTYPE_F32 {
        let stored = if m.desc.dtype == TURBO_DTYPE_F16 { "F16" } else { "BF16" };
        let msg = format!(
            "precision: MODEL computes in the weights' {stored}, and the cpu computes in F32 only; \
             EXACT and FASTEST compute this model in F32"
        );
        return unsafe { refuse_field(err, UNSUPPORTED_OPTION, 3, &msg) };
    }
    if max_seq as u64 + m.desc.position_offset as u64 > m.desc.max_positions as u64 {
        let msg = format!(
            "max_seq {max_seq} from position {} is over the model's {} positions",
            m.desc.position_offset, m.desc.max_positions
        );
        return unsafe { refuse_field(err, UNSUPPORTED_OPTION, 2, &msg) };
    }
    let tensors = m.f32_tensors();
    let packed = match m.packed(&tensors) {
        Ok(p) => p,
        Err(bytes) => {
            return unsafe { refuse(err, OUT_OF_MEMORY, &format!("{bytes} bytes for the model's packed weights")) };
        }
    };
    let pool = pool::Pool::new(threads);
    let size =
        encoder::Size { threads: pool.threads(), max_batch: max_batch as usize, max_seq: max_seq as usize, max_ranges };
    let encoder = match encoder::Encoder::new(&m.desc, tensors, packed, size) {
        Ok(e) => e,
        Err(bytes) => {
            return unsafe { refuse(err, OUT_OF_MEMORY, &format!("{bytes} bytes of scratch for the session")) };
        }
    };
    let s =
        Session { work: Work::Encoder(Box::new(encoder)), pool, output: Box::new(output), max_ranges, written: false };
    unsafe {
        *compute_dtype = TURBO_DTYPE_F32;
        *out = Box::into_raw(Box::new(s)) as *mut c_void;
    }
    0
}

/// What a session computes with: an encoder, or a static model's table.
enum Work {
    Encoder(Box<encoder::Encoder>),
    Static(table::Static),
}

/// A static model's session: no weights to convert or pack, since the
/// table is read in place in its stored dtype and summed in F32 at every
/// precision.
/// `huge`: the model's first session makes copies of its tables in huge
/// pages, which every later session reads.
fn static_session(m: &Model, quantize: bool, huge: bool, shape: &Shape, err: *mut turbo_error) -> Result<Work, i32> {
    let t = m.table.get_or_init(|| std::sync::Arc::new(table::Table::new(&m.tensors, huge))).clone();
    let q = match quantize.then(|| m.quantized.get_or_init(|| table::Quantized::new(&t, huge).map(std::sync::Arc::new)))
    {
        None => None,
        Some(Ok(q)) => Some(q.clone()),
        // The I8 table is FASTEST's, or every precision's under
        // TURBO_CPU_STATIC_TABLE=i8.
        Some(Err(why)) if shape.precision == TURBO_PRECISION_FASTEST => {
            let msg = format!("precision: FASTEST sums an I8 copy of the table, and {why}; EXACT sums the stored table");
            return Err(unsafe { refuse_field(err, UNSUPPORTED_OPTION, 3, &msg) });
        }
        Some(Err(why)) => {
            let msg = format!("TURBO_CPU_STATIC_TABLE=i8 sums an I8 copy of the table, and {why}");
            return Err(unsafe { refuse(err, INVALID_ARGUMENT, &msg) });
        }
    };
    let (max_batch, max_seq, max_ranges) =
        (shape.max_batch as usize, shape.max_seq as usize, shape.max_ranges as usize);
    table::Static::new(t, q, kernels::Isa::detect(), max_batch, max_seq, max_ranges)
        .map(Work::Static)
        .map_err(|bytes| unsafe { refuse(err, OUT_OF_MEMORY, &format!("{bytes} bytes for the session's rows")) })
}

/// Whether a static model's session sums the I8 table: at FASTEST, unless
/// TURBO_CPU_STATIC_TABLE says `stored` (docs/cpu.md); `i8` makes every
/// precision sum it.
fn static_table(precision: u32, var: Option<&str>) -> Result<bool, String> {
    match var.map(str::trim) {
        None | Some("") => Ok(precision == TURBO_PRECISION_FASTEST),
        Some("stored") => Ok(false),
        Some("i8") => Ok(true),
        Some(v) => Err(format!("TURBO_CPU_STATIC_TABLE={v:?}: stored or i8")),
    }
}

/// Whether a static model's tables are read from copies in huge pages:
/// TURBO_CPU_STATIC_PAGES `huge` (docs/cpu.md); unset or `mapped`, the
/// stored table is read where the bundle's file is mapped.
fn static_pages(var: Option<&str>) -> Result<bool, String> {
    match var.map(str::trim) {
        None | Some("") | Some("mapped") => Ok(false),
        Some("huge") => Ok(true),
        Some(v) => Err(format!("TURBO_CPU_STATIC_PAGES={v:?}: mapped or huge")),
    }
}

/// The threads a session runs on (docs/cpu.md): `field`, the session's
/// turbo_session_desc.cpu_threads, when it is not 0; else TURBO_CPU_THREADS,
/// `var`, when set; else one per processor this process may run on, which
/// available_parallelism reads from its affinity mask and cgroup quota.
fn threads(field: u32, var: Option<&str>) -> Result<usize, Refusal> {
    if field != 0 {
        return match field as usize {
            n @ 1..=MAX_THREADS => Ok(n),
            _ => Err(Refusal {
                code: UNSUPPORTED_OPTION,
                field: 6,
                message: format!("cpu_threads {field}: the count must be 1 to {MAX_THREADS}"),
            }),
        };
    }
    match var {
        None => Ok(std::thread::available_parallelism().map_or(1, std::num::NonZero::get)),
        Some(v) => match v.trim().parse::<usize>() {
            Ok(n @ 1..=MAX_THREADS) => Ok(n),
            _ => Err(Refusal {
                code: INVALID_ARGUMENT,
                field: 0,
                message: format!("TURBO_CPU_THREADS {v:?}: a count of threads from 1 to {MAX_THREADS}"),
            }),
        },
    }
}

/// The most threads a session may ask for, by either cpu_threads or
/// TURBO_CPU_THREADS.
const MAX_THREADS: usize = 1024;

/// Why a session is refused: the status, the turbo_session_desc field it
/// names (0 for none), and the message.
#[derive(Debug, PartialEq)]
struct Refusal {
    code: i32,
    field: u32,
    message: String,
}

impl Refusal {
    unsafe fn refuse(self, err: *mut turbo_error) -> i32 {
        unsafe { refuse_field(err, self.code, self.field, &self.message) }
    }
}

unsafe extern "C" fn session_release(session: *mut c_void) {
    drop(unsafe { Box::from_raw(session as *mut Session) });
}

/// The rows, copied into the session: see above.
unsafe extern "C" fn embed_write(
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
    }
    s.written = true;
    0
}

/// The rows and the ranges the run pools them by, copied into the
/// session. The core has checked each range, and that there are at most
/// the session's max_ranges.
unsafe extern "C" fn embed_write_ranges(
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
    }
    s.written = true;
    0
}

/// The ids, mask and types `r` points to, through its last row's last
/// column.
///
/// # Safety
/// The arrays hold what `r` says, for as long as the slices are used.
unsafe fn arrays<'a>(r: &turbo_backend_embed_rows) -> (&'a [i32], &'a [i32], Option<&'a [i32]>) {
    let span = (r.batch as usize - 1) * r.row_stride as usize + r.seq as usize;
    let (ids, mask) = unsafe { (std::slice::from_raw_parts(r.ids, span), std::slice::from_raw_parts(r.mask, span)) };
    let types = (!r.types.is_null()).then(|| unsafe { std::slice::from_raw_parts(r.types, span) });
    (ids, mask, types)
}

unsafe extern "C" fn session_run(session: *mut c_void, out: *mut turbo_backend_run, err: *mut turbo_error) -> i32 {
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
    };
    let encoded = matches!(s.work, Work::Encoder(_));
    out.placement = TURBO_PLACE_HOST;
    out.output = &*s.output as *const Buffer as *mut c_void;
    out.host = s.output.ptr as *mut c_void;
    out.h2d_bytes = 0;
    out.d2h_bytes = 0;
    // Nothing on this path allocates, on this thread or the pool's: the
    // encoder computes in the memory session_create gave it, on the
    // threads it started. tests/allocations.rs holds this to a counting
    // allocator.
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
    fn output_len(&self) -> usize {
        self.output.layout.map_or(0, |l| l.size() / 4)
    }
}

/// What the operating system says about the processor and memory. A value
/// it does not report is empty or 0, which turbo.h reads as unknown.
#[derive(Default)]
struct Host {
    name: String,
    vendor: String,
    memory_total: u64,
    memory_free: u64,
}

impl Host {
    #[cfg(target_os = "linux")]
    fn read() -> Host {
        let mut h = Host::default();
        let field = |text: &str, key: &str| {
            text.lines()
                .find_map(|l| l.split_once(':').filter(|(k, _)| k.trim() == key).map(|(_, v)| v.trim().to_owned()))
        };
        if let Ok(cpu) = std::fs::read_to_string("/proc/cpuinfo") {
            // x86 names the model; Arm cores often do not, so fall back to
            // the implementer and part the kernel reports.
            h.name = field(&cpu, "model name").or_else(|| field(&cpu, "Model")).unwrap_or_default();
            h.vendor = field(&cpu, "vendor_id").or_else(|| field(&cpu, "CPU implementer")).unwrap_or_default();
        }
        if let Ok(mem) = std::fs::read_to_string("/proc/meminfo") {
            let kib = |key| field(&mem, key).and_then(|v| v.trim_end_matches("kB").trim().parse::<u64>().ok());
            h.memory_total = kib("MemTotal").map_or(0, |k| k * 1024);
            h.memory_free = kib("MemAvailable").map_or(0, |k| k * 1024);
        }
        h
    }

    /// macOS names the processor and its memory through sysctl: the free
    /// pages it counts, which leave out what the kernel would give back
    /// on demand, so memory_free is a floor.
    #[cfg(target_os = "macos")]
    fn read() -> Host {
        unsafe extern "C" {
            fn sysctlbyname(
                name: *const c_char,
                oldp: *mut c_void,
                oldlenp: *mut usize,
                newp: *mut c_void,
                newlen: usize,
            ) -> i32;
        }
        let bytes = |name: &std::ffi::CStr| -> Option<Vec<u8>> {
            let mut len = 0usize;
            let null = std::ptr::null_mut();
            if unsafe { sysctlbyname(name.as_ptr(), null, &mut len, null, 0) } != 0 || len == 0 {
                return None;
            }
            let mut buf = vec![0u8; len];
            let rc = unsafe { sysctlbyname(name.as_ptr(), buf.as_mut_ptr() as *mut c_void, &mut len, null, 0) };
            (rc == 0).then(|| {
                buf.truncate(len);
                buf
            })
        };
        let text = |name| {
            bytes(name)
                .map(|b| String::from_utf8_lossy(&b).trim_end_matches('\0').trim().to_owned())
                .unwrap_or_default()
        };
        let number = |name| {
            bytes(name).map_or(0, |b| match b.len() {
                8 => u64::from_ne_bytes(b.try_into().unwrap()),
                4 => u32::from_ne_bytes(b.try_into().unwrap()) as u64,
                _ => 0,
            })
        };
        let name = text(c"machdep.cpu.brand_string");
        // "Apple M2", "Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz": the
        // vendor leads.
        let vendor = name.split([' ', '(']).next().unwrap_or("").to_owned();
        let page = number(c"hw.pagesize");
        let free = number(c"vm.page_free_count") + number(c"vm.page_speculative_count");
        Host { name, vendor, memory_total: number(c"hw.memsize"), memory_free: free * page }
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    fn read() -> Host {
        Host::default()
    }
}

#[cfg(test)]
mod tests {
    use super::threads;
    use crate::status::{INVALID_ARGUMENT, UNSUPPORTED_OPTION};

    #[test]
    fn turbo_cpu_threads_sets_the_count_or_is_refused() {
        let all = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
        assert_eq!(threads(0, None), Ok(all));
        assert_eq!(threads(0, Some("16")), Ok(16));
        assert_eq!(threads(0, Some(" 3 ")), Ok(3));
        for bad in ["0", "-1", "", "sixteen", "1025", "2.5"] {
            let r = threads(0, Some(bad)).unwrap_err();
            assert_eq!((r.code, r.field), (INVALID_ARGUMENT, 0), "{bad}");
            assert!(r.message.starts_with("TURBO_CPU_THREADS"), "{bad}");
        }
    }

    /// cpu_threads wins over the variable, which it does not read: a bad
    /// value there does not refuse a session that names its count.
    #[test]
    fn cpu_threads_sets_the_count_or_is_refused_naming_field_6() {
        for (n, var) in [(1, None), (5, Some("16")), (1024, Some("sixteen"))] {
            assert_eq!(threads(n, var), Ok(n as usize));
        }
        for bad in [1025, 4096, u32::MAX] {
            let r = threads(bad, Some("4")).unwrap_err();
            assert_eq!((r.code, r.field), (UNSUPPORTED_OPTION, 6), "{bad}");
            assert!(r.message.starts_with("cpu_threads"), "{bad}");
        }
    }
}

#[cfg(test)]
mod boundary_tests {
    use std::ffi::c_void;

    use crate::backend::{TURBO_FAMILY_STATIC, turbo_backend_embed_rows, turbo_backend_model, turbo_backend_run};
    use crate::backend::{turbo_backend_session_desc, turbo_backend_tensor};
    use crate::status::{INVALID_ARGUMENT, UNSUPPORTED_OPTION};
    use crate::{
        TURBO_DTYPE_F32, TURBO_NORMALIZE_NONE, TURBO_PLACE_HOST, TURBO_POOLING_MEAN, TURBO_PRECISION_EXACT,
        TURBO_PRECISION_FASTEST, TURBO_TASK_EMBED, turbo_buffer_desc, turbo_error,
    };

    use super::BACKEND;

    fn error() -> turbo_error {
        let mut e: turbo_error = unsafe { std::mem::zeroed() };
        e.struct_size = size_of::<turbo_error>() as u32;
        e
    }

    fn message(e: &turbo_error) -> String {
        crate::backend::cstr(&e.message)
    }

    const VOCAB: usize = 4;
    const DIM: usize = 8;

    /// A static model over `table`, [VOCAB, DIM] F32, with no weights or
    /// mapping, and a session of `precision` on it: the session, or the
    /// refusal's status and message. The model is released before return.
    fn session(table: &[f32], precision: u32) -> Result<Vec<f32>, (i32, u32, String)> {
        let none = turbo_backend_tensor {
            name: std::ptr::null(),
            data: std::ptr::null(),
            shape: [0; 2],
            ndim: 0,
            dtype: 0,
            bytes: 0,
        };
        let tensors = [
            turbo_backend_tensor {
                data: table.as_ptr() as *const c_void,
                shape: [VOCAB as u64, DIM as u64],
                ndim: 2,
                dtype: TURBO_DTYPE_F32,
                bytes: (table.len() * 4) as u64,
                ..none
            },
            none,
            none,
        ];
        let desc = turbo_backend_model {
            struct_size: size_of::<turbo_backend_model>() as u32,
            family: TURBO_FAMILY_STATIC,
            dtype: TURBO_DTYPE_F32,
            layers: 0,
            hidden: DIM as u32,
            heads: 0,
            intermediate: 0,
            vocab_size: VOCAB as u32,
            max_positions: 0,
            token_types: 1,
            layer_norm_eps: 0.0,
            tensor_count: 3,
            position_offset: 0,
            tensors: tensors.as_ptr(),
            format: crate::backend::TURBO_FORMAT_SAFETENSORS,
            graph_input: crate::backend::TURBO_INPUT_TOKEN_IDS,
            graph_output: crate::backend::TURBO_OUTPUT_HIDDEN_STATES,
            compute_dtype: 0,
            fixed_seq: 0,
            fixed_batch: 0,
            artifact: std::ptr::null(),
            artifact_bytes: 0,
            artifact2: std::ptr::null(),
            artifact2_bytes: 0,
        };
        let mut err = error();
        let mut model = std::ptr::null_mut();
        assert_eq!(unsafe { (BACKEND.model_load.unwrap())(std::ptr::null_mut(), &desc, &mut model, &mut err) }, 0);
        let sd = turbo_backend_session_desc {
            struct_size: size_of::<turbo_backend_session_desc>() as u32,
            task: TURBO_TASK_EMBED,
            max_batch: VOCAB as u32,
            max_seq: 1,
            precision,
            cpu_threads: 1,
            max_ranges: VOCAB as u32,
            reserved: 0,
        };
        let (mut dtype, mut s) = (0u32, std::ptr::null_mut());
        let open = BACKEND.session_open.unwrap();
        let rc = unsafe { open(model, &sd, std::ptr::null_mut(), &mut dtype, &mut s, &mut err) };
        let got = if rc != 0 {
            Err((rc, err.field, message(&err)))
        } else {
            // Each token alone in its own row.
            let (ids, mask): (Vec<i32>, Vec<i32>) = ((0..VOCAB as i32).collect(), vec![1; VOCAB]);
            let rows = turbo_backend_embed_rows {
                struct_size: size_of::<turbo_backend_embed_rows>() as u32,
                batch: VOCAB as u32,
                seq: 1,
                row_stride: 1,
                ids: ids.as_ptr(),
                mask: mask.as_ptr(),
                types: std::ptr::null(),
                pooling: TURBO_POOLING_MEAN,
                normalize: TURBO_NORMALIZE_NONE,
                output_dim: DIM as u32,
                reserved: 0,
            };
            assert_eq!(unsafe { (BACKEND.embed_write.unwrap())(s, &rows, &mut err) }, 0);
            let mut run: turbo_backend_run = unsafe { std::mem::zeroed() };
            assert_eq!(unsafe { (BACKEND.session_run.unwrap())(s, &mut run, &mut err) }, 0);
            assert_eq!(run.placement, TURBO_PLACE_HOST);
            let v = unsafe { std::slice::from_raw_parts(run.host as *const f32, VOCAB * DIM) }.to_vec();
            unsafe { (BACKEND.session_release.unwrap())(s) };
            Ok(v)
        };
        unsafe { (BACKEND.model_release.unwrap())(model) };
        got
    }

    /// A table whose row 1 holds a NaN and row 2 an infinity, the others
    /// finite. The stored table's vectors carry them: row 1's NaN in its
    /// column, row 2's infinity in its. The I8 table cannot hold either,
    /// so a FASTEST session on it is refused, naming the first such row,
    /// rather than summing rows that were quantized to something else.
    #[test]
    fn a_fastest_session_is_refused_for_a_table_with_values_that_are_not_finite() {
        let mut table: Vec<f32> = (0..VOCAB * DIM).map(|i| (i as f32 - 13.0) / 7.0).collect();
        table[DIM + 3] = f32::NAN;
        table[2 * DIM + 5] = f32::INFINITY;
        let exact = session(&table, TURBO_PRECISION_EXACT).expect("EXACT sums the stored table");
        assert!(exact[DIM + 3].is_nan() && exact[2 * DIM + 5] == f32::INFINITY);
        assert!(exact[..DIM].iter().chain(&exact[3 * DIM..]).all(|v| v.is_finite()));
        match session(&table, TURBO_PRECISION_FASTEST) {
            Err((code, field, m)) => {
                assert_eq!((code, field), (UNSUPPORTED_OPTION, 3), "{m}");
                assert!(m.contains("row 1") && m.contains("NaN"), "{m}");
            }
            Ok(v) => panic!(
                "FASTEST opened on a table it cannot hold, and gave row 1 {:?} and row 2 {:?}",
                &v[DIM..2 * DIM],
                &v[2 * DIM..3 * DIM]
            ),
        }
        // A finite table still opens at FASTEST.
        table[DIM + 3] = 0.5;
        table[2 * DIM + 5] = -0.5;
        assert!(session(&table, TURBO_PRECISION_FASTEST).is_ok());
    }

    /// A buffer of no bytes is refused: the allocator is not asked for a
    /// zero-size layout, which it may not be given.
    #[test]
    fn a_buffer_of_no_bytes_is_refused() {
        let mut desc: turbo_buffer_desc = unsafe { std::mem::zeroed() };
        desc.placement = TURBO_PLACE_HOST;
        desc.bytes = 0;
        let mut err = error();
        let (mut out, mut host) = (std::ptr::null_mut(), std::ptr::null_mut());
        let rc = unsafe { (BACKEND.buffer_alloc.unwrap())(std::ptr::null_mut(), &desc, &mut out, &mut host, &mut err) };
        assert_eq!(rc, INVALID_ARGUMENT, "{}", message(&err));
        assert!(out.is_null() && host.is_null());
        // One byte is allocated, on the 64-byte boundary.
        desc.bytes = 1;
        let rc = unsafe { (BACKEND.buffer_alloc.unwrap())(std::ptr::null_mut(), &desc, &mut out, &mut host, &mut err) };
        assert_eq!(rc, 0, "{}", message(&err));
        assert_eq!(host as usize % 64, 0);
        unsafe { (BACKEND.buffer_release.unwrap())(out) };
    }
}
