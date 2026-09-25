//! The CPU backend: the host processor, through turbo_backend.h like any
//! other backend. It lists the device and its memory, holds the models
//! loaded on it, and runs embed sessions with the encoder in encoder.rs.

mod encoder;
mod kernels;
mod pool;

use std::alloc::Layout;
use std::ffi::{c_char, c_void};
use std::sync::{Mutex, OnceLock};

use crate::backend::{
    TURBO_BERT_EMBEDDING_TENSORS, TURBO_BERT_LAYER_TENSORS, TURBO_CAP_EXPERIMENTAL, TURBO_FAMILY_BERT,
    TURBO_FORMAT_SAFETENSORS, format_bit, refuse, refuse_field, turbo_backend, turbo_backend_embed_rows,
    turbo_backend_model, turbo_backend_run, turbo_backend_tensor,
};
use crate::status::{
    INVALID_ARGUMENT, INVALID_STATE, OUT_OF_MEMORY, UNSUPPORTED, UNSUPPORTED_OPTION, UNSUPPORTED_TASK,
};
use crate::{
    TURBO_DEVICE_CPU, TURBO_DTYPE_F16, TURBO_DTYPE_F32, TURBO_EMBED_STAGE_DOWNLOAD, TURBO_EMBED_STAGE_ENCODE,
    TURBO_EMBED_STAGE_LOOKUP, TURBO_EMBED_STAGE_NORMALIZE, TURBO_EMBED_STAGE_POOL, TURBO_EMBED_STAGE_UPLOAD,
    TURBO_HANDLE_HOST_PTR, TURBO_NORMALIZE_L2, TURBO_PLACE_DEVICE, TURBO_PLACE_HOST, TURBO_PRECISION_MODEL,
    TURBO_STAGE_HOST, TURBO_STAGE_UNUSED, TURBO_TASK_EMBED, turbo_buffer_desc, turbo_device_info, turbo_error,
    turbo_log_fn, turbo_native_handle, write_str,
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
        // A zero-size layout is not the allocator's to take; the core
        // never asks for one, and a session's output is never empty.
        debug_assert!(layout.size() > 0);
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
    if desc.family != TURBO_FAMILY_BERT {
        return unsafe { refuse(err, UNSUPPORTED, &format!("family {}: the cpu holds BERT encoders", desc.family)) };
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
// embed_write copied in, computes into that memory on those threads, and
// allocates nothing. Releasing the session stops and joins its threads. The CPU's memory is the host's: the rows embed_write
// copies into the session are its input, with no crossing to count, and
// the vectors are where the caller reads them. So UPLOAD and DOWNLOAD do
// not run here, and h2d_bytes and d2h_bytes are 0.

struct Session {
    encoder: encoder::Encoder,
    /// The threads `threads` gives, the caller's among them; see pool.rs.
    pool: pool::Pool,
    /// [max_batch, hidden] F32, the buffer handed out as the run's output.
    output: Box<Buffer>,
    /// Rows are written and not yet run.
    written: bool,
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
    let m = unsafe { &*(model as *const Model) };
    if task != TURBO_TASK_EMBED {
        return unsafe { refuse(err, UNSUPPORTED_TASK, &format!("task {task}: the cpu runs embed")) };
    }
    if precision == TURBO_PRECISION_MODEL && m.desc.dtype != TURBO_DTYPE_F32 {
        let stored = if m.desc.dtype == TURBO_DTYPE_F16 { "F16" } else { "BF16" };
        let msg = format!(
            "precision: MODEL computes in the weights' {stored}, and the cpu computes in F32 only; \
             EXACT and FASTEST compute this model in F32"
        );
        return unsafe { refuse_field(err, UNSUPPORTED_OPTION, 3, &msg) };
    }
    if max_seq > m.desc.max_positions {
        let msg = format!("max_seq {max_seq} is over the model's {} positions", m.desc.max_positions);
        return unsafe { refuse_field(err, UNSUPPORTED_OPTION, 2, &msg) };
    }
    let hidden = m.desc.hidden as usize;
    let bytes = max_batch as usize * hidden * 4;
    let output = Layout::from_size_align(bytes, ALIGN).ok().and_then(Buffer::alloc);
    let Some(output) = output else {
        return unsafe { refuse(err, OUT_OF_MEMORY, &format!("{bytes} bytes for the session's vectors")) };
    };
    let tensors = m.f32_tensors();
    let packed = match m.packed(&tensors) {
        Ok(p) => p,
        Err(bytes) => {
            return unsafe { refuse(err, OUT_OF_MEMORY, &format!("{bytes} bytes for the model's packed weights")) };
        }
    };
    let threads = match threads(std::env::var("TURBO_CPU_THREADS").ok().as_deref()) {
        Ok(n) => n,
        Err(msg) => return unsafe { refuse(err, INVALID_ARGUMENT, &msg) },
    };
    let pool = pool::Pool::new(threads);
    let encoder =
        match encoder::Encoder::new(&m.desc, tensors, packed, pool.threads(), max_batch as usize, max_seq as usize) {
            Ok(e) => e,
            Err(bytes) => {
                return unsafe { refuse(err, OUT_OF_MEMORY, &format!("{bytes} bytes of scratch for the session")) };
            }
        };
    let s = Session { encoder, pool, output: Box::new(output), written: false };
    unsafe {
        *compute_dtype = TURBO_DTYPE_F32;
        *out = Box::into_raw(Box::new(s)) as *mut c_void;
    }
    0
}

/// The threads a session runs on: TURBO_CPU_THREADS when set (docs/cpu.md),
/// else one per processor this process may run on, which
/// available_parallelism reads from its affinity mask and cgroup quota.
fn threads(var: Option<&str>) -> Result<usize, String> {
    match var {
        None => Ok(std::thread::available_parallelism().map_or(1, std::num::NonZero::get)),
        Some(v) => match v.trim().parse::<usize>() {
            Ok(n @ 1..=MAX_THREADS) => Ok(n),
            _ => Err(format!("TURBO_CPU_THREADS {v:?}: a count of threads from 1 to {MAX_THREADS}")),
        },
    }
}

/// The most threads TURBO_CPU_THREADS may ask for.
const MAX_THREADS: usize = 1024;

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
    let span = (r.batch as usize - 1) * r.row_stride as usize + r.seq as usize;
    let (ids, mask) = unsafe { (std::slice::from_raw_parts(r.ids, span), std::slice::from_raw_parts(r.mask, span)) };
    let types = (!r.types.is_null()).then(|| unsafe { std::slice::from_raw_parts(r.types, span) });
    s.encoder.write(r, ids, mask, types);
    s.written = true;
    0
}

unsafe extern "C" fn session_run(session: *mut c_void, out: *mut turbo_backend_run, err: *mut turbo_error) -> i32 {
    let (s, out) = unsafe { (&mut *(session as *mut Session), &mut *out) };
    if !std::mem::take(&mut s.written) {
        return unsafe { refuse(err, INVALID_STATE, "the cpu session has no rows written since its last run") };
    }
    let floats = unsafe { std::slice::from_raw_parts_mut(s.output.ptr as *mut f32, s.output_len()) };
    s.encoder.run(&mut s.pool, floats);
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
    st[TURBO_EMBED_STAGE_ENCODE] = TURBO_STAGE_HOST;
    st[TURBO_EMBED_STAGE_POOL] = TURBO_STAGE_HOST;
    st[TURBO_EMBED_STAGE_NORMALIZE] =
        if s.encoder.normalize() == TURBO_NORMALIZE_L2 { TURBO_STAGE_HOST } else { TURBO_STAGE_UNUSED };
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

    #[cfg(not(target_os = "linux"))]
    fn read() -> Host {
        Host::default()
    }
}

#[cfg(test)]
mod tests {
    use super::threads;

    #[test]
    fn turbo_cpu_threads_sets_the_count_or_is_refused() {
        let all = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
        assert_eq!(threads(None), Ok(all));
        assert_eq!(threads(Some("16")), Ok(16));
        assert_eq!(threads(Some(" 3 ")), Ok(3));
        for bad in ["0", "-1", "", "sixteen", "1025", "2.5"] {
            assert!(threads(Some(bad)).unwrap_err().starts_with("TURBO_CPU_THREADS"), "{bad}");
        }
    }
}
