//! Sessions: everything a run needs, made when the session is.
//!
//! A session is an encoder sized for its largest batch, the threads it runs
//! on, and the buffer its vectors are written to. Every byte they need is
//! allocated here, and every thread started; a run reads the rows
//! embed_write or embed_write_ranges copied in, computes into that memory
//! on those threads, and allocates nothing. Releasing the session stops
//! and joins its threads. The CPU's memory is the host's: the rows copied
//! into the session are its input, with no crossing to count, and the
//! vectors are where the caller reads them. So UPLOAD and DOWNLOAD do not
//! run here, and h2d_bytes and d2h_bytes are 0.

use std::alloc::Layout;
use std::ffi::c_void;

use crate::backend::{TURBO_FAMILY_STATIC, refuse, refuse_field, turbo_backend_session_desc, turbo_backend_tuning};
use crate::status::{INVALID_ARGUMENT, INVALID_STRUCT_SIZE, OUT_OF_MEMORY, UNSUPPORTED_OPTION, UNSUPPORTED_TASK};
use crate::{
    TURBO_DTYPE_F16, TURBO_DTYPE_F32, TURBO_DTYPE_I8, TURBO_PRECISION_FASTEST, TURBO_PRECISION_MODEL, TURBO_TASK_EMBED,
    turbo_error,
};

use super::buffers::*;
use super::model::*;
use super::{encoder, kernels, pool, table};

pub(super) struct Session {
    pub(super) work: Work,
    /// The threads `threads` gives, the caller's among them; see pool.rs.
    pub(super) pool: pool::Pool,
    /// [max(max_batch, max_ranges), hidden] F32, the buffer handed out as
    /// the run's output: a row's vectors after embed_write, a range's
    /// after embed_write_ranges, each output_dim values, packed.
    pub(super) output: Box<Buffer>,
    /// The most ranges one embed_write_ranges may hold.
    pub(super) max_ranges: usize,
    /// Rows are written and not yet run.
    pub(super) written: bool,
}

/// What a session is made for: session_open's description, or
/// session_create's arguments with no thread count and max_batch ranges.
pub(super) struct Shape {
    pub(super) task: u32,
    pub(super) max_batch: u32,
    pub(super) max_seq: u32,
    pub(super) precision: u32,
    /// turbo_session_desc.cpu_threads: 0 leaves the count to `threads`.
    pub(super) cpu_threads: u32,
    pub(super) max_ranges: u32,
}

#[allow(clippy::too_many_arguments)]
pub(super) unsafe extern "C" fn session_create(
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
pub(super) unsafe extern "C" fn session_open(
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
pub(super) unsafe fn open(
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
pub(super) enum Work {
    Encoder(Box<encoder::Encoder>),
    Static(table::Static),
}

/// A static model's session: no weights to convert or pack, since the
/// table is read in place in its stored dtype and summed in F32 at every
/// precision.
/// `huge`: the model's first session makes copies of its tables in huge
/// pages, which every later session reads.
pub(super) fn static_session(
    m: &Model,
    quantize: bool,
    huge: bool,
    shape: &Shape,
    err: *mut turbo_error,
) -> Result<Work, i32> {
    let t = m.table.get_or_init(|| std::sync::Arc::new(table::Table::new(&m.tensors, huge))).clone();
    let q = match quantize.then(|| m.quantized.get_or_init(|| table::Quantized::new(&t, huge).map(std::sync::Arc::new)))
    {
        None => None,
        Some(Ok(q)) => Some(q.clone()),
        // The I8 table is FASTEST's, or every precision's under
        // TURBO_CPU_STATIC_TABLE=i8.
        Some(Err(why)) if shape.precision == TURBO_PRECISION_FASTEST => {
            let msg =
                format!("precision: FASTEST sums an I8 copy of the table, and {why}; EXACT sums the stored table");
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
pub(super) fn static_table(precision: u32, var: Option<&str>) -> Result<bool, String> {
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
pub(super) fn static_pages(var: Option<&str>) -> Result<bool, String> {
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
pub(crate) fn threads(field: u32, var: Option<&str>) -> Result<usize, Refusal> {
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
pub(crate) const MAX_THREADS: usize = 1024;

/// Why a session is refused: the status, the turbo_session_desc field it
/// names (0 for none), and the message.
#[derive(Debug, PartialEq)]
pub(crate) struct Refusal {
    pub(crate) code: i32,
    pub(crate) field: u32,
    pub(crate) message: String,
}

impl Refusal {
    pub(super) unsafe fn refuse(self, err: *mut turbo_error) -> i32 {
        unsafe { refuse_field(err, self.code, self.field, &self.message) }
    }
}

#[cfg(test)]
mod tests {
    use super::threads;
    use crate::status::{INVALID_ARGUMENT, UNSUPPORTED_OPTION};

    #[test]
    pub(super) fn turbo_cpu_threads_sets_the_count_or_is_refused() {
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
    pub(super) fn cpu_threads_sets_the_count_or_is_refused_naming_field_6() {
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
