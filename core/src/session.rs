//! Sessions and results: turbo.h's "Session and run" and "Result", for
//! TURBO_TASK_EMBED. The core checks every argument, tokenizes, resolves
//! each option against the bundle, and keeps the session's state; the
//! backend keeps the rows it was written and computes.
//!
//! State. A write replaces what an earlier one left, and a run takes it,
//! even when the run fails, so every run is of exactly one write. A write that fails leaves nothing
//! written. While a result, or a buffer made from it, is held, the
//! session's output is in use and a write or run is TURBO_E_BUSY; so is a
//! call made while another is still inside the session.
//!
//! The run path. turbo_session_run allocates nothing on the heap: the
//! result handle is made with the session and handed out again by each
//! run, which is possible because a session has at most one result held
//! at a time. The backend's session was likewise sized for the session's
//! largest batch when it was made. So host_allocs is the backend's count
//! plus CORE_RUN_ALLOCS, which is 0; tests/allocations.rs checks the sum
//! against a counting allocator. A released result's handle and the next
//! run's are therefore the same address.

use std::ffi::{c_char, c_void};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, TryLockError};

use crate::backend::{self, turbo_backend, turbo_backend_embed_rows, turbo_backend_run};
use crate::manifest::{PromptRole, Truncation};
use crate::status::{
    BUSY, CAPACITY, Error, INTERNAL, INVALID_ARGUMENT, INVALID_ENUM, INVALID_HANDLE, INVALID_SHAPE, INVALID_STATE,
    Result, UNSUPPORTED, UNSUPPORTED_OPTION, UNSUPPORTED_TASK,
};
use crate::tokenizer::Encode;
use crate::*;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct turbo_session_desc {
    pub struct_size: u32,
    pub max_batch: u32,
    pub max_seq: u32,
    pub precision: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct turbo_embed_options {
    pub struct_size: u32,
    pub truncate: u32,
    pub max_tokens: u32,
    pub prompt_role: u32,
    pub normalize: u32,
    pub pooling: u32,
    pub output_dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct turbo_token_batch {
    pub struct_size: u32,
    pub batch: u32,
    pub seq: u32,
    pub row_stride: u32,
    pub ids: *const i32,
    pub mask: *const i32,
    pub types: *const i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct turbo_session_info {
    pub struct_size: u32,
    pub max_batch: u32,
    pub max_seq: u32,
    pub precision: u32,
    pub compute_dtype: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct turbo_result_info {
    pub struct_size: u32,
    pub task: u32,
    pub batch: u32,
    pub dim: u32,
    pub dtype: u32,
    pub compute_dtype: u32,
    pub placement: u32,
    pub device: u32,
    pub bytes: u64,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
    pub host_allocs: u64,
    pub device_allocs: u64,
    pub stage_count: u32,
    pub reserved: u32,
    pub stage: [u32; TURBO_STAGE_MAX],
    pub backend: [c_char; 32],
    pub arch: [c_char; 32],
    pub runtime_version: [c_char; 64],
    pub manifest_sha256: [c_char; 72],
    pub artifact_sha256: [c_char; 72],
    pub tokenizer_sha256: [c_char; 72],
}

const SESSION_MAGIC: u64 = 0x7475_7262_6f73_7331; // "turboss1"
const RESULT_MAGIC: u64 = 0x7475_7262_6f72_7331; // "turbors1"

/// Fields 1 to 3 of turbo_embed_options (truncate, max_tokens,
/// prompt_role): the core applies them before a backend sees the rows, so
/// every device honors them.
pub(crate) const CORE_HONORED: u32 = 0b111;

/// Heap allocations the core's part of turbo_session_run makes. See the
/// module comment.
const CORE_RUN_ALLOCS: u64 = 0;

/// A session: the backend's, and the state the core keeps for it. Held by
/// the session handle, and by the result and its buffers while they are.
pub(crate) struct SessionInner {
    model: Arc<Model>,
    backend: &'static turbo_backend,
    raw: *mut c_void,
    release: unsafe extern "C" fn(*mut c_void),
    run: unsafe extern "C" fn(*mut c_void, *mut turbo_backend_run, *mut turbo_error) -> i32,
    info: turbo_session_info,
    /// options_honored of the device's cell at the session's precision.
    honored: u32,
    state: Mutex<State>,
    /// The result handle and the buffers made from it that are held: the
    /// session's output is theirs while this is not 0.
    holders: AtomicU32,
    /// The one result handle, made with the session. Freed with it: while
    /// the handle is held it holds the session, so that is after its
    /// release.
    result: *mut turbo_result,
}

// turbo_backend.h: the backend's session may be used from any thread, and
// the core lets one call at a time into it.
unsafe impl Send for SessionInner {}
unsafe impl Sync for SessionInner {}

struct State {
    written: Option<Written>,
    /// Rows turbo_embed_write_text lays out, max_batch x max_seq, made
    /// with the session.
    ids: Vec<i32>,
    mask: Vec<i32>,
}

/// What the backend was last written.
#[derive(Clone, Copy)]
struct Written {
    batch: u32,
    output_dim: u32,
    /// Whether the core tokenized the rows (write_text) or the caller did.
    tokenized: bool,
}

impl SessionInner {
    /// The state, for a call that writes to the session or runs it.
    fn lock(&self) -> Result<MutexGuard<'_, State>> {
        let state = match self.state.try_lock() {
            Ok(g) => g,
            Err(TryLockError::WouldBlock) => {
                return Err(Error::new(BUSY, "another call is using the session"));
            }
            // A panic stopped at the boundary inside an earlier call; what
            // it left written is dropped by the call that comes next.
            Err(TryLockError::Poisoned(p)) => p.into_inner(),
        };
        if self.holders.load(Ordering::Acquire) != 0 {
            return Err(Error::new(BUSY, "the session's result is held: release it and its buffers first"));
        }
        Ok(state)
    }

    /// A holder of the output is released.
    pub(crate) fn let_go(&self) {
        self.holders.fetch_sub(1, Ordering::AcqRel);
    }
}

impl Drop for SessionInner {
    fn drop(&mut self) {
        unsafe {
            (self.release)(self.raw);
            drop(Box::from_raw(self.result));
        }
    }
}

pub struct turbo_session {
    magic: u64,
    inner: Arc<SessionInner>,
}

/// The outputs of one run. The same memory serves every run of its
/// session; magic says whether it is held now.
pub struct turbo_result {
    magic: u64,
    /// The session, while the handle is held.
    hold: Option<Arc<SessionInner>>,
    info: turbo_result_info,
    /// Bytes turbo_result_read copied out, counted in d2h_bytes.
    read: AtomicU64,
    /// The backend's buffer the vectors are in, and its host address.
    output: *mut c_void,
    host: *mut c_void,
}

unsafe fn session<'a>(s: *mut turbo_session) -> Result<&'a turbo_session> {
    match unsafe { s.as_ref() } {
        Some(r) if r.magic == SESSION_MAGIC => Ok(r),
        _ => Err(Error::new(INVALID_HANDLE, "not a turbo_session")),
    }
}

unsafe fn result<'a>(r: *mut turbo_result) -> Result<&'a mut turbo_result> {
    match unsafe { r.as_mut() } {
        Some(r) if r.magic == RESULT_MAGIC => Ok(r),
        _ => Err(Error::new(INVALID_HANDLE, "not a turbo_result, or one already released")),
    }
}

// ---- Session ------------------------------------------------------------------

/// # Safety
/// Pointers are NULL or valid for the call, as turbo.h says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_session_create(
    m: *mut turbo_model,
    desc: *const turbo_session_desc,
    out: *mut *mut turbo_session,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let m = &model_handle(m)?.inner;
            let out = out_ptr(out, "out")?;
            // NULL is 0 in every field, as for turbo_runtime_create.
            let d = match desc.as_ref() {
                Some(d) => {
                    sized(d.struct_size, size_of::<turbo_session_desc>(), "turbo_session_desc")?;
                    *d
                }
                None => turbo_session_desc { struct_size: 0, max_batch: 0, max_seq: 0, precision: 0 },
            };
            let inner = create_session(m, d)?;
            *out = Box::into_raw(Box::new(turbo_session { magic: SESSION_MAGIC, inner: Arc::new(inner) }));
            Ok(())
        })
    }
}

fn create_session(m: &Arc<Model>, d: turbo_session_desc) -> Result<SessionInner> {
    if d.precision > TURBO_PRECISION_EXACT {
        return Err(Error::new(INVALID_ENUM, format!("precision: {} is not a TURBO_PRECISION_* value", d.precision)));
    }
    let mi = &m.info;
    // docs/bundle.md: max_batch is the largest batch the reference was
    // checked at, and a session larger than it is refused; max_seq is the
    // length the model was evaluated at.
    let limit = |field: u32, name: &str, asked: u32, model: u32| {
        if asked > model {
            return Err(Error::field(UNSUPPORTED_OPTION, field, format!("{name} {asked} is over the model's {model}")));
        }
        Ok(if asked == 0 { model } else { asked })
    };
    let max_batch = limit(1, "max_batch", d.max_batch, mi.max_batch)?;
    let max_seq = limit(2, "max_seq", d.max_seq, mi.max_seq)?;

    let c = &m.context;
    let b = c.backend;
    let create = backend::offered!(b, session_create)?;
    let release = backend::offered!(b, session_release)?;
    let run = backend::offered!(b, session_run)?;
    let cap = c.runtime.capability(c.device, mi.task, d.precision)?;
    if cap.status == backend::TURBO_CAP_UNSUPPORTED {
        let reason = backend::cstr(&cap.reason);
        // The task runs at another precision: the precision is the option refused.
        let other = (TURBO_PRECISION_MODEL..=TURBO_PRECISION_EXACT).filter(|&p| p != d.precision).any(|p| {
            c.runtime.capability(c.device, mi.task, p).is_ok_and(|o| o.status != backend::TURBO_CAP_UNSUPPORTED)
        });
        if other {
            return Err(Error::field(
                UNSUPPORTED_OPTION,
                3,
                format!("precision {}: the {} backend does not run it: {reason}", d.precision, b.name()),
            ));
        }
        return Err(Error::new(UNSUPPORTED_TASK, format!("the {} backend does not run embed: {reason}", b.name())));
    }
    let mut compute_dtype = 0;
    let mut raw = std::ptr::null_mut();
    backend::check(b, "session_create", |err| unsafe {
        create(m.raw, mi.task, max_batch, max_seq, d.precision, &mut compute_dtype, &mut raw, err)
    })?;
    if !matches!(compute_dtype, TURBO_DTYPE_I8 | TURBO_DTYPE_F16 | TURBO_DTYPE_BF16 | TURBO_DTYPE_F32) {
        unsafe { release(raw) };
        return Err(Error::new(
            INTERNAL,
            format!("{} backend: compute dtype {compute_dtype} is not a TURBO_DTYPE_* a model computes in", b.name()),
        ));
    }
    let info = turbo_session_info {
        struct_size: size_of::<turbo_session_info>() as u32,
        max_batch,
        max_seq,
        precision: d.precision,
        compute_dtype,
        reserved: 0,
    };
    let rows = max_batch as usize * max_seq as usize;
    let result = Box::into_raw(Box::new(turbo_result {
        magic: 0,
        hold: None,
        info: result_template(m, &info),
        read: AtomicU64::new(0),
        output: std::ptr::null_mut(),
        host: std::ptr::null_mut(),
    }));
    let state = Mutex::new(State { written: None, ids: vec![0; rows], mask: vec![0; rows] });
    // Where the lock is a pthread mutex (macOS), std allocates it on first
    // use; take it once here so no write or run allocates it later.
    drop(state.try_lock());
    Ok(SessionInner {
        model: m.clone(),
        backend: b,
        raw,
        release,
        run,
        info,
        honored: cap.options_honored,
        state,
        holders: AtomicU32::new(0),
        result,
    })
}

/// The fields of turbo_result_info that are the same for every run of the
/// session, filled in once so a run writes only its own.
fn result_template(m: &Model, s: &turbo_session_info) -> turbo_result_info {
    let mut r: turbo_result_info = unsafe { std::mem::zeroed() };
    let c = &m.context;
    let device = &c.runtime.devices[c.device as usize].info;
    r.struct_size = size_of::<turbo_result_info>() as u32;
    r.task = m.info.task;
    r.dtype = TURBO_DTYPE_F32;
    r.compute_dtype = s.compute_dtype;
    r.device = c.device;
    r.stage_count = TURBO_EMBED_STAGE_COUNT as u32;
    r.backend = device.backend;
    r.arch = device.arch;
    r.runtime_version = device.runtime_version;
    r.manifest_sha256 = m.info.manifest_sha256;
    r.artifact_sha256 = m.info.artifact_sha256;
    r.tokenizer_sha256 = m.info.tokenizer_sha256;
    r
}

/// # Safety
/// `s` is NULL or a handle from turbo_session_create, released once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_session_release(s: *mut turbo_session) {
    if unsafe { session(s) }.is_ok() {
        let mut b = unsafe { Box::from_raw(s) };
        b.magic = 0;
    }
}

/// # Safety
/// Pointers are NULL or valid for the call, as turbo.h says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_session_get_info(
    s: *mut turbo_session,
    out: *mut turbo_session_info,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let s = session(s)?;
            let out = out_ptr(out, "out")?;
            sized(out.struct_size, size_of::<turbo_session_info>(), "turbo_session_info")?;
            *out = s.inner.info;
            Ok(())
        })
    }
}

// ---- Writes ---------------------------------------------------------------------

/// turbo_embed_options with every 0 resolved to what the bundle says.
#[derive(Debug, Clone, Copy)]
struct EmbedOptions {
    truncation: Truncation,
    /// 0 when the caller gave none.
    max_tokens: u32,
    prompt: PromptRole,
    pooling: u32,
    normalize: u32,
    output_dim: u32,
    /// The caller's truncate and prompt_role, which rows written as tokens
    /// refuse.
    truncate_given: bool,
    prompt_given: bool,
}

unsafe fn embed_options(s: &SessionInner, opts: *const turbo_embed_options) -> Result<EmbedOptions> {
    let o = match unsafe { opts.as_ref() } {
        Some(o) => {
            sized(o.struct_size, size_of::<turbo_embed_options>(), "turbo_embed_options")?;
            *o
        }
        None => turbo_embed_options {
            struct_size: 0,
            truncate: 0,
            max_tokens: 0,
            prompt_role: 0,
            normalize: 0,
            pooling: 0,
            output_dim: 0,
        },
    };
    let enumerated = |name: &str, v: u32, last: u32, prefix: &str| {
        if v > last {
            return Err(Error::new(INVALID_ENUM, format!("{name}: {v} is not a TURBO_{prefix}_* value")));
        }
        Ok(v)
    };
    enumerated("truncate", o.truncate, TURBO_TRUNCATE_LEFT, "TRUNCATE")?;
    let prompt = prompt_role(o.prompt_role)?;
    enumerated("normalize", o.normalize, TURBO_NORMALIZE_L2, "NORMALIZE")?;
    enumerated("pooling", o.pooling, TURBO_POOLING_LAST, "POOLING")?;

    honored(&o, s.honored, s.backend.name())?;

    if o.max_tokens > s.info.max_seq {
        return Err(Error::new(
            CAPACITY,
            format!("max_tokens {} is over the session's max_seq {}", o.max_tokens, s.info.max_seq),
        ));
    }
    let mi = &s.model.info;
    let output_dim = match o.output_dim {
        0 => mi.dim,
        d if d > mi.dim => {
            return Err(Error::field(
                INVALID_ARGUMENT,
                6,
                format!("output_dim {d} is over the model's dim {}", mi.dim),
            ));
        }
        d if d == mi.dim || s.model.output_dims.contains(&d) => d,
        d => {
            return Err(Error::field(
                UNSUPPORTED_OPTION,
                6,
                format!(
                    "output_dim {d}: the model was trained to be cut to {:?} besides its dim {}",
                    s.model.output_dims, mi.dim
                ),
            ));
        }
    };
    Ok(EmbedOptions {
        truncation: match o.truncate {
            TURBO_TRUNCATE_MODEL => s.model.tokenizer.truncation(),
            TURBO_TRUNCATE_NONE => Truncation::None,
            TURBO_TRUNCATE_RIGHT => Truncation::Right,
            _ => Truncation::Left,
        },
        max_tokens: o.max_tokens,
        prompt,
        pooling: if o.pooling == TURBO_POOLING_MODEL { mi.pooling } else { o.pooling },
        normalize: if o.normalize == TURBO_NORMALIZE_MODEL { mi.normalize } else { o.normalize },
        output_dim,
        truncate_given: o.truncate != 0,
        prompt_given: o.prompt_role != 0,
    })
}

/// A field the device does not honor may only be left to the bundle: any
/// other value is refused by its number.
fn honored(o: &turbo_embed_options, honored: u32, backend: &str) -> Result<()> {
    let fields = [
        ("truncate", o.truncate),
        ("max_tokens", o.max_tokens),
        ("prompt_role", o.prompt_role),
        ("normalize", o.normalize),
        ("pooling", o.pooling),
        ("output_dim", o.output_dim),
    ];
    for (i, (name, v)) in fields.iter().enumerate() {
        if *v != 0 && honored & (1 << i) == 0 {
            return Err(Error::field(
                UNSUPPORTED_OPTION,
                i as u32 + 1,
                format!("{name} {v}: the {backend} backend does not honor it at this precision"),
            ));
        }
    }
    Ok(())
}

/// The start of every write: the session is free, it is an embed
/// session, and nothing is left written until this write succeeds.
fn begin_write(s: &SessionInner) -> Result<MutexGuard<'_, State>> {
    let mut state = s.lock()?;
    state.written = None;
    if s.model.info.task != TURBO_TASK_EMBED {
        return Err(Error::new(UNSUPPORTED_TASK, "the model's task is not embed"));
    }
    Ok(state)
}

/// Hand checked rows to the backend and record them.
fn write_rows(
    s: &SessionInner,
    state: &mut State,
    rows: turbo_backend_embed_rows,
    o: &EmbedOptions,
    tokenized: bool,
) -> Result<()> {
    let b = s.backend;
    let write = backend::offered!(b, embed_write)?;
    backend::check(b, "embed_write", |err| unsafe { write(s.raw, &rows, err) })?;
    state.written = Some(Written { batch: rows.batch, output_dim: o.output_dim, tokenized });
    Ok(())
}

fn backend_rows(o: &EmbedOptions, batch: u32, seq: u32, row_stride: u32) -> turbo_backend_embed_rows {
    turbo_backend_embed_rows {
        struct_size: size_of::<turbo_backend_embed_rows>() as u32,
        batch,
        seq,
        row_stride,
        ids: std::ptr::null(),
        mask: std::ptr::null(),
        types: std::ptr::null(),
        pooling: o.pooling,
        normalize: o.normalize,
        output_dim: o.output_dim,
        reserved: 0,
    }
}

/// # Safety
/// `texts` holds `count` views; other pointers are NULL or valid for the
/// call, as turbo.h says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_embed_write_text(
    s: *mut turbo_session,
    texts: *const turbo_text,
    count: u32,
    opts: *const turbo_embed_options,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let s = &session(s)?.inner;
            let mut state = begin_write(s)?;
            let o = embed_options(s, opts)?;
            if count == 0 {
                return Err(Error::new(INVALID_ARGUMENT, "count is 0: a run needs at least one row"));
            }
            if texts.is_null() {
                return Err(Error::new(INVALID_ARGUMENT, "texts is NULL"));
            }
            if count > s.info.max_batch {
                return Err(Error::new(
                    CAPACITY,
                    format!("{count} texts is over the session's max_batch {}", s.info.max_batch),
                ));
            }
            let tok = &s.model.tokenizer;
            // TURBO_TRUNCATE_MODEL cuts at the bundle's max_seq on every
            // device (docs/bundle.md); a row that is then longer than the
            // session takes is CAPACITY, not cut again.
            let max_tokens = if o.max_tokens == 0 { tok.max_seq } else { o.max_tokens };
            let specials = tok.specials_per_sequence();
            if max_tokens < specials {
                return Err(Error::field(
                    INVALID_ARGUMENT,
                    2,
                    format!("max_tokens {max_tokens} leaves no room beside {specials} special tokens"),
                ));
            }
            let e = Encode { add_special_tokens: true, truncation: o.truncation, max_tokens, prompt: o.prompt };
            let texts = std::slice::from_raw_parts(texts, count as usize);
            let mut rows = Vec::with_capacity(texts.len());
            for (i, &t) in texts.iter().enumerate() {
                let row = tok.encode(text(t, &format!("texts[{i}]"))?, e).map_err(|mut err| {
                    err.message = format!("texts[{i}]: {}", err.message);
                    err
                })?;
                if row.len() > s.info.max_seq as usize {
                    return Err(Error::new(
                        CAPACITY,
                        format!("texts[{i}]: {} tokens is over the session's max_seq {}", row.len(), s.info.max_seq),
                    ));
                }
                rows.push(row);
            }
            let seq = rows.iter().map(Vec::len).max().unwrap_or(0);
            let state = &mut *state;
            for (i, row) in rows.iter().enumerate() {
                let (ids, mask) = (&mut state.ids[i * seq..(i + 1) * seq], &mut state.mask[i * seq..(i + 1) * seq]);
                ids[..row.len()].copy_from_slice(row);
                ids[row.len()..].fill(tok.pad_id);
                mask[..row.len()].fill(1);
                mask[row.len()..].fill(0);
            }
            let mut r = backend_rows(&o, count, seq as u32, seq as u32);
            r.ids = state.ids.as_ptr();
            r.mask = state.mask.as_ptr();
            write_rows(s, state, r, &o, true)
        })
    }
}

/// # Safety
/// `batch` and the arrays it points to hold what turbo.h says; other
/// pointers are NULL or valid for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_embed_write_tokens(
    s: *mut turbo_session,
    batch: *const turbo_token_batch,
    opts: *const turbo_embed_options,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let s = &session(s)?.inner;
            let tb = *batch.as_ref().ok_or_else(|| Error::new(INVALID_ARGUMENT, "batch is NULL"))?;
            sized(tb.struct_size, size_of::<turbo_token_batch>(), "turbo_token_batch")?;
            let mut state = begin_write(s)?;
            let o = embed_options(s, opts)?;
            if o.truncate_given {
                return Err(Error::field(INVALID_ARGUMENT, 1, "truncate: rows written as tokens are already cut"));
            }
            if o.prompt_given {
                return Err(Error::field(
                    INVALID_ARGUMENT,
                    3,
                    "prompt_role: rows written as tokens carry their prefix",
                ));
            }
            check_tokens(s, &tb, o.max_tokens)?;
            let stride = if tb.row_stride == 0 { tb.seq } else { tb.row_stride };
            let mut r = backend_rows(&o, tb.batch, tb.seq, stride);
            r.ids = tb.ids;
            r.mask = tb.mask;
            r.types = tb.types;
            write_rows(s, &mut state, r, &o, false)
        })
    }
}

/// Every row the caller wrote, checked as turbo_backend.h promises the
/// backend: the shape within the session, and each id, mask entry and
/// type one the model has. `max_tokens`, when not 0, bounds each row's
/// length through its last live token.
unsafe fn check_tokens(s: &SessionInner, b: &turbo_token_batch, max_tokens: u32) -> Result<()> {
    if b.batch == 0 || b.seq == 0 {
        return Err(Error::new(INVALID_SHAPE, format!("batch {} by seq {}: neither may be 0", b.batch, b.seq)));
    }
    if b.batch > s.info.max_batch {
        return Err(Error::new(
            CAPACITY,
            format!("batch {} is over the session's max_batch {}", b.batch, s.info.max_batch),
        ));
    }
    if b.seq > s.info.max_seq {
        return Err(Error::new(CAPACITY, format!("seq {} is over the session's max_seq {}", b.seq, s.info.max_seq)));
    }
    let stride = if b.row_stride == 0 { b.seq } else { b.row_stride } as usize;
    if stride < b.seq as usize {
        return Err(Error::new(INVALID_SHAPE, format!("row_stride {stride} is under seq {}", b.seq)));
    }
    if b.ids.is_null() || b.mask.is_null() {
        return Err(Error::new(INVALID_ARGUMENT, "ids and mask may not be NULL"));
    }
    let (vocab, types) = (s.model.weights.vocab_size(), s.model.weights.token_types());
    let (n, seq) = (b.batch as usize, b.seq as usize);
    let span = (n - 1) * stride + seq;
    let (ids, mask) = unsafe { (std::slice::from_raw_parts(b.ids, span), std::slice::from_raw_parts(b.mask, span)) };
    let ty = (!b.types.is_null()).then(|| unsafe { std::slice::from_raw_parts(b.types, span) });
    for r in 0..n {
        let at = r * stride;
        let mut live = 0;
        for p in 0..seq {
            let (id, m) = (ids[at + p], mask[at + p]);
            if id < 0 || id as u32 >= vocab {
                return Err(Error::new(
                    INVALID_ARGUMENT,
                    format!("ids[{r}][{p}] is {id}, outside the model's vocabulary of {vocab}"),
                ));
            }
            match m {
                0 => {}
                1 => live = p + 1,
                _ => return Err(Error::new(INVALID_ARGUMENT, format!("mask[{r}][{p}] is {m}, not 0 or 1"))),
            }
            if let Some(ty) = ty
                && (ty[at + p] < 0 || ty[at + p] as u32 >= types)
            {
                return Err(Error::new(
                    INVALID_ARGUMENT,
                    format!("types[{r}][{p}] is {}, and the model has {types} token types", ty[at + p]),
                ));
            }
        }
        if live == 0 {
            return Err(Error::new(INVALID_ARGUMENT, format!("mask row {r} has no 1: a row needs a token")));
        }
        if max_tokens != 0 && live > max_tokens as usize {
            return Err(Error::new(
                CAPACITY,
                format!("row {r}: {live} tokens through its last live one is over max_tokens {max_tokens}"),
            ));
        }
    }
    Ok(())
}

// ---- Run --------------------------------------------------------------------------

/// # Safety
/// Pointers are NULL or valid for the call, as turbo.h says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_session_run(
    s: *mut turbo_session,
    out: *mut *mut turbo_result,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let handle = session(s)?;
            let s = &handle.inner;
            let out = out_ptr(out, "out")?;
            let mut state = s.lock()?;
            let w = state
                .written
                .take()
                .ok_or_else(|| Error::new(INVALID_STATE, "nothing is written: a run takes one write"))?;
            let mut r: turbo_backend_run = std::mem::zeroed();
            r.struct_size = size_of::<turbo_backend_run>() as u32;
            r.stage[TURBO_EMBED_STAGE_TOKENIZE] = if w.tokenized { TURBO_STAGE_HOST } else { TURBO_STAGE_UNUSED };
            let b = s.backend;
            backend::check(b, "session_run", |err| (s.run)(s.raw, &mut r, err))?;
            check_run(b, &r, w.tokenized)?;

            let res = &mut *s.result;
            let i = &mut res.info;
            i.batch = w.batch;
            i.dim = w.output_dim;
            i.placement = r.placement;
            i.bytes = w.batch as u64 * w.output_dim as u64 * 4;
            i.h2d_bytes = r.h2d_bytes;
            i.d2h_bytes = r.d2h_bytes;
            i.host_allocs = r.host_allocs + CORE_RUN_ALLOCS;
            i.device_allocs = r.device_allocs;
            i.stage = r.stage;
            res.output = r.output;
            res.host = r.host;
            res.read.store(0, Ordering::Relaxed);
            s.holders.store(1, Ordering::Release);
            res.hold = Some(s.clone());
            res.magic = RESULT_MAGIC;
            *out = s.result;
            drop(state);
            Ok(())
        })
    }
}

/// What the backend reported is something turbo.h can say.
fn check_run(b: &turbo_backend, r: &turbo_backend_run, tokenized: bool) -> Result<()> {
    let bad = |what: String| Err(Error::new(INTERNAL, format!("{} backend, session_run: {what}", b.name())));
    if !(TURBO_PLACE_HOST..=TURBO_PLACE_SHARED).contains(&r.placement) {
        return bad(format!("placement {} is not a TURBO_PLACE_* value", r.placement));
    }
    if r.output.is_null() || r.host.is_null() != (r.placement == TURBO_PLACE_DEVICE) {
        return bad(format!("output {:?} at host address {:?} for placement {}", r.output, r.host, r.placement));
    }
    for (i, &st) in r.stage.iter().enumerate() {
        let ok = if i < TURBO_EMBED_STAGE_COUNT { st <= TURBO_STAGE_FUSED } else { st == TURBO_STAGE_UNUSED };
        if !ok {
            return bad(format!("stage[{i}] is {st}"));
        }
    }
    let tokenize = if tokenized { TURBO_STAGE_HOST } else { TURBO_STAGE_UNUSED };
    if r.stage[TURBO_EMBED_STAGE_TOKENIZE] != tokenize {
        return bad("it rewrote the tokenize stage, which is the core's".to_owned());
    }
    Ok(())
}

// ---- Result ---------------------------------------------------------------------

/// # Safety
/// Pointers are NULL or valid for the call, as turbo.h says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_result_get_info(
    r: *mut turbo_result,
    out: *mut turbo_result_info,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let r = result(r)?;
            let out = out_ptr(out, "out")?;
            sized(out.struct_size, size_of::<turbo_result_info>(), "turbo_result_info")?;
            *out = r.info;
            out.d2h_bytes += r.read.load(Ordering::Relaxed);
            Ok(())
        })
    }
}

/// # Safety
/// `dst` is NULL or holds `capacity` bytes; other pointers are NULL or
/// valid for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_result_read(
    r: *mut turbo_result,
    dst: *mut c_void,
    capacity: u64,
    written: *mut u64,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let r = result(r)?;
            if dst.is_null() {
                return Err(Error::new(INVALID_ARGUMENT, "dst is NULL"));
            }
            let bytes = r.info.bytes;
            if capacity < bytes {
                return Err(Error::new(
                    INVALID_ARGUMENT,
                    format!("capacity {capacity} is under the result's {bytes} bytes"),
                ));
            }
            if r.host.is_null() {
                // Device memory: the backend copies it back.
                let b = r.hold.as_ref().expect("a held result holds its session").backend;
                let read = backend::offered!(b, buffer_read).map_err(|_| {
                    Error::new(
                        UNSUPPORTED,
                        format!("the vectors are in device memory, and the {} backend reads none", b.name()),
                    )
                })?;
                backend::check(b, "buffer_read", |err| read(r.output, dst, bytes, err))?;
            } else {
                std::ptr::copy_nonoverlapping(r.host as *const u8, dst as *mut u8, bytes as usize);
            }
            // turbo.h counts a read in d2h_bytes, whether or not the copy
            // crossed a bus: on a CPU it is a copy within host memory.
            r.read.fetch_add(bytes, Ordering::Relaxed);
            if let Some(w) = written.as_mut() {
                *w = bytes;
            }
            Ok(())
        })
    }
}

/// # Safety
/// Pointers are NULL or valid for the call, as turbo.h says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_result_buffer(
    r: *mut turbo_result,
    out: *mut *mut turbo_buffer,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let r = result(r)?;
            let out = out_ptr(out, "out")?;
            let s = r.hold.as_ref().expect("a held result holds its session");
            let i = &r.info;
            let desc = turbo_buffer_desc {
                struct_size: size_of::<turbo_buffer_desc>() as u32,
                placement: i.placement,
                dtype: i.dtype,
                ndim: 2,
                shape: [i.batch as u64, i.dim as u64],
                bytes: i.bytes,
            };
            s.holders.fetch_add(1, Ordering::AcqRel);
            let inner = Buffer {
                context: s.model.context.clone(),
                desc,
                raw: r.output,
                host: r.host,
                release: Release::Result(s.clone()),
            };
            *out = Box::into_raw(Box::new(turbo_buffer { magic: BUFFER_MAGIC, inner }));
            Ok(())
        })
    }
}

/// # Safety
/// `r` is NULL or a handle from turbo_session_run, released once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_result_release(r: *mut turbo_result) {
    if let Ok(res) = unsafe { result(r) } {
        let keep = res.hold.take();
        res.magic = 0;
        if let Some(s) = keep {
            s.let_go();
            // The last reference frees the session and this handle with
            // it, so nothing of it is touched after.
            drop(s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero() -> turbo_embed_options {
        turbo_embed_options {
            struct_size: size_of::<turbo_embed_options>() as u32,
            truncate: 0,
            max_tokens: 0,
            prompt_role: 0,
            normalize: 0,
            pooling: 0,
            output_dim: 0,
        }
    }

    #[test]
    fn an_option_a_device_does_not_honor_is_refused_by_its_number() {
        honored(&zero(), 0, "x").unwrap();
        let o = turbo_embed_options { pooling: TURBO_POOLING_CLS, output_dim: 8, ..zero() };
        // Only the fields the core applies: pooling, field 5, comes first.
        let e = honored(&o, CORE_HONORED, "x").unwrap_err();
        assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 5));
        assert_eq!(e.message, "pooling 2: the x backend does not honor it at this precision");
        let e = honored(&o, CORE_HONORED | 1 << 4, "x").unwrap_err();
        assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 6));
        honored(&o, CORE_HONORED | 0b110000, "x").unwrap();
        let o = turbo_embed_options { max_tokens: 3, ..zero() };
        assert_eq!(honored(&o, 0, "x").unwrap_err().field, 2);
    }
}
