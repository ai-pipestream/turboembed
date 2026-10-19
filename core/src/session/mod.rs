//! Sessions and results: turbo.h's "Session and run" and "Result", for
//! TURBO_TASK_EMBED. The core checks every argument, tokenizes, resolves
//! each option against the bundle, and keeps the session's state; the
//! backend keeps the rows it was written and computes.
//!
//! State. A write replaces what an earlier one left, and a run takes it,
//! even when the run fails, so every run is of exactly one write. A write
//! that fails leaves nothing written. While a result, or a buffer made
//! from it, is held, the session's output is in use and a write or run is
//! TURBO_E_BUSY; so is a call made while another is still inside the
//! session.
//!
//! Ranges. turbo_embed_write_ranges writes rows as turbo_embed_write_tokens
//! does, and the run that takes it returns one vector per range rather
//! than per row, so the result's batch is the count of ranges written.
//!
//! The run path. turbo_session_run allocates nothing on the heap: the
//! result handle is made with the session and handed out again by each
//! run, which is possible because a session has at most one result held
//! at a time. The backend's session was likewise sized when it was made
//! for the session's largest batch, or for its max_ranges vectors when
//! those are more. So host_allocs is the backend's count plus
//! CORE_RUN_ALLOCS, which is 0; tests/allocations.rs checks the sum
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
use crate::tuning;
use crate::*;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct turbo_session_desc {
    pub struct_size: u32,
    pub max_batch: u32,
    pub max_seq: u32,
    pub precision: u32,
    pub tuning: u32,
    pub tuning_budget_ms: u32,
    pub cpu_threads: u32,
    pub max_ranges: u32,
}

/// The size of turbo_session_desc before tuning was appended, which
/// turbo_session_create still accepts.
pub const TURBO_SESSION_DESC_SIZE_V1: usize = std::mem::offset_of!(turbo_session_desc, tuning);

/// The size of turbo_session_desc before cpu_threads was appended, which
/// turbo_session_create still accepts.
pub const TURBO_SESSION_DESC_SIZE_V2: usize = std::mem::offset_of!(turbo_session_desc, cpu_threads);

/// A range of one row's tokens, for turbo_embed_write_ranges.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct turbo_token_range {
    pub row: u32,
    pub start: u32,
    pub end: u32,
}

/// The size of turbo_session_info before tuned was appended, which
/// turbo_session_get_info still accepts.
pub const TURBO_SESSION_INFO_SIZE_V1: usize = std::mem::offset_of!(turbo_session_info, tuned);

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
    pub tuned: u32,
    pub tune_ms: u32,
    pub choices: [c_char; TURBO_CHOICES_LEN],
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
    /// The most ranges one turbo_embed_write_ranges may write: the
    /// caller's max_ranges, or max_batch for 0.
    max_ranges: u32,
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

mod create;
pub use create::*;

mod options;
use options::*;

mod text;
pub use text::*;

mod tokens;
pub use tokens::*;

mod result;
pub use result::*;

#[cfg(test)]
mod tests;
