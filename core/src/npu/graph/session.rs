//! Embed sessions: the host buffers a run binds to the graph.

use std::ffi::{c_char, c_void};
use std::sync::Mutex;

use super::args::*;
use super::buffers::*;
use super::model::*;
use super::*;
use crate::backend::{turbo_backend_session_desc, turbo_backend_tuning};
use crate::status::{INVALID_STRUCT_SIZE, UNSUPPORTED, UNSUPPORTED_OPTION};
use crate::{TURBO_DTYPE_F16, TURBO_PRECISION_EXACT, TURBO_TASK_EMBED, turbo_error, turbo_token_range};

// ---- Sessions ---------------------------------------------------------------------------

/// What embed_write or embed_write_ranges left for the next run.
pub(super) struct State {
    pub(super) batch: u32,
    pub(super) seq: u32,
    pub(super) pooling: u32,
    pub(super) normalize: u32,
    pub(super) output_dim: u32,
    pub(super) vectors: Vectors,
    pub(super) written: bool,
}

/// What a run writes one vector for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Vectors {
    /// Each row written, pooled as the write's pooling says, at the
    /// row's index.
    Rows,
    /// Each range in `Rows::ranges`, the mean over its live tokens, at
    /// the range's index.
    Ranges,
}

pub(crate) struct Session {
    pub(super) model: *const Model,
    pub(super) max_seq: u32,
    /// The most ranges one embed_write_ranges holds.
    pub(super) max_ranges: u32,
    /// One host buffer per graph input, a frame each, bound to the graph
    /// at every run.
    pub(super) frames: Frames,
    /// The graph's hidden states, one frame.
    pub(super) out_buf: Buffer,
    /// The run's vectors, [max(max_batch, max_ranges), hidden] F32 at
    /// most, packed at output_dim: handed to buffer_export, released
    /// with the session.
    pub(super) result: Box<Buffer>,
    /// The rows as written, [max_batch, max_seq] at stride max_seq.
    pub(super) rows: Mutex<Rows>,
}

/// The host buffers of one graph input kind.
pub(super) enum Frames {
    Tokens { ids: Buffer, mask: Buffer, types: Option<Buffer> },
    Embeddings { rows: Buffer, bias: Buffer },
}

pub(super) struct Rows {
    pub(super) ids: Vec<i32>,
    pub(super) mask: Vec<i32>,
    pub(super) types: Vec<i32>,
    /// Pooling scratch, one token's sums: no allocation in a run.
    pub(super) acc: Vec<f64>,
    /// One token's hidden row widened from FP16, so pooling reads f32.
    pub(super) token: Vec<f32>,
    /// One embeddings frame, filled on the host and then written at the
    /// argument's precision. Empty for a token-id graph, and sized when
    /// the session is made so a run allocates nothing.
    pub(super) gathered_rows: Vec<f32>,
    pub(super) gathered_bias: Vec<f32>,
    /// The ranges embed_write_ranges kept, in the order given. Its
    /// capacity is max_ranges from the session's creation, so a write
    /// fills it without allocating.
    pub(super) ranges: Vec<turbo_token_range>,
    pub(super) state: State,
}

unsafe impl Send for Session {}
unsafe impl Sync for Session {}

impl Session {
    #[inline]
    pub(super) fn model(&self) -> &Model {
        unsafe { &*self.model }
    }
}

/// Write `text` into the tuning choices, NUL-terminated.
pub(super) fn write_choices(dst: &mut [c_char; crate::TURBO_CHOICES_LEN], text: &str) {
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
    tuning: *mut turbo_backend_tuning,
    compute_dtype: *mut u32,
    out: *mut *mut c_void,
    err: *mut turbo_error,
) -> i32 {
    let rc = unsafe { session_create(model, task, max_batch, max_seq, precision, compute_dtype, out, err) };
    if rc == 0 {
        unsafe { report_format(model, tuning) };
    }
    rc
}

/// The graph format the driver selected, as the session's choices. The
/// one-path out fields stay: nothing is measured or cached.
///
/// # Safety
/// `model` is a model this backend loaded; `tuning` is NULL or valid.
pub(super) unsafe fn report_format(model: *mut c_void, tuning: *mut turbo_backend_tuning) {
    if let Some(t) = unsafe { tuning.as_mut() } {
        let format = unsafe { (*(model as *const Model)).graph_format };
        t.tuned = crate::TURBO_TUNED_DEFAULT;
        t.tune_ms = 0;
        write_choices(&mut t.choices, format);
    }
}

/// # Safety
/// As turbo_backend.h says for session_open. The session is the one
/// `session_create` makes, with its vectors sized for
/// max(max_batch, max_ranges) rows, and the tuning out fields are
/// filled as `session_create_tuned` fills them.
pub(crate) unsafe extern "C" fn session_open(
    model: *mut c_void,
    desc: *const turbo_backend_session_desc,
    tuning: *mut turbo_backend_tuning,
    compute_dtype: *mut u32,
    out: *mut *mut c_void,
    err: *mut turbo_error,
) -> i32 {
    let rc = unsafe {
        guarded(err, || {
            // struct_size is read alone first: the fields after it are
            // read only when it says they are there.
            let size = (*desc).struct_size as usize;
            let needed = std::mem::offset_of!(turbo_backend_session_desc, max_ranges) + size_of::<u32>();
            if size < needed {
                return Err(fail(
                    INVALID_STRUCT_SIZE,
                    format!("npu: turbo_backend_session_desc.struct_size is {size}, and max_ranges ends at {needed}"),
                ));
            }
            let d = &*desc;
            if d.cpu_threads != 0 {
                return Err(fail_field(
                    UNSUPPORTED_OPTION,
                    6,
                    format!(
                        "npu: cpu_threads {}: the npu backend has no CPU thread count; 0 alone is taken",
                        d.cpu_threads
                    ),
                ));
            }
            let m = &*(model as *const Model);
            let s = open(m, d.task, d.max_batch, d.max_seq, d.precision, d.max_ranges)?;
            *compute_dtype = m.compute_dtype;
            *out = Box::into_raw(Box::new(s)) as *mut c_void;
            Ok(())
        })
    };
    if rc == 0 {
        unsafe { report_format(model, tuning) };
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
            // Without session_open no ranges are written, and the vectors
            // are the rows'.
            let s = open(m, task, max_batch, max_seq, precision, max_batch)?;
            *compute_dtype = m.compute_dtype;
            *out = Box::into_raw(Box::new(s)) as *mut c_void;
            Ok(())
        })
    }
}

/// A session of task on `m`, with every buffer a run needs: the frames,
/// the hidden states, max(max_batch, max_ranges) vectors, and room for
/// max_ranges ranges.
pub(super) fn open(
    m: &Model,
    task: u32,
    max_batch: u32,
    max_seq: u32,
    precision: u32,
    max_ranges: u32,
) -> Res<Session> {
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
    let vectors = u64::from(max_batch.max(max_ranges));
    // In usize: max_batch times max_seq may pass u32.
    let slots = max_batch as usize * max_seq as usize;
    Ok(Session {
        model: m,
        max_seq,
        max_ranges,
        frames,
        out_buf: Buffer::new(ctx, frame * m.hidden as u64 * m.output.elem as u64, "hidden states")?,
        result: Box::new(Buffer::new(ctx, vectors * m.hidden as u64 * 4, "the vectors")?),
        rows: Mutex::new(Rows {
            ids: vec![0; slots],
            mask: vec![0; slots],
            types: vec![0; slots],
            acc: vec![0.0; m.hidden as usize],
            token: vec![0.0; m.hidden as usize],
            gathered_rows,
            gathered_bias,
            ranges: Vec::with_capacity(max_ranges as usize),
            state: State {
                batch: 0,
                seq: 0,
                pooling: 0,
                normalize: 0,
                output_dim: 0,
                vectors: Vectors::Rows,
                written: false,
            },
        }),
    })
}

/// # Safety
/// `session` is a session this backend created, released once.
pub(crate) unsafe extern "C" fn session_release(session: *mut c_void) {
    quietly(|| drop(unsafe { Box::from_raw(session as *mut Session) }));
}
