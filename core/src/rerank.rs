//! Safe cross-encoder sessions. Scores carry the exact bundle identity.

use crate::status::{Error, INTERNAL, INVALID_ARGUMENT, Result, UNSUPPORTED_TASK};
use crate::*;
use std::{ffi::c_char, path::Path, ptr};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RerankIdentity {
    pub model_id: String,
    pub revision: String,
    pub manifest_sha256: String,
    pub artifact_sha256: String,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RerankOptions {
    /// Zero uses the model's query budget.
    pub query_tokens: u32,
    /// Zero uses the session's sequence limit.
    pub max_tokens: u32,
    /// Passage window advance; zero means non-overlapping windows.
    pub stride: u32,
}

#[derive(Clone, Debug)]
pub struct RerankScores {
    pub model: RerankIdentity,
    /// Raw logits in input order, max-pooled across passage windows.
    pub scores: Vec<f32>,
}

/// Where and how a [`Reranker`] session runs.
#[derive(Clone, Debug)]
pub struct RerankSessionOptions {
    /// The backend to run on, by its name in `turbo_device_info.backend`
    /// ("cpu", "cuda", ...): its first device. `None` takes the device
    /// `turbo_runtime_select` picks for the task.
    pub backend: Option<String>,
    /// `TURBO_PRECISION_*`.
    pub precision: u32,
    /// The threads a CPU session computes on; 0 is the core's default.
    pub cpu_threads: u32,
}

impl Default for RerankSessionOptions {
    fn default() -> Self {
        Self { backend: None, precision: TURBO_PRECISION_EXACT, cpu_threads: 0 }
    }
}

pub struct Reranker {
    runtime: *mut turbo_runtime,
    context: *mut turbo_context,
    model: *mut turbo_model,
    session: *mut turbo_session,
    identity: RerankIdentity,
    max_batch: usize,
    backend: String,
    device: u32,
}

// SAFETY: the C handles support movement between threads. Every session
// operation requires exclusive access through &mut self.
unsafe impl Send for Reranker {}

impl Drop for Reranker {
    fn drop(&mut self) {
        unsafe {
            turbo_session_release(self.session);
            turbo_model_release(self.model);
            turbo_context_release(self.context);
            turbo_runtime_release(self.runtime);
        }
    }
}

fn string(chars: &[c_char]) -> String {
    String::from_utf8_lossy(&chars.iter().take_while(|&&c| c != 0).map(|&c| c.to_ne_bytes()[0]).collect::<Vec<_>>())
        .into_owned()
}
fn error() -> turbo_error {
    turbo_error {
        struct_size: size_of::<turbo_error>() as u32,
        code: 0,
        field: 0,
        message: [0; TURBO_ERROR_MESSAGE_LEN],
    }
}
fn check(code: i32, e: &turbo_error) -> Result<()> {
    if code == 0 { Ok(()) } else { Err(Error::field(code, e.field, string(&e.message))) }
}
fn text(s: &str) -> turbo_text {
    turbo_text { ptr: s.as_ptr().cast(), len: s.len() as u64 }
}

impl Reranker {
    /// Load a cross-encoder on the CPU. `threads=0` selects the runtime
    /// default. Another backend is chosen by name through
    /// [`Reranker::open_with`].
    pub fn open(path: &Path, threads: u32) -> Result<Self> {
        Self::open_with(
            path,
            &RerankSessionOptions { backend: Some("cpu".into()), cpu_threads: threads, ..Default::default() },
        )
    }

    pub fn open_with(path: &Path, o: &RerankSessionOptions) -> Result<Self> {
        let path = path.to_str().ok_or_else(|| Error::new(INVALID_ARGUMENT, "bundle path is not UTF-8"))?;
        let mut s = Self {
            runtime: ptr::null_mut(),
            context: ptr::null_mut(),
            model: ptr::null_mut(),
            session: ptr::null_mut(),
            identity: RerankIdentity {
                model_id: String::new(),
                revision: String::new(),
                manifest_sha256: String::new(),
                artifact_sha256: String::new(),
            },
            max_batch: 0,
            backend: String::new(),
            device: 0,
        };
        let mut e = error();
        unsafe {
            check(turbo_runtime_create(ptr::null(), &mut s.runtime, &mut e), &e)?;
            let device = match &o.backend {
                Some(name) => device_of(s.runtime, name)?,
                None => {
                    let mut device = 0;
                    check(
                        turbo_runtime_select(s.runtime, TURBO_TASK_RERANK, &mut device, ptr::null_mut(), 0, &mut e),
                        &e,
                    )?;
                    device
                }
            };
            let mut d: turbo_device_info = std::mem::zeroed();
            d.struct_size = size_of::<turbo_device_info>() as u32;
            check(turbo_runtime_device_info(s.runtime, device, &mut d, &mut e), &e)?;
            s.backend = string(&d.backend);
            s.device = device;
            check(turbo_context_create(s.runtime, device, &mut s.context, &mut e), &e)?;
            check(turbo_model_load(s.context, text(path), &mut s.model, &mut e), &e)?;
            let mut info: turbo_model_info = std::mem::zeroed();
            info.struct_size = size_of::<turbo_model_info>() as u32;
            check(turbo_model_get_info(s.model, &mut info, &mut e), &e)?;
            if info.task != TURBO_TASK_RERANK {
                return Err(Error::new(UNSUPPORTED_TASK, "model.task: expected TASK_RERANK"));
            }
            let desc = turbo_session_desc {
                struct_size: size_of::<turbo_session_desc>() as u32,
                max_batch: 0,
                max_seq: 0,
                precision: o.precision,
                tuning: TURBO_AUTOTUNE_OFF,
                tuning_budget_ms: 0,
                cpu_threads: o.cpu_threads,
                max_ranges: 0,
            };
            check(turbo_session_create(s.model, &desc, &mut s.session, &mut e), &e)?;
            s.max_batch = info.max_batch as usize;
            s.identity = RerankIdentity {
                model_id: string(&info.model_id),
                revision: string(&info.revision),
                manifest_sha256: string(&info.manifest_sha256),
                artifact_sha256: string(&info.artifact_sha256),
            };
        }
        Ok(s)
    }

    /// The device the session runs on.
    pub fn device_info(&self) -> Result<turbo_device_info> {
        let mut d: turbo_device_info = unsafe { std::mem::zeroed() };
        d.struct_size = size_of::<turbo_device_info>() as u32;
        let mut e = error();
        check(unsafe { turbo_runtime_device_info(self.runtime, self.device, &mut d, &mut e) }, &e)?;
        Ok(d)
    }

    /// The backend the session runs on ("cpu", "cuda", ...).
    pub fn backend(&self) -> &str {
        &self.backend
    }

    pub fn identity(&self) -> &RerankIdentity {
        &self.identity
    }

    /// Score any number of pairs in bounded batches. Empty input produces
    /// an empty result. Ties can be broken by the original candidate index.
    pub fn score(&mut self, pairs: &[(&str, &str)], options: RerankOptions) -> Result<RerankScores> {
        let mut scores = Vec::with_capacity(pairs.len());
        let opts = turbo_rerank_options {
            struct_size: size_of::<turbo_rerank_options>() as u32,
            query_tokens: options.query_tokens,
            max_tokens: options.max_tokens,
            stride: options.stride,
        };
        for batch in pairs.chunks(self.max_batch) {
            let inputs: Vec<_> =
                batch.iter().map(|&(q, p)| turbo_text_pair { query: text(q), passage: text(p) }).collect();
            let mut e = error();
            let mut result = OwnedResult(ptr::null_mut());
            unsafe {
                check(turbo_rerank_write_pairs(self.session, inputs.as_ptr(), inputs.len() as u32, &opts, &mut e), &e)?;
                check(turbo_session_run(self.session, &mut result.0, &mut e), &e)?;
                let mut info: turbo_result_info = std::mem::zeroed();
                info.struct_size = size_of::<turbo_result_info>() as u32;
                check(turbo_result_get_info(result.0, &mut info, &mut e), &e)?;
                if info.batch as usize != batch.len() || info.dim != 1 || info.dtype != TURBO_DTYPE_F32 {
                    return Err(Error::new(INTERNAL, "rerank result is not [pairs, 1] F32"));
                }
                let start = scores.len();
                scores.resize(start + batch.len(), 0.0f32);
                let mut written = 0;
                let bytes = (batch.len() * size_of::<f32>()) as u64;
                check(
                    turbo_result_read(result.0, scores[start..].as_mut_ptr().cast(), bytes, &mut written, &mut e),
                    &e,
                )?;
                if written != bytes {
                    return Err(Error::new(INTERNAL, "short rerank result read"));
                }
            }
        }
        Ok(RerankScores { model: self.identity.clone(), scores })
    }
}
/// The index of the first device whose backend is `name`.
unsafe fn device_of(rt: *mut turbo_runtime, name: &str) -> Result<u32> {
    let mut e = error();
    let mut n = 0;
    check(unsafe { turbo_runtime_device_count(rt, &mut n, &mut e) }, &e)?;
    for i in 0..n {
        let mut info: turbo_device_info = unsafe { std::mem::zeroed() };
        info.struct_size = size_of::<turbo_device_info>() as u32;
        check(unsafe { turbo_runtime_device_info(rt, i, &mut info, &mut e) }, &e)?;
        if string(&info.backend) == name {
            return Ok(i);
        }
    }
    Err(Error::new(crate::status::DEVICE_NOT_FOUND, format!("no {name} device is listed among {n}")))
}

struct OwnedResult(*mut turbo_result);
impl Drop for OwnedResult {
    fn drop(&mut self) {
        unsafe {
            turbo_result_release(self.0);
        }
    }
}
