use super::{Answer, DecodeOptions, QaWindow, WindowLogits, decode, windows};
use crate::{
    bundle::Bundle,
    head::Head,
    manifest::{Family, Qa},
    status::{Error, INTERNAL, INVALID_ARGUMENT, Result, UNSUPPORTED_TASK},
    tokenizer::Tokenizer,
    *,
};
use std::{path::Path, ptr};

/// Where and how an [`ExtractiveQa`] session runs.
#[derive(Clone, Debug)]
pub struct QaOptions {
    /// The backend to run on, by its name in `turbo_device_info.backend`
    /// ("cpu", "cuda", ...): its first device. `None` takes the device
    /// `turbo_runtime_select` picks for the task.
    pub backend: Option<String>,
    /// `TURBO_PRECISION_*`.
    pub precision: u32,
    /// The threads a CPU session computes on; 0 is the core's default.
    pub cpu_threads: u32,
    /// Windows per run; 0 is the model's max_batch.
    pub max_batch: u32,
}
impl Default for QaOptions {
    fn default() -> Self {
        Self { backend: None, precision: TURBO_PRECISION_EXACT, cpu_threads: 0, max_batch: 1 }
    }
}

/// Span-head inference. Each token is read through an unnormalized
/// singleton range, so the head receives the encoder's final hidden state.
pub struct ExtractiveQa {
    runtime: *mut turbo_runtime,
    context: *mut turbo_context,
    model: *mut turbo_model,
    session: *mut turbo_session,
    tokenizer: Tokenizer,
    head: Head,
    config: Qa,
    family: Family,
    max_batch: u32,
    device: u32,
    pub model_id: String,
    pub revision: String,
    pub manifest_sha256: String,
}
// SAFETY: handles can move across threads; inference needs &mut self.
unsafe impl Send for ExtractiveQa {}
impl Drop for ExtractiveQa {
    fn drop(&mut self) {
        unsafe {
            turbo_session_release(self.session);
            turbo_model_release(self.model);
            turbo_context_release(self.context);
            turbo_runtime_release(self.runtime);
        }
    }
}
fn error() -> turbo_error {
    turbo_error { struct_size: size_of::<turbo_error>() as u32, code: 0, field: 0, message: [0; 496] }
}
fn check(code: i32, e: &turbo_error) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    let bytes: Vec<_> = e.message.iter().take_while(|&&v| v != 0).map(|&v| v.to_ne_bytes()[0]).collect();
    Err(Error::field(code, e.field, String::from_utf8_lossy(&bytes)))
}
impl ExtractiveQa {
    /// The CPU, one window per run, at exact precision: the path the
    /// bundle check and the annotator take. Another backend is chosen by
    /// name through [`ExtractiveQa::open_with`].
    pub fn open(path: &Path, threads: u32) -> Result<Self> {
        Self::open_with(path, &QaOptions { backend: Some("cpu".into()), cpu_threads: threads, ..Default::default() })
    }
    pub fn open_with(path: &Path, o: &QaOptions) -> Result<Self> {
        let bundle = Bundle::open(path)?;
        if bundle.manifest.task != "TASK_EXTRACTIVE_QA" {
            return Err(Error::new(UNSUPPORTED_TASK, "expected TASK_EXTRACTIVE_QA"));
        }
        let m = &bundle.manifest;
        let config = m.qa.clone().ok_or_else(|| Error::new(INVALID_ARGUMENT, "missing QA configuration"))?;
        let family = m.architecture.as_ref().ok_or_else(|| Error::new(INVALID_ARGUMENT, "missing encoder"))?.family;
        let mut s = Self {
            runtime: ptr::null_mut(),
            context: ptr::null_mut(),
            model: ptr::null_mut(),
            session: ptr::null_mut(),
            tokenizer: Tokenizer::load(&bundle)?,
            head: Head::load(&bundle)?,
            family,
            config,
            max_batch: 0,
            device: 0,
            model_id: m.model.id.clone(),
            revision: m.model.revision.clone(),
            manifest_sha256: bundle.manifest_sha256.clone(),
        };
        let path = path.to_str().ok_or_else(|| Error::new(INVALID_ARGUMENT, "QA bundle path is not UTF-8"))?;
        let mut e = error();
        unsafe {
            check(turbo_runtime_create(ptr::null(), &mut s.runtime, &mut e), &e)?;
            let device = match &o.backend {
                Some(name) => device_of(s.runtime, name)?,
                None => {
                    let mut device = 0;
                    let rt = s.runtime;
                    check(
                        turbo_runtime_select(rt, TURBO_TASK_EXTRACTIVE_QA, &mut device, ptr::null_mut(), 0, &mut e),
                        &e,
                    )?;
                    device
                }
            };
            s.device = device;
            check(turbo_context_create(s.runtime, device, &mut s.context, &mut e), &e)?;
            let text = turbo_text { ptr: path.as_ptr().cast(), len: path.len() as u64 };
            check(turbo_model_load(s.context, text, &mut s.model, &mut e), &e)?;
            let desc = turbo_session_desc {
                struct_size: size_of::<turbo_session_desc>() as u32,
                max_batch: o.max_batch,
                max_seq: s.config.max_seq,
                precision: o.precision,
                tuning: TURBO_AUTOTUNE_OFF,
                tuning_budget_ms: 0,
                cpu_threads: o.cpu_threads,
                max_ranges: 0,
            };
            let mut info: turbo_model_info = std::mem::zeroed();
            info.struct_size = size_of::<turbo_model_info>() as u32;
            check(turbo_model_get_info(s.model, &mut info, &mut e), &e)?;
            s.max_batch = if o.max_batch == 0 { info.max_batch } else { o.max_batch };
            let desc =
                turbo_session_desc { max_batch: s.max_batch, max_ranges: s.max_batch * s.config.max_seq, ..desc };
            check(turbo_session_create(s.model, &desc, &mut s.session, &mut e), &e)?;
        }
        Ok(s)
    }
    /// The device the session runs on.
    pub fn device_info(&self) -> Result<turbo_device_info> {
        let mut info: turbo_device_info = unsafe { std::mem::zeroed() };
        info.struct_size = size_of::<turbo_device_info>() as u32;
        let mut e = error();
        check(unsafe { turbo_runtime_device_info(self.runtime, self.device, &mut info, &mut e) }, &e)?;
        Ok(info)
    }
    /// Windows per run.
    pub fn max_batch(&self) -> u32 {
        self.max_batch
    }
    pub fn config(&self) -> &Qa {
        &self.config
    }

    pub fn answer(&mut self, question: &str, context: &str, options: DecodeOptions) -> Result<Answer> {
        let logits = self.logits(question, context)?;
        decode(context, &logits, options)
    }
    pub fn logits(&mut self, question: &str, context: &str) -> Result<Vec<WindowLogits>> {
        Ok(self.logits_many(&[(question, context)])?.pop().unwrap_or_default())
    }
    /// The window logits of each `(question, context)` pair, the windows of
    /// all pairs run `max_batch` at a time.
    pub fn logits_many(&mut self, pairs: &[(&str, &str)]) -> Result<Vec<Vec<WindowLogits>>> {
        let mut owner = Vec::new();
        let mut inputs = Vec::new();
        for (n, (question, context)) in pairs.iter().enumerate() {
            for w in windows(&self.tokenizer, self.family, &self.config, question, context)? {
                owner.push(n);
                inputs.push(w);
            }
        }
        let mut out: Vec<Vec<WindowLogits>> = vec![Vec::new(); pairs.len()];
        let mut at = 0;
        for chunk in inputs.chunks(self.max_batch as usize) {
            for (w, pairs) in chunk.iter().zip(self.run(chunk)?) {
                out[owner[at]].push(WindowLogits {
                    start: pairs.iter().map(|p| p[0]).collect(),
                    end: pairs.iter().map(|p| p[1]).collect(),
                    context: w.context.clone(),
                    max_context: w.max_context.clone(),
                    cls: 0,
                });
                at += 1;
            }
        }
        Ok(out)
    }
    /// The start and end logits of each token of each window in `ws`.
    fn run(&mut self, ws: &[QaWindow]) -> Result<Vec<Vec<[f32; 2]>>> {
        let seq = ws.iter().map(|w| w.ids.len()).max().unwrap_or(0) as u32;
        let rows = ws.len();
        let fill = self.tokenizer.fill_id();
        let (mut ids, mut types, mut mask) = (Vec::new(), Vec::new(), Vec::new());
        let mut ranges = Vec::new();
        for (r, w) in ws.iter().enumerate() {
            let pad = seq as usize - w.ids.len();
            ids.extend(w.ids.iter().copied().chain(std::iter::repeat_n(fill, pad)));
            types.extend(w.types.iter().copied().chain(std::iter::repeat_n(0, pad)));
            mask.extend(std::iter::repeat_n(1, w.ids.len()).chain(std::iter::repeat_n(0, pad)));
            ranges.extend((0..w.ids.len() as u32).map(|i| turbo_token_range { row: r as u32, start: i, end: i + 1 }));
        }
        let n = ranges.len() as u32;
        let batch = turbo_token_batch {
            struct_size: size_of::<turbo_token_batch>() as u32,
            batch: rows as u32,
            seq,
            row_stride: seq,
            ids: ids.as_ptr(),
            mask: mask.as_ptr(),
            types: types.as_ptr(),
        };
        let opts = turbo_embed_options {
            struct_size: size_of::<turbo_embed_options>() as u32,
            truncate: 0,
            max_tokens: 0,
            prompt_role: 0,
            normalize: TURBO_NORMALIZE_NONE,
            pooling: TURBO_POOLING_MEAN,
            output_dim: 0,
        };
        let mut e = error();
        let mut result = Owned(ptr::null_mut());
        unsafe {
            check(turbo_embed_write_ranges(self.session, &batch, ranges.as_ptr(), n, &opts, &mut e), &e)?;
            check(turbo_session_run(self.session, &mut result.0, &mut e), &e)?;
            let mut info: turbo_result_info = std::mem::zeroed();
            info.struct_size = size_of::<turbo_result_info>() as u32;
            check(turbo_result_get_info(result.0, &mut info, &mut e), &e)?;
            // A device that applies the span head returns each token's two
            // logits (turbo.h); elsewhere its final hidden state. Width 2
            // is also valid on the CPU, so shape cannot identify the path.
            let labels = self.head.classifier.out_dim;
            let on_device = model_handle(self.model)?.inner.span_head.is_some();
            let dim = if on_device { labels } else { self.head.hidden };
            if info.batch != n || info.dim as usize != dim || info.dtype != TURBO_DTYPE_F32 {
                return Err(Error::new(INTERNAL, "QA encoder returned an unexpected state shape"));
            }
            if on_device {
                let mut logits = vec![[0f32; 2]; n as usize];
                let bytes = size_of_val(logits.as_slice()) as u64;
                let mut written = 0;
                check(turbo_result_read(result.0, logits.as_mut_ptr().cast(), bytes, &mut written, &mut e), &e)?;
                if written != bytes {
                    return Err(Error::new(INTERNAL, "QA head returned a short read"));
                }
                if logits.iter().flatten().any(|v| !v.is_finite()) {
                    return Err(Error::new(INTERNAL, "QA head logits overflow F32"));
                }
                let mut tokens = logits.into_iter();
                return Ok(ws.iter().map(|w| tokens.by_ref().take(w.ids.len()).collect()).collect());
            }
            let mut states = vec![0f32; n as usize * self.head.hidden];
            let bytes = size_of_val(states.as_slice()) as u64;
            let mut written = 0;
            check(turbo_result_read(result.0, states.as_mut_ptr().cast(), bytes, &mut written, &mut e), &e)?;
            if written != bytes || states.iter().any(|v| !v.is_finite()) {
                return Err(Error::new(INTERNAL, "QA encoder returned invalid states"));
            }
            let h = &self.head.classifier;
            let mut tokens = states.chunks_exact(h.in_dim);
            ws.iter()
                .map(|w| {
                    tokens
                        .by_ref()
                        .take(w.ids.len())
                        .map(|x| {
                            let mut logits = [0f32; 2];
                            for (label, logit) in logits.iter_mut().enumerate() {
                                let weights = &h.weight[label * h.in_dim..(label + 1) * h.in_dim];
                                *logit = weights
                                    .iter()
                                    .zip(x)
                                    .fold(f64::from(h.bias[label]), |a, (&w, &v)| a + f64::from(w) * f64::from(v))
                                    as f32;
                            }
                            if logits.iter().any(|v| !v.is_finite()) {
                                return Err(Error::new(INTERNAL, "QA head logits overflow F32"));
                            }
                            Ok(logits)
                        })
                        .collect()
                })
                .collect()
        }
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
        let b: Vec<u8> = info.backend.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
        if b == name.as_bytes() {
            return Ok(i);
        }
    }
    Err(Error::new(crate::status::DEVICE_NOT_FOUND, format!("no {name} device is listed among {n}")))
}
struct Owned(*mut turbo_result);
impl Drop for Owned {
    fn drop(&mut self) {
        unsafe { turbo_result_release(self.0) };
    }
}
