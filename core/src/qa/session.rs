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

/// CPU span-head inference. Each token is read through an unnormalized
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
    pub fn open(path: &Path, threads: u32) -> Result<Self> {
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
            model_id: m.model.id.clone(),
            revision: m.model.revision.clone(),
            manifest_sha256: bundle.manifest_sha256.clone(),
        };
        let path = path.to_str().ok_or_else(|| Error::new(INVALID_ARGUMENT, "QA bundle path is not UTF-8"))?;
        let mut e = error();
        unsafe {
            check(turbo_runtime_create(ptr::null(), &mut s.runtime, &mut e), &e)?;
            let mut device = 0;
            check(
                turbo_runtime_select(s.runtime, TURBO_TASK_EXTRACTIVE_QA, &mut device, ptr::null_mut(), 0, &mut e),
                &e,
            )?;
            check(turbo_context_create(s.runtime, device, &mut s.context, &mut e), &e)?;
            let text = turbo_text { ptr: path.as_ptr().cast(), len: path.len() as u64 };
            check(turbo_model_load(s.context, text, &mut s.model, &mut e), &e)?;
            let desc = turbo_session_desc {
                struct_size: size_of::<turbo_session_desc>() as u32,
                max_batch: 1,
                max_seq: s.config.max_seq,
                precision: TURBO_PRECISION_EXACT,
                tuning: TURBO_AUTOTUNE_OFF,
                tuning_budget_ms: 0,
                cpu_threads: threads,
                max_ranges: s.config.max_seq,
            };
            check(turbo_session_create(s.model, &desc, &mut s.session, &mut e), &e)?;
        }
        Ok(s)
    }
    pub fn config(&self) -> &Qa {
        &self.config
    }

    pub fn answer(&mut self, question: &str, context: &str, options: DecodeOptions) -> Result<Answer> {
        let logits = self.logits(question, context)?;
        decode(context, &logits, options)
    }
    pub fn logits(&mut self, question: &str, context: &str) -> Result<Vec<WindowLogits>> {
        let inputs = windows(&self.tokenizer, self.family, &self.config, question, context)?;
        inputs
            .into_iter()
            .map(|w| {
                let pairs = self.run(&w)?;
                Ok(WindowLogits {
                    start: pairs.iter().map(|p| p[0]).collect(),
                    end: pairs.iter().map(|p| p[1]).collect(),
                    context: w.context,
                    max_context: w.max_context,
                    cls: 0,
                })
            })
            .collect()
    }
    fn run(&mut self, w: &QaWindow) -> Result<Vec<[f32; 2]>> {
        let n = w.ids.len() as u32;
        let mask = vec![1; n as usize];
        let batch = turbo_token_batch {
            struct_size: size_of::<turbo_token_batch>() as u32,
            batch: 1,
            seq: n,
            row_stride: n,
            ids: w.ids.as_ptr(),
            mask: mask.as_ptr(),
            types: w.types.as_ptr(),
        };
        let ranges: Vec<_> = (0..n).map(|i| turbo_token_range { row: 0, start: i, end: i + 1 }).collect();
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
            if info.batch != n || info.dim as usize != self.head.hidden || info.dtype != TURBO_DTYPE_F32 {
                return Err(Error::new(INTERNAL, "QA encoder returned an unexpected state shape"));
            }
            let mut states = vec![0f32; n as usize * self.head.hidden];
            let bytes = size_of_val(states.as_slice()) as u64;
            let mut written = 0;
            check(turbo_result_read(result.0, states.as_mut_ptr().cast(), bytes, &mut written, &mut e), &e)?;
            if written != bytes || states.iter().any(|v| !v.is_finite()) {
                return Err(Error::new(INTERNAL, "QA encoder returned invalid states"));
            }
            let h = &self.head.classifier;
            states
                .chunks_exact(h.in_dim)
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
        }
    }
}
struct Owned(*mut turbo_result);
impl Drop for Owned {
    fn drop(&mut self) {
        unsafe { turbo_result_release(self.0) };
    }
}
