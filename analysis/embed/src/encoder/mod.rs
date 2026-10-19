//! The core's runtime, model and session behind one small safe type,
//! driven through the C interface as any other caller drives it: rows of
//! token ids in, and a mean-pooled vector per range of their tokens out
//! (`turbo_embed_write_ranges`).

use std::ffi::c_char;
use std::path::Path;
use std::ptr;

use turbo::status::{DEVICE_NOT_FOUND, Error, INTERNAL, INVALID_ARGUMENT, UNSUPPORTED, UNSUPPORTED_OPTION};
use turbo::*;
use turbo_analysis::Execution;

/// The device a model runs on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Device {
    /// The host's processors, through the cpu backend.
    Cpu,
    /// The first device the backend of this name lists: "cuda", "metal",
    /// "levelzero" and so on, as `turbo_device_info.backend` names it.
    Backend(String),
    /// A device by its index in the runtime's list.
    Index(u32),
    /// The device `turbo_runtime_select` picks for embedding, which is
    /// never a CPU.
    Select,
}

/// The session's precision tier, `TURBO_PRECISION_*`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Precision {
    Model,
    Fastest,
    Exact,
}

impl Precision {
    fn code(self) -> u32 {
        match self {
            Precision::Model => TURBO_PRECISION_MODEL,
            Precision::Fastest => TURBO_PRECISION_FASTEST,
            Precision::Exact => TURBO_PRECISION_EXACT,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Precision::Model => "model",
            Precision::Fastest => "fastest",
            Precision::Exact => "exact",
        }
    }
}

/// Where and how the model runs. A 0 is the default the field names.
#[derive(Clone, Debug)]
pub struct EncoderOptions {
    pub device: Device,
    pub precision: Precision,
    /// Rows per run; 0 is the model's max_batch.
    pub max_batch: u32,
    /// Tokens per row, special tokens included; 0 is the model's max_seq.
    pub max_seq: u32,
    /// Ranges per run; 0 is 16 per row, at most one per token. The
    /// session holds memory for this many vectors.
    pub max_ranges: u32,
    /// The threads a CPU session computes on; 0 is the core's default
    /// (`TURBO_CPU_THREADS`, else one per processor). Another device
    /// takes 0 alone.
    pub cpu_threads: u32,
}

impl Default for EncoderOptions {
    fn default() -> Self {
        EncoderOptions {
            device: Device::Cpu,
            precision: Precision::Model,
            max_batch: 0,
            max_seq: 0,
            max_ranges: 0,
            cpu_threads: 0,
        }
    }
}

/// What a session resolved its options to, and the model's facts the
/// annotator needs.
#[derive(Clone, Debug)]
pub struct EncoderInfo {
    /// Values per vector.
    pub dim: u32,
    pub max_batch: u32,
    pub max_seq: u32,
    pub max_ranges: u32,
    /// The bundle's pooling, `TURBO_POOLING_*`.
    pub pooling: u32,
    /// Whether the bundle normalizes its vectors to unit length.
    pub l2: bool,
    /// The dtype the session computes in, `TURBO_DTYPE_*`.
    pub compute_dtype: u32,
    /// Backend, device and precision, for provenance. The precision is
    /// the dtype the session computes in, then the tier asked for:
    /// "f32 (model)", "f16 (fastest)".
    pub execution: Execution,
}

/// A loaded model and one session on it. A session has one owner at a
/// time, as the C interface says; `&mut self` holds it to that.
pub struct Encoder {
    rt: *mut turbo_runtime,
    ctx: *mut turbo_context,
    model: *mut turbo_model,
    session: *mut turbo_session,
    info: EncoderInfo,
    device: turbo_device_info,
    model_info: turbo_model_info,
}

// SAFETY: the runtime, context and model may be used from any thread, and
// the session from one owner at a time, which `&mut self` on every call
// that uses it makes so.
unsafe impl Send for Encoder {}

impl Drop for Encoder {
    fn drop(&mut self) {
        // Each release takes NULL, so a half-made encoder drops too.
        unsafe {
            turbo_session_release(self.session);
            turbo_model_release(self.model);
            turbo_context_release(self.ctx);
            turbo_runtime_release(self.rt);
        }
    }
}

fn new_error() -> turbo_error {
    turbo_error { struct_size: size_of::<turbo_error>() as u32, code: 0, field: 0, message: [0; 496] }
}

/// A fixed-size C string field as a String.
fn field(b: &[c_char]) -> String {
    let bytes: Vec<u8> = b.iter().take_while(|&&c| c != 0).map(|&c| u8::from_ne_bytes(c.to_ne_bytes())).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// A status from the C interface as the core's error, with its message.
fn check(rc: i32, err: &turbo_error) -> Result<(), Error> {
    if rc == 0 { Ok(()) } else { Err(Error::field(rc, err.field, field(&err.message))) }
}

fn dtype_name(dtype: u32) -> String {
    match dtype {
        TURBO_DTYPE_F32 => "f32".into(),
        TURBO_DTYPE_F16 => "f16".into(),
        TURBO_DTYPE_BF16 => "bf16".into(),
        TURBO_DTYPE_I8 => "i8".into(),
        TURBO_DTYPE_F64 => "f64".into(),
        d => format!("dtype {d}"),
    }
}

impl Encoder {
    /// The bundle at `dir` on the device `o` names, and a session on it.
    pub fn open(dir: &Path, o: &EncoderOptions) -> Result<Encoder, Error> {
        let path =
            dir.to_str().ok_or_else(|| Error::new(INVALID_ARGUMENT, format!("{}: not a UTF-8 path", dir.display())))?;
        let mut err = new_error();
        let mut e = Encoder {
            rt: ptr::null_mut(),
            ctx: ptr::null_mut(),
            model: ptr::null_mut(),
            session: ptr::null_mut(),
            info: EncoderInfo {
                dim: 0,
                max_batch: 0,
                max_seq: 0,
                max_ranges: 0,
                pooling: 0,
                l2: false,
                compute_dtype: 0,
                execution: Execution { backend: String::new(), device: String::new(), precision: String::new() },
            },
            // SAFETY: both are plain C data, for which all zeros is a value.
            device: unsafe { std::mem::zeroed() },
            model_info: unsafe { std::mem::zeroed() },
        };
        check(unsafe { turbo_runtime_create(ptr::null(), &mut e.rt, &mut err) }, &err)?;
        let index = e.device_index(&o.device)?;
        let device = e.device_info_at(index)?;
        check(unsafe { turbo_context_create(e.rt, index, &mut e.ctx, &mut err) }, &err)?;
        let text = turbo_text { ptr: path.as_ptr().cast(), len: path.len() as u64 };
        check(unsafe { turbo_model_load(e.ctx, text, &mut e.model, &mut err) }, &err)?;

        let mut m: turbo_model_info = unsafe { std::mem::zeroed() };
        m.struct_size = size_of::<turbo_model_info>() as u32;
        check(unsafe { turbo_model_get_info(e.model, &mut m, &mut err) }, &err)?;

        let max_batch = if o.max_batch == 0 { m.max_batch } else { o.max_batch };
        let max_seq = if o.max_seq == 0 { m.max_seq } else { o.max_seq };
        let max_ranges = match o.max_ranges {
            0 => max_batch.saturating_mul(16).min(max_batch.saturating_mul(max_seq)),
            n => n,
        };
        let desc = turbo_session_desc {
            struct_size: size_of::<turbo_session_desc>() as u32,
            max_batch,
            max_seq,
            precision: o.precision.code(),
            tuning: TURBO_AUTOTUNE_RUNTIME,
            tuning_budget_ms: 0,
            cpu_threads: o.cpu_threads,
            max_ranges,
        };
        let backend = field(&device.backend);
        check(unsafe { turbo_session_create(e.model, &desc, &mut e.session, &mut err) }, &err)
            .map_err(|x| needs_range_write(x, &backend))?;
        let mut s: turbo_session_info = unsafe { std::mem::zeroed() };
        s.struct_size = size_of::<turbo_session_info>() as u32;
        check(unsafe { turbo_session_get_info(e.session, &mut s, &mut err) }, &err)?;

        e.info = EncoderInfo {
            dim: m.dim,
            max_batch: s.max_batch,
            max_seq: s.max_seq,
            max_ranges,
            pooling: m.pooling,
            l2: m.normalize == TURBO_NORMALIZE_L2,
            compute_dtype: s.compute_dtype,
            execution: Execution {
                backend,
                device: field(&device.name),
                precision: format!("{} ({})", dtype_name(s.compute_dtype), o.precision.name()),
            },
        };
        e.device = device;
        e.model_info = m;
        Ok(e)
    }

    pub fn info(&self) -> &EncoderInfo {
        &self.info
    }

    /// The device the session runs on, as the runtime lists it.
    pub fn device_info(&self) -> &turbo_device_info {
        &self.device
    }

    /// The loaded model as `turbo_model_get_info` reports it: its id,
    /// revision and hashes among the rest.
    pub fn model_info(&self) -> &turbo_model_info {
        &self.model_info
    }

    /// The first CPU device the runtime lists, which names the host
    /// processor; None when this build lists none.
    pub fn host_cpu(&self) -> Result<Option<turbo_device_info>, Error> {
        for i in 0..self.device_count()? {
            let d = self.device_info_at(i)?;
            if d.kind == TURBO_DEVICE_CPU {
                return Ok(Some(d));
            }
        }
        Ok(None)
    }

    fn device_count(&self) -> Result<u32, Error> {
        let (mut n, mut err) = (0, new_error());
        check(unsafe { turbo_runtime_device_count(self.rt, &mut n, &mut err) }, &err)?;
        Ok(n)
    }

    fn device_info_at(&self, index: u32) -> Result<turbo_device_info, Error> {
        let mut info: turbo_device_info = unsafe { std::mem::zeroed() };
        info.struct_size = size_of::<turbo_device_info>() as u32;
        let mut err = new_error();
        check(unsafe { turbo_runtime_device_info(self.rt, index, &mut info, &mut err) }, &err)?;
        Ok(info)
    }

    /// The runtime's index of the device `d` names.
    fn device_index(&self, d: &Device) -> Result<u32, Error> {
        let n = self.device_count()?;
        let find = |what: &str, f: &dyn Fn(&turbo_device_info) -> bool| -> Result<u32, Error> {
            for i in 0..n {
                if f(&self.device_info_at(i)?) {
                    return Ok(i);
                }
            }
            Err(Error::new(DEVICE_NOT_FOUND, format!("no {what} device is listed among {n}")))
        };
        match d {
            Device::Cpu => find("cpu", &|i| i.kind == TURBO_DEVICE_CPU),
            Device::Backend(name) => find(name, &|i| field(&i.backend) == *name),
            Device::Index(i) if *i < n => Ok(*i),
            Device::Index(i) => Err(Error::new(DEVICE_NOT_FOUND, format!("device {i}: the runtime lists {n}"))),
            Device::Select => {
                let (mut out, mut err) = (0, new_error());
                let rc =
                    unsafe { turbo_runtime_select(self.rt, TURBO_TASK_EMBED, &mut out, ptr::null_mut(), 0, &mut err) };
                check(rc, &err)?;
                Ok(out)
            }
        }
    }
}

/// A result of `batch` vectors of `dim` values, as the run was written,
/// or why it is not.
fn result_shape(info: &turbo_result_info, batch: u32, dim: u32) -> Result<(), Error> {
    if info.batch != batch || info.dim != dim {
        return Err(Error::new(
            INTERNAL,
            format!("a result of {} vectors of {}, for {batch} vectors of {dim}", info.batch, info.dim),
        ));
    }
    Ok(())
}

/// The core's refusal of a range write, from a backend that pools no
/// ranges, as what the vector steps need; any other error as it is. The
/// core refuses at the session, when max_ranges is other than max_batch
/// (`UNSUPPORTED_OPTION` on field 7), and at the write, when the backend
/// does not offer `embed_write_ranges` (`UNSUPPORTED`).
fn needs_range_write(e: Error, backend: &str) -> Error {
    let at_session = e.code == UNSUPPORTED_OPTION && e.field == 7 && e.message.contains("pools no ranges");
    let at_write = e.code == UNSUPPORTED && e.message.contains("does not offer embed_write_ranges");
    if !(at_session || at_write) {
        return e;
    }
    Error::new(
        UNSUPPORTED,
        format!(
            "the {backend} backend pools no ranges: sentence and chunk vectors need a backend with the range write \
             (embed_write_ranges); {}",
            e.message
        ),
    )
}

fn options(pooling: u32, normalize: u32) -> turbo_embed_options {
    turbo_embed_options {
        struct_size: size_of::<turbo_embed_options>() as u32,
        truncate: 0,
        max_tokens: 0,
        prompt_role: 0,
        normalize,
        pooling,
        output_dim: 0,
    }
}

/// Scales `v` to unit length as the core's CPU backend does after
/// pooling: by one over the norm, summed in f64, or over 1e-12 when the
/// norm is smaller. A zero vector stays zero.
pub fn normalize(v: &mut [f32]) {
    let norm = v.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt().max(1e-12);
    let inv = (1.0 / norm) as f32;
    for x in v {
        *x *= inv;
    }
}

mod rows;
mod run;
pub use rows::Rows;
#[cfg(test)]
mod tests;
