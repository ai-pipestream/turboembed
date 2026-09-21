//! Conversions between the core's typed structs and the `#[repr(C)]` ABI
//! structs. Used by the C API (`turbo-capi`) in one direction and by the
//! plugin adapter and exporter in both. Every ABI → core conversion validates
//! constants and sizes; every core → ABI conversion writes only the bytes the
//! caller's `struct_size` covers and cuts strings on UTF-8 boundaries.

use std::ffi::c_char;
use std::ptr::NonNull;

use turbo_abi as abi;

use crate::buffer::{BufferDesc, NativeHandle};
use crate::error::{Error, Result};
use crate::provider::{
    Capability, ClassifyOptions, DeviceInfo, EmbedOptions, GenerateDesc, ModelInfo, Options, RerankOptions,
    SessionStats, Span, TensorInfo,
};
use crate::types::{
    Aggregation, CapStatus, DType, DeviceKind, HandleKind, Modality, ModelKind, Normalize, OutputDType, Pooling,
    PromptRole, StagePlacement, StagePlacements, StructuredKind, Task, Truncate,
};

/// Check that a descriptor's `struct_size` is a layout this library
/// understands: the current size, or the end of an earlier field (an older
/// caller's layout; see [`turbo_abi::Versioned`]). Any other value is
/// `TURBO_E_INVALID_STRUCT_SIZE`, since a prefix ending inside a field could
/// hand the library half a pointer or half a count.
pub fn check_size<T: turbo_abi::Versioned>(what: &str, got: u32) -> Result<()> {
    if T::size_is_known(got) {
        return Ok(());
    }
    Err(Error::invalid_struct_size(what, got, core::mem::size_of::<T>()))
}

/// Read an input descriptor, copying only the `struct_size` bytes the caller
/// declared into a zero-initialized full struct. Fields the caller did not
/// declare therefore read as zero ("old behavior") and are never touched in
/// the caller's memory.
///
/// # Safety
/// `ptr` must be NULL or point to at least `struct_size` readable bytes whose
/// first four bytes are the `struct_size` field.
pub unsafe fn read_sized<T: Copy + turbo_abi::Versioned>(ptr: *const T, what: &str) -> Result<T> {
    if ptr.is_null() {
        return Err(Error::invalid_argument(format!("{what} is NULL")));
    }
    // SAFETY: every ABI descriptor starts with `uint32_t struct_size`.
    let size = unsafe { ptr.cast::<u32>().read_unaligned() };
    check_size::<T>(what, size)?;
    // SAFETY: all-zero is a valid value for every ABI descriptor (integers,
    // floats, null pointers, `None` function pointers).
    let mut out: T = unsafe { std::mem::zeroed() };
    // SAFETY: `size <= size_of::<T>()` was checked; the source is readable for `size` bytes.
    unsafe { std::ptr::copy_nonoverlapping(ptr.cast::<u8>(), (&mut out as *mut T).cast::<u8>(), size as usize) };
    Ok(out)
}

/// Copy `src` into `dst`, writing only the first `struct_size` bytes the
/// caller declared (already validated with [`check_size`]).
///
/// # Safety
/// `dst` must point to at least `struct_size` writable bytes.
pub unsafe fn write_sized<T: Copy>(src: &T, dst: *mut T, struct_size: u32) {
    // SAFETY: the caller validated struct_size <= size_of::<T>() and dst is writable for that many bytes.
    unsafe {
        std::ptr::copy_nonoverlapping((src as *const T).cast::<u8>(), dst.cast::<u8>(), struct_size as usize);
    }
}

/// Copy a string into a fixed `c_char` array, NUL-terminated, cut on a UTF-8 boundary.
pub fn put_str(dst: &mut [c_char], s: &str) {
    if dst.is_empty() {
        return;
    }
    let bytes = s.as_bytes();
    let mut n = bytes.len().min(dst.len() - 1);
    while n > 0 && n < bytes.len() && (bytes[n] & 0xC0) == 0x80 {
        n -= 1;
    }
    for (i, b) in bytes[..n].iter().enumerate() {
        dst[i] = *b as c_char;
    }
    dst[n] = 0;
}

/// Read a NUL-terminated fixed `c_char` array as a `String` (lossy on invalid UTF-8).
pub fn get_str(src: &[c_char]) -> String {
    let bytes: Vec<u8> = src.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Borrow a `turbo_text` as `&str`, validating UTF-8.
///
/// # Safety
/// `t.ptr` must be readable for `t.len` bytes for the returned lifetime.
pub unsafe fn text<'a>(t: &abi::turbo_text, what: &str) -> Result<&'a str> {
    if t.len == 0 {
        return Ok("");
    }
    if t.ptr.is_null() {
        return Err(Error::invalid_argument(format!("{what}.ptr is NULL with len {}", t.len)));
    }
    let len = usize::try_from(t.len).map_err(|_| Error::invalid_argument(format!("{what}.len exceeds usize")))?;
    // SAFETY: caller contract.
    let bytes = unsafe { std::slice::from_raw_parts(t.ptr.cast::<u8>(), len) };
    std::str::from_utf8(bytes).map_err(|e| Error::invalid_utf8(&format!("{what} (byte {})", e.valid_up_to())))
}

/// Borrow an array of `turbo_text` as `&str`s.
///
/// # Safety
/// `ptr` must point to `n` readable `turbo_text` entries, each valid per [`text`].
pub unsafe fn texts<'a>(ptr: *const abi::turbo_text, n: u32, what: &str) -> Result<Vec<&'a str>> {
    if n == 0 {
        return Ok(Vec::new());
    }
    if ptr.is_null() {
        return Err(Error::invalid_argument(format!("{what} is NULL with count {n}")));
    }
    // SAFETY: caller contract.
    let raw = unsafe { std::slice::from_raw_parts(ptr, n as usize) };
    raw.iter().enumerate().map(|(i, t)| unsafe { text(t, &format!("{what}[{i}]")) }).collect()
}

/// Borrow key/value options.
///
/// # Safety
/// `ptr` must point to `n` readable `turbo_kv` entries with valid views.
pub unsafe fn kvs(ptr: *const abi::turbo_kv, n: u32, what: &str) -> Result<Options> {
    if n == 0 {
        return Ok(Options::default());
    }
    if ptr.is_null() {
        return Err(Error::invalid_argument(format!("{what} is NULL with count {n}")));
    }
    let raw = unsafe { std::slice::from_raw_parts(ptr, n as usize) };
    let mut out = Vec::with_capacity(raw.len());
    for (i, kv) in raw.iter().enumerate() {
        let k = unsafe { text(&kv.key, &format!("{what}[{i}].key")) }?;
        let v = unsafe { text(&kv.value, &format!("{what}[{i}].value")) }?;
        if k.is_empty() {
            return Err(Error::invalid_argument(format!("{what}[{i}].key is empty")));
        }
        out.push((k.to_string(), v.to_string()));
    }
    Ok(Options(out))
}

/// A `turbo_text` view over a `str`.
pub fn text_of(s: &str) -> abi::turbo_text {
    abi::turbo_text { ptr: s.as_ptr().cast(), len: s.len() as u64 }
}

/// Owned storage for a list of `turbo_kv` views built from `Options`.
pub struct KvViews {
    _owned: Vec<(String, String)>,
    views: Vec<abi::turbo_kv>,
}

impl KvViews {
    /// Build views. The views borrow from this struct; keep it alive for the call.
    pub fn new(options: &Options) -> Self {
        let owned = options.0.clone();
        let views = owned.iter().map(|(k, v)| abi::turbo_kv { key: text_of(k), value: text_of(v) }).collect();
        Self { _owned: owned, views }
    }

    /// Pointer to the views (NULL when empty).
    pub fn ptr(&self) -> *const abi::turbo_kv {
        if self.views.is_empty() {
            std::ptr::null()
        } else {
            self.views.as_ptr()
        }
    }

    /// Count.
    pub fn len(&self) -> u32 {
        self.views.len() as u32
    }

    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.views.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Device info and capability
// ---------------------------------------------------------------------------

/// Core → ABI.
pub fn device_info_to_abi(d: &DeviceInfo, struct_size: u32) -> abi::turbo_device_info {
    let mut out = abi::turbo_device_info {
        struct_size,
        kind: d.kind.as_abi(),
        ordinal: d.ordinal,
        vendor_id: d.vendor_id,
        caps: d.caps,
        memory_total: d.memory_total,
        memory_free: d.memory_free,
        name: [0; 128],
        vendor: [0; 64],
        provider_id: [0; 32],
        provider_version: [0; 32],
        runtime_version: [0; 64],
        driver_version: [0; 64],
    };
    put_str(&mut out.name, &d.name);
    put_str(&mut out.vendor, &d.vendor);
    put_str(&mut out.provider_id, &d.provider_id);
    put_str(&mut out.provider_version, &d.provider_version);
    put_str(&mut out.runtime_version, &d.runtime_version);
    put_str(&mut out.driver_version, &d.driver_version);
    out
}

/// ABI → core.
pub fn device_info_from_abi(d: &abi::turbo_device_info) -> Result<DeviceInfo> {
    Ok(DeviceInfo {
        kind: DeviceKind::from_abi(d.kind)?,
        ordinal: d.ordinal,
        vendor_id: d.vendor_id,
        caps: d.caps,
        memory_total: d.memory_total,
        memory_free: d.memory_free,
        name: get_str(&d.name),
        vendor: get_str(&d.vendor),
        provider_id: get_str(&d.provider_id),
        provider_version: get_str(&d.provider_version),
        runtime_version: get_str(&d.runtime_version),
        driver_version: get_str(&d.driver_version),
    })
}

/// Core → ABI.
pub fn capability_to_abi(c: &Capability, struct_size: u32) -> abi::turbo_capability {
    let mut out = abi::turbo_capability {
        struct_size,
        status: c.status.as_abi(),
        dtype: c.dtype.map(|d| d.as_abi()).unwrap_or(0),
        reference_dtype: c.reference_dtype.map(|d| d.as_abi()).unwrap_or(0),
        cosine_floor: c.cosine_floor,
        max_abs_error: c.max_abs_error,
        deterministic: c.deterministic as u32,
        reserved: 0,
        notes: [0; 128],
    };
    put_str(&mut out.notes, &c.notes);
    out
}

/// ABI → core.
pub fn capability_from_abi(c: &abi::turbo_capability) -> Result<Capability> {
    Ok(Capability {
        status: CapStatus::from_abi(c.status)?,
        dtype: if c.dtype == 0 { None } else { Some(DType::from_abi(c.dtype)?) },
        reference_dtype: if c.reference_dtype == 0 { None } else { Some(DType::from_abi(c.reference_dtype)?) },
        cosine_floor: c.cosine_floor,
        max_abs_error: c.max_abs_error,
        deterministic: c.deterministic != 0,
        notes: get_str(&c.notes),
    })
}

// ---------------------------------------------------------------------------
// Buffers
// ---------------------------------------------------------------------------

/// Core → ABI.
pub fn buffer_desc_to_abi(d: &BufferDesc, struct_size: u32) -> abi::turbo_buffer_desc {
    let mut out = abi::turbo_buffer_desc {
        struct_size,
        placement: d.placement.as_abi(),
        dtype: d.dtype.as_abi(),
        ndim: d.shape.len() as u32,
        shape: [0; abi::TURBO_MAX_RANK],
        strides: [0; abi::TURBO_MAX_RANK],
        bytes: d.bytes,
        next: std::ptr::null(),
    };
    out.shape[..d.shape.len()].copy_from_slice(&d.shape);
    out.strides[..d.strides.len()].copy_from_slice(&d.strides);
    out
}

/// Core → ABI.
pub fn native_handle_to_abi(h: &NativeHandle) -> abi::turbo_native_handle {
    abi::turbo_native_handle {
        struct_size: std::mem::size_of::<abi::turbo_native_handle>() as u32,
        kind: h.kind.as_abi(),
        handle: h.handle,
        aux: h.aux,
        offset: h.offset,
    }
}

/// ABI → core.
pub fn native_handle_from_abi(h: &abi::turbo_native_handle) -> Result<NativeHandle> {
    // SAFETY: a reference is readable for its declared prefix.
    let h = unsafe { read_sized::<abi::turbo_native_handle>(h, "turbo_native_handle") }?;
    Ok(NativeHandle { kind: HandleKind::from_abi(h.kind)?, handle: h.handle, aux: h.aux, offset: h.offset })
}

/// Parse a buffer descriptor plus an optional native handle in `next`.
///
/// # Safety
/// `desc` must be NULL or point to a readable `turbo_buffer_desc`; a non-NULL
/// `next` must point to a readable `turbo_native_handle`.
pub unsafe fn buffer_desc_from_abi(desc: *const abi::turbo_buffer_desc) -> Result<(BufferDesc, Option<NativeHandle>)> {
    if desc.is_null() {
        return Err(Error::invalid_argument("turbo_buffer_desc is NULL"));
    }
    let d = unsafe { read_sized::<abi::turbo_buffer_desc>(desc, "turbo_buffer_desc") }?;
    let bd = BufferDesc::from_abi(&d)?;
    let native = if d.next.is_null() {
        None
    } else {
        Some(native_handle_from_abi(unsafe { &*(d.next as *const abi::turbo_native_handle) })?)
    };
    Ok((bd, native))
}

/// Non-null host pointer from an ABI pointer.
pub fn host_ptr(p: *mut std::ffi::c_void) -> Option<NonNull<u8>> {
    NonNull::new(p.cast::<u8>())
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

/// Core → ABI (labels, inputs, and outputs are exposed through separate calls).
pub fn model_info_to_abi(i: &ModelInfo, struct_size: u32) -> abi::turbo_model_info {
    let mut out = abi::turbo_model_info {
        struct_size,
        task: i.task.as_abi(),
        kind: i.kind.as_abi(),
        modality: i.modality.as_abi(),
        dim: i.dim,
        n_labels: i.labels.len() as u32,
        pooling: i.pooling.map(|p| p.as_abi()).unwrap_or(0),
        normalize: i.normalize.map(|n| n.as_abi()).unwrap_or(0),
        max_seq: i.max_seq,
        max_batch: i.max_batch,
        dtype_used: i.dtype_used.map(|d| d.as_abi()).unwrap_or(0),
        fully_accelerated: i.stages.fully_accelerated() as u32,
        stage_placement: i.stages.as_abi(),
        n_inputs: i.inputs.len() as u32,
        n_outputs: i.outputs.len() as u32,
        vocab_size: i.vocab_size,
        model_id: [0; 128],
        revision: [0; 64],
        tokenizer_sha256: [0; 72],
        provider_id: [0; 32],
        prefix_query: [0; 128],
        prefix_document: [0; 128],
    };
    put_str(&mut out.model_id, &i.model_id);
    put_str(&mut out.revision, &i.revision);
    put_str(&mut out.tokenizer_sha256, &i.tokenizer_sha256);
    put_str(&mut out.provider_id, &i.provider_id);
    put_str(&mut out.prefix_query, &i.prefix_query);
    put_str(&mut out.prefix_document, &i.prefix_document);
    out
}

/// ABI → core. `labels`, `inputs`, `outputs`, and `aggregation` are supplied
/// by the caller, which fetched them through the separate calls.
pub fn model_info_from_abi(
    m: &abi::turbo_model_info,
    labels: Vec<String>,
    inputs: Vec<TensorInfo>,
    outputs: Vec<TensorInfo>,
    aggregation: Option<Aggregation>,
) -> Result<ModelInfo> {
    let mut stages = [StagePlacement::Unused; abi::TURBO_STAGE_COUNT];
    for (s, v) in stages.iter_mut().zip(m.stage_placement.iter()) {
        *s = StagePlacement::from_abi(*v)?;
    }
    Ok(ModelInfo {
        task: Task::from_abi(m.task)?,
        kind: ModelKind::from_abi(m.kind)?,
        modality: Modality::from_abi(m.modality)?,
        dim: m.dim,
        labels,
        pooling: if m.pooling == 0 { None } else { Some(Pooling::from_abi(m.pooling)?) },
        normalize: if m.normalize == 0 { None } else { Some(Normalize::from_abi(m.normalize)?) },
        aggregation,
        max_seq: m.max_seq,
        max_batch: m.max_batch,
        dtype_used: if m.dtype_used == 0 { None } else { Some(DType::from_abi(m.dtype_used)?) },
        stages: StagePlacements(stages),
        inputs,
        outputs,
        vocab_size: m.vocab_size,
        model_id: get_str(&m.model_id),
        revision: get_str(&m.revision),
        tokenizer_sha256: get_str(&m.tokenizer_sha256),
        provider_id: get_str(&m.provider_id),
        prefix_query: get_str(&m.prefix_query),
        prefix_document: get_str(&m.prefix_document),
    })
}

/// Core → ABI.
pub fn tensor_info_to_abi(t: &TensorInfo, struct_size: u32) -> Result<abi::turbo_tensor_info> {
    if t.shape.len() > abi::TURBO_MAX_RANK {
        return Err(Error::internal(format!(
            "tensor `{}` has rank {} > {}",
            t.name,
            t.shape.len(),
            abi::TURBO_MAX_RANK
        )));
    }
    let mut out = abi::turbo_tensor_info {
        struct_size,
        dtype: t.dtype.as_abi(),
        ndim: t.shape.len() as u32,
        reserved: 0,
        shape: [0; abi::TURBO_MAX_RANK],
        name: [0; 64],
    };
    out.shape[..t.shape.len()].copy_from_slice(&t.shape);
    put_str(&mut out.name, &t.name);
    Ok(out)
}

/// ABI → core.
pub fn tensor_info_from_abi(t: &abi::turbo_tensor_info) -> Result<TensorInfo> {
    if t.ndim as usize > abi::TURBO_MAX_RANK {
        return Err(Error::invalid_shape(format!("tensor ndim {} > {}", t.ndim, abi::TURBO_MAX_RANK)));
    }
    Ok(TensorInfo {
        name: get_str(&t.name),
        dtype: DType::from_abi(t.dtype)?,
        shape: t.shape[..t.ndim as usize].to_vec(),
    })
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// ABI → core. NULL means defaults.
///
/// # Safety
/// `opts` must be NULL or point to a readable `turbo_embed_options`.
pub unsafe fn embed_options_from_abi(opts: *const abi::turbo_embed_options) -> Result<EmbedOptions> {
    if opts.is_null() {
        return Ok(EmbedOptions::default());
    }
    let o = unsafe { read_sized::<abi::turbo_embed_options>(opts, "turbo_embed_options") }?;
    Ok(EmbedOptions {
        truncate: Truncate::from_abi(o.truncate).map_err(|e| e.with_field(EmbedOptions::FIELD_TRUNCATE))?,
        max_tokens: o.max_tokens,
        prompt_role: PromptRole::from_abi(o.prompt_role).map_err(|e| e.with_field(EmbedOptions::FIELD_PROMPT_ROLE))?,
        normalize: Normalize::from_abi(o.normalize).map_err(|e| e.with_field(EmbedOptions::FIELD_NORMALIZE))?,
        pooling: Pooling::from_abi(o.pooling).map_err(|e| e.with_field(EmbedOptions::FIELD_POOLING))?,
        output_dim: o.output_dim,
        output_dtype: OutputDType::from_abi(o.output_dtype)
            .map_err(|e| e.with_field(EmbedOptions::FIELD_OUTPUT_DTYPE))?,
    })
}

/// Core → ABI.
pub fn embed_options_to_abi(o: &EmbedOptions) -> abi::turbo_embed_options {
    abi::turbo_embed_options {
        struct_size: std::mem::size_of::<abi::turbo_embed_options>() as u32,
        truncate: o.truncate.as_abi(),
        max_tokens: o.max_tokens,
        prompt_role: o.prompt_role.as_abi(),
        normalize: o.normalize.as_abi(),
        pooling: o.pooling.as_abi(),
        output_dim: o.output_dim,
        output_dtype: o.output_dtype.as_abi(),
    }
}

fn flag(v: u32, name: &str, field: u32) -> Result<bool> {
    match v {
        0 => Ok(false),
        1 => Ok(true),
        other => Err(Error::invalid_argument(format!("{name} must be 0 or 1, got {other}")).with_field(field)),
    }
}

/// ABI → core.
///
/// # Safety
/// `opts` must be NULL or point to a readable `turbo_rerank_options`.
pub unsafe fn rerank_options_from_abi(opts: *const abi::turbo_rerank_options) -> Result<RerankOptions> {
    if opts.is_null() {
        return Ok(RerankOptions::default());
    }
    let o = unsafe { read_sized::<abi::turbo_rerank_options>(opts, "turbo_rerank_options") }?;
    Ok(RerankOptions {
        truncate: Truncate::from_abi(o.truncate).map_err(|e| e.with_field(RerankOptions::FIELD_TRUNCATE))?,
        max_tokens: o.max_tokens,
        top_n: o.top_n,
        return_sorted: flag(o.return_sorted, "return_sorted", RerankOptions::FIELD_RETURN_SORTED)?,
        raw_scores: flag(o.raw_scores, "raw_scores", RerankOptions::FIELD_RAW_SCORES)?,
    })
}

/// Core → ABI.
pub fn rerank_options_to_abi(o: &RerankOptions) -> abi::turbo_rerank_options {
    abi::turbo_rerank_options {
        struct_size: std::mem::size_of::<abi::turbo_rerank_options>() as u32,
        truncate: o.truncate.as_abi(),
        max_tokens: o.max_tokens,
        top_n: o.top_n,
        return_sorted: o.return_sorted as u32,
        raw_scores: o.raw_scores as u32,
    }
}

/// ABI → core.
///
/// # Safety
/// `opts` must be NULL or point to a readable `turbo_classify_options`.
pub unsafe fn classify_options_from_abi(opts: *const abi::turbo_classify_options) -> Result<ClassifyOptions> {
    if opts.is_null() {
        return Ok(ClassifyOptions::default());
    }
    let o = unsafe { read_sized::<abi::turbo_classify_options>(opts, "turbo_classify_options") }?;
    if o.reserved != 0 {
        return Err(Error::invalid_argument("turbo_classify_options.reserved must be 0").with_field(6));
    }
    Ok(ClassifyOptions {
        truncate: Truncate::from_abi(o.truncate).map_err(|e| e.with_field(ClassifyOptions::FIELD_TRUNCATE))?,
        max_tokens: o.max_tokens,
        aggregation: Aggregation::from_abi(o.aggregation)
            .map_err(|e| e.with_field(ClassifyOptions::FIELD_AGGREGATION))?,
        raw_scores: flag(o.raw_scores, "raw_scores", ClassifyOptions::FIELD_RAW_SCORES)?,
    })
}

/// Core → ABI.
pub fn classify_options_to_abi(o: &ClassifyOptions) -> abi::turbo_classify_options {
    abi::turbo_classify_options {
        struct_size: std::mem::size_of::<abi::turbo_classify_options>() as u32,
        truncate: o.truncate.as_abi(),
        max_tokens: o.max_tokens,
        aggregation: o.aggregation.as_abi(),
        raw_scores: o.raw_scores as u32,
        reserved: 0,
    }
}

/// ABI → core.
///
/// # Safety
/// `desc` must be NULL or point to a readable `turbo_generate_desc` whose
/// arrays are readable for their declared counts.
pub unsafe fn generate_desc_from_abi(desc: *const abi::turbo_generate_desc) -> Result<GenerateDesc> {
    if desc.is_null() {
        return Ok(GenerateDesc::default());
    }
    let d = unsafe { read_sized::<abi::turbo_generate_desc>(desc, "turbo_generate_desc") }?;
    let stop =
        unsafe { texts(d.stop, d.n_stop, "turbo_generate_desc.stop") }?.into_iter().map(str::to_string).collect();
    let stop_tokens = if d.n_stop_tokens == 0 {
        Vec::new()
    } else if d.stop_tokens.is_null() {
        return Err(Error::invalid_argument("stop_tokens is NULL with n_stop_tokens > 0").with_field(17));
    } else {
        unsafe { std::slice::from_raw_parts(d.stop_tokens, d.n_stop_tokens as usize) }.to_vec()
    };
    let logit_bias = if d.n_logit_bias == 0 {
        Vec::new()
    } else if d.logit_bias.is_null() {
        return Err(Error::invalid_argument("logit_bias is NULL with n_logit_bias > 0").with_field(20));
    } else {
        unsafe { std::slice::from_raw_parts(d.logit_bias, d.n_logit_bias as usize) }
            .iter()
            .map(|b| (b.token, b.bias))
            .collect()
    };
    let tools =
        unsafe { texts(d.tools, d.n_tools, "turbo_generate_desc.tools") }?.into_iter().map(str::to_string).collect();
    let has_seed = flag(d.has_seed, "has_seed", GenerateDesc::FIELD_HAS_SEED)?;
    let echo = flag(d.echo, "echo", 22)?;
    Ok(GenerateDesc {
        max_new_tokens: d.max_new_tokens,
        min_new_tokens: d.min_new_tokens,
        n_sequences: d.n_sequences.max(1),
        temperature: d.temperature,
        top_k: d.top_k,
        top_p: d.top_p,
        min_p: d.min_p,
        repeat_penalty: d.repeat_penalty,
        presence_penalty: d.presence_penalty,
        frequency_penalty: d.frequency_penalty,
        seed: if has_seed { Some(d.seed) } else { None },
        stop,
        stop_tokens,
        logit_bias,
        logprobs: d.logprobs,
        structured_kind: StructuredKind::from_abi(d.structured_kind)
            .map_err(|e| e.with_field(GenerateDesc::FIELD_STRUCTURED_KIND))?,
        structured: unsafe { text(&d.structured, "turbo_generate_desc.structured") }?.to_string(),
        echo,
        tools,
        options: unsafe { kvs(d.options, d.n_options, "turbo_generate_desc.options") }?,
    })
}

/// Owned storage backing an ABI `turbo_generate_desc` built from a core `GenerateDesc`.
pub struct GenerateDescViews {
    stop: Vec<String>,
    stop_views: Vec<abi::turbo_text>,
    stop_tokens: Vec<i32>,
    logit_bias: Vec<abi::turbo_logit_bias>,
    structured: String,
    tools: Vec<String>,
    tool_views: Vec<abi::turbo_text>,
    kvs: KvViews,
}

impl GenerateDescViews {
    /// Build the backing storage.
    pub fn new(d: &GenerateDesc) -> Self {
        let stop = d.stop.clone();
        let stop_views = stop.iter().map(|s| text_of(s)).collect();
        let tools = d.tools.clone();
        let tool_views = tools.iter().map(|s| text_of(s)).collect();
        Self {
            stop,
            stop_views,
            stop_tokens: d.stop_tokens.clone(),
            logit_bias: d.logit_bias.iter().map(|&(token, bias)| abi::turbo_logit_bias { token, bias }).collect(),
            structured: d.structured.clone(),
            tools,
            tool_views,
            kvs: KvViews::new(&d.options),
        }
    }

    /// The ABI descriptor borrowing from this storage.
    pub fn desc(&self, d: &GenerateDesc) -> abi::turbo_generate_desc {
        let _ = (&self.stop, &self.tools);
        abi::turbo_generate_desc {
            struct_size: std::mem::size_of::<abi::turbo_generate_desc>() as u32,
            max_new_tokens: d.max_new_tokens,
            min_new_tokens: d.min_new_tokens,
            n_sequences: d.n_sequences,
            temperature: d.temperature,
            top_k: d.top_k,
            top_p: d.top_p,
            min_p: d.min_p,
            repeat_penalty: d.repeat_penalty,
            presence_penalty: d.presence_penalty,
            frequency_penalty: d.frequency_penalty,
            has_seed: d.seed.is_some() as u32,
            seed: d.seed.unwrap_or(0),
            n_stop: self.stop_views.len() as u32,
            n_stop_tokens: self.stop_tokens.len() as u32,
            stop: if self.stop_views.is_empty() { std::ptr::null() } else { self.stop_views.as_ptr() },
            stop_tokens: if self.stop_tokens.is_empty() { std::ptr::null() } else { self.stop_tokens.as_ptr() },
            n_logit_bias: self.logit_bias.len() as u32,
            logprobs: d.logprobs,
            logit_bias: if self.logit_bias.is_empty() { std::ptr::null() } else { self.logit_bias.as_ptr() },
            structured_kind: d.structured_kind.as_abi(),
            echo: d.echo as u32,
            structured: text_of(&self.structured),
            n_tools: self.tool_views.len() as u32,
            n_options: self.kvs.len(),
            tools: if self.tool_views.is_empty() { std::ptr::null() } else { self.tool_views.as_ptr() },
            options: self.kvs.ptr(),
        }
    }
}

// ---------------------------------------------------------------------------
// Stats and spans
// ---------------------------------------------------------------------------

/// Core → ABI.
pub fn session_stats_to_abi(s: &SessionStats, struct_size: u32) -> abi::turbo_session_stats {
    abi::turbo_session_stats {
        struct_size,
        reserved: 0,
        runs: s.runs,
        host_allocs: s.host_allocs,
        h2d_bytes: s.h2d_bytes,
        d2h_bytes: s.d2h_bytes,
        input_bytes: s.input_bytes,
        output_bytes: s.output_bytes,
        provider_allocs: s.provider_allocs.unwrap_or(u64::MAX),
    }
}

/// ABI → core.
pub fn session_stats_from_abi(s: &abi::turbo_session_stats) -> SessionStats {
    SessionStats {
        runs: s.runs,
        host_allocs: s.host_allocs,
        h2d_bytes: s.h2d_bytes,
        d2h_bytes: s.d2h_bytes,
        input_bytes: s.input_bytes,
        output_bytes: s.output_bytes,
        provider_allocs: if s.provider_allocs == u64::MAX { None } else { Some(s.provider_allocs) },
    }
}

/// Core → ABI.
pub fn span_to_abi(s: &Span) -> abi::turbo_span {
    abi::turbo_span {
        byte_start: s.byte_start,
        byte_end: s.byte_end,
        row: s.row,
        label: s.label,
        score: s.score,
        reserved: 0,
    }
}

/// ABI → core.
pub fn span_from_abi(s: &abi::turbo_span) -> Span {
    Span { row: s.row, byte_start: s.byte_start, byte_end: s.byte_end, label: s.label, score: s.score }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_str_cuts_on_char_boundary_and_terminates() {
        let mut dst = [0 as c_char; 5];
        put_str(&mut dst, "héllo");
        // "hé" is 3 bytes; "hél" would be 4; 4 bytes fit plus NUL.
        assert_eq!(get_str(&dst), "hél");
        let mut one = [0 as c_char; 1];
        put_str(&mut one, "x");
        assert_eq!(get_str(&one), "");
    }

    #[test]
    fn size_check_accepts_field_boundaries_only() {
        use std::mem::{offset_of, size_of};
        // Current size and every earlier field end are known layouts.
        assert!(check_size::<abi::turbo_embed_options>("o", size_of::<abi::turbo_embed_options>() as u32).is_ok());
        assert!(check_size::<abi::turbo_embed_options>("o", offset_of!(abi::turbo_embed_options, max_tokens) as u32)
            .is_ok());
        // Too small, too large, or ending inside a field: rejected.
        assert!(check_size::<abi::turbo_embed_options>("o", 4).is_err());
        assert!(check_size::<abi::turbo_embed_options>("o", 3).is_err());
        assert!(check_size::<abi::turbo_embed_options>("o", 1000).is_err());
        // Half a pointer: `stop` starts at 64; 68 lands inside it.
        let stop = offset_of!(abi::turbo_generate_desc, stop) as u32;
        assert!(check_size::<abi::turbo_generate_desc>("g", stop).is_ok());
        assert!(check_size::<abi::turbo_generate_desc>("g", stop + 4).is_err());
    }

    #[test]
    fn options_round_trip() {
        let o = EmbedOptions { truncate: Truncate::Left, max_tokens: 7, ..Default::default() };
        let a = embed_options_to_abi(&o);
        let back = unsafe { embed_options_from_abi(&a) }.unwrap();
        assert_eq!(o, back);
        let mut bad = a;
        bad.pooling = 42;
        let e = unsafe { embed_options_from_abi(&bad) }.unwrap_err();
        assert_eq!(e.field(), EmbedOptions::FIELD_POOLING);
    }
}
