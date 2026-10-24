//! Model loading, metadata, and feature-gated inspection entrypoints.

use super::*;

/// docs/bundle.md's loader rules, in order, for the context's device.
pub(super) fn load_model(ctx: &turbo_context, path: &str) -> Result<Model> {
    let c = &ctx.inner;
    let b = c.backend;
    let load = backend::offered!(b, model_load)?;
    let release = backend::offered!(b, model_release)?;
    let bundle = bundle::Bundle::open(Path::new(path))?; // rules 1 to 4
    if bundle.manifest.rerank.is_some() && b.name() != "cpu" {
        return Err(Error::new(status::UNSUPPORTED_TASK, "rerank requires the CPU backend"));
    }
    if bundle.manifest.qa.is_some() && backend::offered!(b, embed_write_ranges).is_err() {
        return Err(Error::new(
            status::UNSUPPORTED_TASK,
            format!("extractive QA needs range writes, which the {} backend does not offer", b.name()),
        ));
    }
    let tokenizer = Tokenizer::load(&bundle)?; // rule 5
    let m = &bundle.manifest;
    let arch = cstr(&c.runtime.devices[c.device as usize].info.arch);
    let index = model::choose(m, b.name(), b.formats(), &arch)?; // rule 6
    let weights = model::Weights::load(&bundle, index)?; // rules 7 and 8
    let art = &m.artifacts[index];

    let e = m.encoder();
    let mut info: turbo_model_info = unsafe { std::mem::zeroed() };
    info.struct_size = size_of::<turbo_model_info>() as u32;
    info.task = if m.rerank.is_some() {
        TURBO_TASK_RERANK
    } else if m.qa.is_some() {
        TURBO_TASK_EXTRACTIVE_QA
    } else {
        TURBO_TASK_EMBED
    };
    info.dim = e.dim;
    info.pooling = match e.pooling {
        manifest::Pooling::Mean => TURBO_POOLING_MEAN,
        manifest::Pooling::Cls => TURBO_POOLING_CLS,
        manifest::Pooling::Last => TURBO_POOLING_LAST,
    };
    info.normalize = match e.normalize {
        manifest::Normalize::None => TURBO_NORMALIZE_NONE,
        manifest::Normalize::L2 => TURBO_NORMALIZE_L2,
    };
    // A fixed-shape artifact reports the smaller of its sequence and the
    // model's. Its fixed_batch is the frame the backend runs, and a backend
    // runs as many frames as a batch needs, so the model's max_batch stands.
    info.max_seq = if art.fixed_seq == 0 { e.max_seq } else { e.max_seq.min(art.fixed_seq) };
    info.max_batch = e.max_batch;
    // A compiled artifact's compute_dtype, else the dtype raw weights are
    // stored in: raw weights fix none (the manifest refuses one).
    info.dtype = weights.info_dtype();
    // The manifest's strings were checked against these buffers when it
    // was parsed (rule 2), and a hash is 64 hex digits: nothing is cut.
    write_str(&mut info.model_id, &m.model.id);
    write_str(&mut info.revision, &m.model.revision);
    write_str(&mut info.manifest_sha256, &bundle.manifest_sha256);
    write_str(&mut info.artifact_sha256, &model::artifact_sha256(m, art));
    write_str(&mut info.tokenizer_sha256, &tokenizer.sha256);
    write_str(&mut info.prefix_query, &e.prefix_query);
    write_str(&mut info.prefix_document, &e.prefix_document);
    // The manifest refuses more than TURBO_OUTPUT_DIMS_MAX (rule 2).
    let mut dims = e.output_dims.clone();
    dims.sort_unstable();
    info.output_dims_count = dims.len() as u32;
    info.output_dims[..dims.len()].copy_from_slice(&dims);

    if let Some(r) = &m.rerank {
        info.rerank_query_tokens = r.query_tokens;
        info.rerank_pair_template = match r.pair_template {
            manifest::PairTemplate::Bert => TURBO_PAIR_BERT,
            manifest::PairTemplate::Roberta => TURBO_PAIR_ROBERTA,
        };
    }
    let rerank_head = m.rerank.as_ref().map(|_| head::Head::load_artifact(&bundle, index)).transpose()?;
    // A backend that applies token heads is given the span head, in F32,
    // and pools QA ranges into logits (turbo_backend.h, token_head).
    let span_head =
        (m.qa.is_some() && b.token_head()).then(|| head::Head::load_artifact(&bundle, index)).transpose()?;
    let tensors = weights.tensors();
    let mut desc = weights.desc(&tensors);
    let head_tensors = span_head.as_ref().map(|h| head_tensors(&h.classifier));
    if let (Some(h), Some(t)) = (&span_head, &head_tensors) {
        desc.head = t.as_ptr();
        desc.head_out = h.classifier.out_dim as u32;
    }
    let mut raw = std::ptr::null_mut();
    backend::check(b, "model_load", |err| unsafe { load(c.raw, &desc, &mut raw, err) })?; // rule 9
    let output_dims = e.output_dims.clone();
    Ok(Model { context: c.clone(), raw, release, weights, tokenizer, info, output_dims, rerank_head, span_head })
}

/// A linear layer as turbo_backend_model.head takes it: the weight, then
/// the bias, both F32, pointing into `l`.
fn head_tensors(l: &head::Linear) -> [backend::turbo_backend_tensor; 2] {
    let f32_tensor =
        |name: &'static std::ffi::CStr, data: &[f32], shape: [u64; 2], ndim: u32| backend::turbo_backend_tensor {
            name: name.as_ptr(),
            data: data.as_ptr().cast(),
            shape,
            ndim,
            dtype: TURBO_DTYPE_F32,
            bytes: size_of_val(data) as u64,
        };
    let (out, hidden) = (l.out_dim as u64, l.in_dim as u64);
    [
        f32_tensor(c"classifier_weight", &l.weight, [out, hidden], 2),
        f32_tensor(c"classifier_bias", &l.bias, [out, 0], 1),
    ]
}

/// # Safety
/// Pointers are NULL or valid for the call, as turbo.h says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_model_load(
    ctx: *mut turbo_context,
    bundle_path: turbo_text,
    out: *mut *mut turbo_model,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let ctx = context(ctx)?;
            let out = out_ptr(out, "out")?;
            let path = text(bundle_path, "bundle_path")?;
            let inner = load_model(ctx, path)?;
            *out = Box::into_raw(Box::new(turbo_model { magic: MODEL_MAGIC, inner: Arc::new(inner) }));
            Ok(())
        })
    }
}

/// # Safety
/// `m` is NULL or a handle from turbo_model_load, released once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_model_release(m: *mut turbo_model) {
    if unsafe { model_handle(m) }.is_ok() {
        let mut b = unsafe { Box::from_raw(m) };
        b.magic = 0;
    }
}

/// # Safety
/// Pointers are NULL or valid for the call, as turbo.h says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_model_get_info(
    m: *mut turbo_model,
    out: *mut turbo_model_info,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let m = model_handle(m)?;
            if out.is_null() {
                return Err(Error::new(INVALID_ARGUMENT, "out is NULL"));
            }
            // A caller built against the struct before output_dims_count
            // has only that much memory: it is read and written through
            // the raw pointer, never as the whole struct.
            let size = (out as *const u32).read();
            let want = size_of::<turbo_model_info>();
            if size as usize != TURBO_MODEL_INFO_SIZE_V1 && size as usize != TURBO_MODEL_INFO_SIZE_V2 {
                sized(size, want, "turbo_model_info")?;
            }
            let mut info = m.inner.info;
            if info.task == TURBO_TASK_RERANK {
                info.dim = 0;
                info.pooling = 0;
                info.normalize = 0;
            }
            std::ptr::copy_nonoverlapping(
                (&info as *const turbo_model_info).cast::<u8>().add(4),
                out.cast::<u8>().add(4),
                size as usize - 4,
            );
            Ok(())
        })
    }
}

/// Where a model's weights are, for the tests that check the CPU backend
/// holds no copy of its own. Not part of the C interface, and built only
/// with the `internals` feature.
#[cfg(feature = "internals")]
pub struct ModelWeights<'a> {
    /// The core's verified bytes of each weights file.
    pub files: Vec<&'a [u8]>,
    /// On the CPU backend, the address of each tensor the backend keeps.
    pub held: Option<Vec<*const c_void>>,
}

/// # Safety
/// `m` is a live handle from turbo_model_load, and outlives what is returned.
#[cfg(feature = "internals")]
pub unsafe fn model_weights<'a>(m: *mut turbo_model) -> Option<ModelWeights<'a>> {
    let m = &unsafe { model_handle(m) }.ok()?.inner;
    #[cfg(feature = "cpu")]
    let held = std::ptr::eq(m.context.backend, &cpu::BACKEND).then(|| unsafe { cpu::tensor_data(m.raw) });
    #[cfg(not(feature = "cpu"))]
    let held = None;
    Some(ModelWeights { files: m.weights.files(), held })
}

/// Where the CPU backend's F32 copy of an F16 or BF16 model's weights is,
/// once a session has made it; for a GPU backend, where each copy of the
/// weights in another dtype than the stored one is. Built only with the
/// `internals` feature.
///
/// # Safety
/// As for model_weights.
#[cfg(feature = "internals")]
pub unsafe fn model_converted_weights(m: *mut turbo_model) -> Option<Vec<*const c_void>> {
    let m = &unsafe { model_handle(m) }.ok()?.inner;
    #[cfg(feature = "cpu")]
    if std::ptr::eq(m.context.backend, &cpu::BACKEND) {
        return unsafe { cpu::converted_data(m.raw) };
    }
    // The CUDA backend widens every tensor into one device allocation, and
    // narrows the GEMM weights to F16 into another: each address stands
    // for its copy, the F32 one first.
    #[cfg(feature = "cuda")]
    if std::ptr::eq(m.context.backend, cuda::backend()) {
        let copies: Vec<_> = unsafe { [cuda::widened(m.raw), cuda::narrowed(m.raw)] }.into_iter().flatten().collect();
        return (!copies.is_empty()).then_some(copies);
    }
    // So does the levelzero backend.
    #[cfg(feature = "levelzero")]
    if std::ptr::eq(m.context.backend, &levelzero::BACKEND) {
        return unsafe { levelzero::widened(m.raw) }.map(|p| vec![p]);
    }
    // So does the Metal backend, into one Metal buffer.
    #[cfg(feature = "metal")]
    if std::ptr::eq(m.context.backend, metal::backend()) {
        return unsafe { metal::widened(m.raw) }.map(|p| vec![p]);
    }
    None
}

/// Whether the Metal backend reads the model's weights where the core
/// holds them. Built only with the `internals` feature.
///
/// # Safety
/// As for model_weights.
#[cfg(all(feature = "internals", feature = "metal"))]
pub unsafe fn model_metal_in_place(m: *mut turbo_model) -> Option<bool> {
    let m = &unsafe { model_handle(m) }.ok()?.inner;
    std::ptr::eq(m.context.backend, metal::backend()).then(|| unsafe { metal::in_place(m.raw) })
}
