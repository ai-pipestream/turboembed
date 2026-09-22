//! Export a Rust `Provider` as a C provider vtable.
//!
//! Rust providers (mock, static, cuda, ggml) use [`export_provider!`] to
//! define the `turbo_provider_get` symbol of their `cdylib`. Every shim here
//! catches panics and converts errors into the caller-owned `turbo_error`, so
//! nothing unwinds across the plugin boundary.
//!
//! Handle representation:
//! - context: `Box<Arc<dyn ProviderContext>>`
//! - buffer: `Box<Arc<dyn ProviderBuffer>>`
//! - model: `Box<ExportModel>` (holds `Arc<dyn ProviderModel>`)
//! - session: `Box<ExportSession>` (holds the session plus the last result's
//!   output descriptors, which stay valid until the next run)
//! - generation: `Box<ExportGeneration>`

use std::ffi::{c_void, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::Arc;

use turbo_abi as abi;

use crate::abi_convert::{
    buffer_desc_from_abi, buffer_desc_to_abi, capability_to_abi, check_size, classify_options_from_abi,
    device_info_to_abi, embed_options_from_abi, generate_desc_from_abi, kvs, model_info_to_abi, native_handle_from_abi,
    native_handle_to_abi, put_str, read_sized, rerank_options_from_abi, session_stats_to_abi, span_to_abi,
    tensor_info_to_abi, text, text_of, texts, write_sized,
};
use crate::buffer::ProviderBuffer;
use crate::bundle::Bundle;
use crate::error::{Error, Result};
use crate::provider::{
    Chunk, ContextDesc, GenerateDesc, Message, ModelDesc, Provider, ProviderContext, ProviderGeneration, ProviderModel,
    ProviderSession, RunOptions, SessionDesc, TokenBatch,
};
use crate::types::{HandleKind, Modality, Task};

/// Provider-global state behind `turbo_provider_vtbl.state`.
pub struct ExportState {
    provider: Arc<dyn Provider>,
    id: &'static CStr,
    version: &'static CStr,
}

struct ExportModel {
    inner: Arc<dyn ProviderModel>,
}

struct ExportSession {
    inner: Box<dyn ProviderSession>,
    // Kept alive and stable until the next run. Output buffer handles use
    // the same representation as allocated buffers (`*const Arc<dyn
    // ProviderBuffer>`), boxed so their address is stable while the result
    // is outstanding; the core never releases these borrowed handles.
    outputs: Vec<abi::turbo_provider_output>,
    #[allow(clippy::vec_box)] // the Box gives each handle a stable address
    keep: Vec<Box<Arc<dyn ProviderBuffer>>>,
    names: Vec<Arc<str>>,
    spans: Vec<abi::turbo_span>,
}

struct ExportGeneration {
    inner: Box<dyn ProviderGeneration>,
    chunk: Chunk,
}

fn fail(err: *mut abi::turbo_error, e: &Error) -> i32 {
    if !err.is_null() {
        // SAFETY: the core passes a valid turbo_error or NULL.
        let out = unsafe { &mut *err };
        let size = out.struct_size as usize;
        if size >= 8 {
            out.code = e.code();
        }
        if size >= 12 {
            out.field = e.field();
        }
        if size > 12 {
            let cap = (size - 12).min(abi::TURBO_ERROR_MESSAGE_LEN);
            put_str(&mut out.message[..cap], e.message());
        }
    }
    e.code()
}

fn boundary(err: *mut abi::turbo_error, f: impl FnOnce() -> Result<()>) -> i32 {
    if !err.is_null() {
        // SAFETY: as in `fail`.
        let out = unsafe { &mut *err };
        if out.struct_size >= 8 {
            out.code = 0;
        }
        if out.struct_size >= 12 {
            out.field = 0;
        }
        if out.struct_size > 12 {
            out.message[0] = 0;
        }
    }
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => abi::TURBO_OK,
        Ok(Err(e)) => fail(err, &e),
        Err(payload) => {
            let msg = payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "non-string panic payload".to_string());
            fail(err, &Error::panic(format!("provider panicked: {msg}")))
        }
    }
}

unsafe fn state<'a>(p: *mut c_void) -> Result<&'a ExportState> {
    if p.is_null() {
        return Err(Error::invalid_handle("provider state"));
    }
    // SAFETY: the vtable's state pointer was set by `make_vtbl`.
    Ok(unsafe { &*(p as *const ExportState) })
}

unsafe fn ctx<'a>(p: *mut c_void) -> Result<&'a Arc<dyn ProviderContext>> {
    if p.is_null() {
        return Err(Error::invalid_handle("context"));
    }
    Ok(unsafe { &*(p as *const Arc<dyn ProviderContext>) })
}

unsafe fn buf<'a>(p: *mut c_void) -> Result<&'a Arc<dyn ProviderBuffer>> {
    if p.is_null() {
        return Err(Error::invalid_handle("buffer"));
    }
    Ok(unsafe { &*(p as *const Arc<dyn ProviderBuffer>) })
}

unsafe fn model<'a>(p: *mut c_void) -> Result<&'a ExportModel> {
    if p.is_null() {
        return Err(Error::invalid_handle("model"));
    }
    Ok(unsafe { &*(p as *const ExportModel) })
}

unsafe fn session<'a>(p: *mut c_void) -> Result<&'a mut ExportSession> {
    if p.is_null() {
        return Err(Error::invalid_handle("session"));
    }
    // SAFETY: the core guarantees single-owner access to sessions.
    Ok(unsafe { &mut *(p as *mut ExportSession) })
}

unsafe fn generation<'a>(p: *mut c_void) -> Result<&'a mut ExportGeneration> {
    if p.is_null() {
        return Err(Error::invalid_handle("generation"));
    }
    Ok(unsafe { &mut *(p as *mut ExportGeneration) })
}

fn out_ptr<'a, T>(out: *mut T, what: &str) -> Result<&'a mut T> {
    if out.is_null() {
        return Err(Error::invalid_argument(format!("{what}: out pointer is NULL")));
    }
    // SAFETY: non-null out pointer supplied by the core.
    Ok(unsafe { &mut *out })
}

fn provider_buffer(b: &Arc<dyn ProviderBuffer>, handle: *mut c_void) -> abi::turbo_provider_buffer {
    abi::turbo_provider_buffer {
        struct_size: std::mem::size_of::<abi::turbo_provider_buffer>() as u32,
        reserved: 0,
        handle,
        host_ptr: b.host_ptr().map(|p| p.as_ptr().cast()).unwrap_or(std::ptr::null_mut()),
        desc: buffer_desc_to_abi(b.desc(), std::mem::size_of::<abi::turbo_buffer_desc>() as u32),
    }
}

unsafe extern "C" fn x_device_count(st: *mut c_void, out: *mut u32, err: *mut abi::turbo_error) -> i32 {
    boundary(err, || {
        let st = unsafe { state(st) }?;
        let o = out_ptr(out, "device_count")?;
        *o = st.provider.devices()?.len() as u32;
        Ok(())
    })
}

unsafe extern "C" fn x_device_info(
    st: *mut c_void,
    ordinal: u32,
    out: *mut abi::turbo_device_info,
    err: *mut abi::turbo_error,
) -> i32 {
    boundary(err, || {
        let st = unsafe { state(st) }?;
        let o = out_ptr(out, "device_info")?;
        check_size::<abi::turbo_device_info>("turbo_device_info", o.struct_size)?;
        let devices = st.provider.devices()?;
        let d = devices
            .get(ordinal as usize)
            .ok_or_else(|| Error::device_not_found(format!("ordinal {ordinal} is out of range")))?;
        let full = device_info_to_abi(d, o.struct_size);
        unsafe { write_sized(&full, out, o.struct_size) };
        Ok(())
    })
}

unsafe extern "C" fn x_capability(
    st: *mut c_void,
    ordinal: u32,
    task: u32,
    modality: u32,
    out: *mut abi::turbo_capability,
    err: *mut abi::turbo_error,
) -> i32 {
    boundary(err, || {
        let st = unsafe { state(st) }?;
        let o = out_ptr(out, "capability")?;
        check_size::<abi::turbo_capability>("turbo_capability", o.struct_size)?;
        let c = st.provider.capability(ordinal, Task::from_abi(task)?, Modality::from_abi(modality)?);
        let full = capability_to_abi(&c, o.struct_size);
        unsafe { write_sized(&full, out, o.struct_size) };
        Ok(())
    })
}

unsafe extern "C" fn x_can_run(
    st: *mut c_void,
    ordinal: u32,
    bundle_dir: abi::turbo_text,
    task: u32,
    modality: u32,
    err: *mut abi::turbo_error,
) -> i32 {
    boundary(err, || {
        let st = unsafe { state(st) }?;
        let dir = unsafe { text(&bundle_dir, "bundle_dir") }?;
        let bundle = Bundle::open(Path::new(dir))?;
        st.provider.can_run(ordinal, &bundle, Task::from_abi(task)?, Modality::from_abi(modality)?)
    })
}

unsafe extern "C" fn x_context_create(
    st: *mut c_void,
    ordinal: u32,
    desc: *const abi::turbo_context_desc,
    out: *mut *mut c_void,
    err: *mut abi::turbo_error,
) -> i32 {
    boundary(err, || {
        let st = unsafe { state(st) }?;
        let o = out_ptr(out, "context_create")?;
        *o = std::ptr::null_mut();
        let mut cd = ContextDesc::default();
        if !desc.is_null() {
            let d = unsafe { read_sized::<abi::turbo_context_desc>(desc, "turbo_context_desc") }?;
            cd.options = unsafe { kvs(d.options, d.n_options, "turbo_context_desc.options") }?;
        }
        let c = st.provider.create_context(ordinal, &cd)?;
        *o = Box::into_raw(Box::new(c)) as *mut c_void;
        Ok(())
    })
}

unsafe extern "C" fn x_context_release(p: *mut c_void) {
    if !p.is_null() {
        // A Drop panic must not unwind across the plugin boundary.
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // SAFETY: created by x_context_create; released once.
            drop(unsafe { Box::from_raw(p as *mut Arc<dyn ProviderContext>) });
        }))
        .is_err()
        {
            eprintln!("turbo plugin: panic inside x_context_release");
        }
    }
}

unsafe extern "C" fn x_buffer_alloc(
    c: *mut c_void,
    desc: *const abi::turbo_buffer_desc,
    out: *mut abi::turbo_provider_buffer,
    err: *mut abi::turbo_error,
) -> i32 {
    boundary(err, || {
        let c = unsafe { ctx(c) }?;
        let o = out_ptr(out, "buffer_alloc")?;
        check_size::<abi::turbo_provider_buffer>("turbo_provider_buffer", o.struct_size)?;
        let (bd, _) = unsafe { buffer_desc_from_abi(desc) }?;
        let b = c.alloc(&bd)?;
        let handle = Box::into_raw(Box::new(b.clone())) as *mut c_void;
        let full = provider_buffer(&b, handle);
        unsafe { write_sized(&full, out, o.struct_size) };
        Ok(())
    })
}

unsafe extern "C" fn x_buffer_import(
    c: *mut c_void,
    desc: *const abi::turbo_buffer_desc,
    handle: *const abi::turbo_native_handle,
    out: *mut abi::turbo_provider_buffer,
    err: *mut abi::turbo_error,
) -> i32 {
    boundary(err, || {
        let c = unsafe { ctx(c) }?;
        let o = out_ptr(out, "buffer_import")?;
        check_size::<abi::turbo_provider_buffer>("turbo_provider_buffer", o.struct_size)?;
        let (bd, _) = unsafe { buffer_desc_from_abi(desc) }?;
        if handle.is_null() {
            return Err(Error::invalid_argument("native handle is NULL"));
        }
        let h = native_handle_from_abi(unsafe { &*handle })?;
        let b = c.import(&bd, &h)?;
        let raw = Box::into_raw(Box::new(b.clone())) as *mut c_void;
        let full = provider_buffer(&b, raw);
        unsafe { write_sized(&full, out, o.struct_size) };
        Ok(())
    })
}

unsafe extern "C" fn x_buffer_read(b: *mut c_void, dst: *mut c_void, bytes: u64, err: *mut abi::turbo_error) -> i32 {
    boundary(err, || {
        let b = unsafe { buf(b) }?;
        if dst.is_null() {
            return Err(Error::invalid_argument("dst is NULL"));
        }
        let n = usize::try_from(bytes).map_err(|_| Error::invalid_argument("bytes exceeds usize"))?;
        // SAFETY: the core passes a writable region of `bytes` bytes.
        let slice = unsafe { std::slice::from_raw_parts_mut(dst.cast::<u8>(), n) };
        b.read_to_host(slice)
    })
}

unsafe extern "C" fn x_buffer_export(
    b: *mut c_void,
    kind: u32,
    out: *mut abi::turbo_native_handle,
    err: *mut abi::turbo_error,
) -> i32 {
    boundary(err, || {
        let b = unsafe { buf(b) }?;
        let o = out_ptr(out, "buffer_export")?;
        let h = b.export(HandleKind::from_abi(kind)?)?;
        *o = native_handle_to_abi(&h);
        Ok(())
    })
}

unsafe extern "C" fn x_buffer_release(b: *mut c_void) {
    if !b.is_null() {
        // A Drop panic must not unwind across the plugin boundary.
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            drop(unsafe { Box::from_raw(b as *mut Arc<dyn ProviderBuffer>) });
        }))
        .is_err()
        {
            eprintln!("turbo plugin: panic inside x_buffer_release");
        }
    }
}

unsafe extern "C" fn x_model_load(
    c: *mut c_void,
    bundle_dir: abi::turbo_text,
    desc: *const abi::turbo_model_desc,
    out: *mut *mut c_void,
    err: *mut abi::turbo_error,
) -> i32 {
    boundary(err, || {
        let c = unsafe { ctx(c) }?;
        let o = out_ptr(out, "model_load")?;
        *o = std::ptr::null_mut();
        let dir = unsafe { text(&bundle_dir, "bundle_dir") }?;
        let bundle = Arc::new(Bundle::open(Path::new(dir))?);
        let mut md = ModelDesc::default();
        if !desc.is_null() {
            let d = unsafe { read_sized::<abi::turbo_model_desc>(desc, "turbo_model_desc") }?;
            md.options = unsafe { kvs(d.options, d.n_options, "turbo_model_desc.options") }?;
        }
        let m = c.load_model(bundle, &md)?;
        *o = Box::into_raw(Box::new(ExportModel { inner: m })) as *mut c_void;
        Ok(())
    })
}

unsafe extern "C" fn x_model_info(m: *mut c_void, out: *mut abi::turbo_model_info, err: *mut abi::turbo_error) -> i32 {
    boundary(err, || {
        let m = unsafe { model(m) }?;
        let o = out_ptr(out, "model_info")?;
        check_size::<abi::turbo_model_info>("turbo_model_info", o.struct_size)?;
        let full = model_info_to_abi(m.inner.info(), o.struct_size);
        unsafe { write_sized(&full, out, o.struct_size) };
        Ok(())
    })
}

unsafe extern "C" fn x_model_label(
    m: *mut c_void,
    index: u32,
    out: *mut abi::turbo_text,
    err: *mut abi::turbo_error,
) -> i32 {
    boundary(err, || {
        let m = unsafe { model(m) }?;
        let o = out_ptr(out, "model_label")?;
        let labels = &m.inner.info().labels;
        let l = labels.get(index as usize).ok_or_else(|| {
            Error::invalid_argument(format!("label index {index} out of range ({} labels)", labels.len()))
        })?;
        *o = text_of(l);
        Ok(())
    })
}

unsafe extern "C" fn x_model_io_info(
    m: *mut c_void,
    direction: u32,
    index: u32,
    out: *mut abi::turbo_tensor_info,
    err: *mut abi::turbo_error,
) -> i32 {
    boundary(err, || {
        let m = unsafe { model(m) }?;
        let o = out_ptr(out, "model_io_info")?;
        check_size::<abi::turbo_tensor_info>("turbo_tensor_info", o.struct_size)?;
        let list = match direction {
            abi::TURBO_IO_INPUT => &m.inner.info().inputs,
            abi::TURBO_IO_OUTPUT => &m.inner.info().outputs,
            other => return Err(Error::invalid_enum("io direction", other)),
        };
        let t = list
            .get(index as usize)
            .ok_or_else(|| Error::invalid_argument(format!("tensor index {index} out of range ({})", list.len())))?;
        let full = tensor_info_to_abi(t, o.struct_size)?;
        unsafe { write_sized(&full, out, o.struct_size) };
        Ok(())
    })
}

unsafe extern "C" fn x_model_release(m: *mut c_void) {
    if !m.is_null() {
        // A Drop panic must not unwind across the plugin boundary.
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            drop(unsafe { Box::from_raw(m as *mut ExportModel) });
        }))
        .is_err()
        {
            eprintln!("turbo plugin: panic inside x_model_release");
        }
    }
}

unsafe extern "C" fn x_session_create(
    m: *mut c_void,
    desc: *const abi::turbo_session_desc,
    out: *mut *mut c_void,
    err: *mut abi::turbo_error,
) -> i32 {
    boundary(err, || {
        let m = unsafe { model(m) }?;
        let o = out_ptr(out, "session_create")?;
        *o = std::ptr::null_mut();
        if desc.is_null() {
            return Err(Error::invalid_argument("turbo_session_desc is NULL"));
        }
        let d = unsafe { read_sized::<abi::turbo_session_desc>(desc, "turbo_session_desc") }?;
        let sd = SessionDesc {
            max_batch: d.max_batch,
            max_seq: d.max_seq,
            options: unsafe { kvs(d.options, d.n_options, "turbo_session_desc.options") }?,
        };
        let s = m.inner.create_session(&sd)?;
        *o = Box::into_raw(Box::new(ExportSession {
            inner: s,
            outputs: Vec::new(),
            keep: Vec::new(),
            names: Vec::new(),
            spans: Vec::new(),
        })) as *mut c_void;
        Ok(())
    })
}

unsafe extern "C" fn x_session_write_text(
    s: *mut c_void,
    t: *const abi::turbo_text,
    count: u32,
    opts: *const abi::turbo_embed_options,
    err: *mut abi::turbo_error,
) -> i32 {
    boundary(err, || {
        let s = unsafe { session(s) }?;
        let list = unsafe { texts(t, count, "texts") }?;
        let o = unsafe { embed_options_from_abi(opts) }?;
        s.inner.write_text(&list, &o)
    })
}

unsafe extern "C" fn x_session_write_tokens(
    s: *mut c_void,
    batch: *const abi::turbo_token_batch,
    err: *mut abi::turbo_error,
) -> i32 {
    boundary(err, || {
        let s = unsafe { session(s) }?;
        if batch.is_null() {
            return Err(Error::invalid_argument("batch is NULL"));
        }
        let b = unsafe { read_sized::<abi::turbo_token_batch>(batch, "turbo_token_batch") }?;
        let row_stride = if b.row_stride == 0 { b.seq } else { b.row_stride };
        let need = TokenBatch::required_len(b.batch, b.seq, row_stride)?;
        if b.ids.is_null() || b.mask.is_null() {
            return Err(Error::invalid_argument("ids or mask is NULL"));
        }
        // SAFETY: the core validated the batch and passes readable arrays.
        let ids = unsafe { std::slice::from_raw_parts(b.ids, need) };
        let mask = unsafe { std::slice::from_raw_parts(b.mask, need) };
        let types = if b.types.is_null() { None } else { Some(unsafe { std::slice::from_raw_parts(b.types, need) }) };
        s.inner.write_tokens(&TokenBatch { batch: b.batch, seq: b.seq, row_stride, ids, mask, types })
    })
}

unsafe extern "C" fn x_session_write_pairs(
    s: *mut c_void,
    query: *const abi::turbo_text,
    docs: *const abi::turbo_text,
    count: u32,
    opts: *const abi::turbo_rerank_options,
    err: *mut abi::turbo_error,
) -> i32 {
    boundary(err, || {
        let s = unsafe { session(s) }?;
        if query.is_null() {
            return Err(Error::invalid_argument("query is NULL"));
        }
        let q = unsafe { text(&*query, "query") }?;
        let d = unsafe { texts(docs, count, "docs") }?;
        let o = unsafe { rerank_options_from_abi(opts) }?;
        s.inner.write_pairs(q, &d, &o)
    })
}

unsafe extern "C" fn x_session_write_text_classify(
    s: *mut c_void,
    t: *const abi::turbo_text,
    count: u32,
    opts: *const abi::turbo_classify_options,
    err: *mut abi::turbo_error,
) -> i32 {
    boundary(err, || {
        let s = unsafe { session(s) }?;
        let list = unsafe { texts(t, count, "texts") }?;
        let o = unsafe { classify_options_from_abi(opts) }?;
        s.inner.write_text_classify(&list, &o)
    })
}

unsafe extern "C" fn x_session_bind(
    s: *mut c_void,
    name: abi::turbo_text,
    b: *mut c_void,
    err: *mut abi::turbo_error,
) -> i32 {
    boundary(err, || {
        let s = unsafe { session(s) }?;
        let n = unsafe { text(&name, "name") }?;
        let b = unsafe { buf(b) }?;
        s.inner.bind(n, b.clone())
    })
}

unsafe extern "C" fn x_session_run(
    s: *mut c_void,
    opts: *const abi::turbo_run_options,
    out: *mut abi::turbo_provider_result,
    err: *mut abi::turbo_error,
) -> i32 {
    boundary(err, || {
        let s = unsafe { session(s) }?;
        let o = out_ptr(out, "session_run")?;
        check_size::<abi::turbo_provider_result>("turbo_provider_result", o.struct_size)?;
        let mut ro = RunOptions::default();
        if !opts.is_null() {
            let r = unsafe { read_sized::<abi::turbo_run_options>(opts, "turbo_run_options") }?;
            ro.params = unsafe { kvs(r.params, r.n_params, "turbo_run_options.params") }?;
        }
        let result = s.inner.run(&ro)?;
        s.outputs.clear();
        s.keep.clear();
        s.names.clear();
        s.spans.clear();
        for (i, output) in result.outputs.iter().enumerate() {
            if output.shape.len() > abi::TURBO_MAX_RANK {
                return Err(Error::internal(format!("output {i} has rank {}", output.shape.len())));
            }
            // Borrowed handle: the exporter keeps the boxed Arc alive in
            // `keep`; `buf()` reads it back as `*const Arc<dyn ProviderBuffer>`.
            let boxed: Box<Arc<dyn ProviderBuffer>> = Box::new(output.buffer.clone());
            let handle = (&*boxed as *const Arc<dyn ProviderBuffer>).cast_mut().cast::<c_void>();
            s.keep.push(boxed);
            s.names.push(output.name.clone());
            let mut shape = [0u64; abi::TURBO_MAX_RANK];
            shape[..output.shape.len()].copy_from_slice(&output.shape);
            s.outputs.push(abi::turbo_provider_output {
                struct_size: std::mem::size_of::<abi::turbo_provider_output>() as u32,
                ndim: output.shape.len() as u32,
                name: text_of(s.names.last().unwrap()),
                buffer: provider_buffer(&output.buffer, handle),
                shape,
            });
        }
        s.spans.extend(result.spans.iter().map(span_to_abi));
        let full = abi::turbo_provider_result {
            struct_size: o.struct_size,
            n_outputs: s.outputs.len() as u32,
            outputs: s.outputs.as_ptr(),
            n_spans: s.spans.len() as u32,
            reserved: 0,
            spans: if s.spans.is_empty() { std::ptr::null() } else { s.spans.as_ptr() },
        };
        unsafe { write_sized(&full, out, o.struct_size) };
        Ok(())
    })
}

unsafe extern "C" fn x_session_stats(
    s: *mut c_void,
    out: *mut abi::turbo_session_stats,
    err: *mut abi::turbo_error,
) -> i32 {
    boundary(err, || {
        let s = unsafe { session(s) }?;
        let o = out_ptr(out, "session_stats")?;
        check_size::<abi::turbo_session_stats>("turbo_session_stats", o.struct_size)?;
        let full = session_stats_to_abi(&s.inner.stats()?, o.struct_size);
        unsafe { write_sized(&full, out, o.struct_size) };
        Ok(())
    })
}

unsafe extern "C" fn x_session_release(s: *mut c_void) {
    if !s.is_null() {
        // A Drop panic must not unwind across the plugin boundary.
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            drop(unsafe { Box::from_raw(s as *mut ExportSession) });
        }))
        .is_err()
        {
            eprintln!("turbo plugin: panic inside x_session_release");
        }
    }
}

unsafe extern "C" fn x_generation_create(
    m: *mut c_void,
    desc: *const abi::turbo_generate_desc,
    out: *mut *mut c_void,
    err: *mut abi::turbo_error,
) -> i32 {
    boundary(err, || {
        let m = unsafe { model(m) }?;
        let o = out_ptr(out, "generation_create")?;
        *o = std::ptr::null_mut();
        let gd: GenerateDesc = unsafe { generate_desc_from_abi(desc) }?;
        let g = m.inner.create_generation(&gd)?;
        *o = Box::into_raw(Box::new(ExportGeneration { inner: g, chunk: Chunk::default() })) as *mut c_void;
        Ok(())
    })
}

unsafe extern "C" fn x_generation_prompt(
    g: *mut c_void,
    messages: *const abi::turbo_message,
    count: u32,
    err: *mut abi::turbo_error,
) -> i32 {
    boundary(err, || {
        let g = unsafe { generation(g) }?;
        if count == 0 || messages.is_null() {
            return Err(Error::invalid_argument("messages is NULL or empty"));
        }
        let raw = unsafe { std::slice::from_raw_parts(messages, count as usize) };
        let mut list = Vec::with_capacity(raw.len());
        for m in raw {
            list.push(Message {
                role: unsafe { text(&m.role, "role") }?,
                content: unsafe { text(&m.content, "content") }?,
            });
        }
        g.inner.prompt(&list)
    })
}

unsafe extern "C" fn x_generation_prompt_tokens(
    g: *mut c_void,
    ids: *const i32,
    count: u32,
    err: *mut abi::turbo_error,
) -> i32 {
    boundary(err, || {
        let g = unsafe { generation(g) }?;
        if count == 0 || ids.is_null() {
            return Err(Error::invalid_argument("ids is NULL or empty"));
        }
        g.inner.prompt_tokens(unsafe { std::slice::from_raw_parts(ids, count as usize) })
    })
}

unsafe extern "C" fn x_generation_step(
    g: *mut c_void,
    out: *mut abi::turbo_generation_chunk,
    err: *mut abi::turbo_error,
) -> i32 {
    boundary(err, || {
        let g = unsafe { generation(g) }?;
        let o = out_ptr(out, "generation_step")?;
        check_size::<abi::turbo_generation_chunk>("turbo_generation_chunk", o.struct_size)?;
        g.chunk.clear();
        g.inner.step(&mut g.chunk)?;
        let c = &g.chunk;
        let full = abi::turbo_generation_chunk {
            struct_size: o.struct_size,
            sequence: c.sequence,
            n_tokens: c.tokens.len() as u32,
            n_logprobs: c.logprobs.len() as u32,
            tokens: c.tokens.as_ptr(),
            text: text_of(&c.text),
            logprobs: if c.logprobs.is_empty() { std::ptr::null() } else { c.logprobs.as_ptr() },
            done: c.done as u32,
            finish_reason: c.finish_reason.as_abi(),
            prompt_tokens: c.prompt_tokens,
            generated_tokens: c.generated_tokens,
        };
        unsafe { write_sized(&full, out, o.struct_size) };
        Ok(())
    })
}

unsafe extern "C" fn x_generation_cancel(g: *mut c_void) {
    if let Ok(g) = unsafe { generation(g) } {
        g.inner.cancel();
    }
}

unsafe extern "C" fn x_generation_release(g: *mut c_void) {
    if !g.is_null() {
        // A Drop panic must not unwind across the plugin boundary.
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            drop(unsafe { Box::from_raw(g as *mut ExportGeneration) });
        }))
        .is_err()
        {
            eprintln!("turbo plugin: panic inside x_generation_release");
        }
    }
}

/// A vtable kept alive for the program's lifetime. The raw pointers inside
/// are static strings and a leaked `ExportState`, safe to share.
pub struct StaticVtbl(Box<abi::turbo_provider_vtbl>);

// SAFETY: the vtable's pointers refer to `'static` data and a leaked state
// whose provider is `Send + Sync`; the shims are plain functions.
unsafe impl Send for StaticVtbl {}
unsafe impl Sync for StaticVtbl {}

impl StaticVtbl {
    /// Pointer to hand to the core.
    pub fn as_ptr(&self) -> *const abi::turbo_provider_vtbl {
        &*self.0
    }
}

/// Environment variable a provider library reads once, in this function
/// only, to report a provider ABI version other than the one it was built
/// with.
///
/// It exists for one conformance case: a provider whose ABI does not match
/// the core's must be refused with `TURBO_E_ABI_MISMATCH` naming both
/// versions, and nothing else in the tree can produce that without shipping
/// a deliberately broken library. It is read where the vtable is built, so
/// it can only affect a provider loaded through `turbo_provider_get`, never
/// a built-in one, and only the first load in a process (the vtable is
/// cached for the program's lifetime). A value that is not a `uint32_t` is
/// ignored and the real version is reported.
pub const ENV_ABI_VERSION_OVERRIDE: &str = "TURBO_PROVIDER_ABI_VERSION_OVERRIDE";

/// The provider ABI version this library reports, which is the one it was
/// built with unless [`ENV_ABI_VERSION_OVERRIDE`] says otherwise.
fn reported_abi_version() -> u32 {
    std::env::var(ENV_ABI_VERSION_OVERRIDE)
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .unwrap_or(abi::TURBO_PROVIDER_ABI_VERSION)
}

/// Build a vtable for a provider. Store the result in a `OnceLock` so it
/// lives for the program's lifetime; [`export_provider!`] does that.
pub fn make_vtbl(provider: Arc<dyn Provider>, id: &'static CStr, version: &'static CStr) -> StaticVtbl {
    let state = Box::into_raw(Box::new(ExportState { provider, id, version }));
    // SAFETY: just allocated.
    let st = unsafe { &*state };
    StaticVtbl(Box::new(abi::turbo_provider_vtbl {
        struct_size: std::mem::size_of::<abi::turbo_provider_vtbl>() as u32,
        abi_version: reported_abi_version(),
        id: st.id.as_ptr(),
        version: st.version.as_ptr(),
        state: state as *mut c_void,
        device_count: Some(x_device_count),
        device_info: Some(x_device_info),
        capability: Some(x_capability),
        can_run: Some(x_can_run),
        context_create: Some(x_context_create),
        context_release: Some(x_context_release),
        buffer_alloc: Some(x_buffer_alloc),
        buffer_import: Some(x_buffer_import),
        buffer_read: Some(x_buffer_read),
        buffer_export: Some(x_buffer_export),
        buffer_release: Some(x_buffer_release),
        model_load: Some(x_model_load),
        model_info: Some(x_model_info),
        model_label: Some(x_model_label),
        model_io_info: Some(x_model_io_info),
        model_release: Some(x_model_release),
        session_create: Some(x_session_create),
        session_write_text: Some(x_session_write_text),
        session_write_tokens: Some(x_session_write_tokens),
        session_write_pairs: Some(x_session_write_pairs),
        session_write_text_classify: Some(x_session_write_text_classify),
        session_bind: Some(x_session_bind),
        session_run: Some(x_session_run),
        session_stats: Some(x_session_stats),
        session_release: Some(x_session_release),
        generation_create: Some(x_generation_create),
        generation_prompt: Some(x_generation_prompt),
        generation_prompt_tokens: Some(x_generation_prompt_tokens),
        generation_step: Some(x_generation_step),
        generation_cancel: Some(x_generation_cancel),
        generation_release: Some(x_generation_release),
    }))
}

/// Define `turbo_provider_get` for a Rust provider crate.
///
/// ```ignore
/// turbo_core::export_provider!(c"mock", c"2.0.0", || std::sync::Arc::new(turbo_core::mock::MockProvider::new()));
/// ```
#[macro_export]
macro_rules! export_provider {
    ($id:expr, $version:expr, $ctor:expr) => {
        /// Provider entry point. Returns NULL for an unsupported core ABI version.
        #[no_mangle]
        pub unsafe extern "C" fn turbo_provider_get(core_abi_version: u32) -> *const $crate::abi::turbo_provider_vtbl {
            static VTBL: ::std::sync::OnceLock<$crate::plugin_export::StaticVtbl> = ::std::sync::OnceLock::new();
            if core_abi_version != $crate::abi::TURBO_PROVIDER_ABI_VERSION {
                return ::std::ptr::null();
            }
            let vt = VTBL.get_or_init(|| {
                let provider: ::std::sync::Arc<dyn $crate::provider::Provider> = ($ctor)();
                $crate::plugin_export::make_vtbl(provider, $id, $version)
            });
            vt.as_ptr()
        }
    };
}
