//! Provider plugins: loading a provider library and adapting its C vtable to
//! the core's provider traits.
//!
//! Loading validates the ABI version, the vtable size, and every required
//! entry point before the provider is registered. After that the adapter is
//! a thin translation layer: it builds ABI descriptors from core types, calls
//! the vtable, and converts results back. Output buffers of a run are
//! borrowed from the provider session; the core's result lease guarantees no
//! further run while they are referenced, and the session handle stays alive
//! through the `Arc` chain.

use std::ffi::{c_void, CStr};
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Arc;

use turbo_abi as abi;

/// Upper bound on outputs one plugin result may carry; a provider reporting
/// more is treated as broken (`TURBO_E_INTERNAL`).
pub const MAX_PLUGIN_OUTPUTS: u32 = 256;
/// Upper bound on spans one plugin result may carry (rows x tokens of the
/// largest sensible session), same rule.
pub const MAX_PLUGIN_SPANS: u32 = 1 << 24;

use crate::abi_convert::{
    buffer_desc_to_abi, capability_from_abi, classify_options_to_abi, device_info_from_abi, embed_options_to_abi,
    get_str, model_info_from_abi, native_handle_from_abi, native_handle_to_abi, rerank_options_to_abi,
    session_stats_from_abi, span_from_abi, tensor_info_from_abi, text_of, GenerateDescViews, KvViews,
};
use crate::buffer::{BufferDesc, NativeHandle, ProviderBuffer};
use crate::bundle::Bundle;
use crate::error::{Error, Result};
use crate::provider::{
    Capability, Chunk, ClassifyOptions, ContextDesc, DeviceInfo, EmbedOptions, GenerateDesc, Message, ModelDesc,
    ModelInfo, Output, Provider, ProviderContext, ProviderGeneration, ProviderModel, ProviderResult, ProviderSession,
    RerankOptions, RunOptions, SessionDesc, SessionStats, TokenBatch,
};
use crate::types::{Aggregation, FinishReason, HandleKind, Modality, Task};

/// A loaded provider library plus the provider adapter over its vtable.
///
/// The library stays mapped for the life of the process. The runtimes
/// providers wrap (CUDA, ONNX Runtime, OpenVINO, llama.cpp) keep worker
/// threads, driver contexts, and global destructors that are not safe to
/// tear down by `dlclose`; unloading one and loading it again in the same
/// process is where the intermittent crashes were.
pub struct LoadedProvider {
    /// The library, leaked on purpose (see the type documentation).
    pub library: &'static libloading::Library,
    /// The adapter.
    pub provider: Arc<PluginProvider>,
}

/// Load a provider library and validate its vtable.
pub fn load(path: &Path) -> Result<LoadedProvider> {
    // SAFETY: loading a shared library runs its initializers; that is the
    // documented contract of a provider library.
    let library: &'static libloading::Library = Box::leak(Box::new(
        unsafe { libloading::Library::new(path) }
            .map_err(|e| Error::provider_load(format!("cannot load provider library `{}`: {e}", path.display())))?,
    ));
    let entry: libloading::Symbol<unsafe extern "C" fn(u32) -> *const abi::turbo_provider_vtbl> =
        // SAFETY: the symbol has the documented signature; a mismatch is the provider's bug.
        unsafe { library.get(b"turbo_provider_get\0") }.map_err(|e| {
            Error::provider_load(format!(
                "provider library `{}` does not export turbo_provider_get: {e}",
                path.display()
            ))
        })?;
    // SAFETY: calling the provider's entry point with our ABI version.
    let vt = unsafe { entry(abi::TURBO_ABI_VERSION) };
    if vt.is_null() {
        return Err(Error::abi_mismatch(format!(
            "provider library `{}` declined ABI version {}",
            path.display(),
            abi::TURBO_ABI_VERSION
        )));
    }
    let provider = PluginProvider::new(vt, &path.display().to_string())?;
    Ok(LoadedProvider { library, provider: Arc::new(provider) })
}

fn new_err() -> abi::turbo_error {
    abi::turbo_error {
        struct_size: std::mem::size_of::<abi::turbo_error>() as u32,
        code: 0,
        field: 0,
        message: [0; abi::TURBO_ERROR_MESSAGE_LEN],
    }
}

/// Convert a provider status into a core error, taking the message from `e`.
fn take(rc: i32, e: &abi::turbo_error, what: &str) -> Result<()> {
    if rc == abi::TURBO_OK {
        return Ok(());
    }
    let msg = get_str(&e.message);
    let code = if e.code != 0 { e.code } else { rc };
    let text =
        if msg.is_empty() { format!("provider {what} failed with {}", crate::error::status_name(code)) } else { msg };
    Err(Error::new(code, text).with_field(e.field))
}

/// Adapter over a provider vtable. Thread-safe by the plugin contract.
pub struct PluginProvider {
    vt: NonNull<abi::turbo_provider_vtbl>,
    id: String,
    version: String,
}

// SAFETY: the plugin contract requires provider-level entry points to be
// callable from any thread; per-handle single-owner rules are enforced by the core.
unsafe impl Send for PluginProvider {}
unsafe impl Sync for PluginProvider {}

macro_rules! require {
    ($vt:expr, $field:ident, $where:expr) => {
        $vt.$field.ok_or_else(|| {
            Error::provider_load(format!(
                "provider library `{}` vtable has a NULL `{}` entry",
                $where,
                stringify!($field)
            ))
        })?
    };
}

impl PluginProvider {
    /// Validate and wrap a vtable pointer.
    pub fn new(vt: *const abi::turbo_provider_vtbl, origin: &str) -> Result<Self> {
        let ptr = NonNull::new(vt as *mut abi::turbo_provider_vtbl)
            .ok_or_else(|| Error::provider_load(format!("provider `{origin}` returned a NULL vtable")))?;
        // SAFETY: non-null vtable from the provider's entry point.
        let v = unsafe { ptr.as_ref() };
        if v.abi_version != abi::TURBO_PROVIDER_ABI_VERSION {
            return Err(Error::abi_mismatch(format!(
                "provider `{origin}` implements provider ABI {} but this core is {}",
                v.abi_version,
                abi::TURBO_PROVIDER_ABI_VERSION
            )));
        }
        let expected = std::mem::size_of::<abi::turbo_provider_vtbl>() as u32;
        if v.struct_size != expected {
            return Err(Error::abi_mismatch(format!(
                "provider `{origin}` vtable struct_size is {} but this core expects {expected}",
                v.struct_size
            )));
        }
        if v.id.is_null() || v.version.is_null() {
            return Err(Error::provider_load(format!("provider `{origin}` vtable has a NULL id or version")));
        }
        // SAFETY: NUL-terminated strings by contract.
        let id = unsafe { CStr::from_ptr(v.id) }
            .to_str()
            .map_err(|_| Error::provider_load(format!("provider `{origin}` id is not UTF-8")))?
            .to_string();
        let version = unsafe { CStr::from_ptr(v.version) }
            .to_str()
            .map_err(|_| Error::provider_load(format!("provider `{origin}` version is not UTF-8")))?
            .to_string();
        if id.is_empty() {
            return Err(Error::provider_load(format!("provider `{origin}` id is empty")));
        }
        // Required entry points.
        require!(v, device_count, origin);
        require!(v, device_info, origin);
        require!(v, capability, origin);
        require!(v, can_run, origin);
        require!(v, context_create, origin);
        require!(v, context_release, origin);
        require!(v, buffer_alloc, origin);
        require!(v, buffer_read, origin);
        require!(v, buffer_release, origin);
        require!(v, model_load, origin);
        require!(v, model_info, origin);
        require!(v, model_label, origin);
        require!(v, model_release, origin);
        require!(v, session_create, origin);
        require!(v, session_run, origin);
        require!(v, session_stats, origin);
        require!(v, session_release, origin);
        if v.generation_create.is_some() {
            require!(v, generation_prompt, origin);
            require!(v, generation_prompt_tokens, origin);
            require!(v, generation_step, origin);
            require!(v, generation_cancel, origin);
            require!(v, generation_release, origin);
        }
        Ok(Self { vt: ptr, id, version })
    }

    fn vt(&self) -> &abi::turbo_provider_vtbl {
        // SAFETY: validated at construction; the library outlives the adapter.
        unsafe { self.vt.as_ref() }
    }
}

impl Provider for PluginProvider {
    fn id(&self) -> &str {
        &self.id
    }

    fn version(&self) -> &str {
        &self.version
    }

    fn devices(&self) -> Result<Vec<DeviceInfo>> {
        let vt = self.vt();
        let mut e = new_err();
        let mut n = 0u32;
        // SAFETY: validated non-null entry point; pointers are valid for the call.
        let rc = unsafe { (vt.device_count.unwrap())(vt.state, &mut n, &mut e) };
        take(rc, &e, "device_count")?;
        let mut out = Vec::with_capacity(n as usize);
        for ordinal in 0..n {
            let mut info = abi::turbo_device_info {
                struct_size: std::mem::size_of::<abi::turbo_device_info>() as u32,
                kind: 0,
                ordinal: 0,
                vendor_id: 0,
                caps: 0,
                memory_total: 0,
                memory_free: 0,
                name: [0; 128],
                vendor: [0; 64],
                provider_id: [0; 32],
                provider_version: [0; 32],
                runtime_version: [0; 64],
                driver_version: [0; 64],
            };
            let mut e = new_err();
            let rc = unsafe { (vt.device_info.unwrap())(vt.state, ordinal, &mut info, &mut e) };
            take(rc, &e, "device_info")?;
            let d = device_info_from_abi(&info)?;
            if d.ordinal != ordinal {
                return Err(Error::internal(format!(
                    "provider `{}` reported ordinal {} for device index {ordinal}",
                    self.id, d.ordinal
                )));
            }
            out.push(d);
        }
        Ok(out)
    }

    fn capability(&self, ordinal: u32, task: Task, modality: Modality) -> Capability {
        let vt = self.vt();
        let mut c = abi::turbo_capability {
            struct_size: std::mem::size_of::<abi::turbo_capability>() as u32,
            status: 0,
            dtype: 0,
            reference_dtype: 0,
            cosine_floor: 0.0,
            max_abs_error: 0.0,
            deterministic: 0,
            reserved: 0,
            notes: [0; 128],
        };
        let mut e = new_err();
        let rc =
            unsafe { (vt.capability.unwrap())(vt.state, ordinal, task.as_abi(), modality.as_abi(), &mut c, &mut e) };
        // A capability query cannot fail softly: a provider error means "unsupported, with a note".
        match take(rc, &e, "capability").and_then(|()| capability_from_abi(&c)) {
            Ok(cap) => cap,
            Err(err) => {
                let mut cap = Capability::unsupported();
                cap.notes = format!("capability query failed: {err}");
                cap
            }
        }
    }

    fn can_run(&self, ordinal: u32, bundle: &Bundle, task: Task, modality: Modality) -> Result<()> {
        let vt = self.vt();
        let dir = bundle.dir().display().to_string();
        let mut e = new_err();
        let rc = unsafe {
            (vt.can_run.unwrap())(vt.state, ordinal, text_of(&dir), task.as_abi(), modality.as_abi(), &mut e)
        };
        take(rc, &e, "can_run")
    }

    fn create_context(&self, ordinal: u32, desc: &ContextDesc) -> Result<Arc<dyn ProviderContext>> {
        let vt = self.vt();
        let kv = KvViews::new(&desc.options);
        let d = abi::turbo_context_desc {
            struct_size: std::mem::size_of::<abi::turbo_context_desc>() as u32,
            flags: 0,
            n_options: kv.len(),
            reserved: 0,
            options: kv.ptr(),
            next: std::ptr::null(),
        };
        let mut ctx: *mut c_void = std::ptr::null_mut();
        let mut e = new_err();
        let rc = unsafe { (vt.context_create.unwrap())(vt.state, ordinal, &d, &mut ctx, &mut e) };
        take(rc, &e, "context_create")?;
        if ctx.is_null() {
            return Err(Error::internal(format!("provider `{}` returned OK with a NULL context", self.id)));
        }
        Ok(Arc::new(PluginContext { vt: self.vt, ctx, ordinal, provider_id: self.id.clone() }))
    }
}

struct PluginContext {
    vt: NonNull<abi::turbo_provider_vtbl>,
    ctx: *mut c_void,
    ordinal: u32,
    provider_id: String,
}
unsafe impl Send for PluginContext {}
unsafe impl Sync for PluginContext {}

impl PluginContext {
    fn vt(&self) -> &abi::turbo_provider_vtbl {
        unsafe { self.vt.as_ref() }
    }

    fn wrap_buffer(&self, b: &abi::turbo_provider_buffer, owned: bool) -> Result<Arc<dyn ProviderBuffer>> {
        if b.handle.is_null() {
            return Err(Error::internal(format!(
                "provider `{}` returned a buffer with a NULL handle",
                self.provider_id
            )));
        }
        let desc = BufferDesc::from_abi(&b.desc)?;
        Ok(Arc::new(PluginBuffer {
            vt: self.vt,
            handle: b.handle,
            host_ptr: NonNull::new(b.host_ptr.cast()),
            desc,
            owned,
        }))
    }
}

impl Drop for PluginContext {
    fn drop(&mut self) {
        // SAFETY: all children are released before the last Arc drops.
        unsafe { (self.vt().context_release.unwrap())(self.ctx) };
    }
}

fn empty_provider_buffer() -> abi::turbo_provider_buffer {
    abi::turbo_provider_buffer {
        struct_size: std::mem::size_of::<abi::turbo_provider_buffer>() as u32,
        reserved: 0,
        handle: std::ptr::null_mut(),
        host_ptr: std::ptr::null_mut(),
        desc: abi::turbo_buffer_desc {
            struct_size: std::mem::size_of::<abi::turbo_buffer_desc>() as u32,
            placement: 0,
            dtype: 0,
            ndim: 0,
            shape: [0; abi::TURBO_MAX_RANK],
            strides: [0; abi::TURBO_MAX_RANK],
            bytes: 0,
            next: std::ptr::null(),
        },
    }
}

impl ProviderContext for PluginContext {
    fn ordinal(&self) -> u32 {
        self.ordinal
    }

    fn alloc(&self, desc: &BufferDesc) -> Result<Arc<dyn ProviderBuffer>> {
        let vt = self.vt();
        let d = buffer_desc_to_abi(desc, std::mem::size_of::<abi::turbo_buffer_desc>() as u32);
        let mut out = empty_provider_buffer();
        let mut e = new_err();
        let rc = unsafe { (vt.buffer_alloc.unwrap())(self.ctx, &d, &mut out, &mut e) };
        take(rc, &e, "buffer_alloc")?;
        self.wrap_buffer(&out, true)
    }

    fn import(&self, desc: &BufferDesc, handle: &NativeHandle) -> Result<Arc<dyn ProviderBuffer>> {
        let vt = self.vt();
        let f = vt
            .buffer_import
            .ok_or_else(|| Error::unsupported(format!("provider `{}` does not import buffers", self.provider_id)))?;
        let d = buffer_desc_to_abi(desc, std::mem::size_of::<abi::turbo_buffer_desc>() as u32);
        let h = native_handle_to_abi(handle);
        let mut out = empty_provider_buffer();
        let mut e = new_err();
        let rc = unsafe { f(self.ctx, &d, &h, &mut out, &mut e) };
        take(rc, &e, "buffer_import")?;
        self.wrap_buffer(&out, true)
    }

    fn load_model(&self, bundle: Arc<Bundle>, desc: &ModelDesc) -> Result<Arc<dyn ProviderModel>> {
        let vt = self.vt();
        let kv = KvViews::new(&desc.options);
        let d = abi::turbo_model_desc {
            struct_size: std::mem::size_of::<abi::turbo_model_desc>() as u32,
            n_options: kv.len(),
            options: kv.ptr(),
            next: std::ptr::null(),
        };
        let dir = bundle.dir().display().to_string();
        let mut model: *mut c_void = std::ptr::null_mut();
        let mut e = new_err();
        let rc = unsafe { (vt.model_load.unwrap())(self.ctx, text_of(&dir), &d, &mut model, &mut e) };
        take(rc, &e, "model_load")?;
        if model.is_null() {
            return Err(Error::internal(format!("provider `{}` returned OK with a NULL model", self.provider_id)));
        }
        // Fetch info eagerly so a broken provider fails at load, not at first use.
        // On failure the model handle is released here, exactly once.
        match fetch_model_info(vt, model, bundle.aggregation()?) {
            Ok(info) => Ok(Arc::new(PluginModel { vt: self.vt, model, info })),
            Err(e) => {
                // SAFETY: the provider returned this handle and nothing else holds it.
                unsafe { (vt.model_release.unwrap())(model) };
                Err(e)
            }
        }
    }
}

struct PluginBuffer {
    vt: NonNull<abi::turbo_provider_vtbl>,
    handle: *mut c_void,
    host_ptr: Option<NonNull<u8>>,
    desc: BufferDesc,
    owned: bool,
}
unsafe impl Send for PluginBuffer {}
unsafe impl Sync for PluginBuffer {}

impl PluginBuffer {
    fn vt(&self) -> &abi::turbo_provider_vtbl {
        unsafe { self.vt.as_ref() }
    }
}

impl Drop for PluginBuffer {
    fn drop(&mut self) {
        if self.owned {
            unsafe { (self.vt().buffer_release.unwrap())(self.handle) };
        }
    }
}

impl ProviderBuffer for PluginBuffer {
    fn desc(&self) -> &BufferDesc {
        &self.desc
    }

    fn host_ptr(&self) -> Option<NonNull<u8>> {
        self.host_ptr
    }

    fn read_to_host(&self, dst: &mut [u8]) -> Result<()> {
        if dst.len() as u64 != self.desc.bytes {
            return Err(Error::capacity(format!(
                "destination holds {} bytes but the buffer is {} bytes",
                dst.len(),
                self.desc.bytes
            )));
        }
        let mut e = new_err();
        let rc =
            unsafe { (self.vt().buffer_read.unwrap())(self.handle, dst.as_mut_ptr().cast(), dst.len() as u64, &mut e) };
        take(rc, &e, "buffer_read")
    }

    fn export(&self, kind: HandleKind) -> Result<NativeHandle> {
        let f = self.vt().buffer_export.ok_or_else(|| Error::unsupported("provider does not export native handles"))?;
        let mut out = abi::turbo_native_handle {
            struct_size: std::mem::size_of::<abi::turbo_native_handle>() as u32,
            kind: 0,
            handle: 0,
            aux: 0,
            offset: 0,
        };
        let mut e = new_err();
        let rc = unsafe { f(self.handle, kind.as_abi(), &mut out, &mut e) };
        take(rc, &e, "buffer_export")?;
        native_handle_from_abi(&out)
    }

    fn plugin_handle(&self) -> Option<(*const c_void, *mut c_void)> {
        Some((self.vt.as_ptr().cast(), self.handle))
    }
}

struct PluginModel {
    vt: NonNull<abi::turbo_provider_vtbl>,
    model: *mut c_void,
    info: ModelInfo,
}
unsafe impl Send for PluginModel {}
unsafe impl Sync for PluginModel {}

impl PluginModel {
    fn vt(&self) -> &abi::turbo_provider_vtbl {
        unsafe { self.vt.as_ref() }
    }
}

/// Fetch a loaded model's info, labels, and named tensors through the vtable.
fn fetch_model_info(
    vt: &abi::turbo_provider_vtbl,
    model: *mut c_void,
    aggregation: Option<Aggregation>,
) -> Result<ModelInfo> {
    {
        let mut m = abi::turbo_model_info {
            struct_size: std::mem::size_of::<abi::turbo_model_info>() as u32,
            task: 0,
            kind: 0,
            modality: 0,
            dim: 0,
            n_labels: 0,
            pooling: 0,
            normalize: 0,
            max_seq: 0,
            max_batch: 0,
            dtype_used: 0,
            fully_accelerated: 0,
            stage_placement: [0; abi::TURBO_STAGE_COUNT],
            n_inputs: 0,
            n_outputs: 0,
            vocab_size: 0,
            model_id: [0; 128],
            revision: [0; 64],
            tokenizer_sha256: [0; 72],
            provider_id: [0; 32],
            prefix_query: [0; 128],
            prefix_document: [0; 128],
        };
        let mut e = new_err();
        let rc = unsafe { (vt.model_info.unwrap())(model, &mut m, &mut e) };
        take(rc, &e, "model_info")?;
        let mut labels = Vec::with_capacity(m.n_labels as usize);
        for i in 0..m.n_labels {
            let mut t = abi::turbo_text { ptr: std::ptr::null(), len: 0 };
            let mut e = new_err();
            let rc = unsafe { (vt.model_label.unwrap())(model, i, &mut t, &mut e) };
            take(rc, &e, "model_label")?;
            labels.push(unsafe { crate::abi_convert::text(&t, "label") }?.to_string());
        }
        let fetch_io = |direction: u32, n: u32| -> Result<Vec<TensorInfoVec>> {
            let mut v = Vec::with_capacity(n as usize);
            if n == 0 {
                return Ok(v);
            }
            let f = vt.model_io_info.ok_or_else(|| {
                Error::provider_load("provider reports named tensors but has no model_io_info entry point")
            })?;
            for i in 0..n {
                let mut t = abi::turbo_tensor_info {
                    struct_size: std::mem::size_of::<abi::turbo_tensor_info>() as u32,
                    dtype: 0,
                    ndim: 0,
                    reserved: 0,
                    shape: [0; abi::TURBO_MAX_RANK],
                    name: [0; 64],
                };
                let mut e = new_err();
                let rc = unsafe { f(model, direction, i, &mut t, &mut e) };
                take(rc, &e, "model_io_info")?;
                v.push(tensor_info_from_abi(&t)?);
            }
            Ok(v)
        };
        let inputs = fetch_io(abi::TURBO_IO_INPUT, m.n_inputs)?;
        let outputs = fetch_io(abi::TURBO_IO_OUTPUT, m.n_outputs)?;
        model_info_from_abi(&m, labels, inputs, outputs, aggregation)
    }
}

type TensorInfoVec = crate::provider::TensorInfo;

impl Drop for PluginModel {
    fn drop(&mut self) {
        unsafe { (self.vt().model_release.unwrap())(self.model) };
    }
}

impl ProviderModel for PluginModel {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn create_session(&self, desc: &SessionDesc) -> Result<Box<dyn ProviderSession>> {
        let vt = self.vt();
        let kv = KvViews::new(&desc.options);
        let d = abi::turbo_session_desc {
            struct_size: std::mem::size_of::<abi::turbo_session_desc>() as u32,
            max_batch: desc.max_batch,
            max_seq: desc.max_seq,
            n_options: kv.len(),
            options: kv.ptr(),
            next: std::ptr::null(),
        };
        let mut s: *mut c_void = std::ptr::null_mut();
        let mut e = new_err();
        let rc = unsafe { (vt.session_create.unwrap())(self.model, &d, &mut s, &mut e) };
        take(rc, &e, "session_create")?;
        if s.is_null() {
            return Err(Error::internal("provider returned OK with a NULL session"));
        }
        Ok(Box::new(PluginSession { vt: self.vt, session: s, provider_id: self.info.provider_id.clone() }))
    }

    fn create_generation(&self, desc: &GenerateDesc) -> Result<Box<dyn ProviderGeneration>> {
        let vt = self.vt();
        let f = vt.generation_create.ok_or_else(|| {
            Error::unsupported_task(format!("provider `{}` does not generate", self.info.provider_id))
        })?;
        let views = GenerateDescViews::new(desc);
        let d = views.desc(desc);
        let mut g: *mut c_void = std::ptr::null_mut();
        let mut e = new_err();
        let rc = unsafe { f(self.model, &d, &mut g, &mut e) };
        take(rc, &e, "generation_create")?;
        if g.is_null() {
            return Err(Error::internal("provider returned OK with a NULL generation"));
        }
        Ok(Box::new(PluginGeneration { vt: self.vt, generation: g }))
    }
}

struct PluginSession {
    vt: NonNull<abi::turbo_provider_vtbl>,
    session: *mut c_void,
    provider_id: String,
}
unsafe impl Send for PluginSession {}

impl PluginSession {
    fn vt(&self) -> &abi::turbo_provider_vtbl {
        unsafe { self.vt.as_ref() }
    }

    fn views(texts: &[&str]) -> Vec<abi::turbo_text> {
        texts.iter().map(|t| text_of(t)).collect()
    }
}

impl Drop for PluginSession {
    fn drop(&mut self) {
        unsafe { (self.vt().session_release.unwrap())(self.session) };
    }
}

impl ProviderSession for PluginSession {
    fn write_text(&mut self, texts: &[&str], opts: &EmbedOptions) -> Result<()> {
        let f = self
            .vt()
            .session_write_text
            .ok_or_else(|| Error::unsupported_task("provider session does not embed text"))?;
        let v = Self::views(texts);
        let o = embed_options_to_abi(opts);
        let mut e = new_err();
        let rc = unsafe { f(self.session, v.as_ptr(), v.len() as u32, &o, &mut e) };
        take(rc, &e, "session_write_text")
    }

    fn write_tokens(&mut self, batch: &TokenBatch<'_>) -> Result<()> {
        let f = self
            .vt()
            .session_write_tokens
            .ok_or_else(|| Error::unsupported_task("provider session does not accept tokens"))?;
        let b = abi::turbo_token_batch {
            struct_size: std::mem::size_of::<abi::turbo_token_batch>() as u32,
            batch: batch.batch,
            seq: batch.seq,
            row_stride: batch.row_stride,
            ids: batch.ids.as_ptr(),
            mask: batch.mask.as_ptr(),
            types: batch.types.map(|t| t.as_ptr()).unwrap_or(std::ptr::null()),
        };
        let mut e = new_err();
        let rc = unsafe { f(self.session, &b, &mut e) };
        take(rc, &e, "session_write_tokens")
    }

    fn write_pairs(&mut self, query: &str, docs: &[&str], opts: &RerankOptions) -> Result<()> {
        let f =
            self.vt().session_write_pairs.ok_or_else(|| Error::unsupported_task("provider session does not rerank"))?;
        let q = text_of(query);
        let v = Self::views(docs);
        let o = rerank_options_to_abi(opts);
        let mut e = new_err();
        let rc = unsafe { f(self.session, &q, v.as_ptr(), v.len() as u32, &o, &mut e) };
        take(rc, &e, "session_write_pairs")
    }

    fn write_text_classify(&mut self, texts: &[&str], opts: &ClassifyOptions) -> Result<()> {
        let f = self
            .vt()
            .session_write_text_classify
            .ok_or_else(|| Error::unsupported_task("provider session does not classify"))?;
        let v = Self::views(texts);
        let o = classify_options_to_abi(opts);
        let mut e = new_err();
        let rc = unsafe { f(self.session, v.as_ptr(), v.len() as u32, &o, &mut e) };
        take(rc, &e, "session_write_text_classify")
    }

    fn bind(&mut self, name: &str, buffer: Arc<dyn ProviderBuffer>) -> Result<()> {
        let f =
            self.vt().session_bind.ok_or_else(|| Error::unsupported_task("provider session does not take bindings"))?;
        let (vt, handle) = buffer.plugin_handle().ok_or_else(|| {
            Error::invalid_argument(format!("buffer was not allocated by provider `{}`", self.provider_id))
        })?;
        if vt != self.vt.as_ptr().cast_const().cast() {
            return Err(Error::invalid_argument(format!(
                "buffer belongs to a different provider than `{}`",
                self.provider_id
            )));
        }
        let mut e = new_err();
        let rc = unsafe { f(self.session, text_of(name), handle, &mut e) };
        take(rc, &e, "session_bind")
    }

    fn run(&mut self, opts: &RunOptions) -> Result<ProviderResult> {
        let vt = self.vt();
        let kv = KvViews::new(&opts.params);
        let o = abi::turbo_run_options {
            struct_size: std::mem::size_of::<abi::turbo_run_options>() as u32,
            n_params: kv.len(),
            params: kv.ptr(),
        };
        let mut r = abi::turbo_provider_result {
            struct_size: std::mem::size_of::<abi::turbo_provider_result>() as u32,
            n_outputs: 0,
            outputs: std::ptr::null(),
            n_spans: 0,
            reserved: 0,
            spans: std::ptr::null(),
        };
        let mut e = new_err();
        let rc = unsafe { (vt.session_run.unwrap())(self.session, &o, &mut r, &mut e) };
        take(rc, &e, "session_run")?;
        if r.n_outputs == 0 || r.outputs.is_null() {
            return Err(Error::internal(format!("provider `{}` returned a result with no outputs", self.provider_id)));
        }
        // A provider that reports absurd counts is broken, not trusted: the
        // slices below would otherwise span most of the address space.
        if r.n_outputs > MAX_PLUGIN_OUTPUTS {
            return Err(Error::internal(format!(
                "provider `{}` reports {} outputs; the plugin contract allows at most {MAX_PLUGIN_OUTPUTS}",
                self.provider_id, r.n_outputs
            )));
        }
        if r.n_spans > MAX_PLUGIN_SPANS {
            return Err(Error::internal(format!(
                "provider `{}` reports {} spans; the plugin contract allows at most {MAX_PLUGIN_SPANS}",
                self.provider_id, r.n_spans
            )));
        }
        // SAFETY: n_outputs readable entries by contract, valid until the next run.
        let raw = unsafe { std::slice::from_raw_parts(r.outputs, r.n_outputs as usize) };
        let mut outputs = Vec::with_capacity(raw.len());
        for (i, o) in raw.iter().enumerate() {
            if o.buffer.handle.is_null() {
                return Err(Error::internal(format!("provider `{}` output {i} has a NULL buffer", self.provider_id)));
            }
            if o.ndim as usize > abi::TURBO_MAX_RANK {
                return Err(Error::internal(format!("provider `{}` output {i} has ndim {}", self.provider_id, o.ndim)));
            }
            let desc = BufferDesc::from_abi(&o.buffer.desc)?;
            let name: Arc<str> = Arc::from(unsafe { crate::abi_convert::text(&o.name, "output name") }?);
            outputs.push(Output {
                name,
                buffer: Arc::new(PluginBuffer {
                    vt: self.vt,
                    handle: o.buffer.handle,
                    host_ptr: NonNull::new(o.buffer.host_ptr.cast()),
                    desc,
                    owned: false,
                }),
                shape: o.shape[..o.ndim as usize].to_vec(),
            });
        }
        let spans = if r.n_spans == 0 {
            Vec::new()
        } else if r.spans.is_null() {
            return Err(Error::internal(format!(
                "provider `{}` reports {} spans with a NULL array",
                self.provider_id, r.n_spans
            )));
        } else {
            unsafe { std::slice::from_raw_parts(r.spans, r.n_spans as usize) }.iter().map(span_from_abi).collect()
        };
        Ok(ProviderResult { outputs, spans })
    }

    fn stats(&self) -> Result<SessionStats> {
        let vt = self.vt();
        let mut s = abi::turbo_session_stats {
            struct_size: std::mem::size_of::<abi::turbo_session_stats>() as u32,
            reserved: 0,
            runs: 0,
            host_allocs: 0,
            h2d_bytes: 0,
            d2h_bytes: 0,
            input_bytes: 0,
            output_bytes: 0,
            provider_allocs: u64::MAX,
        };
        let mut e = new_err();
        let rc = unsafe { (vt.session_stats.unwrap())(self.session, &mut s, &mut e) };
        if rc != abi::TURBO_OK {
            // A failed stats call is an error, never a fabricated zero that a
            // receipt would then publish as a measurement.
            take(rc, &e, "session_stats")?;
            unreachable!("take returns an error for a non-OK status");
        }
        Ok(session_stats_from_abi(&s))
    }
}

struct PluginGeneration {
    vt: NonNull<abi::turbo_provider_vtbl>,
    generation: *mut c_void,
}
unsafe impl Send for PluginGeneration {}

impl PluginGeneration {
    fn vt(&self) -> &abi::turbo_provider_vtbl {
        unsafe { self.vt.as_ref() }
    }
}

impl Drop for PluginGeneration {
    fn drop(&mut self) {
        unsafe { (self.vt().generation_release.unwrap())(self.generation) };
    }
}

impl ProviderGeneration for PluginGeneration {
    fn prompt(&mut self, messages: &[Message<'_>]) -> Result<()> {
        let v: Vec<abi::turbo_message> = messages
            .iter()
            .map(|m| abi::turbo_message { role: text_of(m.role), content: text_of(m.content) })
            .collect();
        let mut e = new_err();
        let rc = unsafe { (self.vt().generation_prompt.unwrap())(self.generation, v.as_ptr(), v.len() as u32, &mut e) };
        take(rc, &e, "generation_prompt")
    }

    fn prompt_tokens(&mut self, ids: &[i32]) -> Result<()> {
        let mut e = new_err();
        let rc = unsafe {
            (self.vt().generation_prompt_tokens.unwrap())(self.generation, ids.as_ptr(), ids.len() as u32, &mut e)
        };
        take(rc, &e, "generation_prompt_tokens")
    }

    fn step(&mut self, out: &mut Chunk) -> Result<()> {
        let mut c = abi::turbo_generation_chunk {
            struct_size: std::mem::size_of::<abi::turbo_generation_chunk>() as u32,
            sequence: 0,
            n_tokens: 0,
            n_logprobs: 0,
            tokens: std::ptr::null(),
            text: abi::turbo_text { ptr: std::ptr::null(), len: 0 },
            logprobs: std::ptr::null(),
            done: 0,
            finish_reason: 0,
            prompt_tokens: 0,
            generated_tokens: 0,
        };
        let mut e = new_err();
        let rc = unsafe { (self.vt().generation_step.unwrap())(self.generation, &mut c, &mut e) };
        take(rc, &e, "generation_step")?;
        out.sequence = c.sequence;
        if c.n_tokens > 0 {
            if c.tokens.is_null() {
                return Err(Error::internal("provider chunk has n_tokens > 0 with NULL tokens"));
            }
            out.tokens.extend_from_slice(unsafe { std::slice::from_raw_parts(c.tokens, c.n_tokens as usize) });
        }
        out.text.push_str(unsafe { crate::abi_convert::text(&c.text, "chunk text") }?);
        if c.n_logprobs > 0 {
            if c.logprobs.is_null() {
                return Err(Error::internal("provider chunk has n_logprobs > 0 with NULL logprobs"));
            }
            out.logprobs.extend_from_slice(unsafe { std::slice::from_raw_parts(c.logprobs, c.n_logprobs as usize) });
        }
        out.done = c.done != 0;
        out.finish_reason = FinishReason::from_abi(c.finish_reason)?;
        out.prompt_tokens = c.prompt_tokens;
        out.generated_tokens = c.generated_tokens;
        Ok(())
    }

    fn cancel(&mut self) {
        unsafe { (self.vt().generation_cancel.unwrap())(self.generation) };
    }
}
