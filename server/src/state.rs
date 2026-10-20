//! Loaded models, session pools, and runtime state. Models come and go
//! while serving: the analysis service's LoadBundle and UnloadBundle add
//! and remove entries, and every listing reads the current list.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use turbo::{TURBO_DTYPE_I32, TURBO_PLACE_PINNED, TURBO_TASK_EMBED, turbo_session_desc, turbo_session_info};

use crate::api::{self, *};
use crate::config::ModelConfig;
use crate::status::{SESSION_DESC, describe};

/// One session of a model, with its three pinned I32 buffers for raw token
/// rows, each [max_batch, max_seq]. Fields are dropped in order, children
/// first: the buffers, then the session.
pub(crate) struct Slot {
    /// ids, mask and types, and their host pointers.
    pub(crate) _buffers: [Buffer; 3],
    pub(crate) ptrs: [*mut u8; 3],
    /// turbo_session_get_info, read once at load: it is fixed at
    /// turbo_session_create.
    pub(crate) info: turbo_session_info,
    pub(crate) session: Session,
}

// The pointers are into the slot's own buffers, written only by the request
// that holds the slot.
unsafe impl Send for Slot {}
unsafe impl Sync for Slot {}

/// A loaded model and its sessions, dropped sessions first, then the model.
pub(crate) struct Loaded {
    pub(crate) slots: Vec<Slot>,
    pub(crate) model: Model,
    pub(crate) device: u32,
    pub(crate) revision: String,
    pub(crate) idle: Mutex<Vec<usize>>,
}

/// A session taken from its model's pool, returned when this is dropped.
pub struct Held {
    loaded: Arc<Loaded>,
    index: usize,
}

impl Held {
    pub(crate) fn slot(&self) -> &Slot {
        &self.loaded.slots[self.index]
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        self.loaded.idle.lock().unwrap_or_else(|e| e.into_inner()).push(self.index);
    }
}

impl Loaded {
    /// An idle session, or TURBO_E_BUSY at once when every one is held.
    pub(crate) fn take(self: &Arc<Loaded>) -> api::Result<Held> {
        let index = self.idle.lock().unwrap_or_else(|e| e.into_inner()).pop().ok_or_else(|| {
            Failure::new(TURBO_E_BUSY, 0, format!("all {} sessions of the model are held", self.slots.len()))
        })?;
        Ok(Held { loaded: self.clone(), index })
    }
}

/// A model the server serves: its name, its configuration, and its load
/// once load() has made it.
pub(crate) struct Served {
    pub(crate) name: String,
    pub(crate) config: ModelConfig,
    pub(crate) loaded: OnceLock<Arc<Loaded>>,
}

/// Dropped models first, then contexts, then the runtime.
pub(crate) struct State {
    /// The served models, in serve order; LoadBundle and UnloadBundle
    /// add and remove entries while serving.
    pub(crate) models: RwLock<Vec<Arc<Served>>>,
    pub(crate) contexts: Mutex<Vec<(u32, Arc<Context>)>>,
    pub(crate) runtime: OnceLock<Runtime>,
    /// Serializes load(): one load at a time, so two LoadBundle calls
    /// never load the same new model twice.
    loading: Mutex<()>,
}

fn not_found(name: &str) -> Failure {
    Failure::new(TURBO_E_BUNDLE_NOT_FOUND, 0, format!("no model {name} is served here"))
}

impl State {
    pub(crate) fn new(names: Vec<String>, models: Vec<ModelConfig>) -> Self {
        Self {
            models: RwLock::new(
                names
                    .into_iter()
                    .zip(models)
                    .map(|(name, config)| Arc::new(Served { name, config, loaded: OnceLock::new() }))
                    .collect(),
            ),
            runtime: OnceLock::new(),
            contexts: Mutex::new(Vec::new()),
            loading: Mutex::new(()),
        }
    }

    /// The served model's name and, when it is loaded, its load. The
    /// answer is a snapshot: a load or unload after it returns does not
    /// change it.
    pub(crate) fn peek(&self, name: &str) -> api::Result<(String, Option<Arc<Loaded>>)> {
        let models = self.models.read().unwrap_or_else(|e| e.into_inner());
        let m = models.iter().find(|m| m.name == name).ok_or_else(|| not_found(name))?;
        Ok((m.name.clone(), m.loaded.get().cloned()))
    }

    /// A ready model, the version empty or its revision.
    pub(crate) fn ready(&self, name: &str, version: &str) -> api::Result<(String, Arc<Loaded>)> {
        let (name, loaded) = self.peek(name)?;
        let l =
            loaded.ok_or_else(|| Failure::new(TURBO_E_INVALID_STATE, 0, format!("model {name} is not loaded yet")))?;
        if !version.is_empty() && version != l.revision {
            return Err(Failure::new(
                TURBO_E_BUNDLE_NOT_FOUND,
                0,
                format!("model {name} has version {}, not {version}", l.revision),
            ));
        }
        Ok((name, l))
    }

    /// The served models as they are now: name, configuration and load,
    /// in serve order.
    pub(crate) fn listing(&self) -> Vec<Arc<Served>> {
        self.models.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Starts serving a model: it lists from here on, not loaded yet, and
    /// load() makes it. The name must not be served already.
    pub(crate) fn add(&self, name: String, config: ModelConfig) -> api::Result<()> {
        let mut models = self.models.write().unwrap_or_else(|e| e.into_inner());
        if models.iter().any(|m| m.name == name) {
            return Err(Failure::new(TURBO_E_INVALID_ARGUMENT, 0, format!("a model named {name} is already served")));
        }
        models.push(Arc::new(Served { name, config, loaded: OnceLock::new() }));
        Ok(())
    }

    /// Stops serving a name. Sessions a request still holds finish: the
    /// load outlives its entry through the holders' Arcs, and is dropped
    /// with the last of them.
    pub(crate) fn remove(&self, name: &str) -> api::Result<()> {
        let mut models = self.models.write().unwrap_or_else(|e| e.into_inner());
        let i = models.iter().position(|m| m.name == name).ok_or_else(|| not_found(name))?;
        models.remove(i);
        Ok(())
    }

    /// The served model by name, as a snapshot.
    pub(crate) fn served(&self, name: &str) -> api::Result<Arc<Served>> {
        self.listing().into_iter().find(|m| m.name == name).ok_or_else(|| not_found(name))
    }

    /// Loads every model in order, stopping at the first failure with the
    /// call, its status, the field it names and the library's message.
    /// One load at a time, so concurrent loads cannot load one new model
    /// twice.
    pub(crate) fn load(&self) -> std::result::Result<(), String> {
        let _one = self.loading.lock().unwrap_or_else(|e| e.into_inner());
        for m in self.listing() {
            if m.loaded.get().is_none() {
                self.load_model(&m)?;
            }
        }
        Ok(())
    }

    /// Loads one named model; an already loaded model is a no-op. Every
    /// other pending registration is untouched, so one LoadBundle's
    /// failure cannot fail or roll back another's.
    pub(crate) fn load_one(&self, name: &str) -> std::result::Result<(), String> {
        let _one = self.loading.lock().unwrap_or_else(|e| e.into_inner());
        let m = self.served(name).map_err(|f| describe(&f, &[]))?;
        if m.loaded.get().is_none() {
            self.load_model(&m)?;
        }
        Ok(())
    }

    /// One model's load: its context, model, sessions and pinned buffers,
    /// published on its entry. The loading mutex is held by the caller, so
    /// this entry cannot load twice.
    fn load_model(&self, m: &Arc<Served>) -> std::result::Result<(), String> {
        let fail = |call: &str, fields: &[&str]| {
            let (name, call) = (m.name.clone(), call.to_string());
            let fields: Vec<String> = fields.iter().map(|s| s.to_string()).collect();
            move |f: Failure| {
                let fields: Vec<&str> = fields.iter().map(String::as_str).collect();
                format!("model {name}: {call}: {}", describe(&f, &fields))
            }
        };
        let rt = match self.runtime.get() {
            Some(rt) => rt,
            None => {
                let rt = Runtime::create().map_err(fail("turbo_runtime_create", &[]))?;
                self.runtime.get_or_init(|| rt)
            }
        };
        let c = &m.config;
        let device = match c.device {
            crate::config::Device::Index(i) => i,
            crate::config::Device::Select => rt.select(TURBO_TASK_EMBED).map_err(fail("turbo_runtime_select", &[]))?,
        };
        let ctx = {
            let mut contexts = self.contexts.lock().unwrap_or_else(|e| e.into_inner());
            if !contexts.iter().any(|(d, _)| *d == device) {
                let ctx = rt.context(device).map_err(fail("turbo_context_create", &[]))?;
                contexts.push((device, Arc::new(ctx)));
            }
            contexts.iter().find(|(d, _)| *d == device).unwrap().1.clone()
        };
        let model = ctx.load(&c.bundle).map_err(fail("turbo_model_load", &[]))?;
        let info = model.info().map_err(fail("turbo_model_get_info", &[]))?;
        let desc = turbo_session_desc {
            struct_size: size_of::<turbo_session_desc>() as u32,
            max_batch: c.max_batch,
            max_seq: c.max_seq,
            precision: c.precision,
            tuning: 0,
            tuning_budget_ms: 0,
            cpu_threads: 0,
            max_ranges: 0,
        };
        let mut slots = Vec::new();
        for _ in 0..c.sessions {
            let session = model.session(&desc).map_err(fail("turbo_session_create", &SESSION_DESC))?;
            let si = session.info().map_err(fail("turbo_session_get_info", &[]))?;
            let alloc = || {
                let b = ctx
                    .alloc(TURBO_PLACE_PINNED, TURBO_DTYPE_I32, si.max_batch, si.max_seq)
                    .map_err(fail("turbo_buffer_alloc", &[]))?;
                let p = b.host_ptr().map_err(fail("turbo_buffer_host_ptr", &[]))?;
                Ok::<_, String>((b, p))
            };
            let ((ids, a), (mask, b), (types, t)) = (alloc()?, alloc()?, alloc()?);
            slots.push(Slot { _buffers: [ids, mask, types], ptrs: [a, b, t], info: si, session });
        }
        let idle = Mutex::new((0..slots.len()).rev().collect());
        let loaded = Loaded { slots, model, device, revision: api::field(&info.revision), idle };
        // The caller holds the loading mutex: no other load can have set
        // this entry's load.
        let _ = m.loaded.set(Arc::new(loaded));
        Ok(())
    }
}

/// The model's properties, read from the library on each call.
pub(crate) fn properties(state: &State, loaded: &Loaded) -> api::Result<HashMap<String, String>> {
    let rt = state.runtime.get().ok_or_else(|| Failure::new(TURBO_E_INVALID_STATE, 0, "no runtime"))?;
    let mi = loaded.model.info()?;
    let si = loaded.slots[0].info;
    let di = rt.device_info(loaded.device)?;
    let cap = rt.capability(loaded.device, mi.task, si.precision)?;
    let dims: Vec<String> = mi
        .output_dims
        .iter()
        .take((mi.output_dims_count as usize).min(mi.output_dims.len()))
        .map(u32::to_string)
        .collect();
    let p: Vec<(&str, String)> = vec![
        ("model_info.task", names::task(mi.task)),
        ("model_info.dim", mi.dim.to_string()),
        ("model_info.pooling", names::pooling(mi.pooling)),
        ("model_info.normalize", names::normalize(mi.normalize)),
        ("model_info.max_seq", mi.max_seq.to_string()),
        ("model_info.max_batch", mi.max_batch.to_string()),
        ("model_info.dtype", names::dtype(mi.dtype)),
        ("model_info.model_id", api::field(&mi.model_id)),
        ("model_info.revision", api::field(&mi.revision)),
        ("model_info.manifest_sha256", api::field(&mi.manifest_sha256)),
        ("model_info.artifact_sha256", api::field(&mi.artifact_sha256)),
        ("model_info.tokenizer_sha256", api::field(&mi.tokenizer_sha256)),
        ("model_info.prefix_query", api::field(&mi.prefix_query)),
        ("model_info.prefix_document", api::field(&mi.prefix_document)),
        ("model_info.output_dims_count", mi.output_dims_count.to_string()),
        ("model_info.output_dims", dims.join(",")),
        ("session_info.max_batch", si.max_batch.to_string()),
        ("session_info.max_seq", si.max_seq.to_string()),
        ("session_info.precision", names::precision(si.precision)),
        ("session_info.compute_dtype", names::dtype(si.compute_dtype)),
        ("device_info.kind", names::device_kind(di.kind)),
        ("device_info.ordinal", di.ordinal.to_string()),
        ("device_info.unified_memory", di.unified_memory.to_string()),
        ("device_info.memory_total", di.memory_total.to_string()),
        ("device_info.arch", api::field(&di.arch)),
        ("device_info.name", api::field(&di.name)),
        ("device_info.vendor", api::field(&di.vendor)),
        ("device_info.backend", api::field(&di.backend)),
        ("device_info.runtime_version", api::field(&di.runtime_version)),
        ("device_info.driver_version", api::field(&di.driver_version)),
        ("capability.status", names::cap(cap.status)),
        ("capability.dtype", names::dtype(cap.dtype)),
        ("capability.options_honored", cap.options_honored.to_string()),
        // Display gives the shortest decimal that reads back as the same f32.
        ("capability.cosine_floor", cap.cosine_floor.to_string()),
        ("capability.speed_ratio", cap.speed_ratio.to_string()),
        ("capability.benchmark", api::field(&cap.benchmark)),
        ("capability.reason", api::field(&cap.reason)),
    ];
    Ok(p.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}
