//! Per-registration cache of progressive pipeline annotators and head weights
//! backed by [`Loaded`]'s pre-loaded model and [`SessionPool`](crate::priority::SessionPool).
//!
//! Storing [`ProgressiveCache`] on [`Served`](crate::state::Served) avoids
//! reloading `model.safetensors`, recomputing SHA-256 digests, re-uploading
//! weights to the GPU, and recapturing CUDA Graphs on every
//! `AnalyzeDocumentProgressive` RPC, while bounding concurrent device runs by
//! `Loaded`'s configured session pool and dropping all cached state when the
//! bundle is unloaded or replaced.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use turbo::bundle::Bundle;
use turbo::status::Error as CoreError;
use turbo_analysis::{Annotator, Error, Execution, KeyId, Locality, ModelRef, Placed, Refs};
use turbo_analysis_embed::{
    Encoder, EncoderOptions, PooledSession, SparseAnnotator, SparseOptions, SubwordAnnotator, VectorAnnotator,
    VectorOptions,
};
use turbo_sat::{SatAnnotator, SatOptions};
use turbo_tagger::WordEncoder;

use super::heads::group::{GroupHead, open_head};
use super::heads::{HeadGroup, HeadStep};
use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT, field};
use crate::priority::{Held, Priority};
use crate::state::Loaded;

thread_local! {
    static ACTIVE_SCOPE: RefCell<Option<(Priority, Option<Arc<AtomicBool>>)>> = const { RefCell::new(None) };
}

/// Request-scoped scheduling priority and cooperative cancellation flag
/// propagated onto synchronous progressive worker threads.
#[derive(Clone, Default)]
pub(crate) struct CallScope {
    priority: Option<Priority>,
    cancel: Option<Arc<AtomicBool>>,
}

pub(crate) struct ScopeGuard {
    prev: Option<(Priority, Option<Arc<AtomicBool>>)>,
}

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        ACTIVE_SCOPE.with(|slot| {
            *slot.borrow_mut() = self.prev.take();
        });
    }
}

impl CallScope {
    pub(crate) fn new(priority: Priority, cancel: Arc<AtomicBool>) -> Self {
        Self { priority: Some(priority), cancel: Some(cancel) }
    }

    pub(crate) fn enter(&self) -> ScopeGuard {
        let next = (self.priority.unwrap_or(Priority::Interactive), self.cancel.clone());
        let prev = ACTIVE_SCOPE.with(|slot| slot.borrow_mut().replace(next));
        ScopeGuard { prev }
    }
}

fn current_scope() -> (Priority, Option<Arc<AtomicBool>>) {
    ACTIVE_SCOPE.with(|slot| slot.borrow().clone().unwrap_or((Priority::Interactive, None)))
}

/// Transparent [`Annotator`] adapter installing [`CallScope`] on the worker
/// thread executing a model-backed progressive step.
pub(crate) struct ScopedAnnotator<A> {
    inner: A,
    scope: CallScope,
}

impl<A> ScopedAnnotator<A> {
    pub(crate) fn new(inner: A, scope: CallScope) -> Self {
        Self { inner, scope }
    }
}

impl<A: Annotator> Annotator for ScopedAnnotator<A> {
    const ID: &'static str = A::ID;
    const VERSION: &'static str = A::VERSION;
    type Reads = A::Reads;
    type Writes = A::Writes;

    fn annotate(&self, input: Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let _g = self.scope.enter();
        self.inner.annotate(input)
    }

    fn annotate_batch<'d>(&self, inputs: Vec<Refs<'d, Self>>) -> Vec<Result<Self::Writes, Error>> {
        let _g = self.scope.enter();
        self.inner.annotate_batch(inputs)
    }

    fn annotate_batch_placed<'d>(&self, inputs: Vec<Refs<'d, Self>>) -> Vec<Placed<Self::Writes>> {
        let _g = self.scope.enter();
        self.inner.annotate_batch_placed(inputs)
    }

    fn requires(&self) -> &[KeyId] {
        self.inner.requires()
    }

    fn produces(&self) -> &[KeyId] {
        self.inner.produces()
    }

    fn model(&self) -> Option<ModelRef> {
        self.inner.model()
    }

    fn execution(&self) -> Option<Execution> {
        self.inner.execution()
    }

    fn config_hash(&self) -> u64 {
        self.inner.config_hash()
    }

    fn is_cacheable(&self) -> bool {
        self.inner.is_cacheable()
    }

    fn locality(&self) -> Locality {
        self.inner.locality()
    }

    fn cost_hint(&self) -> u32 {
        self.inner.cost_hint()
    }
}

struct HeldLease(Held);

impl PooledSession for HeldLease {
    fn as_ptr(&self) -> *mut turbo::turbo_session {
        self.0.slot().session.as_ptr()
    }
}

struct PoolSessionSource {
    loaded: Arc<Loaded>,
}

impl turbo_analysis_embed::SessionSource for PoolSessionSource {
    fn acquire(&self, min_ranges: u32) -> Result<Box<dyn PooledSession + '_>, CoreError> {
        let (priority, cancel) = current_scope();
        let held = self
            .loaded
            .acquire_blocking(priority, cancel.as_deref())
            .map_err(|f| CoreError::field(f.code, f.field, f.message))?;
        if min_ranges > 0 {
            held.reserve_ranges(min_ranges).map_err(|f| CoreError::field(f.code, f.field, f.message))?;
        }
        Ok(Box::new(HeldLease(held)))
    }
}

fn pool_encoder(loaded: &Arc<Loaded>, eo: &EncoderOptions) -> Result<Encoder, CoreError> {
    let source = Arc::new(PoolSessionSource { loaded: Arc::clone(loaded) });
    Encoder::from_pool(source, loaded.model_info, loaded.slots[0].info, loaded.device_info.clone(), eo)
}

#[derive(Default)]
struct CachedProgressive {
    bundle: Option<Arc<Bundle>>,
    subwords: Option<Arc<SubwordAnnotator>>,
    vectors: Option<Arc<VectorAnnotator>>,
    sparse: Option<Arc<SparseAnnotator>>,
    sat: Option<Arc<SatAnnotator>>,
    words: Option<Arc<Mutex<WordEncoder>>>,
    heads: HashMap<(HeadStep, String), Arc<GroupHead>>,
}

impl CachedProgressive {
    fn bundle(&mut self, loaded: &Loaded) -> Result<Arc<Bundle>, CoreError> {
        if let Some(b) = &self.bundle {
            return Ok(Arc::clone(b));
        }
        let b = Bundle::open(Path::new(&loaded.bundle))?;
        let expected = field(&loaded.model_info.manifest_sha256);
        if !expected.is_empty() && b.manifest_sha256 != expected {
            return Err(CoreError::new(
                turbo::status::BUNDLE_INTEGRITY,
                format!("manifest.json: SHA-256 is {}, loaded model expects {expected}", b.manifest_sha256),
            ));
        }
        let arc = Arc::new(b);
        self.bundle = Some(Arc::clone(&arc));
        Ok(arc)
    }
}

/// Cached progressive annotators bound to one [`Served`](crate::state::Served)
/// model registration.
#[derive(Default)]
pub(crate) struct ProgressiveCache {
    inner: Mutex<CachedProgressive>,
}

impl ProgressiveCache {
    fn lock(&self) -> std::sync::MutexGuard<'_, CachedProgressive> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub(crate) fn subword_annotator(&self, loaded: &Arc<Loaded>) -> Result<Arc<SubwordAnnotator>, Failure> {
        let mut g = self.lock();
        if let Some(s) = &g.subwords {
            return Ok(Arc::clone(s));
        }
        let map = |e: CoreError| Failure::new(e.code, e.field, e.message);
        let bundle = g.bundle(loaded).map_err(map)?;
        let ann = Arc::new(SubwordAnnotator::open_bundle(&bundle).map_err(map)?);
        g.subwords = Some(Arc::clone(&ann));
        Ok(ann)
    }

    pub(crate) fn vector_annotator(
        &self,
        loaded: &Arc<Loaded>,
        eo: &EncoderOptions,
    ) -> Result<Arc<VectorAnnotator>, Failure> {
        let mut g = self.lock();
        if let Some(v) = &g.vectors {
            return Ok(Arc::clone(v));
        }
        let map = |e: CoreError| Failure::new(e.code, e.field, e.message);
        let bundle = g.bundle(loaded).map_err(map)?;
        let opts = VectorOptions { encoder: eo.clone(), ..VectorOptions::default() };
        let ann = Arc::new(VectorAnnotator::open_with(&bundle, &opts, |o| pool_encoder(loaded, o)).map_err(map)?);
        g.vectors = Some(Arc::clone(&ann));
        Ok(ann)
    }

    pub(crate) fn sparse_annotator(
        &self,
        loaded: &Arc<Loaded>,
        eo: &EncoderOptions,
    ) -> Result<Arc<SparseAnnotator>, Failure> {
        let mut g = self.lock();
        if let Some(s) = &g.sparse {
            return Ok(Arc::clone(s));
        }
        let map = |e: CoreError| Failure::new(e.code, e.field, e.message);
        let bundle = g.bundle(loaded).map_err(map)?;
        let opts = SparseOptions { encoder: eo.clone(), ..SparseOptions::default() };
        let ann = Arc::new(SparseAnnotator::open_with(&bundle, &opts, |o| pool_encoder(loaded, o)).map_err(map)?);
        g.sparse = Some(Arc::clone(&ann));
        Ok(ann)
    }

    pub(crate) fn sat_annotator(
        &self,
        loaded: &Arc<Loaded>,
        eo: &EncoderOptions,
    ) -> Result<Arc<SatAnnotator>, Failure> {
        let mut g = self.lock();
        if let Some(s) = &g.sat {
            return Ok(Arc::clone(s));
        }
        let map_sat = |e: CoreError| Failure::new(TURBO_E_INVALID_ARGUMENT, 0, format!("turbo:sat: {e}"));
        let bundle = g.bundle(loaded).map_err(map_sat)?;
        let opts = SatOptions { encoder: eo.clone(), ..SatOptions::default() };
        let ann = Arc::new(SatAnnotator::open_with(&bundle, &opts, |o| pool_encoder(loaded, o)).map_err(map_sat)?);
        g.sat = Some(Arc::clone(&ann));
        Ok(ann)
    }

    pub(crate) fn head_group(
        &self,
        loaded: &Arc<Loaded>,
        eo: &EncoderOptions,
        steps: &[(HeadStep, Option<&str>)],
        who: &str,
        scope: CallScope,
    ) -> Result<Arc<HeadGroup>, Failure> {
        let mut g = self.lock();
        let map_head = |e: CoreError| Failure::new(e.code, 0, format!("bundle heads: {}", e.message));
        let bundle = g.bundle(loaded).map_err(map_head)?;
        let mut resolved = Vec::with_capacity(steps.len());
        for &(step, selected) in steps {
            let spec = step.resolve(&bundle, selected).map_err(|e| e.failure(who, &loaded.bundle))?;
            resolved.push((step, spec.clone()));
        }
        let mut heads = Vec::with_capacity(resolved.len());
        for (step, spec) in resolved {
            let key = (step, spec.name.clone());
            let head = match g.heads.get(&key) {
                Some(h) => Arc::clone(h),
                None => {
                    let opened = Arc::new(open_head(step, &bundle, &spec).map_err(map_head)?);
                    g.heads.insert(key, Arc::clone(&opened));
                    opened
                }
            };
            heads.push((step, head));
        }
        let words = match &g.words {
            Some(w) => Arc::clone(w),
            None => {
                let opened = WordEncoder::open_with(&bundle, eo, |o| pool_encoder(loaded, o)).map_err(map_head)?;
                let arc = Arc::new(Mutex::new(opened));
                g.words = Some(Arc::clone(&arc));
                arc
            }
        };
        HeadGroup::from_cached(&bundle, words, heads, scope).map_err(map_head)
    }
}
