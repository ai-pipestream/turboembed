//! The engine: one runtime, every served model with its session pool, and
//! the task functions the routes call. Sessions are fixed-shape and
//! single-owner, so each model keeps a pool per (batch, seq) bucket; a
//! request is served by the smallest bucket that fits it and padded within
//! it, split into bucket-sized chunks when it has more rows than the
//! largest bucket, and rejected with the limit when it is longer than the
//! longest bucket. Nothing is truncated unless the request asks for it.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

use tokio::sync::{mpsc, OwnedSemaphorePermit, Semaphore};
use turbo::handles::{Context, Model, ResultHandle, Session};
use turbo::provider::{
    ClassifyOptions, EmbedOptions, GenerateDesc, Message, ModelDesc, ModelInfo, Options, RerankOptions, RunOptions,
    SessionDesc, Span,
};
use turbo::tokenizer::Tokenizer;
use turbo::types::{Aggregation, DType, FinishReason, ModelKind, Normalize, Placement, Pooling, PromptRole, Truncate};
use turbo::{abi, ContextDesc, DeviceInfo, DeviceSelector, Error as TurboError, Runtime, RuntimeDesc, SelectPolicy};

use crate::config::{Bucket, Config, ModelSpec};
use crate::error::{Result, ServeError};

/// A served model.
pub struct Served {
    /// The specification it was loaded from.
    pub spec: ModelSpec,
    /// Name clients use for this model.
    pub name: String,
    /// The loaded model handle.
    pub model: Arc<Model>,
    /// The bundle's tokenizer, when the core can load one; without it the
    /// server cannot count tokens and the longest bucket serves everything.
    pub tokenizer: Option<Arc<Tokenizer>>,
    /// The device the model runs on.
    pub device: DeviceInfo,
    buckets: Vec<Arc<PoolBucket>>,
    generations: Arc<Semaphore>,
}

struct PoolBucket {
    shape: Bucket,
    idle: Mutex<Vec<Arc<Session>>>,
    slots: Arc<Semaphore>,
}

/// A session checked out of a bucket; returned on drop.
pub struct Leased {
    /// The checked-out session.
    pub session: Arc<Session>,
    /// The bucket's shape; the caller splits its rows by `shape.batch`.
    pub shape: Bucket,
    /// The bucket the session came from; the lease keeps it alive.
    bucket: Arc<PoolBucket>,
    _permit: OwnedSemaphorePermit,
}

impl Drop for Leased {
    fn drop(&mut self) {
        self.bucket.idle.lock().unwrap_or_else(|p| p.into_inner()).push(self.session.clone());
    }
}

/// Every served model, by the name clients use; models load and unload
/// at run time through the repository extension.
pub struct Engine {
    runtime: Arc<Runtime>,
    models: RwLock<BTreeMap<String, Arc<Served>>>,
}

impl Engine {
    /// Load every provider library and model in the configuration. A model
    /// that fails to load fails the server: a half-configured server is
    /// not "ready".
    pub fn load(config: &Config) -> Result<Arc<Self>> {
        let provider_paths = config.provider_libs.iter().map(|p| p.to_string_lossy().into_owned()).collect();
        let runtime = turbo::create_runtime(RuntimeDesc { provider_paths, ..Default::default() })?;
        // A provider library that did not load is a configuration error, not
        // something to serve around: name the first one and stop.
        if let Some(f) = runtime.failures().into_iter().next() {
            return Err(ServeError::internal(format!("provider `{}` failed: {}", f.what, f.error)));
        }
        let engine = Arc::new(Self { runtime, models: RwLock::new(BTreeMap::new()) });
        for spec in &config.models {
            engine.load_model(spec)?;
        }
        if engine.is_empty() {
            return Err(ServeError::bad_request("no models configured; pass --model or --config"));
        }
        Ok(engine)
    }

    /// Load one more model; its name must be new.
    pub fn load_model(&self, spec: &ModelSpec) -> Result<String> {
        spec.validate()?;
        let served = Served::load(&self.runtime, spec)?;
        let name = served.name.clone();
        let mut models = self.models.write().unwrap_or_else(|p| p.into_inner());
        if models.contains_key(&name) {
            return Err(ServeError::bad_request(format!(
                "a model named `{name}` is already served; give the new one a `name=`"
            )));
        }
        eprintln!(
            "model `{}`: {} ({:?}) on {} ordinal {} via `{}`; buckets {}",
            name,
            served.model.info().model_id,
            served.model.info().kind,
            served.device.name,
            spec.ordinal,
            spec.provider,
            served.buckets.iter().map(|b| format!("{}x{}", b.shape.batch, b.shape.seq)).collect::<Vec<_>>().join(", ")
        );
        models.insert(name.clone(), Arc::new(served));
        Ok(name)
    }

    /// Unload a model. Requests already holding its `Arc` finish; new
    /// ones are not found. The sessions and the model are released when
    /// the last holder drops.
    pub fn unload(&self, name: &str) -> Result<()> {
        let mut models = self.models.write().unwrap_or_else(|p| p.into_inner());
        if models.remove(name).is_none() {
            return Err(ServeError::not_found(format!("no model named `{name}` to unload")));
        }
        eprintln!("model `{name}` unloaded");
        Ok(())
    }

    /// The model of that name, or a not-found naming what is served.
    pub fn model(&self, name: &str) -> Result<Arc<Served>> {
        let models = self.models.read().unwrap_or_else(|p| p.into_inner());
        models.get(name).cloned().ok_or_else(|| {
            ServeError::not_found(format!(
                "no model named `{name}`; served: {}",
                models.keys().cloned().collect::<Vec<_>>().join(", ")
            ))
        })
    }

    /// The served names, sorted.
    pub fn names(&self) -> Vec<String> {
        self.models.read().unwrap_or_else(|p| p.into_inner()).keys().cloned().collect()
    }

    /// Every served model, in name order.
    pub fn snapshot(&self) -> Vec<Arc<Served>> {
        self.models.read().unwrap_or_else(|p| p.into_inner()).values().cloned().collect()
    }

    /// True when no model is served.
    pub fn is_empty(&self) -> bool {
        self.models.read().unwrap_or_else(|p| p.into_inner()).is_empty()
    }
}

impl Served {
    fn load(runtime: &Arc<Runtime>, spec: &ModelSpec) -> Result<Self> {
        let index = runtime.select(&DeviceSelector {
            policy: SelectPolicy::Explicit,
            provider_id: spec.provider.clone(),
            ordinal: spec.ordinal,
            ..Default::default()
        })?;
        let device = runtime.device(index)?.info.clone();
        let ctx = Context::create(runtime.clone(), index, &ContextDesc::default())?;
        let model = ctx.load_model(&spec.bundle, &ModelDesc::default())?;
        let info = model.info();
        let name = spec.name.clone().unwrap_or_else(|| {
            info.model_id.rsplit('/').next().filter(|s| !s.is_empty()).unwrap_or(&info.model_id).to_string()
        });
        // The tokenizer is optional (a GGUF bundle carries none the core can
        // load); without it, bucket choice cannot count tokens and the
        // longest bucket serves every request.
        let tokenizer = match Tokenizer::from_bundle(model.bundle()) {
            Ok(t) => Some(t),
            Err(e) => {
                eprintln!("model `{name}`: no core tokenizer ({e}); requests use the longest bucket");
                None
            }
        };
        let shapes = if spec.buckets.is_empty() {
            let mut batches = vec![1u32, 8, info.max_batch];
            batches.retain(|&b| b <= info.max_batch);
            batches.sort_unstable();
            batches.dedup();
            batches.into_iter().map(|batch| Bucket { batch, seq: info.max_seq }).collect()
        } else {
            spec.buckets.clone()
        };
        let mut buckets = Vec::new();
        if info.kind != ModelKind::Generative {
            for shape in shapes {
                if shape.batch > info.max_batch || shape.seq > info.max_seq {
                    return Err(ServeError::bad_request(format!(
                        "model `{name}`: bucket {}x{} exceeds the model's limits {}x{}",
                        shape.batch, shape.seq, info.max_batch, info.max_seq
                    )));
                }
                let mut idle = Vec::new();
                for _ in 0..spec.sessions {
                    idle.push(model.create_session(&SessionDesc {
                        max_batch: shape.batch,
                        max_seq: shape.seq,
                        options: Options::default(),
                    })?);
                }
                buckets.push(Arc::new(PoolBucket {
                    shape,
                    idle: Mutex::new(idle),
                    slots: Arc::new(Semaphore::new(spec.sessions as usize)),
                }));
            }
            buckets.sort_by_key(|b| (b.shape.seq, b.shape.batch));
        }
        Ok(Self {
            spec: spec.clone(),
            name,
            model,
            tokenizer,
            device,
            buckets,
            generations: Arc::new(Semaphore::new(spec.generations as usize)),
        })
    }

    /// What the provider reports about the loaded model.
    pub fn info(&self) -> &ModelInfo {
        self.model.info()
    }

    /// The longest sequence any bucket serves; a generative model, which
    /// keeps no sessions, reports the model's own context length.
    pub fn max_seq(&self) -> u32 {
        self.buckets.iter().map(|b| b.shape.seq).max().unwrap_or(self.info().max_seq)
    }

    /// The widest batch any bucket serves; a generative model reports the
    /// model's own limit.
    pub fn max_batch(&self) -> u32 {
        self.buckets.iter().map(|b| b.shape.batch).max().unwrap_or(self.info().max_batch)
    }

    /// Tokens the longest text needs, counted by the bundle's tokenizer
    /// when there is one; `None` when the count is unknown.
    fn longest(&self, texts: &[&str], extra: &str) -> Result<Option<u32>> {
        let Some(tok) = &self.tokenizer else { return Ok(None) };
        let mut longest = 0;
        for t in texts {
            let n = if extra.is_empty() { tok.count(t, true)? } else { tok.count(&format!("{extra}{t}"), true)? };
            longest = longest.max(n);
        }
        Ok(Some(longest))
    }

    /// Check out a session for `rows` rows of up to `seq` tokens
    /// (`None`: the longest bucket). The caller splits its rows by the
    /// lease's batch.
    async fn lease(&self, rows: usize, seq: Option<u32>, truncating: bool) -> Result<Leased> {
        if self.buckets.is_empty() {
            return Err(ServeError::bad_request(format!(
                "model `{}` serves no sessions (generative model)",
                self.name
            )));
        }
        let max_seq = self.max_seq();
        let need = match seq {
            Some(n) if n > max_seq && !truncating => {
                // A capacity limit, named: the request can shorten the text
                // or ask for truncation (field 2); nothing is cut for it.
                return Err(ServeError::from(
                    TurboError::new(
                        abi::TURBO_E_CAPACITY,
                        format!(
                            "an input needs {n} tokens but the longest session of model `{}` holds {max_seq}; set truncate to right or left to cut it, or shorten it",
                            self.name
                        ),
                    )
                    .with_field(EmbedOptions::FIELD_TRUNCATE),
                ));
            }
            Some(n) => n.min(max_seq),
            None => max_seq,
        };
        let fitting: Vec<&Arc<PoolBucket>> = self.buckets.iter().filter(|b| b.shape.seq >= need).collect();
        let bucket = fitting
            .iter()
            .filter(|b| b.shape.batch as usize >= rows)
            .min_by_key(|b| (b.shape.batch, b.shape.seq))
            .or_else(|| fitting.iter().max_by_key(|b| (b.shape.batch, std::cmp::Reverse(b.shape.seq))))
            .copied()
            .expect("a bucket fits: need <= max_seq");
        let permit =
            bucket.slots.clone().acquire_owned().await.map_err(|_| ServeError::internal("session pool closed"))?;
        let session = bucket
            .idle
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .pop()
            .ok_or_else(|| ServeError::internal("session pool has a permit but no idle session"))?;
        Ok(Leased { session, shape: bucket.shape, bucket: bucket.clone(), _permit: permit })
    }
}

// ---------------------------------------------------------------------------
// Option parsing shared by every surface
// ---------------------------------------------------------------------------

/// `model`, `none`, `right` or `left`; `true` and `false` as the
/// text-embeddings-inference clients spell them.
pub fn parse_truncate(v: Option<&str>) -> Result<Truncate> {
    Ok(match v.map(|s| s.to_ascii_lowercase()).as_deref() {
        None | Some("model") => Truncate::Model,
        Some("none") | Some("false") => Truncate::None,
        Some("right") | Some("true") => Truncate::Right,
        Some("left") => Truncate::Left,
        Some(other) => {
            return Err(ServeError::field(
                EmbedOptions::FIELD_TRUNCATE,
                format!("truncate `{other}` is not model, none, right or left"),
            ))
        }
    })
}

/// `none`, `query` or `document` (`passage` is the same role).
pub fn parse_prompt_role(v: Option<&str>) -> Result<PromptRole> {
    Ok(match v.map(|s| s.to_ascii_lowercase()).as_deref() {
        None | Some("none") => PromptRole::None,
        Some("query") => PromptRole::Query,
        Some("document") | Some("passage") => PromptRole::Document,
        Some(other) => {
            return Err(ServeError::field(
                EmbedOptions::FIELD_PROMPT_ROLE,
                format!("prompt_role `{other}` is not none, query or document"),
            ))
        }
    })
}

/// `model`, `none` or `l2`.
pub fn parse_normalize(v: Option<&str>) -> Result<Normalize> {
    Ok(match v.map(|s| s.to_ascii_lowercase()).as_deref() {
        None | Some("model") => Normalize::Model,
        Some("none") | Some("false") => Normalize::None,
        Some("l2") | Some("true") => Normalize::L2,
        Some(other) => {
            return Err(ServeError::field(
                EmbedOptions::FIELD_NORMALIZE,
                format!("normalize `{other}` is not model, none or l2"),
            ))
        }
    })
}

/// `model`, `mean`, `cls` or `last`.
pub fn parse_pooling(v: Option<&str>) -> Result<Pooling> {
    Ok(match v.map(|s| s.to_ascii_lowercase()).as_deref() {
        None | Some("model") => Pooling::Model,
        Some("mean") => Pooling::Mean,
        Some("cls") => Pooling::Cls,
        Some("last") => Pooling::Last,
        Some(other) => {
            return Err(ServeError::field(
                EmbedOptions::FIELD_POOLING,
                format!("pooling `{other}` is not model, mean, cls or last"),
            ))
        }
    })
}

/// `model`, `none`, `simple`, `first` or `max`.
pub fn parse_aggregation(v: Option<&str>) -> Result<Aggregation> {
    Ok(match v.map(|s| s.to_ascii_lowercase()).as_deref() {
        None | Some("model") => Aggregation::Model,
        Some("none") => Aggregation::None,
        Some("simple") => Aggregation::Simple,
        Some("first") => Aggregation::First,
        Some("max") => Aggregation::Max,
        Some(other) => {
            return Err(ServeError::field(
                ClassifyOptions::FIELD_AGGREGATION,
                format!("aggregation `{other}` is not model, none, simple, first or max"),
            ))
        }
    })
}

// ---------------------------------------------------------------------------
// Tasks
// ---------------------------------------------------------------------------

/// Embedding vectors, one per text, in input order.
#[derive(Debug)]
pub struct Embedded {
    /// One vector per input text, in input order.
    pub vectors: Vec<Vec<f32>>,
    /// Width of each vector.
    pub dim: usize,
    /// Where the provider left the result.
    pub placement: Placement,
    /// Prompt tokens over every text, when a tokenizer counted them.
    pub tokens: Option<u32>,
    /// Wall time of the device work, in milliseconds.
    pub device_ms: f64,
}

/// Embed every text, in chunks of the leased bucket's batch.
pub async fn embed(served: Arc<Served>, texts: Vec<String>, opts: EmbedOptions) -> Result<Embedded> {
    if served.info().kind != ModelKind::Embedding {
        return Err(ServeError::bad_request(format!(
            "model `{}` is {:?}, not an embedding model",
            served.name,
            served.info().kind
        )));
    }
    if texts.is_empty() {
        return Err(ServeError::bad_request("input has no texts"));
    }
    served.model.validate_embed(&opts)?;
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let prefix = match opts.prompt_role {
        PromptRole::Query => served.info().prefix_query.as_str(),
        PromptRole::Document => served.info().prefix_document.as_str(),
        PromptRole::None => "",
    };
    let longest = served.longest(&refs, prefix)?;
    let tokens = served.tokenizer.as_ref().map(|t| refs.iter().map(|s| t.count(s, true).unwrap_or(0)).sum());
    // Only an explicit right or left truncation may cut a text; the
    // model default resolves to a provider convention the caller did not ask for.
    let truncating = matches!(opts.truncate, Truncate::Right | Truncate::Left);
    let lease = served.lease(texts.len(), longest, truncating).await?;
    let batch = lease.shape.batch as usize;
    let dim_out = if opts.output_dim == 0 { served.info().dim as usize } else { opts.output_dim as usize };
    let t0 = Instant::now();
    let (vectors, placement) = tokio::task::spawn_blocking(move || -> Result<(Vec<Vec<f32>>, Placement)> {
        let mut vectors = Vec::with_capacity(texts.len());
        let mut placement = Placement::Host;
        for chunk in texts.chunks(batch) {
            let refs: Vec<&str> = chunk.iter().map(String::as_str).collect();
            lease.session.write_text(&refs, &opts)?;
            let r = lease.session.run(&RunOptions::default())?;
            let out = r.output(0)?;
            placement = out.buffer.desc().placement;
            require_f32(out.buffer.desc().dtype, "embeddings")?;
            let rows = out.shape[0] as usize;
            let width = out.shape[1] as usize;
            if width != dim_out || rows != chunk.len() {
                return Err(ServeError::internal(format!(
                    "provider returned a [{rows}, {width}] result for {} rows of width {dim_out}",
                    chunk.len()
                )));
            }
            let floats = read_f32(&r, 0, rows * width)?;
            for row in floats.chunks(width) {
                vectors.push(row.to_vec());
            }
        }
        Ok((vectors, placement))
    })
    .await
    .map_err(|e| ServeError::internal(format!("embed task failed: {e}")))??;
    Ok(Embedded { vectors, dim: dim_out, placement, tokens, device_ms: t0.elapsed().as_secs_f64() * 1e3 })
}

/// Rerank scores in document order, plus the descending ranking.
pub struct Reranked {
    /// One score per document, in document order.
    pub scores: Vec<f32>,
    /// Document indices, best first, cut to `top_n` when asked.
    pub sorted: Vec<i32>,
    /// Wall time of the device work, in milliseconds.
    pub device_ms: f64,
}

/// Score every document against the query and rank them here, so a
/// request wider than a bucket is still one ranking.
pub async fn rerank(served: Arc<Served>, query: String, docs: Vec<String>, opts: RerankOptions) -> Result<Reranked> {
    if served.info().kind != ModelKind::Reranker {
        return Err(ServeError::bad_request(format!(
            "model `{}` is {:?}, not a reranker",
            served.name,
            served.info().kind
        )));
    }
    if docs.is_empty() {
        return Err(ServeError::bad_request("documents is empty"));
    }
    if opts.top_n as usize > docs.len() {
        return Err(ServeError::field(
            RerankOptions::FIELD_TOP_N,
            format!("top_n {} exceeds the {} documents", opts.top_n, docs.len()),
        ));
    }
    served.model.validate_rerank(&opts)?;
    // A pair is the query plus one document; the longest pair sizes the bucket.
    let longest = match &served.tokenizer {
        Some(tok) => {
            let q = tok.count(&query, true)?;
            let mut longest = 0;
            for d in &docs {
                longest = longest.max(q + tok.count(d, true)?);
            }
            Some(longest)
        }
        None => None,
    };
    // Only an explicit right or left truncation may cut a text; the
    // model default resolves to a provider convention the caller did not ask for.
    let truncating = matches!(opts.truncate, Truncate::Right | Truncate::Left);
    let lease = served.lease(docs.len(), longest, truncating).await?;
    let batch = lease.shape.batch as usize;
    let top_n = opts.top_n;
    let t0 = Instant::now();
    // Every chunk asks for scores only; the ranking over all documents is
    // computed here so a request wider than a bucket is still one ranking.
    let chunk_opts = RerankOptions { top_n: 0, return_sorted: false, ..opts };
    let scores = tokio::task::spawn_blocking(move || -> Result<Vec<f32>> {
        let mut scores = Vec::with_capacity(docs.len());
        for chunk in docs.chunks(batch) {
            let refs: Vec<&str> = chunk.iter().map(String::as_str).collect();
            lease.session.write_pairs(&query, &refs, &chunk_opts)?;
            let r = lease.session.run(&RunOptions::default())?;
            let out = r.output(0)?;
            require_f32(out.buffer.desc().dtype, "scores")?;
            let n = out.shape[0] as usize;
            if n != chunk.len() {
                return Err(ServeError::internal(format!(
                    "provider returned {n} scores for {} documents",
                    chunk.len()
                )));
            }
            scores.extend(read_f32(&r, 0, n)?);
        }
        Ok(scores)
    })
    .await
    .map_err(|e| ServeError::internal(format!("rerank task failed: {e}")))??;
    let mut sorted: Vec<i32> = (0..scores.len() as i32).collect();
    sorted.sort_by(|&a, &b| scores[b as usize].total_cmp(&scores[a as usize]));
    if top_n != 0 {
        sorted.truncate(top_n as usize);
    }
    Ok(Reranked { scores, sorted, device_ms: t0.elapsed().as_secs_f64() * 1e3 })
}

/// Classification scores: `[texts, labels]`.
pub struct Classified {
    /// One row of label scores per input text.
    pub scores: Vec<Vec<f32>>,
    /// The bundle's labels, in score order.
    pub labels: Vec<String>,
    /// Wall time of the device work, in milliseconds.
    pub device_ms: f64,
}

/// Classify every text into the bundle's labels.
pub async fn classify(served: Arc<Served>, texts: Vec<String>, opts: ClassifyOptions) -> Result<Classified> {
    if served.info().kind != ModelKind::Classifier {
        return Err(ServeError::bad_request(format!(
            "model `{}` is {:?}, not a classifier",
            served.name,
            served.info().kind
        )));
    }
    if texts.is_empty() {
        return Err(ServeError::bad_request("input has no texts"));
    }
    served.model.validate_classify(&opts)?;
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let longest = served.longest(&refs, "")?;
    // Only an explicit right or left truncation may cut a text; the
    // model default resolves to a provider convention the caller did not ask for.
    let truncating = matches!(opts.truncate, Truncate::Right | Truncate::Left);
    let lease = served.lease(texts.len(), longest, truncating).await?;
    let batch = lease.shape.batch as usize;
    let labels = served.info().labels.clone();
    let width = labels.len();
    let t0 = Instant::now();
    let scores = tokio::task::spawn_blocking(move || -> Result<Vec<Vec<f32>>> {
        let mut scores = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(batch) {
            let refs: Vec<&str> = chunk.iter().map(String::as_str).collect();
            lease.session.write_text_classify(&refs, &opts)?;
            let r = lease.session.run(&RunOptions::default())?;
            let out = r.output(0)?;
            require_f32(out.buffer.desc().dtype, "scores")?;
            let rows = out.shape[0] as usize;
            let w = out.shape[1] as usize;
            if rows != chunk.len() || w != width {
                return Err(ServeError::internal(format!(
                    "provider returned a [{rows}, {w}] result for {} texts and {width} labels",
                    chunk.len()
                )));
            }
            for row in read_f32(&r, 0, rows * w)?.chunks(w) {
                scores.push(row.to_vec());
            }
        }
        Ok(scores)
    })
    .await
    .map_err(|e| ServeError::internal(format!("classify task failed: {e}")))??;
    Ok(Classified { scores, labels, device_ms: t0.elapsed().as_secs_f64() * 1e3 })
}

/// Entity spans per text.
pub struct Tagged {
    /// The spans the model found, per input text.
    pub spans: Vec<Vec<Span>>,
    /// The bundle's labels; a span's `label` indexes this.
    pub labels: Vec<String>,
    /// Wall time of the device work, in milliseconds.
    pub device_ms: f64,
}

/// Tag every text, returning the spans per row.
pub async fn token_classify(served: Arc<Served>, texts: Vec<String>, opts: ClassifyOptions) -> Result<Tagged> {
    if served.info().kind != ModelKind::TokenClassifier {
        return Err(ServeError::bad_request(format!(
            "model `{}` is {:?}, not a token classifier",
            served.name,
            served.info().kind
        )));
    }
    if texts.is_empty() {
        return Err(ServeError::bad_request("input has no texts"));
    }
    served.model.validate_classify(&opts)?;
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let longest = served.longest(&refs, "")?;
    // Only an explicit right or left truncation may cut a text; the
    // model default resolves to a provider convention the caller did not ask for.
    let truncating = matches!(opts.truncate, Truncate::Right | Truncate::Left);
    let lease = served.lease(texts.len(), longest, truncating).await?;
    let batch = lease.shape.batch as usize;
    let labels = served.info().labels.clone();
    let t0 = Instant::now();
    let spans = tokio::task::spawn_blocking(move || -> Result<Vec<Vec<Span>>> {
        let mut all = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(batch) {
            let refs: Vec<&str> = chunk.iter().map(String::as_str).collect();
            lease.session.write_text_classify(&refs, &opts)?;
            let r = lease.session.run(&RunOptions::default())?;
            let mut rows: Vec<Vec<Span>> = vec![Vec::new(); chunk.len()];
            for s in r.spans() {
                let row = s.row as usize;
                if row >= rows.len() {
                    return Err(ServeError::internal(format!(
                        "provider reported a span on row {row} of {}",
                        rows.len()
                    )));
                }
                rows[row].push(*s);
            }
            all.extend(rows);
        }
        Ok(all)
    })
    .await
    .map_err(|e| ServeError::internal(format!("token-classify task failed: {e}")))??;
    Ok(Tagged { spans, labels, device_ms: t0.elapsed().as_secs_f64() * 1e3 })
}

/// One streamed piece of a generation.
#[derive(Debug, Clone)]
pub struct Piece {
    /// The text this step produced.
    pub text: String,
    /// Set on the last piece of the generation.
    pub done: bool,
    /// Why the generation stopped; meaningful once `done`.
    pub finish_reason: FinishReason,
    /// Tokens the prompt took.
    pub prompt_tokens: u32,
    /// Tokens generated so far.
    pub generated_tokens: u32,
}

/// Start a generation; pieces arrive on the receiver as the model produces
/// them and a dropped receiver cancels the generation on the device. The
/// permit bounds concurrent generations per model.
pub async fn generate(
    served: Arc<Served>,
    messages: Vec<(String, String)>,
    desc: GenerateDesc,
) -> Result<mpsc::Receiver<Result<Piece>>> {
    if served.info().kind != ModelKind::Generative {
        return Err(ServeError::bad_request(format!(
            "model `{}` is {:?}, not a generative model",
            served.name,
            served.info().kind
        )));
    }
    if messages.is_empty() {
        return Err(ServeError::bad_request("messages is empty"));
    }
    let permit =
        served.generations.clone().acquire_owned().await.map_err(|_| ServeError::internal("generation pool closed"))?;
    let (tx, rx) = mpsc::channel::<Result<Piece>>(64);
    let model = served.model.clone();
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let run = || -> Result<()> {
            let g = model.create_generation(&desc)?;
            let msgs: Vec<Message<'_>> = messages
                .iter()
                .map(|(role, content)| Message { role: role.as_str(), content: content.as_str() })
                .collect();
            g.prompt(&msgs)?;
            loop {
                let piece = {
                    let c = g.step()?;
                    Piece {
                        text: c.text.clone(),
                        done: c.done,
                        finish_reason: c.finish_reason,
                        prompt_tokens: c.prompt_tokens,
                        generated_tokens: c.generated_tokens,
                    }
                };
                let done = piece.done;
                if tx.blocking_send(Ok(piece)).is_err() {
                    // The client went away: stop the model, do not run on.
                    g.cancel()?;
                    return Ok(());
                }
                if done {
                    return Ok(());
                }
            }
        };
        if let Err(e) = run() {
            let _ = tx.blocking_send(Err(e));
        }
    });
    Ok(rx)
}

/// A generic RUN model: bound inputs by name, every output read back.
pub struct RunOutput {
    /// The model's name for this output.
    pub name: String,
    /// Element type of the bytes.
    pub dtype: DType,
    /// Extents, outermost first.
    pub shape: Vec<u64>,
    /// Packed little-endian host bytes.
    pub bytes: Vec<u8>,
}

/// One input to bind: packed host bytes of `dtype` with `shape`.
pub struct RunInput {
    /// The model's name for this input.
    pub name: String,
    /// Element type of the bytes.
    pub dtype: DType,
    /// Extents, outermost first.
    pub shape: Vec<u64>,
    /// Packed little-endian host bytes.
    pub bytes: Vec<u8>,
}

/// Bind every input by name, run once, and read every output back.
pub async fn run_generic(
    served: Arc<Served>,
    inputs: Vec<RunInput>,
    params: Vec<(String, String)>,
) -> Result<(Vec<RunOutput>, f64)> {
    if served.info().kind != ModelKind::Generic {
        return Err(ServeError::bad_request(format!(
            "model `{}` is {:?}, not a generic RUN model",
            served.name,
            served.info().kind
        )));
    }
    let lease = served.lease(1, None, true).await?;
    let ctx = served.model.context().clone();
    let t0 = Instant::now();
    let outputs = tokio::task::spawn_blocking(move || -> Result<Vec<RunOutput>> {
        let mut bound = Vec::new();
        for input in &inputs {
            let desc = turbo::buffer::BufferDesc::packed(Placement::Host, input.dtype, &input.shape)?;
            if desc.bytes != input.bytes.len() as u64 {
                return Err(ServeError::bad_request(format!(
                    "input `{}` has {} bytes of data but its shape and datatype need {}",
                    input.name,
                    input.bytes.len(),
                    desc.bytes
                )));
            }
            let buf = ctx.alloc(&desc)?;
            let ptr = buf.host_ptr().ok_or_else(|| ServeError::internal("a host buffer has no host pointer"))?;
            // SAFETY: the buffer is HOST placement of exactly `desc.bytes`
            // bytes, freshly allocated and not yet shared with the provider.
            unsafe { std::ptr::copy_nonoverlapping(input.bytes.as_ptr(), ptr.as_ptr(), input.bytes.len()) };
            lease.session.bind(&input.name, &buf)?;
            bound.push(buf);
        }
        let r = lease.session.run(&RunOptions { params: Options(params) })?;
        let mut outs = Vec::new();
        for (i, o) in r.outputs().iter().enumerate() {
            let desc = o.buffer.desc().clone();
            let mut bytes = vec![0u8; desc.bytes as usize];
            r.read(i as u32, &mut bytes)?;
            outs.push(RunOutput { name: o.name.to_string(), dtype: desc.dtype, shape: o.shape.clone(), bytes });
        }
        Ok(outs)
    })
    .await
    .map_err(|e| ServeError::internal(format!("run task failed: {e}")))??;
    Ok((outputs, t0.elapsed().as_secs_f64() * 1e3))
}

// ---------------------------------------------------------------------------
// Result helpers
// ---------------------------------------------------------------------------

fn require_f32(dtype: DType, what: &str) -> Result<()> {
    if dtype != DType::F32 {
        return Err(ServeError::from(TurboError::new(
            abi::TURBO_E_UNSUPPORTED_DTYPE,
            format!("the server reads f32 {what}; the provider returned {dtype:?}"),
        )));
    }
    Ok(())
}

fn read_f32(r: &Arc<ResultHandle>, index: u32, count: usize) -> Result<Vec<f32>> {
    let mut bytes = vec![0u8; count * 4];
    let n = r.read(index, &mut bytes)?;
    if n != bytes.len() {
        return Err(ServeError::internal(format!("result read gave {n} bytes for {} expected", bytes.len())));
    }
    Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}
