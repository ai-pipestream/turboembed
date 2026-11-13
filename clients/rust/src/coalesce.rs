//! Runtime-scoped micro-batch coalescer for concurrent remote query embeddings (`#2338`).
//!
//! Collects concurrent single-query [`EmbedCoalescer::embed_query`] calls on
//! the same Tokio runtime within a bounded `linger` window (up to `max_batch`)
//! into one batched `ModelInfer` RPC, slicing each caller's vector back out.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::{mpsc, oneshot};

use crate::error::{ClientError, Result};
use crate::remote::{RemoteClient, RemotePriority};

struct CoalesceItem {
    text: String,
    timeout: Option<Duration>,
    reply: oneshot::Sender<Result<Vec<f32>>>,
}

/// Configuration for [`EmbedCoalescer`].
#[derive(Copy, Clone, Debug)]
pub struct CoalesceConfig {
    /// Maximum queries grouped into a single `ModelInfer` call (`>= 1`).
    pub max_batch: usize,
    /// Maximum wait after the first queued item before flushing a partial batch.
    pub linger: Duration,
    /// Bounded channel capacity per runtime worker (`>= max_batch`).
    pub queue_capacity: usize,
}

impl Default for CoalesceConfig {
    fn default() -> Self {
        Self { max_batch: 16, linger: Duration::from_millis(2), queue_capacity: 256 }
    }
}

/// Runtime-scoped concurrent query micro-batcher over [`RemoteClient`].
#[derive(Clone)]
pub struct EmbedCoalescer {
    remote: RemoteClient,
    model_name: String,
    role: Option<String>,
    config: CoalesceConfig,
    workers: Arc<Mutex<HashMap<tokio::runtime::Id, mpsc::Sender<CoalesceItem>>>>,
    batches_dispatched: Arc<AtomicU64>,
}

impl EmbedCoalescer {
    /// Creates a coalescer targeting `model_name` on `remote`.
    pub fn new(
        remote: RemoteClient,
        model_name: impl Into<String>,
        role: Option<String>,
        config: CoalesceConfig,
    ) -> Self {
        let max_batch = config.max_batch.max(1);
        let queue_capacity = config.queue_capacity.max(max_batch);
        Self {
            remote,
            model_name: model_name.into(),
            role,
            config: CoalesceConfig { max_batch, linger: config.linger, queue_capacity },
            workers: Arc::new(Mutex::new(HashMap::new())),
            batches_dispatched: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Total `ModelInfer` RPC batches flushed across all runtimes (for metrics/tests).
    pub fn batches_dispatched(&self) -> u64 {
        self.batches_dispatched.load(Ordering::Relaxed)
    }

    /// Submits a single query `text` to the current Tokio runtime's coalescing
    /// worker and awaits its `dim`-wide embedding row.
    pub async fn embed_query(&self, text: &str, timeout: Option<Duration>) -> Result<Vec<f32>> {
        let handle = tokio::runtime::Handle::try_current()
            .map_err(|_| ClientError::FailedPrecondition("EmbedCoalescer requires an active Tokio runtime".into()))?;
        let tx = self.worker_for(&handle)?;
        let (reply_tx, reply_rx) = oneshot::channel();
        tx.send(CoalesceItem { text: text.to_owned(), timeout, reply: reply_tx })
            .await
            .map_err(|_| ClientError::Unavailable("coalescer worker closed".into()))?;
        reply_rx.await.map_err(|_| ClientError::Unavailable("coalescer worker dropped reply".into()))?
    }

    fn worker_for(&self, handle: &tokio::runtime::Handle) -> Result<mpsc::Sender<CoalesceItem>> {
        let rt_id = handle.id();
        let mut guard = self.workers.lock().map_err(|_| ClientError::Internal("coalescer map poisoned".into()))?;
        if let Some(tx) = guard.get(&rt_id)
            && !tx.is_closed()
        {
            return Ok(tx.clone());
        }
        let (tx, rx) = mpsc::channel(self.config.queue_capacity);
        let remote = self.remote.clone();
        let model_name = self.model_name.clone();
        let role = self.role.clone();
        let cfg = self.config;
        let counter = Arc::clone(&self.batches_dispatched);
        let map_ref = Arc::clone(&self.workers);
        handle.spawn(async move {
            struct Cleanup(tokio::runtime::Id, Arc<Mutex<HashMap<tokio::runtime::Id, mpsc::Sender<CoalesceItem>>>>);
            impl Drop for Cleanup {
                fn drop(&mut self) {
                    if let Ok(mut g) = self.1.lock() {
                        g.remove(&self.0);
                    }
                }
            }
            let _cleanup = Cleanup(rt_id, map_ref);
            run_worker(rx, remote, model_name, role, cfg, counter).await;
        });
        guard.insert(rt_id, tx.clone());
        Ok(tx)
    }
}

async fn run_worker(
    mut rx: mpsc::Receiver<CoalesceItem>,
    remote: RemoteClient,
    model_name: String,
    role: Option<String>,
    cfg: CoalesceConfig,
    counter: Arc<AtomicU64>,
) {
    let mut batch: Vec<CoalesceItem> = Vec::with_capacity(cfg.max_batch);
    while let Some(first) = rx.recv().await {
        batch.clear();
        batch.push(first);
        if cfg.max_batch > 1 {
            let deadline = tokio::time::Instant::now() + cfg.linger;
            while batch.len() < cfg.max_batch {
                match rx.try_recv() {
                    Ok(item) => batch.push(item),
                    Err(mpsc::error::TryRecvError::Empty) => {
                        if cfg.linger.is_zero() {
                            break;
                        }
                        match tokio::time::timeout_at(deadline, rx.recv()).await {
                            Ok(Some(item)) => batch.push(item),
                            _ => break,
                        }
                    }
                    Err(mpsc::error::TryRecvError::Disconnected) => break,
                }
            }
        }
        counter.fetch_add(1, Ordering::Relaxed);
        let min_timeout = batch.iter().filter_map(|it| it.timeout).min();
        let refs: Vec<&str> = batch.iter().map(|it| it.text.as_str()).collect();
        match remote
            .embed_texts(&model_name, &refs, role.as_deref(), Some(RemotePriority::Interactive), min_timeout)
            .await
        {
            Ok((flat, dim)) if dim > 0 && flat.len() == batch.len() * dim => {
                for (i, item) in batch.drain(..).enumerate() {
                    let _ = item.reply.send(Ok(flat[i * dim..(i + 1) * dim].to_vec()));
                }
            }
            Ok((flat, dim)) => {
                let err = ClientError::Internal(format!(
                    "coalesced batch shape mismatch: expected {}x{dim}, got {} floats",
                    batch.len(),
                    flat.len()
                ));
                for item in batch.drain(..) {
                    let _ = item.reply.send(Err(err.clone()));
                }
            }
            Err(err) => {
                for item in batch.drain(..) {
                    let _ = item.reply.send(Err(err.clone()));
                }
            }
        }
    }
}
