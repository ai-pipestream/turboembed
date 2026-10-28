//! Prometheus metrics collection, Tower gRPC instrumentation middleware, and
//! HTTP `/metrics` exposition endpoint (issue `#1300`).

mod http;

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;

use tokio_stream::Stream;
use tonic::Code;

use crate::state::State;

pub(crate) use self::http::{MetricsLayer, start_http_listener};

const LATENCY_BUCKETS: [f64; 14] =
    [0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0];

const BATCH_BUCKETS: [u64; 9] = [1, 2, 4, 8, 16, 32, 64, 128, 256];

#[derive(Default)]
struct LatencyHist {
    buckets: [u64; LATENCY_BUCKETS.len()],
    sum: f64,
    count: u64,
}

impl LatencyHist {
    fn observe(&mut self, secs: f64) {
        for (b, &bound) in self.buckets.iter_mut().zip(&LATENCY_BUCKETS) {
            if secs <= bound {
                *b += 1;
            }
        }
        self.sum += secs;
        self.count += 1;
    }
}

#[derive(Default)]
struct BatchHist {
    buckets: [u64; BATCH_BUCKETS.len()],
    sum: u64,
    count: u64,
}

impl BatchHist {
    fn observe(&mut self, n: usize) {
        let v = n as u64;
        for (b, &bound) in self.buckets.iter_mut().zip(&BATCH_BUCKETS) {
            if v <= bound {
                *b += 1;
            }
        }
        self.sum += v;
        self.count += 1;
    }
}

/// Server-wide Prometheus metrics registry.
#[derive(Default)]
pub struct Metrics {
    rpc_counts: Mutex<BTreeMap<(String, String, &'static str), u64>>,
    rpc_durations: Mutex<BTreeMap<(String, String), LatencyHist>>,
    inflight_streams: AtomicUsize,
    inflight_documents: AtomicUsize,
    batch_sizes: Mutex<BatchHist>,
    device_queued: Mutex<BTreeMap<u32, usize>>,
    bundle_loads_ok: AtomicU64,
    bundle_loads_err: AtomicU64,
    bundle_unloads: AtomicU64,
}

impl Metrics {
    /// Creates an empty [`Metrics`] registry.
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Records completion of one gRPC call.
    pub fn record_rpc(&self, service: &str, method: &str, code: Code, elapsed: Duration) {
        let code_str = http::code_name(code);
        {
            let mut counts = self.rpc_counts.lock().unwrap_or_else(|e| e.into_inner());
            *counts.entry((service.to_owned(), method.to_owned(), code_str)).or_insert(0) += 1;
        }
        {
            let mut durs = self.rpc_durations.lock().unwrap_or_else(|e| e.into_inner());
            durs.entry((service.to_owned(), method.to_owned())).or_default().observe(elapsed.as_secs_f64());
        }
    }

    /// Records the size of one executed inference batch.
    pub fn record_batch_size(&self, batch_size: usize) {
        if batch_size == 0 {
            return;
        }
        self.batch_sizes.lock().unwrap_or_else(|e| e.into_inner()).observe(batch_size);
    }

    /// Records a bundle load attempt (`ok = true` on success, `false` on failure).
    pub fn record_bundle_load(&self, ok: bool) {
        if ok {
            self.bundle_loads_ok.fetch_add(1, Ordering::Relaxed);
        } else {
            self.bundle_loads_err.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Records a bundle unload (`UnloadBundle`).
    pub fn record_bundle_unload(&self) {
        self.bundle_unloads.fetch_add(1, Ordering::Relaxed);
    }

    /// Increments the in-flight stream gauge until the returned guard is dropped.
    pub fn track_stream(self: &Arc<Self>) -> StreamGuard {
        self.inflight_streams.fetch_add(1, Ordering::Relaxed);
        StreamGuard { metrics: Arc::clone(self) }
    }

    /// Increments the in-flight document gauge (and optional device queue depth)
    /// until the returned guard is dropped.
    pub fn track_docs(self: &Arc<Self>, count: usize, device: Option<u32>) -> DocGuard {
        self.inflight_documents.fetch_add(count, Ordering::Relaxed);
        if let Some(d) = device {
            self.add_device_queue(d, count);
        }
        DocGuard { metrics: Arc::clone(self), count, device }
    }

    pub(crate) fn add_device_queue(&self, device: u32, count: usize) {
        let mut map = self.device_queued.lock().unwrap_or_else(|e| e.into_inner());
        *map.entry(device).or_insert(0) += count;
    }

    pub(crate) fn sub_device_queue(&self, device: u32, count: usize) {
        let mut map = self.device_queued.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(v) = map.get_mut(&device) {
            *v = v.saturating_sub(count);
        }
    }

    /// Renders all metrics in Prometheus text exposition format (`0.0.4`).
    pub(crate) fn render(&self, state: &State) -> String {
        let mut out = String::with_capacity(4096);

        let _ = writeln!(
            out,
            "# HELP turbo_kserve_rpc_requests_total Total gRPC requests completed by service, method, and status code."
        );
        let _ = writeln!(out, "# TYPE turbo_kserve_rpc_requests_total counter");
        {
            let counts = self.rpc_counts.lock().unwrap_or_else(|e| e.into_inner());
            for ((svc, method, code), count) in counts.iter() {
                let _ = writeln!(
                    out,
                    "turbo_kserve_rpc_requests_total{{service=\"{svc}\",method=\"{method}\",code=\"{code}\"}} {count}"
                );
            }
        }

        let _ = writeln!(
            out,
            "# HELP turbo_kserve_rpc_duration_seconds gRPC request latency in seconds by service and method."
        );
        let _ = writeln!(out, "# TYPE turbo_kserve_rpc_duration_seconds histogram");
        {
            let durs = self.rpc_durations.lock().unwrap_or_else(|e| e.into_inner());
            for ((svc, method), h) in durs.iter() {
                for (b, &bound) in h.buckets.iter().zip(&LATENCY_BUCKETS) {
                    let _ = writeln!(
                        out,
                        "turbo_kserve_rpc_duration_seconds_bucket{{service=\"{svc}\",method=\"{method}\",le=\"{bound}\"}} {b}"
                    );
                }
                let _ = writeln!(
                    out,
                    "turbo_kserve_rpc_duration_seconds_bucket{{service=\"{svc}\",method=\"{method}\",le=\"+Inf\"}} {}",
                    h.count
                );
                let _ = writeln!(
                    out,
                    "turbo_kserve_rpc_duration_seconds_sum{{service=\"{svc}\",method=\"{method}\"}} {:.6}",
                    h.sum
                );
                let _ = writeln!(
                    out,
                    "turbo_kserve_rpc_duration_seconds_count{{service=\"{svc}\",method=\"{method}\"}} {}",
                    h.count
                );
            }
        }

        let streams = self.inflight_streams.load(Ordering::Relaxed);
        let docs = self.inflight_documents.load(Ordering::Relaxed);
        let stream_stats = state.stream.stats();
        let _ = writeln!(out, "# HELP turbo_kserve_inflight_streams Number of active streaming RPCs.");
        let _ = writeln!(out, "# TYPE turbo_kserve_inflight_streams gauge");
        let _ = writeln!(out, "turbo_kserve_inflight_streams {streams}");

        let _ = writeln!(out, "# HELP turbo_kserve_inflight_documents Number of documents currently in flight.");
        let _ = writeln!(out, "# TYPE turbo_kserve_inflight_documents gauge");
        let _ = writeln!(out, "turbo_kserve_inflight_documents {docs}");

        let _ = writeln!(
            out,
            "# HELP turbo_kserve_queued_documents Number of stream documents currently admitted and awaiting completion."
        );
        let _ = writeln!(out, "# TYPE turbo_kserve_queued_documents gauge");
        let _ = writeln!(out, "turbo_kserve_queued_documents {}", stream_stats.queued_items);

        let _ = writeln!(
            out,
            "# HELP turbo_kserve_queued_bytes Bytes of stream documents currently admitted and awaiting completion."
        );
        let _ = writeln!(out, "# TYPE turbo_kserve_queued_bytes gauge");
        let _ = writeln!(out, "turbo_kserve_queued_bytes {}", stream_stats.queued_bytes);

        let _ = writeln!(out, "# HELP turbo_kserve_batch_size Inference batch size distribution.");
        let _ = writeln!(out, "# TYPE turbo_kserve_batch_size histogram");
        {
            let b = self.batch_sizes.lock().unwrap_or_else(|e| e.into_inner());
            for (cnt, &bound) in b.buckets.iter().zip(&BATCH_BUCKETS) {
                let _ = writeln!(out, "turbo_kserve_batch_size_bucket{{le=\"{bound}\"}} {cnt}");
            }
            let _ = writeln!(out, "turbo_kserve_batch_size_bucket{{le=\"+Inf\"}} {}", b.count);
            let _ = writeln!(out, "turbo_kserve_batch_size_sum {}", b.sum);
            let _ = writeln!(out, "turbo_kserve_batch_size_count {}", b.count);
        }

        let mut device_stats: BTreeMap<u32, (usize, usize, usize)> = BTreeMap::new();
        {
            let q = self.device_queued.lock().unwrap_or_else(|e| e.into_inner());
            for (&dev, &depth) in q.iter() {
                device_stats.entry(dev).or_insert((0, 0, 0)).0 += depth;
            }
        }
        let mut loaded_count = 0usize;
        for served in state.listing() {
            if let Some(loaded) = served.loaded.get() {
                loaded_count += 1;
                let total = loaded.slots.len();
                let idle = loaded.idle.lock().unwrap_or_else(|e| e.into_inner()).len();
                let active = total.saturating_sub(idle);
                let entry = device_stats.entry(loaded.device).or_insert((0, 0, 0));
                entry.1 += active;
                entry.2 += idle;
            }
        }
        if device_stats.is_empty() {
            device_stats.insert(0, (0, 0, 0));
        }

        let _ = writeln!(
            out,
            "# HELP turbo_kserve_device_queue_depth Number of documents queued or executing per runtime device."
        );
        let _ = writeln!(out, "# TYPE turbo_kserve_device_queue_depth gauge");
        for (&dev, &(depth, _, _)) in &device_stats {
            let _ = writeln!(out, "turbo_kserve_device_queue_depth{{device=\"{dev}\"}} {depth}");
        }

        let _ = writeln!(
            out,
            "# HELP turbo_kserve_device_active_sessions Number of held sessions currently executing per runtime device."
        );
        let _ = writeln!(out, "# TYPE turbo_kserve_device_active_sessions gauge");
        for (&dev, &(_, active, _)) in &device_stats {
            let _ = writeln!(out, "turbo_kserve_device_active_sessions{{device=\"{dev}\"}} {active}");
        }

        let _ = writeln!(
            out,
            "# HELP turbo_kserve_device_idle_sessions Number of idle sessions available per runtime device."
        );
        let _ = writeln!(out, "# TYPE turbo_kserve_device_idle_sessions gauge");
        for (&dev, &(_, _, idle)) in &device_stats {
            let _ = writeln!(out, "turbo_kserve_device_idle_sessions{{device=\"{dev}\"}} {idle}");
        }

        let loads_ok = self.bundle_loads_ok.load(Ordering::Relaxed);
        let loads_err = self.bundle_loads_err.load(Ordering::Relaxed);
        let unloads = self.bundle_unloads.load(Ordering::Relaxed);
        let _ = writeln!(out, "# HELP turbo_kserve_bundle_loads_total Total bundle load attempts by status.");
        let _ = writeln!(out, "# TYPE turbo_kserve_bundle_loads_total counter");
        let _ = writeln!(out, "turbo_kserve_bundle_loads_total{{status=\"ok\"}} {loads_ok}");
        let _ = writeln!(out, "turbo_kserve_bundle_loads_total{{status=\"error\"}} {loads_err}");

        let _ = writeln!(out, "# HELP turbo_kserve_bundle_unloads_total Total bundles unloaded.");
        let _ = writeln!(out, "# TYPE turbo_kserve_bundle_unloads_total counter");
        let _ = writeln!(out, "turbo_kserve_bundle_unloads_total {unloads}");

        let _ = writeln!(out, "# HELP turbo_kserve_loaded_bundles Number of currently loaded model bundles.");
        let _ = writeln!(out, "# TYPE turbo_kserve_loaded_bundles gauge");
        let _ = writeln!(out, "turbo_kserve_loaded_bundles {loaded_count}");

        out
    }
}

/// RAII guard tracking one active server/bidi stream.
pub struct StreamGuard {
    metrics: Arc<Metrics>,
}

impl Drop for StreamGuard {
    fn drop(&mut self) {
        self.metrics.inflight_streams.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Stream wrapper holding a [`StreamGuard`] for the stream's lifetime.
pub struct TrackedStream<S> {
    inner: Pin<Box<S>>,
    _guard: StreamGuard,
}

impl<S> TrackedStream<S> {
    pub fn new(inner: S, guard: StreamGuard) -> Self {
        Self { inner: Box::pin(inner), _guard: guard }
    }
}

impl<S: Stream> Stream for TrackedStream<S> {
    type Item = S::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

/// RAII guard tracking in-flight documents and optional device queue depth.
pub struct DocGuard {
    metrics: Arc<Metrics>,
    count: usize,
    device: Option<u32>,
}

impl DocGuard {
    /// Binds this document guard to `device` once the target model is resolved.
    pub fn bind_device(&mut self, device: u32) {
        if self.device.is_none() {
            self.metrics.add_device_queue(device, self.count);
            self.device = Some(device);
        }
    }
}

impl Drop for DocGuard {
    fn drop(&mut self) {
        self.metrics.inflight_documents.fetch_sub(self.count, Ordering::Relaxed);
        if let Some(d) = self.device {
            self.metrics.sub_device_queue(d, self.count);
        }
    }
}
