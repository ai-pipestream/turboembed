//! Bounded work-stealing worker pool with critical-path priority ordering
//! for zero-barrier DAG execution.

use std::cmp::Ordering;
use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::{self, JoinHandle};

/// Priority key ordering ready DAG steps on the critical path.
///
/// Higher `critical_path_cost`, higher `transitive_dependents`, and higher
/// `cost_hint` win first; ties break by lower `topo_rank` and lower `seq`
/// so execution order is deterministic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PriorityKey {
    pub(crate) critical_path_cost: u64,
    pub(crate) transitive_dependents: usize,
    pub(crate) cost_hint: u32,
    pub(crate) topo_rank: usize,
    pub(crate) seq: u64,
}

impl Ord for PriorityKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.critical_path_cost
            .cmp(&other.critical_path_cost)
            .then_with(|| self.transitive_dependents.cmp(&other.transitive_dependents))
            .then_with(|| self.cost_hint.cmp(&other.cost_hint))
            .then_with(|| other.topo_rank.cmp(&self.topo_rank))
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

impl PartialOrd for PriorityKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct PrioritizedJob {
    priority: PriorityKey,
    job: Box<dyn FnOnce() + Send + 'static>,
}

struct PoolQueues {
    locals: Vec<VecDeque<PrioritizedJob>>,
    shutdown: bool,
}

struct PoolState {
    threads: usize,
    queues: Mutex<PoolQueues>,
    cv: Condvar,
    next_seq: AtomicU64,
    steals: AtomicUsize,
}

struct PoolHandle {
    state: Arc<PoolState>,
    workers: Mutex<Vec<JoinHandle<()>>>,
}

impl Drop for PoolHandle {
    fn drop(&mut self) {
        {
            let mut q = self.state.queues.lock().unwrap_or_else(|e| e.into_inner());
            q.shutdown = true;
        }
        self.state.cv.notify_all();
        let handles: Vec<JoinHandle<()>> = std::mem::take(&mut *self.workers.lock().unwrap_or_else(|e| e.into_inner()));
        let current = thread::current().id();
        for handle in handles {
            if handle.thread().id() != current {
                let _ = handle.join();
            }
        }
    }
}

/// A bounded pool of persistent worker threads that execute ready pipeline
/// steps in critical-path priority order with work stealing across per-worker
/// deques.
#[derive(Clone)]
pub struct WorkerPool {
    inner: Arc<PoolHandle>,
}

impl std::fmt::Debug for WorkerPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerPool")
            .field("threads", &self.thread_count())
            .field("steals", &self.steal_count())
            .finish()
    }
}

impl WorkerPool {
    /// Creates a bounded worker pool with `threads.max(1)` persistent worker
    /// threads.
    pub fn new(threads: usize) -> WorkerPool {
        let threads = threads.max(1);
        let locals = (0..threads).map(|_| VecDeque::new()).collect();
        let state = Arc::new(PoolState {
            threads,
            queues: Mutex::new(PoolQueues { locals, shutdown: false }),
            cv: Condvar::new(),
            next_seq: AtomicU64::new(0),
            steals: AtomicUsize::new(0),
        });
        let mut workers = Vec::with_capacity(threads);
        for worker_idx in 0..threads {
            let st = Arc::clone(&state);
            let handle = thread::Builder::new()
                .name(format!("turbo-dag-worker-{worker_idx}"))
                .spawn(move || worker_loop(worker_idx, st))
                .expect("failed to spawn turbo-dag-worker thread");
            workers.push(handle);
        }
        WorkerPool { inner: Arc::new(PoolHandle { state, workers: Mutex::new(workers) }) }
    }

    /// Returns a process-wide shared bounded worker pool sized to available
    /// hardware parallelism (clamped to `1..=8` threads).
    pub fn shared() -> &'static WorkerPool {
        static SHARED: OnceLock<WorkerPool> = OnceLock::new();
        SHARED.get_or_init(|| {
            let n = thread::available_parallelism().map_or(4, |p| p.get()).clamp(1, 8);
            WorkerPool::new(n)
        })
    }

    /// Number of persistent worker threads in this pool.
    pub fn thread_count(&self) -> usize {
        self.inner.state.threads
    }

    /// Total number of cross-worker deque steals performed by this pool.
    pub fn steal_count(&self) -> usize {
        self.inner.state.steals.load(AtomicOrdering::Relaxed)
    }

    pub(crate) fn next_seq(&self) -> u64 {
        self.inner.state.next_seq.fetch_add(1, AtomicOrdering::Relaxed)
    }

    pub(crate) fn spawn(&self, worker_hint: usize, priority: PriorityKey, job: impl FnOnce() + Send + 'static) {
        let st = &self.inner.state;
        let target = worker_hint % st.threads;
        {
            let mut q = st.queues.lock().unwrap_or_else(|e| e.into_inner());
            let deque = &mut q.locals[target];
            let pos = deque.iter().position(|existing| priority > existing.priority).unwrap_or(deque.len());
            deque.insert(pos, PrioritizedJob { priority, job: Box::new(job) });
        }
        st.cv.notify_one();
    }
}

fn worker_loop(worker_idx: usize, state: Arc<PoolState>) {
    loop {
        let next = {
            let mut q = state.queues.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if let Some((source_idx, job)) = pop_highest_priority(&mut q.locals, worker_idx) {
                    if source_idx != worker_idx {
                        state.steals.fetch_add(1, AtomicOrdering::Relaxed);
                    }
                    break Some(job);
                }
                if q.shutdown {
                    break None;
                }
                q = state.cv.wait(q).unwrap_or_else(|e| e.into_inner());
            }
        };
        match next {
            Some(job) => {
                let _ = catch_unwind(AssertUnwindSafe(job.job));
            }
            None => break,
        }
    }
}

fn pop_highest_priority(locals: &mut [VecDeque<PrioritizedJob>], worker_idx: usize) -> Option<(usize, PrioritizedJob)> {
    let n = locals.len();
    let mut best_idx: Option<usize> = None;
    for offset in 0..n {
        let idx = (worker_idx + offset) % n;
        let Some(candidate) = locals[idx].front() else { continue };
        match best_idx {
            None => best_idx = Some(idx),
            Some(cur) => {
                if candidate.priority > locals[cur].front().expect("non-empty").priority {
                    best_idx = Some(idx);
                }
            }
        }
    }
    let idx = best_idx?;
    let job = locals[idx].pop_front().expect("best deque non-empty");
    Some((idx, job))
}
