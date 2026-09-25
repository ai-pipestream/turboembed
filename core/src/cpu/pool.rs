//! The threads a session computes on: made when the session is, parked
//! between runs, and joined when it is released.
//!
//! A run hands the pool one job at a time: a function of a task index and
//! a thread index, and how many tasks there are. The caller's thread is
//! thread 0 and takes tasks too. Each task is taken by exactly one thread,
//! so which thread computes a task varies from run to run; the encoder
//! splits work so that a task's result does not depend on that
//! (encoder.rs).
//!
//! Tasks are claimed from one atomic word that holds the job's number, its
//! task count and the next task, so a claim is either of the current job
//! or fails: a worker that was descheduled over the end of a job cannot
//! take a task of the next one with the old one's function, and the caller
//! waits only for the job's tasks, not for every worker to look in.
//!
//! Nothing here allocates after `new`: a job is published through
//! atomics, and workers park on one Mutex and Condvar made with the pool.

use std::hint::spin_loop;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// A job's function.
type Task<'a> = dyn Fn(usize, usize) + Sync + 'a;

/// How long a worker waits for the next job before it parks. Jobs within
/// a run follow each other in microseconds; between runs it sleeps.
const SPIN: Duration = Duration::from_micros(200);

/// The claim word: the job's number in the top 32 bits, its units in the
/// next 16, the next unit to take in the low 16. A job of more tasks than
/// a unit count holds gives each unit several.
const UNITS: usize = 0xffff;

fn word(job: u64, units: usize, next: usize) -> u64 {
    (job << 32) | ((units as u64) << 16) | next as u64
}

struct Shared {
    /// See UNITS.
    claim: AtomicU64,
    /// The current job's function: a pointer to the caller's reference to
    /// it, on the caller's stack for as long as the job runs.
    job: AtomicPtr<&'static Task<'static>>,
    /// Tasks per unit of the current job, and its tasks.
    per: AtomicUsize,
    tasks: AtomicUsize,
    /// Units of the current job done.
    done: AtomicUsize,
    stop: AtomicBool,
    /// Workers parked on `wake`.
    sleepers: Mutex<usize>,
    wake: Condvar,
}

pub(super) struct Pool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    /// The number of the last job published.
    jobs: u64,
}

/// Aborts the process if a task panics: on a worker, the caller would
/// otherwise wait for its task forever; on the caller, it would leave
/// run() while workers may still use the job it borrowed.
struct AbortOnUnwind;

impl Drop for AbortOnUnwind {
    fn drop(&mut self) {
        if std::thread::panicking() {
            std::process::abort();
        }
    }
}

impl Pool {
    /// A pool of `threads` threads, the caller's among them, so
    /// `threads - 1` are spawned. Fewer are when the system will not
    /// start more; the pool computes the same with any number.
    pub(super) fn new(threads: usize) -> Pool {
        let shared = Arc::new(Shared {
            claim: AtomicU64::new(0),
            job: AtomicPtr::new(std::ptr::null_mut()),
            per: AtomicUsize::new(1),
            tasks: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            sleepers: Mutex::new(0),
            wake: Condvar::new(),
        });
        let mut workers = Vec::with_capacity(threads.saturating_sub(1));
        for id in 1..threads.max(1) {
            let s = Arc::clone(&shared);
            let spawned = std::thread::Builder::new().name(format!("turbo-cpu-{id}")).spawn(move || worker(&s, id));
            match spawned {
                Ok(h) => workers.push(h),
                Err(_) => break,
            }
        }
        Pool { shared, workers, jobs: 0 }
    }

    /// The threads a job runs on, the caller's included.
    pub(super) fn threads(&self) -> usize {
        self.workers.len() + 1
    }

    /// f(task, thread) for every task in 0..tasks, spread over the pool's
    /// threads; returns when all are done. `thread` is under threads().
    pub(super) fn run(&mut self, tasks: usize, f: &Task<'_>) {
        // A task that panics on this thread must not unwind out of here
        // while workers may still call f, which borrows this frame.
        let _abort = AbortOnUnwind;
        if self.workers.is_empty() || tasks <= 1 {
            (0..tasks).for_each(|t| f(t, 0));
            return;
        }
        let s = &*self.shared;
        let per = tasks.div_ceil(UNITS);
        let units = tasks.div_ceil(per);
        self.jobs += 1;
        let mut r: &Task<'_> = f;
        // SAFETY: only the lifetime is erased. A worker dereferences this
        // pointer, and calls f, only after it claims a unit of this job,
        // and every unit is done before this function returns (below),
        // while `r` and `f` are still live.
        let job = unsafe { std::mem::transmute::<*mut &Task<'_>, *mut &'static Task<'static>>(&mut r) };
        s.job.store(job, Ordering::Relaxed);
        s.per.store(per, Ordering::Relaxed);
        s.tasks.store(tasks, Ordering::Relaxed);
        s.done.store(0, Ordering::Relaxed);
        s.claim.store(word(self.jobs, units, 0), Ordering::Release);
        if *lock(&s.sleepers) > 0 {
            s.wake.notify_all();
        }
        take_units(s, 0);
        let mut spins = 0u32;
        while s.done.load(Ordering::Acquire) != units {
            spins += 1;
            if spins < 1 << 12 {
                spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }
}

fn lock(m: &Mutex<usize>) -> MutexGuard<'_, usize> {
    // Nothing panics while holding it; a poisoned count is still a count.
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// Claims and runs units of the current job until none is left. Returns
/// the claim word it saw last.
fn take_units(s: &Shared, thread: usize) -> u64 {
    let mut w = s.claim.load(Ordering::Acquire);
    loop {
        let (units, next) = ((w >> 16) as usize & UNITS, w as usize & UNITS);
        if next >= units {
            return w;
        }
        match s.claim.compare_exchange_weak(w, w + 1, Ordering::AcqRel, Ordering::Acquire) {
            Err(now) => w = now,
            Ok(_) => {
                // SAFETY: the claim succeeded, so the word was this job's
                // and the job was not done: the caller stored its function,
                // per and tasks before the word (Release, which the
                // exchange acquired) and keeps them until every unit is
                // done, this one included.
                let (f, per, tasks) = unsafe {
                    (*s.job.load(Ordering::Relaxed), s.per.load(Ordering::Relaxed), s.tasks.load(Ordering::Relaxed))
                };
                for t in next * per..((next + 1) * per).min(tasks) {
                    f(t, thread);
                }
                s.done.fetch_add(1, Ordering::Release);
                w += 1;
            }
        }
    }
}

fn worker(s: &Shared, id: usize) {
    let _abort = AbortOnUnwind;
    // The word the pool was made with: a worker that starts after the
    // first job, or after the pool stopped, sees the word moved.
    let mut seen = 0;
    loop {
        let start = Instant::now();
        let mut spins = 0u32;
        while s.claim.load(Ordering::Acquire) == seen && !s.stop.load(Ordering::Acquire) {
            spins = spins.wrapping_add(1);
            if !spins.is_multiple_of(256) || start.elapsed() < SPIN {
                spin_loop();
                continue;
            }
            let mut sleepers = lock(&s.sleepers);
            *sleepers += 1;
            while s.claim.load(Ordering::Acquire) == seen && !s.stop.load(Ordering::Acquire) {
                sleepers = s.wake.wait(sleepers).unwrap_or_else(|p| p.into_inner());
            }
            *sleepers -= 1;
        }
        if s.stop.load(Ordering::Acquire) {
            return;
        }
        seen = take_units(s, id);
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        let s = &*self.shared;
        s.stop.store(true, Ordering::Release);
        // A job of no units: every worker sees the word move, and stops.
        s.claim.store(word(self.jobs + 1, 0, 0), Ordering::Release);
        {
            let _sleepers = lock(&s.sleepers);
            s.wake.notify_all();
        }
        for w in self.workers.drain(..) {
            // A worker that panicked aborted the process; there is no
            // error to carry.
            let _ = w.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_task_runs_once_on_a_thread_of_the_pool() {
        for threads in [1, 2, 3, 8] {
            let mut p = Pool::new(threads);
            assert_eq!(p.threads(), threads);
            for tasks in [0, 1, 2, 7, 1000, 3 * UNITS + 5] {
                let hits: Vec<AtomicUsize> = (0..tasks).map(|_| AtomicUsize::new(0)).collect();
                p.run(tasks, &|t, th| {
                    assert!(th < threads);
                    hits[t].fetch_add(1, Ordering::Relaxed);
                });
                assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1), "{threads} threads, {tasks} tasks");
            }
        }
    }

    /// Workers that parked between jobs wake for the next one.
    #[test]
    fn a_parked_pool_wakes() {
        let mut p = Pool::new(4);
        for _ in 0..3 {
            std::thread::sleep(SPIN * 5);
            let n = AtomicUsize::new(0);
            p.run(64, &|_, _| {
                n.fetch_add(1, Ordering::Relaxed);
            });
            assert_eq!(n.load(Ordering::Relaxed), 64);
        }
    }

    /// Many short jobs back to back, as a run's steps are.
    #[test]
    fn short_jobs_follow_each_other() {
        let mut p = Pool::new(4);
        let n = AtomicUsize::new(0);
        for j in 0..20_000 {
            p.run(2 + j % 5, &|_, _| {
                n.fetch_add(1, Ordering::Relaxed);
            });
        }
        assert_eq!(n.load(Ordering::Relaxed), (0..20_000).map(|j| 2 + j % 5).sum::<usize>());
    }
}
